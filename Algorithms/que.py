"""
Robust mean estimation using quantum entropy scoring from the following paper:

Yihe Dong, Samuel B. Hopkins, and Jerry Li. 
Quantum entropy scoring for fast robust mean estimation and improved outlier detection. 
In NeurIPS, 2019. 
URL https://arxiv.org/abs/1906.11366.

Code is taken, with some modifications from https://github.com/twistedcubic/que-outlier-detection

In particular, we reorganize the code to focus on a mean estimation function, que_mean
Many helper functions, including the entire que_utils file are taken directly from the original implementation
"""

import matplotlib
import torch
import numpy as np
import numpy.linalg as linalg
import sklearn.decomposition as decom
import scipy as sp
import que_utils
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def que_mean(data, tau, alpha=4, t=10, early_halt=False, original_threshold=False, fast=True, ev_prune=False):
    """
    Returns mean estimate using quantum entropy scoring

    Parameters:
    data (np.ndarray): A 2D array where rows represent samples and columns represent features.
    tau: expected corruption
    alpha: controls quantum entropy scores: scores closer to 0 converge to weighting based on distance from sample mean
           larger scores converge to spectral scoring as in eigenvalue pruning
    early_halt: True to stop pruning after more that 2tau percent of data is pruned, False otherwise
    fast: True to use fast quantum entropy score calculation
    ev_prune: True to use alternate weighting method equivalent to eigenvalue pruning.
              If this is True, this method will be used regardless of whether ev_prune is true
    """
    n, d = data.shape

    if early_halt:
        min_n = max(math.floor(n * 0.51), math.floor(n * (1-2*tau)))
    
    data = torch.from_numpy(data)

    if original_threshold:
        threshold = 1 + 3 * tau * math.log(1/tau)
    else:
        threshold= (1+ math.sqrt(d/n) + t/math.sqrt(n)) ** 2

    spectral_norm, _ = que_utils.dominant_eval_cov(data) # following the conventions of the original code, directly calculate the spectral norm without

    remove_p = 0.5 * tau # remove 0.5tau percentage of points at every iteration

    # iteratively remove outliers until the spectral norm is less than the threshold or the early_halt condition is met
    while spectral_norm > threshold:
        if tau == 0:
            break # just do nothing if expected corruption is 0
        # the below each use a method to calculate outlier weights and obtain indices to maintain
        if ev_prune:
            # computes weights equivalently to eigenvalue_pruning
            select_idx, _, _ = get_select_idx(data, compute_tau0, remove_p=remove_p, alpha=alpha)
        elif fast:
            # computes weights quickly using JL Chebyshev expansion
            select_idx, _, _ = get_select_idx(data, compute_tau1_fast, remove_p=remove_p, alpha=alpha)
        else:
            # slower exact calculation
            select_idx, _, _ = get_select_idx(data, compute_tau1, remove_p=remove_p, alpha=alpha)

        data = data[select_idx] # prune top 0.5tau percentage of points with largest weights according to some weighting method

        spectral_norm, _ = que_utils.dominant_eval_cov(data)
        n, d = data.shape
        if original_threshold:
            threshold = 1 + 3 * tau * math.log(1/tau)
        else:
            threshold= (1+ math.sqrt(d/n) + t/math.sqrt(n)) ** 2 # the threshold has to be adjusted during pruning as the number of points in the data decreases

        if early_halt and n < min_n:
            break

    return data.mean(dim=0)

def get_select_idx(data, tau_method, remove_p, alpha):
    """
    Return indices of points to remove. Weights are calculated using tau_method, and the points with the
    top remove_p percentage of weights are marked in select_idx to be pruned
    """
    if device == 'cuda':
        select_idx = torch.cuda.LongTensor(list(range(data.size(0))))
    else:
        select_idx = torch.LongTensor(list(range(data.size(0))))
    n_removed = 0
    tau1 = tau_method(data, select_idx, alpha=alpha) # determine weights on points

    #select idx to keep
    cur_select_idx = torch.topk(tau1, k=int(tau1.size(0)*(1-remove_p)), largest=False)[1]

    #note these points are indices of current iteration            
    n_removed += (select_idx.size(0) - cur_select_idx.size(0))

    #print(f"Total Points {select_idx.size(0)}, Points Removed {n_removed}, Expected Removed {select_idx.size(0) * remove_p}")
    select_idx = torch.index_select(select_idx, index=cur_select_idx, dim=0) 
    # print(n_removed)  
    #          
    return select_idx, n_removed, tau1

def compute_tau1_fast(data, select_idx, alpha,):
    """
    Compute quantum entropy scores quickly using JL Chebyshev expansion.
    """

    data = que_utils.pad_to_2power(data)

    data = torch.index_select(data, dim=0, index=select_idx)

    tau1 = que_utils.jl_chebyshev(data, alpha)
    
    return tau1


def compute_tau1(data, select_idx, alpha, noise_vecs=None, **kwargs):
    """
    Compute quantum entropy scores exactly
    """
    data = torch.index_select(data, dim=0, index=select_idx)
    #input should already be centered!
    data_centered = data - data.mean(0, keepdim=True)  
    M = compute_m(data, alpha, noise_vecs) 
    data_m = torch.mm(data_centered, M) #M should be symmetric, so not M.t()
    tau1 = (data_centered*data_m).sum(-1)
        
    return tau1


def compute_m(data, alpha, noise_vecs=None):
    """
    Compute QUE scoring matrix U exactly
    """
    data_cov = (alpha*cov(data))
    #torch svd has bug. U and V not equal up to sign or permutation, for non-duplicate entries.
    #U, D, Vt = (alpha*data_cov).svd()
    
    U, D, Vt = linalg.svd(data_cov.cpu().numpy())
    U = torch.from_numpy(U.astype('float64')).to(device)
    #torch can't take exponential on int64 types.
    D_exp = torch.from_numpy(np.exp(D.astype('float64'))).to(device).diag()
    
    #projection of noise onto the singular vecs. 
    if noise_vecs is not None:
        n_noise = noise_vecs.size(0)
        print(que_utils.inner_mx(noise_vecs, U)[:, :int(1.5*n_noise)])
                    
    m = torch.mm(U, D_exp)
    m = torch.mm(m, U.t())
    
    assert m.max().item() < float('Inf')    
    m_tr =  m.diag().sum()
    m = m / m_tr
    
    return m.to(torch.float64)

def compute_m0(data, alpha, noise_vecs=None):
    data_cov = (alpha*cov(data))
    u,v,w = sp.linalg.svd(data_cov.cpu().numpy())
    m = torch.from_numpy(sp.linalg.expm(alpha * data_cov.cpu().numpy() / v[0])).to(device)
    m_tr =  m.diag().sum()
    m = m / m_tr
    return m
    
'''
Input:
-data: shape (n_sample, n_feat)
'''
def cov(data):    
    data = data - data.mean(dim=0, keepdim=True)    
    cov = torch.mm(data.t(), data) / data.size(0)
    return cov


'''
Input: already centered
'''
def compute_tau0(data, select_idx, n_top_dir=1, noise_vecs=None, **kwargs):
    """
    Compute scores
    """
    data = torch.index_select(data, dim=0, index=select_idx) 

    # center data
    data_sample =data.mean(dim=0)
    centereddata = data-data_sample 

    cov_dir = top_dir(centereddata, n_top_dir, noise_vecs)
    #top dir can be > 1
    cov_dir = cov_dir.sum(dim=0, keepdim=True)

    tau0 = (torch.mm(cov_dir, centereddata.t())**2).squeeze()  
    return tau0


def top_dir(data, n_top_dir=1, noise_vecs=None):
    """
    Compute top cov dir. To compute \tau_old
    Returns:
    -2D array, of shape (1, n_feat)
    """
    data = data - data.mean(dim=0, keepdim=True)    
    data_cov = cov(data)
    if False:
        u, d, v_t = linalg.svd(data_cov.cpu().numpy())
        #pdb.set_trace()
        u = u[:opt.n_top_dir]        
    else:
        #convert to numpy tensor. 
        sv = decom.TruncatedSVD(n_top_dir)
        sv.fit(data.cpu().numpy())
        u = sv.components_
    
    # always None for us
    if noise_vecs is not None:
        
        print('inner of noise with top cov dirs')
        n_noise = noise_vecs.size(0)
        sv1 = decom.TruncatedSVD(n_noise)
        sv1.fit(data.cpu().numpy())
        u1 = torch.from_numpy(sv1.components_).to(device)
        print(que_utils.inner_mx(noise_vecs, u1)[:, :int(1.5*n_noise)])
    
    #U, D, V = svd(data, k=1)    
    return torch.from_numpy(u).to(device)


# NEW NOTE: The below functions are alternate weighting methods present in the original code
# We leave them here for now, but note that they have not been tested

def compute_tau2(data, select_idx, noise_vecs=None, **kwargs):
    """
    compute tau2, v^tM^{-1}v
    """
    data = torch.index_select(data, dim=0, index=select_idx)
    M = cov(data).cpu().numpy()
    M_inv = torch.from_numpy(linalg.pinv(M)).to(device)
    scores = (torch.mm(data, M_inv)*data).sum(-1)
    #cov_dir = top_dir(data, opt, noise_vecs)    
    #top dir can be > 1
    #cov_dir = cov_dir.sum(dim=0, keepdim=True)
    #tau0 = (torch.mm(cov_dir, data.t())**2).squeeze()    
    return scores


def compute_tau1_tau0(data, opt):
    """
    Computes tau1 and tau0.
    Note: after calling this for multiple iterations, use select_idx rather than the scores tau 
    for determining which have been selected as outliers. Since tau's are scores for remaining points after outliers.
    Returns:
    -tau1 and tau0, select indices for each, and n_removed for each
    """
    use_dom_eval = True
    if use_dom_eval:
        #dynamically set alpha now
        #find dominant eval.
        dom_eval, _ = que_utils.dominant_eval_cov(data)
        opt.alpha = 1./dom_eval * opt.alpha_multiplier        
        alpha = opt.alpha        

    #noise_vecs can be used for visualization.
    no_evec = True
    if no_evec:
        noise_vecs = None
        
    def get_select_idx(tau_method):
        if device == 'cuda':
            select_idx = torch.cuda.LongTensor(list(range(data.size(0))))
        else:
            select_idx = torch.LongTensor(list(range(data.size(0))))
        n_removed = 0
        for _ in range(opt.n_iter):
            tau1 = tau_method(data, select_idx, opt, noise_vecs)
            #select idx to keep
            cur_select_idx = torch.topk(tau1, k=int(tau1.size(0)*(1-opt.remove_p)), largest=False)[1]
            #note these points are indices of current iteration            
            n_removed += (select_idx.size(0) - cur_select_idx.size(0))
            select_idx = torch.index_select(select_idx, index=cur_select_idx, dim=0)            
        return select_idx, n_removed, tau1

    if opt.fast_jl:
        select_idx1, n_removed1, tau1 = get_select_idx(compute_tau1_fast)
    else:
        select_idx1, n_removed1, tau1 = get_select_idx(compute_tau1)
    

    select_idx0, n_removed0, tau0 = get_select_idx(compute_tau0)    
    
    return tau1, select_idx1, n_removed1, tau0, select_idx0, n_removed0