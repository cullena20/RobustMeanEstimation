import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import numpy as np
import numpy.linalg as linalg
import sklearn.decomposition as decom
from scipy.stats import ortho_group
import scipy.stats as st
import scipy as sp
import que_utils # ideally put everything here 
import time
import math

'''
Evaluate robust mean estimation using various scorings on data, in terms
of both error from true mean and time.
Input:
-data: input, already corrupted and centered.
-n: number of samples
'''
# to do
# Clean up how these tau_weighting things are handled
def que_mean(data, tau, alpha=4, t=10, original_threshold=False, fast=False, multiplier=False, early_halt=False):
    n, d = data.shape

    if early_halt:
        min_n = max(math.floor(n * 0.51), math.floor(n * (1-2*tau)))
    
    data = torch.from_numpy(data)

    if original_threshold:
        threshold = 1 + 3 * tau * math.log(1/tau)
    else:
        threshold= (1+ math.sqrt(d/n) + t/math.sqrt(n)) ** 2

    spectral_norm, _ = que_utils.dominant_eval_cov(data)
    #run in loop until spectral norm small

    remove_p = tau/2 # why divide by 2

    counter0 = 0

    # sometimes this downweigting step just doesn't work and it keeps iterating and fails catostrophically
    # in these cases no iteration works
    while spectral_norm > threshold:
        if tau == 0:
            break # just do nothing here
        if fast:
            select_idx, _, _ = get_select_idx(data, compute_tau1_fast, remove_p=remove_p, alpha=alpha)
        else:
            select_idx, _, _ = get_select_idx(data, compute_tau0, remove_p=remove_p, alpha=alpha)

        data = data[select_idx]

        spectral_norm, _ = que_utils.dominant_eval_cov(data)
        n, d = data.shape
        if original_threshold:
            threshold = 1 + 3 * tau * math.log(1/tau)
        else:
            threshold= (1+ math.sqrt(d/n) + t/math.sqrt(n)) ** 2
            
        counter0 += 1
        if early_halt and n < min_n:
            break

    return data.mean(dim=0)


device = que_utils.device
NOISE_INN_THRESH = 0.1
DEBUG = False

'''
Compute QUE scoring matrix U.
'''
def compute_m(data, alpha, noise_vecs=None):
    
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

# this is the exact implementaiton I thimk
def compute_m0(data, alpha, noise_vecs=None):
    data_cov = (alpha*cov(data))
    u,v,w = sp.linalg.svd(data_cov.cpu().numpy())
    #pdb.set_trace()
    m = torch.from_numpy(sp.linalg.expm(alpha * data_cov.cpu().numpy() / v[0])).to(que_utils.device)
    m_tr =  m.diag().sum()
    m = m / m_tr
    return m
    
'''
Compute top cov dir. To compute \tau_old
Returns:
-2D array, of shape (1, n_feat)
'''
def top_dir(data, n_top_dir=1, noise_vecs=None):
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
    
'''
Input:
-data: shape (n_sample, n_feat)
'''
def cov(data):    
    data = data - data.mean(dim=0, keepdim=True)    
    cov = torch.mm(data.t(), data) / data.size(0)
    return cov

def compute_tau1_fast(data, select_idx, alpha, noise_vecs=None, **kwargs):

    data = que_utils.pad_to_2power(data)

    data = torch.index_select(data, dim=0, index=select_idx)
    data_centered = data - data.mean(0, keepdim=True)

    tau1 = que_utils.jl_chebyshev(data, alpha)
    
    return tau1

'''
Input:
-data: centered
-select_idx: idx to keep for this iter, 1D tensor.
Output:
-data: updated data
-tau1
'''
def compute_tau1(data, select_idx, alpha, noise_vecs=None, **kwargs):
    
    data = torch.index_select(data, dim=0, index=select_idx)
    #input should already be centered!
    data_centered = data - data.mean(0, keepdim=True)  
    M = compute_m(data, alpha, noise_vecs) 
    data_m = torch.mm(data_centered, M) #M should be symmetric, so not M.t()
    tau1 = (data_centered*data_m).sum(-1)
        
    return tau1

'''
Input: already centered
'''
def compute_tau0(data, select_idx, n_top_dir=1, noise_vecs=None, **kwargs):
    data = torch.index_select(data, dim=0, index=select_idx)

    data_sample =data.mean(dim=0)
    centereddata = data-data_sample 

    cov_dir = top_dir(centereddata, n_top_dir, noise_vecs)
    #top dir can be > 1
    cov_dir = cov_dir.sum(dim=0, keepdim=True)

    tau0 = (torch.mm(cov_dir, centereddata.t())**2).squeeze()    # we are essentially pruning based on squared projections onto top eigenvalue
    # so que works better in certain settings only because of this and the pruning method (no Gaussian concentration inequality)
    # the speed slowdown is from something else
    return tau0

'''
compute tau2, v^tM^{-1}v
'''
def compute_tau2(data, select_idx, noise_vecs=None, **kwargs):
    data = torch.index_select(data, dim=0, index=select_idx)
    M = cov(data).cpu().numpy()
    M_inv = torch.from_numpy(linalg.pinv(M)).to(que_utils.device)
    scores = (torch.mm(data, M_inv)*data).sum(-1)
    #cov_dir = top_dir(data, opt, noise_vecs)    
    #top dir can be > 1
    #cov_dir = cov_dir.sum(dim=0, keepdim=True)
    #tau0 = (torch.mm(cov_dir, data.t())**2).squeeze()    
    return scores

'''
Input:
-data: input, already corrupted
-n: number of samples
'''
# def train(data, noise_idx, outlier_method_l, opt):
    
#     tau1, select_idx1, n_removed1, tau0, select_idx0, n_removed0 = compute_tau1_tau0(data, opt)
    
#     all_idx = torch.zeros(data.size(0), device=device) 
#     ones = torch.ones(noise_idx.size(0), device=device) 
#     all_idx.scatter_add_(dim=0, index=noise_idx.squeeze(), src=ones)
        
#     debug = False
#     if debug:
#         data_cov = cov(data)
#         U, D, V_t = linalg.svd(data_cov.cpu().numpy())
#         U1 = torch.from_numpy(U[0]).to(que_utils.device)
#         '''        
#         all_idx = torch.zeros(data.size(0), device=device)  #torch.cuda.LongTensor(range(data.size(0) )) #  dtype=torch.int64,
#         ones = torch.ones(cor_idx.size(0), device=device) #dtype=torch.int64,        
#         all_idx.scatter_add_(dim=0, index=cor_idx.squeeze(), src=ones)
#         '''
#         good_vecs = data[all_idx==0]
#         w1_norm = (good_vecs**2).sum(-1).mean()
#         good_proj = (good_vecs*U1).sum(-1)
#         que_utils.hist(good_proj, 'inliers_syn', high=500)

#         cor_vecs = torch.index_select(data, dim=0, index=cor_idx.squeeze())
#         w2_norm = (cor_vecs**2).sum(-1).mean()
#         cor_proj = (cor_vecs*U1).sum(-1)
#         que_utils.hist(cor_proj, 'outliers_syn', high=500)
#         pdb.set_trace()

#     #scores of good and bad points
#     good_scores1 = tau1[all_idx==0]
#     bad_scores1 = tau1[all_idx==1]
#     good_scores0 = tau0[all_idx==0]
#     bad_scores0 = tau0[all_idx==1]
    
#     auc1 = que_utils.auc(good_scores1, bad_scores1)
#     auc0 = que_utils.auc(good_scores0, bad_scores0)
#     print('auc0 {} auc1 {}'.format(auc0, auc1))
#     scores_l = [auc1, auc0]
    
#     #default is tau0
#     for method in outlier_method_l:
#         if method == 'iso forest':
#             tau = baselines.isolation_forest(data)
#         elif method == 'lof':
#             tau = baselines.knn_dist_lof(data)
#         elif method == 'ell env':
#             tau = baselines.ellenv(data)
#         elif method == 'dbscan':
#             tau = baselines.dbscan(data)
#         elif method == 'l2':
#             tau = baselines.l2(data)
#         elif method == 'knn':
#             tau = baselines.knn_dist(data)
#         else:
#             raise Exception('method {} not supported'.format(method))
        
#         good_tau = tau[all_idx==0]
#         bad_tau = tau[all_idx==1]    
#         auc = que_utils.auc(good_tau, bad_tau)        
#         scores_l.append(auc)
#         if opt.visualize_scores:
#             pdb.set_trace()
#             que_utils.inlier_outlier_hist(good_tau, bad_tau, method+'syn')
 
#     return scores_l

# def simple_spectral_norm(data):
#     # Calculate the empirical mean of the tensor data along the first axis (batch size).
#     empirical_mean = torch.mean(data, dim=0)
#     # Center the data by subtracting the mean and scaling.
#     centered_data = (data - empirical_mean) / torch.sqrt(torch.tensor(data.size(0), dtype=torch.float32))

#     # Perform Singular Value Decomposition (SVD) on the centered data.
#     U, S, Vt = torch.svd(centered_data)
#     # The spectral norm is the square of the largest singular value.
#     spectral_norm = S[0] ** 2

#     return spectral_norm



'''
Computes tau1 and tau0.
Note: after calling this for multiple iterations, use select_idx rather than the scores tau 
for determining which have been selected as outliers. Since tau's are scores for remaining points after outliers.
Returns:
-tau1 and tau0, select indices for each, and n_removed for each
'''
def compute_tau1_tau0(data, opt):
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
    
    '''
    if device == 'cuda':
        select_idx = torch.cuda.LongTensor(range(data.size(0)))
    else:
        select_idx = torch.LongTensor(range(data.size(0)))
    for _ in range(opt.n_iter):
        tau0 = compute_tau0(data, select_idx, opt)
        cur_select_idx = torch.topk(tau0, k=tau1.size(0)*(1-opt.remove_p), largest=False)[1]
        select_idx = torch.index_select(select_idx, index=cur_select_idx, dim=0)
    '''
    select_idx0, n_removed0, tau0 = get_select_idx(compute_tau0)    
    
    return tau1, select_idx1, n_removed1, tau0, select_idx0, n_removed0

def get_select_idx(data, tau_method, remove_p, alpha, n_iter=1):
    if device == 'cuda':
        select_idx = torch.cuda.LongTensor(list(range(data.size(0))))
    else:
        select_idx = torch.LongTensor(list(range(data.size(0))))
    n_removed = 0
    noise_vecs = None
    for _ in range(n_iter):
        tau1 = tau_method(data, select_idx, alpha=alpha) # determine weights on points
        #select idx to keep
        cur_select_idx = torch.topk(tau1, k=int(tau1.size(0)*(1-remove_p)), largest=False)[1]
        #note these points are indices of current iteration            
        n_removed += (select_idx.size(0) - cur_select_idx.size(0))
        #print(f"Total Points {select_idx.size(0)}, Points Removed {n_removed}, Expected Removed {select_idx.size(0) * remove_p}")
        select_idx = torch.index_select(select_idx, index=cur_select_idx, dim=0) 
    # print(n_removed)           
    return select_idx, n_removed, tau1

 
# if __name__=='__main__':

#     opt = que_utils.parse_args()
#     '''
#     generate_data = True 
#     opt.fast_jl = True     
#     opt.high_dim = True
#     '''
#     generate_data = True 
#     #compute std or confidence interval to measure uncertainty.
#     opt.use_std = True
#     #whether computing differences between scores or raw scores.
#     opt.compute_scores_diff = True
#     #whether to visualize scores, could be useful for debugging.
#     opt.visualize_scores = False
#     #whether to whiten data, note this is void for some datasets.
#     opt.whiten = True
#     #run robust mean estimation rather than outlier detection.
#     #Note this is only supported for evaluating against number of directions.
#     opt.rme = True
#     #e.g. syn_dirs, syn_alpha
#     dataset_name = opt.experiment_type
    
#     if opt.generate_data:
#         opt.dir = 'syn'
#         print('{}'.format(opt.dir))
#         if dataset_name == 'syn_alpha':
#             opt.type = 'alpha'
#             generate_and_score_alpha(opt, 'syn')
#         elif dataset_name == 'syn_dirs':
#             opt.type = 'dirs'
#             generate_and_score(opt, 'syn')            
#     else:
#         #'glove_alpha' #'glove_alpha' #'glove_dirs' #'glove'
#         #dataset_name = 'glove_dirs' 
#         if dataset_name == 'text':
#             opt.dir = 'text'
#             opt.type = '_'
#             test_glove_data(opt)
#         elif dataset_name == 'text_alpha':
#             opt.dir = 'text'
#             opt.type = 'alpha'   
#             test_glove_data_alpha(opt)
#         elif dataset_name == 'text_dirs':
#             opt.dir = 'text'
#             opt.type = 'dirs'   
#             test_glove_data_dirs(opt)
#         elif dataset_name == 'ads':
#             test_ads_data(opt)
#         elif dataset_name == 'genetics':
#             test_genetics_data()
#         elif dataset_name == 'vgg':
#             test_vgg_data()
#         else:
#             raise Exception('Unsupported experiment type {}'.format(opt.experiment_type))