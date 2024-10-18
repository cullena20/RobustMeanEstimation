"""
QUE Utilities function directly from https://github.com/twistedcubic/que-outlier-detection
"""

from __future__ import unicode_literals
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import argparse
import torch
import math
import numpy.linalg as linalg
import scipy.linalg

import pdb

res_dir = 'results'
data_dir = 'data'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
Input:
-X: shape (n_sample, n_feat)
'''
def cov(X):    
    X = X - X.mean(dim=0, keepdim=True)    
    cov = torch.mm(X.t(), X) / X.size(0)
    return cov

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_dir', default=1, type=int, help='Max number of directions' )
    parser.add_argument('--lamb_multiplier', type=float, default=1., help='Set alpha multiplier')
    parser.add_argument('--experiment_type', default='syn_lamb', help='Set type of experiment, e.g. syn_dirs, syn_lamb, text_lamb, text_dirs, image_lamb, image_dirs, representing varying alpha or the number of corruption directions for the respective dataset')
    parser.add_argument('--generate_data', help='Generate synthetic data to run synthetic data experiments', dest='generate_data', action='store_true')
    parser.set_defaults(generate_data=False)
    parser.add_argument('--fast_jl', help='Use fast method to generate approximate QUE scores', dest='fast_jl', action='store_true')
    parser.set_defaults(fast_jl=False)
    parser.add_argument('--fast_whiten', help='Use approximate whitening', dest='fast_whiten', action='store_true')
    parser.set_defaults(fast_whiten=False)    
    parser.add_argument('--high_dim', help='Generate high-dimensional data, if running synthetic data experiments', dest='high_dim', action='store_true')
    parser.set_defaults(high_dim=False)    
    
    opt = parser.parse_args()
    
    if len(opt.experiment_type) > 3 and opt.experiment_type[:3] == 'syn':
        opt.generate_data = True
        
    return opt

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    
create_dir(res_dir)

'''
Get degree and coefficient for kth Chebyshev poly.
'''
def get_chebyshev_deg(k):
    if k == 0:
        coeff = [1]
        deg = [0]   
    elif k == 1:
        coeff = [1]
        deg = [1]
    elif k == 2:
        coeff = [2, -1]
        deg = [2, 0]
    elif k == 3:
        coeff = [4, -3]
        deg = [3, 1]
    elif k == 4:
        coeff = [8, -8, 1]
        deg = [4, 2, 0]
    elif k == 5:
        coeff = [16, -20, 5]
        deg = [5, 3, 1]
    elif k == 6:
        coeff = [32, -48, 18, -1]
        deg = [6, 4, 2, 0]
    else:
        raise Exception('deg {} chebyshev not supported'.format(k))
    return coeff, deg
        
'''
Combination of JL projection and 
Chebyshev expansion of the matrix exponential.
Input:
-X: data matrix, 2D tensor. X is sparse for gene data!
Returns:
-tau1: scores, 1D tensor (n,)
'''
def jl_chebyshev(X, lamb):

    #print(X[:,0].mean(0))
    #assert X[:,0].mean(0) < 1e-4
    X = X - X.mean(0, keepdim=True)
    
    n_data, feat_dim = X.size()
    X_scaled = X/n_data
    
    #if lamb=0 no scaling, so not to magnify bessel[i, 0] in the approximation.
    scale = int(dominant_eval_cov(np.sqrt(lamb)*X)[0]) if lamb > 0 else 1
    
    if scale > 1:
        # print('Scaling M! {}'.format(scale))
        #scale down matrix if matrix norm >= 3, since later scale up when
        #odd power
        if scale%2 == 0:
            scale -= 1
        X_scaled /= scale
    else:
        scale = 1    
    
    subsample_freq = int(feat_dim/math.log(feat_dim, 2)) #100
    k = math.ceil(feat_dim/subsample_freq)

    X_t = X.t() 
    #fast Hadamard transform (ffht) vs transform by multiplication by Hadamard mx.
    ffht_b = False 
    P, H, D = get_jl_mx(feat_dim, k, ffht_b)
    
    I_proj = torch.eye(feat_dim, feat_dim, device=X.device)
    
    M = D
    I_proj = torch.mm(D, I_proj)
    if ffht_b:
        #can be obtained from https://github.com/FALCONN-LIB/FFHT
        #place here so not everyone needs to install.   
        import ffht

        M = M.t()
        #M1 = np.zeros((M.size(0), M.size(1)), dtype=np.double)
        M_np = M.cpu().numpy()
        
        I_np = I_proj.cpu().numpy()
        for i in range(M.size(0)):
            ffht.fht(M_np[i])
        for i in range(I_proj.size(0)):
            ffht.fht(I_np[i])
        #pdb.set_trace()
        M = torch.from_numpy(M_np).to(dtype=M.dtype, device=X.device).t()
        I_proj = torch.from_numpy(I_np).to(dtype=M.dtype, device=X.device)
    else:
        #right now form the matrix exponential here
        M = torch.mm(H, M)
        I_proj = torch.mm(H, I_proj)
            
    #apply P now so downstream multiplications are faster: kd instead of d^2.
    #subsample to get reduced dimension
    subsample = True
    if subsample:
        #random sampling performs well in practice and has lower complexity        
        #select_idx = torch.randint(low=0, high=feat_dim, size=(feat_dim//5,)) <--this produces repeats
        if device == 'cuda':
            #pdb.set_trace()
            select_idx = torch.cuda.LongTensor(list(range(0, feat_dim, subsample_freq)))
        else:
            select_idx = torch.LongTensor(list(range(0, feat_dim, subsample_freq)))
        #M = torch.index_select(M, dim=0, index=select_idx)
        M = M[select_idx]
        #I_proj = torch.index_select(I_proj, dim=0, index=select_idx)
        I_proj = I_proj[select_idx]
    else:
        M = torch.sparse.mm(P, M)
        I_proj = torch.sparse.mm(P, I_proj)

    #M is now the projection mx
    A = M
    for _ in range(scale):
        #(k x d)
        A = sketch_and_apply(lamb, X, X_scaled, A, I_proj)
        
    #Compute tau1 scores
    #M = M / M.diag().sum()
    #M is (k x d)
    #compute tau1 scores (this M is previous M^{1/2})
    tau1 = (torch.mm(A, X_t)**2).sum(0)
    
    return tau1

'''
-M: projection mx
-X, X_scaled, input and scaled input
Returns:
-k x d projected matrix
'''
def sketch_and_apply(lamb, X, X_scaled, M, I_proj):
    M = M.to(X.dtype)
    X_t = X.t()
    M = torch.mm(M, X_t)
    M = torch.mm(M, X_scaled)
    
    check_cov = False
    if check_cov:
        #sanity check, use exact cov mx
        #print('Using real cov mx!')        
        M = cov(X)
        subsample_freq = 1
        feat_dim = X.size(1)
        k = feat_dim
        I_proj = torch.eye(k, k, device=X.device)

    check_exp = False
    #Sanity check, computes exact matrix expoenntial
    if False:
        U, D, V_t = linalg.svd(lamb*M.cpu().numpy())
        pdb.set_trace()
        U = torch.from_numpy(U.astype('float32')).to(device)
        D_exp = torch.from_numpy(np.exp(D.astype('float32'))).to(device).diag()
        m = torch.mm(U, D_exp)
        m = torch.mm(m, U.t())        
        #tau1 = (torch.mm(M, X_t)**2).sum(0)        
        return m
    if check_exp:
        M = torch.from_numpy(scipy.linalg.expm(lamb*M.cpu().numpy())).to(device)
        #pdb.set_trace()
        tau1 = (torch.mm(M, X_t)**2).sum(0)
        #X_m = torch.mm(X, M)
        #tau1 = (X*X_m).sum(-1)
        return M
    
    ## Matrix exponential appx ##
    total_deg = 6
    monomials = [0]*total_deg
    #k x d
    monomials[1] = M
    
    #create monomimials in chebyshev poly. Start with deg 2 since already multiplied with one cov.
    for i in range(2, total_deg):        
        monomials[i] = torch.mm(torch.mm(monomials[i-1], X_t), X_scaled)
    
    monomials[0] = I_proj 
    M = 0
    #M is now (k x d)
    #degrees of terms in deg^th chebyshev poly
    for kk in range(1, total_deg):
        #coefficients and degrees for chebyshev poly. Includes 0th deg.  
        coeff, deg = get_chebyshev_deg(kk)

        T_k = 0
        for i, d in enumerate(deg):
            c = coeff[i]            
            T_k += c*lamb**d*monomials[d]
            
        #includes multiplication with powers of i
        bessel_k = get_bessel('-i', kk)
        M = M + bessel_k*T_k

    #M = I_proj
    #degree 0 term. M is now (k x d)
    #M[:, :k] = 2*M[:, :k] + get_bessel('i', 0) * torch.eye(k, feat_dim, device=X.device) #torch.ones((k,), device=X.device).diag()
    #(k x d) matrix
    M = 2*M + get_bessel('i', 0) * I_proj 

    return M
    
    
'''
Create JL projection matrix.
Input: 
-d: original dim
-k: reduced dim
'''
def get_jl_mx(d, k, ffht_b):
    #M is sparse k x d matrix
    
    P = torch.ones(k, d, device=device) #torch.sparse(  )

    if not ffht_b:
        H = get_hadamard(d)
    else:
        H = None
    #diagonal Rademacher mx
    sign = torch.randint(low=0, high=2, size=(d,), device=device, dtype=torch.float32)
    sign[sign==0] = -1
    D = sign.diag()
    
    return P, H, D

#dict of Hadamard matrices of given dimensions
H2 = {}
'''
-d: dimension of H. Power of 2.
-replace with FFT for d log(d).
'''
def get_hadamard(d):

    if d in H2:
        return H2[d]
    if osp.exists('h{}.pt'.format(d)):
        H2[d] = torch.load('h{}.pt'.format(d), weights_only=False).to(device)
        return H2[d]
    power = math.log(d, 2)
    if power-round(power) != 0:
        raise Exception('Dimension of Hamadard matrix must be power of 2')
    power = int(power)
    #M1 = torch.FloatTensor([[ ], [ ]])
    M2 = torch.FloatTensor([1, 1, 1, -1])
    if device == 'cuda':
        M2 = M2.cuda()
    i = 2
    H = M2
    while i <= power:
        #H = torch.ger(H.view(-1), M2).view(2**i, 2**i)
        H = torch.ger(M2, H.view(-1))
        #reshape into 4 block matrices
        H = H.view(-1, 2**(i-1), 2**(i-1))
        H = torch.cat((torch.cat((H[0], H[1]), dim=1), torch.cat((H[2], H[3]), dim=1)), dim=0)
        #if i == 2:
        #    pdb.set_trace()
        i += 1
    torch.save(H, 'h{}.pt'.format(d))
    H2[d] = H.view(d, d) / np.sqrt(d)
    return H2[d]

'''
Pad to power of 2.
Input: size 2.
'''
def pad_to_2power(X):
    n_data, feat_dim = X.size(0), X.size(-1)
    power = int(math.ceil(math.log(feat_dim, 2)))
    power_diff = 2**power-feat_dim
    if power_diff == 0:
        return X
    padding = torch.zeros(n_data, power_diff, dtype=X.dtype, device=X.device)
    X = torch.cat((X, padding), dim=-1)
    
    return X

'''
Find dominant eval of XX^t (and evec in the process) using the power method.
Without explicitly forming XX^t
Returns:
-dominant eval + corresponding eigenvector
'''
def dominant_eval_cov(X):
    n_data = X.size(0)
    X = X - X.mean(dim=0, keepdim=True)
    X_t = X.t()
    X_t_scaled = X_t/n_data
    n_round = 5
    
    v = torch.randn(X.size(-1), 1, device=X.device, dtype=X.dtype)
    for _ in range(n_round):
        v = torch.mm(X_t_scaled, torch.mm(X, v))
        #scale each time instead of at the end to avoid overflow
        #v = v / (v**2).sum().sqrt()
    v = v / (v**2).sum().sqrt()
    mu = torch.mm(v.t(), torch.mm(X_t_scaled, torch.mm(X, v))) / (v**2).sum()
    
    return mu.item(), v.view(-1)
'''
dominant eval of matrix X
Returns: top eval and evec
'''
def dominant_eval(A):
    '''
    n_data = X.size(0)
    X = X - X.mean(dim=0, keepdim=True)
    X_t = X.t()
    X_t_scaled = X_t/n_data
    '''
    n_round = 5    
    v = torch.randn(A.size(-1), 1, device=A.device)
    for _ in range(n_round):
        v = torch.mm(A, v)
        #scale each time instead of at the end to avoid overflow
        #v = v / (v**2).sum().sqrt()
    v = v / (v**2).sum().sqrt()
    mu = torch.mm(v.t(), torch.mm(A, v)) / (v**2).sum()
    
    return mu.item(), v.view(-1)

'''
Top k eigenvalues of X_c X_c^t rather than top one.
'''
def dominant_eval_k(A, k):
    
    evals = torch.zeros(k).to(device)
    evecs = torch.zeros(k, A.size(-1)).to(device)
    
    for i in range(k):
        
        cur_eval, cur_evec = dominant_eval(A)
        A -= (cur_evec*A).sum(-1, keepdim=True) * (cur_evec/(cur_evec**2).sum())
        
        evals[i] = cur_eval
        evecs[i] = cur_evec
        
    return evals, evecs

'''
Top cov dir, for e.g. visualization + debugging.
'''
def get_top_evals(X, k=10):
    X_cov = cov(X)
    U, D, V_t = linalg.svd(X_cov.cpu().numpy())
    return D[:k]

#bessel function values at i and -i, index is degree.
#sum_{j=0}^\infty ((-1)^j/(2^(2j+k) *j!*(k+j)! )) * (-i)^(2*j+k) for k=0 BesselI(0, 1)
bessel_i = [1.1266066]
bessel_neg_i = [1.266066, -0.565159j, -0.1357476, 0.0221684j, 0.00273712, -0.00027146j]
#includes multipliacation with powers of i, i**k
bessel_neg_i = [1.266066, 0.565159, 0.1357476, 0.0221684, 0.00273712, 0.00027146]

'''
Get precomputed deg^th Bessel function value at input arg.
'''
def get_bessel(arg, deg):
    if arg == 'i':
        if deg > len(bessel_i):
            raise Exception('Bessel i not computed for deg {}'.format(deg))         
        return bessel_i[deg]
    elif arg == '-i':
        if deg > len(bessel_neg_i):
            raise Exception('Bessel -i not computed for deg {}'.format(deg))
        return bessel_neg_i[deg]


'''
Projection (vector) of dirs onto target direction.
'''
def project_onto(tgt, dirs):
    
    projection = (tgt*dirs).sum(-1, keepdims=True) * (tgt/(tgt**2).sum())
    
    return projection