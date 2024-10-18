"""
Robust mean estimation using projected gradient descent from the following paper:

Yu Cheng, Ilias Diakonikolas, Rong Ge, and Mahdi Soltanolkotabi. 
High-dimensional robust mean estimation via gradient descent, 2020. 
URL https://arxiv.org/abs/2005.01378.

Code is based on MatLab code from https://github.com/chycharlie/robust-bn-faster
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs

def grad_descent(data, tau, nItr=10):
    """
    Returns mean estimate using projected gradient descent

    Parameters:
    data (np.ndarray): A 2D array where rows represent samples and columns represent features.
    tau: expected corruption
    nItr: the number of iterations of gradient descent; larger values take more times but lower values are not as robust
    """
    N, d = data.shape
    tauN = round(tau * N)
    stepSz = 1 / N

    w = np.ones(N) / N

    for _ in range(nItr):
        # Define a matrix-free operator for Sigma_w_fun
        def matvec(v):
            return Sigma_w_fun(v, data, w)
        
        # Create a LinearOperator to represent the matrix implicitly
        Sigma_w = LinearOperator((d, d), matvec=matvec)
        
        # Use scipy's eigs to find the leading eigenvalue and eigenvector
        eigenvalue, eigenvector = eigs(Sigma_w, k=1, which='LM')
        u = eigenvector.real  # Extract the first eigenvector and make it real

        data_u = data @ u
        nabla_f_w = (data_u**2 - 2 * (w @ data_u) * data_u).squeeze()

        old_w = w.copy()
        w = w - stepSz * nabla_f_w / np.linalg.norm(nabla_f_w)
        w = project_onto_capped_simplex_simple(w, 1 / (N - tauN))

        # Update the step size based on the change in the largest eigenvalue
        Sigma_w_new = LinearOperator((d, d), matvec=lambda v: Sigma_w_fun(v, data, w))
        new_eigenvalue = eigs(Sigma_w_new, k=1, which='LM', return_eigenvectors=False)
        
        if new_eigenvalue.real < eigenvalue.real:
            stepSz *= 2
        else:
            stepSz /= 4
            w = old_w

    mu = data.T @ w
    return mu

def project_onto_capped_simplex_simple(w, cap):
    tL = np.min(w) - 1
    tR = np.max(w)
    
    for _ in range(50):
        t = (tL + tR) / 2
        projection = np.minimum(np.maximum(w - t, 0), cap)
        if np.sum(projection) < 1:
            tR = t
        else:
            tL = t
            
    return np.minimum(np.maximum(w - t, 0), cap)

def Sigma_w_fun(v, data, w):
    data_w = data.T @ w
    return data.T @ (w * (data @ v)) - data_w * (data_w @ v)