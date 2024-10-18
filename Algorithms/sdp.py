"""
Robust mean estimation using semi definite programming based on the following paper:

Aditya Deshmukh, Jing Liu, and Venugopal V. Veeravalli. 
Robust mean estimation in high dimensions: An outlier fraction agnostic and efficient algorithm, 2022
URL https://arxiv.org/abs/2102.08573.
"""

import cvxpy as cp
import numpy as np
import math


def sdp_mean(data, tau, hard_cutoff=True, cutoff=0.6, original_threshold=False): # cutoff is tau in paper 
    """
    Returns a mean estimate based on semi definite programming.

    Parameters:
    data (np.ndarray): A 2D array where rows represent samples and columns represent features.
    tau: expected corruption
    hard_cutoff: if True, outliers with weights below a cutoff are pruned; otherwise outliers are only weighted
    cutoff: the cutoff used for hard_cutoff
    original_threshold: if False, uses threshold in constraint to account for low data size, if True, uses method
                        described in original paper (fails under low data size)
    """
    n, d = data.shape

    c = 1 + 3 * tau * math.log(1 / tau) # slack term used in SDP constraint

    # initialize mean estimate as coordinate wise median
    mean_estimate = np.median(data, axis=0) 

    # initialize c2 -
    c2_old = 3 * math.sqrt(d) + 2 * c
    t=0

    T = 1 + math.log(c2_old) / math.log(gamma(tau, cutoff))

    # this implements a do while loop as stated in the original paper
    while True: 
        # update h using a SDP solver
        # h is a nx1 outlier indicator vector -> 1 corresponds to outliers, 0 corresponds to inliers, it can take values from 0 to 1
        w = update_w(data, mean_estimate, c, original_threshold) 
        
        # print(w)


  
        # can either return weighted mean with outliers downweighted, or directly prune points with weights less than a cutoff
        # weights are closer to 0 for outliers and closer to 1 for inliers
        if hard_cutoff:
            optional_hard_cutoff = np.where(w <= (1-cutoff), 0, 1) 
            # if w is less than cutoff, it is an outlier -> set value here to 1 (w: 0 is outlier, 1 is inlier)
            
            temp = w * optional_hard_cutoff # just sets outliers to 0, others stay as is
            weighted_mean = temp @ data / np.sum(temp)

            # currently with very low data size everything is considered an outlier and will give a divide by 0 error(this can also happen with eigenvalue pruning)
        else:
            weighted_mean = w @ data / np.sum(w) 

        c2_new = gamma(tau, cutoff) * c2_old + beta(tau, cutoff, c)

        t = t + 1

        if t >= T or c2_new >= c2_old:
            # print("t", t)
            # print("T", T)
            # print()
            # print("c2_new", c2_new)
            # print("c2_old", c2_old)
            # print()
            break

        c2_old = c2_new

        # print("t", t)

    return weighted_mean

def gamma(epsilon, cutoff):
    temp = epsilon / cutoff
    return math.sqrt(temp / ((1 - temp) * (1 - epsilon - temp)))

def beta(epsilon, cutoff, c):
    temp = epsilon / cutoff
    return c * ((1 - temp) ** -1/2 + (1 - epsilon) ** -1/2) * math.sqrt(temp / (1 - epsilon - temp))

# w is 1 if not outlier, 0 if outlier
def update_w(data, mean_estimate, c, original_threshold):
    n, d = data.shape

    # define the decision variable

    w = cp.Variable(n, nonneg=True) # n is the number of data points and is the size of w
    # ideally ones in w should correspond to inliers and zeros should correspond to outliers

    # define the coefficient vector
    ones = np.ones(n)

    # define an objective
    objective = cp.Maximize(ones.T @ w) # maximize the l1 norm of w 
    
    # var_e
    if original_threshold:
        # print("original threshold")
        var_est = 1
        # print("var est", var_est)
    else:
       #  print("our threshold")
        var_est = (1+ math.sqrt(d/n) + 2.45/math.sqrt(n)) ** 2 
       # print(var_est)

    # define B which is an upperbound
    top_left_block = np.eye(n)
    bottom_right_block = np.eye(d) * c * n * var_est # to account for low data size, this should be multiplied by sigma^2 
    B = np.block([
        [top_left_block, np.zeros((n, d))],
        [np.zeros((d, n)), bottom_right_block]
    ])

    # Preallocate As to store n matrices of size (n+d) x (n+d)
    As = np.zeros((n, n + d, n + d))

    # Create the top-left block (diagonal identity matrices)
    # Use np.arange to place 1s along the main diagonal for each slice
    As[np.arange(n), np.arange(n), np.arange(n)] = 1

    # Calculate the error vectors for all rows at once
    errors = data - mean_estimate  # This will be an (n, n) array

    # Compute the outer products for the bottom-right block
    bottom_right_blocks = errors[:, :, np.newaxis] @ errors[:, np.newaxis, :]

    # Assign the bottom-right blocks to the corresponding location in As
    As[:, n:, n:] = bottom_right_blocks

    # Define constraint
    # weighted_sum = cp.sum(cp.multiply(As, w[:, None, None]), axis=0)
    # constraint = weighted_sum << B

    constraint = sum(w[i] * As[i] for i in range(n)) << B # << corresponds to matrix inequality and is the semidefinite constraint
    
    # Define optimization problem
    problem = cp.Problem(objective, [constraint])

    # Solve the problem 
    problem.solve(solver=cp.MOSEK) # cp.SCS is a possible solver, could experiment with others

    # print optimal results
    # print("Optimal value:", problem.value)
    # print("Optimal w:", w.value)

    return w.value 