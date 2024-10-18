import cvxpy as cp
import numpy as np
import math

# with n=500 and d=150, one run of the SDP algorithm took about 3 minutes, versus being nearly instant -> I think this alone is sufficient to sugegst that we don't examine this algorithm
# why should a slower algorithm with no performance gain be considered? We can just discuss this briefly, and maybe cite up to smaller n

# first, define a function to update h
# this is through a semi definite program

# no threshold is when threshold=1
# so maybe don't have this hard_threshold option

# I'm not quite sure if c2 is actually relevant in practice
def sdp_mean(data, tau, threshold=0.6, p=1, c1=1.6, hard_threshold=True, original_threshold=False): # threshold is tau in paper 
    n, d = data.shape

    c1 = 1 + 3 * tau * math.log(1 / tau)

    # initialize mean estimate as coordinate wise median
    mean_estimate = np.median(data, axis=0) 

    # initialize c2 - concern that this might not work in this data size
    # also not really sure how this gets used -> it is unclear how this is translated in the SDP problem (after they simplify it)
    # probably have to go back to the math myself to understand this aspect
    c2_old = 3 * math.sqrt(d) + 2 * c1 # they note that this can be replaced by a smaller value -> we just need distance from initial coordinate median to true mean less than this with high probability

    t=0

    T = 1 + math.log(c2_old) / math.log(gamma(tau, threshold)) # this depends on first c2, gamma(tau, threshold) doesn't change

    # this implements a do while loop as stated in the original paper
    while True: 
        # update h using a SDP solver
        # h is a nx1 outlier indicator vector -> 1 corresponds to outliers, 0 corresponds to inliers, it can take values from 0 to 1
        w = update_w(data, mean_estimate, c1, original_threshold) 
        print(w)
        
        # print("Weights", w)
        # print("Printing weights")
        # print(w)
        # print(data)
        # print(np.linalg.norm(data, axis=1))
        # print(w @ data)
        # print()

  
        # don't get this too well
        # my guess is that this doesn't actually make much of a difference in practice
        if hard_threshold:
            optional_hard_threshold = np.where(w <= (1-threshold), 0, 1) 
            # if w is less than threshold, it is an outlier -> set value here to 1 (w: 0 is outlier, 1 is inlier)
            
            temp = w * optional_hard_threshold # just sets outliers to 0, others stay as is
            weighted_mean = temp @ data / np.sum(temp)

            # as is, with too low data size everything is considered an outlier
            # when this happens we get a divide by 0 error
        else:
            weighted_mean = w @ data / np.sum(w) # probably have to deal with shapes here

        c2_new = gamma(tau, threshold) * c2_old + beta(tau, threshold, c1)

        t = t + 1

        if t >= T or c2_new >= c2_old:
            print("t", t)
            print("T", T)
            print()
            print("c2_new", c2_new)
            print("c2_old", c2_old)
            print()
            break

        c2_old = c2_new

        print("t", t)

    return weighted_mean


# double check these and order of operations if any problems
def gamma(epsilon, threshold):
    temp = epsilon / threshold
    return math.sqrt(temp / ((1 - temp) * (1 - epsilon - temp)))

def beta(epsilon, threshold, c1):
    temp = epsilon / threshold
    return c1 * ((1 - temp) ** -1/2 + (1 - epsilon) ** -1/2) * math.sqrt(temp / (1 - epsilon - temp))

# w is 1 if not outlier, 0 if outlier
# they update h, which is the inverse, but this feels more sensical to me
def update_w(data, mean_estimate, c, original_threshold):
    n, d = data.shape

    # define the decision variable

    w = cp.Variable(n, nonneg=True) # n is the number of data points and is the size of w
    # ideally ones in w should correspond to inliers and zeros should correspond to outliers

    # define the coefficient vector
    ones = np.ones(n) # could maybe omit this but including it to go with the general format

    # define an objective
    objective = cp.Maximize(ones.T @ w) # maximize the l1 norm of w 
    
    if original_threshold:
        print("original threshold")
        var_est = 1
        print("var est", var_est)
    else:
        print("our threshold")
        var_est = (1+ math.sqrt(d/n) + 2.45/math.sqrt(n)) ** 2 / c # get rid of the c here
        print(var_est)

    # define B which is an upperbound
    top_left_block = np.eye(n)
    bottom_right_block = np.eye(d) * c * n * var_est # need to also multiply by sigma^2 here -> translate our eigenvalue threshold into a bound on this
    B = np.block([
        [top_left_block, np.zeros((n, d))],
        [np.zeros((d, n)), bottom_right_block]
    ])

    # print(f"B : {B}")

    # NOTE: Check this -> ChatGPT generated so not sure if this is going to work
    # also it doesn't seem to actually speed anything up

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



# The zero blocks are already correctly set due to the preallocation

    # Preallocate As to store n matrices of size (n+d) x (n+d)
    # As = np.zeros((n, n + d, n + d))

    # for i in range(n):
    #     # Initialize the ith matrix in As

    #     # Top-left block
    #     As[i, i, i] = 1  # Directly set the diagonal element
        
    #     # Bottom-right block
    #     error = (data[i] - mean_estimate).reshape(-1, 1)  # Reshape to n x 1
    #     bottom_right_block = error @ error.T
    #     As[i, n:, n:] = bottom_right_block  # Assign the bottom-right block

    #     # The zero blocks are already correctly set due to the preallocation
        


    # Define As -> there are n of these and each are (n+d) x (n+d)
    # As = np.zeros((n, (n+d), (n+d))) # n As, each of size (n+d)x(n+d)

    # # This for loop drastically slows down performance
    # # need to deal with this somehow
    # # also possible the issue is just high n
    # for i in range(n):
    #     # print(f"i: {i}")
    #     top_left_block = np.zeros((n, n))
    #     top_left_block[i][i] = 1

    #     error = (data[i] - mean_estimate)[:, np.newaxis] # this is going to be n by 1 matching common math notation
    #     bottom_right_block = error @ error.T
    #     A = np.block([
    #         [top_left_block, np.zeros((n, d))],
    #         [np.zeros((d, n)), bottom_right_block]
    #     ])
    #     # print(f"A with i={i}: {A}")
    #     As[i] = A
    #     # print()
    
    # Reshape w to be (n, 1, 1) to align with As for broadcasting

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

# first I have seen that this returns a correct mean with enough data - but prohibitively slow with larger n

# in test.py it also seems to work on corrupted data, although this isn't too scientific yet - it's just too slow
# I wouldn't be surprised if there's some minor issues right now that are causing slightly worse performance

# now with our adapted threshold, it also seems to work with very low data
if __name__ == "__main__":
    # this whole sys business is not good
    import sys
    sys.path.append("/Users/cullen/Desktop/RobustStats/Algorithms")
    sys.path.append("/Users/cullen/Desktop/RobustStats/Utils")
    sys.path.append("/Users/cullen/Desktop/RobustStats/DataGeneration")

    # see if this eigenvalue pruning covariance thing works
    from data_generation import gaussian_data, generate_data_helper
    from noise_generation import dkk_noise, gaussian_noise_one_cluster

    # this seems to work with enough data size currently
    n = 300
    d = 3
    eta = 0.1


    mean_fun = lambda d: np.ones(d)
    # corruption from dkk paper, hard for naive methods
    id_dkk_corruption = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=dkk_noise, mean_fun=mean_fun)

    # cluster corruption sqrt(d) away from true mean, hard for naive methods
    id_gaussian_corruption_one_cluster = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=gaussian_noise_one_cluster, mean_fun=mean_fun)

    true_mean=mean_fun(d)
    gaus_data, _, _ = id_gaussian_corruption_one_cluster(n, d, eta)
    mean = sdp_mean(gaus_data, 0.2) 
    print(f"ERROR: {np.linalg.norm(mean - true_mean)}")


    # data = np.random.multivariate_normal(np.ones(d), np.eye(d), n)

    # print(np.linalg.norm(sdp_mean(data, 0.1) - np.ones(d)))

# with not so high n, this algorithm seems to break down -> how to make it faster?
# i checked and the problem is not the while loop