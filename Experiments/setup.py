# this whole sys business is not good
import sys
sys.path.append("/Users/cullen/Desktop/RobustStats/Algorithms")
sys.path.append("/Users/cullen/Desktop/RobustStats/Utils")
sys.path.append("/Users/cullen/Desktop/RobustStats/DataGeneration")

from eigenvalue_pruning import eigenvalue_pruning_updated, eigenvalue_pruning_unknown_cov, eigenvalue_pruning_original, new_eigenvalue_pruning_unknown
from lee_valiant import lee_valiant_original, lee_valiant_simple
from lrv import lrv, lrvGeneral
from ransac import ransac_mean, ransac_mean_unknown_cov
from simple import sample_mean, coordinate_wise_median, median_of_means, geometric_median, coord_trimmed_mean
from que import que_mean
from grad_descent import grad_descent
from deshmukh import sdp_mean

import numpy as np
import math

from helper import experiment_suite, run_experiment
from data_generation import generate_data_helper, gaussian_data
from noise_generation import dkk_noise, gaussian_noise_one_cluster, gaussian_noise_two_clusters, uniform_noise_top, uniform_noise_whole, obvious_noise, subtractive_noise, obvious_easy, obvious_hard_two_clusters, multiple_corruption
from noise_generation import gaussian_noise_one_cluster_nonspherical, obvious_noise_nonspherical, subtractive_noise_nonspherical, uniform_noise_top_nonspherical

sdp = {
        "sample_mean": lambda data, tau: sample_mean(data),
        "sdp_mean": lambda data, tau: sdp_mean(data, tau),
    }

ev_pruning_comparison = {
    "ev_filtering_low_n": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5),
    "ev_filtering_low_n_random": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5, pruning="random"),
    "ev_filtering_low_n_fixed": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5, pruning="fixed"),
}

ev_que_original_vs_new = {
    #"sample_mean": lambda data, tau: sample_mean(data),
    #"ev_filtering": lambda data, tau: eigenvalue_pruning_original(data, tau, 0.1, 5),
    #"ev_filtering_low_n": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5),
    "que": lambda data, tau: que_mean(data, tau, t=10, original_threshold=True, fast=True, multiplier=False),
    #"que_low_n": lambda data, tau: que_mean(data, tau, t=10)
}

que_estimators = {
    "que-0": lambda data, tau: que_mean(data, tau, t=10, original_threshold=False, fast=True, alpha_multiplier=0),
    "que-0.5": lambda data, tau: que_mean(data, tau, t=10, original_threshold=False, fast=True, alpha_multiplier=0.5),
    "que-1": lambda data, tau: que_mean(data, tau, t=10, original_threshold=False, fast=True, alpha_multiplier=1),
    "que-4": lambda data, tau: que_mean(data, tau, t=10, original_threshold=False, fast=True, alpha_multiplier=4),
    "que-50": lambda data, tau: que_mean(data, tau, t=10, original_threshold=False, fast=True, alpha_multiplier=10),
    "que-200": lambda data, tau: que_mean(data, tau, t=10, original_threshold=False, fast=True, alpha_multiplier=200)
    # "que4_slow": lambda data, tau: que_mean(data, tau, t=10, original_threshold=False, fast=False, alpha_multiplier=4, multiplier=True)
}

# first lay out several corruption schemes
default_n = 500
default_d = 500
default_eps = 0.1
default_tau = None # tau = None means that we will use tau=eps


ev_que_experiment = [["n", np.arange(20, 10021, 500), None, default_d, default_eps, default_tau]]

# default_n = 300
# default_d = 300
# default_eps = 0.1
# default_tau = None # tau = None means that we will use tau=eps

mean_fun = lambda d: np.ones(d) * 5 # this is used the same throughout


# Experiments - Same ranges throughout distributions

experiments = [["n", np.arange(20, 5021, 500), None, default_d, default_eps, default_tau],
               ["n", np.arange(20, 521, 50), None, default_d, default_eps, default_tau],
               ["d", np.arange(20, 1021,  100), default_n, None, default_eps, default_tau],
               ["eps", np.arange(0, 0.46, 0.05), default_n, default_d, None, default_tau]]

time_experiments = [["n", np.arange(20, 5021, 500), None, default_d, default_eps, default_tau],
                    ["d", np.arange(20, 1021,  100), default_n, None, default_eps, default_tau],
                    ["eps", np.arange(0, 0.46, 0.05), default_n, default_d, None, default_tau]
                    ]

uncorrupted_experiments = [["n", np.arange(20, 5021, 500), None, default_d, 0, 0.1],
                            ["n", np.arange(20, 521, 50), None, default_d, 0, 0.1],
                            ["d", np.arange(20, 1021,  100), default_n, None, 0, 0.1],
                            ["tau", np.arange(0, 0.46, 0.05), default_n, default_d, 0, None]]

alt_experiments = [["n", np.arange(20, 10021, 500), None, default_d, default_eps, default_tau],
            ["n", np.arange(20, 1021, 50), None, default_d, default_eps, default_tau],
            ["d", np.arange(20, 1021,  100), default_n, None, default_eps, default_tau],
            ["eps", np.arange(0, 0.46, 0.05), default_n, default_d, None, default_tau]]

experiments_short = [
    ["n", np.arange(20, 5021, 1500), None, default_d, default_eps, default_tau],
    ["n", np.arange(20, 521, 150), None, default_d, default_eps, default_tau],
    ["d", np.arange(20, 1021, 300), default_n, None, default_eps, default_tau],
    ["eps", np.arange(0, 0.46, 0.15), default_n, default_d, None, default_tau]
]

std_experiments = [["data", np.arange(0.1, 201, 25), default_n, default_d, default_eps, default_tau]]

std_experiments_short = [["data", np.arange(0.1, 201, 50), default_n, default_d, default_eps, default_tau]]

ev_comparison_experiment = [["n", np.arange(20, 10021, 2500), None, default_d, default_eps, default_tau]]

# original experiments used
old_experiments = [["n", np.arange(20, 3001, 100), None, 150, 0.1, None],
               ["n", np.arange(20, 300, 50), None, 150, 0.1, None],
               ["d", np.arange(20, 301, 10), 150, None, 0.1, None],
               ["eps", np.arange(0.01, 0.32, 0.05), 150, 150, None, None]]

# abridged original experiments
old_experiments_short = [["n", np.arange(20, 3021, 500), None, 150, 0.1, None],
               ["n", np.arange(20, 321, 80), None, 150, 0.1, None],
               ["d", np.arange(20, 321, 50), 150, None, 0.1, None],
               ["eps", np.arange(0.01, 0.32, 0.1), 150, 150, None, None]]

# huge experiments
huge_experiments = [["d", np.arange(20, 10021, 500), default_n, None, default_eps, default_tau],
                    ["eps", np.arange(0.01, 0.42, 0.05), 10000, 10000, None, default_tau]]

# robustness to true corruption
tau_experiments = [["tau", np.arange(0.01, 0.46, 0.05), default_n, default_d, 0.2, None]]

# time -> same thing as original, minus zoom in
experiments_time = [["n", np.arange(20, 5021, 500), None, default_d, default_eps, default_tau],
               ["d", np.arange(20, 1021,  100), default_n, None, default_eps, default_tau],
               ["eps", np.arange(0, 0.46, 0.05), default_n, default_d, None, default_tau]]


# Identity Covariance No Corruption
uncorrupted_data_scheme = lambda n, d, eps: gaussian_data(n, d, eps, mean_fun=mean_fun)

# Identity Covariance Schemes
# cov_fun is identity by default in generate_data_helper

# corruption from dkk paper, hard for naive methods
id_dkk_corruption = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=dkk_noise, mean_fun=mean_fun)

# cluster corruption sqrt(d) away from true mean, hard for naive methods
id_gaussian_corruption_one_cluster = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=gaussian_noise_one_cluster, mean_fun=mean_fun)

# two clusters of corruption sqrt(d) away from true mean, 45 degrees between each other, 70% one cluster 30% the other, hard for naive methods
id_gaussian_corruption_two_clusters = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=gaussian_noise_two_clusters, mean_fun=mean_fun)

# uniform corruption from true_mean to true_mean + 1 in every coordinate, hard for naive methods
id_uniform_corruption_top = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=uniform_noise_top, mean_fun=mean_fun)

# NOT USED - uniformly place corruption within sqrt(d) from true mean, not hard for naive methods
id_uniform_corruption_whole = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=uniform_noise_whole, mean_fun=mean_fun)

# place two clusters of corruption 10sqrt(d) away from true mean and 20sqrt(d) away from true mean, easy for naive methods - show good methods in hard cases work in easy ones too
id_obvious_corruption = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=obvious_noise, mean_fun=mean_fun)

# place two clusters - one sqrt(d) from the true mean, one 10 * sqrt(d) away from the true mean
id_obvious_hard_corruption = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=obvious_hard_two_clusters, mean_fun=mean_fun)

# subtract epsilon largest points from the projection in a certain direction - nothing performs much worse than sample mean here
id_subtractive_corruption = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=subtractive_noise, mean_fun=mean_fun, additive=False)

# One Easy Cluster One Hard Cluster
# id_obvious_easy_corruption = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=obvious_easy, mean_fun=mean_fun)

id_obvious_gaus_one_corruption = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun= lambda n, d, true_mean: multiple_corruption(n, d, true_mean, obvious_noise, gaussian_noise_one_cluster), mean_fun=mean_fun)

id_obvious_dkk_corruption = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun= lambda n, d, true_mean: multiple_corruption(n, d, true_mean, obvious_noise, dkk_noise), mean_fun=mean_fun)

identity_corruption_schemes = {  "dkk": id_dkk_corruption,
                        "gaus_one": id_gaussian_corruption_one_cluster,
                        "gaus_two": id_gaussian_corruption_two_clusters,
                        "unif_top": id_uniform_corruption_top,
                        "obvious": id_obvious_corruption,
                        "subtractive_corruption": id_subtractive_corruption
}

# Large Outlier Mixes - Identity Covariance

large_outlier_identity_corruption_schemes = {
    #"obvious": id_obvious_corruption,
    "obvious_and_gaus_one": id_obvious_gaus_one_corruption,
    "obvious_and_dkk": id_obvious_dkk_corruption
}

# dependence On True Mean - Identity Covariance
gaus_one_random = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=gaussian_noise_one_cluster, mean_fun= lambda d: np.random.randn(d) * 50)
dkk_random = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=dkk_noise, mean_fun= lambda d: np.random.randn(d) * 50)

identity_true_mean_dependence_schemes = {
    "gaus_one_random": gaus_one_random,
    "dkk_random": dkk_random
}

# Unknown Spherical Covariance

var = 25
std = math.sqrt(var)
cov_fun = lambda d: np.eye(d) * var

# same as identity schemes but scale all noise locations appropriately
sp_dkk_corruption = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun= lambda n, d, true_mean: dkk_noise(n, d, true_mean, std=std), mean_fun=mean_fun, cov_fun = cov_fun)
sp_gaussian_corruption_one_cluster = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=lambda n, d, true_mean: gaussian_noise_one_cluster(n, d, true_mean, std=std), mean_fun=mean_fun, cov_fun = cov_fun)
sp_gaussian_corruption_two_clusters = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=lambda n, d, true_mean: gaussian_noise_two_clusters(n, d, true_mean, std=std), mean_fun=mean_fun, cov_fun = cov_fun)
sp_uniform_corruption_top = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=lambda n, d, true_mean: uniform_noise_top(n, d, true_mean, std=std), mean_fun=mean_fun, cov_fun = cov_fun)
sp_obvious_corruption = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=lambda n, d, true_mean: obvious_noise(n, d, true_mean, std=std), mean_fun=mean_fun, cov_fun = cov_fun)
sp_subtractive_corruption = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=subtractive_noise, mean_fun=mean_fun, cov_fun = cov_fun, additive=False)

# this isn't used and isn't so helfpul
sp_uniform_corruption_whole = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=lambda n, d, true_mean: uniform_noise_whole(n, d, true_mean, std=std), mean_fun=mean_fun, cov_fun = cov_fun)


spherical_corruption_schemes = {
    "dkk": sp_dkk_corruption,
    "gaus_one": sp_gaussian_corruption_one_cluster,
    "gaus_two": sp_gaussian_corruption_two_clusters,
    "unif_top": sp_uniform_corruption_top,
    "obvious": sp_obvious_corruption,
    "subtractive_corruption": sp_subtractive_corruption
}

# experiment with adjusting standard deviation - var parameter is not used here
std_dkk_corruption = lambda n, d, eps, data_std: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun= lambda n, d, true_mean: dkk_noise(n, d, true_mean, std=data_std), mean_fun=mean_fun, cov_fun = lambda d: np.eye(d)*data_std**2)
std_gaussian_corruption_one_cluster =lambda n, d, eps, data_std: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=lambda n, d, true_mean: gaussian_noise_one_cluster(n, d, true_mean, std=data_std), mean_fun=mean_fun, cov_fun = lambda d: np.eye(d)*data_std**2)
std_gaussian_corruption_two_clusters = lambda n, d, eps, data_std: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=lambda n, d, true_mean: gaussian_noise_two_clusters(n, d, true_mean, std=data_std), mean_fun=mean_fun, cov_fun = lambda d: np.eye(d)*data_std**2)
std_uniform_corruption_top = lambda n, d, eps, data_std: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=lambda n, d, true_mean: uniform_noise_top(n, d, true_mean, std=data_std), mean_fun=mean_fun, cov_fun = lambda d: np.eye(d)*data_std**2)
std_obvious_corruption = lambda n, d, eps, data_std: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=lambda n, d, true_mean: obvious_noise(n, d, true_mean, std=data_std), mean_fun=mean_fun, cov_fun =lambda d: np.eye(d)*data_std**2)

std_gaus_one_small = lambda n, d, eps, data_std: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=lambda n, d, true_mean: obvious_noise(n, d, true_mean, std=data_std/5), mean_fun=mean_fun, cov_fun =lambda d: np.eye(d)*data_std**2)

spherical_vary_std_schemes = {
    "dkk": std_dkk_corruption,
    "gaus_one": std_gaussian_corruption_one_cluster,
    "gaus_two": std_gaussian_corruption_two_clusters,
    "unif_top": std_uniform_corruption_top,
    # "obvious": std_obvious_corruption,
}

# Unknown Non Spherical, Non Diagonal Covariance

var = 25

def dim_diag(d, max_var=1):
    diag = np.linspace(1* max_var, 0.1, d) 
    return diag

# this just uses the defaul max_var
def large_dim_diag(d):
    return dim_diag(d, var)

diag_funs = [large_dim_diag]
meta_nonsp_corruption_schemes = {}

# this loop isn't utilized for our main experiments, but more diagonal functions can be examined using it
for diag_fun in diag_funs:
    cov_fun = lambda d: np.diag(diag_fun(d))
    gaussian_corruption_one_cluster = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=lambda n, d, true_mean: gaussian_noise_one_cluster_nonspherical(n, d, true_mean, diag_fun), mean_fun=mean_fun, cov_fun = cov_fun)
    uniform_corruption_top = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=lambda n, d, true_mean: uniform_noise_top_nonspherical(n, d, true_mean, diag_fun), mean_fun=mean_fun, cov_fun = cov_fun)
    obvious_corruption = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=lambda n, d, true_mean: obvious_noise_nonspherical(n, d, true_mean, diag_fun), mean_fun=mean_fun, cov_fun = cov_fun)
    subtractive_corruption = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=lambda data, eps, true_mean: subtractive_noise_nonspherical(data, eps, true_mean, diag_fun), mean_fun=mean_fun, cov_fun = cov_fun, additive=False)

    corruption_schemes = {
        "gaus_one": gaussian_corruption_one_cluster,
        "unif_top": uniform_corruption_top,
        "obvious": obvious_corruption, # note that this may be too obvious in this case and doesn't align with earlier experiments
        "subtractive_corruption": subtractive_corruption # does subtractive here make sense?
    }

    meta_nonsp_corruption_schemes[diag_fun.__name__] = corruption_schemes

# varying standard deviatino experiments
nonsp_std_gaussian_corruption_one_cluster = lambda n, d, eps, max_std: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=lambda n, d, true_mean: gaussian_noise_one_cluster_nonspherical(n, d, true_mean, lambda d: dim_diag(d, max_var=max_std**2)), mean_fun=mean_fun, cov_fun = lambda d: np.diag(dim_diag(d, max_var=max_std**2)))
nonsp_std_uniform_corruption_top = lambda n, d, eps, max_std: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=lambda n, d, true_mean: uniform_noise_top_nonspherical(n, d, true_mean,  lambda d: dim_diag(d, max_var=max_std**2)), mean_fun=mean_fun, cov_fun = lambda d: np.diag(dim_diag(d, max_var=max_std**2)))
    
nonsp_vary_std_schemes = {
    "gaus_one": nonsp_std_gaussian_corruption_one_cluster,
    "unif_top": nonsp_std_uniform_corruption_top,
}

# Mixture Of Noise Distributions For Hyperparameter Tuning

mix_schemes = {
    "id_dkk": id_dkk_corruption,
    "id_sub": id_subtractive_corruption,
    "id_unif_top": id_uniform_corruption_top,
    "sp_gaus_one": sp_gaussian_corruption_one_cluster, 
    "nonsp_gaus_one": meta_nonsp_corruption_schemes[diag_funs[0].__name__]["gaus_one"]
}




# Declare Algorithms To Evaluate
# main_estimators are used across experiments, the others are used for hyperparameter and other comparisons

main_estimators = {
    "sample_mean": lambda data, tau: sample_mean(data),
    "coord_median": lambda data, tau: coordinate_wise_median(data),
    "coord_trimmed_mean": lambda data, tau: coord_trimmed_mean(data, tau=tau),
    "geometric_median": lambda data, tau: geometric_median(data),
    "lee_valiant_simple": lambda data, tau: lee_valiant_simple(data, lambda data: median_of_means(data, 10), tau=tau),
    "median_of_means": lambda data, tau: median_of_means(data, 10),
    "lrv": lambda data, tau: lrv(data, C=1, trace_est_option="robust"),
    "ev_filtering_low_n": lambda data, tau: eigenvalue_pruning_updated(data, tau=tau, p=0.1, cher=5, t=10),
    "que_low_n": lambda data, tau: que_mean(data, tau=tau, fast=True, stop_early2=False),
    "que_halt": lambda data, tau: que_mean(data, tau=tau, fast=True, stop_early2=True),
    "pgd": lambda data, tau: grad_descent(data, tau=tau, nItr=15)
}


# Some last little tests
# selecting t approrpriately matters -> lower results in first runs of experiments come from t being set incorrectly
eigenvalue_t = {
    "sample_mean": lambda data, tau: sample_mean(data),
    "ev_filtering_low_n-2.5": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5, t=2.5),
    "ev_filtering_low_n-5": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5, t=5),
    "ev_filtering_low_n-10": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5, t=10)
}

# eigenvalue pruning methods

eigenvalue_pruning_routines = {
    "ev_filtering_low_n": lambda data, tau: eigenvalue_pruning_updated(data, tau=tau, p=0.1, cher=5, t=10),
    "ev_filtering_low_n_random": lambda data, tau: eigenvalue_pruning_updated(data, tau=tau, p=0.1, cher=5, t=10, pruning="random"),
    "ev_filtering_low_n_fixed": lambda data, tau: eigenvalue_pruning_updated(data, tau=tau, p=0.1, cher=5, t=10, pruning="fixed") 
}

grad_iterations = {
    "pgd-1": lambda data, tau: grad_descent(data, tau, nItr=1),
    "pgd-5": lambda data, tau: grad_descent(data, tau, nItr=5),
    "pgd-10": lambda data, tau: grad_descent(data, tau, nItr=10),
    "pgd-15": lambda data, tau: grad_descent(data, tau, nItr=15),
    "pgd-20": lambda data, tau: grad_descent(data, tau, nItr=20)
}

ev_que = {
    "sample_mean": lambda data, tau: sample_mean(data),
    "ev_filtering_low_n-5": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5),
    "que": lambda data, tau: que_mean(data, tau)
}

# MAIN ESTIMATORS 
# WEIGHTING LRV BY 10 SCREWS LOTS OF THINGS UP, I'LL JUST USE 1 FOR NOW
main_estimators_old = {
    "sample_mean": lambda data, tau: sample_mean(data),
    "coord_median": lambda data, tau: coordinate_wise_median(data),
    "median_of_means": lambda data, tau: median_of_means(data, 10),
    "coord_trimmed_mean": lambda data, tau: coord_trimmed_mean(data, tau),
    "ransac_mean": lambda data, tau: ransac_mean(data, tau),
    "geometric_median": lambda data, tau: geometric_median(data),
    "lee_valiant_simple": lambda data, tau: lee_valiant_simple(data, lambda data: median_of_means(data, 10), tau),
    "lrv": lambda data, tau: lrv(data, C=1, trace_est_option="robust"),
    "ev_filtering_low_n": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5)
}

lrv_options3 = {
    "lrv": lambda data, tau: lrv(data, C=1, trace_est_option="robust"),
    "lrv_sample": lambda data, tau: lrv(data, C=1, trace_est_option="sample"),
    "lrv-1000": lambda data, tau: lrv(data, C=1000, trace_est_option="robust"),
    "lrv-5": lambda data, tau: lrv(data, C=5, trace_est_option="robust"),
    "lrv-0.1": lambda data, tau: lrv(data, C=0.1, trace_est_option="robust")
}

# LEE AND VALIANT EXPERIMENTS
lee_valiant_comparison = {
    "lee_valiant_simple_mom": lambda data, tau: lee_valiant_simple(data, lambda data: median_of_means(data, 10), tau),
    "lee_valiant_original_mom": lambda data, tau: lee_valiant_original(data, lambda data: median_of_means(data, 10), 0.1, tau),
    "lee_valiant_simple_lrv": lambda data, tau: lee_valiant_simple(data, lambda data: lrv(data), tau),
    "lee_valiant_simple_ev": lambda data, tau: lee_valiant_simple(data, lambda data: eigenvalue_pruning_updated(data, tau, 0.1, 5), tau),
    "lee_valiant_original_lrv": lambda data, tau: lee_valiant_original(data, lambda data: lrv(data), 0.1, tau),
    "lee_valiant_original_ev": lambda data, tau: lee_valiant_original(data, lambda data: eigenvalue_pruning_updated(data, tau, 0.1, 5), 0.3, tau),
    "lrv": lambda data, tau: lrv(data, C=1),
    "ev_filtering_low_n": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5)
}

# NEW VS UPDATED EIGENVALUE EXPERIMENTS
eigenvalue_old_vs_new = {
    "sample_mean": lambda data, tau: sample_mean(data),
    "ev_filtering_low_n": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5),
    "eigenvalue_filtering": lambda data, tau: eigenvalue_pruning_original(data, tau, 0.1, 5)
}

# MEDIAN OF MEANS HYPERPARAMETER TUNING
median_of_means_options = {
    "median_of_means-3": lambda data, tau: median_of_means(data, 3),
    "median_of_means-5": lambda data, tau: median_of_means(data, 5),
    "median_of_means-10": lambda data, tau: median_of_means(data, 10),
    "median_of_means-15": lambda data, tau: median_of_means(data, 15),
    "median_of_means-20": lambda data, tau: median_of_means(data, 20),
    "median_of_means-30": lambda data, tau: median_of_means(data, 30)
}

# LRV HYPERPARAMETER TUNING
lrv_options_C = {
    "lrv-0.5": lambda data, tau: lrv(data, 0.5),
    "lrv-1": lambda data, tau: lrv(data, 1),
    "lrv-5": lambda data, tau: lrv(data, 5),
    "lrv-10": lambda data, tau: lrv(data, 10),
    "lrv-20": lambda data, tau: lrv(data, 20),
    "lrv-50": lambda data, tau: lrv(data, 50)    
}

# change to use best lrv - not 10
lrv_options_general_gaus = {
    "lrv": lambda data, tau: lrv(data, 1),
    "lrv_general": lambda data, tau: lrvGeneral(data, tau)
}


# With no data scaling first
# DKK: Near identical, 3's slightly worse
# Gaus One: Near identical, 3's slightly worse but better with more data size
# Unif Top: 3's and 1's are similar for both, 3's are better
# Subtractive: All pretty similar
# Obvious: Sample does meaningfully better with lower data size, then nearly identical - overall sample appears to be the better choice
    #  Hypothesis: Sample results in much larger C value, equivalent to much larger C
    # Results aren't super conclusive, but this seems to be the gist
# Obvious Hard: 3's are meaningfully worse, but 1's for both are about the same
# Gaus Small: 3's are slightly worse (except with enough data), 1's are similar with sample being better with higher corruption
lrv_options_trace_est_variations = {
    "lrv_sample_1": lambda data, tau: lrv(data, C=1, trace_est_option="sample"),
    # "lrv_sample_3": lambda data, tau: lrv(data, C=3, trace_est_option="sample"),
    "lrv_1": lambda data, tau: lrv(data, C=1, trace_est_option="robust"),
    # "lrv_3": lambda data, tau: lrv(data, C=3, trace_est_option="robust"),
    "lrv_100": lambda data, tau: lrv(data, C=100, trace_est_option="robust")
}

# EV HYPERPARAMETER TUNING
ev_options = {
    # "sample_mean": lambda data, tau: sample_mean(data),
    "ev_filtering_low_n-0.5": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 0.5),
    "ev_filtering_low_n-1": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 1),
    "ev_filtering_low_n-2.5": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 2.5),
    "ev_filtering_low_n-5": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5),
    "ev_filtering_low_n-10": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 10),
    "ev_filtering_low_n-20": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 20),
    "ev_filtering_low_n-50": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 50),
}



# All of the below is not necessary anymore as we use the same estimators for everything - scaling data first instead

# Same names for now, might be bad
ev_unknown_cov_options = {
    "eigenvalue_filtering_highd0.5": lambda data, tau: eigenvalue_pruning_unknown_cov(data, tau, 0.1, 0.5),
    "eigenvalue_filtering_highd1": lambda data, tau: eigenvalue_pruning_unknown_cov(data, tau, 0.1, 1),
    "eigenvalue_filtering_highd2.5": lambda data, tau: eigenvalue_pruning_unknown_cov(data, tau, 0.1, 2.5),
    "eigenvalue_filtering_highd5": lambda data, tau: eigenvalue_pruning_unknown_cov(data, tau, 0.1, 5),
    "eigenvalue_filtering_highd10": lambda data, tau: eigenvalue_pruning_unknown_cov(data, tau, 0.1, 10),
    "eigenvalue_filtering_highd20": lambda data, tau: eigenvalue_pruning_unknown_cov(data, tau, 0.1, 20),
    "eigenvalue_filtering_highd50": lambda data, tau: eigenvalue_pruning_unknown_cov(data, tau, 0.1, 50),
}

# UNKNOWN COVARIANCE EXPERIMENTS
main_estimators_unknown_cov = {
    "sample_mean": lambda data, tau: sample_mean(data),
    "coord_median": lambda data, tau: coordinate_wise_median(data),
    "median_of_means": lambda data, tau: median_of_means(data, 10),
    "coord_trimmed_mean": lambda data, tau: coord_trimmed_mean(data, tau),
     "ransac_mean": lambda data, tau: ransac_mean_unknown_cov(data, tau),
    "ransac_mean": lambda data, tau: ransac_mean(data, tau, identity=False),
    "geometric_median": lambda data, tau: geometric_median(data),
    "lee_valiant_simple_mom": lambda data, tau: lee_valiant_simple(data, lambda data: median_of_means(data, 10), tau),
    "lrv": lambda data, tau: lrv(data, C=1),
    "eigenvalue_pruning_unknown_cov": lambda data, tau: eigenvalue_pruning_unknown_cov(data, tau, 0.1, 5, option=1),
    # "eigenvalue_pruning_unknown_cov": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5)
    # "eigenvalue_pruning_scale_data": lambda data, tau: new_eigenvalue_pruning_unknown(data, tau, 0.1, 5, option=1)
}

# UNKNOWN COVARIANCE NONSPHERICAL
main_estimators_unknown_non_sp = {
    "sample_mean": lambda data, tau: sample_mean(data),
    "coord_median": lambda data, tau: coordinate_wise_median(data),
    "median_of_means": lambda data, tau: median_of_means(data, 10),
    "coord_trimmed_mean": lambda data, tau: coord_trimmed_mean(data, tau),
    # "ransac_mean": lambda data, tau: ransac_mean_unknown_cov(data, tau),
    "ransac_mean": lambda data, tau: ransac_mean(data, tau, identity=False),
    "geometric_median": lambda data, tau: geometric_median(data),
    "lee_valiant_simple_mom": lambda data, tau: lee_valiant_simple(data, lambda data: median_of_means(data, 10), tau),
    "lrv": lambda data, tau: lrv(data, C=10),
    # "eigenvalue_pruning_unknown_cov": lambda data, tau: eigenvalue_pruning_unknown_cov(data, tau, 0.1, 5, option=1)
}




# lrv_base_case_test = {
#     "lrv": lambda data, tau: lrv(data, tau),
#     # "lrv2": lambda data, tau: lrv2(data, tau)
# }

# POSSIBLE MORE RECENT LV SECTION
# LV SIMPLE BETTER THAN LV ORIGINAL
# lee_valiant_simple_original = {
#     "lee_valiant_simple_mom": lambda data, tau: lee_valiant_simple(data, lambda data: median_of_means(data, 10), tau),
#     "lee_valiant_original_mom": lambda data, tau: lee_valiant_original(data, lambda data: median_of_means(data, 10), 0.1, tau),
# }

# # LV SIMPLE GETS NO BENEFIT FROM ROBUST ESTIMATORS
# lee_valiant_simple_robust_estimators = {
#     "lee_valiant_simple_mom": lambda data, tau: lee_valiant_simple(data, lambda data: median_of_means(data, 10), tau),
#     "lee_valiant_simple_lrv": lambda data, tau: lee_valiant_simple(data, lambda data: lrv(data, tau), tau),
#     "lee_valiant_original_ev": lambda data, tau: lee_valiant_original(data, lambda data: eigenvalue_pruning_updated(data, tau, 0.1, 5), 0.3, tau),
#     "lrv": lambda data, tau: lrv(data, tau),
#     "updated_threshold_ev": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5)
# }

# # LV ORIGINAL GETS BENEFIT BUT NO BETTER THAN THOSE ROBUST ESTIMATORS ALONE
# lee_valiant_original_robust_estimators = {
#     "lee_valiant_original_mom": lambda data, tau: lee_valiant_original(data, lambda data: median_of_means(data, 10), 0.1, tau),
#     "lee_valiant_original_lrv": lambda data, tau: lee_valiant_original(data, lambda data: lrv(data, tau), 0.1, tau),
#     "lee_valiant_original_ev": lambda data, tau: lee_valiant_original(data, lambda data: eigenvalue_pruning_updated(data, tau, 0.1, 5), 0.3, tau),
#      "lrv": lambda data, tau: lrv(data, tau),
#     "updated_threshold_ev": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5)
# }

# unknown_cov_ev = {
#     "sample_mean": lambda data, tau: sample_mean(data),
#     # "unknown_cov_ev_var": lambda data, tau: eigenvalue_pruning_unknown_spherical(data, tau, 0.1, 5, option=1),
#     # "unknown_cov_ev_std": lambda data, tau: eigenvalue_pruning_unknown_spherical(data, tau, 0.1, 5, option=2),
#     # "unknown_cov_ev_var2": lambda data, tau: eigenvalue_pruning_unknown_spherical(data, tau, 0.1, 5, option=3)
#     "dkk_attempt": lambda data, tau: eigenvalue_pruning_unknown_cov(data, tau, 0.1, 5, option=1),
#     "dkk_attempt_maxvar": lambda data, tau: eigenvalue_pruning_unknown_cov(data, tau, 0.1, 5, option=2)
# }

# main_estimator_w_fite = {
#     "sample_mean": lambda data, tau: sample_mean(data),
#     "median_of_means": lambda data, tau: median_of_means(data, 10),
#     "lrv": lambda data, tau: lrv(data, tau),
#     "updated_threshold_ev": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5),
#     "fite_sample": lambda data, tau: fite(data, sample_mean, tau),
#     "fite_lrv": lambda data, tau: fite(data, lambda data: lrv(data, tau), tau)
# }



# # the original does just not work great here - but not catostrophic

# # simple_estimators = {
# #     "sample_mean": lambda data, tau: sample_mean(data),
# #     "coordinate_wise_median": lambda data, tau: coordinate_wise_median(data),
# #     "median_of_means": lambda data, tau: median_of_means(data, 10),
# #     "naive_prune": lambda data, tau: naive_prune_no_tau(data),
# #     "ransac_mean": lambda data, tau: ransac_mean(data, tau),
# #     "geometric_median": lambda data, tau: geometric_median(data)
# # }

# best_estimators = {
#     "median_of_means": lambda data, tau: median_of_means(data, 10),
#     "lrv": lambda data, tau: lrv(data, tau),
#     "updated_threshold_ev": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5)
# }

# # estimators2 = {
# #     "sample_mean": lambda data, tau: sample_mean(data),
# #     "naive_prune": lambda data, tau: naive_prune_no_tau(data),
# #     "median_of_means": lambda data, tau: median_of_means(data, 10),
# #     "lrv": lambda data, tau: lrv(data, tau),
# #     "updated_threshold_ev": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5)
# # }

# # main_estimators_prune = {
# #     "sample_mean": lambda data, tau: sample_mean(data),
# #     "coordinate_wise_median": lambda data, tau: coordinate_wise_median(data),
# #     "median_of_means": lambda data, tau: median_of_means(data, 10),
# #     # "naive_prune_tau": lambda data, tau: naive_prune0(data, tau),
# #     "naive_prune_no_tau": lambda data, tau: naive_prune_no_tau(data),
# #     # "ransac_mean": lambda data, tau: ransac_mean(data, tau),
# #     # "geometric_median": lambda data, tau: geometric_median(data),
# #     "lee_valiant_simple": lambda data, tau: lee_valiant_simple(data, lambda data: median_of_means(data, 10), tau),
# #     # "lee_valiant_original": lambda data, tau: lee_valiant_original(data, lambda data: median_of_means(data, 10), 0.3, tau),
# #     "lrv": lambda data, tau: lrv(data, tau),
# #     "updated_threshold_ev": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5)
# # }

# main_estimators_no_sample = {
#     "coordinate_wise_median": lambda data, tau: coordinate_wise_median(data),
#     "median_of_means": lambda data, tau: median_of_means(data, 10),
#     # "naive_prune": lambda data, tau: naive_prune(data, tau),
#     # "ransac_mean": lambda data, tau: ransac_mean(data, tau),
#     "geometric_median": lambda data, tau: geometric_median(data),
#     "lee_valiant_simple": lambda data, tau: lee_valiant_simple(data, lambda data: median_of_means(data, 10), tau),
#     # "lee_valiant_original": lambda data, tau: lee_valiant_original(data, lambda data: median_of_means(data, 10), 0.3, tau),
#     "lrv": lambda data, tau: lrv(data, tau),
#     "updated_threshold_ev": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5)
# }

# # main_estimators_spherical = {
# #     "sample_mean": lambda data, tau: sample_mean(data),
# #     "coordinate_wise_median": lambda data, tau: coordinate_wise_median(data),
# #     "median_of_means": lambda data, tau: median_of_means(data, 10),
# #     "geometric_median": lambda data, tau: geometric_median(data),
# #     "naive_prune": lambda data, tau: naive_prune_spherical_no_tau(data),
# #     "lee_valiant_simple": lambda data, tau: lee_valiant_simple(data, lambda data: median_of_means(data, 10), tau),
# #     "lee_valiant_simple_lrv": lambda data, tau: lee_valiant_simple(data, lambda data: lrv(data, tau), tau),
# #     "lee_valiant_original": lambda data, tau: lee_valiant_original(data, lambda data: median_of_means(data, 10), 0.1, tau),
# #     "lee_valiant_original_lrv": lambda data, tau: lee_valiant_original(data, lambda data: lrv(data, tau), 1, tau),
# #     "lrv": lambda data, tau: lrv(data, tau),
# #     "spherical_ev": lambda data, tau: eigenvalue_pruning_unknown_spherical(data, tau, 0.1, 5)
# # }

# best_estimators_spherical = {
#     "sample_mean": lambda data, tau: sample_mean(data),
#      "lrv": lambda data, tau: lrv(data, tau),
#      "lee_valiant_simple_lrv": lambda data, tau: lee_valiant_simple(data, lambda data: lrv(data, tau), tau),
#      "median_of_means": lambda data, tau: median_of_means(data, 10)
# }

# spherical_eigenvalue_attempts = {
#     "sample_mean": lambda data, tau: sample_mean(data),
#     "spherical_ev_var": lambda data, tau: eigenvalue_pruning_unknown_spherical(data, tau, 0.1, 5, option=0),
#     "spherical_ev_var2": lambda data, tau: eigenvalue_pruning_unknown_spherical(data, tau, 0.1, 5, option=1),
#     # "unknown_cov_ev":lambda data, tau: eigenvalue_pruning_unknown_cov(data, tau, 0.1, 5)
# }

# spherical_eigenvalue_attempts2 = {
#     "sample_mean": lambda data, tau: sample_mean(data),
#     "ev_known": lambda data, tau: new_eigenvalue_pruning_unknown(data, tau, 0.1, 5, option=2),
#     "ev_unknown": lambda data, tau: new_eigenvalue_pruning_unknown(data, tau, 0.1, 5, option=1)
# }

# lee_valiant_simple_options = {
#     "lee_valiant_simple_mom3": lambda data, tau: lee_valiant_simple(data, lambda data: median_of_means(data, 3), tau),
#     "lee_valiant_simple_mom10": lambda data, tau: lee_valiant_simple(data, lambda data: median_of_means(data, 10), tau),
#     "lee_valiant_simple_ev": lambda data, tau: lee_valiant_simple(data, lambda data: eigenvalue_pruning_updated(data, tau, 0.1, 2.5), tau),
#     "lee_valiant_simple_lrv": lambda data, tau: lee_valiant_simple(data, lambda data: lrv(data, tau), tau),
#     "lee_valiant_simple_geomed": lambda data, tau: lee_valiant_simple(data, lambda data: geometric_median(data), tau),
#     "lee_valiant_simple_ransac": lambda data, tau: lee_valiant_simple(data, lambda data: ransac_mean(data, tau), tau),
#     "lee_valiant_simple_coordmed": lambda data, tau: lee_valiant_simple(data, lambda data: coordinate_wise_median(data), tau),
#     "lee_valiant_simple_sample": lambda data, tau: lee_valiant_simple(data, lambda data: sample_mean(data), tau)
# }

# lee_valiant_original_options1 = {
#     "lee_valiant_original_mom3": lambda data, tau: lee_valiant_original(data, lambda data: median_of_means(data, 3), 0.3, tau),
#     "lee_valiant_original_mom10": lambda data, tau: lee_valiant_original(data, lambda data: median_of_means(data, 10), 0.3, tau),
#     "lee_valiant_original_ev": lambda data, tau: lee_valiant_original(data, lambda data: eigenvalue_pruning_updated(data, tau, 0.1, 2.5), 0.3, tau),
#     "lee_valiant_original_lrv": lambda data, tau: lee_valiant_original(data, lambda data: lrv(data, tau), 0.3, tau),
#     "lee_valiant_original_geomed": lambda data, tau: lee_valiant_original(data, lambda data: geometric_median(data), 0.3, tau),
#     "lee_valiant_original_ransac": lambda data, tau: lee_valiant_original(data, lambda data: ransac_mean(data, tau), 0.3, tau),
#     "lee_valiant_original_coordmed": lambda data, tau: lee_valiant_original(data, lambda data: coordinate_wise_median(data), 0.3, tau),
#     "lee_valiant_original_sample": lambda data, tau: lee_valiant_original(data, lambda data: sample_mean(data), 0.3, tau)
# }

# lee_valiant_original_mom_options = {
#     "lee_valiant_original_mom0.1": lambda data, tau: lee_valiant_original(data, lambda data: median_of_means(data, 10), 0.1, tau),
#     "lee_valiant_original_mom0.3": lambda data, tau: lee_valiant_original(data, lambda data: median_of_means(data, 10), 0.3, tau),
#     "lee_valiant_original_mom0.5": lambda data, tau: lee_valiant_original(data, lambda data: median_of_means(data, 10), 0.5, tau),
#     "lee_valiant_original_mom0.7": lambda data, tau: lee_valiant_original(data, lambda data: median_of_means(data, 10), 0.7, tau),
#     "lee_valiant_original_mom0.9": lambda data, tau: lee_valiant_original(data, lambda data: median_of_means(data, 10), 0.9, tau),
#     "lee_valiant_original_mom1": lambda data, tau: lee_valiant_original(data, lambda data: median_of_means(data, 10), 1, tau),
# }

# lee_valiant_original_lrv_options = {
#     "lee_valiant_original_lrv0.1": lambda data, tau: lee_valiant_original(data, lambda data: lrv(data, tau), 0.1, tau),
#     "lee_valiant_original_lrv0.3": lambda data, tau: lee_valiant_original(data, lambda data: lrv(data, tau), 0.3, tau),
#     "lee_valiant_original_lrv0.5": lambda data, tau: lee_valiant_original(data, lambda data: lrv(data, tau), 0.5, tau),
#     "lee_valiant_original_lrv0.7": lambda data, tau: lee_valiant_original(data, lambda data: lrv(data, tau), 0.7, tau),
#     "lee_valiant_original_lrv0.9": lambda data, tau: lee_valiant_original(data, lambda data: lrv(data, tau), 0.9, tau),
#     "lee_valiant_original_lrv1": lambda data, tau: lee_valiant_original(data, lambda data: lrv(data, tau), 1, tau),
# }

# eigenvalue_pruning_options = {
#     "updated_threshold_ev-30": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 30),
#     "updated_threshold_ev-15": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 15),
#     "updated_threshold_ev-10": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 10),
#     "updated_threshold_ev-7.5": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 7.5),
#     "updated_threshold_ev-5": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5),
#     "updated_threshold_ev-3.5": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 3.5),
#     "updated_threshold_ev-2.5": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 2.5),
#     "updated_threshold_ev-1.5": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 1.5),
#     "updated_threshold_ev-0.5": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 0.5)
# }

# lrv_options = {
#     "sample_mean": lambda data, tau: sample_mean(data),
#     "lrv": lambda data, tau: lrv(data, tau),
#     "lrv_prune": lambda data, tau: lrv_prune(data, tau)
# }

# Estimators For Uncorrupted Data
# Include Lee & Valiant as Constructed In The Original Paper (median of means, weighting)
# Include our simplified Lee & Valiant
# Also throw in other options for estimators (not corrupted ones -> doesn't help here) - but do mention failure in other cases after
# Sample mean is a good baseline to include
# For an alternate one also include other uncorrupted estimators
# uncorrupted_estimators1 = {
#     "sample_mean": lambda data, tau: sample_mean(data),
#     "lee_valiant_original_mom": lambda data, tau: lee_valiant_original(data, lambda data: median_of_means(data, 10), 0.1, tau),
#     "lee_valiant_simple_mom": lambda data, tau: lee_valiant_simple(data, lambda data: median_of_means(data, 10), tau),
# }


# estimators_tau = {
#     "lee_valiant_simple": lambda data, tau: lee_valiant_simple(data, lambda data: median_of_means(data, 10), tau),
#     "lee_valiant_simple_mom": lambda data, tau: lee_valiant_simple(data, lambda data: median_of_means(data, 10), tau),
#     "lee_valiant_simple_ev": lambda data, tau: lee_valiant_simple(data, lambda data: eigenvalue_pruning_updated(data, tau, 0.1, 2.5), tau),
#     "lee_valiant_original_mom": lambda data, tau: lee_valiant_original(data, lambda data: median_of_means(data, 10), 0.1, tau),
#     "lee_valiant_original_lrv": lambda data, tau: lee_valiant_original(data, lambda data: lrv(data, tau), 1, tau),
#     "lrv": lambda data, tau: lrv(data, tau),
#     "updated_threshold_ev": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5)
# }

# main_estimators = {
#     "sample_mean": lambda data, tau: sample_mean(data),
#     "coordinate_wise_median": lambda data, tau: coordinate_wise_median(data),
#     "median_of_means": lambda data, tau: median_of_means(data, 10),
#     "naive_prune": lambda data, tau: naive_prune_no_tau(data),
#     "ransac_mean": lambda data, tau: ransac_mean(data, tau),
#     "geometric_median": lambda data, tau: geometric_median(data),
#     "lee_valiant_simple_mom": lambda data, tau: lee_valiant_simple(data, lambda data: median_of_means(data, 10), tau),
#     "lee_valiant_simple_ev": lambda data, tau: lee_valiant_simple(data, lambda data: eigenvalue_pruning_updated(data, tau, 0.1, 2.5), tau),
#     "lee_valiant_original_mom": lambda data, tau: lee_valiant_original(data, lambda data: median_of_means(data, 10), 0.1, tau),
#     "lee_valiant_original_lrv": lambda data, tau: lee_valiant_original(data, lambda data: lrv(data, tau), 1, tau),
#     "lrv": lambda data, tau: lrv(data, tau),
#     "updated_threshold_ev": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5)
# }

# just zoomed estimators with lee_valiant_original
# uncorrupted_estimators = {
#     "sample_mean": lambda data, tau: sample_mean(data),
#     "coord_median": lambda data, tau: coordinate_wise_median(data),
#     "median_of_means": lambda data, tau: median_of_means(data, 10),
#     "coord_trimmed_mean": lambda data, tau: coord_trimmed_mean(data, tau),
#     "ransac_mean": lambda data, tau: ransac_mean(data, tau),
#     "geometric_median": lambda data, tau: geometric_median(data),
#     "lee_valiant_simple_mom": lambda data, tau: lee_valiant_simple(data, lambda data: median_of_means(data, 10), tau),
#     "lee_valiant_original_mom": lambda data, tau: lee_valiant_original(data, lambda data: median_of_means(data, 10), 0.1, tau),
#     "lrv": lambda data, tau: lrv(data, tau),
#     "updated_threshold_ev": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5)
# }