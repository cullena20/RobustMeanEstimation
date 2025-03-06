"""
Setup various utilities to perform experiments:
 - Lay out parameters and default values for experiments
 - Define corrupted data schemes
 - Define estimators
"""
import sys
import os

# Add the 'Algorithms' and "DataGeneration" directories to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
algorithms_dir = os.path.abspath(os.path.join(current_dir, '../Algorithms'))
data_dir = os.path.abspath(os.path.join(current_dir, '../DataGeneration'))
sys.path.append(algorithms_dir)
sys.path.append(data_dir)

# Ensure that Algorithms and DataGeneration are in the path

from eigenvalue_pruning import eigenvalue_pruning
from lee_valiant import lee_valiant_original, lee_valiant_simple
from lrv import lrv, lrvGeneral
from simple_estimators import sample_mean, coordinate_wise_median, median_of_means, geometric_median, coord_trimmed_mean
from que import que_mean
from pgd import grad_descent
from sdp import sdp_mean

from data_generation import generate_data_helper
from inlier_data import gaussian_data, t_data, gaussian_mixture_data, uniform_multinomial_data, laplace_data, poisson_data
from corruption_schemes import dkk_noise, gaussian_noise_one_cluster, gaussian_noise_two_clusters, uniform_noise_top, uniform_noise_whole, obvious_noise, subtractive_noise, obvious_easy, obvious_hard_two_clusters, multiple_corruption
from corruption_schemes import gaussian_noise_one_cluster_nonspherical, obvious_noise_nonspherical, subtractive_noise_nonspherical, uniform_noise_top_nonspherical, uniform_mulitnomial_one_cluster, mixture_of_gaussians_noise


import numpy as np
import math

# DEFAULT VALUES
default_n = 500
default_d = 500
default_eta = 0.1
default_tau = None # tau = None means that we will use tau=eta
mean_fun = lambda d: np.ones(d) * 5 


# EXPERIMENTS

# the main experiment used throughout
experiments = [["n", np.arange(20, 5021, 500), None, default_d, default_eta, default_tau],
               ["n", np.arange(20, 521, 50), None, default_d, default_eta, default_tau],
               ["d", np.arange(20, 1021,  100), default_n, None, default_eta, default_tau],
               ["eta", np.arange(0, 0.46, 0.05), default_n, default_d, None, default_tau]]

# experiment over uncorrupted data
uncorrupted_experiments = [["n", np.arange(20, 5021, 500), None, default_d, 0, 0.1],
                            ["n", np.arange(20, 521, 50), None, default_d, 0, 0.1],
                            ["d", np.arange(20, 1021,  100), default_n, None, 0, 0.1],
                            ["tau", np.arange(0, 0.46, 0.05), default_n, default_d, 0, None]]

# robustness to true corruption
tau_experiments = [["tau", np.arange(0.01, 0.46, 0.05), default_n, default_d, 0.2, None]]

# vary top standard deviation or squareroot of top eigenvalue 
std_experiments = [["data", np.arange(0.1, 201, 25), default_n, default_d, default_eta, default_tau]]

# used to compare ev_filtering and que low_n vs original threshold
ev_comparison_experiment = [["n", np.arange(20, 10021, 2500), None, default_d, default_eta, default_tau]]


# Below are various experiments not presented

# main experiments shorter
experiments_short = [
    ["n", np.arange(20, 5021, 1500), None, default_d, default_eta, default_tau],
    ["n", np.arange(20, 521, 150), None, default_d, default_eta, default_tau],
    ["d", np.arange(20, 1021, 300), default_n, None, default_eta, default_tau],
    ["eta", np.arange(0, 0.46, 0.15), default_n, default_d, None, default_tau]
]

# main top std/squareroot of top eigenvalue experiment shorter
std_experiments_short = [["data", np.arange(0.1, 201, 50), default_n, default_d, default_eta, default_tau]]

# not used - gather time estimates which do not require the zoomed in data size
time_experiments = [["n", np.arange(20, 5021, 500), None, default_d, default_eta, default_tau],
                    ["d", np.arange(20, 1021,  100), default_n, None, default_eta, default_tau],
                    ["eta", np.arange(0, 0.46, 0.05), default_n, default_d, None, default_tau]]

# smaller data and dimensionality
old_experiments = [["n", np.arange(20, 3001, 100), None, 150, 0.1, None],
               ["n", np.arange(20, 300, 50), None, 150, 0.1, None],
               ["d", np.arange(20, 301, 10), 150, None, 0.1, None],
               ["eta", np.arange(0.01, 0.32, 0.05), 150, 150, None, None]]

# abridged smaller data and dimensionality
old_experiments_short = [["n", np.arange(20, 3021, 500), None, 150, 0.1, None],
               ["n", np.arange(20, 321, 80), None, 150, 0.1, None],
               ["d", np.arange(20, 321, 50), 150, None, 0.1, None],
               ["eta", np.arange(0.01, 0.32, 0.1), 150, 150, None, None]]

# experiment over uncorrupted data
uncorrupted_experiments_short = [["n", np.arange(20, 3021, 500), None, 150, 0.1, None],
               ["n", np.arange(20, 321, 80), None, 150, 0.1, None],
               ["d", np.arange(20, 321, 50), 150, None, 0.1, None],
                            ["tau", np.arange(0, 0.46, 0.1), 150, 150, 0, None]]

# very large dimensionality
huge_experiments = [["d", np.arange(20, 10021, 500), default_n, None, default_eta, default_tau],
                    ["eta", np.arange(0.01, 0.42, 0.05), 10000, 10000, None, default_tau]]

# slightly larger data size than 
alt_experiments = [["n", np.arange(20, 10021, 500), None, default_d, default_eta, default_tau],
            ["n", np.arange(20, 1021, 50), None, default_d, default_eta, default_tau],
            ["d", np.arange(20, 1021,  100), default_n, None, default_eta, default_tau],
            ["eta", np.arange(0, 0.46, 0.05), default_n, default_d, None, default_tau]]

# DATA SCHEMES

# IDENTITY COVARIANCE NO CORRUPTION
uncorrupted_data_scheme = lambda n, d, eta: gaussian_data(n, d, mean_fun=mean_fun)

# HEAVY TAILED unCCORUTPED
heavy_tail_uncorr_scheme = lambda n, d, eta: t_data(n, d, mean_fun=mean_fun)

# IDENTITY COVARIANCE CORRUPTION
# cov_fun is identity by default in generate_data_helper

# corruption from dkk paper, hard for naive methods
id_dkk_corruption = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=dkk_noise, mean_fun=mean_fun)

# cluster corruption sqrt(d) away from true mean, hard for naive methods
id_gaussian_corruption_one_cluster = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=gaussian_noise_one_cluster, mean_fun=mean_fun)

# two clusters of corruption sqrt(d) away from true mean, 45 degrees between each other, 70% one cluster 30% the other, hard for naive methods
id_gaussian_corruption_two_clusters = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=gaussian_noise_two_clusters, mean_fun=mean_fun)

# uniform corruption from true_mean to true_mean + 1 in every coordinate, hard for naive methods
id_uniform_corruption_top = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=uniform_noise_top, mean_fun=mean_fun)

# uniformly place corruption within sqrt(d) from true mean, not hard for naive methods
id_uniform_corruption_whole = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=uniform_noise_whole, mean_fun=mean_fun)

# place two clusters of corruption 10sqrt(d) away from true mean and 20sqrt(d) away from true mean, easy for naive methods - show good methods in hard cases work in easy ones too
id_obvious_corruption = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=obvious_noise, mean_fun=mean_fun)

# place two clusters - one sqrt(d) from the true mean, one 10 * sqrt(d) away from the true mean
id_obvious_hard_corruption = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=obvious_hard_two_clusters, mean_fun=mean_fun)

# subtract etailon largest points from the projection in a certain direction - nothing performs much worse than sample mean here
id_subtractive_corruption = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=subtractive_noise, mean_fun=mean_fun, additive=False)

# One Easy Cluster One Hard Cluster
id_obvious_easy_corruption = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=obvious_easy, mean_fun=mean_fun)

# Large and Subtle Outlier Mixes
id_obvious_gaus_one_corruption = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun= lambda n, d, true_mean: multiple_corruption(n, d, true_mean, obvious_noise, gaussian_noise_one_cluster), mean_fun=mean_fun)
id_obvious_dkk_corruption = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun= lambda n, d, true_mean: multiple_corruption(n, d, true_mean, obvious_noise, dkk_noise), mean_fun=mean_fun)

# Multinomial T Distribution - One Cluster Noise
nongauss_multinomial_cluster = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=t_data, corruption_fun=gaussian_noise_one_cluster, mean_fun=mean_fun)

# Multinomial T Distribution - Uniform Top Noise
nongauss_multinomial_uniform_top = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=t_data, corruption_fun=uniform_noise_top, mean_fun=mean_fun)

# Mixture Of Gaussians - Additional Gaussian
nongauss_mixture_corruption = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_mixture_data, corruption_fun=mixture_of_gaussians_noise, mean_fun=None)

# Uniform Multinomial Data
nongauss_multinomial_far_cluster = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=uniform_multinomial_data, corruption_fun=gaussian_noise_one_cluster, mean_fun=mean_fun)
nongauss_multinomial_close_cluster = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=uniform_multinomial_data, corruption_fun=uniform_mulitnomial_one_cluster, mean_fun=mean_fun)

# Laplace Data
nongauss_laplace_cluster = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=laplace_data, corruption_fun=gaussian_noise_one_cluster, mean_fun=mean_fun)

# Poisson Data
nongauss_poisson_cluster = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=poisson_data, corruption_fun=gaussian_noise_one_cluster, mean_fun=mean_fun)

# multinomial
nongauss_poisson_cluster = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=poisson_data, corruption_fun=gaussian_noise_one_cluster, mean_fun=mean_fun)


identity_corruption_schemes = {  "dkk": id_dkk_corruption,
                        "gaus_one": id_gaussian_corruption_one_cluster,
                        "gaus_two": id_gaussian_corruption_two_clusters,
                        "unif_top": id_uniform_corruption_top,
                        "obvious": id_obvious_corruption,
                        "subtractive_corruption": id_subtractive_corruption
}

large_outlier_identity_corruption_schemes = {
    "obvious": id_obvious_corruption,
    "obvious_and_gaus_one": id_obvious_gaus_one_corruption,
    "obvious_and_dkk": id_obvious_dkk_corruption
}

# Dependence On True Mean - Identity Covariance
gaus_one_random = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=gaussian_noise_one_cluster, mean_fun= lambda d: np.random.randn(d) * 50)
dkk_random = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=dkk_noise, mean_fun= lambda d: np.random.randn(d) * 50)

identity_true_mean_dependence_schemes = {
    "gaus_one_random": gaus_one_random,
    "dkk_random": dkk_random
}

# NON GAUSSIAN DATA

nongauss_corruption_schemes = {
    "t_far": nongauss_multinomial_cluster,
    "mix_gaussian": nongauss_mixture_corruption,
    #"uniform_multi_far": nongauss_multinomial_far_cluster,
    #"uniform_multi_close": nongauss_multinomial_close_cluster,
    #"t_top": nongauss_multinomial_uniform_top,
    "laplace": nongauss_laplace_cluster,
    "poisson": nongauss_poisson_cluster,
}

# UNKNOWN SPHERICAL COVARIANCE

var = 25
std = math.sqrt(var)
cov_fun = lambda d: np.eye(d) * var

# same as identity schemes but scale all noise locations appropriately
sp_dkk_corruption = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun= lambda n, d, true_mean: dkk_noise(n, d, true_mean, std=std), mean_fun=mean_fun, cov_fun = cov_fun)
sp_gaussian_corruption_one_cluster = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=lambda n, d, true_mean: gaussian_noise_one_cluster(n, d, true_mean, std=std), mean_fun=mean_fun, cov_fun = cov_fun)
sp_gaussian_corruption_two_clusters = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=lambda n, d, true_mean: gaussian_noise_two_clusters(n, d, true_mean, std=std), mean_fun=mean_fun, cov_fun = cov_fun)
sp_uniform_corruption_top = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=lambda n, d, true_mean: uniform_noise_top(n, d, true_mean, std=std), mean_fun=mean_fun, cov_fun = cov_fun)
sp_obvious_corruption = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=lambda n, d, true_mean: obvious_noise(n, d, true_mean, std=std), mean_fun=mean_fun, cov_fun = cov_fun)
sp_subtractive_corruption = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=subtractive_noise, mean_fun=mean_fun, cov_fun = cov_fun, additive=False)

# this isn't used and isn't so interesting
sp_uniform_corruption_whole = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=lambda n, d, true_mean: uniform_noise_whole(n, d, true_mean, std=std), mean_fun=mean_fun, cov_fun = cov_fun)

spherical_corruption_schemes = {
    "dkk": sp_dkk_corruption,
    "gaus_one": sp_gaussian_corruption_one_cluster,
    "gaus_two": sp_gaussian_corruption_two_clusters,
    "unif_top": sp_uniform_corruption_top,
    "obvious": sp_obvious_corruption,
    "subtractive_corruption": sp_subtractive_corruption
}

# experiment with adjusting standard deviation - var parameter is not used here
std_dkk_corruption = lambda n, d, eta, data_std: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun= lambda n, d, true_mean: dkk_noise(n, d, true_mean, std=data_std), mean_fun=mean_fun, cov_fun = lambda d: np.eye(d)*data_std**2)
std_gaussian_corruption_one_cluster =lambda n, d, eta, data_std: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=lambda n, d, true_mean: gaussian_noise_one_cluster(n, d, true_mean, std=data_std), mean_fun=mean_fun, cov_fun = lambda d: np.eye(d)*data_std**2)
std_gaussian_corruption_two_clusters = lambda n, d, eta, data_std: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=lambda n, d, true_mean: gaussian_noise_two_clusters(n, d, true_mean, std=data_std), mean_fun=mean_fun, cov_fun = lambda d: np.eye(d)*data_std**2)
std_uniform_corruption_top = lambda n, d, eta, data_std: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=lambda n, d, true_mean: uniform_noise_top(n, d, true_mean, std=data_std), mean_fun=mean_fun, cov_fun = lambda d: np.eye(d)*data_std**2)
std_obvious_corruption = lambda n, d, eta, data_std: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=lambda n, d, true_mean: obvious_noise(n, d, true_mean, std=data_std), mean_fun=mean_fun, cov_fun =lambda d: np.eye(d)*data_std**2)

std_gaus_one_small = lambda n, d, eta, data_std: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=lambda n, d, true_mean: obvious_noise(n, d, true_mean, std=data_std/5), mean_fun=mean_fun, cov_fun =lambda d: np.eye(d)*data_std**2)

spherical_vary_std_schemes = {
    "dkk": std_dkk_corruption,
    "gaus_one": std_gaussian_corruption_one_cluster,
    "gaus_two": std_gaussian_corruption_two_clusters,
    "unif_top": std_uniform_corruption_top,
    "obvious": std_obvious_corruption,
}

# UNKNOWN NONSPHERICAL COVARIANCE

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
    gaussian_corruption_one_cluster = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=lambda n, d, true_mean: gaussian_noise_one_cluster_nonspherical(n, d, true_mean, diag_fun), mean_fun=mean_fun, cov_fun = cov_fun)
    uniform_corruption_top = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=lambda n, d, true_mean: uniform_noise_top_nonspherical(n, d, true_mean, diag_fun), mean_fun=mean_fun, cov_fun = cov_fun)
    obvious_corruption = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=lambda n, d, true_mean: obvious_noise_nonspherical(n, d, true_mean, diag_fun), mean_fun=mean_fun, cov_fun = cov_fun)
    subtractive_corruption = lambda n, d, eta: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=lambda data, eta, true_mean: subtractive_noise_nonspherical(data, eta, true_mean, diag_fun), mean_fun=mean_fun, cov_fun = cov_fun, additive=False)

    corruption_schemes = {
        "gaus_one": gaussian_corruption_one_cluster,
        "unif_top": uniform_corruption_top,
        "obvious": obvious_corruption, # note that this may be too obvious in this case and doesn't align with earlier experiments
        "subtractive_corruption": subtractive_corruption # does subtractive here make sense?
    }

    meta_nonsp_corruption_schemes[diag_fun.__name__] = corruption_schemes

# varying standard deviation experiments
nonsp_std_gaussian_corruption_one_cluster = lambda n, d, eta, max_std: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=lambda n, d, true_mean: gaussian_noise_one_cluster_nonspherical(n, d, true_mean, lambda d: dim_diag(d, max_var=max_std**2)), mean_fun=mean_fun, cov_fun = lambda d: np.diag(dim_diag(d, max_var=max_std**2)))
nonsp_std_uniform_corruption_top = lambda n, d, eta, max_std: generate_data_helper(n, d, eta, uncorrupted_fun=gaussian_data, corruption_fun=lambda n, d, true_mean: uniform_noise_top_nonspherical(n, d, true_mean,  lambda d: dim_diag(d, max_var=max_std**2)), mean_fun=mean_fun, cov_fun = lambda d: np.diag(dim_diag(d, max_var=max_std**2)))
    
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
    "lee_valiant_simple": lambda data, tau: lee_valiant_simple(data, tau=tau, mean_estimator=lambda data: median_of_means(data, 10)),
    "median_of_means": lambda data, tau: median_of_means(data, 10),
    "lrv": lambda data, tau: lrv(data, C=1, trace_est_option="robust"),
    "ev_filtering_low_n": lambda data, tau: eigenvalue_pruning(data, tau=tau, gamma=5, t=10),
    "que_low_n": lambda data, tau: que_mean(data, tau=tau, fast=True, early_halt=True),
    "que_halt": lambda data, tau: que_mean(data, tau=tau, fast=True, early_halt=True), # used for robustness to expected corruption
    "pgd": lambda data, tau: grad_descent(data, tau=tau, nItr=15)
}


# eigenvalue pruning methods
eigenvalue_pruning_routines = {
    "ev_filtering_low_n": lambda data, tau: eigenvalue_pruning(data, tau=tau, gamma=5, t=10),
    "ev_filtering_low_n_random": lambda data, tau: eigenvalue_pruning(data, tau=tau, gamma=5, t=10, pruning="random"),
    "ev_filtering_low_n_fixed": lambda data, tau: eigenvalue_pruning(data, tau=tau, gamma=5, t=10, pruning="fixed") 
}

grad_iterations = {
    "pgd-1": lambda data, tau: grad_descent(data, tau, nItr=1),
    "pgd-5": lambda data, tau: grad_descent(data, tau, nItr=5),
    "pgd-10": lambda data, tau: grad_descent(data, tau, nItr=10),
    "pgd-15": lambda data, tau: grad_descent(data, tau, nItr=15),
    "pgd-20": lambda data, tau: grad_descent(data, tau, nItr=20)
}


# LEE AND VALIANT EXPERIMENTS
lee_valiant_comparison = {
    "lee_valiant_simple_mom": lambda data, tau: lee_valiant_simple(data, tau, lambda data: median_of_means(data, 10)),
    "lee_valiant_original_mom": lambda data, tau: lee_valiant_original(data, tau, lambda data: median_of_means(data, 10), 0.1),
    "lee_valiant_simple_lrv": lambda data, tau: lee_valiant_simple(data, tau, lambda data: lrv(data)),
    "lee_valiant_simple_ev": lambda data, tau: lee_valiant_simple(data, tau, lambda data: eigenvalue_pruning(data, tau)),
    "lee_valiant_original_lrv": lambda data, tau: lee_valiant_original(data, tau, lambda data: lrv(data), 0.1),
    "lee_valiant_original_ev": lambda data, tau: lee_valiant_original(data, tau, lambda data: eigenvalue_pruning(data, tau), 0.1),
    "lrv": lambda data, tau: lrv(data, C=1),
    "ev_filtering_low_n": lambda data, tau: eigenvalue_pruning(data, tau, 0.1, 5)
}

# NEW VS UPDATED EIGENVALUE EXPERIMENTS
eigenvalue_old_vs_new = {
    "sample_mean": lambda data, tau: sample_mean(data),
    "ev_filtering_low_n": lambda data, tau: eigenvalue_pruning(data, tau),
    "eigenvalue_filtering": lambda data, tau: eigenvalue_pruning(data, tau, threshold="original")
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

# EV HYPERPARAMETER TUNING - tuning gamma
ev_options = {
    # "sample_mean": lambda data, tau: sample_mean(data),
    "ev_filtering_low_n-0.5": lambda data, tau: eigenvalue_pruning(data, tau),
    "ev_filtering_low_n-1": lambda data, tau: eigenvalue_pruning(data, tau, gamma=1),
    "ev_filtering_low_n-2.5": lambda data, tau: eigenvalue_pruning(data, tau, gamma=2.5),
    "ev_filtering_low_n-5": lambda data, tau: eigenvalue_pruning(data, tau, gamma=5),
    "ev_filtering_low_n-10": lambda data, tau: eigenvalue_pruning(data, tau, gamma=10),
    "ev_filtering_low_n-20": lambda data, tau: eigenvalue_pruning(data, tau, gamma=20),
    "ev_filtering_low_n-50": lambda data, tau: eigenvalue_pruning(data, tau, gamma=50),
}

# tuning pruning method
ev_pruning_comparison = {
    "ev_filtering_low_n": lambda data, tau: eigenvalue_pruning(data, tau),
    "ev_filtering_low_n_random": lambda data, tau: eigenvalue_pruning(data, tau, pruning="random"),
    "ev_filtering_low_n_fixed": lambda data, tau: eigenvalue_pruning(data, tau, pruning="fixed"),
}

# LRV weighting method
lrv_options_general_gaus = {
    "lrv": lambda data, tau: lrv(data, 1),
    "lrv_general": lambda data, tau: lrvGeneral(data, tau)
}

# ev and que original threshold vs our threshold comparison
ev_que_original_vs_new = {
    "sample_mean": lambda data, tau: sample_mean(data),
    "ev_filtering": lambda data, tau: eigenvalue_pruning(data, tau, threshold="original"),
    "ev_filtering_low_n": lambda data, tau: eigenvalue_pruning(data, tau),
    "que": lambda data, tau: que_mean(data, tau, t=10, original_threshold=True, fast=True),
    "que_low_n": lambda data, tau: que_mean(data, tau, t=10, fast=True)
}

# EM

# Miscellaneous below
# selecting t approrpriately matters (to get high probability success)
eigenvalue_t = {
    "sample_mean": lambda data, tau: sample_mean(data),
    "ev_filtering_low_n-2.5": lambda data, tau: eigenvalue_pruning(data, tau, t=2.5),
    "ev_filtering_low_n-5": lambda data, tau: eigenvalue_pruning(data, tau, t=5),
    "ev_filtering_low_n-10": lambda data, tau: eigenvalue_pruning(data, tau, t=10)
}

lrv_options_trace_est_variations = {
    "lrv_sample_1": lambda data, tau: lrv(data, C=1, trace_est_option="sample"),
    # "lrv_sample_3": lambda data, tau: lrv(data, C=3, trace_est_option="sample"),
    "lrv_1": lambda data, tau: lrv(data, C=1, trace_est_option="robust"),
    # "lrv_3": lambda data, tau: lrv(data, C=3, trace_est_option="robust"),
    "lrv_100": lambda data, tau: lrv(data, C=100, trace_est_option="robust")
}

ev_que = {
    "sample_mean": lambda data, tau: sample_mean(data),
    "ev_filtering_low_n-5": lambda data, tau: eigenvalue_pruning(data, tau, 0.1, 5),
    "que": lambda data, tau: que_mean(data, tau)
}

lrv_options3 = {
    "lrv": lambda data, tau: lrv(data, C=1, trace_est_option="robust"),
    "lrv_sample": lambda data, tau: lrv(data, C=1, trace_est_option="sample"),
    "lrv-1000": lambda data, tau: lrv(data, C=1000, trace_est_option="robust"),
    "lrv-5": lambda data, tau: lrv(data, C=5, trace_est_option="robust"),
    "lrv-0.1": lambda data, tau: lrv(data, C=0.1, trace_est_option="robust")
}

sdp = {
        "sample_mean": lambda data, tau: sample_mean(data),
        "sdp_mean": lambda data, tau: sdp_mean(data, tau),
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

ev_que_experiment = [["n", np.arange(20, 10021, 500), None, default_d, default_eta, default_tau]]