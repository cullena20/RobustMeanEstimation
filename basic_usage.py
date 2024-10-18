"""
Covers basic usage of this library:
    - Calling robust mean estimation algorithms
    - Defining and running synthetic data experiment
    - Defining and running experiment over your own data
"""

import sys
import os
# Add DataGeneration, Algorithms, and Experiments to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'DataGeneration')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Algorithms')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Experiments')))

from synthetic_setup import main_estimators, id_gaussian_corruption_one_cluster
import numpy as np

from embedding_setup import field_land_bert768, field_study_bert768, estimators_short
from experiment_helper import experiment_suite, embedding_experiment_suite

# PART 1: Perform a simple experiment over additive variance shell noise with identity covariance
#         - Call mean estimators on data
#         - Utilize data generation function
#         - Measure error

n = 200 # data size
d = 200 # dimensionality
eta = 0.1 # corruption percentage
tau = eta # expected corruption

# data can be any n by d numpy array, so you're own data can easily be plugged into this framework
# different data and corruption schemes can be built using generate_data_helper and the tools under DataGeneration
corr_data, good_sample_mean, true_mean = id_gaussian_corruption_one_cluster(n, d, eta)

print(f"Performing Mean Estimation on Data Size {n}, Dimensionality {d}, Corruption {eta}, and Expected Corruption {tau}\n")

print(f"Good Sample Error: {np.linalg.norm(good_sample_mean - true_mean)}")

for key, estimator in main_estimators.items():
    mean_estimate = estimator(corr_data, tau)
    error = np.linalg.norm(mean_estimate - true_mean)
    print(f"{key} Error: {error}")

# Part 2: Perform a simple series of experiments where we vary data size, dimensionality, and corruption
#         - Specify experimental conditions
#         - Run experiment, save plot, return errors and standard deviations
#         - This experimental format is the same as post experiments reported in our paper


print("\nRunning Simple Experiment Over Identity Covariance Additive Variance Shell Corrupted Data")

# each experiment is of form: [varying_variable, varying_range, default_n, default_d, default_eta, default_tau]
# default_<varying_variable> can simply be set to None
# when default_tau is set to None, it will be set to eta
experiments = [["n", np.arange(20, 3021, 500), None, 150, 0.1, None],
               ["n", np.arange(20, 321, 80), None, 150, 0.1, None],
               ["d", np.arange(20, 321, 50), 150, None, 0.1, None],
               ["eta", np.arange(0.01, 0.32, 0.1), 150, 150, None, None]]

# This will take about 30 seconds to run with 2 runs using the above experimental conditions
# experiment_suite takes in a dictionary of estimators, the corrupted data generation function, the experiments, and other experimental parameters
# the data generation function must be of form (n, d, eta) => data, good_sample_mean, true_mean
# the data generation function is called with every choice of the varying variable, at every run
error_dict, std_dict = experiment_suite(main_estimators, id_gaussian_corruption_one_cluster, experiments, runs=2,  save_title=f"id_cov-additive_variance_shell", error_bars=True, plot_good_sample=True, plot=False, sample_scale_cov=False, pickle_results=False, style_dict="main")


# Part 3: Perform a simple experiment over embedding data
#         - Perform experiment over any supplied data
#         - This experiment is of same format as those reported in our paper


# data just has to be an n by d numpy array
# Field Land Bert contains Bert Embeddings of the word field corresponding to "fields of land"
# Field Study Bert contains Bert Embeddings of the word field corresponding to "fields of study"
# To perform a corrupted experiment, both must have same dimensionality
print("Field Land Bert Shape", field_land_bert768.shape)
print("Field Study Bert Shape", field_study_bert768.shape)

# now define an experiment varying corruption and data size
data_var_range = np.arange(10, 410, 100)
data_var_range = np.append(data_var_range, 400)
# the d variable can be set to None because d is determined by the data supplied
llm_experiment = [["eta", np.arange(0.01, 0.42, 0.2), 400, None, None, None],
                  ["n", data_var_range, None, None, 0.1, None]]

# embedding_experiment_suite is a wrapper for experiment_suite which creates a data generation function based on the inlier and outlier data
# and also handles the dimensionality in the experiments supplied
# we use estimators_short as this experiment generally takes very long to run
embedding_experiment_suite(estimators_short, inlier_data=field_land_bert768, outlier_data=field_study_bert768, experiments=llm_experiment, save_title="Bert768-Corruption", sample_scale_cov=True, runs=2)

# now you can easily perform your own experiment on your own data by subbing in your data for the LLM embedding data used here