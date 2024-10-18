"""
Inlier data generation scheme

All data generation schemes should be of type signature:

data_size: integer, dimensions: integer, mean_fun: dimensions => np.ndarray with size (dimensions, ), cov_fun: np.ndarray with size (dimensions, dimensions) 
=> data : np.ndarray with size (data_size, dimensions)), 
   good_sample_mean : np.ndarray with size (dimensions, ),
   mean : np.ndarray with size (dimensions, )

good_sample_mean is the mean of the inliers
mean is the true mean that the distribution is generated from
"""

import numpy as np

def gaussian_data(n, d, mean_fun=None, cov_fun=None):
    """
    Generate Gaussian data with n data points and d dimensions as n by d numpy matrix.
    Defaulte to mean of all ones and identity covariance if no mean_fun or cov_fun are supplied
    """
    if mean_fun is None:
        true_mean = np.ones(d)
    else:
        true_mean = mean_fun(d)
    if cov_fun is None:
        cov = np.eye(d)
    else:
        cov = cov_fun(d)

    data = np.random.multivariate_normal(true_mean, cov, n)
    good_sample_mean = np.mean(data, axis=0)

    return data, good_sample_mean, true_mean    

