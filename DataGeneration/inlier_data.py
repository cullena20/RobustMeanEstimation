"""
Inlier data generation schemes

All data generation schemes should be of form:

data_size, dimensions, mean_fun, cov_fun 
=> data (np.ndarray: data_size by dimensions), 
   good_sample_mean (np.ndarray: dimensions)
   mean (np.ndarray: size dimensions)
"""

import numpy as np

def gaussian_data(n, d, mean_fun=None, cov_fun=None):
    """
    Generate Gaussian data with n data points and d dimensions as n by d numpy matrix

    """
    if mean_fun is None:
        mean = np.ones(d)
    else:
        mean = mean_fun(d)
    if cov_fun is None:
        cov = np.eye(d)
    else:
        cov = cov_fun(d)

    data = np.random.multivariate_normal(mean, cov, n)
    good_sample_mean = np.mean(data, axis=0)

    return data, good_sample_mean, mean    

data, good_sample_mean, _ = gaussian_data(50, 10)
print(data.shape)
print(good_sample_mean.shape)