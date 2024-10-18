import numpy as np
from noise_generation import random_rotation_matrix

# NOTE - I GOT RID OF ADDITIVE NOISE

def generate_data_helper(n, d, eps, uncorrupted_fun, noise_fun, mean_fun=None, cov_fun=None, rotate=False):
    """
    Generate corrupted data given supplied inlier data function and corruption function.
    
    Parameters:
    n: number of data points (including corrupted and uncorrupted points)
    d: number of dimensions of data
    uncorrupted_fun: a function that generates uncorrupted data
                     this function must take in data size, dimensions, 
                     a mean generation function (takes in dimensions and returns a mean), 
                     and a covariance function (takes in dimensions and returns a covariance)
                     It returns the uncorrupted_data, the sample mean from which the data is drawn (good_sample_mean),
                     and the sample mean of the inliers (true_mean)
    mean_fun: function that takes in dimension and generates a mean, used in uncorrupted_fun
    cov_fun: function that takes in dimension and generates a covariance, used in uncorrupted_fun
    rotate: True to randomly rotate corrupted data

    Returns:
    data : numpy.ndarray
        The combined dataset containing both uncorrupted and corrupted data. The array 
        has shape `(n, d)` where `n` is the total number of data points, and `d` is the 
        dimensionality.

    good_sample_mean : numpy.ndarray
        The mean of the uncorrupted data points (inliers)

    true_mean : numpy.ndarray
        The true mean of the uncorrupted data
    """

    uncorrupted_data, good_sample_mean, true_mean = uncorrupted_fun(round(n* (1 - eps)), d, mean_fun=mean_fun, cov_fun=cov_fun)
    noise = noise_fun(round(n * eps), d, true_mean)
    data = np.concatenate((uncorrupted_data, noise), axis=0)

    if rotate:
        rotation_matrix = random_rotation_matrix(d)
        data = data @ rotation_matrix
        good_sample_mean = rotation_matrix @ data
        true_mean = rotation_matrix @ data
   
    return data, good_sample_mean, true_mean
