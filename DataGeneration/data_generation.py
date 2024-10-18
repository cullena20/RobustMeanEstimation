'''
Data generation wrappers:
    - generate_data_helper: generate corrupted data given uncorrupted and corrupted data generating functions
    - create_fun_from_data: build data generators for generate_data_helper given data
'''

import numpy as np
from helper import random_rotation_matrix

def generate_data_helper(n, d, eps, uncorrupted_fun, corruption_fun, additive=True, mean_fun=None, cov_fun=None, rotate=False, draw_from_data=False, **kwargs):
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
    additive: True if noise is additive, False is noise just manipulates the data, e.g. subtractive corruption
    corruption_fun: If additive, returns corruption to be appended to uncorrupted data
                        Takes in number of corrupted points to generate, dimensions, and true mean
                    If not additive, returns manipulated data
                        Takes in data, percentage of corruption, true mean
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
    if additive:
        good_data_size = round(n*(1-eps))
    else:
        good_data_size = n

    uncorrupted_data, good_sample_mean, true_mean = uncorrupted_fun(good_data_size, d, mean_fun=mean_fun, cov_fun=cov_fun)

    if additive:
        noise = corruption_fun(round(n * eps), d, true_mean=true_mean)
        data = np.concatenate((uncorrupted_data, noise), axis=0)   
    else:
        data = corruption_fun(uncorrupted_data, eps, true_mean)


    if rotate:
        rotation_matrix = random_rotation_matrix(d)
        data = data @ rotation_matrix
        good_sample_mean = rotation_matrix @ data
        true_mean = rotation_matrix @ data
   
    return data, good_sample_mean, true_mean

# Use this function to input your own data into generate_data_helper
def create_fun_from_data(data, uncorrupted=True):
    """
    Create data generation function from data to fit into generate_data_helper interface
    uncorrupted = True generates a function that fits uncorrupted_fun interface in generate_data_helper
    uncorrupted = False generates a function that fits corrupted_fun interface in generate_data_helper
    """
    available_data = data.shape[0] 
    true_mean = np.mean(data, axis=0)
    if uncorrupted:
        def draw_from_data(n, d, mean_fun=None, cov_fun=None):
            # unused inputs to fit interface
            random_indices = np.random.choice(available_data, n)
            drawn_data = data[random_indices]
            good_sample_mean = np.mean(drawn_data, axis=0)
            return drawn_data, good_sample_mean, true_mean
    else:
        def draw_from_data(n, d, true_mean=None):
            # unused inputs to fit interface
            random_indices = np.random.choice(available_data, n)
            drawn_data = data[random_indices]
            return drawn_data
    return draw_from_data