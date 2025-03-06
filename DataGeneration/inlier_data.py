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


def uniform_multinomial_data(n, d, mean_fun=None, cov_fun=None):
    """
    Generate data where each sample is a normalized vector of d random values drawn from Uniform[0,1).
    """

    # Generate n samples, each a vector of d random values in Uniform[0,1)
    raw_data = np.random.uniform(0, 1, size=(n, d))
    
    # Normalize each sample so that it sums to 1
    data = raw_data / raw_data.sum(axis=1, keepdims=True)

    # Compute the mean of the generated data
    good_sample_mean = np.mean(data, axis=0)

    # True mean is analytically known: each dimension is expected to have value 1/d
    true_mean = np.full(d, 1 / d)

    return data, good_sample_mean, true_mean

import numpy as np

def t_data(n, d, mean_fun=None, cov_fun=None, df=3):
    """
    Generate data from a multivariate Student's t-distribution with n data points and d dimensions.
    df: degrees of freedom for the t-distribution (default is 3)
    mean_fun: function to generate mean of size d, defaults to ones if None
    cov_fun: function to generate covariance matrix of size (d, d), defaults to identity if None
    """
    # Define the true mean using the provided function or default to ones
    if mean_fun is None:
        true_mean = np.ones(d)
    else:
        true_mean = mean_fun(d)
    
    # Define the covariance matrix using the provided function or default to identity matrix
    if cov_fun is None:
        cov = np.eye(d)
    else:
        cov = cov_fun(d)
    
    # Generating data from multivariate Student's t-distribution
    # The multivariate t-distribution can be expressed as a scaled normal distribution
    # with the scaling factor being drawn from a chi-square distribution
    chi_sq_samples = np.random.chisquare(df, size=n)
    normal_samples = np.random.multivariate_normal(np.zeros(d), cov, size=n)
    
    # Scale the normal samples by the square root of the chi-square samples divided by the degrees of freedom
    data = true_mean + normal_samples * np.sqrt(df / chi_sq_samples[:, None])
    
    # Calculate the good sample mean (mean of the generated data)
    good_sample_mean = np.mean(data, axis=0)
    
    return data, good_sample_mean, true_mean

import numpy as np

import numpy as np

def gaussian_mixture_data(n, d, mean_fun=None, cov_fun=None, weights=None):
    """
    Efficiently generate data from a mixture of 3 Gaussian distributions in d dimensions.
    The function will generate n samples where each sample is drawn from one of the three Gaussian distributions.
    
    n: Number of samples
    d: Number of dimensions
    mean_fun: Function to generate mean of each Gaussian (returns a list of size 3 with means for each Gaussian)
    cov_fun: Function to generate covariance matrices for each Gaussian (returns a list of size 3 with covariance matrices for each Gaussian)
    weights: A list of mixing weights (3 values summing to 1, representing how likely each Gaussian is to be chosen)
    """
    
    if mean_fun is None:
        means = [np.zeros(d), np.ones(d), -np.ones(d)]
    else:
        means = mean_fun(d)
    
    if cov_fun is None:
        covariances = [np.eye(d), np.eye(d), np.eye(d)]
    else:
        covariances = cov_fun(d)
    
    if weights is None:
        weights = [1/3, 1/3, 1/3]
    
    weights = np.array(weights)
    weights /= weights.sum()  # Ensure weights sum to 1

    # Step 1: Generate the component assignments for all points at once
    components = np.random.choice([0, 1, 2], size=n, p=weights)
    
    # Step 2: Generate all samples at once
    data = np.zeros((n, d))  # Initialize the data array
    for i in range(3):
        # Find the indices of the points assigned to component i
        indices = components == i
        
        # Sample all points assigned to component i from the corresponding Gaussian
        if np.any(indices):
            data[indices] = np.random.multivariate_normal(means[i], covariances[i], size=indices.sum())
    
    # Calculate the good sample mean (mean of the generated data)
    good_sample_mean = np.mean(data, axis=0)
    
    # The true mean is the weighted sum of the means
    true_mean = np.dot(weights, means)
    
    return data, good_sample_mean, true_mean

import numpy as np

def laplace_data(n, d, mean_fun=None, cov_fun=None):
    """
    Generate data from a Laplace distribution in d dimensions.
    The function will generate n samples from a Laplace distribution.

    n: Number of samples
    d: Number of dimensions
    mean_fun: Function to generate the mean of the Laplace distribution (returns a vector of size d).
    cov_fun: Function to generate the scale parameter for the Laplace distribution (returns a scalar or a vector of size d).

    Returns:
    - data: n by d numpy matrix with the generated samples
    - good_sample_mean: The mean of the generated data
    - true_mean: The true mean used to generate the data
    """
    
    if mean_fun is None:
        # Default mean is a vector of zeros
        true_mean = np.zeros(d)
    else:
        true_mean = mean_fun(d)
    
    if cov_fun is None:
        # Default scale is 1 for each dimension (Laplace distribution)
        scale = np.ones(d)
    else:
        scale = cov_fun(d)
    
    # Generate data using Laplace distribution
    data = np.random.laplace(loc=true_mean, scale=scale, size=(n, d))
    
    # Calculate the good sample mean (mean of the generated data)
    good_sample_mean = np.mean(data, axis=0)
    
    return data, good_sample_mean, true_mean

def poisson_data(n, d, mean_fun=None, cov_fun=None):
    """
    Generate data from a multivariate Poisson distribution in d dimensions.
    The function will generate n samples from a Poisson distribution for each dimension.

    n: Number of samples
    d: Number of dimensions
    mean_fun: Function to generate the mean (lambda) for each dimension (returns a vector of size d with positive values).
    cov_fun: Function to generate covariance structure (not used for Poisson, but included for interface consistency).

    Returns:
    - data: n by d numpy matrix with the generated samples
    - good_sample_mean: The mean of the generated data
    - true_mean: The true mean used to generate the data
    """
    
    if mean_fun is None:
        # Default mean (lambda) is 1 for each dimension
        true_mean = np.ones(d)
    else:
        true_mean = mean_fun(d)
    
    # Generate data using Poisson distribution
    data = np.random.poisson(lam=true_mean, size=(n, d))
    
    # Calculate the good sample mean (mean of the generated data)
    good_sample_mean = np.mean(data, axis=0)
    
    return data, good_sample_mean, true_mean

