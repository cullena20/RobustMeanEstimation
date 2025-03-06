"""
Corruption Schemes

Note that additive and non additive corruption schemes have different expected interfaces
"""

import numpy as np
import random
import math
from helper import random_rotation_matrix, custom_rotation_matrix

def gaussian_noise_one_cluster(n, d, true_mean, std=1):
    """
    Additive Variance Shell Noise
    """
    # rotate cluster randomly to remove bias possibly induced by coordinate axises
    rotate_basis = random_rotation_matrix(d)
    noise_mean = true_mean + rotate_basis @ np.ones(d) * std
    cov = np.eye(d) / 10
    Y1 = np.random.multivariate_normal(noise_mean, cov, round(n))
    return Y1

def dkk_noise(n, d, true_mean, std=1):
    """
    Noise scheme from eigenvalue pruning paper
    """
    Y1 = np.random.choice([-1*std, 0], size=(round(0.5 * n), d)) + true_mean 
    Y2 = np.concatenate((random.choice([-1*std, 11*std]) * np.ones((round(0.5 * n), 1)), 
                            random.choice([-3*std, -1*std]) * np.ones((round(0.5 * n), 1)), 
                            -1*np.ones((round(0.5 * n), d-2))), axis=1) + true_mean
    return np.concatenate((Y1, Y2), axis=0)

def gaussian_noise_two_clusters(n, d, true_mean, angle=75, std=1, cluster1_percent=0.7, cluster2_percent=0.3):
    """
    Additive Variance Shell Noise with two clusters
    """
    rotate_noise = custom_rotation_matrix(angle, d)
    direction0 = np.concatenate(([math.sqrt(d)], np.zeros(d-1)))
    direction1 = rotate_noise @ direction0

    # rotate both clusters randomly by the same matrix to remove bias possibly induced by coordinate axises
    rotate_basis = random_rotation_matrix(d)
    direction0 = rotate_basis @ direction0
    direction1 = rotate_basis @ direction1

    noise_mean0 = true_mean + direction0 * std
    noise_mean1 = true_mean + direction1 * std
    cov = np.eye(d) / 10
    Y1 = np.random.multivariate_normal(noise_mean0, cov, round(cluster1_percent*n))
    Y2 = np.random.multivariate_normal(noise_mean1, cov, round(cluster2_percent*n))

    return np.concatenate((Y1, Y2), axis=0)

def uniform_noise_top(n, d, true_mean, std=1):
    """
    Uniform Noise distributed on top half of data
    """
    Y1 = np.random.uniform(low= 0, high=2*std, size=(round(n), d))
    Y1 = true_mean + Y1
    return Y1

def obvious_noise(n, d, true_mean, angle=75, std=1):
    """
    Large Outliers with two clusters
    """
    rotate_noise = custom_rotation_matrix(angle, d)
    direction0 = np.concatenate(([1], np.zeros(d-1)))
    direction1 = rotate_noise @ direction0

    # rotate both clusters randomly to remove bias possibly induced by coordinate axises
    rotate_basis = random_rotation_matrix(d)
    direction0 = rotate_basis @ direction0
    direction1 = rotate_basis @ direction1

    noise_mean0 = true_mean + 10 * math.sqrt(d) * std * direction0 # directions have norm 1
    noise_mean1 = true_mean + 20 * math.sqrt(d) * std * direction1

    cov = np.eye(d) / 10
    Y1 = np.random.multivariate_normal(noise_mean0, cov, round(0.7*n))
    Y2 = np.random.multivariate_normal(noise_mean1, cov, round(0.3*n))

    return np.concatenate((Y1, Y2), axis=0)

def subtractive_noise(data, eps, true_mean):
    """
    Subtractive Noise

    Note: This follows a different interface because it is not additive
    """
    n, d = data.shape

    temp = np.random.randn(d)
    v = temp / np.linalg.norm(temp)
    projected_data = np.dot(data - true_mean, v) # project each datapoint onto v through broadcasting, this will be n x 1

    sorted_indices = np.argsort(projected_data)
    projected_data = projected_data[sorted_indices]
    data = data[sorted_indices] # store data based on magnitude of projection

    return data[:math.ceil((1-eps)*n)]


def multiple_corruption(n, d, true_mean, scheme1, scheme2, std=1, scheme1_percent=0.5):
    """
    Generates corruption from two schemes with scheme1_percent of corruption from scheme1
    Used in obvious and subtle corruption mixes
    """
    scheme1_n = math.ceil(scheme1_percent * n)
    scheme2_n = n - scheme1_n

    Y1 = scheme1(scheme1_n, d, true_mean, std=std)
    Y2 = scheme2(scheme2_n, d, true_mean, std=std)

    return np.concatenate((Y1, Y2), axis=0)


def gaussian_noise_one_cluster_nonspherical(n, d, true_mean, diag_fun=None):
    """
    Additive Variance Shell Noise with nonspherical data

    Note: the inlier data must still have diagonal covariance for this to be meaningful
    """
    if diag_fun is None:
        diag = np.ones(d)
    else:
        diag = diag_fun(d)
    noise_mean = true_mean + np.ones(d) * np.sqrt(diag) 
    cov = np.eye(d) / 10
    Y1 = np.random.multivariate_normal(noise_mean, cov, round(n))
    return Y1


def uniform_noise_top_nonspherical(n, d, true_mean, diag_fun=None):
    """
    Uniform Noise distributed on top half of each coordinate on nonspherical data

    Note: the inlier data must still have diagonal covariance for this to be meaningful
    """
    if diag_fun is None:
        diag = np.ones(d)
    else:
        diag = diag_fun(d)
    Y1 = np.random.uniform(low= 0, high=2, size=(round(n), d)) * np.sqrt(diag)
    Y1 = true_mean + Y1
    return Y1


def uniform_mulitnomial_one_cluster(n, d, true_mean, std=1):
    """
    Additive Single Cluster Noise for Uniform Multinomial
    """
    # rotate cluster randomly to remove bias possibly induced by coordinate axises
    rotate_basis = random_rotation_matrix(d)
    noise_mean = true_mean + rotate_basis @ np.ones(d) * (1/np.sqrt(d))
    #noise_mean = true_mean + rotate_basis @ np.ones(d) * std
    cov = np.eye(d) / 10
    Y1 = np.random.multivariate_normal(noise_mean, cov, round(n))
    return Y1

# def multivariate_t_one_cluster(n, d, true_mean, std=1):
#     """
#     Additive Single Cluster Noise for Multivariate T Distribution
#     """
#     # rotate cluster randomly to remove bias possibly induced by coordinate axises
#     rotate_basis = random_rotation_matrix(d)
#     noise_mean = true_mean + rotate_basis @ np.ones(d) * std
#     cov = np.eye(d) / 10
#     Y1 = np.random.multivariate_normal(noise_mean, cov, round(n))
#     return Y1

def mixture_of_gaussians_noise(n, d, true_mean):
    noise_mean = np.ones(d) + 1
    cov = np.eye(d)
    Y1 = np.random.multivariate_normal(noise_mean, cov, round(n))
    return Y1


# Earlier noise generation schemes not used in experiments

def obvious_hard_two_clusters(n, d, true_mean, angle=75, std=1, hard_weight=0.5):
    rotate_noise = custom_rotation_matrix(angle, d)
    direction0 = np.concatenate(([math.sqrt(d)], np.zeros(d-1)))
    direction1 = rotate_noise @ direction0

    # rotate both clusters randomly to remove bias possibly induced by coordinate axises
    rotate_basis = random_rotation_matrix(d)
    direction0 = rotate_basis @ direction0
    direction1 = rotate_basis @ direction1

    hard_noise_mean = true_mean + direction0 * std
    obvious_noise_mean = true_mean + direction1 * std * 10
    cov = np.eye(d) / 10
    Y1 = np.random.multivariate_normal(hard_noise_mean, cov, round(hard_weight*n))
    Y2 = np.random.multivariate_normal(obvious_noise_mean, cov, round((1-hard_weight)*n))

    return np.concatenate((Y1, Y2), axis=0)

def uniform_noise_whole(n, d, true_mean, std=1):
    Y1 = np.random.uniform(low= -2 * std, high=2 * std, size=(round(n), d))
    Y1 = true_mean + Y1
    return Y1

def obvious_easy(n, d, true_mean, std=1):
    noise_mean = true_mean + 10 * np.ones(d) # add sqrt(d) in every direction, making it d * sqrt(d) away from true mean
    cov = np.eye(d) / 10
    return np.random.multivariate_normal(noise_mean, cov, n)

def obvious_noise_nonspherical(n, d, true_mean, diag_fun=None, angle=75):
    if diag_fun is None:
        diag = np.ones(d)
    else:
        diag = diag_fun(d)

    rotate_noise = custom_rotation_matrix(angle, d)
    direction0 = np.concatenate(([1], np.zeros(d-1)))
    direction1 = rotate_noise @ direction0

    std_est = math.sqrt(np.sum(diag) / d)
    noise_mean0 = true_mean + 10 * math.sqrt(d) * std_est * direction0
    noise_mean1 = true_mean + 20 * math.sqrt(d) * std_est * direction1

    cov = np.eye(d) / 10
    Y1 = np.random.multivariate_normal(noise_mean0, cov, round(0.7*n))
    Y2 = np.random.multivariate_normal(noise_mean1, cov, round(0.3*n))

    return np.concatenate((Y1, Y2), axis=0)

def subtractive_noise_nonspherical(data, eps, true_mean, diag_fun=None):
    n, d = data.shape

    if diag_fun is None:
        diag = np.ones(d)
    else:
        diag = diag_fun(d)

    # project onto the distance of most variance
    max_var_idx = np.argmax(diag)
    v = np.zeros(d)
    v[max_var_idx] = 1
    projected_data = np.dot(data - true_mean, v) # project each datapoint onto v through broadcasting, this will be n x 1

    sorted_indices = np.argsort(projected_data)
    projected_data = projected_data[sorted_indices]
    data = data[sorted_indices] # store data based on magnitude of projection

    return data[:math.ceil((1-eps)*n)]