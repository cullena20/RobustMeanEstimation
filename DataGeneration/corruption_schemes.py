import numpy as np
import random
import math

# STEP 1: INVESTIGATE DKK_NOISE CLOSELY TO MAKE SURE NO ISSUES
# STEP 2: DESIGN ALTERNATE EASY HARD NOISE USING GAUS TWO CLUSTERS
# STEP 3: DESIGN ALTERNATE EASY HARD NOISE FOR NON SPHERICAL USING TRACE/n expectation
# note for last step there may be more details to consider, but don't get bogged down.

def multiple_corruption(n, d, true_mean, scheme1, scheme2, std=1, scheme1_percent=0.5):
    scheme1_n = math.ceil(scheme1_percent * n)
    scheme2_n = n - scheme1_n

    Y1 = scheme1(scheme1_n, d, true_mean, std=std)
    Y2 = scheme2(scheme2_n, d, true_mean, std=std)

    return np.concatenate((Y1, Y2), axis=0)


# this doesn't actually seem to be meaningful since sample mean works just fine
# OKAY I FIGURED OUT THE ISSUE
# I THINK THIS IS WRONG -> LOOK INTO AND FIX, BUT TOO LATE NOW
def dkk_noise(n, d, true_mean, std=1):
    Y1 = np.random.choice([-1*std, 0], size=(round(0.5 * n), d)) + true_mean # 0 away or 1 away from true vector (0 or 1 in true)
    Y2 = np.concatenate((random.choice([-1*std, 11*std]) * np.ones((round(0.5 * n), 1)), # 1 or 11 away (12 -1 and 0-1)
                            random.choice([-3*std, -1*std]) * np.ones((round(0.5 * n), 1)), 
                            -1*np.ones((round(0.5 * n), d-2))), axis=1) + true_mean
    return np.concatenate((Y1, Y2), axis=0)


# maybe try to make a non spherical version of the above

# # this might not make any sense
# def dkk_noise_spherical(n, d, true_mean, std):
#     shift = true_mean - np.ones(d)
#     Y1 = np.random.randint(2, size=(round(0.5 * n), d)) * std + shift
#     Y2 = np.concatenate((random.choice([12*std, 0]) * np.ones((round(0.5 * n), 1)),
#                             random.choice([-2*std, 0]) * np.ones((round(0.5 * n), 1)),
#                             np.zeros((round(0.5 * n), d-2))), axis=1) + shift
#     return np.concatenate((Y1, Y2), axis=0)

def gaussian_noise_one_cluster(n, d, true_mean, std=1):
    # rotate cluster randomly to remove bias possibly induced by coordinate axises
    rotate_basis = random_rotation_matrix(d)

    noise_mean = true_mean + rotate_basis @ np.ones(d) * std
    cov = np.eye(d) / 10
    Y1 = np.random.multivariate_normal(noise_mean, cov, round(n))
    return Y1

# not actually sure if this is the best method
def gaussian_noise_one_cluster_nonspherical(n, d, true_mean, diag_fun=None):
    if diag_fun is None:
        diag = np.ones(d)
    else:
        diag = diag_fun(d)
    # max_std = math.sqrt(np.max(diag))
    noise_mean = true_mean + np.ones(d) * np.sqrt(diag) # max_std # I think this makes the most sense but it's also rought
    cov = np.eye(d) / 10
    Y1 = np.random.multivariate_normal(noise_mean, cov, round(n))
    return Y1

# modify this to vary angle
def gaussian_noise_two_clusters(n, d, true_mean, angle=75, std=1):
    rotate_noise = custom_rotation_matrix(angle, d)
    direction0 = np.concatenate(([math.sqrt(d)], np.zeros(d-1)))
    direction1 = rotate_noise @ direction0

    # rotate both clusters randomly by the same matrixto remove bias possibly induced by coordinate axises
    rotate_basis = random_rotation_matrix(d)
    direction0 = rotate_basis @ direction0
    direction1 = rotate_basis @ direction1

    noise_mean0 = true_mean + direction0 * std
    noise_mean1 = true_mean + direction1 * std
    cov = np.eye(d) / 10
    Y1 = np.random.multivariate_normal(noise_mean0, cov, round(0.7*n))
    Y2 = np.random.multivariate_normal(noise_mean1, cov, round(0.3*n))

    return np.concatenate((Y1, Y2), axis=0)

# 0.7 is hard to detect (variance shell additive noise)
# the rest are placed far away (10 times expected value)
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

# maybe make a non spherical version of anobe


# def gaussian_noise3(n, d, true_mean):
#     noise_mean1 = true_mean + math.sqrt(d/n)
#     # noise_mean1 = true_mean + math.sqrt(d)
#     direction1 = np.concatenate(([0], np.ones(d-1)))
#     noise_mean2 = true_mean + direction1 * math.sqrt(d/n)
#     direction2 = np.concatenate((np.ones(d-1), [0]))
#     noise_mean3 = true_mean + direction2 * math.sqrt(d/n)
#     cov = np.eye(d)
#     Y1 = np.random.multivariate_normal(noise_mean1, cov, round(0.33*n))
#     Y2 = np.random.multivariate_normal(noise_mean2, cov, round(0.33*n))
#     Y3 = np.random.multivariate_normal(noise_mean3, cov, round(0.33*n))
#     return np.concatenate((Y1, Y2), axis=0)

# def combined_noise(n, d, true_mean):
#     noise_mean1 = true_mean + math.sqrt(d)
#     cov = np.eye(d)
#     Y1 = np.random.multivariate_normal(noise_mean1, cov, round(0.5*n))
#     Y2 = np.concatenate((random.choice([16, 0]) * np.ones((round(0.5 * n), 1)),
#                             random.choice([-2, 0]) * np.ones((round(0.5 * n), 1)),
#                             np.zeros((round(0.5 * n), d-2))), axis=1) + true_mean
#     return np.concatenate((Y1, Y2), axis=0)

# def uniform_noise(n, d, true_mean, option=0):
#     if option == 0:
#         Y1 = np.random.uniform(low= -math.sqrt(d), high=math.sqrt(d), size=(round(n), d))
#     else:
#         Y1 = np.random.uniform(low= 0, high=math.sqrt(d), size=(round(n), d))
#     Y1 = true_mean + Y1
#     return Y1

def uniform_noise_whole(n, d, true_mean, std=1):
    Y1 = np.random.uniform(low= -2 * std, high=2 * std, size=(round(n), d))
    Y1 = true_mean + Y1
    return Y1

def uniform_noise_top(n, d, true_mean, std=1):
    Y1 = np.random.uniform(low= 0, high=2*std, size=(round(n), d))
    Y1 = true_mean + Y1
    return Y1

# i think this makes sense
def uniform_noise_top_nonspherical(n, d, true_mean, diag_fun=None):
    if diag_fun is None:
        diag = np.ones(d)
    else:
        diag = diag_fun(d)
    Y1 = np.random.uniform(low= 0, high=2, size=(round(n), d)) * np.sqrt(diag)
    Y1 = true_mean + Y1
    return Y1

# def uniform_noise_spherical_whole(n, d, true_mean, std):
#      Y1 = np.random.uniform(low= -2*std, high=2*std, size=(round(n), d))
#      Y1 = true_mean + Y1
#      return Y1
     
# def uniform_noise_spherical_top(n, d, true_mean, std):
#      Y1 = np.random.uniform(low= 0, high=2*std, size=(round(n), d))
#      Y1 = true_mean + Y1
#      return Y1

# def obvious_noise_spherical(n, d, true_mean, std):
#     Y1 = np.ones((round(0.5*n), d)) * 5 * std + true_mean
#     direction = np.concatenate(([0], np.ones(d-1))) / math.sqrt(d-1)
#     Y2 = np.ones((round(0.5*n), d)) * 20 * std * direction + true_mean 
#     return np.concatenate((Y1, Y2), axis=0)

# THE BELOW REPLICATES ORIGINAL RESULTS
def obvious_easy(n, d, true_mean, std=1):
    noise_mean = true_mean + 10 * np.ones(d) # add sqrt(d) in every direction, making it d * sqrt(d) away from true mean
    cov = np.eye(d) / 10
    return np.random.multivariate_normal(noise_mean, cov, n)

def obvious_noise(n, d, true_mean, angle=75, std=1):
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

# I think this makes sense -> all data is enclosed in this ball somehow, and we're putting outliers far from this
def obvious_noise_nonspherical(n, d, true_mean, diag_fun=None, angle=75):
    if diag_fun is None:
        diag = np.ones(d)
    else:
        diag = diag_fun(d)

    rotate_noise = custom_rotation_matrix(angle, d)
    direction0 = np.concatenate(([1], np.zeros(d-1)))
    direction1 = rotate_noise @ direction0

    # max_std = math.sqrt(np.max(diag))

    std_est = math.sqrt(np.sum(diag) / d)
    noise_mean0 = true_mean + 10 * math.sqrt(d) * std_est * direction0
    noise_mean1 = true_mean + 20 * math.sqrt(d) * std_est * direction1

    cov = np.eye(d) / 10
    Y1 = np.random.multivariate_normal(noise_mean0, cov, round(0.7*n))
    Y2 = np.random.multivariate_normal(noise_mean1, cov, round(0.3*n))

    return np.concatenate((Y1, Y2), axis=0)


# think of more noise functions
# potentially think of subtractive noise and mix of subtracting and adding

# this works in spherical covariance
def subtractive_noise(data, eps, true_mean):
    n, d = data.shape

    temp = np.random.randn(d)
    v = temp / np.linalg.norm(temp)
    projected_data = np.dot(data - true_mean, v) # project each datapoint onto v through broadcasting, this will be n x 1

    sorted_indices = np.argsort(projected_data)
    projected_data = projected_data[sorted_indices]
    data = data[sorted_indices] # store data based on magnitude of projection

    return data[:math.ceil((1-eps)*n)]

# I think this makes sense
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

# I don't think this is useful 
def mix_corruption(data, eps, true_mean):
    n, d = data.shape
    data1 = subtractive_noise(data, eps/2, true_mean)
    noise_mean = true_mean - math.ones(d)
    cov = np.eye(d)
    Y1 = np.random.multivariate_normal(noise_mean, cov, math.floor(eps/2 * n))
    return np.concatenate((data1, Y1))

def custom_rotation_matrix(angle, d):
    # Construct rotation matrix in the orthonormal basis
    angle = np.deg2rad(angle)
    R = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    # Construct identity matrix of dimension (n-2)
    I = np.eye(d- 2)
    # Assemble the rotation matrix in the canonical basis
    rotation_matrix = np.block([[R, np.zeros((2, d - 2))], [np.zeros((d - 2, 2)), I]])
    return rotation_matrix

def random_rotation_matrix(d):
    # Generate a random rotation matrix in d dimensions
    random_matrix = np.random.randn(d, d)
    orthogonal_matrix, _ = np.linalg.qr(random_matrix)
    return orthogonal_matrix