"""
A collection of simple mean estimation functions:
  - Sample Mean
  - Coordinate Wise Median
  - Median Of Means
  - Coordinate Wise Trimmed Mean
  - Geometric Median
"""

import numpy as np
import math

def sample_mean(data):
  """
  Computes the sample mean of the given data.

  Parameters:
  data (np.ndarray): A 2D array where rows represent samples and columns represent features.

  Returns:
  np.ndarray: The sample mean vector.
  """

  return np.mean(data, axis=0)

def coordinate_wise_median(data):
  """
  Computes the coordinate-wise median of the given data.

  Parameters:
  data (np.ndarray): A 2D array where rows represent samples and columns represent features.

  Returns:
  np.ndarray: The coordinate-wise median vector.
  """
  return np.median(data, axis = 0)

# num_blocks blocks
def median_of_means(data, num_blocks=10):
  """
  Computes the Median of Means (MoM) estimator.
  
  The MoM estimator divides the data into several blocks, computes the mean of each block, 
  and then takes the coordinate-wise median of these means.

  Parameters:
  data (np.ndarray): A 2D array where rows represent samples and columns represent features.
  num_blocks (int): The number of blocks to divide the data into.

  Returns:
  np.ndarray: The MoM estimator vector.
  """

  n, d = data.shape
  if num_blocks > n:
    num_blocks = n
  block_size = n // num_blocks
  blocks = [data[i*block_size:(i+1)*block_size, :] for i in range(num_blocks)]
  block_means = [np.mean(block, axis=0) for block in blocks]
  return np.median(block_means, axis=0)

# maybe add some error checking below
def coord_trimmed_mean(data, tau):
  """
  Computes the coordinate-wise trimmed mean of the given data.
  
  The coordinate-wise trimmed mean sorts the data along each coordinate, removes a specified 
  fraction of the lowest and highest values, and then computes the sample mean of the remaining data. 
  Data points removed in one coordinate have no effect on data points removed in another coordinate.

  Parameters:
  data (np.ndarray): A 2D array where rows represent samples and columns represent features.
  tau (float): The fraction of data to trim from both ends.

  Returns:
  np.ndarray: The coordinate-wise trimmed mean vector.
  """

  n, d = data.shape
  # sort each coordinate individually
  coord_sort = np.sort(data, axis=0)
  # prune the bottom and top tau percentage in every coordinate
  pruned_data = coord_sort[math.floor(n*tau):n-math.ceil(n*tau)][:]
  # return the mean of the remaining data
  mean = np.mean(pruned_data, axis=0)
  return mean

def geometric_median(data, iters=2):
  """
  Computes the geometric median of the given data using Weiszfeld's algorithm.
  
  The geometric median is a point that minimizes the sum of Euclidean distances to a set of sample points.
  Weiszfeld's algorithm is an iterative procedure to approximate this median.

  Parameters:
  data (np.ndarray): A 2D array where rows represent samples and columns represent features.
  iters (int): The number of iterations for Weiszfeld's algorithm (default is 2).

  Returns:
  np.ndarray: The geometric median vector.
  """

  curr_estimate = np.mean(data, axis=0)

  for _ in range(iters):
    num = 0
    den = 0
    dist = np.linalg.norm(data - curr_estimate, axis=1)[:, np.newaxis] 
    num = (data / dist).sum(axis=0) 
    den = (1 / dist).sum()
    curr_estimate = num / den

  return curr_estimate
