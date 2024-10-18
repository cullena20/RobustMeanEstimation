"""
Mean estimation based on the following paper:

Jasper CH Lee and Paul Valiant.
Optimal sub-gaussian mean estimation in very high dimensions. 
In 13th Innovations in Theoretical Computer Science Conference (ITCS 2022), 2022.
URL https://arxiv.org/abs/2011.08384
"""

import numpy as np
import math
from simple_estimators import median_of_means 

def lee_valiant_original(data, tau, mean_estimator = median_of_means, gamma=0.1):
  """
  Implements the original Lee-Valiant algorithm for robust mean estimation.

  This function first takes a random gamma percentage sample of the data and computes a 
  preliminary mean using a given mean estimator. It calculates an average of the coordinate-wise
  differences of points from this preliminary mean estimate, weighting points included in the original
  mean estimate and the t percentage of points with the largest distance from the sample mean by 0. 
  The final mean is calculated as a sum of the original mean estimate with this average.

  Parameters:
  data (np.ndarray): A 2D array where rows represent samples and columns represent features.
  mean_estimator (function): A function to estimate the mean of the given data.
  gamma (float): The fraction of the data to be used for the preliminary mean estimate. 
  t (float): The fraction of the data to be pruned based on the distance from the preliminary mean.
  
  Note on parameters:
  Instead of using the astronomically large parameters described in the paper, which preclude
  a practical implementation, we set gamma as 0.5 and t as expected corruption.
  """

  n, d = data.shape

  # examine gamma percentage of points
  m = math.floor(gamma * n)
  random_idx = np.random.choice(np.arange(n), size=m, replace=False)

  # preliminary mean estimate on this gamma percentage
  mean = mean_estimator(data[random_idx])

  # calculate distances of points from mean estimate, and sort indices
  distances = np.linalg.norm((data - mean), axis=1)
  sorted_indices = np.argsort(distances)

  # sort differences based on distances from mean estimate
  differences = data - mean
  differences = differences[sorted_indices]

  # create a mask to mask out points included in initial mean estimate
  mask = np.ones(n, dtype=bool)
  mask[random_idx] = False
  mask = mask[sorted_indices]

  # also mask out tau percentage of furthest points from the intitial mean estimate (examining all points)
  s = math.floor(tau * n)
  mask[-s:] = False

  # apply this mask to the differences vector
  differences = differences[mask]

  # calculate final mean using initial estimate and average of differences, masking out certain points
  final_mean = mean + np.sum(differences, axis=0) / n 

  return final_mean

def lee_valiant_simple(data, tau, mean_estimator=median_of_means):
  """
  Implements a simplified version of the Lee-Valiant algorithm for robust mean estimation.
  
  This function first computes a mean estimate of all the data using a given mean estimator,
  calculates the distances of all points from this mean, sorts the points based on these distances,
  prunes the farthest points based on a given threshold, and returns the mean of the remaining data.

  Parameters:
  data (np.ndarray): A 2D array where rows represent samples and columns represent features.
  mean_estimator (function): A function to estimate the mean of the given data.
  tau (float): The fraction of the data to be retained after pruning the farthest points.
  """

  n, d = data.shape

  mean = mean_estimator(data) 
  distances = np.linalg.norm((data - mean), axis=1) 

  # sort indices based on distance from mean, wish to remove greatest projections
  sorted_indices = np.argsort(distances)
  distances = distances[sorted_indices]
  data = data[sorted_indices]

  # prune out furthest tau percent points from mean
  s = math.ceil((1 - tau) * n)
  pruned_data = data[:s][:]

 
  return np.mean(pruned_data, axis=0)