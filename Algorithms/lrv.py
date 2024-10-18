"""
Robust mean estimation based on recursive subspace projection from the following paper:

Kevin A Lai, Anup B Rao, and Santosh Vempala. Agnostic estimation of mean and covariance. In 2016
IEEE 57th Annual Symposium on Foundations of Computer Science (FOCS), pp. 665-674. IEEE, 2016.
URL https://arxiv.org/abs/1604.06968

Code is based on MatLab code from https://github.com/kevinalai/AgnosticMeanAndCovarianceCode
"""

import numpy as np
from scipy.stats import norm

def lrv(data, C=1, trace_est_option="robust"):
  """
  Return the LRV mean estimate using downweighting based on Gaussian assumption

  The LRV method recursively reduces the dimension by half, until 1 or 2 dimension
  remains. In the (≤ 2)-dimensional base case, it returns coord_median. Otherwise
  it recursively projects onto the span of the top half of the singular vectors,
  finding a mean estimate along this direction and projecting back up.

  Parameters:
  data (np.ndarray): A 2D array where rows represent samples and columns represent features.
  C: controls outlier weighting
  trace_est_option: "robust" to use robust trace estimate for LRV paper; anything else to use sample trace
  """
  n, d = data.shape

  if d <= 2:
    return np.median(data, axis=0)

  # gives weights for all points, downweighting outliers
  w = outlier_damping(data, C, trace_est_option) # n by 1

  # calculate the weighted covariance matrix
  mu_hat = w.T @ data / n
  centered_data = data - mu_hat 
  weighted_data = centered_data * np.sqrt(w) 

  weighted_cov = weighted_data.T @ weighted_data / n # beware overflow errors here with corruption too large

  # Calculate the eigendecomposition of the weighted covariance matrix
  eigenvalues, eigenvectors = np.linalg.eig(weighted_cov)

  eigenvalues = np.maximum(np.real(eigenvalues), 1e-10)
  eigenvectors = np.real(eigenvectors)

  # Sort eigenvectors by eigenvalues in increasing order
  sorted_indices = np.argsort(eigenvalues)
  sorted_eigenvectors = eigenvectors[:, sorted_indices] # d by d

  # Compute projection matrix using the first half of sorted eigenvectors (smallest values)
  PW = sorted_eigenvectors[:, :d//2] @ sorted_eigenvectors[:, :d//2].T 

  # Project data using PW and find the mean
  weightedProjData = np.multiply(data @ PW, w) # n by d
  est1 = np.mean(weightedProjData, axis=0) # 1 by d

  # Compute projection matrix using top half of sorted eigenvectors (largest values)
  QV = sorted_eigenvectors[:, d//2:] # d by d // 2

  # Recursively compute estimate on data projected using QW
  est2 = lrv(data @ QV, C, trace_est_option=trace_est_option) # 1 by d // 2
  est2 = est2 @ QV.T # 1 by d

  # return sum of both estimates
  est = est1 + est2

  return est

def outlier_damping(data, C=1, trace_est_option="robust"):
  """
  Computes weights to downweight outliers for the LRV algorithm.
  A weight for data point x_i is calculated as w_i = exp(-norm(x_i-a)^2 / s2)
  where a is an intitial mean estimate, here the coordinate wise median,
  and s2 is the trace estimate of the true covariance matrix.

  Parameters:
  data (np.ndarray): A 2D array where rows represent samples and columns represent features.

  Returns:
  np.ndarray: The n data points by 1 dimension weights matrix where each entry is a weight for
  the corresponding data point
  """

  if trace_est_option == "sample":
    s2, Z = trace_est(data)
  else:
    s2, Z = lrv_trace_est(data)
  # s2 is trace estimate, Z is squared distances from coordinate wise median (n, 1)

  T = np.sum(Z, axis=1)[:, np.newaxis] # n by 1

  temp = (-T / (C*s2))

  w = np.exp(temp) 

  # cap w to handle numerical stability issues that arrise with distant outliers
  w = np.maximum(w, 1e-10)

  return w


def trace_est(data):
  n, d = data.shape
  mu_hat = np.mean(data, axis=0)  # Sample mean
  T = (1 / (n - 1)) * np.sum(np.linalg.norm(data - mu_hat, axis=1)**2)
  med = np.median(data, axis=0)
  Z = np.square(data-med)
  return T, Z

def lrv_trace_est(data):
  """
  Computes a trace estimate for the true covariance matrix of the potentially corrupted data
  following a naive method proposed in LRV. Also returns the squared distances of the data
  from the coordinate wise median, to be used in outlier_damping.

  Project onto d dimensions orthogonal directions (here the standard basic vectors) and compute
  1d estimates of the median and standard deviation. 

  Parameters:
  data (nd.ndarray):  A 2D array where rows represent samples and columns represent features.

  Returns:
  T (float): A trace estimate of the true covariance matrix
  Z (np.ndarray): n data points by d dimension matrix containing the coordinate wise squared difference of the
  coordinate wise median from each data point
  """

  n, d = data.shape
  meds = np.zeros(d)
  I = np.eye(d)

  # UNCOMMENT
  T = 0
  for i in range(d):
    m, sigma2 = estG1D(data, I[i])
    m = m.real
    sigma2=sigma2.real
    meds[i] = m
    T += sigma2
    
  Z = np.square(data - meds)

  return T, Z


# estimate mean and variance of gaussian along a dirrection v
def estG1D(data, v):
    """
    Return 1 dimensional estimates for the mean and standard deviation of data projected along a direction v

    Parameters:
    data (nd.ndarray):  A 2D array where rows represent samples and columns represent features.

    Returns:
    mu (float): A one dimensional mean estimate, simply the median of the data projected on v
    sigma2 (float): A one dimensional standard deviation estimate
    """

    v = v / np.linalg.norm(v) 
    Z = data @ v
    mu = np.median(Z)
    Z -= mu

    # gets spread of the middle 20% of the data
    topQuant = 0.6
    botQuant = 0.4
    diff = np.quantile(Z, topQuant) - np.quantile(Z, botQuant)

    sigma2 = (diff / (norm.ppf(topQuant) - norm.ppf(botQuant))) ** 2
    return mu, sigma2

def lrvGeneral(data, eta):
    """
    Returnsn the LRV mean estimate by completely pruning outliers, not utilizing Gaussian assumption

    Parameters:
    data (np.ndarray): A 2D array where rows represent samples and columns represent features.
    eta (float): Noise fraction.
    """
    n = data.shape[1]

    if n <= 1:
        temp = np.array([estGeneral1D(data, np.ones(n), eta)])
        return temp
    
    w = outRemBall(data, eta)
    newdata = data[w > 0]

    S = np.cov(newdata, rowvar=False)
    D, V = np.linalg.eigh(S)

    # Ensure the eigenvalues are in ascending order
    sorted_indices = np.argsort(D)
    V = V[:, sorted_indices]
    
    k = n // 2

    PW = V[:, :k] @ V[:, :k].T
    weightedProjdata = newdata @ PW
    est1 = np.mean(weightedProjdata, axis=0)

    QV = V[:, k:]
    est2 = lrvGeneral(data @ QV, eta)
    est2 = est2 @ QV.T
    
    est = est1 + est2
    return est

def estGeneral1D(data, v, eta):
    """
    Estimate the mean of the projection of the data data onto the vector v,
    excluding the fraction eta of outliers.
    """
    # Normalize the vector v
    v = v / np.linalg.norm(v)
    
    # Project the data onto v
    Z = np.dot(data, v)
    
    # Sort the projections
    Z = np.sort(Z)
    
    # Calculate the interval width
    m = len(Z)
    intervalWidth = int(np.floor(m * (1 - eta)**2))
    
    # Calculate the lengths of intervals
    lengths = np.zeros(m - intervalWidth + 1)
    for i in range(m - intervalWidth + 1):
        lengths[i] = Z[i + intervalWidth - 1] - Z[i]
    
    # Find the index of the smallest interval
    ind = np.argmin(lengths)
    
    # Calculate the mean of the smallest interval
    mu = np.mean(Z[ind:ind + intervalWidth])
    
    return mu

def outRemBall(data, eta):
    """
    Removes points outside of a ball containing (1-eta)^2 fraction of the
    points. The ball is centered at the coordinate-wise median.
    The weight vector returned has 0 weight for points from data that are
    outside this ball.

    Returns:
    numpy.ndarray: Weight (column) vector w that is 0 for "removed" points.
    """
    m, n = data.shape
    med = np.median(data, axis=0)
    
    w = np.ones(m)
    
    Z = data - med
    T = np.sum(Z**2, axis=1)
    thresh = np.percentile(T, 100 * (1 - eta)**2)
    
    w[T > thresh] = 0
    
    return w


