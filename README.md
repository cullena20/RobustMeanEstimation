# Robust Mean Estimation Suite

This code accompanies the paper "Robust High-Dimensional Mean Estimation With Low Data Size". 
It includes a suite of robust mean estimation algorithms and experimental infrastructure to run your own experiments and recreate the experiments in the paper.

## Calling Robust Mean Estimation Algorithms

Mean estimators are located in Algorithms. Each mean estimator is of the form: mean_estimator(data, tau, **kwargs) => mean_estimate. Data is in the form of n by d numpy arrays, where n is the data size, and d is the dimensionality of the data. Tau is the expected corruption of the data. **kwargs are keyword arguments specific to the mean estimator. Mean_estimate is a d dimensional mean estimate.

To call a mean estimator on your data, simply pass in the data and expected corruption. You can experiment with hyperparameters, but default values are already supplied for all estimators. Most estimators are not overly sensitive to the choice of expected corruption, so overestimates tend to work fine.

For example, to return means found by quantum entropy scoring, eigenvalue filtering, and median of means of your data simply run the following.

`
from eigenvalue_pruning import eigenvalue_pruning
from que import que_mean
from simple_estimators import median_of_means
import numpy as np

data = ... # sub in your data, ensuring that it is a numpy array of the appropriate convention
tau = 0.3

ev_mean_estimate = eigenvalue_pruning(data, tau)
que_mean_estimate = que_mean(data, tau)
med_mean_estimate = median_of_means(tau)
`

## Data Generation

Data generation functions are located in DataGeneration. data_generation.py contains the main data generation function, generate_data_helper. This function will take in an inlier data generation function, a corruption function, along with the desired data size, dimensionality, and corruption level, and return corrupted data. Different choices of parameters, inlier, and corruption functions can be fed into this function to allow for experimentation over new settings using our infrastructure. inlier_data.py contains inlier data generation functions, currently just a function to generate Gaussian data. corruption_schemes.py contains several corruption schemes, including both additive corruption schemes - corrupted points are generated and then appended to inlier data - and general corruption schemes - the corruption scheme directly manipulates a percentage of the data, such as subtractive corruption. By following the appropriate interfaces, additional inlier and corruption schemes can be developed and tested.

For example, say you want to generate gaussian data corrupted with additive variance shell noise and examine the mean.

`
from data_generation import generate_data_helper
from inlier_data import gaussian_data
from corruption_schemes import gaussian_noise_one_cluster

n = 500 # data size
d = 1000 # dimensions
eta = 0.2 # corruption percentage

data, good_sample_mean, true_mean = generate_data_helper(n, d, eta, gaussian_data, gaussian_noise_one_cluster)
`

Now you can perform experiments using this data, such as examining the error of mean estimates compared to the good_sample_mean or true_mean. To continue we may have the following.

`
from que import que_mean
import numpy as np

que_mean_estimate = que_mean(data)

sample_error = np.linalg.norm(que_mean_estimate - good_sample_mean)
true_error = np.linalg.norm(que_mean_estimate - true_mean)
`

## Running Experiments

More extensive experiments can be performed using the functions in helper.py under Experiments. setup.py and main_experiments.py setup the experiments in our paper. We defer the reader to these files to see how to run more sophisticated experiments.

