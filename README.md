# Robust Mean Estimation Suite

This code accompanies the paper "Robust High-Dimensional Mean Estimation With Low Data Size". 
It includes a suite of robust mean estimation algorithms and experimental infrastructure to run your own experiments over your own settings, algorithms, and data. We also provide files to recreate the experiments in the paper.

Here we cover some main usage points of this library. See basic_usage.py for a code tutorial to 1) Use mean estimators and data generating functions 2) recreate synthetic data experiments similar to those seen in the paper and define your own experiments 3) perform experiments over your own data and recreate embedding experiments seen in the paper.

## Calling Robust Mean Estimation Algorithms

Mean estimators are located in Algorithms. Each mean estimator is of the form: mean_estimator(data, tau, **kwargs) => mean_estimate. Data is in the form of n by d numpy arrays, where n is the data size, and d is the dimensionality of the data. Tau is the expected corruption of the data. **kwargs are keyword arguments specific to the mean estimator. Mean_estimate is a d dimensional mean estimate.

To call a mean estimator on your data, simply pass in the data and expected corruption. You can experiment with hyperparameters, but default values are already supplied for all estimators. Most estimators are not overly sensitive to the choice of expected corruption, so overestimates of tau tend to work fine.

For example, to return means found by quantum entropy scoring, eigenvalue filtering, and median of means of your data simply run the following.

```python
from eigenvalue_pruning import eigenvalue_pruning
from que import que_mean
from simple_estimators import median_of_means
import numpy as np

data = ... # sub in your data, ensuring that it is a numpy array of the appropriate convention
tau = 0.3 # an overestimate of true corruption tends to work fine

ev_mean_estimate = eigenvalue_pruning(data, tau)
que_mean_estimate = que_mean(data, tau)
med_mean_estimate = median_of_means(data, tau)
```

## Data Generation

Data generation functions are located in DataGeneration. data_generation.py contains the main data generation function, generate_data_helper. This function will take in an inlier data generation function, a corruption function, along with the desired data size, dimensionality, and corruption level, and return corrupted data. Different choices of parameters, inlier, and corruption functions can be fed into this function to allow for experimentation over new settings using our infrastructure. inlier_data.py contains inlier data generation functions, currently just a function to generate Gaussian data. corruption_schemes.py contains several corruption schemes, including both additive corruption schemes - corrupted points are generated and then appended to inlier data - and general corruption schemes - the corruption scheme directly manipulates a percentage of the data, such as subtractive corruption. By following the appropriate interfaces, additional inlier and corruption schemes can be developed and tested.

For example, say you want to generate gaussian data corrupted with additive variance shell noise and examine the mean.

```python
from data_generation import generate_data_helper
from inlier_data import gaussian_data
from corruption_schemes import gaussian_noise_one_cluster

n = 500 # data size
d = 1000 # dimensions
eta = 0.2 # corruption percentage

data, good_sample_mean, true_mean = generate_data_helper(n, d, eta, gaussian_data, gaussian_noise_one_cluster)
```

Now you can perform experiments using this data, such as examining the error of mean estimates compared to the good_sample_mean or true_mean. To continue we may have the following.

```python
from que import que_mean
import numpy as np

que_mean_estimate = que_mean(data)

sample_error = np.linalg.norm(que_mean_estimate - good_sample_mean)
true_error = np.linalg.norm(que_mean_estimate - true_mean)
```

You can also fit your own data into this interface using create_fun_from_data. This function takes in data and returns a data generation function that randomly draws from the data supplied. This can be used as an uncorrupted or corrupted function in generate_data helper, hence allowing experiments to be run using our interfaces.

## Running Experiments

More extensive experiments can be performed using the functions in helper.py under Experiments. We build infrastructure to perform experiments to examine error (defined as the Euclidean distance from a mean estimate to the true mean or to the sample mean of the inliers) as we vary data size, dimensionality, true corruption, expected corruption, or in the case of non identity covariance the top eigenvalue. Each run of an experiment will return a plot of errors versus the variable being varied, along with a pickle file with the errors and standard deviations of each algorithm. To run a set of experiments, first define the default variables and variables being varied. Each experiment will be an array of the following form: [varying_variable, varying_range, default_n, default_d, default_eps, default_tau] where the default value for the variable being varied is set to None. A list of individual experiments can be made to define a series of experiments that will be ran and plotted in a grid. To recreate the majority of experiments in the paper, this can be done as follows:

```python
default_n = 500
default_d = 500
default_eps = 0.1
default_tau = None # tau = None means that we will use tau=eps

experiments = [["n", np.arange(20, 5021, 500), None, default_d, default_eps, default_tau],
               ["n", np.arange(20, 521, 50), None, default_d, default_eps, default_tau],
               ["d", np.arange(20, 1021,  100), default_n, None, default_eps, default_tau],
               ["eps", np.arange(0, 0.46, 0.05), default_n, default_d, None, default_tau]]
```

Now define a data generation function that takes in data size, dimensionality, and corruption at every iteration of the experiment. This can be done using generate_data_helper as follows, where we recreate additive variance shell noise over identity covariance data:

```python
id_gaussian_corruption_one_cluster = lambda n, d, eps: generate_data_helper(n, d, eps, uncorrupted_fun=gaussian_data, noise_fun=gaussian_noise_one_cluster, mean_fun=mean_fun)
```

To utilize different distributions, we can modify the inlier data generating function and the corruption generating function.

Then, we must define a dictionary of estimators that we wish to evalute. Say we wish to just evaluate quantum entropy scoring, the sample mean, and median of means. Then we can define a dictionary of estimators as follows:

```python
from que import que_mean
from simple_estimators import sample_mean, median_of_means

main_estimators = {
    "sample_mean": lambda data, tau: sample_mean(data),
    "median_of_means": lambda data, tau: median_of_means(data, 10),
    "que_low_n": lambda data, tau: que_mean(data, tau),
}
```

Note how we are using lambda functions to enforce the appropriate interfaces. Then a simple experiment can be run as follows:

```python
errors, stds = experiment_suite(estimators, id_gaussian_corruption_one_cluster, experiments, runs=5, save_title="simple_experiment")
```

This call will return a dictionary of errors and standard deviations for each estimator and will also save the plotted results under "simple_experiment.png".

This experimental infrastructure can be adapted to your own supplied data by creating a data generation function that draws from this data. embedding_experiment_suite does this for you and allows for experiments where inlier data and corruption are drawn from two provided data sets of the same dimensionality. To further personalize this, you can directly define data generation functions to allow for more sophisticated corruption schemes based on your own data.

synthetic_setup.py and synthetic_experiments.py setup and run the synthetic experiments in our paper. embedding_setup.py and embedding_experiments.py setup and run the embedding experiments in our paper. These can be run to replicate our results and can be examined to see how you might run more sophisticated experiments. 

## Embeddings

We additionally provide the embeddings that we utilized for experiments on large language model, deep pretrained image model, and context free word embeddings under Embeddings. 