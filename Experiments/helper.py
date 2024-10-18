"""
Contains helper functions for running experiments, notably experiment_suite
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import math
from simple_estimators import median_of_means
from lrv import estG1D
import pickle

def experiment_suite(estimators, generate_data, experiments, runs=5, compare_to_true=True, error_bars=False, save_title=None, plot_good_sample=False, plot=False, style_dict="main", rotate=False, legend=True, sample_scale_cov=False, prune_obvious=False,  pickle_results=False, columns=2, option="error", plot_optimal_error=False, xlabel=None, **kwargs):
    """
    Runs a set of experiments using the supplied estimators, data generation function, and experiments, along with hyperparameters

    Parameters:
        estimators - a dictionary of mean estimators of type: data, tau => mean_estimate
        generate_data - a data generation function that returns data, good_sample_mean (mean of inlieres), and true_mean (mean data is generated from)
        experiments - array of experiments - each individual experiment is of form [varying_variable, varying_range, n_fixed, d_fixed, eps_fixed, tau_fixed]
                    note eps here is true corruption percentage, as opposed to eta
                    note varying_variable is "n", "d", "eps", "tau", or "data")
                    note [varying_variable]_fixed should be set to 0 (except if varying_variable is "data")
                    A seperate plot will be generated for each experiment, and they will be merged into a grid
                    The default setting is to generate a 2 by 2 grid for 4 experiments
        runs - the number of runs averaged over.
        compare_to_true - true to compare to true mean, otherwise compares to inlier sample mean. This is always True in our experiments
        error_bars - visualize 1 std with error bars
        save_title - the name of the saved plot and possible pickle files
        plot_good_sample - True to plot error of inlier sample mean from true mean
        plot - True to plot directly (if False will not call plt.show() but can save figure)
        style_dict - Set name to choose styles for each estimator (each style dict is defined in this file and is given a name 
        rotate - True to rotate data in data generation, this is used for unconstrained covariance experiments
        legend - True to plot with a legend (will only be displayed on first plot if there are a series of experiments)
        sample_scale_cov - True to employ the sample covariance scaling heuristic on eigenvalue_filtering and que 
                            used whenever these estimators are called on non identity covariance data (does not effect other estimators) 
        prune_obvious - Employ a method to prune large outliers first
        pickle_results - True to save pickled experimental results with save_title
        columns - the number of columns in the final plot - set to 1 for single plots

        unutilized
        option - "error" by default, "time" to run same experiments but plot times instead of errors
        plot_optimal_error - plots O(sqrt(d/n)) which is the error we expect with no corruption
    """ 

    errors = []
    std_dev_list = []
    rows = (len(experiments) + columns - 1) // columns
    fig, axs = plt.subplots(rows, columns, figsize=(15, 5*rows))
    for idx, experiment in enumerate(experiments):
        print(f"Running Experiment {idx}")
        start_time = time.time()

        varying_variable = experiment[0]
        varying_range = experiment[1]
        n_fixed = experiment[2]
        d_fixed = experiment[3]
        eps_fixed = experiment[4]
        tau_fixed = experiment[5]

        if option=="error":
            error, std_devs = run_experiment(estimators, generate_data, varying_variable, varying_range, n_fixed=n_fixed, d_fixed=d_fixed, eps_fixed=eps_fixed, tau_fixed=tau_fixed, runs=runs, compare_to_true=compare_to_true, plot_good_sample=plot_good_sample, plot_optimal_error=plot_optimal_error, rotate=rotate, sample_scale_cov=sample_scale_cov, prune_obvious=prune_obvious, **kwargs)
        elif option=="time":
            error, std_devs = run_time_experiment(estimators, generate_data, varying_variable, varying_range, n_fixed=n_fixed, d_fixed=d_fixed, eps_fixed=eps_fixed, tau_fixed=tau_fixed, runs=runs, compare_to_true=compare_to_true, plot_good_sample=plot_good_sample, plot_optimal_error=plot_optimal_error, rotate=rotate, sample_scale_cov=sample_scale_cov, prune_obvious=prune_obvious, **kwargs)
        errors.append(error)
        std_dev_list.append(std_devs)
        
        end_time = time.time()
        print(f"Finished Running Experiment {idx} in {end_time-start_time:.2f} seconds")

        if varying_variable == "n":
            xlabel = "Data Size"
        elif varying_variable == "d":
            xlabel = "Dimensions"
        elif varying_variable == "eps":
            xlabel = "Corruption"
        elif varying_variable == "tau":
            xlabel = "Expected Corruption"
        # just need to manually change this label
        elif varying_variable == "data":
            # xlabel = "Coordinate Wise Variance"
            # xlabel = "Coordinate Wise STD" # for spherical
            xlabel = "Squareroot Of Top Eigenvalue" # for non spherical

        row = idx // columns
        col = idx % columns
        ax = axs[row, col] if rows > 1 else axs[col]
        try: 
            plot_results(error, varying_range, xlabel, std_devs=std_devs, error_bars=error_bars, n_fixed=n_fixed, d_fixed=d_fixed, eps_fixed=eps_fixed, tau_fixed=tau_fixed, ax=ax, style_dict=style_dict, option=option)
            if idx == 0 and legend:
                ax.legend()
            plt.tight_layout()
        except Exception as e: # if plotting fails, move on to the next run
            print(f"Couldn't plot results with exception {e}")
            continue
    
    if pickle_results and save_title:
        filename = f"ExperimentResults/{save_title}.pkl"

        data_to_pickle = {
            "errors": errors,
            "stds": std_dev_list,
            "experiments": experiments, 
        }

        with open(filename, "wb") as f:
            pickle.dump(data_to_pickle, f)
    if save_title:
        plt.savefig(f"ExperimentResults/{save_title}")
    
    if plot:
        plt.show()

    return errors, std_devs

# use eps for true corruption, tau for expected corruption
def run_experiment(estimators, generate_data, varying_variable, varying_range, compare_to_true=True, runs=5, n_fixed=None, d_fixed=None, eps_fixed=None, tau_fixed=None, plot_good_sample=False, plot_optimal_error=False, rotate=False, sample_scale_cov=False, prune_obvious=False, **kwargs):
    """
    Run a single experiment 
    """
    errors = {key: [np.empty(runs) for _ in range(len(varying_range))] for key in estimators.keys()}
    average_errors = {key: [] for key in estimators.keys()}
    std = {key: [] for key in estimators.keys()}
    if plot_good_sample:
        errors["good_sample_mean"] = [np.empty(runs) for _ in range(len(varying_range))]
        average_errors["good_sample_mean"] = []
        std["good_sample_mean"] = []
    if plot_optimal_error: # not useful
        errors["optimal_error1"] = [np.empty(runs) for _ in range(len(varying_range))]
        average_errors["optimal_error1"] = []
        std["optimal_error1"] = []

    for idx in range(len(varying_range)):
        for name, _ in estimators.items():
            errors[name][idx] = np.empty(runs)

    for run in range(runs):
        print(f"Run Number {run}")
        for idx in range(len(varying_range)):
            value = varying_range[idx]
            if varying_variable == "data":
                n = n_fixed
                d = d_fixed
                eps = eps_fixed
                if tau_fixed is None:
                    tau = eps
                else:
                    tau = tau_fixed
                X, good_sample_mean, true_mean = generate_data(n, d, eps, value)
            else:
                if varying_variable == "n": # data points
                    n = value
                    d = d_fixed
                    eps = eps_fixed
                    if tau_fixed is None:
                        tau = eps
                    else:
                        tau = tau_fixed
                elif varying_variable == "d": # dimensions
                    d = value
                    n = n_fixed
                    eps = eps_fixed
                    if tau_fixed is None:
                        tau = eps
                    else:
                        tau = tau_fixed
                elif varying_variable == "eps": # true corruption
                    eps = value
                    n = n_fixed
                    d = d_fixed
                    if tau_fixed is None:
                        tau = eps
                    else:
                        tau = tau_fixed
                elif varying_variable == "tau": # expected corruption
                    n = n_fixed
                    d = d_fixed
                    eps = eps_fixed
                    tau = value

                X, good_sample_mean, true_mean = generate_data(n, d, eps)
            
            if rotate:
                rotation_matrix = random_rotation_matrix(d)
                X = (rotation_matrix @ X.T).T
                good_sample_mean = rotation_matrix @ good_sample_mean
                true_mean = rotation_matrix @ true_mean

            if prune_obvious:
                # this method prunes large outliers by estimating a shell around the data and removing points far outside of this shell
                X = prune_obvious_fun(X)

            if sample_scale_cov:
                trace = np.trace(np.cov(X, rowvar=False)) # Just calculate sample covariance
                std_est = math.sqrt(trace/d) # now we have a coordinate wise variance estimate to scale data by
                scaled_X = X / std_est # data should now have identity covariance

                # now we test ev_filtering and que_low_n on this scaled version of X

            if compare_to_true:
                compare_to_mean = true_mean
            else:
                compare_to_mean = good_sample_mean

            for name, function in estimators.items():
                # name specific which isn't very nice
                if sample_scale_cov and (name.startswith("ev_filtering_low_n") or name == "ransac_mean" or name.startswith("que")): 
                    mean = function(scaled_X, tau=tau)
                    mean = mean * std_est
                else:
                     mean = function(X, tau=tau)
                error = np.linalg.norm(mean - compare_to_mean)
                errors[name][idx][run] = error
            if plot_good_sample:
                error = np.linalg.norm(good_sample_mean - compare_to_mean)
                errors["good_sample_mean"][idx][run] = error
            # we don't ever use below
            if plot_optimal_error and compare_to_true:
                error1 = math.sqrt(d/n) + eps
                errors["optimal_error1"][idx][run] = error1
            elif plot_optimal_error and not compare_to_true:
                error1 = eps
                errors["optimal_error1"][idx][run] = error1
  

    for name in errors:
        for idx in range(len(varying_range)):
            errors_for_estimator = errors[name][idx]

            # Calculate average error
            avg_error = np.mean(errors_for_estimator)
            average_errors[name].append(avg_error)

            # Calculate standard deviation
            std_dev = np.std(errors_for_estimator)
            std[name].append(std_dev)

    return average_errors, std

# need to change how I deal with this style dict business
def plot_results(errors, varying_range, xlabel, std_devs=None, error_bars=False, n_fixed=None, d_fixed=None, eps_fixed=None, tau_fixed=None, ax=None, style_dict=None, option="error"):
    '''
    Plots filter error and sample error vs varying_variable. Sample error is subtracted from error so that 0 error represents 0 good sample
    error. The varying_variable is labeled by x_label. Other values held constant may be supplied to print out more information.
    '''
    if ax is None:
        ax = plt.gca()

    for name, error in errors.items():
        if style_dict is None:
            # below is to shade in instead of using lines
            # somehow lee and valiant has way higher standard deviation than others
            if error_bars and std_devs is not None:
                std_dev = std_devs[name]
                ax.plot(varying_range, error, label=f"{name} Error", alpha=0.8)
                ax.fill_between(varying_range, [e - sd for e, sd in zip(error, std_dev)], 
                                [e + sd for e, sd in zip(error, std_dev)], alpha=0.3)
            else:
                ax.plot(varying_range, error, label=f"{name} Error", alpha=0.5)
        else:
            if style_dict == "main":
                style = main_style_dict[name]
            elif style_dict == "lee_valiant":
                style = lee_valiant_style_dict[name]
            elif style_dict == "med_mean":
                style = median_of_means_style_dict[name]
            elif style_dict == "lrv":
                style = lrv_style_dict[name]
            elif style_dict == "ev":
                style = ev_hyperparameter_style_dict[name]
            elif style_dict == "ev_compare":
                style = ev_style_dict[name]
            elif style_dict == "lrv_sample":
                style = lrv_sample_style_dict[name]
            elif style_dict=="pgd":
                style = pgd_style_dict[name]
            elif style_dict == "ev_pruning":
                style = ev_pruning_style_dict[name]
            elif style_dict == "que":
                style = que_style_dict[name]

            # below is to shade in instead of using lines
            if error_bars and std_devs is not None:
                std_dev = std_devs[name]
                ax.plot(varying_range, error, label=f"{name} Error", 
                        color=style['color'], marker=style['marker'], linestyle=style['linestyle'], 
                        linewidth=style['linewidth'], alpha=0.8)
                ax.fill_between(varying_range, [e - sd for e, sd in zip(error, std_dev)], 
                                [e + sd for e, sd in zip(error, std_dev)], 
                                color=style['color'], alpha=0.3)
            else:
                ax.plot(varying_range, error, label=f"{name} Error", 
                        color=style['color'], marker=style['marker'], linestyle=style['linestyle'], 
                        linewidth=style['linewidth'], alpha=0.5)

    ax.set_xlabel(xlabel)
    if option == "time":
        ax.set_ylabel("Time (seconds)")
    else:
        ax.set_ylabel('Error')

    if n_fixed is not None or d_fixed is not None or eps_fixed is not None or tau_fixed is not None:
        if option == "time":
            title = f'Time vs {xlabel}\n'
        else:
            title = f'Error vs {xlabel}\n'
        if n_fixed is not None:
            title += f'Data Size: {n_fixed}, '
        if d_fixed is not None:
            title += f'Dimensions: {d_fixed}, '
        if tau_fixed is not None:
            title += f'Expected Corruption: {tau_fixed:.2f}, '
        if eps_fixed is not None:
            title += f'Corruption Percentage: {eps_fixed:.2f}, '
        ax.set_title(title[:-2])

def random_rotation_matrix(d):
    """
    Generate a random rotation matrix in d dimensions
    """
    random_matrix = np.random.randn(d, d)
    orthogonal_matrix, _ = np.linalg.qr(random_matrix)
    return orthogonal_matrix


def scale_data(data, cov):
    """
    Scale data to have identity covariance given known covariance
    Return scaled data and the inverse transformation to scale its mean back
    """
    # Compute the eigenvalue decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

     # Form the whitening matrix
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.maximum(eigenvalues, 1e-8))))

    # Form the transformation matrix
    transform_matrix = eigenvectors @ D_inv_sqrt @ eigenvectors.T # d by d
    
    # Scale the data
    data = data @ transform_matrix.T 

    # Generate inverse transform matrix
    inverse_transform = np.linalg.pinv(transform_matrix)

    return data, inverse_transform


def prune_obvious_fun(data, C=5, mean_option=None, trace_option=None):
    """
    Prunes large outliers based on those laying far outside of a shell around the data
    This shell is estimated using median of means as its center and a robust trace estimate as its radius
    """
    n, d = data.shape

    if trace_option is None:
        trace = d # assume identity covariance default
    elif trace_option == "sample":
        trace = np.trace(np.cov(data, rowvar=False))
    else:
        trace, _ = trace_est(data) # robust trace
        
    std_est = math.sqrt(trace/d) 

    if mean_option is None:
        mean = median_of_means(data, num_blocks=3) # median of means default
    else:
        mean = np.mean(data, axis=0) # only use sample mean if nothing specified
        
    dist_to_mean = np.linalg.norm(data - mean, axis=1)

    inlier_indices = np.where(dist_to_mean <= C * math.sqrt(d) * std_est)
    return data[inlier_indices]

def trace_est(data):
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

  T = 0
  for i in range(d):
    m, sigma2 = estG1D(data, I[i])
    meds[i] = m
    T += sigma2

  Z = np.square(data - meds) # n by d

  return T, Z

# brute force copy and paste experiment to measure times, really this should be done while running experiments
def run_time_experiment(estimators, generate_data, varying_variable, varying_range, compare_to_true=True, runs=10, n_fixed=None, d_fixed=None, eps_fixed=None, tau_fixed=None, plot_good_sample=False, plot_optimal_error=False, rotate=False, sample_scale_cov=False, prune_obvious=False, **kwargs):
    errors = {key: [np.empty(runs) for _ in range(len(varying_range))] for key in estimators.keys()}
    average_errors = {key: [] for key in estimators.keys()}
    std = {key: [] for key in estimators.keys()}

    for idx in range(len(varying_range)):
        for name, _ in estimators.items():
            errors[name][idx] = np.empty(runs)

    for run in range(runs):
        print(f"Run Number {run}")
        for idx in range(len(varying_range)):
            value = varying_range[idx]
            if varying_variable == "data":
                n = n_fixed
                d = d_fixed
                eps = eps_fixed
                if tau_fixed is None:
                    tau = eps
                else:
                    tau = tau_fixed
                X, good_sample_mean, true_mean = generate_data(n, d, eps, value)
            else:
                if varying_variable == "n": # data points
                    n = value
                    d = d_fixed
                    eps = eps_fixed
                    if tau_fixed is None:
                        tau = eps
                    else:
                        tau = tau_fixed
                elif varying_variable == "d": # dimensions
                    d = value
                    n = n_fixed
                    eps = eps_fixed
                    if tau_fixed is None:
                        tau = eps
                    else:
                        tau = tau_fixed
                elif varying_variable == "eps": # true corruption
                    eps = value
                    n = n_fixed
                    d = d_fixed
                    if tau_fixed is None:
                        tau = eps
                    else:
                        tau = tau_fixed
                elif varying_variable == "tau": # expected corruption
                    n = n_fixed
                    d = d_fixed
                    eps = eps_fixed
                    tau = value

                X, good_sample_mean, true_mean = generate_data(n, d, eps)
            

            # pretty sure the below makes sense, should double check
            if rotate:
                rotation_matrix = random_rotation_matrix(d)
                X = (rotation_matrix @ X.T).T
                good_sample_mean = rotation_matrix @ good_sample_mean
                true_mean = rotation_matrix @ true_mean

            # ISSUE - we should probably rotate data then prune? Or is this equivalent
            # only issue with doing this is we no longer scale on only eigenvalue
            # for the purposes of our work - just don't even use the trace option (boom resolved)
            if prune_obvious:
                X = prune_obvious_fun(X) # prune assuming identity trace and using med means as center for now
                # this won't work for non identity covariance for now

            if sample_scale_cov:
                # scale data by sample covariance estimate

                # sample trace
                trace = np.trace(np.cov(X, rowvar=False)) # Just calculate sample covariance
                print("sample", trace)

                # lrv trace
                # trace, _ = trace_est(X)
                # #trace =trace *2
                # print("lrv * 2", trace)
                # print()

                # print(f"SAMPLE TRACE: {trace}, LRV TRACE: {trace_lrv} ")

                std_est = math.sqrt(trace/d) # now we have a coordinate wise variance estimate to scale data by
                scaled_X = X / std_est # data should now have identity covariance

                # now we test algorithms on this scaled version of X


            for name, function in estimators.items():
                # name specific
                if sample_scale_cov and (name.startswith("ev_filtering_low_n") or name == "ransac_mean" or name.startswith("que")): # really poor placement of this
                    mean = function(scaled_X, tau=tau)
                    mean = mean * std_est
                else:
                    start_time= time.time()
                    mean = function(X, tau=tau)
                    end_time = time.time()
                error = end_time-start_time
                errors[name][idx][run] = error

    for name in errors:
        for idx in range(len(varying_range)):
            errors_for_estimator = errors[name][idx]

            # Calculate average error
            avg_error = np.mean(errors_for_estimator)
            average_errors[name].append(avg_error)

            # Calculate standard deviation
            std_dev = np.std(errors_for_estimator)
            std[name].append(std_dev)

    return average_errors, std

# STYLE DICTIONARIES 

# use - linestyle for baselines, other colors
# use : linestyle for less important
# use -. for most important
main_style_dict = {
    'sample_mean': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'linewidth': 2},  # C0
    'coord_median': {'color': '#7f7f7f', 'marker': 's', 'linestyle': ':', 'linewidth': 2}, 
    'median_of_means': {'color': '#2ca02c', 'marker': 'd', 'linestyle': ':', 'linewidth': 2},  # C2
    'coord_trimmed_mean': {'color': '#ba7213', 'marker': '^', 'linestyle': ':', 'linewidth': 2},  # C7
    # 'ransac_mean': {'color': '#9467bd', 'marker': 'v', 'linestyle': ':', 'linewidth': 2},  # C4
    'geometric_median': {'color': '#8c564b', 'marker': '<', 'linestyle': ':', 'linewidth': 2},  # C5
     # doubled to not break earlier stuff
    # 'lee_valiant_simple_mom': {'color': '#bcbd22', 'marker': '>', 'linestyle': ':', 'linewidth': 2},  # C8,
    'lee_valiant_simple': {'color': '#bcbd22', 'marker': '>', 'linestyle': ':', 'linewidth': 2},  # C8
    'lrv': {'color': '#d62728', 'marker': 'p', 'linestyle': '-.', 'linewidth': 3},  # C3
    'ev_filtering': {'color': '#2ca02c', 'marker': 'd', 'linestyle': ':', 'linewidth': 2},
    'ev_filtering_low_n': {'color': '#e377c2', 'marker': 'h', 'linestyle': '-.', 'linewidth': 3},  # C6
    # 'eigenvalue_pruning_unknown_cov': {'color': '#e377c2', 'marker': 'h', 'linestyle': '-.', 'linewidth': 3},  # C6
    'good_sample_mean': {'color': '#17becf', 'marker': '*', 'linestyle': '-', 'linewidth': 2},  # C9
    'que': {'color': '#d62728', 'marker': 'p', 'linestyle': '-', 'linewidth': 2},
    'que_low_n': {'color': '#9467bd', 'marker': 'v', 'linestyle': '-.', 'linewidth': 3},
    'pgd': {'color': '#ff7f0e', 'marker': 'P', 'linestyle': '-.', 'linewidth': 2},  # C1
    'sdp': {'color': '#1a5f91', 'marker': 'X', 'linestyle': '-.', 'linewidth': 3},
    'que_halt': {'color': '#CBA6FF', 'marker': 'p', 'linestyle': '-', 'linewidth': 2},  # C3 - only for robustness to expected corruption
}

pgd_style_dict = {
    'pgd-1': {'color': '#2ca02c', 'marker': 'd', 'linestyle': ':', 'linewidth': 2},
    'pgd-5': {'color': '#d62728', 'marker': 'p', 'linestyle': ':', 'linewidth': 2},
    'pgd-10': {'color': '#bcbd22', 'marker': '>', 'linestyle': ':', 'linewidth': 2},
    'pgd-15': {'color': '#ff7f0e', 'marker': 'P', 'linestyle': '-.', 'linewidth': 3},
    'pgd-20': {'color': '#9467bd', 'marker': 'v', 'linestyle': ':', 'linewidth': 2}
}

que_style_dict = {
    'que-0': {'color': '#2ca02c', 'marker': 'd', 'linestyle': ':', 'linewidth': 2},
    'que-0.5': {'color': '#1a5f91', 'marker': 'X', 'linestyle': ':', 'linewidth': 2},
    'que-1': {'color': '#d62728', 'marker': 'p', 'linestyle': ':', 'linewidth': 2},
    'que-4': {'color': '#9467bd', 'marker': 'v', 'linestyle': '-.', 'linewidth': 3},
    'que-50': {'color': '#ff7f0e', 'marker': 'P', 'linestyle': ':', 'linewidth': 2},
    'que-200': {'color': '#bcbd22', 'marker': '>', 'linestyle': ':', 'linewidth': 2}
}

ev_pruning_style_dict = {
    'ev_filtering_low_n': {'color': '#e377c2', 'marker': 'h', 'linestyle': '-.', 'linewidth': 3},
    'ev_filtering_low_n_random': {'color': '#ff7f0e', 'marker': 'P', 'linestyle': '-.', 'linewidth': 2},
    'ev_filtering_low_n_fixed': {'color': '#d62728', 'marker': 'p', 'linestyle': '-.', 'linewidth': 2}
}

uncorrupted_style_dict = {
    'sample_mean': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'linewidth': 2},  # C0
    'coord_median': {'color': '#ff7f0e', 'marker': 's', 'linestyle': ':', 'linewidth': 2},  # C1
    'median_of_means': {'color': '#2ca02c', 'marker': 'd', 'linestyle': ':', 'linewidth': 2},  # C2
    'coord_trimmed_mean': {'color': '#7f7f7f', 'marker': '^', 'linestyle': ':', 'linewidth': 2},  # C7
    'ransac_mean': {'color': '#9467bd', 'marker': 'v', 'linestyle': ':', 'linewidth': 2},  # C4
    'geometric_median': {'color': '#8c564b', 'marker': '<', 'linestyle': ':', 'linewidth': 2},  # C5
    'lee_valiant_simple_mom': {'color': '#bcbd22', 'marker': '>', 'linestyle': ':', 'linewidth': 2},  # C8
    'lrv': {'color': '#d62728', 'marker': 'p', 'linestyle': '-.', 'linewidth': 3},  # C3
    'updated_threshold_ev': {'color': '#e377c2', 'marker': 'h', 'linestyle': '-.', 'linewidth': 3},  # C6
    'lee_valiant_original_mom': {'color': '#17becf', 'marker': 'o', 'linestyle': ':', 'linewidth': 2}  # C9 no need to have good sample mean when this is used
}

lee_valiant_style_dict = {
    'lee_valiant_simple_mom': {'color': '#bcbd22', 'marker': 'o', 'linestyle': '-', 'linewidth': 3},  # C8
    "lee_valiant_original_mom": {'color': '#2ca02c', 'marker': '>', 'linestyle': ':', 'linewidth': 2}, # C2
    "lee_valiant_simple_lrv": {'color': '#ff7f0e', 'marker': 'o', 'linestyle': ':', 'linewidth': 2},  # C1,
    "lee_valiant_original_lrv": {'color': '#8c564b', 'marker': '>', 'linestyle': ':', 'linewidth': 2},  # C5,
    "lee_valiant_simple_ev": {'color': '#7f7f7f', 'marker': 'o', 'linestyle': ':', 'linewidth': 2}, # C7
    "lee_valiant_original_ev": {'color': '#9467bd', 'marker': '>', 'linestyle': ':', 'linewidth': 2}, # C4,
    'lrv': {'color': '#d62728', 'marker': 'p', 'linestyle': '-.', 'linewidth': 2},  # C3
    'ev_filtering_low_n': {'color': '#e377c2', 'marker': 'h', 'linestyle': '-.', 'linewidth': 2},  # C6
    'good_sample_mean': {'color': '#17becf', 'marker': '*', 'linestyle': '-', 'linewidth': 2}  # C9
}

ev_style_dict = {
    'sample_mean': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'linewidth': 2},  # C0
    'ev_filtering_low_n': {'color': '#e377c2', 'marker': 'h', 'linestyle': '-.', 'linewidth': 3},  # C6
    'eigenvalue_filtering': {'color': '#2ca02c', 'marker': '>', 'linestyle': ':', 'linewidth': 3},  # C2
    'good_sample_mean': {'color': '#17becf', 'marker': '*', 'linestyle': '-', 'linewidth': 2}  # C9
}

median_of_means_style_dict = {
    "median_of_means-3": {'color': '#1f77b4', 'marker': 'o', 'linestyle': ':', 'linewidth': 2}, # C0
    "median_of_means-5": {'color': '#ff7f0e', 'marker': 's', 'linestyle': ':', 'linewidth': 2}, # C1 
    "median_of_means-10": {'color': '#2ca02c', 'marker': 'd', 'linestyle': '-', 'linewidth': 3}, # C2
    "median_of_means-15": {'color': '#d62728', 'marker': 'p', 'linestyle': ':', 'linewidth': 2}, # C3
    "median_of_means-20": {'color': '#9467bd', 'marker': 'v', 'linestyle': ':', 'linewidth': 2}, # C4
    "median_of_means-30": {'color': '#8c564b', 'marker': '<', 'linestyle': ':', 'linewidth': 2} # C5
}

lrv_style_dict = {
    "lrv-0.1": {'color': '#1f77b4', 'marker': 'o', 'linestyle': ':', 'linewidth': 2}, # C0
    "lrv-0.5": {'color': '#ff7f0e', 'marker': 's', 'linestyle': ':', 'linewidth': 2}, # C1 
    "lrv-1": {'color': '#d62728', 'marker': 'p', 'linestyle': '-.', 'linewidth': 3} , # C3 (used in original),
    "lrv": {'color': '#d62728', 'marker': 'p', 'linestyle': '-.', 'linewidth': 3} , # C3 (used in original)
    "lrv-5": {'color': '#8c564b', 'marker': '<', 'linestyle': ':', 'linewidth': 2}, # C5
    "lrv-10": {'color': '#2ca02c', 'marker': 'd', 'linestyle': ':', 'linewidth': 2},  # C2
    "lrv-20": {'color': '#9467bd', 'marker': 'v', 'linestyle': ':', 'linewidth': 2}, # C4
    "lrv-50": {'color': '#e377c2', 'marker': 'h', 'linestyle': ':', 'linewidth': 2}, # C6
    'lrv_general': {'color': '#ff7f0e', 'marker': 'P', 'linestyle': '-.', 'linewidth': 2}
}

ev_hyperparameter_style_dict = {
    "ev_filtering_low_n-0.1": {'color': '#1f77b4', 'marker': 'o', 'linestyle': ':', 'linewidth': 2}, # C0
    "ev_filtering_low_n-0.5": {'color': '#ff7f0e', 'marker': 's', 'linestyle': ':', 'linewidth': 2}, # C1 
    "ev_filtering_low_n-1": {'color': '#2ca02c', 'marker': 'd', 'linestyle': ':', 'linewidth': 2}, # C2
    "ev_filtering_low_n-2.5": {'color': '#7f7f7f', 'marker': 'o', 'linestyle': ':', 'linewidth': 2}, # C7
    "ev_filtering_low_n-5": {'color': '#e377c2', 'marker': 'h', 'linestyle': '-.', 'linewidth': 4},  # C6 (used in original)
    "ev_filtering_low_n-10": {'color': '#9467bd', 'marker': 'v', 'linestyle': ':', 'linewidth': 2}, # C4
    "ev_filtering_low_n-20": {'color': '#d62728', 'marker': 'p', 'linestyle': ':', 'linewidth': 2},  # C3 
    "ev_filtering_low_n-50": {'color': '#8c564b', 'marker': '<', 'linestyle': ':', 'linewidth': 2}, # C5
}

lrv_sample_style_dict = {
    "lrv": {'color': '#d62728', 'marker': 'p', 'linestyle': '-.', 'linewidth': 3},
    "lrv_sample": {'color': '#e377c2', 'marker': 'h', 'linestyle': '-.', 'linewidth': 4},
    "lrv-1000": {'color': '#9467bd', 'marker': 'v', 'linestyle': ':', 'linewidth': 2},
    "lrv-100": {'color': '#9467bd', 'marker': 'v', 'linestyle': ':', 'linewidth': 2},
    "lrv-5": {'color': '#8c564b', 'marker': '<', 'linestyle': ':', 'linewidth': 2},
    "lrv-0.1": {'color': '#1f77b4', 'marker': 'o', 'linestyle': ':', 'linewidth': 2},
    'good_sample_mean': {'color': '#17becf', 'marker': '*', 'linestyle': '-', 'linewidth': 2}  # C9
}


