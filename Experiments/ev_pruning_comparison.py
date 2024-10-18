"""
Plot threshold comparisons
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from synthetic_setup import gaussian_data

d = 500
tau = 0.1
mean_fun = lambda d: np.ones(d) * 5
cov_fun = lambda d: np.eye(d)

og_threshold_fun = lambda n, d, tau: 1 + 3 * tau * math.log(1 / tau)

t = 10
new_threshold_fun = lambda n, d, tau:  (1+ math.sqrt(d/n) + t/math.sqrt(n)) ** 2

def calculate_top_eigenvalue(data):
    n, d = data.shape
    empirical_mean = np.mean(data, axis=0)
    centered_data = (data-empirical_mean) / math.sqrt(n)
    U, S, Vt = np.linalg.svd(centered_data, full_matrices=False)
    return S[0] ** 2

runs = 5

# goal is to plot gaussian identity threshold vs eigenvalues
varying_range = np.arange(20, 10021, 500)
true_eigenvalues_runs = np.zeros((runs, len(varying_range)))
og_threshold_values = np.zeros(len(varying_range))
new_threshold_values = np.zeros(len(varying_range))

for run in range(runs):
    for i, n in enumerate(varying_range):
        if run == 0:
            og_threshold_values[i] = og_threshold_fun(n, d, tau)
            new_threshold_values[i] = new_threshold_fun(n, d, tau)

        data, _, _ = gaussian_data(n, d, tau, mean_fun=mean_fun, cov_fun=cov_fun)
        true_eigenvalues_runs[run][i] = calculate_top_eigenvalue(data)
    
true_eigenvalues = np.mean(true_eigenvalues_runs, axis=0)
true_eigenvalues_std = np.std(true_eigenvalues_runs, axis=0)

# Assuming `varying_range`, `true_eigenvalues`, `og_threshold_values`, and `new_threshold_values` are already defined
# Calculate where true_eigenvalues become less than og_threshold_values
for i, (true_val, og_val) in enumerate(zip(true_eigenvalues, og_threshold_values)):
    if true_val < og_val:
        break

style_dict = {
    'new_threshold': {'color': '#e377c2', 'marker': 'h', 'linestyle': '-.', 'linewidth': 3},  # C6
    'og_threshold': {'color': '#2ca02c', 'marker': '>', 'linestyle': ':', 'linewidth': 3},  # C6
    'true_eigenvalue': {'color': '#17becf', 'marker': '*', 'linestyle': '-', 'linewidth': 2}  # C9
}

# Plot the values with error bars filled
og_threshold_style = style_dict['og_threshold']
new_threshold_style = style_dict['new_threshold']
true_eigenvalue_style = style_dict['true_eigenvalue']

# plot true eigenvalues and stds
plt.plot(varying_range, true_eigenvalues, label='True Eigenvalues', color=true_eigenvalue_style['color'], marker=true_eigenvalue_style['marker'], linestyle=true_eigenvalue_style['linestyle'], 
                        linewidth=true_eigenvalue_style['linewidth'], alpha=0.8)

plt.fill_between(varying_range, 
                 true_eigenvalues - true_eigenvalues_std, 
                 true_eigenvalues + true_eigenvalues_std, color = true_eigenvalue_style['color'],
                 alpha=0.3)

# plot original threshold values
plt.plot(varying_range, og_threshold_values, color=og_threshold_style['color'], marker=og_threshold_style['marker'], linestyle=og_threshold_style['linestyle'], 
                        linewidth=og_threshold_style['linewidth'], label='Original Threshold Values')

# plot our threshold values
plt.plot(varying_range, new_threshold_values, color=new_threshold_style['color'], marker=new_threshold_style['marker'], linestyle=new_threshold_style['linestyle'], 
                        linewidth=new_threshold_style['linewidth'], label='New Threshold Values')

# plot the vertical line at the identified point
plt.axvline(x=varying_range[i], color='#d62728', linestyle='--', label='True Eigenvalue < Naive Threshold')

plt.legend()
plt.title("True Eigenvalue vs Data Size\n Dimensions: 500, Expected Corruption: 0.1")

plt.xlabel("Data Size")

plt.savefig(f"threshold_vs_eigenvalue")
