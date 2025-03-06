"""
Run synthetic experiments presented in paper
"""

# import corruption schemes, experiments, and estimators 
from synthetic_setup import identity_corruption_schemes, spherical_corruption_schemes, spherical_vary_std_schemes, meta_nonsp_corruption_schemes, nonsp_vary_std_schemes, uncorrupted_data_scheme, mix_schemes, id_gaussian_corruption_one_cluster, nongauss_corruption_schemes, heavy_tail_uncorr_scheme
from synthetic_setup import experiments, std_experiments, tau_experiments, uncorrupted_experiments, ev_que_experiment
from synthetic_setup import main_estimators, lee_valiant_comparison, median_of_means_options, lrv_options_C, lrv_options_general_gaus, ev_options, large_outlier_identity_corruption_schemes, grad_iterations, ev_que_original_vs_new, que_estimators, ev_pruning_comparison, eigenvalue_pruning_routines

# use experiment_suite to run experiments
from experiment_helper import experiment_suite

from synthetic_setup import old_experiments_short, uncorrupted_experiments_short

# 4.1 - Uncorrupted Identity Covariance

print("Uncorrupted Identity Covariance")
try:
    experiment_suite(main_estimators, uncorrupted_data_scheme, uncorrupted_experiments, runs=5, plot_good_sample=False, error_bars=True, save_title="uncorrupted", style_dict="main")
except Exception as e:
    print(f"Failed Uncorrupted Identity Covariance with Exception: {e}")
print()

# 4.2 - Identity Covariance Experiments (and extra for appendix)

print("Identity Battleground Experiments") 
for key, scheme in identity_corruption_schemes.items(): # 6 schemes
    print(f"Corruption Scheme: {key}")
    try:
        experiment_suite(main_estimators, scheme, experiments, runs=5, error_bars=True, plot_good_sample=True, plot=False, sample_scale_cov=False, save_title=f"id_cov-{key}", pickle_results=True, style_dict="main")
    except Exception as e:
        print(f"Failed Identity On Scheme {scheme}: {e}")
    print()

# 4.3 - Unknown Spherical Covariance Experiments (and extra for appendix)

print("Unknown Spherical Covariance Battleground Experiments") 
for key, scheme in spherical_corruption_schemes.items(): # 6 schemes
    print(f"Corruption Scheme: {key}")
    try:
        experiment_suite(main_estimators, scheme, experiments, runs=5, error_bars=True, plot_good_sample=True, save_title=f"large_sp-{key}", style_dict="main", sample_scale_cov=True, prune_obvious=False, pickle_results=True)
    except Exception as e:
        print(f"Failed Spherical On Scheme {scheme}: {e}")
    print()


print("Vary Standard Deviation Experiments")
for key, scheme in spherical_vary_std_schemes.items(): # 4 schemes
    print(f"Corruption Scheme: {key}")
    if key == "gaus_one":
        legend=True
    else:
        legend=False
    try: 
        experiment_suite(main_estimators, scheme, std_experiments, runs=5, error_bars=True, plot_good_sample=True, plot=False, legend=legend, save_title=f"sp_std_dependence-{key}", style_dict="main", sample_scale_cov=True, prune_obvious=False, pickle_results=True)
    except Exception as e:
        print(f"Failed Spherical STD On Scheme {scheme}: {e}")
    print()

# 4.4 - Unknown Non Sperhical Covariance Experiments: Diagonal

print("Unknown Non Spherical Covariance Battleground")
for diag_fun, corruption_schemes in meta_nonsp_corruption_schemes.items(): # 4 schemes
    print(f"New Diagonal Function: {diag_fun}")
    for key, scheme in corruption_schemes.items():
        print(f"Corruption Scheme: {key}") 
        try: 
            experiment_suite(main_estimators, scheme, experiments, runs=5, error_bars=True, plot_good_sample=True, style_dict="main", plot=False, save_title=f"large_non_sp-{diag_fun}_{key}", sample_scale_cov=True, prune_obvious=False, pickle_results=True)
        except Exception as e:
            print(f"Failed Non Spherical On Scheme {scheme} with exception: {e}")
        print()


print("Running Varying Top STD Experiment")
for key, scheme in nonsp_vary_std_schemes.items(): # 2 schemes
    print(f"Corruption Scheme: {key}")
    try: 
        if key == "gaus_one":
            legend=True
        else:
            legend=False
        experiment_suite(main_estimators, scheme, std_experiments, runs=5, error_bars=True, plot_good_sample=True, plot=False, style_dict="main", save_title=f"non_sp_std_dependence-{key}", sample_scale_cov=True, prune_obvious=False, legend=legend, pickle_results=True)
    except Exception as e:
        print(f"Failed Non Spherical STD with exception: {e}")
    print()

# 4.4 - Unknown Non Spherical Covariance Experiments: Non Diagonal
print("Rotated Experiments")
for key, scheme in mix_schemes.items():
    print(f"Corruption Scheme: {key}")
    try:
        if key.startswith("id"):
            experiment_suite(main_estimators, scheme, experiments, runs=5, error_bars=True, plot_good_sample=True, plot=False, save_title=f"rotate-{key}", style_dict="main", sample_scale_cov=False, rotate=True,pickle_results=True)
        else:
            experiment_suite(main_estimators, scheme, experiments, runs=5, error_bars=True, plot_good_sample=True, plot=False, save_title=f"rotate-{key}", style_dict="main", sample_scale_cov=True, rotate=True,pickle_results=True)
    except Exception as e:
        print(f"Failed Rotated On Scheme {scheme}: {e}")
    print()
    
# 6 - Non Gaussian Synthetic Experiments

print("Non Gaussian Synthetic Experiments") 
for key, scheme in nongauss_corruption_schemes.items(): # 6 schemes
    print(f"Corruption Scheme: {key}")
    try:
        experiment_suite(main_estimators, scheme, experiments, runs=3, error_bars=True, plot_good_sample=True, plot=False, sample_scale_cov=True, save_title=f"{key}", pickle_results=True, style_dict="main")
    except Exception as e:
        print(f"Failed Non-Gauss On Scheme {scheme}: {e}")
    print()

# 7 - Lee Valiant Comparison (Eigenvalue Pruning In Seperate File)

print("Lee Valiant Comparison")
try:
    experiment_suite(lee_valiant_comparison, id_gaussian_corruption_one_cluster, experiments, runs=5, error_bars=True, plot_good_sample=True, style_dict="lee_valiant", plot=False, save_title=f"lee_valiant_comparison_gaus_one", sample_scale_cov=False)
except Exception as e:
    print(f"Failed Lee Valiant Comparison with exception: {e}")
print()

# 7 - Original vs Updated EV + QUE
print("Original Vs Updated EV + QUE Experiment")
experiment_suite(ev_que_original_vs_new, identity_corruption_schemes["dkk"], ev_que_experiment, runs=5, plot_good_sample=False, error_bars=True, save_title="ev_que_comparison", pickle_results=True, style_dict="main")


# Identity Covariance Large Outlier Mixes without Pruning
# Prune outliers here - currently this is done in a way that assumes identity covariance, but should be adaptable by
print("Large Outlier - Identity Covariance")
for key, scheme in large_outlier_identity_corruption_schemes.items():
    print(f"Corruption Scheme: {key}")
    try:
        experiment_suite(main_estimators, scheme, experiments, runs=5, error_bars=True, plot_good_sample=True, plot=False, sample_scale_cov=False, save_title=f"id_cov-large-{key}", style_dict="main", prune_obvious=False, pickle_results=True)
    except Exception as e:
        print(f"Failed Identity Large Outliers With Pruning On Scheme {scheme}: {e}")
    print()

# Identity Covariance Dependence On True Mean
print("Dependence On True Mean - Identity Covariance")
for key, scheme in identity_corruption_schemes.items():
    if key == "gaus_one" or key == "dkk":
        print(f"Corruption Scheme: {key}")
        try:
            experiment_suite(main_estimators, scheme, experiments, runs=5, error_bars=True, plot_good_sample=True, plot=False, sample_scale_cov=False, save_title=f"mean_dependence-{key}", style_dict="main")
        except Exception as e:
            print(f"Failed Identity Dependence On True Mean On Scheme {scheme}: {e}")
        print()


# Robustness To Expected Corruption 
print("Robustness To Expected Corruption")
for key, scheme in mix_schemes.items():
    print(f"Corruption Scheme: {key}")
    try:
        if key == "id_dkk":
            legend = True
        else:
            legend = False
        if key.startswith("id"):
            experiment_suite(main_estimators, scheme, tau_experiments, runs=5, error_bars=True, plot_good_sample=True, plot=False, save_title=f"expected_corruption_robustness-{key}", style_dict="main", sample_scale_cov=False, legend=legend, pickle_results=True)  
        else:
            experiment_suite(main_estimators, scheme, tau_experiments, runs=5, error_bars=True, plot_good_sample=True, plot=False, save_title=f"expected_corruption_robustness-{key}", style_dict="main", sample_scale_cov=True, legend=legend, pickle_results=True)
    except Exception as e:
        print(f"Failed Robustness To Expected Corruption On Scheme {scheme}: {e}")
    print()


# Hyperparameter Tuning

# Median Of How Many Means?  
print("Median Of Means Hyperparameter Tuning")
for key, scheme in mix_schemes.items():
    print(f"Corruption Scheme: {key}")
    if key != "id_unif_top":
        continue
    try:
        experiment_suite(median_of_means_options, scheme, experiments, runs=5, error_bars=True, plot_good_sample=False, plot=False, save_title=f"med_mean_tuning/{key}", style_dict="med_mean")
    except Exception as e:
        print(f"Failed Median Of Means Hyperparameter Tuning On  Scheme {scheme}: {e}")
    print()

# LRV Weighting Procedure - Weight and Vs General
# Maybe even redo general for LLM Experiment
print("LRV Hyperparameter Tuning - C")
for key, scheme in mix_schemes.items():
    print(f"Corruption Scheme: {key}")
    try:
        experiment_suite(lrv_options_C, scheme, experiments, runs=5, error_bars=True, plot_good_sample=False, plot=False, save_title=f"lrv_tuning/{key}", style_dict="lrv")
    except Exception as e:
        print(f"Failed LRV Hyperparameter Tuning On Scheme {scheme}: {e}")
    print()

print("LRV Hyperparameter Tuning - General vs Gaus")
for key, scheme in mix_schemes.items():
    print(f"Corruption Scheme: {key}")
    try:
        experiment_suite(lrv_options_general_gaus, scheme, experiments, runs=5, error_bars=True, plot_good_sample=False, plot=False, save_title=f"lrv_tuning_{key}", style_dict="lrv")
    except Exception as e:
        print(f"Failed LRV Hyperparameter Tuning On Scheme {scheme}: {e}")
    print()


# Eigenvalue Hyperparameter Tuning 
print("Eigenvalue Hyperparameter Tuning")
for key, scheme in mix_schemes.items():
    print(f"Corruption Scheme: {key}")
    try:
        if key == "sp_gaus_one" or key == "nonsp_gaus_one":
            experiment_suite(ev_options, scheme, experiments, runs=5, error_bars=True, plot_good_sample=False, plot=False, save_title=f"ev_tuning/{key}", style_dict="ev", sample_scale_cov=True)
        else:
            experiment_suite(ev_options, scheme, experiments, runs=5, error_bars=True, plot_good_sample=False, plot=False, save_title=f"ev_tuning/{key}", style_dict="ev", sample_scale_cov=False)
    except Exception as e:
        print(f"Failed Eigenvalue Hyperparameter Tuning On Scheme {scheme}: {e}")
    print()

#Eigenvalue Hyperparameter Tuning - different pruning routines
print("Eigenvalue Hyperparameter Tuning")
for key, scheme in mix_schemes.items():
    print(f"Corruption Scheme: {key}")
    if key != "id_unif_top":
        continue
    try:
        if key == "sp_gaus_one" or key == "nonsp_gaus_one":
            experiment_suite(eigenvalue_pruning_routines, scheme, experiments, runs=5, error_bars=True, plot_good_sample=False, plot=False, save_title=f"ev_tuning/{key}", style_dict="ev_pruning", sample_scale_cov=True)
        else:
            experiment_suite(eigenvalue_pruning_routines, scheme, experiments, runs=5, error_bars=True, plot_good_sample=False, plot=False, save_title=f"ev_tuning/{key}", style_dict="ev_pruning", sample_scale_cov=False)
    except Exception as e:
        print(f"Failed Eigenvalue Hyperparameter Tuning On Scheme {scheme}: {e}")
    print()

print("PGD Options")
for key, scheme in mix_schemes.items():
    print(f"Corruption Scheme: {key}")
    if key != "id_unif_top":
        continue
    try:
        experiment_suite(grad_iterations, scheme, experiments, runs=5, error_bars=True, plot_good_sample=False, plot=False, save_title=f"pgd_tuning-{key}", style_dict="pgd", pickle_results=True)
    except Exception as e:
        print(f"Failed PGD Hyperparameter Tuning On Scheme {scheme}: {e}")
    print()

print("EV Pruning Comparison")
for key, scheme in mix_schemes.items():
    print(f"Corruption Scheme: {key}")
    try:
        if key.startswith("id"):
            experiment_suite(ev_pruning_comparison, scheme, experiments, runs=5, error_bars=True, plot_good_sample=False, plot=False, save_title=f"ev_pruning-{key}", style_dict="ev_pruning", pickle_results=False)
        else:
            experiment_suite(ev_pruning_comparison, scheme, experiments, runs=5, error_bars=True, plot_good_sample=False, plot=False, save_title=f"ev_pruning-{key}", style_dict="ev_pruning", pickle_results=False, sample_scale_cov=True)
    except Exception as e:
        print(f"Failed PGD Hyperparameter Tuning On Scheme {scheme}: {e}")
    print()

print("QUE Alpha Comparison")
for key, scheme in mix_schemes.items():
    print(f"Corruption Scheme: {key}")
    try:
        if key.startswith("id"):
            experiment_suite(que_estimators, scheme, experiments, runs=5, error_bars=True, plot_good_sample=False, plot=False, save_title=f"que_tuning-{key}", style_dict="que", pickle_results=True)
        else:
            experiment_suite(que_estimators, scheme, experiments, runs=5, error_bars=True, plot_good_sample=False, plot=False, save_title=f"que_tuning-{key}", style_dict="que", pickle_results=True, sample_scale_cov=True)
    except Exception as e:
        print(f"Failed PGD Hyperparameter Tuning On Scheme {scheme}: {e}")
    print()