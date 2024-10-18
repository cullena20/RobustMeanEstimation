# this is a place to run most experiments

from setup import identity_corruption_schemes, spherical_corruption_schemes, spherical_vary_std_schemes, meta_nonsp_corruption_schemes, nonsp_vary_std_schemes, uncorrupted_data_scheme, mix_schemes, identity_true_mean_dependence_schemes, id_dkk_corruption, id_gaussian_corruption_one_cluster
from setup import experiments, experiments_short, std_experiments, std_experiments_short, old_experiments_short, old_experiments, tau_experiments, ev_comparison_experiment, uncorrupted_experiments, experiments_time, time_experiments, ev_que_experiment
from setup import main_estimators, lee_valiant_comparison, median_of_means_options, lrv_options_C, lrv_options_general_gaus, ev_options, large_outlier_identity_corruption_schemes, eigenvalue_old_vs_new, lrv_options3, grad_iterations, ev_que_original_vs_new, que_estimators, ev_pruning_comparison, eigenvalue_pruning_routines
from helper import experiment_suite

from setup import sdp

# random little experiments here

from setup import eigenvalue_t, grad_iterations, ev_que

# Problem: eigenvalue still breaks down under low data size - does changing t help? (we might want to explore this as a hyperparamter too)
# RESULT - Changing t solves issue -> fix in appropriate graphs (this was also an issue with que but somehow it is less sensitive)

#experiment_suite(eigenvalue_t, identity_corruption_schemes["unif_top"], experiments_short, runs=2, plot_good_sample=True, error_bars=True, save_title="Test/eigenvalue_t-id_unif_top", pickle_results=False)

# Problem: grad is super slow, can we reduce nItr? may also be worth exploring as full hyperparameter - I think worth exploring
# also see if different nItr may improve dkk or uniftop (it degrades worse with n and eta)
# RESULT: Reducing nItr results in worse performance (although halving it is indeed nearly a 2x speedup)
# nItr=15 solves the above issue with degredation

# experiment_suite(grad_iterations, identity_corruption_schemes["gaus_one"], old_experiments_short, runs=2, plot_good_sample=True, error_bars=True, save_title="Test/grad_nItr-id_gaus_one", pickle_results=False)
#experiment_suite(grad_iterations, identity_corruption_schemes["gaus_one"], old_experiments_short, option="time", runs=2, plot_good_sample=False, error_bars=True, save_title="Test/grad_nItr-time-id_gaus_one", pickle_results=False)
#experiment_suite(grad_iterations, identity_corruption_schemes["unif_top"], old_experiments_short, runs=2, plot_good_sample=True, error_bars=True, save_title="Test/grad_nItr-id_unif_top", pickle_results=False)

# Problem: Our sample trace estimate heuristic seems to break down - 
# notably large_sp_dkk fails badly for ev, and que deteriorates too - run these again for ev and que with different trace methods
# RESULT: LRV trace does return a lower trace as previously described, so use sample trace
# For the writeup we might want to give specific numbers

#experiment_suite(ev_que, spherical_corruption_schemes["dkk"], old_experiments_short, runs=2, error_bars=True, plot_good_sample=True, save_title=f"Test/sample_scale-large_sp_dkk", pickle_results=False, sample_scale_cov=True, prune_obvious=False)

# # MAIN PAPER EXPERIMENTS
#experiment_suite(main_estimators, uncorrupted_data_scheme, experiments, runs=5, plot_good_sample=False, error_bars=True, save_title="NEWuncorrupted", pickle_results=True, style_dict="main")

# rerun this to fix eigenvalue
#experiment_suite(main_estimators, identity_corruption_schemes["gaus_one"], experiments, runs=5, plot_good_sample=True, error_bars=True, save_title="NO_PGD-id_gaus_one", pickle_results=True, style_dict="main")

# rerun this to fix scaling stuff
#experiment_suite(main_estimators, spherical_corruption_schemes["gaus_one"], experiments, runs=5, error_bars=True, plot_good_sample=True, save_title=f"NO_PGD-large_sp_gaus_one", style_dict="main", pickle_results=True, sample_scale_cov=True, prune_obvious=False)

# rerun to fix pgd performance
#experiment_suite(main_estimators, identity_corruption_schemes["dkk"], experiments, runs=5, plot_good_sample=True, error_bars=True, save_title="id_dkk", pickle_results=True, style_dict="main")
#experiment_suite(main_estimators, identity_corruption_schemes["unif_top"], experiments, runs=5, plot_good_sample=True, error_bars=True, save_title="id_unif_top", pickle_results=True, style_dict="main")

# experiment_suite(main_estimators, identity_corruption_schemes["subtractive_corruption"], experiments, runs=5, plot_good_sample=True, error_bars=True, save_title="id_sub", pickle_results=True, style_dict="main")

# # Extra Basics - Large Dim Diag, Time Analysis
# experiment_suite(main_estimators, meta_nonsp_corruption_schemes["large_dim_diag"]["gaus_one"], experiments, runs=5, error_bars=True, plot_good_sample=True, save_title=f"large_dim_diag_gaus_one", style_dict="main", pickle_results=True, sample_scale_cov=True, prune_obvious=False)
# experiment_suite(main_estimators, identity_corruption_schemes["gaus_one"], experiments, option="time", runs=5, plot_good_sample=False, error_bars=True, save_title="time-id_gaus_one", pickle_results=True, style_dict="main")

# 4.1 - Uncorrupted Identity Covariance

# print("Uncorrupted Identity Covariance")
# try:
#     experiment_suite(main_estimators, uncorrupted_data_scheme, uncorrupted_experiments, runs=10, plot_good_sample=False, error_bars=True, save_title="uncorrupted", style_dict="main")
#     # experiment_suite(main_estimators, uncorrupted_data_scheme, experiments_short, runs=3, plot_good_sample=False, error_bars=True, save_title="uncorrupted", style_dict="main")
# except Exception as e:
#     print(f"Failed Uncorrupted Identity Covariance with Exception: {e}")
# print()

# 4.2 - Identity Covariance Experiments

# print("Identity Battleground Experiments") # - seems to work
# for key, scheme in identity_corruption_schemes.items(): # 6 schemes
#     print(f"Corruption Scheme: {key}")
#     if key != "gaus_two" and key != "obvious":
#         pass
#     else:
#         try:
#             experiment_suite(main_estimators, scheme, experiments, runs=5, error_bars=True, plot_good_sample=True, plot=False, sample_scale_cov=False, save_title=f"NEWid_cov-{key}", pickle_results=True, style_dict="main")
#             # experiment_suite(main_estimators, scheme, experiments_short, runs=3, error_bars=True, plot_good_sample=True, plot=False, sample_scale_cov=False, save_title=f"id_cov/{key}", style_dict="main")
#         except Exception as e:
#             # this exception can get messed up when scheme is replaced by another variable
#             print(f"Failed Identity On Scheme {scheme}: {e}")
#     print()

# FOR QUE
# print("Identity Battleground Experiments") # - seems to work
# for key, scheme in identity_corruption_schemes.items(): # 6 schemes
#     print(f"Corruption Scheme: {key}")
#     try:
#         experiment_suite(que_estimators, scheme, experiments_short, runs=2, error_bars=True, plot_good_sample=True, plot=False, sample_scale_cov=False, save_title=f"Test/que_test_{key}", pickle_results=False)
#         # experiment_suite(main_estimators, scheme, experiments_short, runs=3, error_bars=True, plot_good_sample=True, plot=False, sample_scale_cov=False, save_title=f"id_cov/{key}", style_dict="main")
#     except Exception as e:
#         # this exception can get messed up when scheme is replaced by another variable
#         print(f"Failed Identity On Scheme {scheme}: {e}")
#     print()


# 4.3 - Unknown Spherical Covariance Experiments

# pgd for gaus_two, 

# print("Unknown Spherical Covariance Battleground Experiments") # - seems to work, but now ev doesn't work as well, not sure what happened
# for key, scheme in spherical_corruption_schemes.items(): # 6 schemes
#     print(f"Corruption Scheme: {key}")
#     if key!="gaus_one" and key!="gaus_two" and key!="dkk": # unif_top, sub (omitting obvious for now)
#         pass
#     else:
#         try:
#             experiment_suite(main_estimators, scheme, experiments, runs=5, error_bars=True, plot_good_sample=True, save_title=f"NEWlarge_sp-{key}", style_dict="main", sample_scale_cov=True, prune_obvious=False, pickle_results=True)
#             #experiment_suite(main_estimators, scheme, experiments_short, runs=2, error_bars=True, plot_good_sample=True, save_title=f"obvious/large_sp_{key}", style_dict="main", sample_scale_cov=True, prune_obvious=False, plot=True)
#         except Exception as e:
#             print(f"Failed Spherical On Scheme {scheme}: {e}")
#     print()

# FOR QUE
# print("Unknown Spherical Covariance Battleground Experiments") # - seems to work, but now ev doesn't work as well, not sure what happened
# for key, scheme in spherical_corruption_schemes.items(): # 6 schemes
#     print(f"Corruption Scheme: {key}")
#     try:
#         experiment_suite(que_estimators, scheme, experiments_short, runs=2, error_bars=True, plot_good_sample=True, save_title=f"Test/que_test_largesp_{key}", sample_scale_cov=True, prune_obvious=False, pickle_results=False)
#         #experiment_suite(main_estimators, scheme, experiments_short, runs=2, error_bars=True, plot_good_sample=True, save_title=f"obvious/large_sp_{key}", style_dict="main", sample_scale_cov=True, prune_obvious=False, plot=True)
#     except Exception as e:
#         print(f"Failed Spherical On Scheme {scheme}: {e}")
#     print()

# omiting obvious for now
# print("Vary Standard Deviation Experiments")
# for key, scheme in spherical_vary_std_schemes.items(): # 4 schemes
#     print(f"Corruption Scheme: {key}")
#     if key == "gaus_one":
#         legend=True
#     else:
#         legend=False
#     try: 
#         experiment_suite(main_estimators, scheme, std_experiments, runs=3, error_bars=True, plot_good_sample=True, plot=False, legend=legend, save_title=f"sp_std_dependence-{key}", style_dict="main", sample_scale_cov=True, prune_obvious=False, pickle_results=True)
#         #experiment_suite(main_estimators, scheme, std_experiments_short, runs=3, error_bars=True, plot_good_sample=True, plot=False, legend=legend, save_title=f"obvious/sp_std_{key}", style_dict="main", sample_scale_cov=True, prune_obvious=False)
#     except Exception as e:
#         print(f"Failed Spherical STD On Scheme {scheme}: {e}")
#     print()

# 4.4 - Unknown Non Sperhical Covariance Experiments: Diagonal

# print("Unknown Non Spherical Covariance Battleground")
# for diag_fun, corruption_schemes in meta_nonsp_corruption_schemes.items(): # 4 schemes
#     print(f"New Diagonal Function: {diag_fun}")
#     for key, scheme in corruption_schemes.items():
#         print(f"Corruption Scheme: {key}") 
#         if key != "unif_top" and key != "subtractive_corruption":
#             continue
#         try: 
#             experiment_suite(main_estimators, scheme, experiments, runs=3, error_bars=True, plot_good_sample=True, style_dict="main", plot=False, save_title=f"large_non_sp-{diag_fun}_{key}", sample_scale_cov=True, prune_obvious=False, pickle_results=True)
#             #experiment_suite(main_estimators, scheme, experiments_short, runs=2, error_bars=True, plot_good_sample=True, style_dict="main", save_title=f"obvious/non_sp_{key}", sample_scale_cov=True, prune_obvious=False, plot=True)
#         except Exception as e:
#             print(f"Failed Non Spherical On Scheme {scheme} with exception: {e}")
#         print()

# QUE TEST
# print("Unknown Non Spherical Covariance Battleground")
# for diag_fun, corruption_schemes in meta_nonsp_corruption_schemes.items(): # 4 schemes
#     print(f"New Diagonal Function: {diag_fun}")
#     for key, scheme in corruption_schemes.items():
#         print(f"Corruption Scheme: {key}") 
#         try: 
#             experiment_suite(que_estimators, scheme, experiments, runs=2, error_bars=True, plot_good_sample=True, plot=False, save_title=f"Test/que_test_nonsp_{key}", sample_scale_cov=True, prune_obvious=False, pickle_results=False)
#             #experiment_suite(main_estimators, scheme, experiments_short, runs=2, error_bars=True, plot_good_sample=True, style_dict="main", save_title=f"obvious/non_sp_{key}", sample_scale_cov=True, prune_obvious=False, plot=True)
#         except Exception as e:
#             print(f"Failed Non Spherical On Scheme {scheme} with exception: {e}")
#         print()

# EV Test
# print("Unknown Non Spherical Covariance Battleground")
# for diag_fun, corruption_schemes in meta_nonsp_corruption_schemes.items(): # 4 schemes
#     print(f"New Diagonal Function: {diag_fun}")
#     for key, scheme in corruption_schemes.items():
#         print(f"Corruption Scheme: {key}") 
#         try: 
#             experiment_suite(ev_pruning_comparison, scheme, experiments, runs=2, error_bars=True, plot_good_sample=True, plot=False, save_title=f"Test/ev_pruning_{key}", sample_scale_cov=True, prune_obvious=False, pickle_results=False)
#             #experiment_suite(main_estimators, scheme, experiments_short, runs=2, error_bars=True, plot_good_sample=True, style_dict="main", save_title=f"obvious/non_sp_{key}", sample_scale_cov=True, prune_obvious=False, plot=True)
#         except Exception as e:
#             print(f"Failed Non Spherical On Scheme {scheme} with exception: {e}")
#         print()


# print("Running Varying Top STD Experiment")
# for key, scheme in nonsp_vary_std_schemes.items(): # 2 schemes
#     print(f"Corruption Scheme: {key}")
#     try: 
#         if key == "gaus_one":
#             legend=True
#         else:
#             legend=False
#         experiment_suite(main_estimators, scheme, std_experiments, runs=3, error_bars=True, plot_good_sample=True, plot=False, style_dict="main", save_title=f"non_sp_std_dependence-{key}", sample_scale_cov=True, prune_obvious=False, legend=legend, pickle_results=True)
#         #experiment_suite(main_estimators, scheme, std_experiments_short, runs=3, error_bars=True, plot_good_sample=True, plot=False, style_dict="main", save_title=f"obvious/non_sp_std_{key}", sample_scale_cov=True, prune_obvious=False)
#     except Exception as e:
#         print(f"Failed Non Spherical STD with exception: {e}")
#     print()
        
# 6 - Lee Valiant Comparison (Eigenvalue Pruning In Seperate File)

# print("Lee Valiant Comparison")
# try:
#     experiment_suite(lee_valiant_comparison, id_gaussian_corruption_one_cluster, experiments, runs=10, error_bars=True, plot_good_sample=True, style_dict="lee_valiant", plot=False, save_title=f"lee_valiant_comparison_gaus_one", sample_scale_cov=False)
#     # experiment_suite(lee_valiant_comparison, id_gaussian_corruption_one_cluster, experiments_short, runs=3, error_bars=True, plot_good_sample=True, style_dict="lee_valiant", plot=False, save_title=f"lee_valiant_comparison_gaus_one", sample_scale_cov=False)
# except Exception as e:
#     print(f"Failed Lee Valiant Comparison with exception: {e}")
# print()

# Hyperparameter Tuning

# Median Of How Many Means? - currently k = 10 
# print("Median Of Means Hyperparameter Tuning")
# for key, scheme in mix_schemes.items():
#     print(f"Corruption Scheme: {key}")
#     if key != "id_unif_top":
#         continue
#     try:
#         experiment_suite(median_of_means_options, scheme, experiments, runs=5, error_bars=True, plot_good_sample=False, plot=False, save_title=f"med_mean_tuning/{key}", style_dict="med_mean")
#         # experiment_suite(median_of_means_options, scheme, experiments_short, runs=3, error_bars=True, plot_good_sample=False, plot=False, save_title=f"med_mean_tuning/{key}", style_dict="med_mean")
#     except Exception as e:
#         print(f"Failed Median Of Means Hyperparameter Tuning On  Scheme {scheme}: {e}")
#     print()

# LRV Weighting Procedure - Weight and Vs General - currently C = 1
# Maybe even redo general for LLM Experiment
# print("LRV Hyperparameter Tuning - C")
# for key, scheme in mix_schemes.items():
#     print(f"Corruption Scheme: {key}")
#     try:
#         experiment_suite(lrv_options_C, scheme, experiments, runs=10, error_bars=True, plot_good_sample=False, plot=False, save_title=f"lrv_tuning/{key}", style_dict="lrv")
#         # experiment_suite(lrv_options_C, scheme, experiments_short, runs=3, error_bars=True, plot_good_sample=False, plot=False, save_title=f"lrv_tuning/{key}", style_dict="lrv")
#     except Exception as e:
#         print(f"Failed LRV Hyperparameter Tuning On Scheme {scheme}: {e}")
#     print()

# print("LRV Hyperparameter Tuning - General vs Gaus")
# for key, scheme in mix_schemes.items():
#     print(f"Corruption Scheme: {key}")
#     try:
#         experiment_suite(lrv_options_general_gaus, scheme, experiments, runs=5, error_bars=True, plot_good_sample=False, plot=False, save_title=f"lrv_tuning_{key}", style_dict="lrv")
#         # experiment_suite(lrv_options_C,s cheme, experiments_short, runs=3, error_bars=True, plot_good_sample=False, plot=False, save_title=f"lrv_tuning/{key}", style_dict="lrv")
#     except Exception as e:
#         print(f"Failed LRV Hyperparameter Tuning On Scheme {scheme}: {e}")
#     print()


# Eigenvalue Hyperparameter Tuning - currently cher = 5
# print("Eigenvalue Hyperparameter Tuning")
# for key, scheme in mix_schemes.items():
#     print(f"Corruption Scheme: {key}")
#     try:
#         if key == "sp_gaus_one" or key == "nonsp_gaus_one":
#             print("scale sample cov")
#             experiment_suite(ev_options, scheme, experiments, runs=10, error_bars=True, plot_good_sample=False, plot=False, save_title=f"ev_tuning/{key}", style_dict="ev", sample_scale_cov=True)
#             # experiment_suite(ev_options, scheme, experiments_short, runs=3, error_bars=True, plot_good_sample=False, plot=False, save_title=f"ev_tuning/{key}", style_dict="ev", sample_scale_cov=True)
#         else:
#             experiment_suite(ev_options, scheme, experiments, runs=10, error_bars=True, plot_good_sample=False, plot=False, save_title=f"ev_tuning/{key}", style_dict="ev", sample_scale_cov=False)
#             # experiment_suite(ev_options, scheme, experiments_short, runs=3, error_bars=True, plot_good_sample=False, plot=False, save_title=f"ev_tuning/{key}", style_dict="ev", sample_scale_cov=False)
#     except Exception as e:
#         print(f"Failed Eigenvalue Hyperparameter Tuning On Scheme {scheme}: {e}")
#     print()

#Eigenvalue Hyperparameter Tuning - different pruning routines
# print("Eigenvalue Hyperparameter Tuning")
# for key, scheme in mix_schemes.items():
#     print(f"Corruption Scheme: {key}")
#     if key != "id_unif_top":
#         continue
#     try:
#         if key == "sp_gaus_one" or key == "nonsp_gaus_one":
#             experiment_suite(eigenvalue_pruning_routines, scheme, experiments, runs=5, error_bars=True, plot_good_sample=False, plot=False, save_title=f"ev_tuning/{key}", style_dict="ev_pruning", sample_scale_cov=True)
#             # experiment_suite(ev_options, scheme, experiments_short, runs=3, error_bars=True, plot_good_sample=False, plot=False, save_title=f"ev_tuning/{key}", style_dict="ev", sample_scale_cov=True)
#         else:
#             experiment_suite(eigenvalue_pruning_routines, scheme, experiments, runs=5, error_bars=True, plot_good_sample=False, plot=False, save_title=f"ev_tuning/{key}", style_dict="ev_pruning", sample_scale_cov=False)
#             # experiment_suite(ev_options, scheme, experiments_short, runs=3, error_bars=True, plot_good_sample=False, plot=False, save_title=f"ev_tuning/{key}", style_dict="ev", sample_scale_cov=False)
#     except Exception as e:
#         print(f"Failed Eigenvalue Hyperparameter Tuning On Scheme {scheme}: {e}")
#     print()

# Identity Covariance Large Outlier Mixes with Pruning
# Prune outliers here - currently this is done in a way that assumes identity covariance, but should be adaptable by
# print("Large Outlier With Pruning - Identity Covariance")
# for key, scheme in large_outlier_identity_corruption_schemes.items():
#     print(f"Corruption Scheme: {key}")
#     try:
#         # experiment_suite(main_estimators, scheme, experiments, runs=10, error_bars=True, plot_good_sample=True, plot=False, sample_scale_cov=False, save_title=f"id_cov/pruned-{key}", style_dict="main", prune_obvious=True)
#         experiment_suite(main_estimators, scheme, experiments_short, runs=3, error_bars=True, plot_good_sample=True, plot=True, sample_scale_cov=False, prune_obvious=True, save_title=f"id_cov/fixed_pruned-{key}", style_dict="main")
#     except Exception as e:
#         print(f"Failed Identity Large Outliers With Pruning On Scheme {scheme}: {e}")
#     print()

# Identity Covariance Large Outlier Mixes without Pruning
# Prune outliers here - currently this is done in a way that assumes identity covariance, but should be adaptable by
# print("Large Outlier - Identity Covariance")
# for key, scheme in large_outlier_identity_corruption_schemes.items():
#     print(f"Corruption Scheme: {key}")
#     try:
#         experiment_suite(main_estimators, scheme, experiments, runs=3, error_bars=True, plot_good_sample=True, plot=False, sample_scale_cov=False, save_title=f"NEWid_cov-large-{key}", style_dict="main", prune_obvious=False, pickle_results=True)
#         #experiment_suite(main_estimators, scheme, experiments_short, runs=3, error_bars=True, plot_good_sample=True, plot=True, sample_scale_cov=False, prune_obvious=True, save_title=f"obvious/id_{key}", style_dict="main")
#     except Exception as e:
#         print(f"Failed Identity Large Outliers With Pruning On Scheme {scheme}: {e}")
#     print()

# Might want to extend above for non identity too

# Identity Covariance Dependence On True Mean - Just shuffle mean with every iteration
# Only examine on DKK and Gaus One
# print("Dependence On True Mean - Identity Covariance")
# for key, scheme in identity_corruption_schemes.items():
#     if key == "gaus_one" or key == "dkk":
#         print(f"Corruption Scheme: {key}")
#         try:
#             experiment_suite(main_estimators, scheme, experiments, runs=3, error_bars=True, plot_good_sample=True, plot=False, sample_scale_cov=False, save_title=f"mean_dependence-{key}", style_dict="main")
#             # experiment_suite(main_estimators, scheme, experiments_short, runs=3, error_bars=True, plot_good_sample=True, plot=False, sample_scale_cov=False, save_title=f"id_cov/mean_dependence-{key}", style_dict="main")
#         except Exception as e:
#             print(f"Failed Identity Dependence On True Mean On Scheme {scheme}: {e}")
#         print()

# Robustness To Expected Corruption - probably just use same schemes as hyperparameter tuning
# If we do show this prune obvious for spherical then we should do it here too
print("Robustness To Expected Corruption")
for key, scheme in mix_schemes.items():
    print(f"Corruption Scheme: {key}")
    if key == "id_unif_top":
        continue
    try:
        if key == "id_dkk":
            legend = True
        else:
            legend = False
        if key.startswith("id"):
            experiment_suite(main_estimators, scheme, tau_experiments, runs=5, error_bars=True, plot_good_sample=True, plot=False, save_title=f"NEWexpected_corruption_robustness-{key}", style_dict="main", sample_scale_cov=False, legend=legend, pickle_results=True)  
        else:
            experiment_suite(main_estimators, scheme, tau_experiments, runs=5, error_bars=True, plot_good_sample=True, plot=False, save_title=f"NEWexpected_corruption_robustness-{key}", style_dict="main", sample_scale_cov=True, legend=legend, pickle_results=True)
    except Exception as e:
        print(f"Failed Robustness To Expected Corruption On Scheme {scheme}: {e}")
    print()

exit()

# RERUN - PLOTTING ISSUE (Pickle should've accounted for but doesn't)

# print("Original Vs Updated EV + QUE Experiment")
# experiment_suite(ev_que_original_vs_new, identity_corruption_schemes["dkk"], ev_que_experiment, runs=3, plot_good_sample=False, error_bars=True, save_title="que2_catostrophic", pickle_results=True, style_dict="main")

# print("PGD Options")
# for key, scheme in mix_schemes.items():
#     print(f"Corruption Scheme: {key}")
#     if key != "id_unif_top":
#         continue
#     try:
#         experiment_suite(grad_iterations, scheme, experiments, runs=2, error_bars=True, plot_good_sample=False, plot=False, save_title=f"pgd_tuning-{key}", style_dict="pgd", pickle_results=True)
#         #experiment_suite(lrv_options3, scheme, old_experiments_short, runs=2, error_bars=True, plot_good_sample=True, plot=False, save_title=f"lrv_sample/knownid_{key}", style_dict="lrv_sample")
#     except Exception as e:
#         print(f"Failed PGD Hyperparameter Tuning On Scheme {scheme}: {e}")
#     print()

# print("EV Pruning Comparison")
# for key, scheme in mix_schemes.items():
#     print(f"Corruption Scheme: {key}")
#     try:
#         if key.startswith("id"):
#             experiment_suite(ev_pruning_comparison, scheme, experiments, runs=3, error_bars=True, plot_good_sample=False, plot=False, save_title=f"ev_pruning-{key}", style_dict="ev_pruning", pickle_results=False)
#         else:
#             experiment_suite(ev_pruning_comparison, scheme, experiments, runs=3, error_bars=True, plot_good_sample=False, plot=False, save_title=f"ev_pruning-{key}", style_dict="ev_pruning", pickle_results=False, sample_scale_cov=True)
#         #experiment_suite(lrv_options3, scheme, old_experiments_short, runs=2, error_bars=True, plot_good_sample=True, plot=False, save_title=f"lrv_sample/knownid_{key}", style_dict="lrv_sample")
#     except Exception as e:
#         print(f"Failed PGD Hyperparameter Tuning On Scheme {scheme}: {e}")
#     print()

# print("QUE Alpha Comparison")
# for key, scheme in mix_schemes.items():
#     print(f"Corruption Scheme: {key}")
#     try:
#         if key.startswith("id"):
#             experiment_suite(que_estimators, scheme, experiments, runs=3, error_bars=True, plot_good_sample=False, plot=False, save_title=f"que_tuning-{key}", style_dict="que", pickle_results=True)
#         else:
#             experiment_suite(que_estimators, scheme, experiments, runs=3, error_bars=True, plot_good_sample=False, plot=False, save_title=f"que_tuning-{key}", style_dict="que", pickle_results=True, sample_scale_cov=True)
#         #experiment_suite(lrv_options3, scheme, old_experiments_short, runs=2, error_bars=True, plot_good_sample=True, plot=False, save_title=f"lrv_sample/knownid_{key}", style_dict="lrv_sample")
#     except Exception as e:
#         print(f"Failed PGD Hyperparameter Tuning On Scheme {scheme}: {e}")
#     print()

# print("QUE tau0 vs tau1 fast Comparison")
# for key, scheme in mix_schemes.items():
#     print(f"Corruption Scheme: {key}")
#     try:
#         experiment_suite(que_estimators, scheme, experiments, runs=2, error_bars=True, plot_good_sample=False, plot=False, save_title=f"Test/que_old_vs_new_{key}", pickle_results=False)
#         #experiment_suite(lrv_options3, scheme, old_experiments_short, runs=2, error_bars=True, plot_good_sample=True, plot=False, save_title=f"lrv_sample/knownid_{key}", style_dict="lrv_sample")
#     except Exception as e:
#         print(f"Failed PGD Hyperparameter Tuning On Scheme {scheme}: {e}")
#     print()

print("Time Experiment")
experiment_suite(main_estimators, identity_corruption_schemes["gaus_one"], time_experiments, option="time", runs=3, plot_good_sample=False, error_bars=True, save_title="NoPGDtime-id_gaus_one", pickle_results=True, style_dict="main")
experiment_suite(main_estimators, identity_corruption_schemes["subtractive_corruption"], time_experiments, option="time", runs=3, plot_good_sample=False, error_bars=True, save_title="NoPGDtime-id_sub", pickle_results=True, style_dict="main")

# 6 - Eigenvalue Filtering Comparison

# print("Eigenvalue Filtering Old Vs Updated Threshold")
# experiment_suite(eigenvalue_old_vs_new, id_dkk_corruption, ev_comparison_experiment, runs=2, error_bars=True, plot_good_sample=False, plot=False, save_title=f"eigenvalue_comparison", style_dict="ev_compare", sample_scale_cov=False)
# print()

# 4.4 - Unknown Non Spherical Covariance Experiments: Non Diagonal

# this might not actually make sense for diminishing diagonal
# print("Rotated Experiments")
# for key, scheme in mix_schemes.items():
#     print(f"Corruption Scheme: {key}")
#     try:
#         if key.startswith("id"):
#             experiment_suite(main_estimators, scheme, experiments, runs=3, error_bars=True, plot_good_sample=True, plot=False, save_title=f"rotate-{key}", style_dict="main", sample_scale_cov=False, rotate=True,pickle_results=True)
#         else:
#             experiment_suite(main_estimators, scheme, experiments, runs=3, error_bars=True, plot_good_sample=True, plot=False, save_title=f"rotate-{key}", style_dict="main", sample_scale_cov=True, rotate=True,pickle_results=True)
#         # experiment_suite(main_estimators, scheme, experiments_short, runs=3, error_bars=True, plot_good_sample=True, plot=False, save_title=f"rotate/{key}", style_dict="main", sample_scale_cov=True, rotate=True)
#     except Exception as e:
#         # this exception can get messed up when scheme is replaced by another variable
#         print(f"Failed Rotated On Scheme {scheme}: {e}")
#     print()


# print("LRV Options")
# for key, scheme in mix_schemes.items():
#     print(f"Corruption Scheme: {key}")
#     try:
#         experiment_suite(lrv_options_C,scheme, experiments, runs=5, error_bars=True, plot_good_sample=False, plot=False, save_title=f"lrv_tuning_{key}", style_dict="lrv")
#         #experiment_suite(lrv_options_C,scheme, old_experiments_short, runs=2, error_bars=True, plot_good_sample=True, plot=False, save_title=f"lrv_sample/knownid_{key}", style_dict="lrv_sample")
#     except Exception as e:
#         print(f"Failed LRV Hyperparameter Tuning On Scheme {scheme}: {e}")
#     print()

# for key, scheme in large_outlier_identity_corruption_schemes.items():
#     print(f"Corruption Scheme: {key}")
#     try:
#         experiment_suite(lrv_options_C,scheme, experiments, runs=10, error_bars=True, plot_good_sample=False, plot=False, save_title=f"lrv_tuning/{key}", style_dict="lrv_sample")
#         # experiment_suite(lrv_options3, scheme, experiments_short, runs=3, error_bars=True, plot_good_sample=True, plot=True, style_dict="lrv_sample")
#         # experiment_suite(main_estimators, scheme, old_experiments_short, runs=2, error_bars=True, plot_good_sample=True, plot=True, style_dict="lrv_sample")
#     except Exception as e:
#         print(f"Failed LRV Hyperparameter Tuning On Scheme {scheme}: {e}")
#     print()

