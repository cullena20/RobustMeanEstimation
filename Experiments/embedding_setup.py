"""
Setup for embedding experiments. Loads embeddings, defines experiments, and estimators
"""

from experiment_helper import unpickle
import numpy as np
from eigenvalue_pruning import eigenvalue_pruning
from lee_valiant import lee_valiant_simple
from lrv import lrv
from simple_estimators import sample_mean, coordinate_wise_median, median_of_means, geometric_median, coord_trimmed_mean
from que import que_mean
from pgd import grad_descent
import os

# Set current_dir to one level above the current directory
current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# LOAD LLM WORD EMBEDDINGS
bert768 = np.load(os.path.join(current_dir, "Embeddings/LLMEmbeddings/Bert768field_embeddings.npz"))
field_study_bert768, field_land_bert768 = bert768["array1"], bert768["array2"]

minillm384 = np.load(os.path.join(current_dir, "Embeddings/LLMEmbeddings/MiniLLM384field_embeddings.npz"))
field_study_minillm384, field_land_minillm384 = minillm384["array1"], minillm384["array2"]

albert768 = np.load(os.path.join(current_dir, "Embeddings/LLMEmbeddings/ALBERT768field_embeddings.npz"))
field_study_albert768, field_land_albert768 = albert768["array1"], albert768["array2"]

t5_512 = np.load(os.path.join(current_dir, "Embeddings/LLMEmbeddings/t5_512field_embeddings.npz"))
field_study_t5_512, field_land_t5_512 = t5_512["array1"], t5_512["array2"]

# LOAD IMAGE MODEL EMBEDDINGS
resnet18_512 = np.load(os.path.join(current_dir, "Embeddings/ImageEmbeddings/resnet18_embeddings.pkl"), allow_pickle=True)
resnet18_512_cat, resnet18_512_dog = np.array(resnet18_512["cat"]), np.array(resnet18_512["dog"])

mobilenet_960 = np.load(os.path.join(current_dir, "Embeddings/ImageEmbeddings/mobilenet_v3_embeddings.pkl"), allow_pickle=True)
mobilenet_960_cat, mobilenet_960_dog = np.array(mobilenet_960["cat"]), np.array(mobilenet_960["dog"])

effecientnet_1280 = np.load(os.path.join(current_dir, "Embeddings/ImageEmbeddings/efficientnet_b0_embeddings.pkl"), allow_pickle=True)
effecientnet_1280_cat, effecientnet_1280_dog = np.array(effecientnet_1280["cat"]), np.array(effecientnet_1280["dog"])

resnet50_2048 = np.load(os.path.join(current_dir, "Embeddings/ImageEmbeddings/resnet50_embeddings.pkl"), allow_pickle=True)
resnet50_2048_cat, resnet50_2048_dog = np.array(resnet50_2048["cat"]), np.array(resnet50_2048["dog"])

# LOAD GLOVE WORD EMBEDDINGS
pleasant50 = unpickle("Embeddings/GloVeEmbeddings/glove50_pleasant")
pleasant100 = unpickle("Embeddings/GloVeEmbeddings/glove100_pleasant")
pleasant200 = unpickle("Embeddings/GloVeEmbeddings/glove200_pleasant")
pleasant300 = unpickle("Embeddings/GloVeEmbeddings/glove300_pleasant")
pleasant_dict = {"50": pleasant50, "100": pleasant100, "200": pleasant200, "300": pleasant300}

unpleasant50 = unpickle("Embeddings/GloVeEmbeddings/glove50_unpleasant")
unpleasant100 = unpickle("Embeddings/GloVeEmbeddings/glove100_unpleasant")
unpleasant200 = unpickle("Embeddings/GloVeEmbeddings/glove200_unpleasant")
unpleasant300 = unpickle("Embeddings/GloVeEmbeddings/glove300_unpleasant")

# LOOCV data varying ranges (these go into different format than standard experiment suite)
# up to 400 LLM word embeddings
llm_loocv_var_range = np.arange(10, 411, 50)
llm_loocv_var_range = np.append(llm_loocv_var_range, 400)

# up to 1000 Image embeddings
img_loocv_var_range = np.arange(10, 1001, 100)
img_loocv_var_range = np.append(img_loocv_var_range, 1000)

# up to 100 GloVe word embeddings
glove_loocv_var_range = np.arange(10, 101, 10)


# Corrupted Data Experiments - Default Vs Eps experiments
# data size, default_d is None because this is handled by the embedding_experiment_suite (wrapper for experiment_suite)
# default_eta and default_tau are None because eta is varied and tau will take on eta value 
llm_corrupted_experiment = [["eta", np.arange(0.01, 0.46, 0.05), 400, None, None, None]]
img_corrupted_experiment = [["eta", np.arange(0.01, 0.46, 0.05), 1000, None, None, None]]
glove_corrupted_experiment = [["eta", np.arange(0.01, 0.46, 0.05), 100, None, None, None]]

# Corrupted Data Experiments - Vs Data Size
data_var_range_large = np.arange(100, 5000, 500)
data_var_range_large = np.append(data_var_range_large, 5000)

data_var_range_small = np.arange(100, 1000, 200)
data_var_range_small = np.append(data_var_range_small, 1000)

img_corrupted_experiment_vs_n = [
    ["n", data_var_range_large, None, None, 0.1, None],
    ["n", data_var_range_small, None, None, 0.1, None]]

test_estimators = {
    "sample_mean": lambda data, tau: sample_mean(data),
    "median_of_means": lambda data, tau: median_of_means(data, 10)
}

main_estimators = {
    "sample_mean": lambda data, tau: sample_mean(data),
    "coord_median": lambda data, tau: coordinate_wise_median(data),
    "coord_trimmed_mean": lambda data, tau: coord_trimmed_mean(data, tau),
    "geometric_median": lambda data, tau: geometric_median(data),
    "lee_valiant_simple": lambda data, tau: lee_valiant_simple(data, tau, lambda data: median_of_means(data, 10)),
    "median_of_means": lambda data, tau: median_of_means(data, 10),
    "lrv": lambda data, tau: lrv(data, C=1, trace_est_option="robust"),
    "ev_filtering_low_n": lambda data, tau: eigenvalue_pruning(data, tau, gamma=5, t=10, early_halt=False), # do not use halting by default
    "que_low_n": lambda data, tau: que_mean(data, tau, t=10, fast=True, early_halt=True), # use halting by default
    "pgd": lambda data, eta: grad_descent(data, eta, nItr=15)

    # commented out estimators are utlized in ablations  (see if I can fix this)

    #"lrv_general": lambda data, tau: lrvGeneral(data, eta=tau),

    #"ev_filtering_halt": lambda data, tau: eigenvalue_pruning_updated(data, tau, gamma=5, t=10, early_halt=True),
    #"ev_filtering_no_halt": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5, t=10, early_halt=False),
    #"ev_filtering_low_n_random": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5, t=10, early_halt=True, pruning="random"),
    #"ev_filtering_low_n_fixed": lambda data, tau: eigenvalue_pruning_updated(data, tau, 0.1, 5, t=10, early_halt=True, pruning="fixed"),

    #"que_halt": lambda data, tau: que_mean(data, tau, t=10, fast=True, early_halt=True),
    #"que_no_halt": lambda data, tau: que_mean(data, tau, t=10, fast=True, early_halt=False),
    #"que_low_n_halt": lambda data, tau: que_mean(data, tau, t=10, fast=True, early_halt=True),
    #"que_original_halt": lambda data, tau: que_mean(data, tau, t=10, original_threshold=True, fast=True, early_halt=True),
    #"que_no_halt": lambda data, tau: que_mean(data, tau, t=10, fast=True, early_halt=False),

}

estimators_short = {
    "sample_mean": lambda data, tau: sample_mean(data),
    "coord_median": lambda data, tau: coordinate_wise_median(data),
    "coord_trimmed_mean": lambda data, tau: coord_trimmed_mean(data, tau),
    "geometric_median": lambda data, tau: geometric_median(data),
    "lee_valiant_simple": lambda data, tau: lee_valiant_simple(data, tau, lambda data: median_of_means(data, 10)),
    "median_of_means": lambda data, tau: median_of_means(data, 10),
    "que_low_n": lambda data, tau: que_mean(data, tau, t=10, fast=True, early_halt=True), # use halting by default
}
