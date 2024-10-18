"""
Run real world experiments over LLM embeddings, deep pretrained image model embeddings, and GloVe context free word embeddings.
"""


from embedding_setup import field_study_bert768, field_land_bert768, field_study_minillm384, field_land_minillm384,field_study_albert768, field_land_albert768, field_study_t5_512, field_land_t5_512
from embedding_setup import resnet18_512_cat, resnet18_512_dog, mobilenet_960_cat, mobilenet_960_dog, effecientnet_1280_cat, effecientnet_1280_dog, resnet50_2048_cat, resnet50_2048_dog
from embedding_setup import pleasant50, unpleasant50, pleasant100, unpleasant100, pleasant200, unpleasant200, pleasant300, unpleasant300
from embedding_setup import main_estimators
from embedding_setup import llm_loocv_var_range, img_loocv_var_range, glove_loocv_var_range
from embedding_setup import llm_corrupted_experiment, img_corrupted_experiment, glove_corrupted_experiment, img_corrupted_experiment_vs_n
from experiment_helper import loocv_data_size_experiment, embedding_experiment_suite

num_runs = 5

# LOOCV experiments
print("RUNNING ALL LOOCV EXPERIMENTS \n")

print("LLM LOOCV Experiments - Field Of Land")
loocv_data_size_experiment(field_land_minillm384, main_estimators, llm_loocv_var_range, num_runs=num_runs, save_title="MiniLM384_LOOCV_Land", sample_scale_cov="sample", legend=True)
loocv_data_size_experiment(field_land_t5_512, main_estimators, llm_loocv_var_range, num_runs=num_runs, save_title="T5_512_LOOCV_Land", sample_scale_cov="sample", legend=False)
loocv_data_size_experiment(field_land_albert768, main_estimators, llm_loocv_var_range, num_runs=num_runs, save_title="Albert768_LOOCV_Land", sample_scale_cov="sample", legend=False)
loocv_data_size_experiment(field_land_bert768, main_estimators, llm_loocv_var_range, num_runs=num_runs, save_title="Bert768_LOOCV_Land", sample_scale_cov="sample", legend=False)

print("LLM LOOCV Experiments - Field Of Study")
loocv_data_size_experiment(field_study_minillm384, main_estimators, llm_loocv_var_range, num_runs=num_runs, save_title="MiniLM384_LOOCV_Study", sample_scale_cov="sample", legend=True)
loocv_data_size_experiment(field_study_t5_512, main_estimators, llm_loocv_var_range, num_runs=num_runs, save_title="T5_512_LOOCV_Study", sample_scale_cov="sample", legend=False)
loocv_data_size_experiment(field_study_albert768, main_estimators, llm_loocv_var_range, num_runs=num_runs, save_title="Albert768_LOOCV_Study", sample_scale_cov="sample", legend=False)
loocv_data_size_experiment(field_study_bert768, main_estimators, llm_loocv_var_range, num_runs=num_runs, save_title="Bert768_LOOCV_Study", sample_scale_cov="sample", legend=False)

print("Image LOOCV Experiments - Cats")
loocv_data_size_experiment(resnet18_512_cat, main_estimators, img_loocv_var_range, num_runs=num_runs, save_title="Resnet512LOOCV", sample_scale_cov="sample", legend=True)
loocv_data_size_experiment(mobilenet_960_cat, main_estimators, img_loocv_var_range, num_runs=num_runs, save_title="MobileNet960LOOCV", sample_scale_cov="sample", legend=False)
loocv_data_size_experiment(effecientnet_1280_cat, main_estimators, img_loocv_var_range, num_runs=num_runs, save_title="EfficientNet1280LOOCV", sample_scale_cov="sample", legend=False)
loocv_data_size_experiment(resnet50_2048_cat, main_estimators, img_loocv_var_range, num_runs=num_runs, save_title="ResNet2048LOOCV", sample_scale_cov="sample", legend=False)

print("Image LOOCV Experiments - Dogs")
loocv_data_size_experiment(resnet18_512_cat, main_estimators, img_loocv_var_range, num_runs=num_runs, save_title="Resnet512LOOCV", sample_scale_cov="sample", legend=True)
loocv_data_size_experiment(mobilenet_960_cat, main_estimators, img_loocv_var_range, num_runs=num_runs, save_title="MobileNet960LOOCV", sample_scale_cov="sample", legend=False)
loocv_data_size_experiment(effecientnet_1280_cat, main_estimators, img_loocv_var_range, num_runs=num_runs, save_title="EfficientNet1280LOOCV", sample_scale_cov="sample", legend=False)
loocv_data_size_experiment(resnet50_2048_cat, main_estimators, img_loocv_var_range, num_runs=num_runs, save_title="ResNet2048LOOCV", sample_scale_cov="sample", legend=False)

print("GloVe LOOCV Experiments - Pleasant")
loocv_data_size_experiment(pleasant50, main_estimators, glove_loocv_var_range, num_runs=num_runs, save_title="LOOCVGloVe/pleasant50", sample_scale_cov="sample", legend=True)
loocv_data_size_experiment(pleasant100, main_estimators, glove_loocv_var_range, num_runs=num_runs, save_title="LOOCVGloVe/pleasant100", sample_scale_cov="sample", legend=False)
loocv_data_size_experiment(pleasant200, main_estimators, glove_loocv_var_range, num_runs=num_runs, save_title="LOOCVGloVe/pleasant200", sample_scale_cov="sample", legend=False)
loocv_data_size_experiment(pleasant300, main_estimators, glove_loocv_var_range, num_runs=num_runs, save_title="LOOCVGloVe/pleasant300", sample_scale_cov="sample", legend=False)

print("GloVe LOOCV Experiments - Unpleasant")
loocv_data_size_experiment(unpleasant50, main_estimators, glove_loocv_var_range, num_runs=num_runs, save_title="LOOCVGloVe/unpleasant50", sample_scale_cov="sample", legend=True)
loocv_data_size_experiment(unpleasant100, main_estimators, glove_loocv_var_range, num_runs=num_runs, save_title="LOOCVGloVe/unpleasant100", sample_scale_cov="sample", legend=False)
loocv_data_size_experiment(unpleasant200, main_estimators, glove_loocv_var_range, num_runs=num_runs, save_title="LOOCVGloVe/unpleasant200", sample_scale_cov="sample", legend=False)
loocv_data_size_experiment(unpleasant300, main_estimators, glove_loocv_var_range, num_runs=num_runs, save_title="LOOCVGloVe/unpleasant300", sample_scale_cov="sample", legend=False)

print("LOOCV EXPERIMENTS COMPLETE \n\n")

# Corrupted Experiments
print("RUNNING CORRUPTED DATA EXPERIMENTS \n")

print("Image Corrupted Experiments - Cat corrupted with Dog")
embedding_experiment_suite(main_estimators, resnet18_512_cat, resnet18_512_dog,  img_corrupted_experiment, save_title="ResNet512_Corruption", sample_scale_cov=True, num_runs=num_runs, legend=True)
embedding_experiment_suite(main_estimators, mobilenet_960_cat, mobilenet_960_dog, img_corrupted_experiment, save_title="MobileNet960_Corruption", sample_scale_cov=True, num_runs=num_runs, legend=False)
embedding_experiment_suite(main_estimators, effecientnet_1280_cat, effecientnet_1280_dog, img_corrupted_experiment, save_title="EfficientNet1280_Corruption", sample_scale_cov=True, num_runs=num_runs, legend=False)
embedding_experiment_suite(main_estimators, resnet50_2048_cat, resnet50_2048_dog, img_corrupted_experiment, save_title="ResNet2048_Corruption", sample_scale_cov=True, num_runs=num_runs, legend=False)

print("Image Corrupted Experiments - Dog corrupted with Cat")
embedding_experiment_suite(main_estimators, resnet18_512_dog, resnet18_512_cat, img_corrupted_experiment, save_title="ResNet512_Corruption_Inv", sample_scale_cov=True, num_runs=num_runs, legend=True)
embedding_experiment_suite(main_estimators, mobilenet_960_dog, mobilenet_960_cat, img_corrupted_experiment, save_title="MobileNet960_Corruption_Inv", sample_scale_cov=True, num_runs=num_runs, legend=False)
embedding_experiment_suite(main_estimators, effecientnet_1280_dog, effecientnet_1280_cat, img_corrupted_experiment, save_title="EfficientNet1280_Corruption_Inv", sample_scale_cov=True, num_runs=num_runs, legend=False)
embedding_experiment_suite(main_estimators, resnet50_2048_dog, resnet50_2048_cat, img_corrupted_experiment, save_title="ResNet2048_Corruption_Inv", sample_scale_cov=True, num_runs=num_runs, legend=False)

print("GloVe Corrupted Experiments - Pleasant corrupted with Unpleasant")
embedding_experiment_suite(main_estimators, pleasant50, unpleasant50, glove_corrupted_experiment, save_title="GloVe50_Corruption", sample_scale_cov=True, num_runs=num_runs, legend=True)
embedding_experiment_suite(main_estimators, pleasant100, unpleasant100, glove_corrupted_experiment, save_title="GloVe100_Corruption", sample_scale_cov=True, num_runs=num_runs, legend=False)
embedding_experiment_suite(main_estimators, pleasant200, unpleasant200, glove_corrupted_experiment, save_title="GloVe200_Corruption", sample_scale_cov=True, num_runs=num_runs, legend=False)
embedding_experiment_suite(main_estimators, pleasant300, unpleasant300, glove_corrupted_experiment, save_title="GloVe300_Corruption", sample_scale_cov=True, num_runs=num_runs, legend=False)

print("GloVe Corrupted Experiments - Unpleasant corrupted with Pleasant")
embedding_experiment_suite(main_estimators, unpleasant50, pleasant50, glove_corrupted_experiment, save_title="GloVe50_Corruption_Inv", sample_scale_cov=True, num_runs=num_runs, legend=True)
embedding_experiment_suite(main_estimators, unpleasant100, pleasant100, glove_corrupted_experiment, save_title="GloVe100_Corruption_Inv", sample_scale_cov=True, num_runs=num_runs, legend=False)
embedding_experiment_suite(main_estimators, unpleasant200, pleasant200, glove_corrupted_experiment, save_title="GloVe200_Corruption_Inv", sample_scale_cov=True, num_runs=num_runs, legend=False)
embedding_experiment_suite(main_estimators, unpleasant300, pleasant300, glove_corrupted_experiment, save_title="GloVe300_Corruption_Inv", sample_scale_cov=True, num_runs=num_runs, legend=False)

print("LLM Corrupted Experiment - Land corrupted with Study")
embedding_experiment_suite(main_estimators, field_land_minillm384, field_study_minillm384, llm_corrupted_experiment, save_title="MiniLLM384_Corruption", sample_scale_cov=True, num_runs=num_runs, legend=True)
embedding_experiment_suite(main_estimators, field_land_t5_512, field_study_t5_512, llm_corrupted_experiment, save_title="T5512_Corruption", sample_scale_cov=True, num_runs=num_runs, legend=False)
embedding_experiment_suite(main_estimators, field_land_bert768, field_study_bert768, llm_corrupted_experiment, save_title="BERT768_Corruption", sample_scale_cov=True, num_runs=num_runs, legend=False)
embedding_experiment_suite(main_estimators, field_land_albert768, field_study_albert768, llm_corrupted_experiment, save_title="ALBERT768_Corruption", sample_scale_cov=True, num_runs=num_runs, legend=False)

print("LLM Corrupted Experiment - Study corrupted with Land")
embedding_experiment_suite(main_estimators, field_study_minillm384, field_land_minillm384, llm_corrupted_experiment, save_title="MiniLLM384_Corruption_Inv", sample_scale_cov=True, num_runs=num_runs, legend=True)
embedding_experiment_suite(main_estimators, field_study_t5_512, field_land_t5_512, llm_corrupted_experiment, save_title="T5512_Corruption_Inv", sample_scale_cov=True, num_runs=num_runs, legend=False)
embedding_experiment_suite(main_estimators, field_study_bert768, field_land_bert768, llm_corrupted_experiment, save_title="BERT768_Corruption_Inv", sample_scale_cov=True, num_runs=num_runs, legend=False)
embedding_experiment_suite(main_estimators, field_study_albert768, field_land_albert768, llm_corrupted_experiment, save_title="ALBERT768_Corruption_Inv", sample_scale_cov=True, num_runs=num_runs, legend=False)

print("Image Corrupted Experiments VS DATA SIZE - Cat corrupted with Dog")
embedding_experiment_suite(main_estimators, resnet18_512_cat, resnet18_512_dog, img_corrupted_experiment_vs_n, save_title="ResNet512_CorruptionVsN", sample_scale_cov=True, num_runs=num_runs, legend=True)
embedding_experiment_suite(main_estimators, mobilenet_960_cat, mobilenet_960_dog, img_corrupted_experiment_vs_n, save_title="MobileNet960_CorruptionVsN", sample_scale_cov=True, num_runs=num_runs, legend=False)
embedding_experiment_suite(main_estimators, effecientnet_1280_cat, effecientnet_1280_dog, img_corrupted_experiment_vs_n, save_title="EfficientNet1280_CorruptionVsN", sample_scale_cov=True, num_runs=num_runs, legend=False)
embedding_experiment_suite(main_estimators, resnet50_2048_cat, resnet50_2048_dog, img_corrupted_experiment_vs_n, save_title="ResNet2048_CorruptionVsN", sample_scale_cov=True, num_runs=num_runs, legend=False)

print("CORRUPTED DATA EXPERIMENTS COMPLETE")