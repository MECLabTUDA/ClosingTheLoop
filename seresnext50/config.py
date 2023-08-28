extract_raw_image_config = dict(
    
    base_path = "/local/scratch/rsna-str-pulmonary-embolism-detection/test/",
    image_list_test = "/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/test/image_list_test.pickle",
    image_dict_test = "/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/test/image_dict_test.pickle",
    bbox_dict_test = "/local/scratch/continualLearning_results/trainval_small/lung_localization/split_fold3/bbox_dict_test.pickle",

    hyperparam_batch_size = 1,
    hyperparameter_image_size = 576,
    
    #model_state path excluding checkpoint from where the state should be loaded
    # Adapt the checkpoint in the code corresponding to your model structure
    model_state_path = "/local/scratch/continualLearning_results/trainval_small/seresnext50/split_fold3/weights/",

    #out dir for results
    out_dir ="/local/scratch/rsna_results/trainval_small/seresnext50/split_fold3/test_attention_maps0/",
    
    #output pd.df with pred_score for the study id and the positively evaluated slices
    result_path = "/gris/gris-f/homestud/mofuchs/EVA_KI_data/transparency/output_final.xlsx"    
)

extract_test_attention_maps_config = dict(
    base_path= "/local/scratch/rsna-str-pulmonary-embolism-detection/test/",
    image_list_test = "/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/test/image_list_test.pickle",
    image_dict_test = "/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/test/image_dict_test.pickle",
    bbox_dict_test = "/local/scratch/continualLearning_results/trainval_small/lung_localization/split_fold3/bbox_dict_test.pickle",

    hyperparam_batch_size = 1,
    hyperparameter_image_size = 576,
    
    #model_state path excluding checkpoint from where the state should be loaded
    # Adapt the checkpoint in the code corresponding to your model structure
    model_state_path = "/local/scratch/continualLearning_results/trainval_small/seresnext50/split_fold3/weights/",

    # output_dir for Inject model with M3d-CAM
    output_dir= "/local/scratch/rsna_results/trainval_small/seresnext50/split_fold3/test_attention_maps0",
    
    #output pd.df with pred_score for the study id and the positively evaluated slices
    result_path = "/gris/gris-f/homestud/mofuchs/EVA_KI_data/transparency/output_final.xlsx"    

)


save_test_features_config = dict(
    data_base_path = '/local/scratch/rsna-str-pulmonary-embolism-detection/test/',
    image_list_test = "/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/test/image_list_test.pickle",
    image_dict_test = "/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/test/image_dict_test.pickle",
    bbox_dict_test = "/local/scratch/continualLearning_results/trainval_small/lung_localization/split_fold3/bbox_dict_test.pickle",
    
    #model_state path excluding checkpoint from where the state should be loaded
    # Adapt the checkpoint in the code corresponding to your model structure
    model_state_path = "/local/scratch/continualLearning_results/trainval_small/seresnext50/split_fold3/weights/",
    
    #hyperparameter
    batch_size = 96,
    image_size = 576,
    
    #out dir for features 
    out_dir = '/local/scratch/continualLearning_results/trainval_small/seresnext50/split_fold3/features0'


)

save_train_features_config = dict(
    data_base_path = '/local/scratch/rsna-str-pulmonary-embolism-detection/train/',
    image_list_train = '/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/image_list_train.pickle',
    image_dict_train = '/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/image_dict.pickle', 
    bbox_dict_train = '/local/scratch/continualLearning_results/trainval_small/lung_localization/split_fold3/bbox_dict_train.pickle',
    
    #model_state path excluding checkpoint from where the state should be loaded
    # Adapt the checkpoint in the code corresponding to your model structure
    model_state_path = "/local/scratch/continualLearning_results/trainval_small/seresnext50/split_fold3/weights/",
    
    #out dir for features 
    out_dir = '/local/scratch/continualLearning_results/trainval_small/seresnext50/split_fold3/features0',
    
    #hyperparameter
    batch_size = 96,
    image_size = 576

)

save_valid_features_config = dict(
    data_base_path = '/local/scratch/rsna-str-pulmonary-embolism-detection/train/',
    image_list_valid = '/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/image_list_valid.pickle',    
    image_dict_valid = '/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/image_dict.pickle',
    bbox_dict_valid = '/local/scratch/continualLearning_results/trainval_small/lung_localization/split_fold3/bbox_dict_valid.pickle',
    
    #model_state path excluding checkpoint from where the state should be loaded
    model_state_path = '/local/scratch/continualLearning_results/trainval_small/seresnext50/split_fold3/weights/',
    
    #hyperparameter
    batch_size = 96,
    image_size = 576,
    
    #out dir for features    
    out_dir = '/local/scratch/continualLearning_results/trainval_small/seresnext50/split_fold3/features0',

)

train_config = dict(
    data_base_path = '/local/scratch/rsna-str-pulmonary-embolism-detection/train/',
    image_list_train = '/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/image_list_train.pickle',    
    image_dict_train = '/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/image_dict.pickle',
    bbox_dict_train = '/local/scratch/continualLearning_results/trainval_small/lung_localization/split_fold3/bbox_dict_train.pickle',

    #model_state path excluding checkpoint from where the state should be loaded
    # Adapt the checkpoint in the code corresponding to your model structure
    model_state_path = '/local/scratch/continualLearning_results/trainval_small/seresnext50/split_fold3/weights/',
    
    #if you want to resume a previous training, adapt the following two parameters.
    # Note: State the complete path for the checkpoint path
    resume_training = True,
    checkpoint_path = '/local/scratch/continualLearning_results/trainval_small/seresnext50/split_fold3/weights/epoch0',

    #hyperparameter
    learning_rate = 0.0004,
    batch_size = 24,
    image_size = 576,
    num_epoch = 1,
    
    #out dir to save model
    model_out_dir = '/local/scratch/continualLearning_results/trainval_small/seresnext50/split_fold3/weights/',
)

valid_config = dict(
    data_base_path = '/local/scratch/rsna-str-pulmonary-embolism-detection/train/',
    image_list_valid = '/local/scratch/continualLearning_results/trainval_all/process_input/split_fold3/image_list_train.pickle',    
    image_dict_valid = '/local/scratch/continualLearning_results/trainval_all/process_input/split_fold3/image_dict.pickle',
    bbox_dict_valid = '/local/scratch/continualLearning_results/trainval_all/lung_localization/split_fold3/bbox_dict_train.pickle',

    #model_state path excluding checkpoint from where the state should be loaded
    # Adapt the checkpoint in the code corresponding to your model structure
    model_state_path = '/local/scratch/continualLearning_results/trainval_all/seresnext50/split_fold3/weights/',
    
    #hyperparameter
    batch_size = 96,
    image_size = 576
)
