save_bbox_test_config = dict(
    series_base_path = '/local/scratch/rsna-str-pulmonary-embolism-detection/test/',
    series_list_test = '/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/test/series_list_test.pickle',
    csv_path = '/local/scratch/continualLearning_results/trainval_small/lung_localization/lung_bbox.csv',
    model_state_path = '/local/scratch/continualLearning_results/trainval_small/lung_localization/split_fold3/weights/epoch34_polyak',
    bbox_out_path = '/local/scratch/continualLearning_results/trainval_small/lung_localization/split_fold3/bbox_dict_test.pickle',
)

save_bbox_train_config = dict(
    series_base_path = '/local/scratch/rsna-str-pulmonary-embolism-detection/train/',
    series_list_train = '/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/series_list_train.pickle',
    csv_path = '/local/scratch/continualLearning_results/trainval_small/lung_localization/lung_bbox.csv',
    model_state_path = '/local/scratch/continualLearning_results/trainval_small/lung_localization/split_fold3/weights/epoch34_polyak''/local/scratch/continualLearning_results/trainval_small/lung_localization/split_fold3/weights/epoch34_polyak',
    bbox_out_path = '/local/scratch/continualLearning_results/trainval_small/lung_localization/split_fold3/bbox_dict_train.pickle',  
)

save_bbox_valid_config = dict(
    series_base_path =  '/local/scratch/rsna-str-pulmonary-embolism-detection/train/' ,
)

train_config = dict(
    data_base_path = '/local/scratch/rsna-str-pulmonary-embolism-detection/train/',
    series_list_train = '/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/series_list_train.pickle',
    series_dict_train = '/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/series_dict.pickle',
    image_dict_train = '/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/image_dict.pickle',
    csv_path = '/local/scratch/continualLearning_results/trainval_small/lung_localization/lung_bbox.csv',
    
    #hyperparameter set 1
    learning_rate = 0.0004,
    batch_size = 32,
    image_size = 512,
    num_polyak = 32,
    num_epoch = 20,
    start_epoch = 0,
    
    # hyperparameters set 2
    #learning_rate = 0.00004,
    #batch_size = 32,
    #image_size = 512,
    #num_polyak = 32,
    #num_epoch = 30,
    #start_epoch = 20,
   
    ## hyperparameters set 3
    #learning_rate = 0.000008,
    #batch_size = 32,
    #image_size = 512,
    #num_polyak = 32,
    #num_epoch = 35,
    #start_epoch = 30,
    
    load_model = False,
    #model_state path including checkpoint from where the state should be loaded
    #Note that you need to enter the complete path
    model_state_path= '/local/scratch/continualLearning_results/trainval_small/lung_localization/split_fold3/weights/epoch19',


    #out dir to save model    
    out_dir = '/local/scratch/continualLearning_results/trainval_small/lung_localization/split_fold3/weights/',
    
)


valid_config = dict(
    data_base_path = '/local/scratch/rsna-str-pulmonary-embolism-detection/train/',
    series_list_valid = '/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/series_list_valid.pickle',
    series_dict_valid = '/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/series_dict.pickle',
    image_dict_valid = '/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/image_dict.pickle',
    
    
    #hyperparameter
    batch_size = 256,
    image_size = 512,
    csv_path = '/local/scratch/continualLearning_results/trainval_small/lung_localization/lung_bbox.csv',
    model_state_path  = '/local/scratch/continualLearning_results/trainval_small/lung_localization/split_fold3/weights/',
    
)
