validation_128_2class_config = dict(
    series_list_valid ='/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/series_list_valid.pickle',
    image_list_valid = '/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/image_list_valid.pickle', 
    image_dict = '/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/image_dict.pickle',
    series_dict = '/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/series_dict.pickle',
    feature_valid = '/local/scratch/continualLearning_results/trainval_small/seresnext50/split_fold3/features0/feature_valid.npy',

    # hyperparameters
    seq_len = 128,
    feature_size = 2048*3,
    lstm_size = 512,
    learning_rate = 0.0005,
    batch_size = 64,
       
    #model_state path excluding checkpoint from where the state should be loaded
    # Adapt the checkpoint in the code corresponding to your model structure
    model_state_dict = '/local/scratch/continualLearning_results/trainval_small/2nd_level/split_fold3/weights/',
    
)

seresnext50_128_2class_config = dict(
    series_list_train = '/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/series_list_train.pickle', 
    series_list_valid = '/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/series_list_valid.pickle', 
    image_list_train = '/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/image_list_train.pickle', 
    image_list_valid = '/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/image_list_valid.pickle', 
    image_dict = 'local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/image_dict.pickle',
    series_dict = '/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/series_dict.pickle',
    feature_train = '/local/scratch/continualLearning_results/trainval_small/seresnext50/split_fold3/features0/feature_train.npy',
    feature_valid = '/local/scratch/continualLearning_results/trainval_small/seresnext50/split_fold3/features0/feature_valid.npy',
    
    # hyperparameters
    seq_len = 128,
    feature_size = 2048*3,
    lstm_size = 512,
    learning_rate = 0.0005,
    batch_size = 64,
    num_epoch = 25,
    
    result_out_dir = '/local/scratch/continualLearning_results/trainval_small/2nd_level/split_fold3/predictions_2class/',
    #base path to save model
    model_out_dir = '/local/scratch/continualLearning_results/trainval_small/2nd_level/split_fold3/weights/',


)