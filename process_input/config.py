process_input_split2_config = dict(
    
    csv_path = '/local/scratch/continualLearning_results/new_train_jpegs_5fold.csv',
    series_base_path = '/local/scratch/rsna-str-pulmonary-embolism-detection/train/',
    out_dir = '/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/'
)

process_input_test_config = dict(
    csv_path = '/local/scratch/rsna-str-pulmonary-embolism-detection/test.csv',
    series_path = '/local/scratch/rsna-str-pulmonary-embolism-detection/test/',
    out_dir = '/local/scratch/continualLearning_results/trainval_small/process_input/split_fold3/test/'
)