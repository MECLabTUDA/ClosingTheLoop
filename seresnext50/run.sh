python -m torch.distributed.launch --nproc_per_node=2 train.py > /local/scratch/continualLearning_results/trainval_small/seresnext50/split_fold3/train0.txt

python train.py > /local/scratch/continualLearning_results/trainval_small/seresnext50/split_fold3/test0.txt
python valid.py > /local/scratch/continualLearning_results/trainval_small/seresnext50/split_fold3/valid0.txt
python save_valid_features.py > /local/scratch/continualLearning_results/trainval_small/seresnext50/split_fold3/save_valid_features0.txt
python save_train_features.py > /local/scratch/continualLearning_results/trainval_small/seresnext50/split_fold3/save_train_features0.txt
python save_test_features.py > /local/scratch/continualLearning_results/trainval_small/seresnext50/split_fold3/save_test_features0.txt