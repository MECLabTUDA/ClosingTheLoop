cd 2class

CUDA_VISIBLE_DEVICES="3" python seresnext50_128_2class.py > /local/scratch/continualLearning_results/trainval_small/2nd_level/split_fold3/seresnext50_128_2class.txt
CUDA_VISIBLE_DEVICES="3" python validation_128_2class.py > /local/scratch/continualLearning_results/trainval_small/2nd_level/split_fold3/seresnext50_128_validation_2class.txt

cd ..
cd 1class

CUDA_VISIBLE_DEVICES="3,4" python seresnext50_256_1class.py > /local/scratch/continualLearning_results/trainval_small/2nd_level/split_fold3/seresnext50_256_1class.txt
CUDA_VISIBLE_DEVICES="3,4" python validation_256_1class.py > /local/scratch/continualLearning_results/trainval_small/2nd_level/split_fold3/seresnext50_256_validation_1class.txt
