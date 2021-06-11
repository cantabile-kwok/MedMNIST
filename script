#!/bin/bash
#SBATCH --job-name=chest_new_50
#SBATCH --partition=2080ti
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --output=5-25-chestmnist_new_50.out
module add cuda/10.1
module add gcc/8.4.0
python mytrain.py --data_name chestmnist --start_epoch 0 --end_epoch 100 --model_name resnet50
