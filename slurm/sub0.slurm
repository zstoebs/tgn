#!/bin/bash

#SBATCH --account=h_oguz_lab_acc
#SBATCH --partition=pascal
#SBATCH --gres=gpu:1	

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --time=5-00:00:00
#SBATCH --output=gpu-job.log

# Max job duration
#SBATCH --job-name=sub0

# Communication
#SBATCH --mail-user=zachary.a.stoebner@vanderbilt.edu
#SBATCH --mail-type=ALL

ml Anaconda3/5.0.1
source activate my_root 
#conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
#pip install torchvision --upgrade
#pip install torch-summary
#pip install torchio==0.17.0

python ../train_self_supervised.py --use_memory --prefix tgn-attn --n_runs 10 --use_validation
python ../train_supervised.py --use_memory --prefix tgn-attn --n_runs 10 --use_validation
