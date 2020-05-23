#!/bin/bash
#
#SBATCH --partition=sc-quick 
#SBATCH --nodelist=scq2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 16
#SBATCH --mem 100GB

#SBATCH --x11

python $1
