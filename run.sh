#!/bin/bash
#SBATCH -J Jobname
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH -t 10:00:00 
#SBATCH --gres=gpu:2
python Q_learning_without_DNN.py

