#!/bin/bash
#SBATCH --account=vadmanuel
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=10
#SBATCH --time=00:20:00
#SBATCH --error=myjobresults-%J.err
#SBATCH --output=myjobresults-%J.out
#SBATCH --job-name=pa2-module1

# Load Modules

echo "SLURM nodes assigned :$SLURM_JOB_NODELIST"
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
module load cuda time argparse pytorch CNN torch vision torch

python main.py
