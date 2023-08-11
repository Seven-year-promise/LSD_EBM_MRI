#!/bin/bash
#SBATCH  --output=log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=40G
#SBATCH  --time=2820
#SBATCH  --constraint='titan_xp'

source /itet-stor/yankwang/net_scratch/conda/etc/profile.d/conda.sh
conda activate lsdebm_env
python -u $1
