#!/bin/bash
#SBATCH  --output=log/%j.out
#SBATCH  --gres=gpu:2
#SBATCH  --mem=80G
#SBATCH  --time=2020

source /itet-stor/yankwang/net_scratch/conda/etc/profile.d/conda.sh
conda activate ve_py37_lsd_A100_20230202
python -u inference_time.py "$@"