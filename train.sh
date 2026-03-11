#!/bin/bash
#SBATCH --job-name=train_pi0_real
#SBATCH --output=logs/%x-%j.err
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=gu-compute
#SBATCH --gres=gpu:1
#SBATCH --qos=gu-med
#SBATCH --time=48:00:00
#SBATCH --mem=256G

eval "$(conda shell.bash hook)"
conda activate openpi

export XLA_PYTHON_CLIENT_MEM_FRACTION=1.0

python scripts/train.py \
  pi0_realworld_yam \
  --exp-name=train_pi0_real \
  --overwrite
