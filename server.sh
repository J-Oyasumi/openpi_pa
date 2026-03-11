#!/bin/bash
#SBATCH --job-name=eval_server
#SBATCH --output=logs/%x-%j.err
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=gu-compute
#SBATCH --gres=gpu:1
#SBATCH --qos=gu-med
#SBATCH --time=24:00:00
#SBATCH --mem=128G


python scripts/serve_policy.py \
--port=8000 policy:checkpoint \
--policy.config=pi0_realworld_yam \
--policy.dir=./checkpoints/pi0_robocasa_ours/train_pi0_real/9999 \