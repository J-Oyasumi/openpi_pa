#!/bin/bash
#SBATCH --job-name=eval_policy
#SBATCH --output=logs/%x-%j.err
#SBATCH --error=logs/%x-%j.out
#SBATCH --partition=gu-compute
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --qos=gu-med
#SBATCH --mem=128G


python examples/robocasa/main.py \
--args.port 8000 \
--args.task_set unused_tasks \
--args.split pretrain \
--args.log_dir  /home/hanjiang/work/baseline/openpi/checkpoints/pi0_robocasa_ours/train_pi0/9999 \
--args.num_trials 100