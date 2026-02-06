#!/bin/bash
#SBATCH --job-name=sf-dmd
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dgx-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --exclude=dgx017

# ============================================================
# Pi-Flow Policy Distillation Training Script
# Supports GMM (Gaussian Mixture Model) and DX (Direct x0) policies
# ============================================================

module purge
module load cuda/12.8
source /vast/projects/jgu32/lab/yhpark/miniconda3/etc/profile.d/conda.sh
conda activate self_forcing 

MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
NUM_GPUS=$SLURM_GPUS_ON_NODE
cd $SLURM_SUBMIT_DIR

# ============================================================
# Config: DMD Baseline (Self-Forcing)
# ============================================================
config_name="self_forcing_dmd"

# ============================================================
# Run Training
# ============================================================
echo "=========================================="
echo "Self-Forcing DMD Distillation"
echo "Config: $config_name"
echo "GPUs: $NUM_GPUS"
echo "=========================================="

# Export so child processes inherit the variable
export DEBUG_GPU_MEMORY=0
export DEBUG_INFERENCE=0

torchrun --nproc_per_node=$NUM_GPUS --standalone --master_port=$MASTER_PORT \
  train.py \
  --config_path configs/$config_name.yaml \
  --logdir "outputs/${config_name}_${SLURM_JOB_ID}" \
  --disable-wandb

# torchrun --nproc_per_node=$NUM_GPUS --standalone --master_port=$MASTER_PORT \
#   train.py \
#   --config_path configs/$config_name.yaml \
#   --logdir "outputs/${config_name}_${SLURM_JOB_ID}" \
#   --disable-wandb
