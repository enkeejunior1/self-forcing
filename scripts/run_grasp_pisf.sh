#!/bin/bash
#SBATCH --job-name=sf-piflow
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gu-compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --qos=gu-high
#SBATCH --mem=64G

# ============================================================
# Pi-Flow + Self-Forcing (PiSF) Training Script
# Combines pi-flow velocity matching with DMD distribution matching
# DMD timestep sampling is segment-aware
# ============================================================

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate self_forcing 

MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
NUM_GPUS=$SLURM_GPUS_ON_NODE
cd $SLURM_SUBMIT_DIR
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# ============================================================
# Config: PiSF (Pi-Flow + DMD with segment-aware sampling)
# ============================================================
config_name="self_forcing_pisf_gmm"

# ============================================================
# Run Training
# ============================================================
echo "=========================================="
echo "PiSF: Pi-Flow + Self-Forcing Distillation"
echo "Config: $config_name"
echo "GPUs: $NUM_GPUS"
echo "=========================================="

export DEBUG_GPU_MEMORY=0
export DEBUG_INFERENCE=0

torchrun --nproc_per_node=$NUM_GPUS --standalone --master_port=$MASTER_PORT \
  train.py \
  --config_path configs/$config_name.yaml \
  --logdir "outputs/${config_name}_${SLURM_JOB_ID}" \
  --disable-wandb
