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
# Pi-Flow Policy Distillation Training Script
# Supports GMM (Gaussian Mixture Model) and DX (Direct x0) policies
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

# ============================================================
# Policy Type Selection
# Options: "gmm" or "dx"
#   - gmm: Gaussian Mixture Model policy (K Gaussians)
#   - dx:  Direct x0 grid policy (K grid points)
# ============================================================
POLICY_TYPE="gmm" # Default to GMM, override with POLICY_TYPE=dx

if [ "$POLICY_TYPE" = "dx" ]; then
    config_name="self_forcing_piflow_dx"
else
    config_name="self_forcing_piflow_gmm"
fi

# ============================================================
# Run Training
# ============================================================
echo "=========================================="
echo "Pi-Flow Policy Distillation"
echo "Policy Type: $POLICY_TYPE"
echo "Config: $config_name"
echo "GPUs: $NUM_GPUS"
echo "=========================================="

torchrun --nproc_per_node=$NUM_GPUS --standalone --master_port=$MASTER_PORT \
  train.py \
  --config_path configs/$config_name.yaml \
  --logdir "outputs/${config_name}_${SLURM_JOB_ID}" \
  --disable-wandb
