#!/bin/bash
#SBATCH --job-name=install
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gu-compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --qos=gu-high
#SBATCH --mem=64G

# Environment setup
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate self_forcing 

# pip install uv
# uv pip install -r requirements.txt
# uv pip install flash-attn --no-build-isolation

# huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
# huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-14B
# huggingface-cli download gdhe17/Self-Forcing checkpoints/ode_init.pt --local-dir .
# huggingface-cli download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir prompts
# huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .