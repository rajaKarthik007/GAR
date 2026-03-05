#!/bin/bash
#SBATCH --job-name=gar-train
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=gpu
#SBATCH --output=logs/%j_train_gar.log
#
# Joint adversarial RL training (GAR) using 8 A100 GPUs with PyTorch DDP.
# Effective global batch = per_device(1) * grad_accum(24) * 8 GPUs = 192.
#
# Submit from your project root:
#   cd $SCRATCH/gar && sbatch scripts/slurm/04_train_gar.sh

export HF_HOME="$SCRATCH/hf_cache"

ml purge
ml GCCcore/13.3.0
ml Miniconda3/23.10.0-1
ml CUDA/12.3.0
ml cuDNN/9.4.0.58-CUDA-12.3.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SCRATCH/conda_envs/gar_env"
TORCHRUN="$SCRATCH/conda_envs/gar_env/bin/torchrun"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints

echo "Starting GAR training on 8x A100 at $(date)"
echo "Node: $(hostname), GPUs: $CUDA_VISIBLE_DEVICES"

"$TORCHRUN" \
    --nproc_per_node=8 \
    --master_port=29501 \
    scripts/train_gar.py \
    --config configs/qwen_gar_paper.yaml

echo "Done at $(date)"
