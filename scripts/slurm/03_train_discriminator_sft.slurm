#!/bin/bash
#SBATCH --job-name=gar-disc-sft
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=gpu
#SBATCH --output=logs/%j_disc_sft.log
#
# Fine-tunes the discriminator (DeepSeek-R1-Distill-Qwen-1.5B) on the labeled
# SFT dataset using 8 A100 GPUs with PyTorch DDP via torchrun.
#
# Submit from your project root:
#   cd $SCRATCH/gar && sbatch scripts/slurm/03_train_discriminator_sft.sh

export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

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

echo "Starting discriminator SFT on 8x A100 at $(date)"
echo "Node: $(hostname), GPUs: $CUDA_VISIBLE_DEVICES"

"$TORCHRUN" \
    --nproc_per_node=8 \
    --master_port=29500 \
    scripts/train_discriminator_sft.py \
    --config configs/qwen_gar_paper.yaml

echo "Done at $(date)"
