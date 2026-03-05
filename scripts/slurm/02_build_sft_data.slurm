#!/bin/bash
#SBATCH --job-name=gar-build-sft
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --output=logs/%j_build_sft.log
#
# Builds the discriminator SFT dataset by labeling reasoning slices via the
# TAMUS AI API. Submit from your project root:
#   cd $SCRATCH/gar && sbatch scripts/slurm/02_build_sft_data.sh
#
# REQUIRES: TAMUS_AI_CHAT_API_KEY exported in your ~/.bashrc on the cluster.

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
PYTHON="$SCRATCH/conda_envs/gar_env/bin/python"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data

echo "Starting SFT data build at $(date)"

"$PYTHON" scripts/build_discriminator_sft_data.py \
    --config configs/qwen_gar_paper.yaml \
    --api_base https://chat-api.tamu.ai/api \
    --api_key_env TAMUS_AI_CHAT_API_KEY \
    --openai_model protected.o4-mini

echo "Done at $(date)"
