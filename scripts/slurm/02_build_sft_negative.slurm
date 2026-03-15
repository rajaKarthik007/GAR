#!/bin/bash
#SBATCH --job-name=sft_neg
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:2
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%j_sft_neg.log

#
# Stage 2b: Generate and label negative (incorrect) reasoning traces.
# Run on a single A100 GPU after Stage 2a (positive) completes.
#
# Submit from your project root:
#   cd $SCRATCH/gar && sbatch scripts/slurm/02_build_sft_negative.slurm

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

echo "Starting SFT negative generation at $(date)"
echo "Node: $(hostname), GPU: $CUDA_VISIBLE_DEVICES"

"$PYTHON" scripts/build_discriminator_sft_negative.py \
    --config configs/qwen_gar_paper.yaml \
    --api_base https://chat-api.tamu.ai/api \
    --api_key_env TAMUS_AI_CHAT_API_KEY \
    --openai_model protected.o4-mini \
    --batch_size 2

echo "Done at $(date)"
