#!/bin/bash
#SBATCH --job-name=gar-eval
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --output=logs/%j_eval.log
#
# Evaluates the trained reasoner and the base model on the math benchmark.
# Runs on a single A100 (the 7B model + 32K KV cache fits in 80 GB).
#
# Submit from your project root:
#   cd $SCRATCH/gar && sbatch scripts/slurm/05_eval.sh

export HF_HOME="$SCRATCH/hf_cache"

ml purge
ml GCCcore/13.3.0
ml Miniconda3/23.10.0-1
ml CUDA/12.4.0
ml cuDNN/9.4.0.58-CUDA-12.3.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SCRATCH/conda_envs/gar_env"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "=== Evaluating GAR-trained reasoner at $(date) ==="
python scripts/eval_math.py \
    --config configs/qwen_gar_paper.yaml \
    --input data/eval_comprehensive.jsonl \
    --max_tokens 32768 \
    --num_samples 30

echo "=== Evaluating base model (DeepSeek-R1-Distill-Qwen-7B) at $(date) ==="
python scripts/eval_math.py \
    --config configs/qwen_gar_paper.yaml \
    --input data/eval_comprehensive.jsonl \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --max_tokens 32768 \
    --num_samples 30

echo "Done at $(date)"
