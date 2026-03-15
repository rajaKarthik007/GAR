#!/usr/bin/env bash
# Run this on the FASTER login node to pre-download HuggingFace models and
# the dataset to SCRATCH before submitting training jobs (compute nodes may
# not have outbound internet access).
#
# Run interactively from your project directory:
#   bash scripts/slurm/01_download_models.sh

export HF_HOME="$SCRATCH/hf_cache"
mkdir -p "$HF_HOME"

ml purge
ml GCCcore/13.3.0
ml Miniconda3/23.10.0-1
ml CUDA/12.3.0
ml cuDNN/9.4.0.58-CUDA-12.3.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SCRATCH/conda_envs/gar_env"
PYTHON="$SCRATCH/conda_envs/gar_env/bin/python"

echo "=== Downloading DeepSeek-R1-Distill-Qwen-7B (reasoner) ==="
"$PYTHON" -m pip install -q huggingface_hub
"$PYTHON" -m huggingface_hub.cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
echo "Reasoner downloaded."

echo "=== Downloading DeepSeek-R1-Distill-Qwen-1.5B (discriminator) ==="
"$PYTHON" -m huggingface_hub.cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
echo "Discriminator downloaded."

echo "=== Downloading OpenR1-Math-220k dataset ==="
"$PYTHON" -c "
from datasets import load_dataset
load_dataset('open-r1/OpenR1-Math-220k', split='train')
print('Dataset downloaded.')
"

echo ""
echo "=== All downloads complete. Models cached at: $HF_HOME ==="
echo "You can now submit jobs with: sbatch scripts/slurm/02_build_sft_data.slurm"
