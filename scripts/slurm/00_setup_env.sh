#!/usr/bin/env bash
# Run this ONCE on the FASTER login node to create the conda environment.
# This is NOT submitted as a SLURM job — run it interactively:
#   bash scripts/slurm/00_setup_env.sh
#
# Prerequisites:
#   - You are in your project directory on SCRATCH (e.g. cd $SCRATCH/gar)
#   - Your TAMUS API key is ready to add to ~/.bashrc

PROJECT_DIR="$(pwd)"
ENV_PREFIX="$SCRATCH/conda_envs/gar_env"

ml purge
ml GCCcore/13.3.0
ml Miniconda3/23.10.0-1
ml CUDA/12.3.0
ml cuDNN/9.4.0.58-CUDA-12.3.0

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "=== Creating conda environment at $ENV_PREFIX ==="
if [ ! -d "$ENV_PREFIX" ]; then
    conda create --prefix "$ENV_PREFIX" python=3.10 -y
else
    echo "Environment already exists, skipping creation."
fi

conda activate "$ENV_PREFIX"

# Use the explicit env pip to avoid accidentally using the system pip
PIP="$ENV_PREFIX/bin/pip"

echo "=== Installing PyTorch with CUDA 12.1 (compatible with CUDA 12.3 runtime) ==="
"$PIP" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing project dependencies ==="
"$PIP" install -e "$PROJECT_DIR"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "ACTION REQUIRED: Add your TAMUS API key to ~/.bashrc:"
echo "  echo 'export TAMUS_AI_CHAT_API_KEY=<your-key-here>' >> ~/.bashrc"
echo "  source ~/.bashrc"
echo ""
echo "Then run: bash scripts/slurm/01_download_models.sh"
