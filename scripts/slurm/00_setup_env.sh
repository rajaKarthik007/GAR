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

# Initialize conda shell functions (required in non-interactive scripts).
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "=== Creating conda environment at $ENV_PREFIX ==="
conda create --prefix "$ENV_PREFIX" python=3.10 -y

conda activate "$ENV_PREFIX"

echo "=== Installing PyTorch with CUDA 12.1 ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing project dependencies ==="
cd "$PROJECT_DIR"
pip install -e "."

echo ""
echo "=== Setup complete! ==="
echo ""
echo "ACTION REQUIRED: Add your TAMUS API key to ~/.bashrc:"
echo "  echo 'export TAMUS_AI_CHAT_API_KEY=<your-key-here>' >> ~/.bashrc"
echo "  source ~/.bashrc"
echo ""
echo "Then run: bash scripts/slurm/01_download_models.sh"
