# GAR Reproduction (Qwen Variant)

Reproduction of **Generative Adversarial Reasoner (GAR)** ([arXiv:2512.16917](https://arxiv.org/abs/2512.16917)), a joint adversarial RL framework that co-evolves an LLM reasoner and an LLM-based discriminator to improve mathematical reasoning.

**Paper setup (Qwen variant):**
- Reasoner: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- Discriminator: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- Dataset: OpenR1-Math-220k
- Hardware: 8x NVIDIA H100 GPUs

## Repository Structure

```
.
├── configs/
│   ├── qwen_gar.yaml           # Local MacBook profile (MPS, small models)
│   └── qwen_gar_paper.yaml     # Paper-scale profile (CUDA, full models)
├── scripts/
│   ├── build_discriminator_sft_data.py   # Step 1: Label slices via OpenAI API
│   ├── train_discriminator_sft.py        # Step 2: SFT the discriminator
│   ├── train_gar.py                      # Step 3: Joint adversarial RL training
│   ├── eval_math.py                      # Step 4: Evaluate Pass@1
│   └── setup_tamus_llm.sh               # Optional: TAMUS AI endpoint setup
├── src/gar/                    # Core library
│   ├── config.py               # Configuration dataclasses + YAML loader
│   ├── data.py                 # Dataset loading and preprocessing
│   ├── modeling.py             # Model loading, generation, log-prob computation
│   ├── parsing.py              # Output parsing (think/answer, yes/no, math answers)
│   ├── prompts.py              # Prompt formatting (uses tokenizer chat templates)
│   ├── rewards.py              # Reward functions (reasoner + discriminator)
│   ├── slicing.py              # Reasoning trace segmentation
│   ├── openai_labeler.py       # OpenAI API integration for slice labeling
│   └── utils.py                # Seed setting
├── data/
│   ├── train_local.jsonl       # Small local training set (8 examples, for smoke tests)
│   ├── discriminator_sft_local.jsonl   # Pre-built SFT data for local testing
│   └── eval_comprehensive.jsonl        # 115-question math evaluation benchmark
├── checkpoints/                # Saved model checkpoints (created during training)
└── pyproject.toml              # Package definition and dependencies
```

## Training Pipeline Overview

GAR training has two stages, matching the paper:

1. **Discriminator SFT**: Fine-tune a smaller LM to evaluate reasoning slices in an analysis-score-rationale format, using labels from GPT-o4-mini.
2. **Joint Adversarial RL**: Co-train the reasoner and discriminator using GRPO with four reward signals:
   - **R^m** (exact match): Binary reward for correct final answers
   - **R^s** (slice reward): Mean discriminator score across reasoning slices
   - **R^d** (discriminator reward): GAN-style real/fake classification loss
   - **R^a** (alignment reward): BCE between slice scores and answer correctness

The overall rewards are:
- Reasoner: `R^rea = λ1 * R^m + λ2 * R^s`
- Discriminator: `R^dis = λ3 * R^d + λ4 * R^a`

---

## Part 1: Running Locally on MacBook (Apple Silicon)

The local profile uses small models (`Qwen2.5-0.5B-Instruct`) and reduced hyperparameters to validate that the full pipeline runs correctly on a MacBook with MPS.

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+ (system Python or Homebrew; **not** Anaconda — see note below)
- ~8 GB free memory

### Step 1: Environment Setup

```bash
# Clone or navigate to the repository
cd /path/to/Code

# Create virtual environment (use system Python, not conda)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -U pip
pip install -e .
```

> **Note on Conda**: If your default `python3` comes from Anaconda/Miniconda, the editable install (`pip install -e .`) may not properly register the `gar` package due to a known `.pth` file processing issue. The scripts work regardless because they use `sys.path.insert` internally. If you need `import gar` from a Python shell, use:
> ```bash
> PYTHONPATH=src python -c "import gar; print(gar.__file__)"
> ```

### Step 2: Build Discriminator SFT Data

This step labels reasoning slices using an OpenAI-compatible API. For local smoke testing, pre-built SFT data is included (`data/discriminator_sft_local.jsonl`), so you can skip to Step 3.

**To regenerate SFT data:**

```bash
export OPENAI_API_KEY=<your-key>
python scripts/build_discriminator_sft_data.py --config configs/qwen_gar.yaml
```

This script:
1. Loads training data (local JSONL or OpenR1-Math-220k from Hugging Face)
2. Samples a subset (0.5% for local, 10% for paper)
3. Segments reasoning traces into slices (max 320 tokens each)
4. Labels each slice via GPT-o4-mini (analysis + YES/NO verdict + rationale)
5. Balances to 1:1 YES/NO ratio
6. Writes to `data/discriminator_sft_local.jsonl`

**Using TAMUS AI (Texas A&M) instead of OpenAI:**

```bash
# One-time setup
bash scripts/setup_tamus_llm.sh \
  --endpoint https://chat-api.tamu.ai \
  --key-id chat.tamu.ai \
  --default-model protected.o4-mini

# Run labeling
export TAMUS_AI_CHAT_API_KEY="$(llm keys get chat.tamu.ai)"
python scripts/build_discriminator_sft_data.py \
  --config configs/qwen_gar.yaml \
  --api_base https://chat-api.tamu.ai/api \
  --api_key_env TAMUS_AI_CHAT_API_KEY \
  --openai_model protected.o4-mini
```

### Step 3: Train Discriminator (SFT)

```bash
python scripts/train_discriminator_sft.py --config configs/qwen_gar.yaml
```

Local defaults: 30 steps, batch size 1, lr=5e-5, max sequence length 512.

Output: `checkpoints/discriminator_local/` (model weights + tokenizer).

The SFT training properly masks prompt tokens so the loss is only computed on the target (analysis + verdict + rationale).

### Step 4: Joint GAR Training

```bash
python scripts/train_gar.py --config configs/qwen_gar.yaml
```

Local defaults: 20 steps, batch size 1, 2 generations per question, max 128 reasoner tokens.

The script:
1. Loads both models (reasoner + SFT'd discriminator)
2. Enables gradient checkpointing on MPS to reduce memory
3. For each step: generates reasoning, segments into slices, computes rewards, updates both models
4. Saves checkpoints to `checkpoints/reasoner_local/` and `checkpoints/discriminator_local/`

> **Memory**: Joint training loads two models simultaneously. If you hit OOM, reduce `max_reasoner_tokens` or `max_discriminator_tokens` in the config. The script sets `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to use all available MPS memory.

### Step 5: Evaluate

```bash
# Evaluate the trained reasoner
python scripts/eval_math.py \
  --config configs/qwen_gar.yaml \
  --input data/eval_comprehensive.jsonl

# Evaluate the base model for comparison
python scripts/eval_math.py \
  --config configs/qwen_gar.yaml \
  --input data/eval_comprehensive.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct

# Paper-style evaluation with more samples
python scripts/eval_math.py \
  --config configs/qwen_gar.yaml \
  --input data/eval_comprehensive.jsonl \
  --num_samples 30 \
  --max_tokens 2048
```

Options:
- `--model`: Override model path (defaults to trained reasoner checkpoint)
- `--max_tokens`: Override generation length (paper uses 32768 for evaluation)
- `--num_samples`: Number of samples per question for Pass@1 (paper uses 30)

> **Expected local results**: The `Qwen2.5-0.5B-Instruct` model is not trained for `<think>/<answer>` format, so local accuracy will be very low. This is expected — the local profile validates that the code runs correctly, not that the small model achieves good accuracy. Meaningful results require the DeepSeek-R1-Distill models on GPU.

---

## Part 2: Running on a GPU Cluster (TAMU FASTER)

Use `configs/qwen_gar_paper.yaml` for paper-scale hyperparameters.

### Config Differences from Local

| Setting | Local (`qwen_gar.yaml`) | Paper (`qwen_gar_paper.yaml`) |
|---|---|---|
| Reasoner | `Qwen2.5-0.5B-Instruct` | `DeepSeek-R1-Distill-Qwen-7B` |
| Discriminator | `Qwen2.5-0.5B-Instruct` | `DeepSeek-R1-Distill-Qwen-1.5B` |
| dtype | float32 (MPS forces this) | bfloat16 |
| SFT data ratio | 0.5% | 10% |
| SFT max steps | 30 | 500 |
| SFT batch (global) | 1 | 128 (4 per GPU × 4 grad_accum × 8 GPUs) |
| SFT learning rate | 5e-5 | 1e-4 |
| Joint training max steps | 20 | 400 |
| Joint training batch (global) | 1 | 192 (1 per GPU × 24 grad_accum × 8 GPUs) |
| Num generations | 2 | 4 |
| Max reasoner tokens | 128 | 2048 |
| Max discriminator tokens | 64 | 128 |
| Reward lambdas | λ1=λ2=λ3=1, λ4=0.5 | λ1=λ2=λ3=1, λ4=0.5 |

### Cluster Details

Scripts target TAMU's FASTER cluster:
- **GPU node**: `gpu:a100:8` (8× A100 80GB, 1005 GB RAM)
- **Modules used**: `CUDA/12.3.0` + `cuDNN/9.4.0.58-CUDA-12.3.0` (loading the cuDNN module automatically pins CUDA to 12.3)
- **PyTorch**: `torch==2.5.1+cu121` (cu121 wheel is forward-compatible with CUDA 12.3 runtime)
- **All work must live under `$SCRATCH`** — home directory has a small quota that will be exceeded by model weights

All SLURM scripts use explicit binary paths (`$SCRATCH/conda_envs/gar_env/bin/python`, `…/bin/torchrun`) rather than relying on `conda activate` to update `PATH` in non-interactive shells.

---

### Step 0: One-time Environment Setup

Run **once** on the login node from your project directory:

```bash
cd $SCRATCH
git clone <your-github-repo-url> gar
cd $SCRATCH/gar
bash scripts/slurm/00_setup_env.sh
```

This creates `$SCRATCH/conda_envs/gar_env` with Python 3.10, installs PyTorch (cu121), and installs the project package.

**How to verify it succeeded:**
```bash
$SCRATCH/conda_envs/gar_env/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.5.1+cu121 False
# (False is correct on the login node — it has no GPU)

$SCRATCH/conda_envs/gar_env/bin/python -c "import gar; print('gar OK')"
# Expected: gar OK
```

If either import fails, re-run `bash scripts/slurm/00_setup_env.sh` — the script skips env creation if it already exists but always re-runs the pip installs.

After setup, add your TAMUS AI API key to `~/.bashrc` (needed for Step 2):

```bash
echo 'export TAMUS_AI_CHAT_API_KEY=<your-key-here>' >> ~/.bashrc
source ~/.bashrc
echo $TAMUS_AI_CHAT_API_KEY   # should print the key, not empty
```

---

### Step 1: Pre-download Models and Dataset

Compute nodes typically don't have outbound internet access. Download everything from the login node first:

```bash
cd $SCRATCH/gar
bash scripts/slurm/01_download_models.sh
```

Downloads ~15 GB total to `$SCRATCH/hf_cache`:
- `DeepSeek-R1-Distill-Qwen-7B` (~15 GB)
- `DeepSeek-R1-Distill-Qwen-1.5B` (~3 GB)
- `OpenR1-Math-220k` dataset (~2 GB)

**How to verify it succeeded:**

The script prints a success line for each download:
```
Reasoner downloaded.
Discriminator downloaded.
Dataset downloaded.
=== All downloads complete. Models cached at: /scratch/user/<you>/hf_cache ===
```

If you see `ModuleNotFoundError` for `transformers` or `datasets`, the wrong Python was used — check that `$SCRATCH/conda_envs/gar_env/bin/python` exists and that `git pull` picked up the latest `01_download_models.sh`.

You can also confirm the cache exists:
```bash
ls $SCRATCH/hf_cache/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/
# Should show: blobs/  refs/  snapshots/
```

---

### Step 2: Build Discriminator SFT Data

> **Run on the login node, not as a SLURM job.** This step calls the TAMUS AI API over the internet; compute nodes on FASTER do not have outbound internet access. The script needs no GPU — it is CPU-only.

Use `screen` or `tmux` so the job survives if your SSH session drops:

```bash
screen -S sft_build   # start a persistent session

cd $SCRATCH/gar
export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

$SCRATCH/conda_envs/gar_env/bin/python scripts/build_discriminator_sft_data.py \
    --config configs/qwen_gar_paper.yaml \
    --api_base https://chat-api.tamu.ai/api \
    --api_key_env TAMUS_AI_CHAT_API_KEY \
    --openai_model protected.o4-mini \
    2>&1 | tee logs/sft_build_$(date +%Y%m%d_%H%M%S).log
```

Detach without killing: `Ctrl+A` then `D`. Re-attach: `screen -r sft_build`.

Allow up to 8 hours (~28k API calls across ~9k sampled examples).

**How to verify it succeeded:**

The script prints when done:
```
Wrote XXXX rows to data/discriminator_sft.jsonl
```

Check that the output file exists and is non-empty:
```bash
wc -l data/discriminator_sft.jsonl
# Expect: several thousand lines (one JSON object per labeled slice)
```

If empty or missing, check the log for `APIConnectionError` (network issue) or `TAMUS_AI_CHAT_API_KEY not found` (key not set in `~/.bashrc`).

---

### Step 3: Train Discriminator (SFT)

```bash
cd $SCRATCH/gar
# Only submit after Step 2's output file exists
sbatch scripts/slurm/03_train_discriminator_sft.slurm
```

Fine-tunes `DeepSeek-R1-Distill-Qwen-1.5B` with HuggingFace Trainer + PyTorch DDP across 8 A100s via `torchrun`. Global batch = 4 × 4 × 8 = 128. 500 steps, ~30–60 minutes.

**How to monitor:**
```bash
squeue -u $USER
tail -f logs/<jobid>_disc_sft.log
```

**How to verify it succeeded:**

The log should show HuggingFace Trainer progress and end with:
```
Done at <timestamp>
```

Check that the checkpoint directory was created:
```bash
ls checkpoints/discriminator/
# Expect: config.json  model.safetensors (or pytorch_model.bin)  tokenizer.json  etc.
```

If the job fails immediately (exit code non-zero within seconds), it's usually a DDP init issue — check the log for `NCCL` or `RuntimeError` lines.

---

### Step 4: Joint GAR Training

```bash
cd $SCRATCH/gar
# Only submit after Step 3's checkpoint directory exists
sbatch scripts/slurm/04_train_gar.slurm
```

Co-trains the reasoner (7B) and discriminator (1.5B) with GRPO via PyTorch DDP across 8 A100s. Each GPU holds both models and processes its own mini-batch; DDP syncs gradients on each backward pass. Global batch = 1 × 24 × 8 = 192. 400 steps, allow up to 24 hours.

**How to monitor:**
```bash
squeue -u $USER
tail -f logs/<jobid>_train_gar.log
```

Look for per-step log lines like:
```
Step 1/400 | r_loss=... | d_loss=... | reward=...
```

**How to verify it succeeded:**

Log ends with:
```
Done at <timestamp>
```

Check checkpoints exist:
```bash
ls checkpoints/reasoner/    # model weights + tokenizer
ls checkpoints/discriminator/   # updated discriminator weights
```

If the job crashes with OOM, see the "Adjusting for Fewer GPUs" table below and reduce `--nproc_per_node` or increase `gradient_accumulation_steps` in the config.

---

### Step 5: Evaluate

```bash
cd $SCRATCH/gar
# Only submit after Step 4's reasoner checkpoint exists
sbatch scripts/slurm/05_eval.slurm
```

Runs Pass@1 on `data/eval_comprehensive.jsonl` with 30 samples per question and max 32,768 tokens. Evaluates both the GAR-trained reasoner and the base `DeepSeek-R1-Distill-Qwen-7B` for comparison. Single A100, allow up to 8 hours.

**How to monitor:**
```bash
squeue -u $USER
tail -f logs/<jobid>_eval.log
```

**How to verify it succeeded:**

The log prints accuracy after each model:
```
=== Evaluating GAR-trained reasoner at <timestamp> ===
...
Pass@1: 0.XX
=== Evaluating base model (DeepSeek-R1-Distill-Qwen-7B) at <timestamp> ===
...
Pass@1: 0.XX
Done at <timestamp>
```

Compare your results against the paper's Table 1 (see Paper Results section below).

---

### Adjusting for Fewer GPUs

If you have fewer than 8 A100s, keep the global batch size constant by adjusting `gradient_accumulation_steps` in `configs/qwen_gar_paper.yaml` and `--nproc_per_node` + `--gres=gpu:a100:N` in the SLURM scripts:

| GPUs | `per_device_batch_size` | `gradient_accumulation_steps` | Global batch |
|------|------------------------|-------------------------------|--------------|
| 8    | 1                      | 24                            | 192          |
| 4    | 1                      | 48                            | 192          |
| 2    | 1                      | 96                            | 192          |
| 1    | 1                      | 192                           | 192          |

---

## Key Implementation Details

### Prompt Formatting
Prompts are formatted using each model's native chat template via `tokenizer.apply_chat_template()`. This ensures compatibility with any HuggingFace model (Qwen uses ChatML, DeepSeek uses its own format, etc.).

### Reasoning Slicing
Following the paper, reasoning traces are segmented by:
1. Splitting on paragraph breaks and sentence boundaries
2. Respecting semantic break markers ("Therefore", "Thus", "Hence", "So", "Next", "Finally")
3. Enforcing max/min token limits per slice (default: 48-320 tokens)

### Reward Computation
- **Reasoner reward**: GRPO with group-relative advantage normalization (z-score within each question's generation group)
- **Discriminator reward**: GAN-style loss (real/fake discrimination) + BCE alignment loss

### SFT Label Masking
The discriminator SFT correctly masks prompt tokens (setting labels to -100) so the cross-entropy loss is only computed on the target completion (analysis + verdict + rationale), not the prompt.

---

## Paper Reference

**Generative Adversarial Reasoner: Enhancing LLM Reasoning with Adversarial Reinforcement Learning**
Qihao Liu, Luoxin Ye, Wufei Ma, Yu-Cheng Chou, Alan Yuille
Johns Hopkins University
[arXiv:2512.16917](https://arxiv.org/abs/2512.16917)

### Paper Results (Qwen variant, Table 1)

| Benchmark | DS-R1-Distill-Qwen-7B | + GAR |
|---|---|---|
| AIME24 | 54.0 | 61.3 (+7.3) |
| AIME25 | 38.0 | 44.3 (+6.3) |
| MATH500 | 94.3 | 94.8 (+0.5) |
| GSM8K | 90.6 | 92.2 (+1.6) |
| AMC23 | 90.3 | 92.5 (+2.2) |
| OlympiadBench | 52.5 | 54.8 (+2.3) |
| LiveMathBench-Hard | 18.4 | 24.9 (+6.5) |
