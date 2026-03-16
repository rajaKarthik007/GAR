#!/usr/bin/env python
"""
Check how many samples in the dataset have reasoning traces.
Run on login node (CPU-only).

Usage:
  python scripts/check_reasoning_coverage.py --config configs/qwen_gar_paper.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gar.config import load_config
from gar.data import load_math_dataset, sample_ratio
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check reasoning field coverage in dataset")
    p.add_argument("--config", type=str, required=True, help="Path to config YAML")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Load full dataset
    print("Loading full dataset...")
    data = load_math_dataset(
        dataset_name=cfg.data.dataset_name,
        dataset_config=cfg.data.dataset_config,
        split=cfg.data.train_split,
        local_jsonl=cfg.data.local_jsonl,
        question_keys=cfg.data.question_key_candidates,
        answer_keys=cfg.data.answer_key_candidates,
        reasoning_keys=cfg.data.reasoning_key_candidates,
        max_examples=cfg.data.max_train_examples,
    )
    print(f"Loaded {len(data)} total examples\n")

    # Check full dataset
    has_reasoning = sum(1 for ex in tqdm(data, desc="Checking full dataset") if ex.reasoning and ex.reasoning.strip())
    missing_reasoning = len(data) - has_reasoning

    print(f"Full dataset reasoning coverage:")
    print(f"  With reasoning: {has_reasoning} ({100*has_reasoning/len(data):.1f}%)")
    print(f"  Missing reasoning: {missing_reasoning} ({100*missing_reasoning/len(data):.1f}%)\n")

    # Check sampled subset (what Stage 2a will use)
    sampled = sample_ratio(data, cfg.data.sft_sample_ratio, cfg.seed)
    print(f"After sampling ({cfg.data.sft_sample_ratio:.1%}):")
    print(f"  Total sampled examples: {len(sampled)}")

    has_reasoning_sampled = sum(
        1 for ex in tqdm(sampled, desc="Checking sampled subset") if ex.reasoning and ex.reasoning.strip()
    )
    missing_reasoning_sampled = len(sampled) - has_reasoning_sampled

    print(f"  With reasoning: {has_reasoning_sampled} ({100*has_reasoning_sampled/len(sampled):.1f}%)")
    print(f"  Missing reasoning: {missing_reasoning_sampled} ({100*missing_reasoning_sampled/len(sampled):.1f}%)")


if __name__ == "__main__":
    main()
