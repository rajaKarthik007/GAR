#!/usr/bin/env python
"""
Stage 1: Extract and label positive (ground-truth) reasoning traces.
Run on login node (CPU-only, no GPU needed).

Usage:
  python scripts/build_discriminator_sft_positive.py \
    --config configs/qwen_gar_paper.yaml \
    --api_base https://chat-api.tamu.ai/api \
    --api_key_env TAMUS_AI_CHAT_API_KEY \
    --openai_model protected.o4-mini
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from gar.config import load_config
from gar.data import load_math_dataset, sample_ratio
from gar.openai_labeler import label_slice_with_openai
from gar.prompts import discriminator_slice_prompt
from gar.slicing import SliceConfig, segment_reasoning
from gar.utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract and label positive (ground-truth) reasoning traces.")
    p.add_argument("--config", type=str, required=True, help="Path to config YAML")
    p.add_argument("--openai_model", type=str, default="gpt-o4-mini", help="OpenAI model name")
    p.add_argument(
        "--api_base",
        type=str,
        default=None,
        help="Optional OpenAI-compatible base URL",
    )
    p.add_argument(
        "--api_key_env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key",
    )
    return p.parse_args()


def main() -> None:
    print("Starting Stage 2a: Extract positive traces...", flush=True)
    args = parse_args()
    print(f"Args parsed. Config: {args.config}", flush=True)
    cfg = load_config(args.config)
    print(f"Config loaded. Seed: {cfg.seed}", flush=True)
    set_seed(cfg.seed)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.models.discriminator_name, use_fast=True)
    print("Tokenizer loaded.", flush=True)

    slice_cfg = SliceConfig(
        max_slice_tokens=cfg.slicing.max_slice_tokens,
        min_slice_tokens=cfg.slicing.min_slice_tokens,
        semantic_break_markers=tuple(cfg.slicing.semantic_break_markers),
    )
    print("Slice config created.", flush=True)

    # Load and sample dataset
    print("Loading dataset...", flush=True)
    print(f"Dataset name: {cfg.data.dataset_name}, split: {cfg.data.train_split}", flush=True)
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
    print(f"Dataset loaded. Total examples: {len(data)}", flush=True)
    sampled = sample_ratio(data, cfg.data.sft_sample_ratio, cfg.seed)
    print(f"Loaded {len(sampled)} sampled examples", flush=True)

    # Set up OpenAI client
    print(f"Setting up OpenAI client with key from {args.api_key_env}...", flush=True)
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key in env var {args.api_key_env}. "
            f"Set it before running this script."
        )

    print(f"API key found. Base URL: {args.api_base}", flush=True)
    if args.api_base:
        client = OpenAI(api_key=api_key, base_url=args.api_base)
    else:
        client = OpenAI(api_key=api_key)
    print("OpenAI client ready.", flush=True)

    # Extract ground-truth reasoning traces from dataset
    print("Extracting ground-truth reasoning traces...")
    traces_to_label = []
    skipped_no_reasoning = 0
    for ex in tqdm(sampled):
        # Only use examples with actual reasoning field (not fallback to answer)
        if ex.reasoning and ex.reasoning.strip():
            traces_to_label.append((ex.question, ex.reasoning))
        else:
            skipped_no_reasoning += 1

    print(f"Extracted {len(traces_to_label)} positive traces (skipped {skipped_no_reasoning} with missing reasoning)")

    # Label with OpenAI
    print("Labeling positive traces with OpenAI...")
    rows = []
    debug_count = 0
    for q, trace in tqdm(traces_to_label, desc="Labeling slices"):
        slices = segment_reasoning(trace, tokenizer, slice_cfg)
        for s in slices:
            # Enable debug logging for first few slices to diagnose label distribution
            debug = debug_count < 5
            lbl = label_slice_with_openai(client, q, s, model=args.openai_model, debug=debug)
            if debug:
                debug_count += 1
            target = f"{lbl.analysis}\n\n{'**YES**' if lbl.yes_no else '**NO**'}\n\n{lbl.rationale}".strip()
            rows.append(
                {
                    "question": q,
                    "slice": s,
                    "label": lbl.yes_no,
                    "prompt": discriminator_slice_prompt(q, s, tokenizer),
                    "target": target,
                }
            )

    # Write positive examples
    output_path = Path("data/discriminator_sft_positive.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} labeled positive slices to {output_path}")

    # Report label distribution
    labels = [r["label"] for r in rows]
    yes_count = sum(labels)
    no_count = len(labels) - yes_count
    print(f"Label distribution: YES={yes_count}, NO={no_count}")
    if len(labels) > 0:
        print(f"  YES (label=1): {yes_count} ({100*yes_count/len(labels):.1f}%)")
        print(f"  NO (label=0): {no_count} ({100*no_count/len(labels):.1f}%)")


if __name__ == "__main__":
    main()
