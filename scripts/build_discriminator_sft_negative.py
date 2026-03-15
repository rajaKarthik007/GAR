#!/usr/bin/env python
"""
Stage 2: Generate and label negative (incorrect model-generated) reasoning traces.
Run on GPU node via SLURM to use reasoner for generation.

Usage:
  python scripts/build_discriminator_sft_negative.py \
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
from gar.prompts import reasoner_prompt, discriminator_slice_prompt
from gar.slicing import SliceConfig, segment_reasoning
from gar.utils import set_seed
from gar.modeling import load_causal_lm, generate_text
from gar.parsing import extract_think_answer, exact_match


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate and label negative (incorrect) reasoning traces.")
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
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.models.discriminator_name, use_fast=True)
    slice_cfg = SliceConfig(
        max_slice_tokens=cfg.slicing.max_slice_tokens,
        min_slice_tokens=cfg.slicing.min_slice_tokens,
        semantic_break_markers=tuple(cfg.slicing.semantic_break_markers),
    )

    # Load and sample dataset
    print("Loading dataset...")
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
    sampled = sample_ratio(data, cfg.data.sft_sample_ratio, cfg.seed)
    print(f"Loaded {len(sampled)} sampled examples")

    # Set up OpenAI client
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key in env var {args.api_key_env}. "
            f"Set it before running this script."
        )

    if args.api_base:
        client = OpenAI(api_key=api_key, base_url=args.api_base)
    else:
        client = OpenAI(api_key=api_key)

    # Load reasoner model
    print("Loading reasoner model...")
    reasoner = load_causal_lm(
        cfg.models.reasoner_name,
        dtype=cfg.runtime.dtype,
        device=cfg.runtime.device,
        use_mps_if_available=cfg.runtime.use_mps_if_available,
        trust_remote_code=cfg.runtime.trust_remote_code,
    )

    # Generate incorrect reasoning traces from reasoner
    print("Generating responses from reasoner to find incorrect reasoning traces...")
    traces_to_label = []
    for i in tqdm(range(0, len(sampled), cfg.training.per_device_batch_size)):
        chunk = sampled[i : i + cfg.training.per_device_batch_size]
        prompts = [reasoner_prompt(ex.question, reasoner.tokenizer) for ex in chunk]
        completions = generate_text(
            reasoner.model,
            reasoner.tokenizer,
            prompts,
            max_new_tokens=cfg.training.max_reasoner_tokens,
        )

        for ex, comp in zip(chunk, completions):
            think, answer = extract_think_answer(comp)
            # Only add generated trace if it led to an incorrect answer (negative example)
            if exact_match(answer, ex.answer) == 0:
                if think and think.strip():
                    traces_to_label.append((ex.question, think))

    # Free memory
    del reasoner
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Generated {len(traces_to_label)} negative traces")

    if len(traces_to_label) == 0:
        print("Warning: No incorrect reasoning traces generated. File will be empty.")

    # Label with OpenAI
    print("Labeling negative traces with OpenAI...")
    rows = []
    for q, trace in tqdm(traces_to_label, desc="Labeling slices"):
        slices = segment_reasoning(trace, tokenizer, slice_cfg)
        for s in slices:
            lbl = label_slice_with_openai(client, q, s, model=args.openai_model)
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

    # Write negative examples
    output_path = Path("data/discriminator_sft_negative.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(rows)} labeled negative slices to {output_path}")


if __name__ == "__main__":
    main()
