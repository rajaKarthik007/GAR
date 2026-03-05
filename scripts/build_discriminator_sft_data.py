#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from gar.config import load_config
from gar.data import load_math_dataset, sample_ratio, write_jsonl
from gar.openai_labeler import label_slice_with_openai
from gar.prompts import DISCRIMINATOR_SYSTEM_PROMPT, discriminator_slice_prompt
from gar.slicing import SliceConfig, segment_reasoning
from gar.utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--openai_model", type=str, default="gpt-o4-mini")
    p.add_argument(
        "--api_base",
        type=str,
        default=None,
        help="Optional OpenAI-compatible base URL (e.g., https://chat-api.tamu.ai/api).",
    )
    p.add_argument(
        "--api_key_env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key.",
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
    rows = []

    for ex in tqdm(sampled, desc="Labeling slices"):
        # Local/debug robustness: many datasets store reasoning in the solution/answer field.
        trace = ex.reasoning if ex.reasoning and ex.reasoning.strip() else ex.answer
        if not trace or not trace.strip():
            continue
        slices = segment_reasoning(trace, tokenizer, slice_cfg)
        for s in slices:
            lbl = label_slice_with_openai(client, ex.question, s, model=args.openai_model)
            target = f"{lbl.analysis}\n\n{'**YES**' if lbl.yes_no else '**NO**'}\n\n{lbl.rationale}".strip()
            rows.append(
                {
                    "question": ex.question,
                    "slice": s,
                    "label": lbl.yes_no,
                    "prompt": discriminator_slice_prompt(ex.question, s, tokenizer),
                    "target": target,
                }
            )

    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    if len(rows) == 0:
        raise RuntimeError(
            "No labeled slices were created. Check dataset access, reasoning field mapping, and OPENAI_API_KEY."
        )

    n = min(len(by_label[0]), len(by_label[1]))
    if n == 0:
        # FULL_SCALE_REVERT: for strict paper replication, keep a 1:1 class ratio.
        print(
            "Warning: could not build 1:1 YES/NO balance (one class is missing). "
            "Proceeding with unbalanced local SFT data."
        )
        balanced = rows
    else:
        balanced = random.sample(by_label[0], n) + random.sample(by_label[1], n)
        random.shuffle(balanced)

    write_jsonl(cfg.data.sft_output_path, balanced)
    print(f"Wrote {len(balanced)} rows to {cfg.data.sft_output_path}")


if __name__ == "__main__":
    main()
