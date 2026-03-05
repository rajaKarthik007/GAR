#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm

from gar.config import load_config
from gar.modeling import generate_text, load_causal_lm
from gar.parsing import exact_match, extract_think_answer
from gar.prompts import reasoner_prompt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--max_tokens", type=int, default=None,
                   help="Override max_new_tokens for evaluation (paper uses 32768).")
    p.add_argument("--num_samples", type=int, default=1,
                   help="Number of samples per question for Pass@1 (paper uses 30).")
    return p.parse_args()


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    model_name = args.model or cfg.models.reasoner_output_dir
    bundle = load_causal_lm(
        model_name,
        dtype=cfg.runtime.dtype,
        device=cfg.runtime.device,
        use_mps_if_available=cfg.runtime.use_mps_if_available,
        trust_remote_code=cfg.runtime.trust_remote_code,
    )
    model, tokenizer = bundle.model, bundle.tokenizer

    max_tokens = args.max_tokens or cfg.training.max_reasoner_tokens
    num_samples = args.num_samples

    rows = load_jsonl(args.input)
    correct = 0
    total = 0

    for row in tqdm(rows, desc="Evaluating"):
        question = row["question"]
        gold = row["answer"]
        prompt = reasoner_prompt(question, tokenizer)

        sample_correct = 0
        for _ in range(num_samples):
            out = generate_text(
                model,
                tokenizer,
                [prompt],
                max_new_tokens=max_tokens,
                temperature=0.6,
                top_p=0.95,
            )[0]
            _, answer = extract_think_answer(out)
            sample_correct += exact_match(answer, gold)

        correct += sample_correct
        total += num_samples

    acc = correct / max(1, total)
    print(f"Pass@1: {acc:.4f} ({correct}/{total})")
    if num_samples > 1:
        print(f"  (averaged over {num_samples} samples per question, {len(rows)} questions)")


if __name__ == "__main__":
    main()
