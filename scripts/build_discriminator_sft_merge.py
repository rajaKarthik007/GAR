#!/usr/bin/env python
"""
Stage 3: Merge positive and negative samples, balance 1:1, and create final SFT dataset.
Run on login node after stages 1 and 2 complete.

Usage:
  python scripts/build_discriminator_sft_merge.py
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from gar.utils import set_seed


def main() -> None:
    # Load positive and negative examples
    positive_path = Path("data/discriminator_sft_positive.jsonl")
    negative_path = Path("data/discriminator_sft_negative.jsonl")

    if not positive_path.exists():
        raise RuntimeError(f"Positive examples not found: {positive_path}")

    print("Loading positive examples...")
    positive = []
    with open(positive_path) as f:
        for line in f:
            positive.append(json.loads(line))

    print(f"Loaded {len(positive)} positive slices")

    # Negative examples are optional (may be empty if reasoner doesn't generate many wrong answers)
    negative = []
    if negative_path.exists():
        print("Loading negative examples...")
        with open(negative_path) as f:
            for line in f:
                negative.append(json.loads(line))
        print(f"Loaded {len(negative)} negative slices")
    else:
        print(f"Negative examples not found: {negative_path} (skipping)")

    # Combine and balance
    all_rows = positive + negative
    by_label: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in all_rows:
        by_label[int(r["label"])].append(r)

    print(f"Label distribution before balancing:")
    print(f"  YES (label=1): {len(by_label[1])}")
    print(f"  NO (label=0): {len(by_label[0])}")

    if len(all_rows) == 0:
        raise RuntimeError("No labeled slices found in positive or negative files.")

    n = min(len(by_label[0]), len(by_label[1]))
    if n == 0:
        print("Warning: Could not build 1:1 YES/NO balance (one class is missing).")
        balanced = all_rows
    else:
        print(f"Balancing to {n} examples per class...")
        balanced = random.sample(by_label[0], n) + random.sample(by_label[1], n)
        random.shuffle(balanced)

    # Write final merged dataset
    output_path = Path("data/discriminator_sft.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in balanced:
            f.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(balanced)} balanced rows to {output_path}")
    print(f"Label distribution after balancing:")
    balanced_labels = [int(r["label"]) for r in balanced]
    print(f"  YES (label=1): {sum(balanced_labels)}")
    print(f"  NO (label=0): {len(balanced_labels) - sum(balanced_labels)}")


if __name__ == "__main__":
    # Use a fixed seed for reproducible balancing
    set_seed(42)
    main()
