from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


@dataclass
class MathExample:
    question: str
    answer: str
    reasoning: str | None = None


def _pick(record: dict[str, Any], keys: list[str]) -> str | None:
    for k in keys:
        if k in record and record[k] is not None and str(record[k]).strip():
            return str(record[k])
    return None


def load_math_dataset(
    dataset_name: str | None,
    dataset_config: str | None,
    split: str,
    local_jsonl: str | None,
    question_keys: list[str],
    answer_keys: list[str],
    reasoning_keys: list[str],
    max_examples: int | None = None,
) -> list[MathExample]:
    rows: list[dict[str, Any]]

    if local_jsonl:
        rows = [json.loads(line) for line in Path(local_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        ds = load_dataset(dataset_name, dataset_config, split=split)
        rows = [dict(r) for r in ds]

    examples: list[MathExample] = []
    for r in rows:
        q = _pick(r, question_keys)
        a = _pick(r, answer_keys)
        c = _pick(r, reasoning_keys)
        if q and a:
            examples.append(MathExample(question=q, answer=a, reasoning=c))

    if max_examples is not None:
        return examples[:max_examples]
    return examples


def sample_ratio(items: list[MathExample], ratio: float, seed: int) -> list[MathExample]:
    n = max(1, int(len(items) * ratio))
    rng = random.Random(seed)
    return rng.sample(items, n)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def jsonl_to_dataset(path: str | Path) -> Dataset:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return Dataset.from_list(rows)
