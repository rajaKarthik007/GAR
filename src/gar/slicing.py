from __future__ import annotations

import re
from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase


@dataclass
class SliceConfig:
    max_slice_tokens: int = 320
    min_slice_tokens: int = 48
    semantic_break_markers: tuple[str, ...] = ("Therefore", "Thus", "Hence", "So", "Next", "Finally")


def _count_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _initial_segments(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n|(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]


def segment_reasoning(text: str, tokenizer: PreTrainedTokenizerBase, cfg: SliceConfig) -> list[str]:
    segments = _initial_segments(text)
    if not segments:
        return [text.strip()] if text.strip() else []

    slices: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            slices.append("\n".join(buffer).strip())
            buffer.clear()

    for seg in segments:
        candidate = "\n".join(buffer + [seg]).strip()
        tok_count = _count_tokens(tokenizer, candidate)
        has_semantic_reset = any(seg.startswith(m) for m in cfg.semantic_break_markers)

        if buffer and (tok_count > cfg.max_slice_tokens or (has_semantic_reset and tok_count >= cfg.min_slice_tokens)):
            flush()
            buffer.append(seg)
        else:
            buffer.append(seg)

    flush()
    return slices
