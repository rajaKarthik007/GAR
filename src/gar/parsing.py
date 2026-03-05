from __future__ import annotations

import re

THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
YES_RE = re.compile(r"\*\*\s*YES\s*\*\*|\bYES\b", re.IGNORECASE)
NO_RE = re.compile(r"\*\*\s*NO\s*\*\*|\bNO\b", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def extract_think_answer(text: str) -> tuple[str, str]:
    think_match = THINK_RE.search(text)
    answer_match = ANSWER_RE.search(text)
    think = think_match.group(1).strip() if think_match else text.strip()
    answer = answer_match.group(1).strip() if answer_match else text.strip().splitlines()[-1].strip()
    return think, answer


def parse_yes_no(text: str) -> int:
    yes = YES_RE.search(text)
    no = NO_RE.search(text)
    if yes and not no:
        return 1
    if no and not yes:
        return 0
    if yes and no:
        return int(yes.start() < no.start())
    return 0


def canonicalize_math_answer(text: str) -> str:
    text = text.strip()
    boxed = BOXED_RE.search(text)
    if boxed:
        text = boxed.group(1)
    text = text.strip().strip(".")
    text = re.sub(r"\s+", "", text)
    return text.lower()


def exact_match(pred: str, gold: str) -> int:
    return int(canonicalize_math_answer(pred) == canonicalize_math_answer(gold))
