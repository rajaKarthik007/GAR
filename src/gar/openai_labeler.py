from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from gar.parsing import parse_yes_no


@dataclass
class SliceLabel:
    analysis: str
    yes_no: int
    rationale: str
    raw: str


SYSTEM_PROMPT = (
    "You are an evaluator for math reasoning slices. "
    "Reply with exactly three sections: "
    "1) brief analysis, 2) verdict as **YES** or **NO**, 3) brief rationale."
)


def _extract_chat_text(chat_rsp: object) -> str:
    # Standard OpenAI SDK object.
    if hasattr(chat_rsp, "choices"):
        try:
            return (chat_rsp.choices[0].message.content or "").strip()  # type: ignore[attr-defined]
        except Exception:
            pass

    # Dict-like payload from OpenAI-compatible gateways.
    if isinstance(chat_rsp, dict):
        choices = chat_rsp.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message", {})
            content = msg.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                pieces: list[str] = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        pieces.append(str(item["text"]))
                return "\n".join(pieces).strip()
        for key in ("output_text", "text", "response", "content"):
            val = chat_rsp.get(key)
            if isinstance(val, str):
                return val.strip()

    # Some proxies return plain text.
    if isinstance(chat_rsp, str):
        return chat_rsp.strip()

    return str(chat_rsp).strip()


def label_slice_with_openai(client: OpenAI, question: str, slice_text: str, model: str = "gpt-o4-mini", debug: bool = False) -> SliceLabel:
    prompt = (
        f"Question:\n{question}\n\n"
        f"Reasoning slice:\n{slice_text}\n\n"
        "Assess logical soundness."
    )

    # Use chat.completions — universally supported by OpenAI and all compatible proxies
    # (the newer responses API is not reliably supported by all endpoints).
    chat_rsp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    raw = _extract_chat_text(chat_rsp)

    verdict = parse_yes_no(raw)
    if debug:
        print(f"[DEBUG] Slice: {slice_text[:50]}... -> Verdict: {verdict} -> Raw: {raw[:100]}...")

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    analysis = lines[0] if lines else ""
    rationale = lines[-1] if len(lines) > 1 else ""

    return SliceLabel(analysis=analysis, yes_no=verdict, rationale=rationale, raw=raw)
