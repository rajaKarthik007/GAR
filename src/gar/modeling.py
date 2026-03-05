from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class LMWithTokenizer:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase


def _unwrap(model: PreTrainedModel) -> PreTrainedModel:
    """Return the underlying model, unwrapping DistributedDataParallel if needed."""
    return model.module if hasattr(model, "module") else model  # type: ignore[return-value]


def _device(model: PreTrainedModel) -> torch.device:
    """Return the device of a model, handling DDP wrappers."""
    return _unwrap(model).device


def resolve_device(requested: str = "auto", use_mps_if_available: bool = True) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if use_mps_if_available and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    # MPS is prone to NaN/Inf sampling instability with fp16/bf16 on some decoder models.
    # FULL_SCALE_REVERT: CUDA/H100 setups can keep fp16/bf16.
    if device.type == "mps":
        return torch.float32

    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        # FULL_SCALE_REVERT: bfloat16 is preferred on H100/CUDA paper setup.
        if device.type == "cuda":
            return torch.bfloat16
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_causal_lm(
    name: str,
    *,
    dtype: str = "bfloat16",
    device: str = "auto",
    use_mps_if_available: bool = True,
    trust_remote_code: bool = False,
) -> LMWithTokenizer:
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Decoder-only generation is more stable/correct with left padding.
    tok.padding_side = "left"

    resolved_device = resolve_device(device, use_mps_if_available=use_mps_if_available)
    torch_dtype = resolve_dtype(dtype, resolved_device)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            name,
            dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
    except TypeError:
        # Backward compatibility for transformers builds that still expect `torch_dtype`.
        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
    model.to(resolved_device)
    model.config.pad_token_id = tok.pad_token_id
    return LMWithTokenizer(model=model, tokenizer=tok)


@torch.no_grad()
def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float = 0.6,
    top_p: float = 0.95,
) -> list[str]:
    raw = _unwrap(model)
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    enc = {k: v.to(_device(model)) for k, v in enc.items()}

    try:
        out = raw.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    except RuntimeError as e:
        if "probability tensor contains either `inf`, `nan` or element < 0" not in str(e):
            raise
        # Fallback for numerically unstable sampling runs on MPS.
        out = raw.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = out[:, enc["input_ids"].shape[1] :]
    return tokenizer.batch_decode(gen, skip_special_tokens=True)


def generated_logprobs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    completions: list[str],
) -> torch.Tensor:
    texts = [p + c for p, c in zip(prompts, completions)]
    prompt_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]

    # MPS memory is tight: compute logprobs in micro-batches to avoid OOM.
    # FULL_SCALE_REVERT: on large CUDA machines, this can be processed in one batch.
    chunk_size = 1 if _device(model).type == "mps" else len(texts)
    outputs: list[torch.Tensor] = []

    for i in range(0, len(texts), chunk_size):
        chunk_texts = texts[i : i + chunk_size]
        chunk_prompt_lens = prompt_lens[i : i + chunk_size]

        enc = tokenizer(chunk_texts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(_device(model)) for k, v in enc.items()}

        with torch.set_grad_enabled(True):
            logits = model(**enc).logits[:, :-1, :]
            labels = enc["input_ids"][:, 1:]
            log_probs = torch.log_softmax(logits, dim=-1)
            token_logp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        max_len = token_logp.shape[1]
        mask = torch.zeros_like(token_logp)
        for j, plen in enumerate(chunk_prompt_lens):
            start = max(plen - 1, 0)
            if start < max_len:
                mask[j, start:] = 1.0

        summed = (token_logp * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        outputs.append(summed / denom)

    return torch.cat(outputs, dim=0)


def yes_probability(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
) -> torch.Tensor:
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    enc = {k: v.to(_device(model)) for k, v in enc.items()}

    logits = model(**enc).logits[:, -1, :]
    yes_id = tokenizer.encode(" YES", add_special_tokens=False)
    no_id = tokenizer.encode(" NO", add_special_tokens=False)
    if not yes_id or not no_id:
        raise ValueError("Tokenizer must encode ' YES' and ' NO'.")

    yes_logit = logits[:, yes_id[0]]
    no_logit = logits[:, no_id[0]]
    probs = torch.softmax(torch.stack([no_logit, yes_logit], dim=-1), dim=-1)[:, 1]
    return probs


def save_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, out_dir: str) -> None:
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
