#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from gar.config import load_config
from gar.data import jsonl_to_dataset
from gar.modeling import resolve_device, resolve_dtype
from gar.utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.models.discriminator_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = resolve_device(cfg.runtime.device, use_mps_if_available=cfg.runtime.use_mps_if_available)
    torch_dtype = resolve_dtype(cfg.runtime.dtype, device)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.models.discriminator_name,
            dtype=torch_dtype,
            trust_remote_code=cfg.runtime.trust_remote_code,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.models.discriminator_name,
            torch_dtype=torch_dtype,
            trust_remote_code=cfg.runtime.trust_remote_code,
        )
    model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    if device.type == "cuda":
        model.gradient_checkpointing_enable()

    ds = jsonl_to_dataset(cfg.data.sft_output_path)
    if len(ds) == 0:
        raise RuntimeError(
            f"SFT dataset is empty at {cfg.data.sft_output_path}. "
            "Run build_discriminator_sft_data.py first and ensure it writes labeled rows."
        )

    def tok_fn(batch: dict) -> dict:
        texts = [p + t for p, t in zip(batch["prompt"], batch["target"])]
        # Tokenize prompt-only to find where the target begins.
        prompt_toks = tokenizer(
            batch["prompt"],
            truncation=True,
            max_length=cfg.sft.max_seq_length,
            add_special_tokens=False,
        )
        toks = tokenizer(
            texts,
            truncation=True,
            max_length=cfg.sft.max_seq_length,
            padding="max_length",
        )
        # Mask prompt tokens in labels so loss is only on the completion.
        labels = []
        for i, ids in enumerate(toks["input_ids"]):
            prompt_len = len(prompt_toks["input_ids"][i])
            label = list(ids)
            for j in range(min(prompt_len, len(label))):
                label[j] = -100
            # Also mask padding tokens.
            for j in range(len(label)):
                if ids[j] == tokenizer.pad_token_id:
                    label[j] = -100
            labels.append(label)
        toks["labels"] = labels
        return toks

    tokenized = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)
    if len(tokenized) == 0:
        raise RuntimeError(
            "Tokenized SFT dataset is empty. This usually means your input SFT JSONL has no valid rows."
        )

    args_train = TrainingArguments(
        output_dir=cfg.models.discriminator_output_dir,
        per_device_train_batch_size=cfg.sft.per_device_batch_size,
        gradient_accumulation_steps=cfg.sft.gradient_accumulation_steps,
        max_steps=cfg.sft.max_steps,
        learning_rate=cfg.sft.learning_rate,
        warmup_steps=cfg.sft.warmup_steps,
        weight_decay=cfg.sft.weight_decay,
        bf16=cfg.runtime.sft_use_bf16,
        fp16=cfg.runtime.sft_use_fp16,
        gradient_checkpointing=device.type == "cuda",
        logging_steps=cfg.logging.log_every,
        save_steps=cfg.logging.save_every,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=[],
    )

    # Compatibility: some transformers versions removed/changed the tokenizer init arg.
    trainer = Trainer(model=model, args=args_train, train_dataset=tokenized)
    trainer.train()
    trainer.save_model(cfg.models.discriminator_output_dir)
    tokenizer.save_pretrained(cfg.models.discriminator_output_dir)


if __name__ == "__main__":
    main()
