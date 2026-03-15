#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from tqdm import tqdm

# Allow MPS to use all available memory (prevents premature OOM on Apple Silicon).
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

from gar.config import load_config
from gar.data import load_math_dataset
from gar.modeling import generate_text, generated_logprobs, load_causal_lm, save_model, yes_probability
from gar.parsing import exact_match, extract_think_answer, parse_yes_no
from gar.prompts import (
    discriminator_alignment_prompt,
    discriminator_real_fake_prompt,
    discriminator_slice_prompt,
    reasoner_prompt,
)
from gar.rewards import discriminator_bce_loss, group_relative_advantages, reasoner_reward
from gar.slicing import SliceConfig, segment_reasoning
from gar.utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()


def cosine_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


from typing import Any, List
def chunked(xs: List[Any], n: int) -> List[List[Any]]:
    return [xs[i : i + n] for i in range(0, len(xs), n)]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # --- Distributed setup ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = (local_rank == 0)

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    # Use a different seed per process so each GPU samples different batches.
    set_seed(cfg.seed + local_rank)

    reasoner_bundle = load_causal_lm(
        cfg.models.reasoner_name,
        dtype=cfg.runtime.dtype,
        device=cfg.runtime.device,
        use_mps_if_available=cfg.runtime.use_mps_if_available,
        trust_remote_code=cfg.runtime.trust_remote_code,
    )
    disc_ckpt = cfg.models.discriminator_output_dir if Path(cfg.models.discriminator_output_dir).exists() else cfg.models.discriminator_name
    discriminator_bundle = load_causal_lm(
        disc_ckpt,
        dtype=cfg.runtime.dtype,
        device=cfg.runtime.device,
        use_mps_if_available=cfg.runtime.use_mps_if_available,
        trust_remote_code=cfg.runtime.trust_remote_code,
    )

    reasoner_tok = reasoner_bundle.tokenizer
    discr_tok = discriminator_bundle.tokenizer

    # Keep references to the raw models for generation and checkpoint saving.
    raw_reasoner = reasoner_bundle.model
    raw_discriminator = discriminator_bundle.model

    # Load frozen reference reasoner for GRPO KL penalty computation
    ref_reasoner_bundle = load_causal_lm(
        cfg.models.reasoner_name,
        dtype=cfg.runtime.dtype,
        device=cfg.runtime.device,
        use_mps_if_available=cfg.runtime.use_mps_if_available,
        trust_remote_code=cfg.runtime.trust_remote_code,
    )
    ref_reasoner = ref_reasoner_bundle.model
    ref_reasoner.eval()
    ref_reasoner.requires_grad_(False)

    # Enable gradient checkpointing to reduce peak memory on MPS and CUDA.
    if raw_reasoner.device.type in ("mps", "cuda"):
        raw_reasoner.gradient_checkpointing_enable()
        raw_discriminator.gradient_checkpointing_enable()

    # Wrap in DDP for multi-GPU training; fall back to plain models for single-GPU.
    if world_size > 1:
        reasoner = DDP(raw_reasoner, device_ids=[local_rank], find_unused_parameters=True)
        discriminator = DDP(raw_discriminator, device_ids=[local_rank], find_unused_parameters=True)
    else:
        reasoner = raw_reasoner
        discriminator = raw_discriminator

    reasoner.train()
    discriminator.train()

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
    if not data:
        raise RuntimeError("No training data loaded.")

    slice_cfg = SliceConfig(
        max_slice_tokens=cfg.slicing.max_slice_tokens,
        min_slice_tokens=cfg.slicing.min_slice_tokens,
        semantic_break_markers=tuple(cfg.slicing.semantic_break_markers),
    )

    opt_r = AdamW(raw_reasoner.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    opt_d = AdamW(raw_discriminator.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    opt_r.zero_grad(set_to_none=True)
    opt_d.zero_grad(set_to_none=True)

    warmup_steps = int(cfg.training.max_steps * cfg.training.warmup_ratio)
    bsz = cfg.training.per_device_batch_size
    g = cfg.training.num_generations

    grad_accum = max(1, cfg.training.gradient_accumulation_steps)
    for step in tqdm(range(cfg.training.max_steps), desc="GAR training", disable=not is_main):
        lr = cosine_lr(
            step=step,
            total_steps=cfg.training.max_steps,
            warmup_steps=warmup_steps,
            max_lr=cfg.training.learning_rate,
            min_lr=cfg.training.min_learning_rate,
        )
        for pg in opt_r.param_groups:
            pg["lr"] = lr
        for pg in opt_d.param_groups:
            pg["lr"] = lr

        batch = random.sample(data, bsz)
        questions = [b.question for b in batch]
        gold_answers = [b.answer for b in batch]

        grouped_prompts: list[list[str]] = []
        grouped_completions: list[list[str]] = []
        grouped_rm: list[list[float]] = []
        grouped_rs: list[list[float]] = []
        grouped_gen_slices: list[list[str]] = []
        grouped_final_correct: list[list[int]] = []

        for q, gold in zip(questions, gold_answers):
            prompts = [reasoner_prompt(q, reasoner_tok)] * g
            # Use raw model for generation (DDP wrapper does not support .generate()).
            completions = generate_text(
                raw_reasoner,
                reasoner_tok,
                prompts,
                max_new_tokens=cfg.training.max_reasoner_tokens,
                temperature=0.6,
                top_p=0.95,
            )
            rm_group: list[float] = []
            rs_group: list[float] = []
            gen_slice_group: list[str] = []
            final_corr_group: list[int] = []

            for c in completions:
                think, answer = extract_think_answer(c)
                rm = float(exact_match(answer, gold))
                final_corr_group.append(int(rm > 0.5))

                slices = segment_reasoning(think, discr_tok, slice_cfg)
                if not slices:
                    slices = [think.strip() or c.strip()]
                gen_slice_group.extend(slices)

                judge_prompts = [discriminator_slice_prompt(q, s, discr_tok) for s in slices]
                judge_outputs = generate_text(
                    raw_discriminator,
                    discr_tok,
                    judge_prompts,
                    max_new_tokens=cfg.training.max_discriminator_tokens,
                    temperature=0.2,
                    top_p=0.95,
                )
                slice_scores = [parse_yes_no(o) for o in judge_outputs]
                rs = float(sum(slice_scores) / max(1, len(slice_scores)))

                rm_group.append(rm)
                rs_group.append(rs)

            grouped_prompts.append(prompts)
            grouped_completions.append(completions)
            grouped_rm.append(rm_group)
            grouped_rs.append(rs_group)
            grouped_gen_slices.append(gen_slice_group)
            grouped_final_correct.append(final_corr_group)

        # Free MPS cache after generation phase before gradient computation.
        if raw_reasoner.device.type == "mps":
            torch.mps.empty_cache()

        flat_prompts = [p for gp in grouped_prompts for p in gp]
        flat_completions = [c for gc in grouped_completions for c in gc]

        rewards = []
        for rm_list, rs_list in zip(grouped_rm, grouped_rs):
            r = reasoner_reward(
                torch.tensor(rm_list, dtype=torch.float32),
                torch.tensor(rs_list, dtype=torch.float32),
                cfg.reward.lambda1_exact_match,
                cfg.reward.lambda2_slice,
            )
            rewards.append(r)
        reward_t = torch.stack(rewards, dim=0).to(raw_reasoner.device)
        adv = group_relative_advantages(reward_t).reshape(-1)

        # Use DDP-wrapped reasoner so gradients are synced across GPUs on backward.
        logp = generated_logprobs(reasoner, reasoner_tok, flat_prompts, flat_completions)
        with torch.no_grad():
            ref_logp = generated_logprobs(ref_reasoner, reasoner_tok, flat_prompts, flat_completions)
            
        # Standard GRPO KL penalty approximation
        beta = 0.04
        kl_penalty = logp - ref_logp
        reasoner_loss = -(adv * logp - beta * kl_penalty).mean()

        ref_slices = []
        for ex in batch:
            if not ex.reasoning:
                continue
            ref_slices.extend(segment_reasoning(ex.reasoning, discr_tok, slice_cfg))

        gen_slices = [s for gs in grouped_gen_slices for s in gs]
        n_pair = min(len(ref_slices), len(gen_slices))
        if n_pair == 0:
            # DDP Deadlock Fix: We MUST call forward on the DDP wrapper even when loss is 0.
            dummy_prompt = discriminator_real_fake_prompt("dummy", discr_tok)
            prob_ref = yes_probability(discriminator, discr_tok, [dummy_prompt])
            discriminator_loss = 0.0 * prob_ref.sum()
        else:
            ref_subset = random.sample(ref_slices, n_pair)
            gen_subset = random.sample(gen_slices, n_pair)

            # Use DDP-wrapped discriminator so gradients are synced across GPUs on backward.
            prob_ref = yes_probability(
                discriminator,
                discr_tok,
                [discriminator_real_fake_prompt(s, discr_tok) for s in ref_subset],
            )
            prob_gen = yes_probability(
                discriminator,
                discr_tok,
                [discriminator_real_fake_prompt(s, discr_tok) for s in gen_subset],
            )

            align_prompts = []
            align_targets = []
            for sl_group, corr_group in zip(grouped_gen_slices, grouped_final_correct):
                for sl in sl_group:
                    target = float(sum(corr_group) / max(1, len(corr_group)) >= 0.5)
                    align_prompts.append(discriminator_alignment_prompt(sl, bool(target), discr_tok))
                    align_targets.append(target)

            if align_prompts:
                prob_align = yes_probability(discriminator, discr_tok, align_prompts)
                align_t = torch.tensor(align_targets, dtype=torch.float32, device=prob_align.device)
            else:
                prob_align = torch.full((1,), 0.5, dtype=torch.float32, device=prob_ref.device)
                align_t = torch.full((1,), 0.5, dtype=torch.float32, device=prob_ref.device)

            discriminator_loss = discriminator_bce_loss(
                prob_real_on_ref=prob_ref,
                prob_real_on_gen=prob_gen,
                prob_align=prob_align,
                align_target=align_t,
                lambda3=cfg.reward.lambda3_disc,
                lambda4=cfg.reward.lambda4_align,
            )

        loss = reasoner_loss + discriminator_loss
        scaled_loss = loss / grad_accum
        scaled_loss.backward()

        should_step = ((step + 1) % grad_accum == 0) or ((step + 1) == cfg.training.max_steps)
        if should_step:
            torch.nn.utils.clip_grad_norm_(raw_reasoner.parameters(), cfg.training.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(raw_discriminator.parameters(), cfg.training.max_grad_norm)

            opt_r.step()
            opt_d.step()
            opt_r.zero_grad(set_to_none=True)
            opt_d.zero_grad(set_to_none=True)

        if is_main and (step + 1) % cfg.logging.log_every == 0:
            print(
                f"step={step+1} lr={lr:.2e} total_loss={float(loss):.4f} "
                f"reasoner_loss={float(reasoner_loss):.4f} discriminator_loss={float(discriminator_loss):.4f}"
            )

        if is_main and (step + 1) % cfg.logging.save_every == 0:
            rdir = Path(cfg.models.reasoner_output_dir) / f"step_{step+1}"
            ddir = Path(cfg.models.discriminator_output_dir) / f"step_{step+1}"
            rdir.mkdir(parents=True, exist_ok=True)
            ddir.mkdir(parents=True, exist_ok=True)
            save_model(raw_reasoner, reasoner_tok, str(rdir))
            save_model(raw_discriminator, discr_tok, str(ddir))

    if is_main:
        Path(cfg.models.reasoner_output_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.models.discriminator_output_dir).mkdir(parents=True, exist_ok=True)
        save_model(raw_reasoner, reasoner_tok, cfg.models.reasoner_output_dir)
        save_model(raw_discriminator, discr_tok, cfg.models.discriminator_output_dir)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
