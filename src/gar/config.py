from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelsConfig:
    reasoner_name: str
    discriminator_name: str
    reasoner_output_dir: str
    discriminator_output_dir: str


@dataclass
class RuntimeConfig:
    # FULL_SCALE_REVERT: set profile=paper, device=auto, dtype=bfloat16 on CUDA/H100.
    profile: str
    device: str
    dtype: str
    trust_remote_code: bool
    use_mps_if_available: bool
    sft_use_fp16: bool
    sft_use_bf16: bool


@dataclass
class DataConfig:
    dataset_name: str | None
    dataset_config: str | None
    train_split: str
    eval_split: str
    local_jsonl: str | None
    question_key_candidates: list[str]
    answer_key_candidates: list[str]
    reasoning_key_candidates: list[str]
    sft_sample_ratio: float
    sft_output_path: str
    max_train_examples: int | None = None
    max_eval_examples: int | None = None


@dataclass
class SlicingConfig:
    max_slice_tokens: int
    min_slice_tokens: int
    semantic_break_markers: list[str]


@dataclass
class TrainConfig:
    max_reasoner_tokens: int
    max_discriminator_tokens: int
    num_generations: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    learning_rate: float
    min_learning_rate: float
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float


@dataclass
class SFTConfig:
    per_device_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    max_seq_length: int


@dataclass
class RewardConfig:
    lambda1_exact_match: float
    lambda2_slice: float
    lambda3_disc: float
    lambda4_align: float


@dataclass
class LoggingConfig:
    log_every: int
    save_every: int


@dataclass
class GARConfig:
    seed: int
    runtime: RuntimeConfig
    models: ModelsConfig
    data: DataConfig
    slicing: SlicingConfig
    training: TrainConfig
    sft: SFTConfig
    reward: RewardConfig
    logging: LoggingConfig


def _build(cls: type, raw: dict[str, Any]) -> Any:
    return cls(**raw)


def load_config(path: str | Path) -> GARConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return GARConfig(
        seed=raw["seed"],
        runtime=_build(RuntimeConfig, raw["runtime"]),
        models=_build(ModelsConfig, raw["models"]),
        data=_build(DataConfig, raw["data"]),
        slicing=_build(SlicingConfig, raw["slicing"]),
        training=_build(TrainConfig, raw["training"]),
        sft=_build(SFTConfig, raw["sft"]),
        reward=_build(RewardConfig, raw["reward"]),
        logging=_build(LoggingConfig, raw["logging"]),
    )
