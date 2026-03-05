from __future__ import annotations

import torch


def reasoner_reward(rm: torch.Tensor, rs: torch.Tensor, lambda1: float, lambda2: float) -> torch.Tensor:
    return lambda1 * rm + lambda2 * rs


def group_relative_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # rewards shape: [batch, group]
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True).clamp_min(eps)
    return (rewards - mean) / std


def discriminator_bce_loss(
    prob_real_on_ref: torch.Tensor,
    prob_real_on_gen: torch.Tensor,
    prob_align: torch.Tensor,
    align_target: torch.Tensor,
    lambda3: float,
    lambda4: float,
) -> torch.Tensor:
    eps = 1e-8
    rd = torch.log(prob_real_on_ref.clamp_min(eps)).mean() + torch.log((1 - prob_real_on_gen).clamp_min(eps)).mean()

    bce = -(
        align_target * torch.log(prob_align.clamp_min(eps))
        + (1 - align_target) * torch.log((1 - prob_align).clamp_min(eps))
    ).mean()

    return -lambda3 * rd + lambda4 * bce
