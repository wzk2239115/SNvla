"""Intention conditioning via FiLM / gated residual.

Locked spec (plan.md v3 §2.5):
  - Brain generates intention text → take <|plan_end|> hidden state [2560]
  - 2560 → Linear → 1024 = plan_embed
  - FiLM: gamma, beta, gate = plan_proj(plan_embed).chunk(3)
  - h_cond = h_t + sigmoid(gate) * (gamma * rmsnorm(h_t) + beta)
  - NOT cross-attention (too expensive for 60Hz path).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class IntentionCondition(nn.Module):
    """FiLM conditioning: injects brain intention into the fast action path.

    Args:
        brain_dim: dimension of brain hidden state (Qwen3 = 2560).
        fast_dim: dimension of fast path (1024).
    """

    def __init__(self, brain_dim: int = 2560, fast_dim: int = 1024):
        super().__init__()
        self.proj_in = nn.Linear(brain_dim, fast_dim)
        # FiLM parameters: gamma, beta, gate (each fast_dim)
        self.film = nn.Linear(fast_dim, fast_dim * 3)
        self.norm = RMSNorm(fast_dim)

    def forward(self, h_t: torch.Tensor, brain_hidden: torch.Tensor | None) -> torch.Tensor:
        """
        Args:
            h_t: [B, fast_dim] temporal memory output.
            brain_hidden: [B, brain_dim] or None (when brain hasn't fired yet).

        Returns:
            h_cond: [B, fast_dim] conditioned state for fast action head.
        """
        if brain_hidden is None:
            return h_t

        plan_embed = self.proj_in(brain_hidden)  # [B, fast_dim]
        gamma, beta, gate = self.film(plan_embed).chunk(3, dim=-1)

        h_norm = self.norm(h_t)
        h_cond = h_t + torch.sigmoid(gate) * (gamma * h_norm + beta)
        return h_cond
