"""Replan Gate: lightweight MLP on shared h_t.

Locked spec (plan.md v3 §2.4):
  Does NOT reuse full StreamMindGate. The TemporalMemory's h_t already encodes
  long-range temporal state, so the gate only needs a small MLP.

  LayerNorm(1024) → Linear(1024, 256) → GELU → Linear(256, 2)
  Output: continue / replan logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ReplanGate(nn.Module):
    """Lightweight replan gate operating on shared temporal memory output."""

    def __init__(self, d_model: int = 1024, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """h_t: [B, D] → [B, 2] continue/replan logits."""
        return self.net(h_t)

    @staticmethod
    def compute_loss(
        gate_logits: torch.Tensor,
        replan_labels: torch.Tensor,
        weight: tuple[float, float] = (0.7, 0.3),
    ) -> torch.Tensor:
        """Class-weighted CE for continue/replan."""
        w = torch.tensor(weight, device=gate_logits.device, dtype=gate_logits.dtype)
        return nn.functional.cross_entropy(gate_logits, replan_labels, weight=w)
