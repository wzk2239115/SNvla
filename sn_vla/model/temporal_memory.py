"""Temporal Memory: Mamba SSM (preferred) with GRU fallback.

Locked spec (plan.md v3 §2.2):
  - d_model = 1024
  - layers = 2 (Phase 1) → 4 (Phase 2+)
  - Input sequence = [vision_patches..., prev_action_embed, ACT_TICK]
  - Output = ACT_TICK position hidden state = h_t [B, 1024]

The ACT_TICK token solves the variable-patch aggregation problem:
  regardless of how many codec patches arrive (5 or 50), the Mamba output
  at the ACT_TICK position serves as the per-frame action state.

Including prev_action reduces action jitter (the controller knows what it
pressed last tick).
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from mamba_ssm.modules.mamba_simple import Mamba as _MambaBlock
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


def _make_ssm_layer(d_model: int, layer_idx: int) -> nn.Module:
    """Create one SSM layer. Uses Mamba if available, else GRU fallback."""
    if HAS_MAMBA:
        return _MambaBlock(
            d_model=d_model,
            d_state=64,
            d_conv=4,
            expand=2,
            layer_idx=layer_idx,
        )
    else:
        return _GRUFallback(d_model)


class _GRUFallback(nn.Module):
    """GRU fallback when mamba_ssm is unavailable.

    Provides the same interface as Mamba: takes [B, L, D] and returns
    [B, L, D]. Maintains hidden state internally for incremental use.
    Not as expressive as SSM but sufficient for development/testing.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, inference_params=None):
        out, _ = self.gru(x)
        return self.norm(out), x  # (hidden, residual) to match Mamba interface


class TemporalMemory(nn.Module):
    """Incremental temporal memory over codec visual features.

    Processes a sequence of [vision_patches + prev_action + ACT_TICK] through
    SSM layers and returns the ACT_TICK position as the per-frame state h_t.
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_layers: int = 2,
        n_keys: int = 256,
        n_mouse_buckets: int = 64,
        n_buttons: int = 5,
        n_wheel: int = 3,
    ):
        super().__init__()
        self.d_model = d_model

        # [ACT_TICK] learned embedding — appended to every frame's sequence
        self.act_tick = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Previous action embedding (single token summarizing last tick's action)
        self.prev_kbd_embed = nn.Linear(n_keys, d_model)
        self.prev_mdx_embed = nn.Embedding(n_mouse_buckets, d_model)
        self.prev_mdy_embed = nn.Embedding(n_mouse_buckets, d_model)
        self.prev_btn_embed = nn.Linear(n_buttons, d_model)
        self.prev_wheel_embed = nn.Embedding(n_wheel, d_model)

        # SSM layers
        self.ssms = nn.ModuleList([
            _make_ssm_layer(d_model, i) for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def _encode_prev_action(
        self,
        prev_kbd: torch.Tensor,       # [B, 256]
        prev_mdx_bucket: torch.Tensor, # [B]
        prev_mdy_bucket: torch.Tensor, # [B]
        prev_btn: torch.Tensor,        # [B, 5]
        prev_wheel: torch.Tensor,      # [B]
    ) -> torch.Tensor:
        """Encode previous action state into a single [B, D] token."""
        return (
            self.prev_kbd_embed(prev_kbd)
            + self.prev_mdx_embed(prev_mdx_bucket)
            + self.prev_mdy_embed(prev_mdy_bucket)
            + self.prev_btn_embed(prev_btn)
            + self.prev_wheel_embed(prev_wheel)
        )

    def forward(
        self,
        vision_features: torch.Tensor,  # [B, P, D]
        prev_kbd: torch.Tensor,          # [B, 256]
        prev_mdx_bucket: torch.Tensor,   # [B]
        prev_mdy_bucket: torch.Tensor,   # [B]
        prev_btn: torch.Tensor,          # [B, 5]
        prev_wheel: torch.Tensor,        # [B]
        canvas_mask: torch.Tensor | None = None,  # [B, P] bool, True = valid
    ) -> torch.Tensor:
        """Run temporal memory for one tick. Returns h_t [B, D].

        Args:
            vision_features: codec patch features for this tick [B, P, D].
            prev_*: previous tick's action state.
            canvas_mask: optional validity mask for padding.
        """
        B = vision_features.shape[0]

        # Encode previous action
        prev_emb = self._encode_prev_action(
            prev_kbd, prev_mdx_bucket, prev_mdy_bucket, prev_btn, prev_wheel
        )  # [B, D]
        prev_emb = prev_emb.unsqueeze(1)  # [B, 1, D]

        # ACT_TICK token
        tick = self.act_tick.expand(B, 1, -1)  # [B, 1, D]

        # Build sequence: [vision_patches, prev_action, ACT_TICK]
        if canvas_mask is not None:
            # Mask out padding patches by zeroing
            mask = canvas_mask.unsqueeze(-1).float()  # [B, P, 1]
            vision_features = vision_features * mask

        seq = torch.cat([vision_features, prev_emb, tick], dim=1)  # [B, P+2, D]

        # Run SSM layers
        x = seq
        for ssm in self.ssms:
            result = ssm(x)
            if HAS_MAMBA:
                x_new, residual = result
                x = x_new + (residual if residual is not None else 0)
            else:
                x_new, residual = result
                x = x_new

        x = self.norm(x)

        # Extract ACT_TICK position (last token) as h_t
        h_t = x[:, -1, :]  # [B, D]
        return h_t
