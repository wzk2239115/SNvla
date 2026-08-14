"""Fast Action Head: parallel multi-dimensional action prediction (cerebellum).

Locked spec (plan.md v3 §2.3):
  - d_model = 1024
  - keyboard:  256 sigmoid (multi-label, masked weighted BCE)
  - mouse X/Y: log-bucket classification + residual regression
  - buttons:   5 sigmoid (none = all-zero)
  - wheel:     3-class softmax (none/up/down)
  - auxiliary: press_event / release_event heads (training only)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ActionOutput:
    """All action logits from a single forward pass."""
    kbd: torch.Tensor           # [B, 256] raw logits → sigmoid
    mdx_bucket: torch.Tensor    # [B, N_BUCKETS]
    mdx_resid: torch.Tensor     # [B, 1]
    mdy_bucket: torch.Tensor    # [B, N_BUCKETS]
    mdy_resid: torch.Tensor     # [B, 1]
    btn: torch.Tensor           # [B, 5] raw logits → sigmoid
    wheel: torch.Tensor         # [B, 3]
    press: torch.Tensor         # [B, 256] auxiliary
    release: torch.Tensor       # [B, 256] auxiliary


class FastActionHead(nn.Module):
    """Parallel action prediction head. Single forward → all dimensions."""

    def __init__(
        self,
        d_model: int = 1024,
        n_keys: int = 256,
        n_mouse_buckets: int = 64,
        n_buttons: int = 5,
        n_wheel: int = 3,
    ):
        super().__init__()
        self.n_keys = n_keys
        self.n_mouse_buckets = n_mouse_buckets
        self.n_buttons = n_buttons
        self.n_wheel = n_wheel

        # keyboard: full 256 VK output (execution layer does whitelist mask)
        self.kbd_head = nn.Linear(d_model, n_keys)

        # mouse: bucket classification + residual refinement
        self.mdx_bucket_head = nn.Linear(d_model, n_mouse_buckets)
        self.mdx_resid_head = nn.Linear(d_model, 1)
        self.mdy_bucket_head = nn.Linear(d_model, n_mouse_buckets)
        self.mdy_resid_head = nn.Linear(d_model, 1)

        # mouse buttons: 5-dimensional multi-label (none = all zeros)
        self.btn_head = nn.Linear(d_model, n_buttons)

        # wheel: 3-class (none/up/down)
        self.wheel_head = nn.Linear(d_model, n_wheel)

        # auxiliary: onset/release timing (training only)
        self.press_event_head = nn.Linear(d_model, n_keys)
        self.release_event_head = nn.Linear(d_model, n_keys)

    def forward(self, h_t: torch.Tensor) -> ActionOutput:
        """h_t: [B, D] → ActionOutput with all logits."""
        return ActionOutput(
            kbd=self.kbd_head(h_t),
            mdx_bucket=self.mdx_bucket_head(h_t),
            mdx_resid=self.mdx_resid_head(h_t),
            mdy_bucket=self.mdy_bucket_head(h_t),
            mdy_resid=self.mdy_resid_head(h_t),
            btn=self.btn_head(h_t),
            wheel=self.wheel_head(h_t),
            press=self.press_event_head(h_t),
            release=self.release_event_head(h_t),
        )


def fast_action_loss(
    pred: ActionOutput,
    target_kbd: torch.Tensor,         # [B, 256]
    target_mdx_bucket: torch.Tensor,  # [B]
    target_mdx_resid: torch.Tensor,   # [B]
    target_mdy_bucket: torch.Tensor,  # [B]
    target_mdy_resid: torch.Tensor,   # [B]
    target_btn: torch.Tensor,         # [B, 5]
    target_wheel: torch.Tensor,       # [B]
    target_press: torch.Tensor,       # [B, 256]
    target_release: torch.Tensor,     # [B, 256]
    valid_key_mask: torch.Tensor | None = None,   # [256] bool
    key_pos_weight: torch.Tensor | None = None,   # [256] float
    mouse_resid_weight: float = 0.25,
    aux_weight: float = 0.3,
) -> dict[str, torch.Tensor]:
    """Compute all action losses. Returns dict of individual + total.

    Locked spec (plan.md v3 §5.1):
      keyboard: masked weighted BCE (per-key pos_weight + valid_key_mask)
      mouse:    bucket CE + 0.25 * residual SmoothL1
      buttons:  sigmoid BCE
      wheel:    3-class CE
      aux:      0.3 * press/release BCE
    """
    losses = {}

    # === keyboard: masked weighted BCE ===
    kbd_loss = F.binary_cross_entropy_with_logits(
        pred.kbd, target_kbd,
        pos_weight=key_pos_weight,
        reduction="none",
    )  # [B, 256]
    if valid_key_mask is not None:
        mask = valid_key_mask.unsqueeze(0).float()  # [1, 256]
        kbd_loss = (kbd_loss * mask).sum() / mask.sum().clamp(min=1)
    else:
        kbd_loss = kbd_loss.mean()
    losses["kbd"] = kbd_loss

    # === mouse: bucket CE + residual SmoothL1 ===
    losses["mdx_bucket"] = F.cross_entropy(pred.mdx_bucket, target_mdx_bucket)
    losses["mdy_bucket"] = F.cross_entropy(pred.mdy_bucket, target_mdy_bucket)
    losses["mdx_resid"] = F.smooth_l1_loss(pred.mdx_resid.squeeze(-1), target_mdx_resid)
    losses["mdy_resid"] = F.smooth_l1_loss(pred.mdy_resid.squeeze(-1), target_mdy_resid)
    losses["mouse"] = (
        losses["mdx_bucket"] + losses["mdy_bucket"]
        + mouse_resid_weight * (losses["mdx_resid"] + losses["mdy_resid"])
    )

    # === buttons: 5-dim sigmoid BCE ===
    losses["btn"] = F.binary_cross_entropy_with_logits(pred.btn, target_btn)

    # === wheel: 3-class CE ===
    losses["wheel"] = F.cross_entropy(pred.wheel, target_wheel)

    # === auxiliary: press/release event heads ===
    losses["press"] = F.binary_cross_entropy_with_logits(pred.press, target_press)
    losses["release"] = F.binary_cross_entropy_with_logits(pred.release, target_release)
    losses["aux"] = aux_weight * (losses["press"] + losses["release"])

    # === total ===
    losses["total"] = (
        losses["kbd"] + losses["mouse"] + losses["btn"]
        + losses["wheel"] + losses["aux"]
    )
    return losses
