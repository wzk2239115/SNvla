"""SN-VLA: Screen-Native Vision-Language-Action Model.

Locked spec (plan.md v3 §0.2):
  Mage-ViT (eyes, 1024 dim) → Temporal Memory → Fast Action Head (cerebellum)
                                         ↕
                                Replan Gate + Qwen3 Brain (via FiLM)

Phase 1: ViT frozen + Memory + Fast Head only (no brain, no gate).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .temporal_memory import TemporalMemory
from .fast_action_head import FastActionHead, ActionOutput
from .replan_gate import ReplanGate
from .intention import IntentionCondition


@dataclass
class SNVLAConfig:
    d_vision: int = 1024
    d_fast: int = 1024
    d_brain: int = 2560
    n_memory_layers: int = 2
    n_keys: int = 256
    n_mouse_buckets: int = 64
    n_buttons: int = 5
    n_wheel: int = 3
    use_gate: bool = False      # Phase 3+
    use_brain: bool = False     # Phase 3+


class SimpleViT(nn.Module):
    """Lightweight ViT for development without Mage-ViT checkpoint.

    Takes canvas images [B, n_canvas, 3, H, W] and produces patch features
    [B, P, d_vision]. Used when the full Mage-ViT is unavailable.
    """

    def __init__(self, d_model: int = 1024, patch_size: int = 16, n_layers: int = 6):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, 4096, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            self._make_block(d_model) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def _make_block(self, d: int) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(d),
            nn.MultiheadAttention(d, num_heads=8, batch_first=True),
            nn.LayerNorm(d),
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d),
        )

    def forward(self, canvases: torch.Tensor, canvas_mask: torch.Tensor | None = None) -> torch.Tensor:
        """canvases: [B, n_canvas, 3, H, W] → [B, P, d_model]."""
        B, N, C, H, W = canvases.shape
        flat = canvases.view(B * N, C, H, W)
        patches = self.patch_embed(flat)  # [B*N, d, h, w]
        d, h, w = patches.shape[1:]
        patches = patches.flatten(2).transpose(1, 2)  # [B*N, h*w, d]

        # Add positional embedding
        P = patches.shape[1]
        patches = patches + self.pos_embed[:, :P]

        for blk in self.blocks:
            # Simple residual transformer block
            normed = blk[0](patches)
            attn_out, _ = blk[1](normed, normed, normed)
            patches = patches + attn_out
            normed2 = blk[2](patches)
            ff_out = blk[5](blk[4](blk[3](normed2)))
            patches = patches + ff_out

        patches = self.norm(patches)
        patches = patches.view(B, N * P, -1)  # [B, N*P, d]
        return patches


class SNVLA(nn.Module):
    """SN-VLA model: codec-native visual → temporal memory → fast action head.

    Phase 1 (default): ViT frozen + TemporalMemory + FastActionHead.
    Phase 3: add ReplanGate + IntentionCondition + Qwen3 Brain.
    """

    def __init__(self, config: SNVLAConfig, visual_encoder: nn.Module | None = None):
        super().__init__()
        self.config = config

        # === Eyes ===
        self.visual = visual_encoder or SimpleViT(d_model=config.d_vision)

        # === Cerebellum ===
        self.memory = TemporalMemory(
            d_model=config.d_fast,
            n_layers=config.n_memory_layers,
            n_keys=config.n_keys,
            n_mouse_buckets=config.n_mouse_buckets,
            n_buttons=config.n_buttons,
            n_wheel=config.n_wheel,
        )
        self.fast_head = FastActionHead(
            d_model=config.d_fast,
            n_keys=config.n_keys,
            n_mouse_buckets=config.n_mouse_buckets,
            n_buttons=config.n_buttons,
            n_wheel=config.n_wheel,
        )

        # === Gate (Phase 3+) ===
        self.gate = ReplanGate(d_model=config.d_fast) if config.use_gate else None

        # === Brain conditioning (Phase 3+) ===
        self.intention_cond = IntentionCondition(
            brain_dim=config.d_brain, fast_dim=config.d_fast
        ) if config.use_brain else None

    def forward(
        self,
        visual: torch.Tensor,            # [B, n_canvas, 3, H, W]
        prev_kbd: torch.Tensor,           # [B, 256]
        prev_mdx_bucket: torch.Tensor,    # [B]
        prev_mdy_bucket: torch.Tensor,    # [B]
        prev_btn: torch.Tensor,           # [B, 5]
        prev_wheel: torch.Tensor,         # [B]
        canvas_mask: torch.Tensor | None = None,
        brain_hidden: torch.Tensor | None = None,
    ) -> tuple[ActionOutput, torch.Tensor | None]:
        """Full forward pass for one tick.

        Returns:
            action: ActionOutput with all action logits.
            gate_logits: [B, 2] or None (if gate disabled).
        """
        # 1. Visual encoding → [B, P, d_vision]
        vis_feat = self.visual(visual, canvas_mask) if isinstance(self.visual, SimpleViT) \
            else self.visual(visual)

        # 2. Temporal memory → h_t [B, d_fast]
        h_t = self.memory(
            vis_feat,
            prev_kbd, prev_mdx_bucket, prev_mdy_bucket, prev_btn, prev_wheel,
            canvas_mask=canvas_mask,
        )

        # 3. Intention conditioning (Phase 3+)
        if self.intention_cond is not None:
            h_t = self.intention_cond(h_t, brain_hidden)

        # 4. Fast action (every tick)
        action = self.fast_head(h_t)

        # 5. Gate (Phase 3+)
        gate_logits = self.gate(h_t) if self.gate is not None else None

        return action, gate_logits

    @torch.no_grad()
    def freeze_visual(self):
        """Freeze visual encoder (Phase 1, 2a)."""
        for p in self.visual.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def unfreeze_visual_last_n(self, n: int):
        """Unfreeze last n layers of visual encoder (Phase 2b)."""
        self.freeze_visual()
        if isinstance(self.visual, SimpleViT):
            for blk in self.visual.blocks[-n:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.visual.norm.parameters():
                p.requires_grad = True

    @torch.no_grad()
    def unfreeze_visual_all(self):
        """Unfreeze entire visual encoder (Phase 2c)."""
        for p in self.visual.parameters():
            p.requires_grad = True
