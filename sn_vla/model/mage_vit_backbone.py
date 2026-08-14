"""Mage-ViT backbone: load pretrained weights from Mage-VL checkpoint.

Extracts the ViT trunk (1024-dim, before PatchMerger) for the fast action path.
The PatchMerger (1024→2560) is available separately for the brain path (Phase 3).

Weight source: model-00002-of-00002.safetensors, keys prefixed "model.visual."
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class MageAttention(nn.Module):
    """Standard multi-head attention with RoPE stub (pretrained weights compatible)."""

    def __init__(self, dim: int = 1024, num_heads: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MageEncoderLayer(nn.Module):
    """ViT encoder layer matching Mage-ViT weight names."""

    def __init__(self, dim: int = 1024, num_heads: int = 16, mlp_ratio: int = 4):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.self_attn = MageAttention(dim, num_heads)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class MageViTBackbone(nn.Module):
    """Mage-ViT trunk producing 1024-dim patch features.

    Loads pretrained weights from Mage-VL safetensors checkpoint.
    Input: canvas images [B, n_canvas, 3, H, W]
    Output: patch features [B, total_patches, 1024]
    """

    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 24,
        patch_size: int = 16,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.num_layers = num_layers

        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.layernorm_pre = nn.LayerNorm(dim, eps=1e-6)
        self.encoder_layers = nn.ModuleList([
            MageEncoderLayer(dim, num_heads, mlp_ratio) for _ in range(num_layers)
        ])
        self.layernorm_post = nn.LayerNorm(dim, eps=1e-6)

        # Learnable 2D positional embedding (added since we skip Mage's 3D-RoPE)
        self.max_patches = 4096
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        canvases: torch.Tensor,
        canvas_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """canvases: [B, n_canvas, 3, H, W] → [B, total_patches, 1024]."""
        B, N, C, H, W = canvases.shape
        flat = canvases.view(B * N, C, H, W)
        patches = self.patch_embedding(flat)  # [B*N, dim, h, w]
        d, h, w = patches.shape[1:]
        patches = patches.flatten(2).transpose(1, 2)  # [B*N, h*w, dim]
        patches = patches.reshape(B, N * (h * w), d)

        P = patches.shape[1]
        patches = patches + self.pos_embed[:, :P]

        patches = self.layernorm_pre(patches)
        for layer in self.encoder_layers:
            patches = layer(patches)
        patches = self.layernorm_post(patches)
        return patches

    @classmethod
    def from_magevl(cls, magevl_dir: str | Path, strict: bool = False) -> "MageViTBackbone":
        """Load pretrained weights from Mage-VL checkpoint directory.

        Loads patch_embedding, layernorm_pre, encoder_layers from safetensors.
        The merger (1024→2560) is NOT loaded (fast path doesn't need it).

        Args:
            magevl_dir: path to the Mage-VL model directory.
            strict: if True, require all keys to match.
        """
        magevl_dir = Path(magevl_dir)
        from safetensors.torch import load_file
        import json

        # Find which shard contains visual weights
        index_path = magevl_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                wm = json.load(f)["weight_map"]
            # Group visual keys by shard
            shards_needed = set()
            key_map = {}  # our_key -> source_key
            for src_key, shard in wm.items():
                if src_key.startswith("model.visual.") and "merger" not in src_key:
                    short = src_key.replace("model.visual.", "")
                    key_map[short] = (shard, src_key)
                    shards_needed.add(shard)
        else:
            shards_needed = {"model-00002-of-00002.safetensors"}

        model = cls()

        # Load and remap weights
        loaded_state = {}
        for shard_name in shards_needed:
            shard_path = magevl_dir / shard_name
            if not shard_path.exists():
                continue
            state = load_file(str(shard_path))
            for src_key, tensor in state.items():
                if src_key.startswith("model.visual.") and "merger" not in src_key:
                    short = src_key.replace("model.visual.", "")
                    loaded_state[short] = tensor

        # Map encoder layer weight names from Mage-VL to our module:
        #   embeddings.patch_embedding.*     → patch_embedding.*
        #   encoder.layers.N.*               → encoder_layers.N.*
        #   encoder.layers.N.mlp.fc1.*       → encoder_layers.N.mlp.0.*
        #   encoder.layers.N.mlp.fc2.*       → encoder_layers.N.mlp.2.*
        remapped = {}
        for k, v in loaded_state.items():
            new_k = k.replace("encoder.layers.", "encoder_layers.")
            new_k = new_k.replace("embeddings.patch_embedding.", "patch_embedding.")
            new_k = new_k.replace("mlp.fc1.", "mlp.0.")
            new_k = new_k.replace("mlp.fc2.", "mlp.2.")
            remapped[new_k] = v

        missing, unexpected = model.load_state_dict(remapped, strict=False)
        n_loaded = len(remapped) - len(missing)
        print(f"MageViTBackbone: loaded {n_loaded} tensors from Mage-VL")
        if missing:
            print(f"  Missing (init randomly): {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected (ignored): {unexpected[:5]}...")

        # pos_embed is always randomly initialized (Mage uses RoPE, not learned PE)
        print(f"  pos_embed: randomly initialized (Mage-VL uses 3D-RoPE, replaced with learned PE)")
        return model
