"""Phase 1: Fast BC PoC training.

Locked spec (plan.md v3 §7):
  - ViT frozen
  - Train TemporalMemory + FastActionHead
  - Loss: L_fast_action only
  - Data: D2E-480p (or sample), single game for PoC
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sn_vla.data.dataset import D2EDataset, collate_fn
from sn_vla.data.mouse_buckets import MouseBucketer
from sn_vla.model.sn_vla import SNVLA, SNVLAConfig
from sn_vla.model.fast_action_head import fast_action_loss


def compute_key_stats(dataset: D2EDataset) -> tuple[np.ndarray, np.ndarray]:
    """Compute valid_key_mask and pos_weight from the dataset.

    valid_key_mask: [256] bool — True for keys that appear at least once.
    key_pos_weight: [256] float — inverse frequency for positive class.
    """
    n_keys = 256
    pos_count = np.zeros(n_keys, dtype=np.float64)
    total = 0

    for arr_idx, idx in enumerate(dataset._episode_indices):
        pos_count += idx.kbd_multi_hot.sum(axis=0)
        total += idx.kbd_multi_hot.shape[0]

    valid_key_mask = pos_count > 0
    freq = pos_count / max(total, 1)
    # pos_weight = neg_count / pos_count per key (clamped)
    pos_weight = np.where(
        valid_key_mask,
        np.clip((1.0 - freq) / np.maximum(freq, 1e-6), 1.0, 50.0),
        0.0,
    ).astype(np.float32)

    return valid_key_mask, pos_weight


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # === Data ===
    bucketer = MouseBucketer(d_max=args.d_max, n_buckets=args.n_mouse_buckets)
    dataset = D2EDataset(
        manifest_path=args.manifest,
        codec_cache_dir=args.codec_cache,
        tick_hz=args.tick_hz,
        clip_frames=args.clip_frames,
        bucketer=bucketer,
    )
    print(f"Dataset: {len(dataset)} samples")

    valid_key_mask, key_pos_weight = compute_key_stats(dataset)
    n_valid_keys = valid_key_mask.sum()
    print(f"Valid keys: {n_valid_keys}/256")
    valid_vk = np.flatnonzero(valid_key_mask)
    if n_valid_keys > 0:
        for vk in valid_vk:
            print(f"  vk={vk:3d} (0x{vk:02X}): pos_weight={key_pos_weight[vk]:.2f}")

    valid_key_mask_t = torch.from_numpy(valid_key_mask).to(device)
    key_pos_weight_t = torch.from_numpy(key_pos_weight).to(device)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # === Model ===
    config = SNVLAConfig(
        n_mouse_buckets=args.n_mouse_buckets,
        n_memory_layers=args.n_memory_layers,
    )
    model = SNVLA(config)
    model.freeze_visual()
    model.to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,}")

    # === Optimizer ===    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(loader))

    # === Training loop ===
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []

        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(device)

            action, _ = model(
                visual=batch["visual"],
                prev_kbd=batch["prev_kbd"],
                prev_mdx_bucket=batch["prev_mdx_bucket"],
                prev_mdy_bucket=batch["prev_mdy_bucket"],
                prev_btn=batch["prev_btn"],
                prev_wheel=batch["prev_wheel"],
                canvas_mask=batch["canvas_mask"],
            )

            losses = fast_action_loss(
                action,
                target_kbd=batch["target_kbd"],
                target_mdx_bucket=batch["target_mdx_bucket"],
                target_mdx_resid=batch["target_mdx_residual"],
                target_mdy_bucket=batch["target_mdy_bucket"],
                target_mdy_resid=batch["target_mdy_residual"],
                target_btn=batch["target_btn"],
                target_wheel=batch["target_wheel"],
                target_press=batch["target_press"],
                target_release=batch["target_release"],
                valid_key_mask=valid_key_mask_t,
                key_pos_weight=key_pos_weight_t,
            )

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()

            epoch_losses.append(losses["total"].item())
            global_step += 1

            if global_step % args.log_every == 0:
                parts = {k: f"{v:.4f}" for k, v in losses.items() if k != "total"}
                print(f"  [ep{epoch} step{global_step}] total={losses['total']:.4f} {parts}")

        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint
        ckpt_path = output_dir / f"checkpoint_ep{epoch}.pt"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": avg_loss,
            "config": config.__dict__,
            "valid_key_mask": valid_key_mask,
            "key_pos_weight": key_pos_weight,
            "bucketer": {"d_max": bucketer.d_max, "n_buckets": bucketer.n_buckets},
        }, ckpt_path)
        print(f"  Saved {ckpt_path}")

    print("Training complete.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="Path to manifest JSON")
    ap.add_argument("--codec-cache", default=None, help="Codec canvas cache directory")
    ap.add_argument("--output-dir", default="checkpoints/phase1")
    ap.add_argument("--tick-hz", type=int, default=60)
    ap.add_argument("--clip-frames", type=int, default=64)
    ap.add_argument("--d-max", type=float, default=200.0)
    ap.add_argument("--n-mouse-buckets", type=int, default=64)
    ap.add_argument("--n-memory-layers", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
