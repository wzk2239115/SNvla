#!/usr/bin/env python3
"""Phase 1 training launch script for remote H100 machine.

Usage:
    python scripts/train_phase1.py \
        --d2e-dir /home/jovyan/exploitgym/D2E-Original \
        --magevl-dir /home/jovyan/exploitgym/Mage-VL \
        --output-dir /home/jovyan/exploitgym/SNvla/checkpoints/phase1 \
        [--max-episodes 50] [--games "Super Bunny Man"] [--epochs 20]

Steps:
    1. Explore D2E data structure
    2. Build manifest (+ B-frame check)
    3. Scan mouse/raw distribution → calibrate bucketer
    4. Train with pretrained Mage-ViT (frozen) + TemporalMemory + FastActionHead
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def explore_d2e(d2e_dir: Path, max_show: int = 30):
    """Print D2E directory structure summary."""
    print(f"\n{'='*60}")
    print(f"Exploring D2E data: {d2e_dir}")
    print(f"{'='*60}")

    games = sorted([d for d in d2e_dir.iterdir() if d.is_dir()])
    print(f"Found {len(games)} game directories")
    for g in games[:max_show]:
        mcaps = list(g.rglob("*.mcap"))
        mkvs = list(g.rglob("*.mkv"))
        print(f"  {g.name}: {len(mcaps)} mcap, {len(mkvs)} mkv")
    if len(games) > max_show:
        print(f"  ... and {len(games) - max_show} more")

    total_mcap = sum(1 for _ in d2e_dir.rglob("*.mcap"))
    total_mkv = sum(1 for _ in d2e_dir.rglob("*.mkv"))
    print(f"\nTotal: {total_mcap} mcap files, {total_mkv} mkv files")
    return games


def run_phase1(args):
    d2e_dir = Path(args.d2e_dir)
    magevl_dir = Path(args.magevl_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === 1. Explore data ===
    games = explore_d2e(d2e_dir)

    # Filter games if specified
    if args.games:
        game_filter = [g.strip() for g in args.games.split(",")]
        games = [g for g in games if g.name in game_filter]
        print(f"\nFiltered to {len(games)} games: {[g.name for g in games]}")

    if not games:
        print("ERROR: No games found matching filter")
        sys.exit(1)

    # === 2. Build manifest ===
    print(f"\n{'='*60}")
    print("Building manifest...")
    print(f"{'='*60}")
    from sn_vla.data.manifest import build_manifest

    manifest_path = output_dir / "manifest.json"

    # Create a temp dir with symlinks if filtering games
    if args.games:
        import tempfile, os
        tmp_dir = Path(tempfile.mkdtemp())
        for g in games:
            (tmp_dir / g.name).symlink_to(g.resolve())
        data_root = tmp_dir
    else:
        data_root = d2e_dir

    manifest = build_manifest(
        root_dir=data_root,
        output_path=manifest_path,
        tick_hz=args.tick_hz,
        window_frames=args.window_frames,
        check_causality=not args.skip_causality,
        skip_b_frame_episodes=True,
    )

    if manifest["n_episodes"] == 0:
        print("ERROR: No valid episodes found (all may have B-frames)")
        sys.exit(1)

    print(f"\nManifest: {manifest['n_episodes']} episodes, {manifest['n_samples']:,} samples")

    # === 3. Calibrate mouse bucketer ===
    print(f"\n{'='*60}")
    print("Scanning mouse/raw distribution...")
    print(f"{'='*60}")
    from sn_vla.data.percentile_scan import scan_mouse_distribution

    scan_result = scan_mouse_distribution(data_root, max_episodes=50)
    d_max = scan_result.get("d_max", args.d_max)
    if d_max <= 0:
        d_max = args.d_max  # fallback
    print(f"D_MAX = {d_max:.1f}")

    # === 4. Build model with real Mage-ViT ===
    print(f"\n{'='*60}")
    print("Building model...")
    print(f"{'='*60}")
    import torch
    from sn_vla.model.mage_vit_backbone import MageViTBackbone
    from sn_vla.model.sn_vla import SNVLA, SNVLAConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Device: {device}, GPUs available: {n_gpus}")

    backbone = MageViTBackbone.from_magevl(magevl_dir)
    config = SNVLAConfig(
        n_mouse_buckets=args.n_mouse_buckets,
        n_memory_layers=args.n_memory_layers,
    )
    model = SNVLA(config, visual_encoder=backbone)
    model.freeze_visual()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_trainable:,} trainable / {n_total:,} total params")
    print(f"  ViT frozen, training Memory + FastActionHead only")

    model.to(device)
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"  Using DataParallel across {n_gpus} GPUs")

    # === 5. Dataset ===
    from sn_vla.data.dataset import D2EDataset, collate_fn
    from sn_vla.data.mouse_buckets import MouseBucketer
    from torch.utils.data import DataLoader

    bucketer = MouseBucketer(d_max=d_max, n_buckets=args.n_mouse_buckets)
    dataset = D2EDataset(
        manifest_path=manifest_path,
        tick_hz=args.tick_hz,
        window_frames=args.window_frames,
        clip_frames=args.window_frames,
        bucketer=bucketer,
    )
    print(f"Dataset: {len(dataset):,} samples")

    # Compute key stats
    from sn_vla.train.phase1_fast_bc import compute_key_stats
    valid_key_mask, key_pos_weight = compute_key_stats(dataset)
    print(f"Valid keys: {valid_key_mask.sum()}/256")

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

    # === 6. Training ===
    from sn_vla.model.fast_action_head import fast_action_loss

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    print(f"\n{'='*60}")
    print(f"Starting Phase 1 training: {args.epochs} epochs, {len(loader)} steps/epoch")
    print(f"  batch_size={args.batch_size}, lr={args.lr}, total_steps={total_steps}")
    print(f"{'='*60}\n")

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []

        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(device)

            m = model.module if isinstance(model, torch.nn.DataParallel) else model
            action, _ = m(
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
                kbd_acc = ((torch.sigmoid(action.kbd) > 0.5).float() == batch["target_kbd"]).float().mean().item()
                print(f"  [ep{epoch} step{global_step:>6}] loss={losses['total']:.4f} "
                      f"kbd={losses['kbd']:.4f} mouse={losses['mouse']:.4f} "
                      f"kbd_acc={kbd_acc:.3f} lr={scheduler.get_last_lr()[0]:.2e}")

        avg = np.mean(epoch_losses)
        print(f"Epoch {epoch}: avg_loss={avg:.4f}")

        # Save checkpoint
        ckpt_path = output_dir / f"checkpoint_ep{epoch}.pt"
        state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        torch.save({
            "model": state,
            "epoch": epoch,
            "loss": avg,
            "valid_key_mask": valid_key_mask,
            "key_pos_weight": key_pos_weight,
            "bucketer": {"d_max": d_max, "n_buckets": args.n_mouse_buckets},
        }, ckpt_path)
        print(f"  Saved {ckpt_path}")

    print("\nPhase 1 training complete.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d2e-dir", required=True, help="D2E dataset directory")
    ap.add_argument("--magevl-dir", required=True, help="Mage-VL model directory")
    ap.add_argument("--output-dir", default="checkpoints/phase1")
    ap.add_argument("--games", default=None, help="Comma-separated game names to filter")
    ap.add_argument("--tick-hz", type=int, default=60)
    ap.add_argument("--window-frames", type=int, default=64)
    ap.add_argument("--n-mouse-buckets", type=int, default=64)
    ap.add_argument("--n-memory-layers", type=int, default=2)
    ap.add_argument("--d-max", type=float, default=200.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--skip-causality", action="store_true", help="Skip B-frame check")
    args = ap.parse_args()
    run_phase1(args)


if __name__ == "__main__":
    main()
