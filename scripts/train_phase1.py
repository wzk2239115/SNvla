#!/usr/bin/env python3
"""Phase 1 training launch script (DDP, multi-GPU).

Usage:
    torchrun --nproc_per_node=8 scripts/train_phase1.py \
        --d2e-dir /home/jovyan/exploitgym/D2E-Original \
        --magevl-dir /home/jovyan/exploitgym/Mage-VL \
        --frame-cache /home/jovyan/exploitgym/frame_cache \
        --games "Brotato" \
        --output-dir checkpoints/phase1_brotato

    NOTE: --batch-size is PER GPU. Total batch = batch_size * nproc_per_node.

Steps:
    1. Build manifest (rank 0 only, then barrier)
    2. Scan mouse/raw distribution → calibrate bucketer (rank 0)
    3. Train frozen Mage-ViT + TemporalMemory + FastActionHead via DDP
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def setup_ddp():
    """Initialize DDP if launched via torchrun. Returns (rank, local_rank, world_size, device)."""
    import torch
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        import torch.distributed as dist
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl", device_id=device)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return rank, local_rank, world_size, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 0, 1, device


def barrier(world_size):
    if world_size > 1:
        import torch
        import torch.distributed as dist
        dist.barrier(device_ids=[torch.cuda.current_device()])


def explore_d2e(d2e_dir: Path, max_show: int = 40):
    games = sorted([d for d in d2e_dir.iterdir() if d.is_dir()])
    print(f"Found {len(games)} game directories")
    for g in games[:max_show]:
        n = sum(1 for _ in g.rglob("*.mcap"))
        print(f"  {g.name}: {n} episodes")
    return games


def run_phase1(args):
    import torch

    rank, local_rank, world_size, device = setup_ddp()
    is_main = rank == 0

    d2e_dir = Path(args.d2e_dir)
    magevl_dir = Path(args.magevl_dir)
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    # === 1. Manifest (rank 0 builds, others wait) ===
    if is_main:
        print(f"\n{'='*60}\nBuilding manifest...\n{'='*60}")
        from sn_vla.data.manifest import build_manifest
        games = explore_d2e(d2e_dir)
        if args.games:
            game_filter = [g.strip() for g in args.games.split(",")]
            games = [g for g in games if g.name in game_filter]
            print(f"\nFiltered to {len(games)} games: {[g.name for g in games]}")
        if not games:
            sys.exit(1)

        manifest_path = output_dir / "manifest.json"
        if args.games and len(games) == 1:
            data_root = games[0]
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
            print("ERROR: no valid episodes")
            sys.exit(1)
        print(f"\nManifest: {manifest['n_episodes']} episodes, {manifest['n_samples']:,} samples")

    barrier(world_size)

    manifest_path = output_dir / "manifest.json"

    # === 2. Data ===
    from sn_vla.data.dataset import D2EDataset, collate_fn
    from sn_vla.data.mouse_buckets import MouseBucketer
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    # Calibrate D_MAX (rank 0 computes, reuse from manifest dir for simplicity —
    # cheap default fallback for non-FPS games)
    calib_path = output_dir / "mouse_calibration.json"
    if is_main and not calib_path.exists():
        from sn_vla.data.percentile_scan import scan_mouse_distribution
        games = sorted([d for d in d2e_dir.iterdir() if d.is_dir()])
        if args.games:
            gf = [g.strip() for g in args.games.split(",")]
            games = [g for g in games if g.name in gf]
        result = scan_mouse_distribution(games[0] if len(games) == 1 else d2e_dir, max_episodes=30)
        calib_path.write_text(json.dumps(result))
    barrier(world_size)
    d_max = args.d_max
    if calib_path.exists():
        d_max = json.loads(calib_path.read_text()).get("d_max", args.d_max) or args.d_max
    if is_main:
        print(f"D_MAX = {d_max}")

    bucketer = MouseBucketer(d_max=d_max, n_buckets=args.n_mouse_buckets)
    dataset = D2EDataset(
        manifest_path=manifest_path,
        frame_cache_dir=args.frame_cache,
        tick_hz=args.tick_hz,
        window_frames=args.window_frames,
        clip_frames=args.window_frames,
        bucketer=bucketer,
    )

    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                      shuffle=True, drop_last=True)
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,          # per-GPU
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    # === 3. Key stats ===
    from sn_vla.train.phase1_fast_bc import compute_key_stats
    valid_key_mask, key_pos_weight = compute_key_stats(dataset)
    valid_key_mask_t = torch.from_numpy(valid_key_mask).to(device)
    key_pos_weight_t = torch.from_numpy(key_pos_weight).to(device)

    # === 4. Model ===
    if is_main:
        print(f"\n{'='*60}\nBuilding model...\n{'='*60}")
    from sn_vla.model.mage_vit_backbone import MageViTBackbone
    from sn_vla.model.sn_vla import SNVLA, SNVLAConfig

    backbone = MageViTBackbone.from_magevl(magevl_dir)
    config = SNVLAConfig(
        n_mouse_buckets=args.n_mouse_buckets,
        n_memory_layers=args.n_memory_layers,
    )
    model = SNVLA(config, visual_encoder=backbone)
    model.freeze_visual()
    model.to(device)

    if world_size > 1:
        import torch.distributed as dist
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )
    else:
        ddp_model = model

    if is_main:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"Model: {n_trainable:,} trainable / {n_total:,} total (ViT frozen)")
        print(f"DDP world_size={world_size}, per-GPU batch={args.batch_size}, "
              f"total batch={args.batch_size * world_size}")
        print(f"Dataset: {len(dataset):,} samples, {len(loader)} steps/epoch/GPU")
        print(f"Valid keys: {valid_key_mask.sum()}/256")

    # === 5. Optimizer ===
    from sn_vla.model.fast_action_head import fast_action_loss

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    if is_main:
        print(f"\n{'='*60}\nTraining: {args.epochs} epochs\n{'='*60}\n")

    import time
    global_step = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        epoch_losses = []

        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)

            action, _ = ddp_model(
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

            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()

            epoch_losses.append(losses["total"].item())
            global_step += 1

            if is_main and global_step % args.log_every == 0:
                with torch.no_grad():
                    kbd_acc = ((torch.sigmoid(action.kbd) > 0.5).float()
                               == batch["target_kbd"]).float().mean().item()
                eps = args.batch_size * world_size / max(time.time() - t0, 1e-6)
                print(f"  [ep{epoch} step{global_step:>6}] loss={losses['total']:.4f} "
                      f"kbd={losses['kbd']:.4f} mouse={losses['mouse']:.4f} "
                      f"kbd_acc={kbd_acc:.3f} {eps:.0f} samples/s lr={scheduler.get_last_lr()[0]:.2e}")

        barrier(world_size)
        if is_main:
            avg = float(np.mean(epoch_losses))
            print(f"Epoch {epoch}: avg_loss={avg:.4f} ({time.time()-t0:.0f}s)")
            ckpt_path = output_dir / f"checkpoint_ep{epoch}.pt"
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "loss": avg,
                "valid_key_mask": valid_key_mask,
                "key_pos_weight": key_pos_weight,
                "bucketer": {"d_max": d_max, "n_buckets": args.n_mouse_buckets},
                "config": config.__dict__,
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")

    barrier(world_size)
    if is_main:
        print("\nPhase 1 training complete.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d2e-dir", required=True)
    ap.add_argument("--magevl-dir", required=True)
    ap.add_argument("--frame-cache", default=None, help="JPEG frame cache dir (strongly recommended)")
    ap.add_argument("--output-dir", default="checkpoints/phase1")
    ap.add_argument("--games", default=None)
    ap.add_argument("--tick-hz", type=int, default=60)
    ap.add_argument("--window-frames", type=int, default=64)
    ap.add_argument("--n-mouse-buckets", type=int, default=64)
    ap.add_argument("--n-memory-layers", type=int, default=2)
    ap.add_argument("--d-max", type=float, default=200.0)
    ap.add_argument("--batch-size", type=int, default=64, help="PER GPU")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--num-workers", type=int, default=8, help="per GPU")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--skip-causality", action="store_true")
    args = ap.parse_args()
    run_phase1(args)


if __name__ == "__main__":
    main()
