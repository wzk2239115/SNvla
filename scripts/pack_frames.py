#!/usr/bin/env python3
"""Pack per-episode JPEG frame caches into single uint8 .npy files.

Random-access JPEG decode is the training bottleneck (~8 cv2.imread calls per
sample). Packing frames into one [n_frames, 224, 224, 3] uint8 array per
episode enables zero-copy np.memmap reads at train time.

Layout: <pack_root>/<game>/<episode_stem>.npy + .npy.done marker
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sn_vla.data.manifest import find_episodes


def pack_episode(job):
    frame_dir, out_path, size = job
    frame_dir, out_path = Path(frame_dir), Path(out_path)
    if (out_path.parent / (out_path.name + ".done")).exists():
        return out_path.name, "skip"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import cv2
        files = sorted(frame_dir.glob("f_*.jpg"))
        if not files:
            return out_path.name, "no-frames"
        arr = np.empty((len(files), size, size, 3), dtype=np.uint8)
        for i, fp in enumerate(files):
            img = cv2.imread(str(fp))
            if img is None:
                img = np.zeros((size, size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            arr[i] = img
        np.save(out_path, arr)
        (out_path.parent / (out_path.name + ".done")).write_text(
            json.dumps({"n_frames": len(files), "size": size}))
        return out_path.name, "ok"
    except Exception as e:
        return out_path.name, f"error: {e}"


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("d2e_dir")
    ap.add_argument("frame_cache", help="JPEG cache root")
    ap.add_argument("pack_root", help="output dir for packed .npy files")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    pairs = find_episodes(args.d2e_dir)
    jobs = []
    for mcap, mkv in pairs:
        game = mkv.parent.name
        stem = mkv.stem
        frame_dir = Path(args.frame_cache) / game / stem
        out_dir = Path(args.pack_root) / game
        out_dir.mkdir(parents=True, exist_ok=True)
        jobs.append((frame_dir, out_dir / f"{stem}.npy", args.size))

    print(f"Packing {len(jobs)} episodes → {args.pack_root} "
          f"({args.workers} workers)")
    from tqdm import tqdm
    ok = skip = err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for name, status in tqdm(ex.map(pack_episode, jobs), total=len(jobs),
                                 desc="Packing", unit="ep"):
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                err += 1
                print(f"  {name}: {status}")
    print(f"Done: {ok} packed, {skip} skipped, {err} errors")


if __name__ == "__main__":
    main()
