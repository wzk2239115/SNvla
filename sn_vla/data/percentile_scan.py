"""Scan D2E to calibrate mouse delta distribution and determine D_MAX.

Locked spec (plan.md v3 §3.3):
  D_MAX = P99.9 of absolute mouse/raw deltas across the training set.
  The bucketer uses non-uniform signed-log spacing with D_MAX as the clamp bound.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .mcap_reader import parse_mcap
from .manifest import find_episodes


def scan_mouse_distribution(root_dir: str | Path, max_episodes: int = 0) -> dict:
    """Scan all episodes and collect mouse/raw delta statistics.

    Args:
        root_dir: directory containing .mcap/.mkv pairs.
        max_episodes: 0 = scan all; otherwise limit.

    Returns:
        dict with percentile info.
    """
    pairs = find_episodes(root_dir)
    if max_episodes > 0:
        pairs = pairs[:max_episodes]

    all_abs = []
    for mcap_path, _ in tqdm(pairs, desc="Scanning mouse/raw"):
        try:
            episode = parse_mcap(mcap_path)
        except Exception:
            continue
        for ev in episode.raw_mouse_events:
            all_abs.append(abs(ev.dx))
            all_abs.append(abs(ev.dy))

    if not all_abs:
        print("No mouse/raw events found. Mouse may use mouse/state instead.")
        return {"d_max": 200.0, "note": "no raw mouse data; using default"}

    all_abs = np.array(all_abs, dtype=np.float64)
    percentiles = {p: float(np.percentile(all_abs, p)) for p in [50, 90, 95, 99, 99.9, 100]}

    print("Mouse/raw |delta| distribution:")
    for p, v in percentiles.items():
        print(f"  P{p:<5}: {v:>10.1f} px")

    return {
        "d_max": percentiles[99.9],
        "percentiles": percentiles,
        "n_events": len(all_abs),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root_dir", help="Directory with .mcap/.mkv pairs")
    ap.add_argument("--output", default="mouse_calibration.json")
    ap.add_argument("--max-episodes", type=int, default=0)
    args = ap.parse_args()

    result = scan_mouse_distribution(args.root_dir, args.max_episodes)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nD_MAX (P99.9) = {result['d_max']:.1f} → saved to {args.output}")


if __name__ == "__main__":
    main()
