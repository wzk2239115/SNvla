"""Dataset manifest: index all episodes and build the global sample list.

Scans a directory tree for .mcap/.mkv pairs, builds EpisodeIndex for each,
and produces a flat list of training samples with per-game grouping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from .align import EpisodeIndex, build_episode_index, episode_index_to_samples, TrainingSample
from .causality import CausalityReport, ffprobe_codec, ffprobe_fps


def find_episodes(root_dir: str | Path) -> list[tuple[Path, Path]]:
    """Find all .mcap/.mkv pairs under root_dir.

    Returns list of (mcap_path, mkv_path).
    """
    root = Path(root_dir)
    pairs = []
    for mcap in sorted(root.rglob("*.mcap")):
        mkv = mcap.with_suffix(".mkv")
        if mkv.exists():
            pairs.append((mcap, mkv))
    return pairs


def build_manifest(
    root_dir: str | Path,
    output_path: str | Path,
    tick_hz: int = 60,
    window_frames: int = 64,
    check_causality: bool = True,
    skip_b_frame_episodes: bool = True,
) -> dict:
    """Build a manifest JSON from a directory of D2E recordings.

    Args:
        root_dir: root directory containing .mcap/.mkv pairs.
        output_path: path to write the manifest JSON.
        tick_hz: action tick frequency.
        window_frames: visual window size.
        check_causality: if True, probe each episode for B-frames.
        skip_b_frame_episodes: if True (and check_causality), skip non-causal episodes.
    """
    pairs = find_episodes(root_dir)
    print(f"Found {len(pairs)} episodes in {root_dir}")

    manifest = {
        "root_dir": str(root_dir),
        "tick_hz": tick_hz,
        "window_frames": window_frames,
        "episodes": [],
        "n_samples": 0,
    }

    total_samples = 0
    for mcap_path, mkv_path in tqdm(pairs, desc="Indexing episodes"):
        if check_causality:
            report = CausalityReport(mkv_path)
            if not report.is_causal_safe:
                msg = f"  WARNING: {mkv_path.name} has B-frames"
                if skip_b_frame_episodes:
                    print(f"{msg} — skipping")
                    continue
                print(msg)

        try:
            idx = build_episode_index(
                mcap_path=mcap_path,
                mkv_path=mkv_path,
                tick_hz=tick_hz,
                window_frames=window_frames,
            )
        except Exception as e:
            print(f"  ERROR indexing {mcap_path.name}: {e}")
            continue

        ep_entry = {
            "episode_id": idx.episode_id,
            "mcap_path": str(mcap_path),
            "mkv_path": str(mkv_path),
            "game_title": idx.game_title,
            "duration_ns": int(idx.duration_ns),
            "fps": float(idx.fps),
            "n_ticks": idx.n_ticks,
            "n_samples": idx.n_ticks,
            "codec": report.codec if check_causality else ffprobe_codec(mkv_path),
        }
        manifest["episodes"].append(ep_entry)
        total_samples += idx.n_ticks

    manifest["n_samples"] = total_samples
    manifest["n_episodes"] = len(manifest["episodes"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    games = {}
    for ep in manifest["episodes"]:
        g = ep["game_title"]
        games[g] = games.get(g, 0) + ep["n_samples"]
    print(f"\nManifest: {manifest['n_episodes']} episodes, {total_samples} samples")
    print("Per-game sample counts:")
    for g, c in sorted(games.items(), key=lambda x: -x[1]):
        print(f"  {g:40s} {c:>10d}")

    return manifest


def load_manifest(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def get_game_list(manifest: dict) -> list[str]:
    return sorted({ep["game_title"] for ep in manifest["episodes"]})
