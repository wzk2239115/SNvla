"""Offline frame cache builder.

Extracts every `stride`-th frame from each episode mkv, resizes, and stores
JPEGs in per-episode folders. Enables fast random access during training.

On-the-fly cv2 seeking is ~100x slower (each seek decodes from the nearest
keyframe). ffmpeg here does sequential decode + scale + encode in one pass.

Cache layout:  <cache_root>/<game>/<episode_stem>/f_%06d.jpg
Mapping:       file f_000001.jpg = original frame 0, f_000002.jpg = frame
`stride`, ... so cache index j (0-based) = original_frame // stride.
"""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .manifest import find_episodes


def cache_dir_for(cache_root: Path, mkv_path: Path) -> Path:
    return cache_root / mkv_path.parent.name / mkv_path.stem


def build_episode_cache(
    mkv_path: Path,
    cache_root: Path,
    stride: int = 8,
    size: int = 224,
    quality: int = 3,
    timeout_s: int = 7200,
) -> dict:
    """Extract frames for one episode. Idempotent (skips if done marker exists)."""
    out_dir = cache_dir_for(cache_root, mkv_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "done.json"
    if marker.exists():
        return json.loads(marker.read_text())

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(mkv_path),
        "-vf", f"select=not(mod(n\\,{stride})),scale={size}:{size}",
        "-vsync", "0",
        "-q:v", str(quality),
        str(out_dir / "f_%06d.jpg"),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-500:])
    except Exception as e:
        print(f"ERROR {mkv_path.name}: {e}")
        return {"status": "error", "error": str(e)}

    n_frames = len(list(out_dir.glob("f_*.jpg")))
    meta = {
        "status": "ok",
        "stride": stride,
        "size": size,
        "n_frames": n_frames,
        "mkv_path": str(mkv_path),
    }
    marker.write_text(json.dumps(meta))
    return meta


def _worker(args):
    mkv, cache_root, stride, size, quality = args
    meta = build_episode_cache(mkv, Path(cache_root), stride, size, quality)
    return mkv.name, meta["status"]


def build_frame_cache(
    d2e_dir: str | Path,
    cache_root: str | Path,
    stride: int = 8,
    size: int = 224,
    quality: int = 3,
    workers: int = 16,
    max_episodes: int = 0,
):
    """Build frame cache for all episodes under d2e_dir (parallel ffmpeg jobs)."""
    pairs = find_episodes(d2e_dir)
    if max_episodes > 0:
        pairs = pairs[:max_episodes]
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    print(f"Building frame cache: {len(pairs)} episodes, stride={stride}, size={size}, workers={workers}")

    jobs = [(mkv, cache_root, stride, size, quality) for _, mkv in pairs]
    done = skip = error = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for name, status in ex.map(_worker, jobs):
            if status == "ok":
                done += 1
            elif status == "error":
                error += 1
            print(f"  [{done + skip + error}/{len(jobs)}] {name}: {status}")

    print(f"Done: {done} extracted, {error} errors. Cache at {cache_root}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("d2e_dir")
    ap.add_argument("cache_root")
    ap.add_argument("--stride", type=int, default=8, help="Keep every Nth frame (8 → 7.5fps)")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--quality", type=int, default=3, help="JPEG quality (2=best, 31=worst)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-episodes", type=int, default=0)
    args = ap.parse_args()
    build_frame_cache(args.d2e_dir, args.cache_root, args.stride, args.size,
                      args.quality, args.workers, args.max_episodes)


if __name__ == "__main__":
    main()
