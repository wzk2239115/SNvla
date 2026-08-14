"""Offline preprocessing: extract video clips and compute codec canvas cache.

For each episode, divides the video into non-overlapping segments of
``clip_frames`` (default 64 frames ≈ 1s at 60fps), then for each segment:

  1. Extract the sub-clip as a short .mkv (ffmpeg).
  2. Run the codec patchifier (cv-preinfer HEVC/H.264 path, or DCVC-RT).
  3. Save canvas images + src_patch_position.npy + meta.json.

At training time, each tick maps to the segment covering its visual window.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from .causality import ffprobe_codec, ffprobe_fps, ffprobe_duration
from .manifest import find_episodes


def extract_clip(
    mkv_path: Path,
    output_path: Path,
    start_frame: int,
    n_frames: int,
    fps: float,
) -> Path:
    """Extract a sub-clip from mkv using ffmpeg (stream copy when possible)."""
    start_s = start_frame / fps
    duration_s = n_frames / fps
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "quiet",
        "-ss", f"{start_s:.6f}",
        "-i", str(mkv_path),
        "-t", f"{duration_s:.6f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0 or not output_path.exists():
        # Fallback: re-encode if stream copy fails
        cmd_fb = [
            "ffmpeg", "-y", "-v", "quiet",
            "-ss", f"{start_s:.6f}",
            "-i", str(mkv_path),
            "-t", f"{duration_s:.6f}",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            "-an",
            str(output_path),
        ]
        result2 = subprocess.run(cmd_fb, capture_output=True, text=True, timeout=300)
        if result2.returncode != 0:
            raise RuntimeError(
                f"ffmpeg clip extraction failed: {result2.stderr[:300]}"
            )
    return output_path


def extract_clip_frames(
    mkv_path: Path,
    output_dir: Path,
    start_frame: int,
    n_frames: int,
    fps: float,
) -> list[Path]:
    """Extract individual frames as PNG (fallback when codec unavailable)."""
    import cv2
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(mkv_path))
    frame_paths = []
    for i in range(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i)
        ok, frame = cap.read()
        if not ok:
            break
        path = output_dir / f"frame_{i:04d}.png"
        cv2.imwrite(str(path), frame)
        frame_paths.append(path)
    cap.release()
    return frame_paths


def run_codec_on_clip(
    clip_path: Path,
    output_dir: Path,
    target_canvas: int = 8,
    group_size: int = 32,
    images_per_group: int = 4,
    patch: int = 16,
    max_pixels: int = 150000,
    codec_engine: str = "hevc",
) -> dict:
    """Run the Mage-VL codec patchifier on a single clip.

    Tries the codec path first (cv-preinfer), falls back to frame extraction.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())

    # Try codec-video-prep
    cv_preinfer_bin = os.environ.get("CV_PREINFER_BIN", "cv-preinfer")
    if shutil.which(cv_preinfer_bin):
        num_sampled = (target_canvas // images_per_group) * group_size
        cmd = [
            cv_preinfer_bin,
            "--video", str(clip_path),
            "--out_dir", str(output_dir),
            "--num_sampled_frames", str(num_sampled),
            "--grouping_mode", "readiness",
            "--group_size", str(group_size),
            "--images_per_group", str(images_per_group),
            "--patch", str(patch),
            "--max_pixels", str(max_pixels),
            "--readiness_sum_threshold", "0",
            "--canvas_format", "jpg",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and meta_path.exists():
            return json.loads(meta_path.read_text())

    # Fallback: simple frame extraction (multi-image mode)
    fps = ffprobe_fps(clip_path)
    frame_dir = output_dir / "frames"
    frame_paths = extract_clip_frames(clip_path, frame_dir, 0, target_canvas * images_per_group, fps)
    meta = {
        "engine": "frame_fallback",
        "canvas_files": [str(p.relative_to(output_dir)) for p in frame_paths],
        "fps": fps,
        "n_frames": len(frame_paths),
        "note": "codec patchifier unavailable; using uniform frame extraction",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def preprocess_episode(
    mcap_path: Path,
    mkv_path: Path,
    output_root: Path,
    clip_frames: int = 64,
    target_canvas: int = 8,
    group_size: int = 32,
    images_per_group: int = 4,
    patch: int = 16,
    max_pixels: int = 150000,
    codec_engine: str | None = None,
) -> list[dict]:
    """Preprocess one episode into per-clip codec caches.

    Returns list of clip metadata dicts with segment index ranges.
    """
    fps = ffprobe_fps(mkv_path)
    if fps <= 0:
        fps = 60.0
    duration_s = ffprobe_duration(mkv_path)
    total_frames = int(duration_s * fps)
    if codec_engine is None:
        codec_engine = ffprobe_codec(mkv_path)

    episode_id = mcap_path.stem
    episode_dir = output_root / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)

    n_clips = total_frames // clip_frames
    clip_infos = []

    for clip_idx in range(n_clips):
        start_frame = clip_idx * clip_frames
        clip_dir = episode_dir / f"clip_{clip_idx:05d}"
        meta_path = clip_dir / "meta.json"

        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        else:
            clip_mkv = clip_dir / "clip.mkv"
            try:
                extract_clip(mkv_path, clip_mkv, start_frame, clip_frames, fps)
                meta = run_codec_on_clip(
                    clip_mkv, clip_dir,
                    target_canvas=target_canvas,
                    group_size=group_size,
                    images_per_group=images_per_group,
                    patch=patch,
                    max_pixels=max_pixels,
                    codec_engine=codec_engine,
                )
            except Exception as e:
                print(f"  ERROR clip {clip_idx}: {e}")
                continue

        clip_infos.append({
            "clip_idx": clip_idx,
            "clip_dir": str(clip_dir),
            "start_frame": start_frame,
            "n_frames": clip_frames,
            "meta": meta,
        })

    episode_meta = {
        "episode_id": episode_id,
        "mkv_path": str(mkv_path),
        "fps": fps,
        "total_frames": total_frames,
        "clip_frames": clip_frames,
        "n_clips": len(clip_infos),
        "clips": clip_infos,
    }
    (episode_dir / "episode_meta.json").write_text(json.dumps(episode_meta, indent=2))
    return clip_infos


def preprocess_dataset(
    root_dir: str | Path,
    output_root: str | Path,
    clip_frames: int = 64,
    target_canvas: int = 8,
    max_episodes: int = 0,
):
    """Preprocess all episodes in root_dir."""
    pairs = find_episodes(root_dir)
    if max_episodes > 0:
        pairs = pairs[:max_episodes]

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for mcap_path, mkv_path in tqdm(pairs, desc="Preprocessing"):
        preprocess_episode(
            mcap_path=Path(mcap_path),
            mkv_path=Path(mkv_path),
            output_root=output_root,
            clip_frames=clip_frames,
            target_canvas=target_canvas,
        )


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root_dir", help="Directory with .mcap/.mkv pairs")
    ap.add_argument("output_root", help="Output directory for codec cache")
    ap.add_argument("--clip-frames", type=int, default=64)
    ap.add_argument("--target-canvas", type=int, default=8)
    ap.add_argument("--max-episodes", type=int, default=0)
    args = ap.parse_args()

    preprocess_dataset(
        root_dir=args.root_dir,
        output_root=args.output_root,
        clip_frames=args.clip_frames,
        target_canvas=args.target_canvas,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
