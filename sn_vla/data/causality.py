"""Causality checks: B-frame detection and future-leakage prevention.

Locked spec (plan.md v3 §4.4):
  B-frames in H.264/H.265 use bidirectional prediction → codec bit-cost may
  contain future context. D2E-480p disables B-frames; D2E-Original must be
  checked per-file. Offline BC that accidentally includes future frames will
  show inflated accuracy but fail in closed-loop.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np


def ffprobe_frames(video_path: str | Path) -> list[dict]:
    """Get per-frame metadata via ffprobe (pict_type, pts, etc.)."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_frames",
            "-show_entries", "frame=pict_type,pts_time,coded_picture_number",
            "-of", "json",
            str(video_path),
        ],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr[:500]}")
    return json.loads(result.stdout).get("frames", [])


def has_b_frames(video_path: str | Path) -> bool:
    """Check if the video stream contains B-frames.

    B-frames use bidirectional prediction and would violate causality for
    codec-native bit-cost readiness.
    """
    frames = ffprobe_frames(video_path)
    pict_types = {f.get("pict_type", "?") for f in frames}
    return "B" in pict_types


def ffprobe_codec(video_path: str | Path) -> str:
    """Probe the video codec name (e.g. 'h264', 'hevc')."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name",
            "-of", "json",
            str(video_path),
        ],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe codec failed: {result.stderr[:500]}")
    streams = json.loads(result.stdout).get("streams", [])
    return streams[0]["codec_name"] if streams else "unknown"


def ffprobe_fps(video_path: str | Path) -> float:
    """Get average frame rate."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate",
            "-of", "json",
            str(video_path),
        ],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        return 0.0
    streams = json.loads(result.stdout).get("streams", [])
    fr_str = streams[0].get("avg_frame_rate", "0/1") if streams else "0/1"
    num, den = fr_str.split("/")
    return float(num) / float(den) if float(den) != 0 else 0.0


def ffprobe_duration(video_path: str | Path) -> float:
    """Get video duration in seconds."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-of", "json",
            str(video_path),
        ],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        return 0.0
    streams = json.loads(result.stdout).get("streams", [])
    return float(streams[0].get("duration", 0)) if streams else 0.0


def assert_no_future_leakage(
    decision_pts_ns: np.ndarray,
    visual_window_max_pts: np.ndarray,
    action_tick_start_ns: np.ndarray,
) -> None:
    """Assert that no decision point uses future visual or past action.

    Locked action time definition:
      O_t = visual frames with PTS ≤ t
      A_t = action during [t, t+Δ)

    Args:
        decision_pts_ns: [N] the decision timestamp t for each sample.
        visual_window_max_pts: [N] max PTS in the visual window for each sample.
        action_tick_start_ns: [N] start of the action tick (should be > t).
    """
    for i in range(len(decision_pts_ns)):
        t = decision_pts_ns[i]
        if visual_window_max_pts[i] > t:
            raise AssertionError(
                f"FUTURE LEAKAGE at sample {i}: visual max PTS {visual_window_max_pts[i]} "
                f"> decision time {t}. Visual window must only contain frames with PTS ≤ t."
            )
        if action_tick_start_ns[i] <= t:
            raise AssertionError(
                f"ACTION-TIME ERROR at sample {i}: action tick starts at "
                f"{action_tick_start_ns[i]} ≤ decision time {t}. "
                f"Action must be in (t, t+Δ]."
            )


class CausalityReport:
    """Result of a full causality audit on one episode."""

    def __init__(self, video_path: str | Path):
        self.video_path = Path(video_path)
        self.codec = ffprobe_codec(video_path)
        self.has_b_frames = has_b_frames(video_path)
        self.fps = ffprobe_fps(video_path)
        self.duration_s = ffprobe_duration(video_path)

    def __repr__(self):
        status = "PASS" if not self.has_b_frames else "FAIL (B-frames detected!)"
        return (
            f"CausalityReport({self.video_path.name})\n"
            f"  codec:      {self.codec}\n"
            f"  fps:        {self.fps:.1f}\n"
            f"  duration:   {self.duration_s:.1f}s\n"
            f"  b-frames:   {status}"
        )

    @property
    def is_causal_safe(self) -> bool:
        return not self.has_b_frames
