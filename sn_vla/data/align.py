"""Align episodes into training samples.

Locked spec (plan.md v3 §4):
  Each training sample = (visual window, action target at next tick, gate label).

  Action time definition (locked):
    O_t = all visual frames with PTS ≤ t
    A_t = action executed during [t, t+Δ),  Δ = 1/tick_hz

  Decision points are placed at tick boundaries so that:
    - The visual window ends exactly at t (no future frames).
    - The action target is the NEXT tick's executed action.

  For the fast policy (Phase 1-2), every tick is a decision point.
  For the gate/brain (Phase 3), decision points are subsampled to 2-10 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .mcap_reader import EpisodeData, parse_mcap
from .reconstruct import TickAction, build_tick_actions


@dataclass
class TrainingSample:
    """One aligned training sample.

    Attributes:
        mkv_path: path to the source video for frame extraction.
        visual_frame_indices: indices into the video (PTS-ordered) forming O_t.
            All these frames have PTS ≤ decision_time_ns.
        decision_time_ns: the decision timestamp t.
        action: the TickAction for [t, t+Δ).
        gate_label: "continue" or "replan" (Phase 3; Phase 1-2 leave as "continue").
        episode_id: identifier for the source episode.
        tick_index: index within the episode's tick sequence.
    """
    mkv_path: str
    visual_frame_indices: np.ndarray
    decision_time_ns: int
    action: TickAction
    gate_label: str = "continue"
    episode_id: str = ""
    tick_index: int = 0


@dataclass
class EpisodeIndex:
    """Pre-computed index for one episode (avoids re-parsing mcap every epoch)."""
    episode_id: str
    mcap_path: str
    mkv_path: str
    game_title: str
    duration_ns: int
    fps: float
    n_ticks: int
    # Per-tick data (numpy arrays for efficiency)
    decision_times_ns: np.ndarray       # [n_ticks]
    visual_end_frame_idx: np.ndarray    # [n_ticks] last valid frame index (PTS ≤ t)
    kbd_multi_hot: np.ndarray           # [n_ticks, 256]
    press_events: np.ndarray            # [n_ticks, 256]
    release_events: np.ndarray           # [n_ticks, 256]
    mouse_dx: np.ndarray                # [n_ticks]
    mouse_dy: np.ndarray                # [n_ticks]
    btn_multi_hot: np.ndarray           # [n_ticks, 5]
    wheel: np.ndarray                   # [n_ticks]


def build_screen_pts_array(episode: EpisodeData) -> np.ndarray:
    """Extract the sorted PTS array from screen messages."""
    return np.array([s.pts_ns for s in episode.screen], dtype=np.int64)


def find_visual_end_frame(pts_array: np.ndarray, t_ns: int) -> int:
    """Find the index of the last frame with PTS ≤ t_ns (causal).

    Returns -1 if no frame satisfies the condition.
    """
    idx = np.searchsorted(pts_array, t_ns, side="right") - 1
    return int(idx)


def build_episode_index(
    mcap_path: str | Path,
    mkv_path: str | Path | None = None,
    tick_hz: int = 60,
    window_frames: int = 64,
    episode_id: str | None = None,
) -> EpisodeIndex:
    """Parse an episode and build the pre-computed index.

    Args:
        mcap_path: path to .mcap file.
        mkv_path: path to .mkv file (defaults to mcap_path with .mkv suffix).
        tick_hz: action tick frequency (default 60).
        window_frames: number of recent frames in the visual window O_t.
        episode_id: override identifier (defaults to mcap stem).
    """
    mcap_path = Path(mcap_path)
    if mkv_path is None:
        mkv_path = mcap_path.with_suffix(".mkv")
    else:
        mkv_path = Path(mkv_path)
    episode_id = episode_id or mcap_path.stem

    episode = parse_mcap(mcap_path)
    if not episode.screen:
        raise ValueError(f"No screen messages in {mcap_path}")

    game_title = episode.window_infos[0].title if episode.window_infos else "unknown"

    # Screen messages bridge two clocks: log_time_ns (recorder) ↔ pts_ns (video).
    # All events use log_time; tick boundaries are in log_time.
    log_time_array = np.array([s.log_time_ns for s in episode.screen], dtype=np.int64)
    start_ns = int(log_time_array[0])
    duration_ns = int(log_time_array[-1] - start_ns)
    fps = episode.fps

    tick_actions = build_tick_actions(episode, tick_hz=tick_hz)
    n_ticks = len(tick_actions)
    tick_ns = int(1e9 / tick_hz)

    # Decision times in log_time domain (matches keyboard/mouse events)
    decision_times = np.array([
        start_ns + i * tick_ns for i in range(n_ticks)
    ], dtype=np.int64)

    # Visual end frame: last screen frame whose log_time ≤ decision time.
    # This gives the video frame index (0-based) that is causally valid.
    visual_end_indices = np.array([
        int(np.searchsorted(log_time_array, t, side="right") - 1)
        for t in decision_times
    ], dtype=np.int64)

    kbd_multi_hot = np.stack([a.kbd_multi_hot for a in tick_actions]) if tick_actions else np.zeros((0, 256), dtype=np.float32)
    press_events = np.stack([a.press_events for a in tick_actions]) if tick_actions else np.zeros((0, 256), dtype=np.float32)
    release_events = np.stack([a.release_events for a in tick_actions]) if tick_actions else np.zeros((0, 256), dtype=np.float32)
    mouse_dx = np.array([a.mouse_dx for a in tick_actions], dtype=np.int32) if tick_actions else np.zeros(0, dtype=np.int32)
    mouse_dy = np.array([a.mouse_dy for a in tick_actions], dtype=np.int32) if tick_actions else np.zeros(0, dtype=np.int32)
    btn_multi_hot = np.stack([a.btn_multi_hot for a in tick_actions]) if tick_actions else np.zeros((0, 5), dtype=np.float32)
    wheel = np.array([a.wheel for a in tick_actions], dtype=np.int64) if tick_actions else np.zeros(0, dtype=np.int64)

    return EpisodeIndex(
        episode_id=episode_id,
        mcap_path=str(mcap_path),
        mkv_path=str(mkv_path),
        game_title=game_title,
        duration_ns=duration_ns,
        fps=fps,
        n_ticks=n_ticks,
        decision_times_ns=decision_times,
        visual_end_frame_idx=visual_end_indices,
        kbd_multi_hot=kbd_multi_hot,
        press_events=press_events,
        release_events=release_events,
        mouse_dx=mouse_dx,
        mouse_dy=mouse_dy,
        btn_multi_hot=btn_multi_hot,
        wheel=wheel,
    )


def episode_index_to_samples(
    idx: EpisodeIndex,
    window_frames: int = 64,
    min_tick: int = 0,
    max_tick: int | None = None,
) -> list[TrainingSample]:
    """Convert an EpisodeIndex into a list of TrainingSamples.

    Each sample's visual window = the most recent `window_frames` frames up to
    and including the decision time (all PTS ≤ t).
    """
    max_tick = max_tick or idx.n_ticks
    samples = []
    for tick in range(min_tick, min(max_tick, idx.n_ticks)):
        end_frame = int(idx.visual_end_frame_idx[tick])
        if end_frame < 0:
            continue
        start_frame = max(0, end_frame - window_frames + 1)
        frame_indices = np.arange(start_frame, end_frame + 1, dtype=np.int64)

        action = TickAction(
            tick_start_ns=int(idx.decision_times_ns[tick]),
            tick_end_ns=int(idx.decision_times_ns[tick]) + int(1e9 / 60),
            kbd_held=frozenset(np.flatnonzero(idx.kbd_multi_hot[tick]).tolist()),
            kbd_multi_hot=idx.kbd_multi_hot[tick],
            press_events=idx.press_events[tick],
            release_events=idx.release_events[tick],
            mouse_dx=int(idx.mouse_dx[tick]),
            mouse_dy=int(idx.mouse_dy[tick]),
            mouse_buttons=frozenset(),
            btn_multi_hot=idx.btn_multi_hot[tick],
            wheel=int(idx.wheel[tick]),
        )

        samples.append(TrainingSample(
            mkv_path=idx.mkv_path,
            visual_frame_indices=frame_indices,
            decision_time_ns=int(idx.decision_times_ns[tick]),
            action=action,
            episode_id=idx.episode_id,
            tick_index=tick,
        ))
    return samples
