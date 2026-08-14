"""Reconstruct high-frequency (60Hz) keyboard/mouse state from event streams.

Locked spec (plan.md v3 §4):
  - Keyboard ground truth = event stream (press/release), NOT keyboard/state 1Hz snapshot.
  - Mouse deltas = mouse/raw HID deltas aggregated per tick window.
  - keyboard/state used only for sanity-check validation.

Action time definition (locked):
  A_t = action executed during [t, t+Δ),  Δ = 1/tick_hz seconds
    mouse:   Σ raw deltas in [t, t+Δ)
    keyboard: held state at t+Δ^- (just before next tick)
    buttons:  held state at t+Δ^-
    wheel:    events within [t, t+Δ)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .mcap_reader import EpisodeData, KeyboardEvent, RawMouseEvent


@dataclass
class TickAction:
    """One tick's full action target (matches FastActionHead output dims)."""
    tick_start_ns: int
    tick_end_ns: int
    kbd_held: frozenset[int]          # desired held state at tick_end^-
    kbd_multi_hot: np.ndarray         # [256] float32
    press_events: np.ndarray          # [256] float32, 1.0 if key pressed during tick
    release_events: np.ndarray        # [256] float32, 1.0 if key released during tick
    mouse_dx: int                     # Σ raw dx in [t, t+Δ)
    mouse_dy: int                     # Σ raw dy in [t, t+Δ)
    mouse_buttons: frozenset[str]     # desired button state at tick_end^-
    btn_multi_hot: np.ndarray         # [5] float32
    wheel: int                        # 0=none, 1=up, 2=down


N_VK = 256
MOUSE_BUTTON_NAMES = ["left", "middle", "right", "x1", "x2"]
N_MOUSE_BUTTONS = 5


def _kbd_to_multi_hot(held: frozenset[int]) -> np.ndarray:
    v = np.zeros(N_VK, dtype=np.float32)
    for vk in held:
        if 0 <= vk < N_VK:
            v[vk] = 1.0
    return v


def _btn_to_multi_hot(buttons: frozenset[str]) -> np.ndarray:
    v = np.zeros(N_MOUSE_BUTTONS, dtype=np.float32)
    for i, name in enumerate(MOUSE_BUTTON_NAMES):
        if name in buttons:
            v[i] = 1.0
    return v


def reconstruct_keyboard_timeline(
    kbd_events: list[KeyboardEvent],
    duration_ns: int,
    tick_hz: int = 60,
    start_ns: int = 0,
) -> list[frozenset[int]]:
    """Rebuild exact keyboard held-state at each tick boundary from events.

    Returns list of length n_ticks; element i = held keys at tick i end boundary.
    The event stream is ground truth; keyboard/state is only for validation.
    """
    tick_ns = int(1e9 / tick_hz)
    n_ticks = duration_ns // tick_ns
    held: set[int] = set()
    timeline: list[frozenset[int]] = []
    ei = 0
    events_sorted = sorted(kbd_events, key=lambda e: e.time_ns)

    for i in range(n_ticks):
        t_end = start_ns + (i + 1) * tick_ns
        while ei < len(events_sorted) and events_sorted[ei].time_ns < t_end:
            ev = events_sorted[ei]
            if ev.event_type == "press":
                held.add(ev.vk)
            elif ev.event_type == "release":
                held.discard(ev.vk)
            ei += 1
        timeline.append(frozenset(held))
    return timeline


def reconstruct_press_release_events(
    kbd_events: list[KeyboardEvent],
    duration_ns: int,
    tick_hz: int = 60,
    start_ns: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-tick press/release event labels for auxiliary heads.

    Returns:
        press_events:   [n_ticks, 256]  1.0 if key vk was pressed during that tick
        release_events: [n_ticks, 256]  1.0 if key vk was released during that tick
    """
    tick_ns = int(1e9 / tick_hz)
    n_ticks = duration_ns // tick_ns
    press = np.zeros((n_ticks, N_VK), dtype=np.float32)
    release = np.zeros((n_ticks, N_VK), dtype=np.float32)
    events_sorted = sorted(kbd_events, key=lambda e: e.time_ns)

    for ev in events_sorted:
        rel_tick = (ev.time_ns - start_ns) // tick_ns
        if rel_tick < 0 or rel_tick >= n_ticks:
            continue
        if 0 <= ev.vk < N_VK:
            if ev.event_type == "press":
                press[rel_tick, ev.vk] = 1.0
            elif ev.event_type == "release":
                release[rel_tick, ev.vk] = 1.0
    return press, release


def aggregate_mouse_raw(
    raw_events: list[RawMouseEvent],
    duration_ns: int,
    tick_hz: int = 60,
    start_ns: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate HID mouse deltas per tick window.

    Returns:
        dx_per_tick: [n_ticks] int32
        dy_per_tick: [n_ticks] int32
    """
    tick_ns = int(1e9 / tick_hz)
    n_ticks = duration_ns // tick_ns
    dx_arr = np.zeros(n_ticks, dtype=np.int32)
    dy_arr = np.zeros(n_ticks, dtype=np.int32)
    for ev in raw_events:
        rel_tick = (ev.time_ns - start_ns) // tick_ns
        if rel_tick < 0 or rel_tick >= n_ticks:
            continue
        dx_arr[rel_tick] += ev.dx
        dy_arr[rel_tick] += ev.dy
    return dx_arr, dy_arr


def reconstruct_mouse_buttons_timeline(
    mouse_events: list,
    duration_ns: int,
    tick_hz: int = 60,
    start_ns: int = 0,
) -> list[frozenset[str]]:
    """Rebuild mouse button held-state at each tick boundary from mouse events."""
    tick_ns = int(1e9 / tick_hz)
    n_ticks = duration_ns // tick_ns
    held: set[str] = set()
    timeline: list[frozenset[str]] = []
    ei = 0
    events_sorted = sorted(mouse_events, key=lambda e: e.time_ns)

    for i in range(n_ticks):
        t_end = start_ns + (i + 1) * tick_ns
        while ei < len(events_sorted) and events_sorted[ei].time_ns < t_end:
            ev = events_sorted[ei]
            if ev.event_type == "press" and ev.button:
                held.add(ev.button)
            elif ev.event_type == "release" and ev.button:
                held.discard(ev.button)
            ei += 1
        timeline.append(frozenset(held))
    return timeline


def aggregate_wheel(
    mouse_events: list,
    duration_ns: int,
    tick_hz: int = 60,
    start_ns: int = 0,
) -> np.ndarray:
    """Per-tick wheel classification: 0=none, 1=up, 2=down.

    If both up and down events occur in the same tick, net direction wins.
    """
    tick_ns = int(1e9 / tick_hz)
    n_ticks = duration_ns // tick_ns
    wheel = np.zeros(n_ticks, dtype=np.int64)  # 0=none
    net = np.zeros(n_ticks, dtype=np.int32)

    for ev in sorted(mouse_events, key=lambda e: e.time_ns):
        if ev.event_type == "vertical_wheel":
            rel_tick = (ev.time_ns - start_ns) // tick_ns
            if 0 <= rel_tick < n_ticks:
                net[rel_tick] += ev.dy  # positive=up, negative=down (convention)

    wheel[net > 0] = 1   # up
    wheel[net < 0] = 2   # down
    return wheel


def validate_keyboard_reconstruction(
    reconstructed: list[frozenset[int]],
    kbd_state_snapshots: list,
    tick_hz: int = 60,
    start_ns: int = 0,
) -> bool:
    """Sanity-check: compare reconstructed state against 1Hz keyboard/state snapshots.

    Returns True if all snapshots match.
    """
    tick_ns = int(1e9 / tick_hz)
    ok = True
    for snap in kbd_state_snapshots:
        rel_tick = (snap.time_ns - start_ns) // tick_ns
        if rel_tick < 0 or rel_tick >= len(reconstructed):
            continue
        recon = reconstructed[rel_tick]
        expected = snap.buttons
        if recon != expected:
            ok = False
    return ok


def build_tick_actions(
    episode: EpisodeData,
    tick_hz: int = 60,
) -> list[TickAction]:
    """Full pipeline: episode → list of TickAction (one per tick).

    Uses the screen topic's first pts_ns as the time origin (t=0).
    Duration = last screen pts - first screen pts.
    """
    if not episode.screen:
        return []

    # Use the recorder monotonic clock (log_time) as the primary time domain,
    # since all keyboard/mouse events use log_time. Screen pts_ns is only for
    # video frame extraction (see align.py).
    start_ns = episode.screen[0].log_time_ns
    duration_ns = episode.screen[-1].log_time_ns - start_ns

    kbd_timeline = reconstruct_keyboard_timeline(
        episode.keyboard_events, duration_ns, tick_hz, start_ns
    )
    press_ev, release_ev = reconstruct_press_release_events(
        episode.keyboard_events, duration_ns, tick_hz, start_ns
    )
    dx_arr, dy_arr = aggregate_mouse_raw(
        episode.raw_mouse_events, duration_ns, tick_hz, start_ns
    )
    btn_timeline = reconstruct_mouse_buttons_timeline(
        episode.mouse_events, duration_ns, tick_hz, start_ns
    )
    wheel_arr = aggregate_wheel(
        episode.mouse_events, duration_ns, tick_hz, start_ns
    )

    tick_ns = int(1e9 / tick_hz)
    n_ticks = min(len(kbd_timeline), len(dx_arr), len(btn_timeline), len(wheel_arr))

    actions: list[TickAction] = []
    for i in range(n_ticks):
        t_start = start_ns + i * tick_ns
        t_end = t_start + tick_ns
        actions.append(TickAction(
            tick_start_ns=t_start,
            tick_end_ns=t_end,
            kbd_held=kbd_timeline[i],
            kbd_multi_hot=_kbd_to_multi_hot(kbd_timeline[i]),
            press_events=press_ev[i],
            release_events=release_ev[i],
            mouse_dx=int(dx_arr[i]),
            mouse_dy=int(dy_arr[i]),
            mouse_buttons=btn_timeline[i],
            btn_multi_hot=_btn_to_multi_hot(btn_timeline[i]),
            wheel=int(wheel_arr[i]),
        ))
    return actions
