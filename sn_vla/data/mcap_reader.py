"""OWAMcap file reader.

Parses a .mcap event file into per-topic lists of typed events with nanosecond
timestamps. Decodes JSON payloads for the desktop/* schemas used by ocap/D2E.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from mcap.reader import make_reader


@dataclass
class ScreenMsg:
    log_time_ns: int           # mcap message timestamp (recorder monotonic clock)
    pts_ns: int                # video frame timestamp (0-based, from mkv)
    utc_ns: int                # wall clock
    uri: str                   # mkv file reference
    shape: tuple[int, int]     # (H, W)


@dataclass
class KeyboardEvent:
    time_ns: int              # log_time (monotonic, recorder clock)
    event_type: str           # "press" | "release"
    vk: int                   # Windows virtual key code


@dataclass
class RawMouseEvent:
    time_ns: int
    dx: int                   # HID delta x
    dy: int                   # HID delta y


@dataclass
class MouseEvent:
    time_ns: int
    event_type: str           # "press"/"release"/"double_click"/"vertical_wheel"/...
    button: str               # "left"/"middle"/"right"/"x1"/"x2" or ""
    dx: int = 0               # for wheel: scroll amount
    dy: int = 0


@dataclass
class KeyboardState:
    time_ns: int
    buttons: frozenset[int]   # set of pressed vk codes


@dataclass
class MouseState:
    time_ns: int
    x: int                    # absolute screen x
    y: int                    # absolute screen y
    buttons: frozenset[str]   # {"left", "middle", "right", "x1", "x2"}


@dataclass
class WindowInfo:
    time_ns: int
    title: str
    rect: tuple[int, int, int, int]  # (left, top, right, bottom)
    hwnd: int


@dataclass
class EpisodeData:
    """All parsed data from one .mcap file."""
    mcap_path: Path
    screen: list[ScreenMsg] = field(default_factory=list)
    keyboard_events: list[KeyboardEvent] = field(default_factory=list)
    raw_mouse_events: list[RawMouseEvent] = field(default_factory=list)
    mouse_events: list[MouseEvent] = field(default_factory=list)
    keyboard_states: list[KeyboardState] = field(default_factory=list)
    mouse_states: list[MouseState] = field(default_factory=list)
    window_infos: list[WindowInfo] = field(default_factory=list)
    env_metadata: dict = field(default_factory=dict)

    @property
    def duration_ns(self) -> int:
        if not self.screen:
            return 0
        return self.screen[-1].pts_ns - self.screen[0].pts_ns

    @property
    def fps(self) -> float:
        if len(self.screen) < 2:
            return 0.0
        return (len(self.screen) - 1) / (self.duration_ns / 1e9)


def parse_mcap(mcap_path: str | Path) -> EpisodeData:
    """Parse an OWAMcap .mcap file into typed events.

    Args:
        mcap_path: path to the .mcap file.

    Returns:
        EpisodeData with all topics populated.
    """
    mcap_path = Path(mcap_path)
    ep = EpisodeData(mcap_path=mcap_path)

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        schema_names = {sid: s.name for sid, s in summary.schemas.items()} if summary else {}

        for schema, channel, message in reader.iter_messages():
            if not message.data:
                continue
            data = json.loads(message.data.decode("utf-8"))
            topic = channel.topic
            log_t = message.log_time

            if topic == "screen":
                media_ref = data.get("media_ref", {})
                shape = tuple(data.get("shape", [0, 0]))
                ep.screen.append(ScreenMsg(
                    log_time_ns=log_t,
                    pts_ns=media_ref.get("pts_ns", 0),
                    utc_ns=data.get("utc_ns", 0),
                    uri=media_ref.get("uri", ""),
                    shape=(shape[0], shape[1]),
                ))
            elif topic == "keyboard":
                ep.keyboard_events.append(KeyboardEvent(
                    time_ns=log_t,
                    event_type=data["event_type"],
                    vk=data["vk"],
                ))
            elif topic == "mouse/raw":
                ep.raw_mouse_events.append(RawMouseEvent(
                    time_ns=log_t,
                    dx=data.get("last_x", data.get("dx", 0)),
                    dy=data.get("last_y", data.get("dy", 0)),
                ))
            elif topic == "mouse":
                ep.mouse_events.append(MouseEvent(
                    time_ns=log_t,
                    event_type=data.get("event_type", ""),
                    button=data.get("button", ""),
                    dx=data.get("dx", 0),
                    dy=data.get("dy", 0),
                ))
            elif topic == "keyboard/state":
                ep.keyboard_states.append(KeyboardState(
                    time_ns=log_t,
                    buttons=frozenset(data.get("buttons", [])),
                ))
            elif topic == "mouse/state":
                ep.mouse_states.append(MouseState(
                    time_ns=log_t,
                    x=data.get("x", 0),
                    y=data.get("y", 0),
                    buttons=frozenset(data.get("buttons", [])),
                ))
            elif topic == "window":
                rect = data.get("rect", [0, 0, 0, 0])
                ep.window_infos.append(WindowInfo(
                    time_ns=log_t,
                    title=data.get("title", ""),
                    rect=(rect[0], rect[1], rect[2], rect[3]),
                    hwnd=data.get("hWnd", 0),
                ))

    ep.keyboard_events.sort(key=lambda e: e.time_ns)
    ep.raw_mouse_events.sort(key=lambda e: e.time_ns)
    ep.mouse_events.sort(key=lambda e: e.time_ns)
    return ep
