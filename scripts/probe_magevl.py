#!/usr/bin/env python3
"""Probe: can Mage-VL understand gameplay footage?

Samples short clips from D2E episodes around keyboard/mouse activity peaks,
asks Mage-VL to describe what's happening, and prints side-by-side with
ground-truth action facts (structured reconstruction from the event stream).

Usage:
    python scripts/probe_magevl.py \
        --d2e-dir /home/jovyan/exploitgym/D2E-Original \
        --magevl-dir /home/jovyan/exploitgym/Mage-VL \
        --frame-cache /home/jovyan/exploitgym/frame_cache \
        --games "Brotato,Counter-Strike_2,Stardew_Valley" \
        --episodes-per-game 1 \
        --clips-per-episode 4 \
        --output probes/probe_results.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sn_vla.data.mcap_reader import parse_mcap
from sn_vla.data.manifest import find_episodes
from sn_vla.data.reconstruct import build_tick_actions

VK_NAMES = {
    0x57: "W", 0x41: "A", 0x53: "S", 0x44: "D", 0x20: "SPACE", 0x10: "SHIFT",
    0x11: "CTRL", 0x12: "ALT", 0x0D: "ENTER", 0x1B: "ESC", 0x09: "TAB",
    0x45: "E", 0x51: "Q", 0x52: "R", 0x46: "F", 0x43: "C", 0x56: "V",
    0x31: "1", 0x32: "2", 0x33: "3", 0x34: "4", 0x35: "5",
}


def vk_name(vk: int) -> str:
    return VK_NAMES.get(vk, f"vk{vk}")


def find_activity_peaks(actions, clip_s: float, n_clips: int) -> list[int]:
    """Find tick indices with high keyboard/mouse activity (good probe spots)."""
    tick_hz = 60
    win = int(clip_s * tick_hz)
    scores = np.zeros(len(actions))
    for i, a in enumerate(actions):
        sc = len(a.kbd_held) + abs(a.mouse_dx) / 20.0 + abs(a.mouse_dy) / 20.0
        scores[i] = sc
    # Smooth over clip window and pick non-overlapping top peaks
    kernel = np.ones(win) / win
    smooth = np.convolve(scores, kernel, mode="valid")
    order = np.argsort(smooth)[::-1]
    chosen = []
    for idx in order:
        if all(abs(idx - c) >= win for c in chosen):
            chosen.append(int(idx))
        if len(chosen) >= n_clips:
            break
    return sorted(chosen)


def actions_to_facts(actions, start_tick: int, end_tick: int) -> str:
    """Structured ground-truth facts for [start_tick, end_tick)."""
    seg = actions[start_tick:end_tick]
    press_events = []
    release_events = []
    held_counts: dict[int, int] = {}
    dx_sum = sum(a.mouse_dx for a in seg)
    dy_sum = sum(a.mouse_dy for a in seg)
    clicks = sum(1 for a in seg if len(a.mouse_buttons) > 0)

    prev_held = set()
    for a in seg:
        cur = set(a.kbd_held)
        for vk in cur - prev_held:
            press_events.append(vk)
        for vk in prev_held - cur:
            release_events.append(vk)
        prev_held = cur
        for vk in cur:
            held_counts[vk] = held_counts.get(vk, 0) + 1

    lines = []
    dur = (end_tick - start_tick) / 60
    lines.append(f"Duration: {dur:.1f}s ({end_tick - start_tick} ticks @60Hz)")
    if held_counts:
        top = sorted(held_counts.items(), key=lambda x: -x[1])[:6]
        held_str = ", ".join(
            f"{vk_name(vk)}({cnt/60:.1f}s)" for vk, cnt in top
        )
        lines.append(f"Keys held: {held_str}")
    else:
        lines.append("Keys held: (none)")
    if press_events:
        lines.append(f"Key presses: {', '.join(vk_name(v) for v in press_events[:12])}")
    if release_events:
        lines.append(f"Key releases: {', '.join(vk_name(v) for v in release_events[:12])}")
    lines.append(f"Mouse: total dx={dx_sum:+d}px dy={dy_sum:+d}px, button-held ticks={clicks}")
    if abs(dx_sum) > 30 or abs(dy_sum) > 30:
        lines.append(f"  → significant mouse movement ({abs(dx_sum)+abs(dy_sum)}px total)")
    if clicks > 10:
        lines.append(f"  → clicking/holding ({clicks} ticks with buttons)")
    return "\n".join(lines)


def load_clip_frames_from_cache(mkv_path: Path, frame_cache: Path,
                                 start_frame: int, n_frames: int,
                                 stride: int = 8, size: int = 224):
    """Load n_frames cached JPEGs (already stride-8 subsampled)."""
    import cv2
    cache_dir = frame_cache / mkv_path.parent.name / mkv_path.stem
    frames = []
    j_start = start_frame // stride
    for j in range(j_start, j_start + n_frames):
        fp = cache_dir / f"f_{j + 1:06d}.jpg"
        if not fp.exists():
            continue
        img = cv2.imread(str(fp))
        if img is not None:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


def load_clip_frames_from_mkv(mkv_path: Path, start_frame: int, n_frames: int):
    """Fallback: extract frames directly from mkv."""
    import cv2
    cap = cv2.VideoCapture(str(mkv_path))
    frames = []
    for i in range(n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i)
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d2e-dir", required=True)
    ap.add_argument("--magevl-dir", required=True)
    ap.add_argument("--frame-cache", default=None)
    ap.add_argument("--games", required=True, help="Comma-separated game names")
    ap.add_argument("--episodes-per-game", type=int, default=1)
    ap.add_argument("--clips-per-episode", type=int, default=4)
    ap.add_argument("--clip-seconds", type=float, default=4.0)
    ap.add_argument("--frames-per-clip", type=int, default=16)
    ap.add_argument("--output", default="probes/probe_results.md")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    game_filter = [g.strip() for g in args.games.split(",")]

    # === Model ===
    print("Loading Mage-VL...")
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.magevl_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.magevl_dir, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map=args.device,
    ).eval()

    # === Collect episodes ===
    pairs = find_episodes(args.d2e_dir)
    by_game = {}
    for mcap, mkv in pairs:
        by_game.setdefault(mcap.parent.name, []).append((mcap, mkv))

    results = []
    for game in game_filter:
        eps = by_game.get(game, [])[:args.episodes_per_game]
        if not eps:
            print(f"WARNING: no episodes for {game}")
            continue
        for mcap_path, mkv_path in eps:
            print(f"\n=== {game} / {mcap_path.stem} ===")
            ep = parse_mcap(mcap_path)
            actions = build_tick_actions(ep, tick_hz=60)
            if not actions:
                print("  no actions, skip")
                continue

            peaks = find_activity_peaks(actions, args.clip_seconds, args.clips_per_episode)
            for peak in peaks:
                start_tick = max(0, peak)
                end_tick = min(len(actions), start_tick + int(args.clip_seconds * 60))
                facts = actions_to_facts(actions, start_tick, end_tick)

                # Load frames: cache stride=8, so take every 8th tick frame
                start_frame = start_tick  # 1 tick = 1 frame @60fps
                if args.frame_cache:
                    frames = load_clip_frames_from_cache(
                        Path(mkv_path), Path(args.frame_cache),
                        start_frame, args.frames_per_clip,
                    )
                else:
                    frames = load_clip_frames_from_mkv(
                        Path(mkv_path), start_frame, args.frames_per_clip * 8
                    )
                if len(frames) < 4:
                    print(f"  clip@{start_tick}: no frames, skip")
                    continue

                # === Ask Mage-VL ===
                t_sec = start_tick / 60
                prompt = (
                    "Watch this gameplay clip and answer two questions concisely:\n"
                    "1. What is happening on screen? (game state, player/character actions, "
                    "visible events, UI changes)\n"
                    "2. What keyboard/mouse inputs is the player most likely performing "
                    "right now? (movement keys, camera/mouse movement, clicks)"
                )
                messages = [{"role": "user", "content": [
                    {"type": "video"}, {"type": "text", "text": prompt},
                ]}]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                try:
                    inputs = processor(
                        text=[text], videos=[frames],
                        return_tensors="pt", padding=True,
                    )
                except Exception as e:
                    # some checkpoints want video_backend
                    try:
                        inputs = processor(
                            text=[text], videos=[frames], video_backend="frames",
                            return_tensors="pt", padding=True,
                        )
                    except Exception as e2:
                        print(f"  clip@{t_sec:.1f}s: processor error: {e2}")
                        continue
                inputs = {k: (v.to(model.device) if hasattr(v, "to") else v)
                          for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

                with torch.inference_mode():
                    out = model.generate(
                        **inputs, max_new_tokens=220, do_sample=False
                    )
                desc = processor.tokenizer.decode(
                    out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()

                results.append({
                    "game": game, "episode": mcap_path.stem,
                    "t_start_s": round(t_sec, 1),
                    "frames_used": len(frames),
                    "description": desc,
                    "facts": facts,
                })
                print(f"  clip@{t_sec:.1f}s ({len(frames)} frames)")
                print(f"  DESC: {desc[:200]}")
                print(f"  FACTS: {facts.splitlines()[0]}")

    # === Write report ===
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# Mage-VL Gameplay Understanding Probe\n\n")
        f.write(f"Games: {game_filter} | clips: {len(results)}\n\n")
        f.write("Legend: DESC = Mage-VL output | FACTS = ground-truth from event stream\n\n---\n\n")
        for i, r in enumerate(results):
            f.write(f"## Clip {i+1}: {r['game']} @ {r['t_start_s']}s\n\n")
            f.write(f"**DESC (Mage-VL):**\n\n> {r['description']}\n\n")
            f.write(f"**FACTS (ground truth):**\n\n```\n{r['facts']}\n```\n\n---\n\n")
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()
