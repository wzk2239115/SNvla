#!/usr/bin/env python3
"""Data augmentation pipeline: video captions + action rationales.

Stage A (visual): Mage-VL watches clip frames (from JPEG cache) → scene description.
Stage B (text):   same model, text-only, combines description + ground-truth
                  action facts → intention label + rationale + consistency check.

Anti-hallucination guards:
  - Facts are deterministic (event-stream reconstruction), never LLM-invented.
  - Rationale prompt forbids introducing entities absent from the description.
  - Post-hoc entity-overlap check; failing samples are flagged low-confidence.

Output: JSONL, one line per segment:
  {episode_id, game, t_start_s, t_end_s, frame_ref, description, facts,
   intention, rationale, consistency_ok, entity_overlap}

Launch 4 workers (one per GPU) on the 4xH100 machine:
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python scripts/augment_rationale.py \
        --d2e-dir ... --magevl-dir ... --frame-cache ... \
        --shard $i --num-shards 4 \
        --output probes/aug_shard$i.jsonl &
  done; wait
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
    0x26: "UP", 0x28: "DOWN", 0x25: "LEFT", 0x27: "RIGHT",
}


def vk_name(vk: int) -> str:
    return VK_NAMES.get(vk, f"vk{vk}")


def actions_to_facts(actions, start_tick: int, n_ticks: int) -> str:
    """Deterministic structured facts for [start_tick, start_tick+n_ticks)."""
    seg = actions[start_tick:start_tick + n_ticks]
    press_events, release_events = [], []
    held_counts: dict[int, int] = {}
    dx_sum = sum(a.mouse_dx for a in seg)
    dy_sum = sum(a.mouse_dy for a in seg)
    clicks = sum(1 for a in seg if len(a.mouse_buttons) > 0)
    wheel_up = sum(1 for a in seg if a.wheel == 1)
    wheel_dn = sum(1 for a in seg if a.wheel == 2)

    prev_held = set()
    for a in seg:
        cur = set(a.kbd_held)
        press_events.extend(cur - prev_held)
        release_events.extend(prev_held - cur)
        prev_held = cur
        for vk in cur:
            held_counts[vk] = held_counts.get(vk, 0) + 1

    dur = n_ticks / 60
    lines = [f"Duration: {dur:.1f}s @60Hz"]
    if held_counts:
        top = sorted(held_counts.items(), key=lambda x: -x[1])[:8]
        lines.append("Keys held: " + ", ".join(
            f"{vk_name(v)}({c/60:.1f}s)" for v, c in top))
    else:
        lines.append("Keys held: none")
    if press_events:
        lines.append("Presses: " + ", ".join(vk_name(v) for v in press_events[:15]))
    if release_events:
        lines.append("Releases: " + ", ".join(vk_name(v) for v in release_events[:15]))
    mparts = [f"mouse dx={dx_sum:+d}px dy={dy_sum:+d}px"]
    if clicks > 5:
        mparts.append(f"buttons held {clicks} ticks")
    if wheel_up or wheel_dn:
        mparts.append(f"wheel up:{wheel_up} down:{wheel_dn}")
    lines.append(" | ".join(mparts))
    return "\n".join(lines)


def select_segments(actions, seg_ticks: int, stride_ticks: int,
                    n_peaks: int) -> list[int]:
    """Mix of uniform coverage + activity peaks. Returns start ticks, sorted."""
    n = len(actions)
    starts = list(range(0, max(0, n - seg_ticks), stride_ticks))
    # Activity peaks
    scores = np.array([
        len(a.kbd_held) + abs(a.mouse_dx) / 20.0 + abs(a.mouse_dy) / 20.0
        for a in actions
    ])
    smooth = np.convolve(scores, np.ones(seg_ticks) / seg_ticks, mode="valid")
    for idx in np.argsort(smooth)[::-1][: n_peaks * 3]:
        idx = int(idx)
        if all(abs(idx - s) >= seg_ticks for s in starts):
            starts.append(idx)
        if len(starts) >= len(range(0, n, stride_ticks)) + n_peaks:
            break
    return sorted(set(s for s in starts if 0 <= s < n - seg_ticks))


def entity_overlap_check(description: str, rationale: str, facts: str = "") -> bool:
    """Cheap guard: nouns in rationale that appear nowhere in description/facts.

    Uses a small stopword list; flags samples with many novel entities.
    """
    import re
    stop = {"the", "a", "an", "is", "are", "was", "to", "of", "in", "on",
            "and", "or", "for", "with", "that", "this", "it", "its", "as",
            "be", "by", "at", "from", "he", "she", "they", "player", "game",
            "enemy", "enemies", "weapon", "gun", "screen", "move", "moving",
            "press", "pressing", "presses", "release", "releasing",
            "holding", "held", "key", "keys", "mouse", "keyboard", "aim",
            "aiming", "attack", "attacking", "navigate", "navigating",
            "using", "use", "using", "control", "character", "movement"}
    words = lambda s: set(w for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in stop)
    novel = words(rationale) - words(description) - words(facts)
    return len(novel) <= 4  # allow a few generic verbs/nouns


ACTION_VERBS = {
    "attack": ["j", "k", "l", "space", "left", "mouse"],
    "health": [], "heal": [], "jump": ["space", "w"],
    "reload": ["r"], "interact": ["e", "f"], "sprint": ["shift"],
    "crouch": ["ctrl", "c"], "ability": ["q", "e", "r", "f", "1", "2", "3", "4"],
    "spell": ["q", "e", "r", "f", "1", "2", "3", "4"],
    "shoot": ["left"], "fire": ["left"], "aim": ["right"],
}


def key_hallucination_check(rationale: str, facts: str) -> bool:
    """Detect fabricated input actions in rationale.

    Two patterns:
    1. Named keys not in ground truth (mentions 'R key' but no R in facts).
    2. Action-verb + 'key' phrasing ('pressing the attack key') where the
       facts contain none of the keys commonly bound to that action AND the
       game inputs are pure movement (WASD-only) — a strong hallucination sign.
    """
    import re
    rl = rationale.lower()

    # --- Pattern 1: named single-letter/space keys absent from facts ---
    gt_keys = set()
    for m in re.finditer(r"\b([A-Z]{1,10})\(", facts):
        gt_keys.add(m.group(1).lower())
    for m in re.finditer(r"Presses: ([^\n]+)", facts):
        gt_keys.update(k.strip().lower() for k in m.group(1).split(",") if k.strip())
    for m in re.finditer(r"Releases: ([^\n]+)", facts):
        gt_keys.update(k.strip().lower() for k in m.group(1).split(",") if k.strip())
    if "space" in rl and "space" not in gt_keys and "space" in facts.lower() is False:
        pass  # 'space' as word is ambiguous; handled by pattern 2
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ".lower():
        if re.search(rf"\b{letter}\s+key\b", rl) and letter not in gt_keys:
            return True

    # --- Pattern 2: '<action> key' with no plausible binding in facts ---
    facts_lower = facts.lower()
    clicks_present = "buttons held" in facts_lower and "buttons held 0" not in facts_lower
    for verb, bindings in ACTION_VERBS.items():
        if re.search(rf"\b{verb}\s+key", rl) or re.search(rf"\b{verb}\s+button", rl):
            has_binding = any(b in gt_keys for b in bindings)
            if verb in ("shoot", "fire", "aim") and clicks_present:
                has_binding = True
            if not has_binding:
                return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d2e-dir", required=True)
    ap.add_argument("--magevl-dir", required=True)
    ap.add_argument("--frame-cache", required=True)
    ap.add_argument("--games", default=None, help="Comma-separated filter")
    ap.add_argument("--segment-frames", type=int, default=32,
                    help="cache frames per segment (32×8=256 ticks ≈ 4.3s)")
    ap.add_argument("--stride-seconds", type=float, default=60.0,
                    help="uniform sampling period; peaks added on top")
    ap.add_argument("--peaks-per-episode", type=int, default=6)
    ap.add_argument("--max-segments", type=int, default=0, help="0 = all")
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    CACHE_STRIDE = 8
    seg_ticks = args.segment_frames * CACHE_STRIDE
    stride_ticks = int(args.stride_seconds * 60)

    # === Model (one per process) ===
    print(f"[shard {args.shard}] loading Mage-VL...", flush=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.magevl_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.magevl_dir, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda:0",
    ).eval()

    def generate(messages, max_new_tokens=256):
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        has_media = any(c.get("type") in ("image", "video")
                        for m in messages for c in m["content"])
        kwargs = dict(text=[text], return_tensors="pt", padding=True)
        if has_media:
            kwargs["videos"] = [cur_frames]
        inputs = processor(**kwargs)
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False)
        return processor.tokenizer.decode(
            out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # === Episodes for this shard ===
    pairs = find_episodes(args.d2e_dir)
    if args.games:
        gf = {g.strip() for g in args.games.split(",")}
        pairs = [p for p in pairs if p[0].parent.name in gf]
    pairs = pairs[args.shard::args.num_shards]
    if args.max_episodes:
        pairs = pairs[:args.max_episodes]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_episodes = set()
    if out_path.exists():  # resume support
        with open(out_path) as f:
            for line in f:
                try:
                    done_episodes.add(json.loads(line)["episode_id"])
                except Exception:
                    pass

    import cv2
    global cur_frames

    n_out = 0
    with open(out_path, "a") as fout:
        for mcap_path, mkv_path in pairs:
            if mcap_path.stem in done_episodes:
                continue
            game = mcap_path.parent.name
            try:
                ep = parse_mcap(mcap_path)
                actions = build_tick_actions(ep, tick_hz=60)
                if len(actions) < seg_ticks * 2:
                    continue
            except Exception as e:
                print(f"[shard {args.shard}] SKIP {mcap_path.name}: {e}", flush=True)
                continue

            cache_dir = Path(args.frame_cache) / game / mkv_path.stem
            segments = select_segments(actions, seg_ticks, stride_ticks,
                                       args.peaks_per_episode)
            if args.max_segments:
                segments = segments[:args.max_segments]

            ep_records = []
            for st in segments:
                frames = []
                for j in range(st // CACHE_STRIDE,
                               st // CACHE_STRIDE + args.segment_frames):
                    fp = cache_dir / f"f_{j + 1:06d}.jpg"
                    if not fp.exists():
                        continue
                    img = cv2.imread(str(fp))
                    if img is not None:
                        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if len(frames) < 8:
                    continue
                cur_frames = frames

                t0, t1 = st / 60, (st + seg_ticks) / 60
                facts = actions_to_facts(actions, st, seg_ticks)

                # ---- Stage A: describe ----
                try:
                    desc = generate([{"role": "user", "content": [
                        {"type": "video"},
                        {"type": "text", "text":
                         "Describe this gameplay clip factually in 2-3 sentences: "
                         "game genre/perspective, what the player-character is doing, "
                         "visible events or UI state, and apparent player intent."},
                    ]}], max_new_tokens=180)
                except Exception as e:
                    print(f"  desc fail @{t0:.1f}s: {e}", flush=True)
                    continue

                # ---- Stage B: rationale (text-only) ----
                try:
                    gt_keys_line = next(
                        (l for l in facts.splitlines() if l.startswith("Keys held")),
                        "Keys held: none")
                    prompt = (
                        "You are analyzing a gameplay clip. You are given:\n"
                        "(1) A visual description of the clip.\n"
                        "(2) The player's EXACT recorded keyboard/mouse inputs.\n\n"
                        f"### Visual description\n{desc}\n\n"
                        f"### Recorded inputs (ground truth, exact and complete)\n{facts}\n\n"
                        f"IMPORTANT: The keys listed above are ALL the keys used. "
                        f"Do NOT mention any key, button, or input that does not "
                        f"appear in the recorded inputs. If an action you want to "
                        f"explain (e.g. attacking) has no corresponding input in "
                        f"the list, treat it as automatic game behavior, not a "
                        f"player key press.\n\n"
                        "Based ONLY on the entities and events in the description, "
                        "answer in this exact format:\n"
                        "INTENTION: <one short sentence: what the player is trying to achieve>\n"
                        "RATIONALE: <2-3 sentences explaining why these specific inputs "
                        "serve that intention. Only reference entities visible in the "
                        "description and keys present in the recorded inputs. If the "
                        "inputs seem random or unrelated to the scene, say so instead "
                        "of inventing a reason.>"
                    )
                    rat = generate([{"role": "user", "content": [
                        {"type": "text", "text": prompt}]}], max_new_tokens=200)
                except Exception as e:
                    print(f"  rat fail @{t0:.1f}s: {e}", flush=True)
                    rat = ""

                ok = entity_overlap_check(desc, rat, facts)
                key_hall = key_hallucination_check(rat, facts)
                ep_records.append({
                    "episode_id": mcap_path.stem,
                    "game": game,
                    "t_start_s": round(t0, 2),
                    "t_end_s": round(t1, 2),
                    "frame_ref": str(cache_dir),
                    "description": desc,
                    "facts": facts,
                    "intention": rat.split("RATIONALE:")[0].replace("INTENTION:", "").strip()
                                  if "INTENTION:" in rat else "",
                    "rationale": rat.split("RATIONALE:")[-1].strip()
                                  if "RATIONALE:" in rat else rat,
                    "entity_overlap_ok": bool(ok),
                    "key_hallucination": bool(key_hall),
                })

            for r in ep_records:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            n_out += len(ep_records)
            print(f"[shard {args.shard}] {game}/{mcap_path.stem}: "
                  f"{len(ep_records)} segments (total {n_out})", flush=True)

    print(f"[shard {args.shard}] DONE: {n_out} segments → {out_path}", flush=True)


if __name__ == "__main__":
    main()
