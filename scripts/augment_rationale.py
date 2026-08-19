#!/usr/bin/env python3
"""Data augmentation pipeline v2: split describe / explain across two models.

Stage A (Mage-VL, visual):    clip frames → OPERATIONAL description
                              (what moves, where, UI changes — no motivation)
Stage B (strong LLM, text):   description + ground-truth facts
                              → INTENTION + RATIONALE ("why")

The strong LLM is queried via OpenAI-compatible API (vLLM / any provider):
    --api-base http://localhost:8000/v1 --api-model Qwen3-32B

Launch pattern (4xH100 machine):
  # terminal 1: vLLM serving the strong LLM (2 or 3 GPUs)
  vllm serve Qwen/Qwen3-32B --tensor-parallel-size 2 --port 8000
  # terminal 2+: sharded describe jobs on remaining GPUs
  for i in 1 2; do
    CUDA_VISIBLE_DEVICES=$i python scripts/augment_rationale.py \
        --stage describe --shard $i --num-shards 2 ... &
  done
  # then one process for the explain stage (API-bound, no GPU)
  python scripts/augment_rationale.py --stage explain ...
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
    # modifiers (frequent in FPS: sprint/crouch/walk)
    0xA0: "LSHIFT", 0xA1: "RSHIFT", 0xA2: "LCTRL", 0xA3: "RCTRL",
    0xA4: "LALT", 0xA5: "RALT",
    0x70: "F1", 0x71: "F2", 0x72: "F3", 0x73: "F4", 0x74: "F5", 0x75: "F6",
    0x76: "F7", 0x77: "F8", 0x78: "F9", 0x79: "F10", 0x7A: "F11", 0x7B: "F12",
    0x4D: "M", 0x50: "P", 0x47: "G", 0x58: "X", 0x5A: "Z", 0x42: "B", 0x4E: "N",
    0x36: "6", 0x37: "7", 0x38: "8", 0x39: "9", 0x30: "0",
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
    n = len(actions)
    starts = list(range(0, max(0, n - seg_ticks), stride_ticks))
    scores = np.array([
        len(a.kbd_held) + abs(a.mouse_dx) / 20.0 + abs(a.mouse_dy) / 20.0
        for a in actions
    ])
    smooth = np.convolve(scores, np.ones(seg_ticks) / seg_ticks, mode="valid")
    added = 0
    for idx in np.argsort(smooth)[::-1]:
        idx = int(idx)
        if all(abs(idx - s) >= seg_ticks for s in starts):
            starts.append(idx)
            added += 1
        if added >= n_peaks:
            break
    return sorted(set(s for s in starts if 0 <= s < n - seg_ticks))


# ---------------------------------------------------------- hallucination guards

def entity_overlap_check(description: str, rationale: str, facts: str = "") -> bool:
    stop = {"the", "a", "an", "is", "are", "was", "to", "of", "in", "on",
            "and", "or", "for", "with", "that", "this", "it", "its", "as",
            "be", "by", "at", "from", "he", "she", "they", "player", "game",
            "enemy", "enemies", "weapon", "gun", "screen", "move", "moving",
            "press", "pressing", "presses", "release", "releasing",
            "holding", "held", "key", "keys", "mouse", "keyboard", "aim",
            "aiming", "attack", "attacking", "navigate", "navigating",
            "using", "use", "control", "character", "movement", "because",
            "likely", "probably", "order", "trying", "would", "could"}
    words = lambda s: set(w for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in stop)
    novel = words(rationale) - words(description) - words(facts)
    return len(novel) <= 4


ACTION_VERBS = {
    "attack": ["j", "k", "l", "space", "left", "mouse"],
    "health": [], "heal": [], "jump": ["space", "w"],
    "reload": ["r"], "interact": ["e", "f"], "sprint": ["shift"],
    "crouch": ["ctrl", "c"], "ability": ["q", "e", "r", "f", "1", "2", "3", "4"],
    "spell": ["q", "e", "r", "f", "1", "2", "3", "4"],
    "shoot": ["left"], "fire": ["left"], "aim": ["right"],
}


def key_hallucination_check(rationale: str, facts: str) -> bool:
    rl = rationale.lower()
    gt_keys = set()
    for m in re.finditer(r"\b([A-Z]{1,10})\(", facts):
        gt_keys.add(m.group(1).lower())
    for m in re.finditer(r"Presses: ([^\n]+)", facts):
        gt_keys.update(k.strip().lower() for k in m.group(1).split(",") if k.strip())
    for m in re.finditer(r"Releases: ([^\n]+)", facts):
        gt_keys.update(k.strip().lower() for k in m.group(1).split(",") if k.strip())
    for letter in "abcdefghijklmnopqrstuvwxyz":
        if re.search(rf"\b{letter}\s+key\b", rl) and letter not in gt_keys:
            return True
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


# ---------------------------------------------------------- describe stage

DESC_PROMPT = (
    "Watch this gameplay clip and describe ONLY what is observably happening, "
    "in 2-4 sentences. Focus on OPERATIONS, not motivation:\n"
    "- What the player-character does: movement direction, speed, actions "
    "(running, shooting, jumping, interacting)\n"
    "- Camera/view changes (turning, aiming, zooming)\n"
    "- UI state and changes (menus, inventory, health bar, scores, timers)\n"
    "- Which of these are visibly controlled by the player vs automatic\n"
    "Do NOT guess reasons or strategy. Do NOT mention specific keyboard keys "
    "(you cannot see them). Just describe visible behavior."
)


def truncate_repetition(text: str, max_repeat: int = 2) -> str:
    """Cut degenerate repetition loops ('then moves forward again... again...').

    Splits into sentences; when a sentence repeats, keeps at most `max_repeat`
    occurrences and truncates there. Returns cleaned text.
    """
    import re
    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # split on sentence boundary followed by capital (crude but effective)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    if len(sentences) < 4:
        return text
    seen: dict[str, int] = {}
    out = []
    for s in sentences:
        key = re.sub(r"\W+", "", s.lower())[:80]  # fuzzy sentence key
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > max_repeat:
            break
        out.append(s)
    return " ".join(out).strip()


def run_describe(args):
    """Stage A: Mage-VL → operational descriptions. Output: describe.jsonl"""
    import torch
    import cv2
    from transformers import AutoModelForCausalLM, AutoProcessor

    CACHE_STRIDE = 8
    seg_ticks = args.segment_frames * CACHE_STRIDE
    stride_ticks = int(args.stride_seconds * 60)

    print(f"[describe shard {args.shard}/{args.num_shards}] loading Mage-VL...", flush=True)
    processor = AutoProcessor.from_pretrained(args.magevl_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.magevl_dir, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda:0",
    ).eval()

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
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done_episodes.add(json.loads(line)["episode_id"])
                except Exception:
                    pass

    cur_frames: list = []

    def generate_desc(frames):
        messages = [{"role": "user", "content": [
            {"type": "video"}, {"type": "text", "text": DESC_PROMPT},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = processor(text=[text], videos=[frames],
                           return_tensors="pt", padding=True)
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=180, do_sample=False)
        return processor.tokenizer.decode(
            out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

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
                print(f"[describe] SKIP {mcap_path.name}: {e}", flush=True)
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
                t0, t1 = st / 60, (st + seg_ticks) / 60
                facts = actions_to_facts(actions, st, seg_ticks)
                try:
                    desc = truncate_repetition(generate_desc(frames))
                except Exception as e:
                    print(f"  desc fail @{t0:.1f}s: {e}", flush=True)
                    continue
                ep_records.append({
                    "episode_id": mcap_path.stem,
                    "game": game,
                    "t_start_s": round(t0, 2),
                    "t_end_s": round(t1, 2),
                    "frame_ref": str(cache_dir),
                    "description": desc,
                    "facts": facts,
                })

            for r in ep_records:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            n_out += len(ep_records)
            print(f"[describe] {game}/{mcap_path.stem}: {len(ep_records)} segs "
                  f"(total {n_out})", flush=True)

    print(f"[describe] DONE: {n_out} → {out_path}", flush=True)


# ---------------------------------------------------------- explain stage

EXPLAIN_SYSTEM = (
    "You are an expert gamer analyzing recorded gameplay. You receive:\n"
    "(1) A visual description of observable behavior in a clip.\n"
    "(2) The player's EXACT recorded keyboard/mouse inputs (ground truth).\n"
    "Your job: infer WHY the player made these inputs — tactical reasoning, "
    "game-state response, or UI navigation logic.\n"
    "Hard rules:\n"
    "- The listed inputs are ALL the inputs. Never mention a key/click/scroll "
    "that does not appear in the recorded inputs.\n"
    "- Behavior with no corresponding input (e.g. auto-attacks) is game "
    "automatics, not player action.\n"
    "- Only reference entities/events present in the visual description.\n"
    "- If the description CONTRADICTS the inputs (e.g. described movement but "
    "no keys pressed, or described stillness but W held + large mouse deltas), "
    "start RATIONALE with exactly 'INCONSISTENT:' followed by one short note "
    "(likely spectator/replay/cutscene, death cam, menu overlay, or sampling "
    "miss). Do NOT force an explanation.\n"
    "- If inputs look random or you cannot find a coherent reason, say so "
    "plainly — do not invent one.\n"
    "Answer in this exact format:\n"
    "INTENTION: <one sentence: what the player is trying to achieve; "
    "'unclear' if inconsistent>\n"
    "RATIONALE: <2-4 sentences: why these specific inputs accomplish it, "
    "grounded in the described scene state>"
)


def run_explain(args):
    """Stage B: strong LLM (API) → intention + rationale. Output: final.jsonl"""
    from openai import OpenAI

    client = OpenAI(base_url=args.api_base, api_key=args.api_key or "EMPTY")

    in_path = Path(args.describe_output or args.output)
    out_path = Path(args.output)
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["episode_id"])
                except Exception:
                    pass

    records = []
    with open(in_path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    print(f"[explain] {len(records)} records in, {len(done)} episodes already done",
          flush=True)

    n_out = 0
    with open(out_path, "a") as fout:
        for i, r in enumerate(records):
            if r["episode_id"] in done:
                continue
            prompt = (
                f"### Visual description (observable behavior)\n{r['description']}\n\n"
                f"### Recorded inputs (ground truth, exact and complete)\n{r['facts']}"
            )
            try:
                resp = client.chat.completions.create(
                    model=args.api_model,
                    messages=[
                        {"role": "system", "content": EXPLAIN_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=320, temperature=0.2,
                )
                rat = resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"  explain fail [{i}] {r['episode_id']}@{r['t_start_s']}s: {e}",
                      flush=True)
                continue

            r["intention"] = (rat.split("RATIONALE:")[0]
                              .replace("INTENTION:", "").strip()
                              if "INTENTION:" in rat else "")
            r["rationale"] = (rat.split("RATIONALE:")[-1].strip()
                              if "RATIONALE:" in rat else rat)
            r["entity_overlap_ok"] = entity_overlap_check(
                r["description"], r["rationale"], r["facts"])
            r["key_hallucination"] = key_hallucination_check(
                r["rationale"], r["facts"])
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_out += 1
            if n_out % 50 == 0:
                fout.flush()
                print(f"[explain] {n_out} done", flush=True)

        fout.flush()
    print(f"[explain] DONE: {n_out} → {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["describe", "explain", "both"],
                    default="both",
                    help="describe=Mage-VL visual; explain=strong LLM API; both=legacy single-model")
    # data
    ap.add_argument("--d2e-dir", required=True)
    ap.add_argument("--magevl-dir", required=True)
    ap.add_argument("--frame-cache", required=True)
    ap.add_argument("--games", default=None)
    ap.add_argument("--segment-frames", type=int, default=32)
    ap.add_argument("--stride-seconds", type=float, default=60.0)
    ap.add_argument("--peaks-per-episode", type=int, default=6)
    ap.add_argument("--max-segments", type=int, default=0)
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    # describe stage
    ap.add_argument("--output", required=True)
    # explain stage
    ap.add_argument("--describe-output", default=None,
                    help="input jsonl from describe stage (default: <output>.describe.jsonl)")
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--api-model", default="Qwen/Qwen3-32B")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    args = ap.parse_args()

    if args.stage == "explain" and not args.describe_output:
        args.describe_output = str(Path(args.output).with_suffix("")) + ".describe.jsonl"
    if args.stage == "describe" and not args.describe_output:
        args.describe_output = args.output

    if args.stage in ("describe", "both"):
        run_describe(args)
    if args.stage in ("explain", "both"):
        run_explain(args)


if __name__ == "__main__":
    main()
