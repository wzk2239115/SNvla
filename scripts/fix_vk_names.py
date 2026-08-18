#!/usr/bin/env python3
"""Replace raw vk codes in describe-output facts with readable key names.

Option-B fix: the 17,937 already-described segments have facts like
'vk160(0.6s)' which block the explain LLM from reasoning about
sprint/crouch/walk. This rewrites vkNNN -> LSHIFT/LCTRL/... in place.

Usage:
    python scripts/fix_vk_names.py probes/all_desc.jsonl -o probes/all_desc_fixed.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# keep in sync with scripts/augment_rationale.py VK_NAMES
VK_NAMES = {
    0x57: "W", 0x41: "A", 0x53: "S", 0x44: "D", 0x20: "SPACE", 0x10: "SHIFT",
    0x11: "CTRL", 0x12: "ALT", 0x0D: "ENTER", 0x1B: "ESC", 0x09: "TAB",
    0x45: "E", 0x51: "Q", 0x52: "R", 0x46: "F", 0x43: "C", 0x56: "V",
    0x31: "1", 0x32: "2", 0x33: "3", 0x34: "4", 0x35: "5",
    0x26: "UP", 0x28: "DOWN", 0x25: "LEFT", 0x27: "RIGHT",
    0xA0: "LSHIFT", 0xA1: "RSHIFT", 0xA2: "LCTRL", 0xA3: "RCTRL",
    0xA4: "LALT", 0xA5: "RALT",
    0x70: "F1", 0x71: "F2", 0x72: "F3", 0x73: "F4", 0x74: "F5", 0x75: "F6",
    0x76: "F7", 0x77: "F8", 0x78: "F9", 0x79: "F10", 0x7A: "F11", 0x7B: "F12",
    0x4D: "M", 0x50: "P", 0x47: "G", 0x58: "X", 0x5A: "Z", 0x42: "B", 0x4E: "N",
    0x36: "6", 0x37: "7", 0x38: "8", 0x39: "9", 0x30: "0",
}

# vk112 = F21? no — vk codes >= 0x70 are F-keys; anything else keep vkNNN.
PAT = re.compile(r"\bvk(\d+)\b")


def fix_facts(facts: str) -> str:
    def repl(m):
        code = int(m.group(1))
        return VK_NAMES.get(code, m.group(0))
    return PAT.sub(repl, facts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="describe-stage jsonl")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    n = replaced = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            old = rec.get("facts", "")
            new = fix_facts(old)
            if new != old:
                replaced += 1
            rec["facts"] = new
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} records, {replaced} had vk codes replaced → {args.output}")

    # quick sanity: remaining raw vk codes
    with open(args.output) as f:
        left = sum(PAT.search(json.loads(l).get("facts", "")) is not None for l in f)
    print(f"records still containing raw vkNNN: {left} (unknown keys kept as-is)")


if __name__ == "__main__":
    main()
