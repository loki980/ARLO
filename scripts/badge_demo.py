# ruff: noqa: INP001
"""Draw badges on every slot of a capture, the way the live surface would.

Usage: uv run python scripts/badge_demo.py CAPTURE.png OUT.png [--slots S.json]

Finds the slots with grid.find_slots when this branch has it. Otherwise the
slot rects come from a JSON list of [left, top, right, bottom] (or of
{"rect": [...]} objects), given with --slots or, when that is left out,
read from OUT's name with .slots.json in place of .png. Each slot gets a gold
border and a placeholder price tag, drawn by the badge helper's PNG mode and
composited over the capture.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from arc_helper.badge_surface import Border
from arc_helper.badge_surface import Tag
from arc_helper.badge_surface import render_png

PLACEHOLDER_PRICE = "Φ 1k"


def slot_rects(capture: Image.Image, slots_file: Path) -> tuple[list, str]:
    """The slot rects and where they came from."""
    try:
        from arc_helper.grid import find_slots
    except ImportError:
        if not slots_file.is_file():
            msg = f"grid.find_slots is not on this branch and {slots_file} is missing"
            raise SystemExit(msg) from None
        entries = json.loads(slots_file.read_text(encoding="utf-8"))
        rects = [tuple(e["rect"] if isinstance(e, dict) else e) for e in entries]
        return rects, str(slots_file)
    slots = find_slots(np.asarray(capture.convert("RGB")))
    return [tuple(s.rect) for s in slots], "grid.find_slots"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("capture", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--slots", type=Path, help="JSON slot rects, see above")
    args = parser.parse_args()
    capture = Image.open(args.capture).convert("RGBA")
    slots_file = args.slots or args.out.with_suffix(".slots.json")
    rects, source = slot_rects(capture, slots_file)
    badges = []
    for rect in rects:
        badges.extend((Border(rect), Tag(rect, PLACEHOLDER_PRICE)))
    layer_path = args.out.with_suffix(".layer.png")
    if not render_png(badges, capture.size, layer_path):
        print("The badge helper could not draw the PNG, see the log above.")
        return 1
    layer = Image.open(layer_path).convert("RGBA")
    capture.alpha_composite(layer)
    capture.convert("RGB").save(args.out)
    print(f"{len(rects)} slots from {source}, badges drawn to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
