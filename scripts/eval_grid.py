# ruff: noqa: INP001
"""Run the slot lattice over saved full-screen captures.

Usage: uv run python scripts/eval_grid.py <captures folder> <output folder>

Runs find_slots on every 3840x2160 PNG in the captures folder and writes to
the output folder a debug PNG per capture (half size, each found slot boxed
in its panel's colour, a dot in filled slots) and grid.csv with one row per
capture: file, ms for find_slots, slots found per panel and how many of them
are filled. It ends with the median ms over all captures. The captures
folder is only read.
"""

import csv
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from arc_helper.grid import PANELS
from arc_helper.grid import find_slots

FULL_SCREEN = (3840, 2160)
PANEL_COLOURS = {
    "stash": (255, 0, 255),
    "equipment": (255, 160, 0),
    "backpack": (0, 255, 255),
    "quick use": (255, 255, 0),
    "augmented slots": (0, 255, 0),
    "safe pocket": (255, 80, 80),
}


def _debug_image(rgb: np.ndarray, slots: list) -> np.ndarray:
    half = cv2.resize(rgb, (rgb.shape[1] // 2, rgb.shape[0] // 2))
    for slot in slots:
        colour = PANEL_COLOURS[slot.panel]
        left, top, right, bottom = (v // 2 for v in slot.rect)
        cv2.rectangle(half, (left, top), (right - 1, bottom - 1), colour, 2)
        if slot.filled:
            cv2.circle(half, (left + 14, top + 14), 9, colour, -1)
    y = 20
    for panel in PANELS:
        count = sum(slot.panel == panel for slot in slots)
        text = f"{panel}: {count}"
        cv2.putText(
            half, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, PANEL_COLOURS[panel], 2
        )
        y += 22
    return half


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    source = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for path in sorted(source.glob("*.png")):
        with Image.open(path) as image:
            if image.size == FULL_SCREEN:
                paths.append(path)
    if not paths:
        sys.exit("no full-screen captures found")
    find_slots(np.asarray(Image.open(paths[0]).convert("RGB")))
    times = []
    header = ["file", "ms"]
    header += list(PANELS)
    header += [f"{panel} filled" for panel in PANELS]
    with (out / "grid.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for path in paths:
            rgb = np.asarray(Image.open(path).convert("RGB"))
            start = time.perf_counter()
            slots = find_slots(rgb)
            ms = (time.perf_counter() - start) * 1000
            times.append(ms)
            counts = [sum(s.panel == panel for s in slots) for panel in PANELS]
            filled = [
                sum(s.panel == panel and s.filled for s in slots) for panel in PANELS
            ]
            writer.writerow([path.name, f"{ms:.1f}", *counts, *filled])
            debug = _debug_image(rgb, slots)
            Image.fromarray(debug).save(out / path.name)
            print(path.name, f"{ms:.1f} ms", *counts)
    print(f"{len(paths)} captures, median {statistics.median(times):.1f} ms")


if __name__ == "__main__":
    main()
