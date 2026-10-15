# ruff: noqa: INP001
"""Read the stash and loadout headers on a fixed sample of saved captures.

Usage: uv run python scripts/eval_inventory_header.py <captures folder> <output folder>

Takes every 10th full-screen 3840x2160 PNG in name order, starting with the
first, and runs read_stash_header and read_loadout_header on it. Writes to
the output folder header_reads.csv (file, what each reader returned, ms for
the two readers) and header_sheet.png, one contact sheet of the header crops
with the reads printed under each, to check by eye. Ends with the median ms
per frame. The captures folder is only read.
"""

import csv
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from arc_helper.grid import EQUIPMENT
from arc_helper.grid import STASH
from arc_helper.grid import find_slots
from arc_helper.inventory_header import read_loadout_header
from arc_helper.inventory_header import read_stash_header

FULL_SCREEN = (3840, 2160)
STEP = 10
CROP_TOP, CROP_BOTTOM = 270, 410
STASH_SPAN = (-10, 560)
LOADOUT_SPAN = (200, 600)
SHEET_SCALE = 0.5
COLUMNS = 2
FALLBACK_LEFT = {STASH_SPAN: 302, LOADOUT_SPAN: 1384}


def _sample(folder: Path) -> list[Path]:
    full = []
    for path in sorted(folder.glob("*.png")):
        with Image.open(path) as image:
            if image.size == FULL_SCREEN:
                full.append(path)
    return full[::STEP]


def _crop(rgb: np.ndarray, left: int | None, span: tuple[int, int]) -> np.ndarray:
    width = span[1] - span[0]
    if left is None:
        # No slots found: show the crop at the usual left edge, labelled in
        # the caption, so the header can still be checked by eye
        left = FALLBACK_LEFT[span]
    start = max(0, left + span[0])
    crop = rgb[CROP_TOP:CROP_BOTTOM, start : start + width]
    pad = width - crop.shape[1]
    return np.pad(crop, ((0, 0), (0, pad), (0, 0)))


def _tile(rgb: np.ndarray, stash, equipment, label: str, stash_read, loadout_read):
    left_s = min((s.rect.left for s in stash), default=None)
    left_e = min((s.rect.left for s in equipment), default=None)
    row = np.hstack([_crop(rgb, left_s, STASH_SPAN), _crop(rgb, left_e, LOADOUT_SPAN)])
    row = cv2.resize(
        row, None, fx=SHEET_SCALE, fy=SHEET_SCALE, interpolation=cv2.INTER_AREA
    )
    caption = np.zeros((34, row.shape[1], 3), np.uint8)
    text = f"{label} S:{stash_read} L:{loadout_read}"
    if left_s is None or left_e is None:
        text += " (fallback crop, no slots)"
    cv2.putText(
        caption, text, (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1
    )
    return np.vstack([row, caption])


def main() -> None:
    folder, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rows, tiles, times = [], [], []
    for path in _sample(folder):
        rgb = np.array(Image.open(path).convert("RGB"))
        slots = find_slots(rgb)
        stash = [s for s in slots if s.panel == STASH]
        equipment = [s for s in slots if s.panel == EQUIPMENT]
        start = time.perf_counter()
        stash_read = read_stash_header(rgb, stash)
        loadout_read = read_loadout_header(rgb, equipment)
        ms = (time.perf_counter() - start) * 1000
        times.append(ms)
        rows.append([path.name, stash_read, loadout_read, f"{ms:.0f}"])
        label = path.name[9:15] + path.name[16:19]
        tiles.append(_tile(rgb, stash, equipment, label, stash_read, loadout_read))
        print(rows[-1])
    with (out / "header_reads.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "stash", "loadout", "ms"])
        writer.writerows(rows)
    per_column = -(-len(tiles) // COLUMNS)
    blank = np.zeros_like(tiles[0])
    columns = []
    for c in range(COLUMNS):
        part = tiles[c * per_column : (c + 1) * per_column]
        part += [blank] * (per_column - len(part))
        columns.append(np.vstack(part))
    Image.fromarray(np.hstack(columns)).save(out / "header_sheet.png")
    print(f"frames {len(times)}, median ms {statistics.median(times):.0f}")


if __name__ == "__main__":
    main()
