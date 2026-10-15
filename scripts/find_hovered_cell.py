# ruff: noqa: INP001
"""Experiment: find the hovered inventory cell beside the tooltip card.

Usage:
    uv run python scripts/find_hovered_cell.py --captures <folder> --out <folder>
        [--labels <labels.csv>]

The game draws an animated glow ring (white, purple and cyan, partly dark)
about 4 px wide and 3 to 4 px outside the hovered cell's thin grey outline.
Other cells have only the thin outline. The card opens beside the hovered
cell, so the search runs in a window on each side of the card returned by
find_tooltip_card (cursor taken as the image centre):

1. lit mask: max channel at least 70, or a colourful pixel (max minus min at
   least 50). The white parts of the ring are the card's cream colour, so
   cream is kept and the card is left out by the window instead.
2. close with a 5 px square, which merges the glow ring and the grey outline
   beside it into one band about 10 px thick.
3. open with a 7 px square, which removes thin outlines, rarity borders and
   most text, and keeps that band. Whatever survives an opening with a
   21 px square (filled blobs such as icons and rarity wedges) is removed
   from it, so a wedge touching the outline does not swell the ring.
4. seeds: opened components that are line shaped (longest side at least
   80 px, area over width plus height at most 14) and reach within 80 px of
   the card edge.
5. the left side of the card is searched first, the right side only when
   the left has no seed. Seeds are tried largest first. The cell is the
   bounding box of the seed's component in the closed mask, which also
   follows the dim parts of the ring through the grey outline. When that box
   is not cell sized (120 to 720 px a side) the seed box is used, and when
   neither is, the next seed is tried.

Writes results.csv and, for every 3840x2160 frame, a debug PNG with the card
in blue and the cell in gold. With --labels it also runs over the labeled
card=yes crops in the captures folder and writes rows for them (debug PNGs
for crops go to a crops subfolder).
"""

import argparse
import csv
import statistics
import time
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw

from arc_helper.tooltip_layout import Point
from arc_helper.tooltip_layout import Rect
from arc_helper.tooltip_layout import find_tooltip_card

FULL_SIZE = (3840, 2160)

LIT_MIN = 70
COLOURFUL_SPREAD = 50
CLOSE_SIDE = 5
OPEN_SIDE = 7
BLOB_SIDE = 21
SEED_MIN_LENGTH = 80
SEED_MAX_THICKNESS = 14
EDGE_BAND = 80
CARD_GAP = 3
WINDOW_WIDTH = 720
WINDOW_ABOVE = 100
WINDOW_BELOW = 100
CELL_MIN_SIDE = 120
CELL_MAX_SIDE = 720


class Found(NamedTuple):
    cell: Rect
    side: str
    note: str


def _lit_mask(window: np.ndarray) -> np.ndarray:
    red, green, blue = cv2.split(np.ascontiguousarray(window))
    high = cv2.max(cv2.max(red, green), blue)
    low = cv2.min(cv2.min(red, green), blue)
    bright = cv2.threshold(high, LIT_MIN - 1, 255, cv2.THRESH_BINARY)[1]
    spread = cv2.threshold(
        cv2.subtract(high, low), COLOURFUL_SPREAD - 1, 255, cv2.THRESH_BINARY
    )[1]
    return cv2.bitwise_or(bright, spread)


def _search_side(rgb: np.ndarray, card: Rect, side: str) -> tuple[int, Rect] | None:
    """Return (seed area, cell rect in image coordinates) on one card side."""
    height, width = rgb.shape[:2]
    if side == "left":
        x0, x1 = max(0, card.left - WINDOW_WIDTH), card.left - CARD_GAP
    else:
        x0, x1 = card.right + CARD_GAP, min(width, card.right + WINDOW_WIDTH)
    y0, y1 = max(0, card.top - WINDOW_ABOVE), min(height, card.bottom + WINDOW_BELOW)
    if x1 - x0 < CELL_MIN_SIDE or y1 - y0 < CELL_MIN_SIDE:
        return None

    lit = _lit_mask(rgb[y0:y1, x0:x1, :3])
    closed = cv2.morphologyEx(
        lit, cv2.MORPH_CLOSE, np.ones((CLOSE_SIDE, CLOSE_SIDE), np.uint8)
    )
    thick = cv2.morphologyEx(
        closed, cv2.MORPH_OPEN, np.ones((OPEN_SIDE, OPEN_SIDE), np.uint8)
    )
    blobs = cv2.morphologyEx(
        closed, cv2.MORPH_OPEN, np.ones((BLOB_SIDE, BLOB_SIDE), np.uint8)
    )
    thick[blobs > 0] = 0

    edge = card.left - x0 if side == "left" else card.right - x0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(thick)
    seeds = []
    for label in range(1, count):
        left, _, w, h, area = (int(v) for v in stats[label][:5])
        if max(w, h) < SEED_MIN_LENGTH or area / (w + h) > SEED_MAX_THICKNESS:
            continue
        near = left + w >= edge - EDGE_BAND if side == "left" else left <= EDGE_BAND
        if near:
            seeds.append((area, label))
    if not seeds:
        return None

    # Follow each seed into the closed mask to recover the dim ring parts,
    # largest seed first, and keep the first one with a cell sized box
    _, closed_labels, closed_stats, _ = cv2.connectedComponentsWithStats(closed)
    for area, label in sorted(seeds, reverse=True):
        ys, xs = np.nonzero(labels == label)
        owner = int(closed_labels[ys[0], xs[0]])
        cl, ct, cw, ch = (int(v) for v in closed_stats[owner][:4])
        if not _cell_sized(cw, ch):
            cl, ct, cw, ch = (int(v) for v in stats[label][:4])
            if not _cell_sized(cw, ch):
                continue
        return area, Rect(cl + x0, ct + y0, cl + cw + x0, ct + ch + y0)
    return None


def _cell_sized(width: int, height: int) -> bool:
    return (
        CELL_MIN_SIDE <= width <= CELL_MAX_SIDE
        and CELL_MIN_SIDE <= height <= CELL_MAX_SIDE
    )


def find_hovered_cell(rgb: np.ndarray, card: Rect) -> Found | None:
    """Return the hovered cell rect beside the card, or None.

    The game opens the card to the right of the hovered cell, so the left
    side is searched first and the right side only when the left side has
    no seed (a card flipped left near the right screen edge).
    """
    left = _search_side(rgb, card, "left")
    if left is not None:
        return Found(left[1], "left", f"seed area {left[0]} on left")
    right = _search_side(rgb, card, "right")
    if right is not None:
        return Found(right[1], "right", f"seed area {right[0]} on right")
    return None


def _fmt(rect: Rect | None) -> str:
    if rect is None:
        return "none"
    return f"{rect.left} {rect.top} {rect.right} {rect.bottom}"


def _debug_png(
    rgb: np.ndarray, card: Rect | None, found: Found | None, path: Path
) -> None:
    image = Image.fromarray(rgb[:, :, :3])
    draw = ImageDraw.Draw(image)
    if card is not None:
        draw.rectangle(
            (*card[:2], card.right - 1, card.bottom - 1), outline=(0, 120, 255), width=8
        )
    if found is not None:
        cell = found.cell
        draw.rectangle(
            (*cell[:2], cell.right - 1, cell.bottom - 1), outline=(255, 200, 0), width=8
        )
    scale = 1600 / image.width
    image.resize((1600, round(image.height * scale))).save(path)


def _labeled_crops(labels_path: Path) -> set[str]:
    with labels_path.open(encoding="utf-8", newline="") as f:
        return {
            row["file"].strip()
            for row in csv.DictReader(f)
            if (row.get("card") or "").strip() == "yes"
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    crops = _labeled_crops(args.labels) if args.labels else set()
    if crops:
        (args.out / "crops").mkdir(exist_ok=True)

    cell_ms: list[float] = []
    total_ms: list[float] = []
    rows = []
    for path in sorted(args.captures.glob("*.png")):
        with Image.open(path) as image:
            size = image.size
            primary = size == FULL_SIZE
            if not primary and path.name not in crops:
                continue
            rgb = np.ascontiguousarray(np.asarray(image.convert("RGB")))

        start = time.perf_counter()
        card = find_tooltip_card(rgb, Point(rgb.shape[1] // 2, rgb.shape[0] // 2))
        middle = time.perf_counter()
        found = find_hovered_cell(rgb, card) if card is not None else None
        end = time.perf_counter()
        if primary:
            cell_ms.append((end - middle) * 1000)
            total_ms.append((end - start) * 1000)

        kind = "primary" if primary else "crop"
        if card is None:
            note = "no card"
        elif found is None:
            note = "no seed beside the card"
        else:
            note = found.note
        rows.append(
            [
                path.name,
                kind,
                f"{size[0]}x{size[1]}",
                _fmt(card),
                _fmt(found.cell if found else None),
                note,
            ]
        )
        target = args.out / path.name if primary else args.out / "crops" / path.name
        _debug_png(rgb, card, found, target)

    with (args.out / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "set", "size", "card", "cell", "notes"])
        writer.writerows(rows)

    for kind in ("primary", "crop"):
        subset = [r for r in rows if r[1] == kind]
        if not subset:
            continue
        cards = sum(r[3] != "none" for r in subset)
        cells = sum(r[4] != "none" for r in subset)
        print(f"{kind}: {len(subset)} frames, card found {cards}, cell found {cells}")
    if cell_ms:
        print(
            f"median ms per 3840x2160 frame: cell search {statistics.median(cell_ms):.1f}, "
            f"with find_tooltip_card {statistics.median(total_ms):.1f}"
        )
    print(f"wrote {args.out / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
