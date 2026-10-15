# ruff: noqa: INP001
"""Check the learned blueprint reader and the hovered frame snap on real images.

Usage:
    uv run python scripts/eval_blueprint_check.py <checks image> <frame image>
        --output-dir <folder> [--cursor X,Y] [--captures <folder>]
        [--scan-csv <stash_scan.csv>]

1. The checks image is Jeffrey's 15 Oct loadout screenshot with the
   learned Vertical Grip III Blueprint and the unlearned Bettina Blueprint,
   stretched from 2000x1406 to 3840x2160. find_slots needs square slots of
   about the 4K size, so the stretch is undone (the width squeezed by
   STRETCH_UNDO), the result scaled so its slots are 194 px and placed on a
   3840x2160 canvas of the screenshot's corner colour at the offset that
   finds the most slots. The slot rects are mapped back to the given image
   and learned_check runs on the given image, for every filled slot.
2. The frame image is the Stiletto IV screenshot. The item card is found
   from the cursor (default the Stiletto slot's centre), then
   main.hovered_slot_on_screen finds the glow ring and snaps it to the slot
   under it, as the app does. Writes frame_snap.png (half size) with the
   ring box in blue and the snapped frame in gold, and times the ring search
   alone against the whole lookup.
3. learned_check runs on every filled slot of the stash captures named in
   the bar-patch-s1 stash_scan.csv, and every True read is listed with the
   item identified in that CSV and its catalog type.

Everything printed is also written to real-data.txt in the output folder.
"""

import argparse
import csv
import statistics
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw

from arc_helper.database import Database
from arc_helper.grid import find_slots
from arc_helper.hovered_cell import find_hovered_cell
from arc_helper.main import hovered_slot_on_screen
from arc_helper.slot_marks import learned_check
from arc_helper.tooltip_layout import Point
from arc_helper.tooltip_layout import Rect
from arc_helper.tooltip_layout import find_tooltip_card

# 2000x1406 stretched to 3840x2160: the width grew 1.92 times, the height
# 1.536 times, so 1.536 / 1.92 of the width undoes it
STRETCH_UNDO = (2160 / 1406) / (3840 / 2000)
SLOT_4K = 194
OFFSETS = ((0, 0), (300, 150), (600, 200))
# Points in the stretched checks image inside the two named tiles
NAMED = {
    "Vertical Grip III Blueprint": (2410, 1546),
    "Bettina Blueprint": (2863, 1606),
}
DEFAULT_CURSOR = (1689, 985)
TIMING_RUNS = 9
GOLD = (242, 177, 52)
BLUE = (60, 140, 255)


class Log:
    """Print and keep every line for real-data.txt."""

    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)


def _repo_root() -> Path:
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).parent,
    ).stdout.strip()
    return Path(common).parent


def _load(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _unstretched_slots(rgb: np.ndarray, log: Log) -> list:
    """Slots of the stretched screenshot, as (slot, rect in the given image)."""
    height, width = rgb.shape[:2]
    squeezed = cv2.resize(
        rgb, (round(width * STRETCH_UNDO), height), interpolation=cv2.INTER_AREA
    )
    first = find_slots(squeezed)
    if not first:
        return []
    side = float(np.median([min(s.rect.width, s.rect.height) for s in first]))
    scale = SLOT_4K / side
    small = cv2.resize(squeezed, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    corner = np.median(squeezed[:20, :20].reshape(-1, 3), axis=0).astype(np.uint8)
    best: tuple[list, tuple[int, int]] = ([], (0, 0))
    for ox, oy in OFFSETS:
        canvas = np.empty((height, width, 3), np.uint8)
        canvas[:] = corner
        part = small[: height - oy, : width - ox]
        canvas[oy : oy + part.shape[0], ox : ox + part.shape[1]] = part
        slots = find_slots(canvas)
        if len(slots) > len(best[0]):
            best = (slots, (ox, oy))
    slots, (ox, oy) = best
    log(
        f"stretch undone: width x{STRETCH_UNDO:.3f}, then x{scale:.3f} to "
        f"{SLOT_4K} px slots at offset ({ox}, {oy}): {len(slots)} slots"
    )

    def back(rect: Rect) -> Rect:
        return Rect(
            round((rect.left - ox) / scale / STRETCH_UNDO),
            round((rect.top - oy) / scale),
            round((rect.right - ox) / scale / STRETCH_UNDO),
            round((rect.bottom - oy) / scale),
        )

    return [(slot, back(slot.rect)) for slot in slots]


def _contains(rect: Rect, point: tuple[int, int]) -> bool:
    return rect.left <= point[0] < rect.right and rect.top <= point[1] < rect.bottom


def check_image(path: Path, log: Log) -> None:
    log(f"== learned_check on {path.name}")
    rgb = _load(path)
    direct = find_slots(rgb)
    log(f"find_slots on the image as given: {len(direct)} slots")
    pairs = [(s, s.rect) for s in direct] or _unstretched_slots(rgb, log)
    counts = {True: 0, False: 0, None: 0}
    for slot, rect in pairs:
        if not slot.filled:
            continue
        read = learned_check(rgb, rect)
        counts[read] += 1
        name = next((n for n, p in NAMED.items() if _contains(rect, p)), "")
        log(f"  {tuple(rect)!s:28} {slot.panel:16} {read!s:5} {name}")
    log(f"filled slots: {sum(counts.values())}, reads {counts}")


def frame_image(path: Path, cursor: tuple[int, int], out_dir: Path, log: Log) -> None:
    log(f"== hovered frame snap on {path.name}")
    rgb = _load(path)
    card = find_tooltip_card(rgb, Point(*cursor))
    log(f"cursor {cursor}, card {card}")
    if card is None:
        log("no card found, nothing to snap")
        return
    ring = find_hovered_cell(rgb, card)
    capture = Image.fromarray(rgb)
    hovered = hovered_slot_on_screen(capture, Point(0, 0), card, [])
    log(f"ring box {ring}")
    log(f"snapped frame {hovered.frame}")
    for slot in find_slots(rgb):
        if hovered.frame is not None and slot.rect == hovered.frame:
            log(f"frame is the {slot.panel} slot {tuple(slot.rect)}")

    parts: dict[str, list[float]] = {
        "ring search": [],
        "find_slots": [],
        "learned_check": [],
        "whole lookup": [],
    }
    slot_rect = hovered.frame or Rect(0, 0, 1, 1)
    steps = {
        "ring search": lambda: find_hovered_cell(rgb, card),
        "find_slots": lambda: find_slots(rgb),
        "learned_check": lambda: learned_check(rgb, slot_rect),
        "whole lookup": lambda: hovered_slot_on_screen(
            capture, Point(0, 0), card, [], read_check=True
        ),
    }
    for _ in range(TIMING_RUNS):
        for name, step in steps.items():
            start = time.perf_counter()
            step()
            parts[name].append((time.perf_counter() - start) * 1000)
    medians = {name: statistics.median(ms) for name, ms in parts.items()}
    log(
        f"timing, median of {TIMING_RUNS} runs: "
        + ", ".join(f"{name} {ms:.1f} ms" for name, ms in medians.items())
    )
    log(
        "added per new card with chips on (the ring search ran before too): "
        f"{medians['find_slots'] + medians['learned_check']:.1f} ms. With chips "
        f"off the whole lookup is new: {medians['whole lookup']:.1f} ms"
    )

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    if ring is not None:
        draw.rectangle(tuple(ring), outline=BLUE, width=6)
    if hovered.frame is not None:
        draw.rectangle(tuple(hovered.frame), outline=GOLD, width=10)
    draw.rectangle(tuple(card), outline=(255, 255, 255), width=4)
    image = image.resize((image.width // 2, image.height // 2))
    target = out_dir / "frame_snap.png"
    image.save(target)
    log(f"wrote {target}")


def captures(folder: Path, scan_csv: Path, log: Log) -> None:
    log(f"== learned_check on the stash captures in {scan_csv}")
    with scan_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    db = Database()
    frames = sorted({row["frame"] for row in rows})
    counts = {True: 0, False: 0, None: 0}
    wrong = 0
    for frame in frames:
        rgb = _load(folder / frame)
        for slot in find_slots(rgb):
            if not slot.filled:
                continue
            read = learned_check(rgb, slot.rect)
            counts[read] += 1
            if read is not True:
                continue
            row = next(
                (
                    r
                    for r in rows
                    if r["frame"] == frame
                    and abs(int(r["left"]) - slot.rect.left) <= 4
                    and abs(int(r["top"]) - slot.rect.top) <= 4
                ),
                None,
            )
            item = (row["item"] or row["guess"]) if row else ""
            found = db.lookup(item) if item else None
            category = found.category if found else None
            if category != "Blueprint":
                wrong += 1
            log(
                f"  True {frame} {tuple(slot.rect)} {slot.panel}: "
                f"{item or 'unidentified'} ({category or 'no catalog type'})"
            )
    log(f"{len(frames)} captures, filled slots {sum(counts.values())}, reads {counts}")
    log(f"True reads on items whose catalog type is not Blueprint: {wrong}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("checks", type=Path)
    parser.add_argument("frame", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cursor", default=",".join(map(str, DEFAULT_CURSOR)))
    parser.add_argument("--captures", type=Path)
    parser.add_argument("--scan-csv", type=Path)
    args = parser.parse_args()
    root = _repo_root()
    folder = args.captures or root / "debug" / "captures"
    scan_csv = args.scan_csv or (
        root
        / ".claude/worktrees/overlay-roadmap/.claude/handoff/bar-patch-s1"
        / "attempt-1/stash_scan.csv"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    x, y = (int(v) for v in args.cursor.split(","))

    log = Log()
    check_image(args.checks, log)
    log()
    frame_image(args.frame, (x, y), args.output_dir, log)
    log()
    captures(folder, scan_csv, log)
    (args.output_dir / "real-data.txt").write_text("\n".join(log.lines) + "\n")


if __name__ == "__main__":
    main()
