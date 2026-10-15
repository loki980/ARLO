# ruff: noqa: INP001
"""Experiment: find the hovered inventory cell beside the tooltip card.

Usage:
    uv run python scripts/find_hovered_cell.py --captures <folder> --out <folder>
        [--include <list.txt>] [--labels <labels.csv>]

Runs arc_helper.hovered_cell.locate_hovered_cell (the method is described
in that module, find_hovered_cell is its app-facing wrapper) beside the card
returned by find_tooltip_card, with the cursor taken as the image centre.

Writes results.csv and, for every 3840x2160 frame, a debug PNG with the card
in blue and the cell in gold.

--include names a text file with one capture file name per line, and only
those files are run.

--labels takes one of two CSV kinds. A file with a cell_left column scores
the run: its header is file,card_opened,cell_left,cell_top,cell_right,
cell_bottom,notes in full-screen px, empty cells meaning no hovered cell. A
labeled cell matches when every edge of the found cell is within 12 px, an
empty label matches when nothing is found, and the pass counts are printed
for all labeled frames and for those whose card opened left. A file with a
card column instead (the older crop labels) adds the card=yes crops in the
captures folder to the run (debug PNGs for crops go to a crops subfolder).
"""

import argparse
import csv
import statistics
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image
from PIL import ImageDraw

from arc_helper.hovered_cell import locate_hovered_cell
from arc_helper.tooltip_layout import Point
from arc_helper.tooltip_layout import Rect
from arc_helper.tooltip_layout import find_tooltip_card

FULL_SIZE = (3840, 2160)
MATCH_PX = 12


class Found(NamedTuple):
    cell: Rect
    side: str
    note: str


def find_hovered_cell(rgb: np.ndarray, card: Rect) -> Found | None:
    """Run the app's cell search and describe how it found the cell."""
    found = locate_hovered_cell(rgb, card)
    if found is None:
        return None
    note = f"seed area {found.seed_area} on {found.side}"
    return Found(found.rect, found.side, note)


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


class Label(NamedTuple):
    opened: str
    cell: Rect | None


def _read_labels(labels_path: Path) -> tuple[set[str], dict[str, Label]]:
    """Return the crop names to add and the cell labels to score against."""
    with labels_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "cell_left" not in rows[0]:
        crops = {
            r["file"].strip() for r in rows if (r.get("card") or "").strip() == "yes"
        }
        return crops, {}
    labels = {}
    for row in rows:
        edges = [
            row[k].strip()
            for k in ("cell_left", "cell_top", "cell_right", "cell_bottom")
        ]
        cell = Rect(*(int(v) for v in edges)) if all(edges) else None
        labels[row["file"].strip()] = Label(row["card_opened"].strip(), cell)
    return set(), labels


def _matches(found: Rect | None, label: Rect | None) -> bool:
    if found is None or label is None:
        return found is None and label is None
    return all(abs(a - b) <= MATCH_PX for a, b in zip(found, label, strict=True))


def _score(
    rows: list[list[str]], found: dict[str, Rect | None], labels: dict[str, Label]
) -> None:
    """Add a verdict column to rows and print the pass counts."""
    groups: dict[str, list[bool]] = {"all": [], "cell": [], "left": [], "right": []}
    for row in rows:
        label = labels.get(row[0])
        if label is None:
            row.append("unlabeled")
            continue
        ok = _matches(found[row[0]], label.cell)
        row.append("match" if ok else f"miss, label {_fmt(label.cell)}")
        groups["all"].append(ok)
        if label.cell is not None:
            groups["cell"].append(ok)
            groups[label.opened].append(ok)
    for name, text in (
        ("all", "labeled frames"),
        ("cell", "frames with a hovered cell"),
        ("left", "card opened left"),
        ("right", "card opened right"),
    ):
        hits = groups[name]
        print(f"{text}: {sum(hits)} of {len(hits)} right")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--include", type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    crops, labels = _read_labels(args.labels) if args.labels else (set(), {})
    include = None
    if args.include:
        lines = args.include.read_text(encoding="utf-8").splitlines()
        include = {line.strip() for line in lines if line.strip()}
    if crops:
        (args.out / "crops").mkdir(exist_ok=True)

    cell_ms: list[float] = []
    total_ms: list[float] = []
    rows = []
    found_cells: dict[str, Rect | None] = {}
    for path in sorted(args.captures.glob("*.png")):
        if include is not None and path.name not in include:
            continue
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

        found_cells[path.name] = found.cell if found else None
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

    header = ["file", "set", "size", "card", "cell", "notes"]
    if labels:
        header.append("verdict")
        _score(rows, found_cells, labels)
    with (args.out / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
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
