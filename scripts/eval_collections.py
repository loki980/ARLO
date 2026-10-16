# ruff: noqa: INP001
"""Read Collections screen frames and draw what was read.

Usage: uv run python scripts/eval_collections.py --out <folder>
           [--captures <folder>] [--count 5] frame.png [...]

Runs read_collections on each frame and writes, in the output folder,
<frame>_overlay.png with every cell boxed (green for found, grey for empty,
the grid index in each box) and the FOUND read at the top. Then runs
record_collections on a throwaway user database in a temporary folder, never
the real arlo_user.db. Then runs read_collections on --count stash captures
(*_grid.png, spread over the captures folder by name), each expected None.
Writes real-data.txt in the output folder: the FOUND read, the cell counts,
the found grid indices with their blueprint names, the unknown found slots,
the timing, and the capture results. The captures folder is only read.
"""

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageDraw

from arc_helper.collections_screen import read_collections
from arc_helper.collections_screen import record_collections
from arc_helper.database import Database

CAPTURES = Path("/home/jefflamb/workspace/ARLO/debug/captures")


def _overlay(image: Image.Image, read, path: Path) -> None:
    draw = ImageDraw.Draw(image)
    for cell in read.cells:
        r = cell.rect
        colour = (0, 255, 0) if cell.found else (160, 160, 160)
        draw.rectangle((r.left, r.top, r.right - 1, r.bottom - 1), outline=colour)
        draw.text((r.left + 4, r.top + 4), str(cell.grid_index), fill=colour)
    draw.text(
        (10, 10),
        f"FOUND {read.found_count}/{read.total}  cells {len(read.cells)}  "
        f"found {len(read.found_indices)}  consistent {read.consistent}",
        fill=(255, 255, 0),
    )
    image.save(path)


def _frame_lines(path: Path, out: Path, db: Database) -> list[str]:
    image = Image.open(path).convert("RGB")
    start = time.perf_counter()
    read = read_collections(np.asarray(image))
    elapsed = time.perf_counter() - start
    lines = [f"== {path}", f"size {image.width}x{image.height}  {elapsed:.2f} s"]
    if read is None:
        return [*lines, "read: None"]
    lines += [
        f"FOUND {read.found_count}/{read.total}",
        (
            f"cells {len(read.cells)}  found cells {len(read.found_indices)}  "
            f"consistent {read.consistent}"
        ),
        "found grid indices and blueprints:",
    ]
    lines += [f"  {i:3d}  {db.blueprint_at(i)}" for i in read.found_indices]
    unknown = record_collections(db, read)
    lines.append(f"record_collections (throwaway user db): unknown found {unknown}")
    lines.append(f"  owned after record: {len(db.owned_blueprints())}")
    overlay = out / f"{path.stem}_overlay.png"
    _overlay(image, read, overlay)
    lines.append(f"overlay {overlay}")
    return lines


def _capture_lines(folder: Path, count: int) -> list[str]:
    captures = sorted(folder.glob("*_grid.png"))
    if not captures:
        return [f"== no *_grid.png captures in {folder}"]
    step = max(1, len(captures) // count)
    picked = captures[::step][:count]
    lines = [f"== {len(picked)} stash captures from {folder}, each expected None"]
    for path in picked:
        read = read_collections(np.asarray(Image.open(path).convert("RGB")))
        lines.append(f"  {path.name}: {'None' if read is None else read[:2]}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--captures", type=Path, default=CAPTURES)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("frames", type=Path, nargs="+")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    with tempfile.TemporaryDirectory() as scratch:
        db = Database(user_db_path=Path(scratch) / "arlo_user.db")
        for frame in args.frames:
            lines += _frame_lines(frame, args.out, db)
    lines += _capture_lines(args.captures, args.count)
    text = "\n".join(lines) + "\n"
    (args.out / "real-data.txt").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
