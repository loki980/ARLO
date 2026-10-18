# ruff: noqa: INP001, PLC2701
"""Build the stored stash map from a capture run, frame by frame, and check it.

    uv run --extra icons --extra wayland python scripts/eval_stash_map.py \\
        --out <folder> <frame.png>...

The frames go in name order through stash_map.StashMapStore.observe on a
user database in a temporary folder (never arlo_user.db), identified as
eval_stash_stitch does. Then stash_stitch.stitch runs on the same frames
for comparison. Then the last frame is shown again to a new store on the
saved database, once as it is and once with one filled slot painted over
with an empty slot of the same frame. Writes real-data.txt to the out
folder: one line per frame (how it was placed, the rows it showed, the
cells it changed), the map against the stitch's (filled cells, rows, names
and stacks), the simulated change, the status, and a reload of the saved
map compared with the map in memory.
"""

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np
from eval_stash_stitch import _Identify
from PIL import Image

from arc_helper.database import Database
from arc_helper.grid import STASH
from arc_helper.grid import find_slots
from arc_helper.stash_map import StashMapStore
from arc_helper.stash_stitch import stitch
from arc_helper.stash_stitch import whole_slots


def _load(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _items(identify, index, rgb):
    stash = [s for s in find_slots(rgb) if s.panel == STASH]
    filled = [s for s in whole_slots(stash) if s.filled]
    return identify(index, rgb, filled) if filled else []


def _same_cell(a, b) -> bool:
    if a is None or b is None:
        return a is b
    looks = (a.look is None and b.look is None) or (
        a.look is not None and b.look is not None and np.array_equal(a.look, b.look)
    )
    return a[:7] == b[:7] and a.last_seen == b.last_seen and looks


def _same_maps(a, b) -> list[str]:
    """Where two stored maps differ, empty when they are the same."""
    out = []
    if a[:6] != b[:6]:
        out.append(f"view or header: {a[:6]} against {b[:6]}")
    if set(a.cells) != set(b.cells):
        out.append(f"cells {sorted(set(a.cells) ^ set(b.cells))}")
    for key in sorted(set(a.cells) & set(b.cells)):
        ca, cb = a.cells[key], b.cells[key]
        if not _same_cell(ca, cb):
            out.append(
                f"cell {key}: {ca[:7]} {ca.last_seen} against {cb[:7]} {cb.last_seen}"
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("frames", nargs="+")
    args = parser.parse_args()
    paths = sorted(Path(p) for p in args.frames)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logging.getLogger("arc_helper").setLevel(logging.ERROR)
    identify = _Identify()
    temp = Path(tempfile.mkdtemp(prefix="stash_map_"))
    db = Database(user_db_path=temp / "arlo_user.db")
    lines = [
        f"{len(paths)} frames, {paths[0].name} to {paths[-1].name}",
        f"user database: {db.user_db_path}",
        "",
        "build, frame by frame: index name | how | row 0 top y | rows shown | cells changed",
    ]
    store = StashMapStore(db)
    for index, path in enumerate(paths):
        rgb = _load(path)
        seen = store.observe(rgb, _items(identify, index, rgb))
        origin = f"{seen.origin:.0f}" if seen.origin is not None else "-"
        lines.append(
            f"  {index:2d} {path.name} | {seen.how} | {origin} | {seen.rows} | "
            f"{len(seen.changed)}"
        )
    built = store.map
    filled = {k for k, c in built.cells.items() if c.filled}
    rows = sorted({r for r, _ in built.cells})
    lines += [
        "",
        (
            f"map: view {built.view}, pitch {built.pitch * 2160:.1f} px at 2160, "
            f"{len(built.cells)} cells, {len(filled)} filled, rows {rows[0]} to {rows[-1]}"
        ),
        f"status: {store.status()}",
    ]

    result = stitch((_load(p) for p in paths), identify)
    cells = result.cells
    stitched = {k for k, c in cells.items() if c.filled}
    names = [
        (k, built.cells[k].name, c.slot.item.name if c.slot and c.slot.item else None)
        for k, c in cells.items()
        if k in built.cells
        and c.filled
        and built.cells[k].name
        != (c.slot.item.name if c.slot and c.slot.item else None)
    ]
    stacks = [
        (k, built.cells[k].stack, c.slot.stack)
        for k, c in cells.items()
        if k in built.cells
        and c.filled
        and c.slot is not None
        and built.cells[k].stack != c.slot.stack
    ]
    srows = sorted({r for r, _ in cells})
    lines += [
        "",
        f"stitch: {len(cells)} cells, {len(stitched)} filled, rows {srows[0]} to {srows[-1]}",
        (
            f"same cells as the stitch: {set(cells) == set(built.cells)}, "
            f"same filled cells: {stitched == filled}"
        ),
        f"  filled in the map only: {sorted(filled - stitched)}",
        f"  filled in the stitch only: {sorted(stitched - filled)}",
        f"cells whose name differs from the stitch's ({len(names)}): {names}",
        f"cells whose stack differs from the stitch's ({len(stacks)}): {stacks}",
    ]

    last = len(paths) - 1
    rgb = _load(paths[last])
    again = StashMapStore(db)
    reload = _same_maps(built, again.map)
    lines += [
        "",
        f"reload: a new store on the saved database, same map as in memory: {not reload}",
        *(f"  {d}" for d in reload),
    ]
    seen = again.observe(rgb, _items(identify, last, rgb))
    lines.append(
        f"frame {last} as it is, to the new store: {seen.how}, rows {seen.rows}, "
        f"changed {list(seen.changed)}"
    )
    whole = whole_slots([s for s in find_slots(rgb) if s.panel == STASH])
    empty = next(s for s in whole if not s.filled)
    full = max((s for s in whole if s.filled), key=lambda s: s.rect.top)
    blanked = rgb.copy()
    e, f = empty.rect, full.rect
    blanked[f.top : f.bottom, f.left : f.right] = rgb[
        e.top : e.bottom, e.left : e.right
    ]
    before = again.map
    seen = again.observe(blanked, _items(identify, last, blanked))
    after = again.map
    differs = [
        k for k in after.cells if not _same_cell(after.cells[k], before.cells.get(k))
    ]
    lines.append(
        f"frame {last} with the slot at {tuple(f)} painted over with the empty "
        f"slot at {tuple(e)}: {seen.how}, changed {list(seen.changed)}, cells that "
        f"differ from before {differs}"
    )
    for k in differs:
        b, a = before.cells[k], after.cells[k]
        lines.append(
            f"  cell {k}: filled {b.filled} {b.name} x{b.stack} seen {b.last_seen} "
            f"-> filled {a.filled} {a.name} x{a.stack} seen {a.last_seen}"
        )
    lines.append(f"status: {again.status()}")
    third = StashMapStore(db)
    reload = _same_maps(after, third.map)
    lines += [
        f"reload after the change: same map as in memory: {not reload}",
        *(f"  {d}" for d in reload),
    ]
    text = "\n".join(lines) + "\n"
    (out / "real-data.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
