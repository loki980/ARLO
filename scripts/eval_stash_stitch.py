# ruff: noqa: INP001, PLC2701
"""Stitch a capture run of the stash scrolled top to bottom and report the map.

    uv run --extra icons --extra wayland python scripts/eval_stash_stitch.py \\
        --out <folder> <frame.png>...

The frames go through stash_stitch.stitch in name order, identified with
stash_scan.identify_slots (stack reads on, one SlotCache over the run, the
gallery as eval_stash_scan builds it). Writes to the out folder:

- real-data.txt: one line per frame (the header as read on its own slots
  and as stitched, slot counts, how it was placed, its scroll in rows), the
  joins, every map per view with its segments, the rows they cover, the
  filled count against the header's used count, the overlap check of
  consecutive frames, cells whose frames disagree, and the tally (item to
  total count).
- stash_map.png: the first view's first segment as one tall image, each
  absolute row once, every cell the crop of the frame it was taken from,
  with its row and column, name, tier and stack on it (green identified,
  red not, grey empty).
"""

import argparse
import logging
import textwrap
import time
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np
from eval_stash_scan import _gallery_icons
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from arc_helper import icon_embed
from arc_helper.config import get_settings
from arc_helper.database import Database
from arc_helper.grid import STASH
from arc_helper.grid import find_slots
from arc_helper.inventory_header import read_stash_header
from arc_helper.stash_scan import SlotCache
from arc_helper.stash_scan import distinct_gallery
from arc_helper.stash_scan import identify_slots
from arc_helper.stash_stitch import filled_cells
from arc_helper.stash_stitch import stitch
from arc_helper.stash_stitch import tally

TILE = 128
LABEL = 44
LINE = 13
GREEN = (90, 220, 90)
RED = (240, 90, 90)
GREY = (150, 150, 150)
WHITE = (230, 230, 230)


def _frames(paths: list[Path]):
    for path in paths:
        with Image.open(path) as image:
            yield np.asarray(image.convert("RGB"))


class _Identify:
    """identify_slots over the run, keeping a thumbnail of every slot and the
    header as read on the frame's own slots."""

    def __init__(self):
        cache_dir = get_settings().icon_cache_dir
        icons = _gallery_icons(cache_dir)
        self.embedder = icon_embed.default_embedder()
        gallery = icon_embed.build_gallery(
            icons, self.embedder, cache_dir / "stash_gallery.npz"
        )
        self.gallery, self.twins = distinct_gallery(gallery)
        self.db = Database()
        self.cache = SlotCache()
        self.thumbs: dict[tuple[int, tuple], Image.Image] = {}
        self.own_header: dict[int, object] = {}
        self.top_slot: dict[int, int] = {}
        self.seconds = 0.0

    def __call__(self, index, rgb, slots):
        stash = [s for s in find_slots(rgb) if s.panel == STASH]
        self.own_header[index] = read_stash_header(rgb, stash)
        if stash:
            self.top_slot[index] = min(s.rect.top for s in stash)
        for s in stash:
            r = s.rect
            thumb = Image.fromarray(rgb[r.top : r.bottom, r.left : r.right])
            thumb.thumbnail((TILE, TILE))
            self.thumbs[index, tuple(r)] = thumb
        start = time.perf_counter()
        found = identify_slots(
            rgb,
            self.gallery,
            self.db.lookup,
            embedder=self.embedder,
            read_stacks=True,
            slots=slots,
            twins=self.twins,
            cache=self.cache,
        )
        self.seconds += time.perf_counter() - start
        return found


def _cell_text(cell) -> tuple[str, tuple[int, int, int]]:
    if not cell.filled:
        return "empty", GREY
    slot = cell.slot
    if slot is None:
        return "? (not identified)", RED
    if slot.item is None:
        return f"? {slot.guess}", RED
    stack = f" x{slot.stack}" if slot.stack is not None else ""
    return f"{slot.item.name}{stack}", GREEN


def _sheet(cells, thumbs, path: Path) -> None:
    font = ImageFont.load_default(size=12)
    rows = sorted({r for r, _ in cells})
    columns = sorted({c for _, c in cells})
    top, first_col = rows[0], columns[0]
    width = LABEL + (columns[-1] - first_col + 1) * TILE
    height = (rows[-1] - top + 1) * TILE
    sheet = Image.new("RGB", (width, height), (8, 8, 8))
    draw = ImageDraw.Draw(sheet)
    for row in range(top, rows[-1] + 1):
        y = (row - top) * TILE
        draw.text((4, y + TILE // 2 - 6), f"r{row}", fill=WHITE, font=font)
    for (row, col), cell in cells.items():
        x, y = LABEL + (col - first_col) * TILE, (row - top) * TILE
        thumb = None
        if cell.slot is not None and cell.frame is not None:
            thumb = thumbs.get((cell.frame, tuple(cell.slot.rect)))
        if thumb is not None:
            sheet.paste(thumb, (x, y))
        else:
            draw.rectangle((x + 2, y + 2, x + TILE - 3, y + TILE - 3), outline=GREY)
        text, colour = _cell_text(cell)
        parts = textwrap.wrap(text, 18)[:3]
        draw.rectangle((x, y, x + TILE - 1, y + LINE * len(parts) + 2), fill=(0, 0, 0))
        for k, part in enumerate(parts):
            draw.text((x + 2, y + k * LINE), part, fill=colour, font=font)
        draw.text(
            (x + 2, y + TILE - LINE - 2),
            f"{row},{col} f{cell.frame}",
            fill=WHITE,
            font=font,
        )
    sheet.save(path)


def _rows_line(cells) -> str:
    rows = sorted({r for r, _ in cells})
    missing = sorted(set(range(rows[0], rows[-1] + 1)) - set(rows))
    short = [r for r in rows if sum((r, c) in cells for c in range(4)) < 4]
    return (
        f"rows {rows[0]} to {rows[-1]} ({len(rows)} rows), missing rows {missing or 'none'}, "
        f"rows with fewer than 4 cells {short or 'none'}"
    )


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
    start = time.perf_counter()
    result = stitch(_frames(paths), identify)
    total = time.perf_counter() - start
    lines = [
        f"{len(paths)} frames, {paths[0].name} to {paths[-1].name}",
        f"stitch with identification: {total:.1f} s, of which identify_slots {identify.seconds:.1f} s",
        "",
        (
            "frames: name | header on its own slots | header as stitched | top slot y | "
            "stash slots | whole | whole filled | placed by | offset px | segment | "
            "scroll rows | dropped"
        ),
    ]
    for note in result.frames:
        own = identify.own_header.get(note.index)
        scroll = (
            f"{note.scroll / result.maps[0].pitch:.2f}"
            if note.scroll is not None
            else "-"
        )
        lines.append(
            f"  {note.index:2d} {paths[note.index].name} | {own and tuple(own)} | "
            f"{note.header and tuple(note.header)} | {identify.top_slot.get(note.index)} | "
            f"{note.slots} | {note.whole} | {note.filled} | {note.how} | {note.offset} | "
            f"{note.segment} | {scroll} | {note.dropped}"
        )
    views = Counter(n.view for n in result.frames)
    lines += ["", f"views (tab, slot size, grid left): {dict(views)}"]
    for m, view_map in enumerate(result.maps):
        lines += [
            "",
            f"map {m}: view {view_map.view}, header {view_map.header}, pitch {view_map.pitch:.1f} px",
            f"  joins after the last frame (frame a, frame b, offset): {[tuple(j) for j in view_map.joins]}",
        ]
        for s, segment in enumerate(view_map.segments):
            cells = segment.cells
            filled = filled_cells(cells)
            lines += [
                f"  segment {s}: frames {list(segment.frames)}",
                f"    {len(cells)} cells, {filled} filled, {_rows_line(cells)}",
            ]
            if s == 0 and view_map.header is not None:
                lines.append(
                    f"    filled {filled} against header used {view_map.header.used}: "
                    f"difference {filled - view_map.header.used}"
                )
            seen = Counter(c.seen for c in cells.values())
            dropped = sum(result.frames[i].dropped for i in segment.frames)
            lines.append(
                f"    cells by how many frames showed them whole: {dict(sorted(seen.items()))}, "
                f"observations dropped (off any row line, or a second one on a cell "
                f"of the same frame): {dropped}"
            )
            agree = sum(o.agree for o in segment.overlaps)
            disagree = sum(o.disagree for o in segment.overlaps)
            unsure = sum(o.unsure for o in segment.overlaps)
            shared = sum(o.shared for o in segment.overlaps)
            lines.append(
                f"    overlap of consecutive frames: {shared} shared cells, {agree} same look, "
                f"{disagree} different look, {unsure} unsure"
            )
            for o in segment.overlaps:
                if o.disagree:
                    lines.append(
                        f"      frames {o.a} and {o.b}: {o.shared} shared, {o.agree} same, "
                        f"{o.disagree} different, {o.unsure} unsure"
                    )
            for cell in cells.values():
                if cell.filled_seen not in {0, cell.seen} or len(cell.names) > 1:
                    lines.append(
                        f"    cell {cell.row},{cell.column}: seen {cell.seen}, filled on "
                        f"{cell.filled_seen}, names {list(cell.names)}, counted "
                        f"{'filled' if cell.filled else 'empty'}"
                    )
    if result.maps:
        cells = result.cells
        unknown = [
            c
            for c in cells.values()
            if c.filled and (c.slot is None or c.slot.item is None)
        ]
        lines += [
            "",
            (
                f"tally of map 0, segment 0 ({len(tally(cells))} items, "
                f"{sum(tally(cells).values())} pieces, {len(unknown)} filled cells "
                "not identified):"
            ),
        ]
        for name, count in sorted(tally(cells).items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"  {count:5d}  {name}")
        for c in unknown:
            guess = c.slot.guess if c.slot is not None else None
            lines.append(f"  not identified: cell {c.row},{c.column} guess {guess}")
        # The stash was sorted by name, so in reading order the identified
        # names should never go back in the alphabet. A row placed twice or
        # out of order shows up as a step back
        named = [
            (key, c.slot.item.name)
            for key, c in sorted(cells.items())
            if c.filled and c.slot is not None and c.slot.item is not None
        ]
        steps_back = [
            (ka, a, kb, b)
            for (ka, a), (kb, b) in pairwise(named)
            if b.casefold() < a.casefold()
        ]
        lines.append(
            f"\nname order check (sorted by NAME): {len(named)} identified cells in "
            f"reading order, {len(steps_back)} steps back in the alphabet"
        )
        for ka, a, kb, b in steps_back:
            lines.append(f"  {ka} {a} then {kb} {b}")
        _sheet(cells, identify.thumbs, out / "stash_map.png")
        lines.append(f"\ncontact sheet: {out / 'stash_map.png'}")
    text = "\n".join(lines) + "\n"
    (out / "real-data.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
