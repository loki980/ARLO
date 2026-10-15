# ruff: noqa: INP001, PLC2701
"""Measure the stash pass (stash_scan.identify_slots) on a fixed sample of frames.

    uv run --extra icons python scripts/eval_stash_scan.py <captures> <out folder>

Sample rule: the 3840x2160 captures in name order, every 12th of them
(the 1st, 13th, 25th and so on), kept when grid.find_slots finds at least
MIN_FILLED filled slots, then SAMPLE frames spread evenly over those.

The gallery is every arctracker item outside the types that never sit in an
inventory slot (EXCLUDED_TYPES), its icons downloaded once into the icon
cache, cut to one entry per distinct icon (distinct_gallery). Items resolve
through Database.lookup and badges use verdict_kinds.

Each frame is timed twice through the whole pass (find_slots, embedding,
tier reads, lookups), without and with stack reads. The model is loaded and
warmed up first. Writes to the out folder: sheet_NN_<frame>.png (one tile
per filled slot: the slot crop, the top gallery icon, the name drawn on the
slot, then tier, stack, price and verdict), stash_scan.csv (one row per
filled slot) and summary.txt (sample, timings and badge counts).
"""

import csv
import logging
import statistics
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
from icon_hash_baseline import _arctracker_items
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from arc_helper import icon_embed
from arc_helper.config import get_settings
from arc_helper.database import Database
from arc_helper.grid import find_slots
from arc_helper.icons import IconDownloadError
from arc_helper.icons import blueprint_icon_path
from arc_helper.stash_scan import badges_for
from arc_helper.stash_scan import distinct_gallery
from arc_helper.stash_scan import format_price
from arc_helper.stash_scan import identify_slots
from arc_helper.stash_scan import verdict_kinds

FULL_SIZE = (3840, 2160)
EVERY = 12
SAMPLE = 12
MIN_FILLED = 1
EXCLUDED_TYPES = {"Outpost Furniture", "Outpost Room", "Design"}
TILE = 112
COLUMNS = 6
TEXT_ROWS = 4
LINE = 15
BACKDROP = (22, 30, 38, 255)
GREEN = (90, 220, 90)
RED = (240, 90, 90)
GOLD = (242, 177, 52)
GREY = (200, 200, 200)


def _sample(folder: Path) -> tuple[list[Path], list[str]]:
    """The fixed sample and a note per candidate frame."""
    full = []
    for path in sorted(folder.glob("*.png")):
        with Image.open(path) as image:
            if image.size == FULL_SIZE:
                full.append(path)
    candidates = full[::EVERY]
    kept, notes = [], [f"{len(full)} full-size captures, {len(candidates)} candidates"]
    for path in candidates:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        filled = sum(s.filled for s in find_slots(rgb))
        if filled >= MIN_FILLED:
            kept.append(path)
        notes.append(f"  {path.name}: {filled} filled")
    if len(kept) <= SAMPLE:
        return kept, notes
    step = len(kept) / SAMPLE
    return [kept[int(n * step)] for n in range(SAMPLE)], notes


def _gallery_icons(cache_dir: Path) -> dict[str, tuple[str, Path]]:
    icons = {}
    for item in _arctracker_items(cache_dir):
        if item["type"] in EXCLUDED_TYPES:
            continue
        try:
            icons[item["id"]] = (
                item["name"]["en"],
                blueprint_icon_path(item["id"], cache_dir),
            )
        except IconDownloadError as e:
            print(f"skipped icon: {e}")
    return icons


def _icon_tile(path: Path) -> Image.Image:
    with Image.open(path) as icon:
        flat = Image.new("RGBA", icon.size, BACKDROP)
        tile = Image.alpha_composite(flat, icon.convert("RGBA")).convert("RGB")
    tile.thumbnail((TILE, TILE))
    return tile


def _sheet(rgb: np.ndarray, rows: list[dict], icon_paths: dict, path: Path) -> None:
    font = ImageFont.load_default(size=13)
    cell_w, cell_h = 2 * TILE + 12, TILE + TEXT_ROWS * LINE + 8
    count = max(1, (len(rows) + COLUMNS - 1) // COLUMNS)
    sheet = Image.new("RGB", (cell_w * COLUMNS, cell_h * count), (8, 8, 8))
    draw = ImageDraw.Draw(sheet)
    for n, r in enumerate(rows):
        x, y = (n % COLUMNS) * cell_w, (n // COLUMNS) * cell_h
        crop = Image.fromarray(rgb[r["top"] : r["bottom"], r["left"] : r["right"]])
        crop.thumbnail((TILE, TILE))
        crop_draw = ImageDraw.Draw(crop)
        name = r["item"] or f"? {r['guess']}"
        crop_draw.rectangle(
            (0, crop.height - 2 * LINE, crop.width, crop.height), fill=(0, 0, 0)
        )
        for k, part in enumerate(textwrap.wrap(name, 16)[:2]):
            crop_draw.text(
                (2, crop.height - 2 * LINE + k * LINE),
                part,
                fill=GREEN if r["item"] else RED,
                font=font,
            )
        sheet.paste(crop, (x, y))
        if r["guess"] in icon_paths:
            sheet.paste(_icon_tile(icon_paths[r["guess"]]), (x + TILE + 6, y))
        lines = [
            (
                f"{n + 1}. {r['panel']} sim {r['confidence']:.2f} lead {r['lead']:.2f}",
                GREY,
            ),
            (f"top: {r['guess']}", GREY),
            (
                f"tier {r['tier'] or '-'} stack {r['stack'] or '-'} price {r['price']}",
                GREY,
            ),
            (
                f"verdict {r['kind']}",
                GOLD if r["kind"] in {"keep", "keep_some"} else GREY,
            ),
        ]
        for k, (text, colour) in enumerate(lines):
            draw.text((x + 2, y + TILE + 4 + k * LINE), text, fill=colour, font=font)
    sheet.save(path)


def _row(path: Path, s, kind_of) -> dict:
    """One CSV and sheet row for a filled slot."""
    return {
        "frame": path.name,
        "panel": s.panel,
        "left": s.rect.left,
        "top": s.rect.top,
        "right": s.rect.right,
        "bottom": s.rect.bottom,
        "guess": s.guess,
        "confidence": s.confidence,
        "lead": s.lead,
        "item": s.item.name if s.item else "",
        "tier": s.tier or "",
        "stack": s.stack if s.stack is not None else "",
        "sell_price": s.item.sell_price if s.item and s.item.sell_price else "",
        "price": format_price(s.item.sell_price)
        if s.item and s.item.sell_price
        else "-",
        "kind": kind_of(s) if s.item else "",
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: eval_stash_scan.py <captures folder> <out folder>")
        return 2
    folder, out = Path(sys.argv[1]), Path(sys.argv[2])
    if not folder.is_dir():
        print(f"not a folder: {folder}")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    logging.getLogger("arc_helper").setLevel(logging.ERROR)

    sample, notes = _sample(folder)
    cache_dir = get_settings().icon_cache_dir
    icons = _gallery_icons(cache_dir)
    icon_paths = dict(icons.values())
    embedder = icon_embed.default_embedder()
    start = time.perf_counter()
    gallery = icon_embed.build_gallery(icons, embedder, cache_dir / "stash_gallery.npz")
    built = time.perf_counter() - start
    full_size = len(gallery)
    gallery = distinct_gallery(gallery)
    db = Database()
    kind_of = verdict_kinds(db)
    embedder.embed(
        [np.zeros((icon_embed.INPUT_SIDE, icon_embed.INPUT_SIDE, 3), np.uint8)]
    )

    lines = [
        (
            f"sample rule: every {EVERY}th 3840x2160 capture in name order, at least "
            f"{MIN_FILLED} filled slot, {SAMPLE} spread evenly"
        ),
        *notes,
        f"gallery: {full_size} icons, {len(gallery)} distinct ({built:.1f} s to build or load)",
        "",
    ]
    plain_times, stack_times = [], []
    rows_out = []
    for n, path in enumerate(sample, 1):
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        start = time.perf_counter()
        identify_slots(rgb, gallery, db.lookup, embedder=embedder)
        plain = time.perf_counter() - start
        start = time.perf_counter()
        found = identify_slots(
            rgb, gallery, db.lookup, embedder=embedder, read_stacks=True
        )
        stacked = time.perf_counter() - start
        plain_times.append(plain)
        stack_times.append(stacked)
        badges = badges_for(found, kind_of)
        rows = [_row(path, s, kind_of) for s in found]
        rows_out.extend(rows)
        _sheet(rgb, rows, icon_paths, out / f"sheet_{n:02d}_{path.stem}.png")
        identified = sum(1 for s in found if s.item)
        lines.append(
            f"{n:2d} {path.name}: {len(found)} filled, {identified} identified, "
            f"{sum(b['kind'] == 'border' for b in badges)} borders, "
            f"{sum(b['kind'] == 'tag' for b in badges)} price tags, "
            f"pass {1000 * plain:.0f} ms, with stacks {1000 * stacked:.0f} ms"
        )
    lines += [
        "",
        f"median full pass without stack reads: {1000 * statistics.median(plain_times):.0f} ms",
        f"median full pass with stack reads: {1000 * statistics.median(stack_times):.0f} ms",
    ]
    columns = [k for k in rows_out[0] if k != "price"] if rows_out else []
    with (out / "stash_scan.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_out)
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
