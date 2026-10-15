# ruff: noqa: INP001, SLF001
"""Measure the perceptual-hash icon baseline on slot crops with known items.

Usage:
    uv run python scripts/icon_hash_baseline.py <captures folder> <out folder>

Every 3840x2160 capture from 20261012-001659 on that has a card is read the
way the app reads it (title reader, then Database.lookup), and the hovered
slot is found with find_hovered_cell. Slot crops with a resolved item name
are the labeled set. The gallery is every arctracker item of the same types
as the labeled items (the labeled items plus their distractors), downloaded
once into the configured icon cache.

Writes to the out folder: labeled.csv (one row per crop with its top 5),
results.txt (the accuracy tables) and contact_<method>.png (crops beside
their top-1 match).
"""

import csv
import json
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from PIL import ImageDraw

from arc_helper.config import get_settings
from arc_helper.database import Database
from arc_helper.hovered_cell import find_hovered_cell
from arc_helper.icon_match import METHODS
from arc_helper.icon_match import build_gallery
from arc_helper.icon_match import timed_match
from arc_helper.icons import USER_AGENT
from arc_helper.icons import IconDownloadError
from arc_helper.icons import blueprint_icon_path
from arc_helper.ocr import OCREngine
from arc_helper.tooltip_layout import Point

FULL_SIZE = (3840, 2160)
FIRST_CAPTURE = "20261012-001659"
ITEMS_URL = "https://arctracker.io/api/items"
SHEET_COLUMNS = 6
TILE = 128
# Capture file name prefix to why the label is not a fair test, checked by eye
BAD_LABELS = {
    "20261014-211224": "shows a Great Mullein, not the labeled Defibrillator",
    "20261014-211227": "shows a Great Mullein, not the labeled Defibrillator",
    "20261014-211303": "shows a Zipline, not the labeled Adrenaline Shot",
    "20261014-210936": "hovered cell is not the Grappling Hook",
    "20261014-211019": "hovered cell is not the Grappling Hook",
    "20261014-211021": "hovered cell is not the Grappling Hook",
    "20261014-211024": "hovered cell is not the Grappling Hook",
    "20261014-211226": "hovered cell is not the Grappling Hook",
    "20261014-210220": "tooltip fragment lies over the slot",
    "20261014-210922": "two-cell crop, not one slot",
}


def _arctracker_items(cache_dir: Path) -> list[dict]:
    path = cache_dir / "arctracker_items.json"
    if not path.is_file():
        resp = requests.get(ITEMS_URL, timeout=60, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
    return json.loads(path.read_text(encoding="utf-8"))["items"]


def _label_crops(folder: Path) -> tuple[list[dict], list[str]]:
    """Return (labeled crops, skipped capture notes)."""
    logging.getLogger("arc_helper").setLevel(logging.WARNING)
    engine = OCREngine()
    db = Database()
    labeled: list[dict] = []
    skipped: list[str] = []
    for path in sorted(folder.glob("*.png")):
        if path.name < FIRST_CAPTURE:
            continue
        with Image.open(path) as image:
            if image.size != FULL_SIZE:
                continue
            rgb = image.convert("RGB")
        engine._last_capture, engine._last_origin = rgb, Point(0, 0)
        engine._last_card = None
        crop = engine._locate_card(Point(rgb.width // 2, rgb.height // 2))
        card = engine._last_card
        if card is None:
            skipped.append(f"{path.name}: no card")
            continue
        read = engine._read_title(crop or rgb, card_found=crop is not None)
        item = db.lookup(read, tier_vote=engine.last_tier_vote) if read else None
        if item is None:
            skipped.append(f"{path.name}: title {read!r} not resolved")
            continue
        array = np.asarray(rgb)
        cell = find_hovered_cell(array, card)
        if cell is None:
            skipped.append(f"{path.name}: no hovered cell")
            continue
        slot = array[cell.top : cell.bottom, cell.left : cell.right]
        labeled.append({"file": path.name, "name": item.name, "crop": slot})
    return labeled, skipped


def _id_for(name: str, by_name: dict[str, dict]) -> str | None:
    key = name.lower()
    for candidate in (key, key.removesuffix(" stencil")):
        if candidate in by_name:
            return by_name[candidate]["id"]
    return None


def _gallery_icons(
    labeled: list[dict], items: list[dict], cache_dir: Path
) -> tuple[dict[str, tuple[str, Path]], dict[str, str], int]:
    """Download the icons for the labeled items' types. Return the icon map, the
    name to id map for labeled names, and the number of fresh downloads."""
    by_name = {i["name"]["en"].lower(): i for i in items}
    by_id = {i["id"]: i for i in items}
    label_ids: dict[str, str] = {}
    for name in {row["name"] for row in labeled}:
        item_id = _id_for(name, by_name)
        if item_id is None:
            print(f"no arctracker id for labeled item {name!r}")
            continue
        label_ids[name] = item_id
    types = {by_id[i]["type"] for i in label_ids.values()}
    icons: dict[str, tuple[str, Path]] = {}
    fresh = 0
    for item in items:
        if item["type"] not in types:
            continue
        existed = (cache_dir / f"{item['id']}.png").is_file()
        try:
            path = blueprint_icon_path(item["id"], cache_dir)
        except IconDownloadError as e:
            print(f"skipped icon: {e}")
            continue
        fresh += not existed
        icons[item["id"]] = (item["name"]["en"], path)
    return icons, label_ids, fresh


def _pct(hits: int, total: int) -> str:
    return f"{100 * hits / total:.1f}% ({hits}/{total})" if total else "-"


def _measure(
    labeled: list[dict],
    label_ids: dict[str, str],
    icons: dict,
    method: str,
    title: str = "",
) -> tuple[list[str], list[dict]]:
    gallery = build_gallery(icons, method)
    rows = []
    times = []
    for row in labeled:
        item_id = label_ids.get(row["name"])
        if item_id is None:
            continue
        found, seconds = timed_match(row["crop"], gallery)
        times.append(seconds)
        ids = [c.item_id for c in found]
        rows.append(
            {
                **row,
                "id": item_id,
                "found": found,
                "top1": ids[:1] == [item_id],
                "top5": item_id in ids,
            }
        )
    lines = [title or f"== {method} ({len(gallery)} icons in the gallery) =="]
    lines.append(
        f"top-1 per crop: {_pct(sum(r['top1'] for r in rows), len(rows))}"
        f"  top-5 per crop: {_pct(sum(r['top5'] for r in rows), len(rows))}"
    )
    per_item: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_item[r["name"]].append(r)
    top1_items = sum(
        sum(r["top1"] for r in rs) * 2 > len(rs) for rs in per_item.values()
    )
    top5_items = sum(
        sum(r["top5"] for r in rs) * 2 > len(rs) for rs in per_item.values()
    )
    lines.extend(
        [
            (
                f"per distinct item (majority of its crops): top-1 "
                f"{_pct(top1_items, len(per_item))}  top-5 "
                f"{_pct(top5_items, len(per_item))}"
            ),
            (
                f"runtime per crop: median {1000 * statistics.median(times):.2f} ms, "
                f"max {1000 * max(times):.2f} ms (normalise, hash, rank)"
            ),
            "item: crops, top-1 hits, top-5 hits, top-1 answers",
        ]
    )
    for name, rs in sorted(per_item.items()):
        answers = defaultdict(int)
        for r in rs:
            answers[r["found"][0].name] += 1
        said = ", ".join(f"{n} x{c}" for n, c in sorted(answers.items()))
        lines.append(
            f"  {name}: {len(rs)}, {sum(r['top1'] for r in rs)}, "
            f"{sum(r['top5'] for r in rs)}, {said}"
        )
    confusions: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        if not r["top1"]:
            confusions[r["name"], r["found"][0].name] += 1
    lines.append("confusions (true item -> top-1 answer, crops):")
    lines.extend(
        f"  {a} -> {b}: {n}"
        for (a, b), n in sorted(confusions.items(), key=lambda x: -x[1])
    )
    return lines, rows


def _contact_sheet(rows: list[dict], icons: dict, path: Path) -> None:
    seen: set[tuple[str, str]] = set()
    picked = []
    for r in rows:
        key = (r["name"], r["found"][0].name)
        if key not in seen:
            seen.add(key)
            picked.append(r)
    cell_w, cell_h = TILE * 2 + 10, TILE + 30
    sheet_rows = (len(picked) + SHEET_COLUMNS - 1) // SHEET_COLUMNS
    sheet = Image.new("RGB", (cell_w * SHEET_COLUMNS, cell_h * max(1, sheet_rows)))
    draw = ImageDraw.Draw(sheet)
    for n, r in enumerate(picked):
        x, y = (n % SHEET_COLUMNS) * cell_w, (n // SHEET_COLUMNS) * cell_h
        crop = Image.fromarray(r["crop"])
        crop.thumbnail((TILE, TILE))
        sheet.paste(crop, (x, y))
        top = r["found"][0]
        with Image.open(icons[top.item_id][1]) as icon:
            flat = Image.new("RGBA", icon.size, (22, 30, 38, 255))
            tile = Image.alpha_composite(flat, icon.convert("RGBA")).convert("RGB")
        tile.thumbnail((TILE, TILE))
        sheet.paste(tile, (x + TILE + 10, y))
        colour = (90, 220, 90) if r["top1"] else (240, 90, 90)
        draw.text((x + 2, y + TILE + 2), f"true: {r['name']}", fill=(230, 230, 230))
        draw.text(
            (x + 2, y + TILE + 14), f"top1: {top.name} d={top.distance}", fill=colour
        )
    sheet.save(path)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: icon_hash_baseline.py <captures folder> <out folder>")
        return 2
    folder, out = Path(sys.argv[1]), Path(sys.argv[2])
    if not folder.is_dir():
        print(f"not a folder: {folder}")
        return 2
    out.mkdir(parents=True, exist_ok=True)

    labeled, skipped = _label_crops(folder)
    cache_dir = get_settings().icon_cache_dir
    items = _arctracker_items(cache_dir)
    icons, label_ids, fresh = _gallery_icons(labeled, items, cache_dir)

    report = [
        f"labeled crops: {len(labeled)}, skipped captures: {len(skipped)}",
        f"distinct labeled items: {len({r['name'] for r in labeled})}",
        f"gallery: {len(icons)} icons, {fresh} downloaded this run",
        "",
    ]
    report.extend(f"skipped {s}" for s in skipped)
    report.extend(["", "known bad labels, excluded from the cleaned tables only:"])
    report.extend(
        f"  {r['file']}: {BAD_LABELS[k]}"
        for r in labeled
        for k in BAD_LABELS
        if r["file"].startswith(k)
    )
    report.append("")
    for method in METHODS:
        lines, rows = _measure(labeled, label_ids, icons, method)
        report.extend(lines)
        report.append("")
        clean = [r for r in labeled if not r["file"].startswith(tuple(BAD_LABELS))]
        clean_lines, _ = _measure(
            clean,
            label_ids,
            icons,
            method,
            f"== {method}, known bad labels removed ({len(labeled) - len(clean)}) ==",
        )
        report.extend(clean_lines)
        report.append("")
        _contact_sheet(rows, icons, out / f"contact_{method}.png")
        with (out / f"labeled_{method}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "item", "top1", "top5", "answers"])
            for r in rows:
                answers = "; ".join(f"{c.name} {c.distance}" for c in r["found"])
                writer.writerow([r["file"], r["name"], r["top1"], r["top5"], answers])
    text = "\n".join(report)
    (out / "results.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
