# ruff: noqa: INP001, PLC2701
"""Measure the slot tier numeral and stack count readers on labeled crops.

Usage:
    uv run python scripts/eval_slot_marks.py <captures folder> <out folder>
        [--embed-csv <labeled_embed.csv>]

The labeled crops are the icon baseline's (hovered cell plus the title read,
see icon_hash_baseline.py) with its BAD_LABELS removed. The expected tier is
the tier in the item name. Expected stack counts were labeled by eye from the
slot bars (STACK_LABELS). The combined figure pairs the base item of the
embedding's top-1 answer from icon-embed-s1's labeled_embed.csv with the
tier read here. By default that file is looked up in the handoff folder next
to the out folder's (../../icon-embed-s1/attempt-1/labeled_embed.csv).

Writes results.txt (the tables), reads.csv (one row per crop) and
contact_bars.png (every slot bar with its reads) to the out folder.
"""

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from icon_hash_baseline import BAD_LABELS
from icon_hash_baseline import _label_crops
from PIL import Image
from PIL import ImageDraw

from arc_helper.slot_marks import read_stack
from arc_helper.slot_marks import read_tier
from arc_helper.tiers import split_tier

# Capture prefix to the stack count on its slot bar, read by eye from the
# bars. None is a non-weapon slot with no count. Weapons are not listed,
# they show a numeral and are expected to give None.
STACK_LABELS: dict[str, int | None] = {
    "20261012-001659": 20,
    "20261012-001700": 80,
    "20261012-001702": 80,
    "20261014-210134": 2,
    "20261014-210137": 2,
    "20261014-210138": None,
    "20261014-210140": None,
    "20261014-210143": None,
    "20261014-210144": 1,
    "20261014-210147": 1,
    "20261014-210150": 1,
    "20261014-210151": 9,
    "20261014-210154": 9,
    "20261014-210156": 9,
    "20261014-210158": 3,
    "20261014-210201": 3,
    "20261014-210202": 1,
    "20261014-210900": 1,
    "20261014-210903": 1,
    "20261014-210905": 1,
    "20261014-210908": 1,
    "20261014-210911": 1,
    "20261014-210912": 1,
    "20261014-210915": 1,
    "20261014-210916": 2,
    "20261014-210919": 2,
    "20261014-210931": None,
    "20261014-210953": 1,
    "20261014-210956": 1,
    "20261014-210959": 1,
    "20261014-211002": 1,
    "20261014-211005": 1,
    "20261014-211007": 1,
    "20261014-211010": 1,
    "20261014-211013": 90,
    "20261014-211014": 1,
    "20261014-211149": 2,
    "20261014-211151": 2,
    "20261014-211154": 2,
    "20261014-211156": 2,
    "20261014-211232": 1,
    "20261014-211235": None,
    "20261014-211237": None,
    "20261014-211239": None,
    "20261014-211241": None,
    "20261014-211243": None,
    "20261014-211246": None,
    "20261014-211248": None,
    "20261014-211252": None,
    "20261014-211253": None,
    "20261014-211258": 1,
    "20261014-211300": 1,
    "20261014-211306": 1,
    "20261014-211309": 1,
    "20261014-211312": 1,
    "20261014-211315": 1,
    "20261014-211318": 1,
    "20261014-211321": 1,
    "20261014-211324": 1,
    "20261014-211327": 1,
    "20261014-211328": 2,
    "20261014-211331": 9,
    "20261014-211334": 9,
    "20261014-211336": 3,
    "20261014-211337": 2,
    "20261014-211343": 80,
}
PREFIX = len("20261012-001659")
WIDE_ASPECT = 1.3
SHEET_COLUMNS = 4
TILE_W, TILE_H = 300, 64


def _base(name: str) -> str:
    return split_tier(name)[0].removesuffix(" Stencil")


def _pct(hits: int, total: int) -> str:
    return f"{100 * hits / total:.1f}% ({hits}/{total})" if total else "-"


def _ms(seconds: list[float]) -> str:
    if not seconds:
        return "-"
    median = 1000 * statistics.median(seconds)
    return f"median {median:.2f} ms, max {1000 * max(seconds):.2f} ms"


def _embed_top1(path: Path) -> dict[str, str]:
    """File name to the embedding's top-1 item name."""
    top1 = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            first = row["answers (similarity x100)"].split("; ")[0]
            top1[row["file"]] = first.rsplit(" ", 1)[0]
    return top1


def _read_all(labeled: list[dict]) -> list[dict]:
    rows = []
    for row in labeled:
        crop = row["crop"]
        start = time.perf_counter()
        tier = read_tier(crop)
        middle = time.perf_counter()
        stack = read_stack(crop)
        end = time.perf_counter()
        height, width = crop.shape[:2]
        rows.append(
            {
                **row,
                "true_tier": split_tier(row["name"])[1],
                "wide": width / height > WIDE_ASPECT,
                "tier": tier,
                "stack": stack,
                "tier_s": middle - start,
                "stack_s": end - middle,
            }
        )
    return rows


def _tier_lines(rows: list[dict]) -> list[str]:
    weapons = [r for r in rows if r["true_tier"]]
    others = [r for r in rows if not r["true_tier"]]
    lines = ["== tier numerals =="]
    for label, group in (
        ("square weapon slots", [r for r in weapons if not r["wide"]]),
        ("wide weapon slots", [r for r in weapons if r["wide"]]),
        ("all weapon slots", weapons),
    ):
        hits = sum(
            r["tier"] is not None and r["tier"].tier == r["true_tier"] for r in group
        )
        misses = sum(r["tier"] is None for r in group)
        lines.append(
            f"{label}: right {_pct(hits, len(group))}, no read {misses}, "
            f"wrong {len(group) - hits - misses}"
        )
    false_reads = [r for r in others if r["tier"] is not None]
    lines.append(
        f"non-weapon slots: false tier reads {len(false_reads)} of {len(others)}"
    )
    lines.extend(f"  {r['file']}: {r['name']} read {r['tier']}" for r in false_reads)
    lines.extend(
        f"  wrong or missing: {r['file']}: {r['name']} read {r['tier']}"
        for r in weapons
        if r["tier"] is None or r["tier"].tier != r["true_tier"]
    )
    confidences = [r["tier"].confidence for r in rows if r["tier"]]
    if confidences:
        lines.append(
            f"confidence of reads: min {min(confidences)}, max {max(confidences)}"
        )
    lines.append(f"runtime per slot: {_ms([r['tier_s'] for r in rows])}")
    for wide in (False, True):
        times = [r["tier_s"] for r in rows if r["wide"] == wide]
        lines.append(f"  {'wide' if wide else 'square'} slots: {_ms(times)}")
    return lines


def _stack_lines(rows: list[dict]) -> list[str]:
    lines = ["== stack counts =="]
    labeled = [r for r in rows if r["file"][:PREFIX] in STACK_LABELS or r["true_tier"]]
    for r in labeled:
        r["true_stack"] = STACK_LABELS.get(r["file"][:PREFIX])
    with_count = [r for r in labeled if r["true_stack"] is not None]
    without = [r for r in labeled if r["true_stack"] is None]
    right = sum(r["stack"] == r["true_stack"] for r in with_count)
    missed = sum(r["stack"] is None for r in with_count)
    weapons_without = sum(bool(r["true_tier"]) for r in without)
    lines.extend(
        [
            (
                f"hand-labeled sample: {len(labeled)} crops, {len(with_count)} with "
                f"a count, {len(without)} without (of them {weapons_without} weapons)"
            ),
            (
                f"slots with a count: right {_pct(right, len(with_count))}, "
                f"no read {missed}, wrong {len(with_count) - right - missed}"
            ),
        ]
    )
    false_reads = [r for r in without if r["stack"] is not None]
    lines.append(
        f"slots without a count: false reads {len(false_reads)} of {len(without)}"
    )
    lines.extend(
        f"  wrong or missing: {r['file']}: {r['name']} expected {r['true_stack']} "
        f"read {r['stack']}"
        for r in labeled
        if r["stack"] != r["true_stack"]
    )
    ocr = [r["stack_s"] for r in rows if r["stack"] is not None]
    skipped = [r["stack_s"] for r in rows if r["stack"] is None]
    lines.extend(
        [
            f"runtime with OCR (a count read): {_ms(ocr)}",
            f"runtime with no count read (a failed OCR counts here): {_ms(skipped)}",
        ]
    )
    return lines


def _combined_lines(rows: list[dict], embed_csv: Path) -> list[str]:
    lines = ["== base item from the embedding plus the tier from the slot =="]
    if not embed_csv.is_file():
        return [*lines, f"skipped, no file at {embed_csv}"]
    top1 = _embed_top1(embed_csv)
    paired = [r for r in rows if r["file"] in top1]
    lines.append(f"embedding results: {embed_csv} ({len(paired)} crops paired)")
    for label, group in (
        ("weapons", [r for r in paired if r["true_tier"]]),
        ("non-weapons", [r for r in paired if not r["true_tier"]]),
        ("all", paired),
    ):
        embed_only = sum(
            _base(top1[r["file"]]) + str(split_tier(top1[r["file"]])[1])
            == _base(r["name"]) + str(r["true_tier"])
            for r in group
        )
        base = sum(_base(top1[r["file"]]) == _base(r["name"]) for r in group)
        combined = sum(
            _base(top1[r["file"]]) == _base(r["name"])
            and (r["tier"].tier if r["tier"] else None) == r["true_tier"]
            for r in group
        )
        lines.append(
            f"{label}: embedding top-1 alone {_pct(embed_only, len(group))}, "
            f"base item {_pct(base, len(group))}, "
            f"base item plus slot tier {_pct(combined, len(group))}"
        )
    return lines


def _contact_sheet(rows: list[dict], path: Path) -> None:
    count = len(rows)
    sheet_rows = (count + SHEET_COLUMNS - 1) // SHEET_COLUMNS
    tile_h = TILE_H + 26
    sheet = Image.new("RGB", (TILE_W * SHEET_COLUMNS, tile_h * max(1, sheet_rows)))
    draw = ImageDraw.Draw(sheet)
    for n, r in enumerate(rows):
        x, y = (n % SHEET_COLUMNS) * TILE_W, (n // SHEET_COLUMNS) * tile_h
        crop = Image.fromarray(r["crop"])
        bar = crop.crop((0, int(crop.height * 0.68), crop.width, crop.height))
        bar.thumbnail((TILE_W - 4, TILE_H))
        sheet.paste(bar, (x, y))
        tier = r["tier"].tier if r["tier"] else None
        expected_stack = r.get("true_stack")
        ok = tier == r["true_tier"] and r["stack"] == expected_stack
        colour = (90, 220, 90) if ok else (240, 90, 90)
        draw.text(
            (x + 2, y + TILE_H + 1),
            f"{r['name']} {r['file'][9:15]}",
            fill=(230, 230, 230),
        )
        draw.text(
            (x + 2, y + TILE_H + 13),
            f"tier {tier} stack {r['stack']}",
            fill=colour,
        )
    sheet.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("captures", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--embed-csv", type=Path)
    args = parser.parse_args()
    if not args.captures.is_dir():
        print(f"not a folder: {args.captures}")
        return 2
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    embed_csv = args.embed_csv or (
        out.resolve().parent.parent
        / "icon-embed-s1"
        / "attempt-1"
        / "labeled_embed.csv"
    )

    labeled, skipped = _label_crops(args.captures)
    clean = [r for r in labeled if not r["file"].startswith(tuple(BAD_LABELS))]
    read_tier(np.zeros((8, 8, 3), np.uint8))  # first call warms up numpy
    rows = _read_all(clean)

    weapons = [r for r in rows if r["true_tier"]]
    report = [
        (
            f"labeled crops: {len(labeled)}, known bad labels removed: "
            f"{len(labeled) - len(clean)}, measured: {len(clean)}, "
            f"skipped captures: {len(skipped)}"
        ),
        f"weapon crops: {len(weapons)} ({sum(r['wide'] for r in weapons)} wide)",
        "",
    ]
    report.extend(_tier_lines(rows))
    report.append("")
    report.extend(_stack_lines(rows))
    report.append("")
    report.extend(_combined_lines(rows, embed_csv))
    _contact_sheet(rows, out / "contact_bars.png")
    with (out / "reads.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "file",
                "item",
                "wide",
                "true tier",
                "tier",
                "confidence",
                "detail",
                "true stack",
                "stack",
                "tier ms",
                "stack ms",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["file"],
                    r["name"],
                    r["wide"],
                    r["true_tier"],
                    r["tier"].tier if r["tier"] else None,
                    r["tier"].confidence if r["tier"] else None,
                    r["tier"].detail if r["tier"] else None,
                    r.get("true_stack"),
                    r["stack"],
                    f"{1000 * r['tier_s']:.2f}",
                    f"{1000 * r['stack_s']:.1f}",
                ]
            )
    text = "\n".join(report)
    (out / "results.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
