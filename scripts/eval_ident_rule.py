# ruff: noqa: INP001, PLC2701
"""Why do filled slots stay unidentified? Measure the stash acceptance rule.

    uv run --extra icons python scripts/eval_ident_rule.py --out <folder>
        --captures <folder> --names <stash_scan.csv> <fixture.png> ...

Two sets of frames: the sample captures (the frame names in the first
column of --names, read from --captures) and the fixtures given as
arguments. Every filled slot is embedded once on the app's gallery and
twins (main.load_stash_gallery) and ranked as identify_slots ranks it, then
judged twice: by BASELINE_RULE, a frozen copy of the rule before this work,
and by stash_scan._resolve as it is now.

For every slot the baseline leaves unidentified the report gives every
applicable reason (similarity below MIN_SIMILARITY, base lead below
MIN_LEAD, ambiguous twins, unresolved catalog lookup) and the by-eye mark of
the top guess. Then right, wrong and unidentified per set, before and after.

By-eye marks come from --labels (default <out>/labels.csv when present), a
CSV of slot, mark, note, where slot is <frame stem>@<left>,<top> and mark is
right (the top guess names the item, for a tiered item its base item),
wrong (it does not), empty (an empty, locked or covered slot) or unsure
(cannot be told by eye). An identified slot without a mark counts as
right: the earlier by-eye checks of these frames (stash-scan-s1 and
loadout-ledger-s1) found every identification right except the ones marked
wrong in labels.csv. A slot whose decision changes without a mark is listed
under "unjudged".

Writes to --out: real-data.txt (the report), slots.csv (every filled slot
with its top TOP_N, tier read, lead and both decisions) and
unidentified_NN.png (one tile per baseline-unidentified slot, with its
slot id and top guess, for marking by eye).
"""

import argparse
import csv
import shutil
import tempfile
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from arc_helper.config import get_settings
from arc_helper.database import Database
from arc_helper.grid import find_slots
from arc_helper.icon_embed import crop_square
from arc_helper.icon_embed import default_embedder
from arc_helper.main import load_stash_gallery
from arc_helper.slot_marks import read_tier
from arc_helper.stash_scan import _rank
from arc_helper.stash_scan import _resolve
from arc_helper.stash_scan import base_name
from arc_helper.stash_scan import name_tier

BASE_MIN_SIMILARITY = 0.65
BASE_MIN_LEAD = 0.04
BASE_TOP_N = 5
TILE = 160
COLUMNS = 8
TEXT_ROWS = 4


def baseline_reasons(found, read, lookup, twins) -> tuple[list[str], object]:
    """Every part of the baseline rule that rejects the slot, and its item.

    The reasons are [] when the slot is accepted, the item None otherwise.
    """
    found = found[:BASE_TOP_N]
    reasons = []
    top = found[0]
    if top.similarity < BASE_MIN_SIMILARITY:
        reasons.append("similarity")
    if _baseline_lead(found) < BASE_MIN_LEAD:
        reasons.append("lead")
    catalog_tier = name_tier(top.name)
    settled = catalog_tier is not None and read is not None
    alike = twins.get(top.name, ())
    if alike and not (settled and all(name_tier(t) is None for t in alike)):
        reasons.append("twins")
    name = f"{base_name(top.name)} {read}" if settled else top.name
    item = lookup(name)
    if item is None:
        reasons.append("lookup")
    return reasons, None if reasons else item


def _baseline_lead(found) -> float:
    """base_lead before this work: only tier siblings are passed over."""
    top = base_name(found[0].name).casefold()
    others = [c.similarity for c in found[1:] if base_name(c.name).casefold() != top]
    rest = others[0] if others else found[-1].similarity
    return found[0].similarity - rest if len(found) > 1 else found[0].similarity


def _frames(args) -> list[tuple[str, Path]]:
    with args.names.open(encoding="utf-8") as handle:
        names = list(dict.fromkeys(row["frame"] for row in csv.DictReader(handle)))
    frames = [("sample", args.captures / name) for name in names]
    return frames + [("fixture", path) for path in args.images]


def _labels(path: Path | None) -> dict[str, tuple[str, str]]:
    if path is None or not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {
            row["slot"]: (row["mark"].strip(), row.get("note", "").strip())
            for row in csv.DictReader(handle)
        }


def _sheets(tiles: list[tuple[str, np.ndarray, list[Path]]], out: Path) -> None:
    font = ImageFont.load_default()
    per = COLUMNS * 6
    width = TILE * 2
    for n in range(0, len(tiles), per):
        part = tiles[n : n + per]
        rows = -(-len(part) // COLUMNS)
        height = TILE + 12 * TEXT_ROWS
        sheet = Image.new("RGB", (COLUMNS * width, rows * height), (24, 24, 24))
        draw = ImageDraw.Draw(sheet)
        for i, (text, crop, icons) in enumerate(part):
            x, y = (i % COLUMNS) * width, (i // COLUMNS) * height
            tile = Image.fromarray(crop)
            tile.thumbnail((TILE - 4, TILE - 4))
            sheet.paste(tile, (x + 2, y + 2))
            side = TILE // 2 - 2
            for k, icon_path in enumerate(icons[:4]):
                if not icon_path.is_file():
                    continue
                with Image.open(icon_path) as source:
                    icon = source.convert("RGBA")
                icon.thumbnail((side, side))
                spot = (x + TILE + (k % 2) * (side + 2), y + (k // 2) * (side + 2))
                sheet.paste(icon, spot, icon)
            for k, line in enumerate(textwrap.wrap(text, 52)[:TEXT_ROWS]):
                draw.text((x + 2, y + TILE + 12 * k), line, fill="white", font=font)
        sheet.save(out / f"unidentified_{n // per + 1:02d}.png")


def _decide(*, identified: bool, mark: str) -> str:
    if not identified:
        return "unidentified"
    if mark == "unsure":
        return "unjudged"
    return "wrong" if mark in {"wrong", "empty"} else "right"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--names", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    labels = _labels(args.labels or args.out / "labels.csv")
    settings = get_settings()
    gallery, twins = load_stash_gallery(settings.icon_cache_dir)
    embedder = default_embedder()
    rows, tiles = [], []
    with tempfile.TemporaryDirectory() as tmp:
        db_copy = Path(tmp) / "items.db"
        shutil.copy(settings.database_path, db_copy)
        db = Database(db_path=db_copy, user_db_path=Path(tmp) / "arlo_user.db")
        for group, path in _frames(args):
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"))
            filled = [s for s in find_slots(rgb) if s.filled]
            crops = [
                rgb[s.rect.top : s.rect.bottom, s.rect.left : s.rect.right]
                for s in filled
            ]
            vectors = embedder.embed([crop_square(c) for c in crops])
            for slot, crop, found in zip(
                filled, crops, _rank(vectors, gallery), strict=True
            ):
                tier_read = read_tier(crop)
                read = tier_read.tier if tier_read else None
                reasons, before = baseline_reasons(found, read, db.lookup, twins)
                after, _, lead = _resolve(found, read, db.lookup, twins)
                sid = f"{path.stem}@{slot.rect.left},{slot.rect.top}"
                mark, note = labels.get(sid, ("", ""))
                row = {
                    "set": group,
                    "slot": sid,
                    "panel": slot.panel,
                    "guess": found[0].name,
                    "similarity": round(found[0].similarity, 4),
                    "lead": round(lead, 4),
                    "read_tier": read or "",
                    "twins": ";".join(twins.get(found[0].name, ())),
                    "reasons": ";".join(reasons),
                    "before": before.name if before else "",
                    "after": after.name if after else "",
                    "mark": mark,
                    "note": note,
                    "top": " | ".join(f"{c.name} {c.similarity:.4f}" for c in found),
                }
                row["before_result"] = _decide(identified=not reasons, mark=mark)
                row["after_result"] = _decide(identified=after is not None, mark=mark)
                rows.append(row)
                if reasons:
                    icons = [
                        settings.icon_cache_dir / f"{c.item_id}.png" for c in found
                    ]
                    names = ", ".join(c.name for c in found[:4])
                    tiles.append((f"{len(tiles)} {sid} | {names}", crop, icons))
    with (args.out / "slots.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _sheets(tiles, args.out)
    text = "\n".join(_report(rows))
    print(text)
    (args.out / "real-data.txt").write_text(text + "\n", encoding="utf-8")


def _report(rows: list[dict]) -> list[str]:
    lines = [
        (
            "baseline rule: similarity >= 0.65, base lead >= 0.04, twins settled by"
            " the numeral, catalog lookup resolves"
        ),
        (
            "new rule: the same, and a slot whose tiered top name is settled by a"
            " read numeral passes over untiered candidates in the lead, TOP_N 10"
        ),
        (
            "marks: right (top guess names the item, its base item when tiered),"
            " wrong, empty (empty, locked or covered slot), unsure, blank"
            " (identified and not marked: right by the earlier by-eye checks)"
        ),
        "",
    ]
    for group in ("sample", "fixture"):
        part = [r for r in rows if r["set"] == group]
        lines.extend(
            (
                f"== {group}: {len(part)} filled slots",
                (
                    "unidentified by the baseline (slot, guess, sim, lead, tier"
                    " read, reasons, mark, after):"
                ),
            )
        )
        for r in part:
            if r["before_result"] != "unidentified":
                continue
            lines.append(
                f"  {r['slot']} [{r['panel']}] {r['guess']} sim {r['similarity']}"
                f" lead {r['lead']} tier {r['read_tier'] or '-'}"
                f" reasons {r['reasons']} mark {r['mark'] or '-'}"
                f" after {r['after'] or '?'}" + (f" ({r['note']})" if r["note"] else "")
            )
        for when in ("before", "after"):
            counts = {
                k: sum(r[f"{when}_result"] == k for r in part)
                for k in ("right", "wrong", "unidentified", "unjudged")
            }
            lines.append(
                f"{when}: {counts['right']} right, {counts['wrong']} wrong,"
                f" {counts['unidentified']} unidentified, {counts['unjudged']}"
                " identified but unsure by eye"
            )
        guesses: dict[str, int] = {}
        for r in part:
            if r["before_result"] == "unidentified":
                key = r["mark"] or "unmarked"
                guesses[key] = guesses.get(key, 0) + 1
        lines.append(
            "top guess of the baseline-unidentified slots by eye: "
            + ", ".join(f"{k} {v}" for k, v in sorted(guesses.items()))
        )
        reasons: dict[str, int] = {}
        for r in part:
            if r["reasons"]:
                reasons[r["reasons"]] = reasons.get(r["reasons"], 0) + 1
        lines.append(
            "baseline rejections by reason set: "
            + ", ".join(
                f"{k} {v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])
            )
        )
        changed = [r for r in part if r["before"] != r["after"]]
        lines.append(f"changed by the new rule: {len(changed)}")
        lines.extend(
            f"  {r['slot']} {r['before'] or '?'} -> {r['after'] or '?'}"
            f" ({r['before_result']} -> {r['after_result']})"
            for r in changed
        )
        unjudged = [
            r
            for r in changed
            if r["after_result"] != "unidentified" and r["mark"] in {"", "unsure"}
        ]
        lines.append(f"unjudged: {len(unjudged)}")
        lines.extend(f"  {r['slot']}" for r in unjudged)
        lines.append("")
    return lines


if __name__ == "__main__":
    main()
