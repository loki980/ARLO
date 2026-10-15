# ruff: noqa: INP001, PLC2701, SLF001
"""Compare the card title readers on a labeled capture library.

Usage: uv run python scripts/bakeoff_title_readers.py <captures> <labels.csv>

labels.csv has the columns file,expected,card,notes. Only the files it lists
are read, so a library that keeps growing gives the same result. Rows with
card no are skipped. Rows whose expected is "unreadable" (a card with its
title cut off) are counted separately, together with how often each reader
still resolved them to some catalog name. Every other row with card yes is
scored.

The card finder runs once per image with the cursor assumed at the image
centre, and its crop goes to every reader. A row where the finder finds no
card is listed under crop failures and left out of every reader's accuracy.
The whole screenshot is never given to a reader.
Each read resolves through Database.lookup with the reader's tier vote, and
once more with the last-resort tolerant step turned off, so the gain from
that step alone is visible.

Prints per reader: exact title accuracy (the read equals the expected name
ignoring case, spaces and punctuation), resolved catalog accuracy with and
without the tolerant step, the median time per card, and the misses with
read versus expected. A focus section then shows every reader's read for the
names that failed in game (Rattler III, Anvil IV, Burletta IV, Anvil I),
hit or miss.

Needs Tesseract and items.db, no display. The paddle reader needs the
rapidocr-onnxruntime package (uv sync --extra paddle) and is reported as
unavailable without it.
"""

import csv
import logging
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from arc_helper.database import Database
from arc_helper.database import Item
from arc_helper.database import _match_key
from arc_helper.ocr import crop_card
from arc_helper.title_reader import ENGINES
from arc_helper.title_reader import TitleRead
from arc_helper.title_reader import TitleReaderUnavailableError
from arc_helper.title_reader import make_reader
from arc_helper.title_reader import timed_read
from arc_helper.tooltip_layout import Point

UNREADABLE = "unreadable"
FOCUS = ("Rattler III", "Anvil IV", "Burletta IV", "Anvil I")
BASELINE = "tesseract"


def _load_labels(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return [
            {k: (v or "").strip() for k, v in row.items() if k}
            for row in csv.DictReader(f)
            if (row.get("file") or "").strip()
        ]


def _percent(hits: int, total: int) -> str:
    return f"{hits}/{total} ({100 * hits / total:.1f}%)" if total else "0/0"


def _vote_text(read: TitleRead) -> str:
    vote = read.tier_vote
    return f"{vote.tier} {vote.confidence:.2f}" if vote and vote.tier else "-"


def _lookup(db: Database, read: TitleRead) -> Item | None:
    return db.lookup(read.text, tier_vote=read.tier_vote) if read.text else None


def _median_ms(times: list[float]) -> float:
    return 1000 * statistics.median(times) if times else 0.0


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: bakeoff_title_readers.py <captures> <labels.csv>")
        return 2
    folder, labels_path = Path(sys.argv[1]), Path(sys.argv[2])
    if not folder.is_dir() or not labels_path.is_file():
        print(f"need a folder and a labels file: {folder} {labels_path}")
        return 2

    logging.getLogger("arc_helper").setLevel(logging.WARNING)
    db = Database()
    # The same catalog with the last-resort tolerant step turned off
    strict_db = Database()
    strict_db._resolve_tolerant = lambda _name, _vote: None

    readers = {}
    unavailable: dict[str, str] = {}
    for engine in ENGINES:
        try:
            readers[engine] = make_reader(engine)
        except TitleReaderUnavailableError as e:
            unavailable[engine] = str(e)

    labels = _load_labels(labels_path)
    rows = [r for r in labels if r.get("card") == "yes"]
    scored = [r for r in rows if r.get("expected") and r["expected"] != UNREADABLE]
    unreadable = [r for r in rows if r.get("expected") == UNREADABLE]
    skipped = len(labels) - len(rows)

    exact = dict.fromkeys(readers, 0)
    resolved = dict.fromkeys(readers, 0)
    resolved_strict = dict.fromkeys(readers, 0)
    false_reads = dict.fromkeys(readers, 0)
    times: dict[str, list[float]] = {name: [] for name in readers}
    misses: dict[str, list[str]] = {name: [] for name in readers}
    focus: dict[str, list[str]] = {name: [] for name in FOCUS}
    crop_failures: list[tuple[str, str]] = []
    missing_files: list[str] = []
    read_scored = 0
    read_unreadable = 0

    for row in scored + unreadable:
        expected = row["expected"]
        path = folder / row["file"]
        if not path.is_file():
            missing_files.append(row["file"])
            continue
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        found = crop_card(rgb, Point(rgb.width // 2, rgb.height // 2))
        if found is None:
            crop_failures.append((row["file"], expected))
            if expected in FOCUS:
                focus[expected].append(f"  {row['file']}  no card found, not read")
            continue
        card = np.array(found[0])
        if expected == UNREADABLE:
            read_unreadable += 1
        else:
            read_scored += 1

        focus_parts: list[str] = []
        for name, reader in readers.items():
            read, seconds = timed_read(reader, card)
            times[name].append(seconds)
            item = _lookup(db, read)
            if expected == UNREADABLE:
                false_reads[name] += item is not None
                continue
            strict_item = _lookup(strict_db, read)
            exact_hit = _match_key(read.text or "") == _match_key(expected)
            resolved_hit = item is not None and item.name.casefold() == (
                expected.casefold()
            )
            exact[name] += exact_hit
            resolved[name] += resolved_hit
            resolved_strict[name] += strict_item is not None and (
                strict_item.name.casefold() == expected.casefold()
            )
            outcome = (
                f"read={read.text or '-'!s}  vote={_vote_text(read)}  "
                f"resolved={item.name if item else '-'}"
            )
            if not (exact_hit and resolved_hit):
                misses[name].append(f"  {row['file']}  {outcome}  expected={expected}")
            if expected in FOCUS:
                verdict = "hit" if resolved_hit else "miss"
                focus_parts.append(
                    f"{name}: {read.text or '-'} ({_vote_text(read)}) -> "
                    f"{item.name if item else '-'} {verdict}"
                )
        if focus_parts:
            focus[expected].append(f"  {row['file']}  " + " | ".join(focus_parts))

    print(f"labels: {labels_path}")
    print(
        f"scored rows: {len(scored)}  unreadable rows: {len(unreadable)}  "
        f"card no (skipped): {skipped}  missing files: {len(missing_files)}"
    )
    print(
        f"crop failures: {len(crop_failures)}  "
        f"scored rows read: {read_scored}  unreadable rows read: {read_unreadable}"
    )
    print()
    header = (
        f"{'reader':<16}  {'exact title':<16}  {'resolved catalog':<16}  "
        f"{'no tolerance':<16}  {'median ms/card':>14}  {'unreadable resolved':>19}"
    )
    print(header)
    print("-" * len(header))
    for name in readers:
        print(
            f"{name:<16}  {_percent(exact[name], read_scored):<16}  "
            f"{_percent(resolved[name], read_scored):<16}  "
            f"{_percent(resolved_strict[name], read_scored):<16}  "
            f"{_median_ms(times[name]):>14.0f}  "
            f"{false_reads[name]:>8}/{read_unreadable:<10}"
        )
    for name, why in unavailable.items():
        print(f"{name:<16}  unavailable: {why}")

    print()
    print(f"crop failures, no card found, not read ({len(crop_failures)}):")
    for file, expected in crop_failures:
        print(f"  {file}  expected={expected}")
    if missing_files:
        print()
        print(f"missing files ({len(missing_files)}):")
        for file in missing_files:
            print(f"  {file}")

    for name in readers:
        print()
        print(f"{name} misses ({len(misses[name])}):")
        for line in misses[name]:
            print(line)

    print()
    print("focus rows, every reader's read (vote) -> resolved, hit or miss:")
    for expected, lines in focus.items():
        print(f" {expected} ({len(lines)} rows)")
        for line in lines:
            print(line)

    if readers:
        winner = max(readers, key=lambda n: (resolved[n], -_median_ms(times[n])))
        print()
        print(f"winner by resolved catalog accuracy, ties by time: {winner}")
        if BASELINE in readers and winner != BASELINE:
            margin = resolved[winner] - resolved[BASELINE]
            print(
                f"margin over {BASELINE}: {margin:+d} of {read_scored} scored rows, "
                f"{_median_ms(times[winner]):.0f} ms against "
                f"{_median_ms(times[BASELINE]):.0f} ms per card"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
