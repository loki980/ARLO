"""Compare the card title readers on a labeled capture library.

Usage: uv run python scripts/bakeoff_title_readers.py <captures> <labels.csv>

labels.csv has the columns file,expected,card,notes. Only the files it lists
are read, so a library that keeps growing gives the same result. Rows with
card no are skipped. Rows whose expected is "unreadable" (a card with its
title cut off) are counted separately, together with how often each reader
still resolved them to some catalog name. Every other row with card yes is
scored.

The card finder runs once per image with the cursor assumed at the image
centre, and its crop goes to every reader. With no card found the whole
image goes to the readers, as the scanner falls back to its fixed box.
Each read resolves through Database.lookup with the reader's tier vote.

Prints per reader: exact title accuracy (the read equals the expected name
ignoring case, spaces and punctuation), resolved catalog accuracy, the
median time per card, and the misses with read versus expected.

Needs Tesseract and items.db, no display. The paddle reader needs the
rapidocr-onnxruntime package and is reported as unavailable without it.
"""

import csv
import logging
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from arc_helper.database import Database
from arc_helper.database import _match_key
from arc_helper.ocr import crop_card
from arc_helper.title_reader import ENGINES
from arc_helper.title_reader import TitleReaderUnavailable
from arc_helper.title_reader import make_reader
from arc_helper.title_reader import timed_read
from arc_helper.tooltip_layout import Point

UNREADABLE = "unreadable"


def _load_labels(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return [
            {k: (v or "").strip() for k, v in row.items() if k}
            for row in csv.DictReader(f)
            if (row.get("file") or "").strip()
        ]


def _percent(hits: int, total: int) -> str:
    return f"{hits}/{total} ({100 * hits / total:.1f}%)" if total else "0/0"


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

    readers = {}
    for engine in ENGINES:
        try:
            readers[engine] = make_reader(engine)
        except TitleReaderUnavailable as e:
            print(f"{engine}: unavailable, {e}")

    rows = [r for r in _load_labels(labels_path) if r.get("card") == "yes"]
    scored = [r for r in rows if r.get("expected") and r["expected"] != UNREADABLE]
    unreadable = [r for r in rows if r.get("expected") == UNREADABLE]
    skipped = sum(1 for r in _load_labels(labels_path) if r.get("card") != "yes")

    exact = dict.fromkeys(readers, 0)
    resolved = dict.fromkeys(readers, 0)
    false_reads = dict.fromkeys(readers, 0)
    times: dict[str, list[float]] = {name: [] for name in readers}
    misses: dict[str, list[str]] = {name: [] for name in readers}
    no_card = 0
    missing_files = 0

    for row in scored + unreadable:
        path = folder / row["file"]
        if not path.is_file():
            missing_files += 1
            continue
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        found = crop_card(rgb, Point(rgb.width // 2, rgb.height // 2))
        if found is None:
            no_card += 1
        else:
            rgb = found[0]
        card = np.array(rgb)

        expected = row["expected"]
        for name, reader in readers.items():
            read, seconds = timed_read(reader, card)
            times[name].append(seconds)
            item = db.lookup(read.text, tier_vote=read.tier_vote) if read.text else None
            if expected == UNREADABLE:
                false_reads[name] += item is not None
                continue
            exact_hit = _match_key(read.text or "") == _match_key(expected)
            resolved_hit = item is not None and item.name.casefold() == (
                expected.casefold()
            )
            exact[name] += exact_hit
            resolved[name] += resolved_hit
            if not (exact_hit and resolved_hit):
                vote = read.tier_vote
                vote_text = (
                    f"{vote.tier} {vote.confidence:.2f}" if vote and vote.tier else "-"
                )
                misses[name].append(
                    f"  {row['file']}  read={read.text or '-'!s}  vote={vote_text}  "
                    f"resolved={item.name if item else '-'}  expected={expected}"
                )

    print(f"labels: {labels_path}")
    print(
        f"scored rows: {len(scored)}  unreadable rows: {len(unreadable)}  "
        f"card no (skipped): {skipped}  no card found: {no_card}  "
        f"missing files: {missing_files}"
    )
    print()
    header = (
        f"{'reader':<16}  {'exact title':<16}  {'resolved catalog':<16}  "
        f"{'median ms/card':>14}  {'unreadable resolved':>19}"
    )
    print(header)
    print("-" * len(header))
    for name in readers:
        median_ms = 1000 * statistics.median(times[name]) if times[name] else 0.0
        print(
            f"{name:<16}  {_percent(exact[name], len(scored)):<16}  "
            f"{_percent(resolved[name], len(scored)):<16}  {median_ms:>14.0f}  "
            f"{false_reads[name]:>8}/{len(unreadable):<10}"
        )

    for name in readers:
        print()
        print(f"{name} misses ({len(misses[name])}):")
        for line in misses[name]:
            print(line)

    if readers:
        winner = max(
            readers,
            key=lambda n: (resolved[n], -statistics.median(times[n] or [0.0])),
        )
        print()
        print(f"winner by resolved catalog accuracy, ties by time: {winner}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
