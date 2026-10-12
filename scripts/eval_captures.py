# ruff: noqa: INP001, PLC2701, SLF001
"""Evaluate the item name read and tier resolution over saved captures.

Usage: uv run python scripts/eval_captures.py <folder>

Runs the real OCR name read and database lookup on every PNG in the folder
(for example the capture library written with CAPTURE_LIBRARY=true) and
prints one row per file: file, read text, tier vote and confidence, and the
resolved catalog name. With a labels.csv in the folder (columns
file,expected) it also prints the accuracy of the read text, of the
resolver without the vote, and of the resolver with the vote.

Needs Tesseract and items.db, no display.
"""

import csv
import logging
import sys
from pathlib import Path

from PIL import Image

from arc_helper.database import Database
from arc_helper.database import _match_key
from arc_helper.ocr import OCREngine


def _load_labels(folder: Path) -> dict[str, str]:
    path = folder / "labels.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        return {
            row["file"].strip(): (row.get("expected") or "").strip()
            for row in csv.DictReader(f)
            if row.get("file")
        }


def _same(name: str | None, expected: str) -> bool:
    return (name or "").casefold() == expected.casefold()


def _percent(hits: int, total: int) -> str:
    return f"{hits}/{total} ({100 * hits / total:.1f}%)" if total else "0/0"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: eval_captures.py <folder>")
        return 2
    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"not a folder: {folder}")
        return 2

    # Keep the output to the table, even with DEBUG_MODE on
    logging.getLogger("arc_helper").setLevel(logging.WARNING)

    engine = OCREngine()
    db = Database()
    labels = _load_labels(folder)
    files = sorted(folder.glob("*.png"))
    width = max((len(f.name) for f in files), default=4)

    print(f"{'file':<{width}}  {'read':<24}  {'vote':<9}  resolved")
    hits = {"read": 0, "no_vote": 0, "vote": 0}
    labeled = 0
    for path in files:
        with Image.open(path) as image:
            processed = engine.preprocess_tooltip(image.convert("RGB"))
        read = engine._read_item_name(processed)
        vote = engine.last_tier_vote
        with_vote = db.lookup(read, tier_vote=vote) if read else None

        vote_text = f"{vote.tier} {vote.confidence:.2f}" if vote and vote.tier else "-"
        resolved = with_vote.name if with_vote else "-"
        print(f"{path.name:<{width}}  {read or '-':<24}  {vote_text:<9}  {resolved}")

        if path.name in labels:
            expected = labels[path.name]
            labeled += 1
            without_vote = db.lookup(read) if read else None
            hits["read"] += _match_key(read or "") == _match_key(expected)
            hits["no_vote"] += _same(without_vote and without_vote.name, expected)
            hits["vote"] += _same(with_vote and with_vote.name, expected)

    if labels:
        print()
        print(f"labeled files: {labeled}")
        print(f"read text:              {_percent(hits['read'], labeled)}")
        print(f"resolver without vote:  {_percent(hits['no_vote'], labeled)}")
        print(f"resolver with vote:     {_percent(hits['vote'], labeled)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
