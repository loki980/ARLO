# ruff: noqa: INP001, PLC2701, SLF001
"""Evaluate the item name read and tier resolution over saved captures.

Usage: uv run python scripts/eval_captures.py <folder> [labels.csv]

Runs the real OCR name read and database lookup on every PNG in the folder
(for example the capture library written by uv run arc-helper --capture) and
prints one row per file: file, located card, read text, tier vote and
confidence, and the resolved catalog name. Each image is read the way the
scanner reads it, with the reader OCR_ENGINE names: the card is located with
the cursor from the capture's sidecar (<name>.json, the cursor minus the
image origin) or, without one, assumed at the image centre, and the reader
runs on its crop. The cursor column says which was used. With
no card the tesseract engine reads the whole image, as for an old fixed-box
capture, and the other engines read nothing. With a labels.csv in the folder,
or given after it (columns file,expected), it also prints the accuracy of the
read text, of the resolver without the vote, and of the resolver with the
vote. It ends with the effective reader (a fallback shows here) and, over the
captures with a card, the median ms for the full name read and for
layout_from_last_capture.

Needs Tesseract and items.db, no display.
"""

import csv
import json
import logging
import statistics
import sys
import time
from pathlib import Path

from PIL import Image

from arc_helper.database import Database
from arc_helper.database import _match_key
from arc_helper.ocr import OCREngine
from arc_helper.tooltip_layout import Point


def _load_labels(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        return {
            row["file"].strip(): (row.get("expected") or "").strip()
            for row in csv.DictReader(f)
            if row.get("file")
        }


def _cursor_for(path: Path, image: Image.Image) -> tuple[Point, str]:
    """The locate cursor in image coordinates, and whether it came from the sidecar.

    Uses the sidecar's cursor minus its origin when <name>.json contains a
    cursor, else the image centre.
    """
    centre = Point(image.width // 2, image.height // 2)
    try:
        data = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        x, y = data["cursor"]
        ox, oy = data.get("origin") or (0, 0)
        return Point(int(x) - int(ox), int(y) - int(oy)), "sidecar"
    except (OSError, ValueError, TypeError, KeyError):
        return centre, "centre"


def _same(name: str | None, expected: str) -> bool:
    return (name or "").casefold() == expected.casefold()


def _percent(hits: int, total: int) -> str:
    return f"{hits}/{total} ({100 * hits / total:.1f}%)" if total else "0/0"


def _median_ms(seconds: list[float]) -> str:
    return f"{1000 * statistics.median(seconds):.0f} ms" if seconds else "-"


def main() -> int:
    if len(sys.argv) not in {2, 3}:
        print("usage: eval_captures.py <folder> [labels.csv]")
        return 2
    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"not a folder: {folder}")
        return 2

    # Keep the output to the table, even with DEBUG_MODE on
    logging.getLogger("arc_helper").setLevel(logging.WARNING)

    engine = OCREngine()
    db = Database()
    labels_path = Path(sys.argv[2]) if len(sys.argv) == 3 else folder / "labels.csv"
    labels = _load_labels(labels_path)
    files = sorted(folder.glob("*.png"))
    width = max((len(f.name) for f in files), default=4)

    print(
        f"{'file':<{width}}  {'cursor':<7}  {'card':<21}  {'read':<24}  {'vote':<9}  resolved"
    )
    hits = {"read": 0, "no_vote": 0, "vote": 0}
    labeled = 0
    skipped = 0
    read_times: list[float] = []
    layout_times: list[float] = []
    for path in files:
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
        except (OSError, ValueError) as e:
            print(f"skipped {path.name}: {e}")
            skipped += 1
            continue
        # The capture as the scanner kept it before reading, at origin 0,0 so
        # the locate cursor is in image coordinates
        engine._last_capture, engine._last_origin = rgb, Point(0, 0)
        engine._last_card = None
        start = time.perf_counter()
        cursor, cursor_source = _cursor_for(path, rgb)
        crop = engine._locate_card(cursor)
        card = engine._last_card
        card_text = "-"
        if card is not None:
            card_text = f"{card.left},{card.top} {card.width}x{card.height}"
        read = engine._read_title(crop or rgb, card_found=crop is not None)
        if card is not None:
            read_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            engine.layout_from_last_capture()
            layout_times.append(time.perf_counter() - start)
        vote = engine.last_tier_vote
        with_vote = db.lookup(read, tier_vote=vote) if read else None

        vote_text = f"{vote.tier} {vote.confidence:.2f}" if vote and vote.tier else "-"
        resolved = with_vote.name if with_vote else "-"
        print(
            f"{path.name:<{width}}  {cursor_source:<7}  {card_text:<21}  {read or '-':<24}  "
            f"{vote_text:<9}  {resolved}"
        )

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
    print(f"engine: {engine.reader.name}")
    print(f"captures with a card:   {len(read_times)} of {len(files) - skipped}")
    print(f"skipped unreadable:     {skipped}")
    print(f"median full read:       {_median_ms(read_times)} per card")
    print(f"median layout:          {_median_ms(layout_times)} per card")
    return 0


if __name__ == "__main__":
    sys.exit(main())
