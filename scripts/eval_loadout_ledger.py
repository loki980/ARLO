# ruff: noqa: INP001
"""Tally the loadout panels on still full-screen frames and check the header.

    uv run --extra icons python scripts/eval_loadout_ledger.py <image>... --out <folder>

Each image goes through find_slots, identify_slots on the app's gallery
(main.stash_identifier), read_loadout_header and loadout_tally, then the
tally is saved to and read back from a temporary user database (a copy of
items.db in a temporary folder, so neither items.db nor arlo_user.db is
written). Prints, and writes to <folder>/loadout_ledger.txt, the per-panel
slot counts, every loadout slot with its guess and count, the per-panel item
counts, the computed value against the header, agrees and its reason. Also
writes <folder>/ledger_<image>.png, the frame at half size with each loadout
slot boxed and labelled with its name and count (red when unidentified), to
check by eye.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from arc_helper.config import get_settings
from arc_helper.database import Database
from arc_helper.grid import EQUIPMENT
from arc_helper.grid import STASH
from arc_helper.grid import find_slots
from arc_helper.inventory_header import read_loadout_header
from arc_helper.inventory_ledger import VALUE_TOLERANCE_FRACTION
from arc_helper.inventory_ledger import loadout_tally
from arc_helper.main import stash_identifier

SHEET_SCALE = 0.5
GREEN = (90, 220, 90)
RED = (240, 90, 90)


def _label(slot) -> str:
    if slot.item is None:
        return f"? {slot.guess}"
    count = slot.stack if slot.stack is not None else 1
    return f"{slot.item.name} x{count}"


def _sheet(rgb: np.ndarray, slot_items, path: Path) -> None:
    image = Image.fromarray(rgb)
    image = image.resize(
        (int(image.width * SHEET_SCALE), int(image.height * SHEET_SCALE))
    )
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=12)
    for slot in slot_items:
        if slot.panel == STASH:
            continue
        r = slot.rect
        box = [int(v * SHEET_SCALE) for v in (r.left, r.top, r.right, r.bottom)]
        colour = GREEN if slot.item else RED
        draw.rectangle(box, outline=colour, width=2)
        draw.text((box[0] + 2, box[1] + 2), _label(slot), fill=colour, font=font)
    image.save(path)


def _report(path: Path, rgb: np.ndarray, db: Database, slot_items) -> list[str]:
    equipment = [s for s in find_slots(rgb) if s.panel == EQUIPMENT]
    header = read_loadout_header(rgb, equipment)
    tally = loadout_tally(slot_items, header)
    db.save_loadout(tally)
    stored = db.loadout_counts()
    lines = [f"== {path.name} ({rgb.shape[1]}x{rgb.shape[0]})", f"header: {header}"]
    lines.append("slots per panel (identified / unidentified):")
    lines.extend(
        f"  {panel}: {tally.identified[panel]} / {tally.unidentified[panel]}"
        for panel in tally.identified
    )
    lines.append("loadout slots (panel, top, left: name or ? guess, count):")
    lines.extend(
        f"  {s.panel}, {s.rect.top}, {s.rect.left}: {_label(s)}"
        f" (stack read {s.stack}, sim {s.confidence}, price"
        f" {s.item.sell_price if s.item else '-'})"
        for s in slot_items
        if s.panel != STASH
    )
    lines.append("item counts per panel:")
    for panel, counts in tally.panel_counts.items():
        lines.append(f"  {panel}:")
        lines.extend(f"    {name}: {count}" for name, count in sorted(counts.items()))
    if tally.header_value:
        gap = tally.computed_value - tally.header_value
        versus = (
            f"header {tally.header_value:,}, gap {gap:+,}"
            f" ({gap / tally.header_value:+.1%})"
        )
    else:
        versus = "no header value"
    lines += [
        f"computed value {tally.computed_value:,} vs {versus}",
        f"tolerance {VALUE_TOLERANCE_FRACTION:.0%}",
        f"unpriced: {tally.unpriced}",
        f"agrees: {tally.agrees} ({tally.reason})",
        f"stored totals match tally: {stored == tally.totals}",
        "",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    with tempfile.TemporaryDirectory() as tmp:
        db_copy = Path(tmp) / "items.db"
        shutil.copy(settings.database_path, db_copy)
        db = Database(db_path=db_copy, user_db_path=Path(tmp) / "arlo_user.db")
        identify = stash_identifier(db, settings.icon_cache_dir)
        lines = []
        for path in args.images:
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"))
            slot_items = identify(rgb)
            lines += _report(path, rgb, db, slot_items)
            _sheet(rgb, slot_items, args.out / f"ledger_{path.stem}.png")
    text = "\n".join(lines)
    print(text)
    (args.out / "loadout_ledger.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
