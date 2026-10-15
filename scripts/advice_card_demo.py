# ruff: noqa: INP001
"""Render the advice card for four real catalog items to PNGs.

Usage: uv run python scripts/advice_card_demo.py OUTPUT_DIR [SCALE]

Writes advice_<item>.png (owned unknown) and advice_<item>_owned2.png
(owned = 2) for Snitch Scanner, Bronze Statuette, Microscope and
ARC Powercell, using the configured database.
"""

import datetime
import sys
from pathlib import Path

from arc_helper.advice_card import build_advice_card
from arc_helper.advice_card import render_advice_card
from arc_helper.database import HOME
from arc_helper.database import Database
from arc_helper.verdict import compute_verdict

ITEMS = ("Snitch Scanner", "Bronze Statuette", "Microscope", "ARC Powercell")


def main() -> None:
    out = Path(sys.argv[1])
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    out.mkdir(parents=True, exist_ok=True)
    db = Database()
    today = datetime.date.today()  # noqa: DTZ011
    for name in ITEMS:
        item = db.lookup(name)
        needs = db.needs_for(name)
        home = [o for o in db.recycle_outputs(name) if o.location == HOME]
        for owned, suffix in ((None, ""), (2, "_owned2")):
            verdict = compute_verdict(
                item,
                needs,
                db.progress_map(),
                owned,
                today,
                False,  # noqa: FBT003
                blueprint=db.blueprint_for(name),
                home_outputs=home,
            )
            card = build_advice_card(verdict, needs, owned)
            image = render_advice_card(card, scale)
            path = out / f"advice_{name.replace(' ', '_').lower()}{suffix}.png"
            image.save(path)
            print(path, card.kind, card.title, image.size)


if __name__ == "__main__":
    main()
