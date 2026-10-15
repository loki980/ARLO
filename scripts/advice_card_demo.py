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
from arc_helper.advice_card import needed_parts
from arc_helper.advice_card import render_advice_card
from arc_helper.database import HOME
from arc_helper.database import Database
from arc_helper.verdict import KEEP
from arc_helper.verdict import KEEP_SOME
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
        outputs = db.recycle_outputs(name)
        home = [o for o in outputs if o.location == HOME]

        def is_needed(component, needs_c):
            v = compute_verdict(
                db.lookup(component),
                needs_c,
                db.progress_map(),
                None,
                today,
                False,  # noqa: FBT003
                blueprint=db.blueprint_for(component),
            )
            return v.kind in {KEEP, KEEP_SOME}

        needed = needed_parts(home, db.needs_for, is_needed)
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
            card = build_advice_card(verdict, needs, owned, outputs, needed)
            image = render_advice_card(card, scale)
            path = out / f"advice_{name.replace(' ', '_').lower()}{suffix}.png"
            image.save(path)
            print(path, card.kind, card.title, card.subtitle, image.size)


if __name__ == "__main__":
    main()
