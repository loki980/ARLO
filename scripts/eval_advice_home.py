# ruff: noqa: INP001
"""Check the advice card's home beneath the second gun on real screenshots.

Usage: uv run python scripts/eval_advice_home.py IMAGE... [--out DIR]
       [--card LEFT,TOP,RIGHT,BOTTOM]

For each full-screen capture, prints the weapon slots find_slots found, the
home box beneath the second gun and the rect of a real advice card rendered
into it, checks that the card lies inside the gun's width and overlaps no
slot, and draws the card's rect on a copy named home_<first word>.png
(base-stiletto-4k.png gives home_base.png). DIR defaults to attempt-1 beside
the first image when it exists, else the current directory.

On base-stiletto-4k.png, or any image given --card, it also reports where
the card goes while the game's item card (default the visible Stiletto IV
card, x 2024 to 2784, y 350 to 1700) is up: home when the item card leaves
home free, else the escape spot.
"""

import argparse
import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageDraw

from arc_helper.advice_card import build_advice_card
from arc_helper.advice_card import needed_parts
from arc_helper.advice_window import render_in_box
from arc_helper.database import HOME
from arc_helper.database import Database
from arc_helper.grid import find_slots
from arc_helper.layout_home import fixed_home
from arc_helper.layout_home import home_below_weapons
from arc_helper.layout_home import weapon_slots
from arc_helper.main import POPUP_CORNER
from arc_helper.main import POPUP_HOME
from arc_helper.main import home_is_free
from arc_helper.main import popup_escape_spot
from arc_helper.main import rects_intersect
from arc_helper.tooltip_layout import Rect
from arc_helper.verdict import KEEP
from arc_helper.verdict import KEEP_SOME
from arc_helper.verdict import compute_verdict

# Real catalog items, from short to tall cards. The tallest decides
ITEMS = (
    "Stiletto IV",
    "Snitch Scanner",
    "Bronze Statuette",
    "Microscope",
    "ARC Powercell",
    "Anvil I",
)
# The Stiletto IV card visible on base-stiletto-4k.png
STILETTO_CARD = Rect(2024, 350, 2784, 1700)
# OVERLAY_X and OVERLAY_Y 57 tk at 1.75 scale, the corner spot
CORNER = (100, 100)


def _cards(db: Database):
    """Advice cards for ITEMS found in the database, owned unknown."""
    today = datetime.date.today()  # noqa: DTZ011

    def is_needed(component, needs_c):
        verdict = compute_verdict(
            db.lookup(component),
            needs_c,
            db.progress_map(),
            None,
            today,
            False,  # noqa: FBT003
            blueprint=db.blueprint_for(component),
        )
        return verdict.kind in {KEEP, KEEP_SOME}

    cards = []
    for name in ITEMS:
        item = db.lookup(name)
        if item is None:
            continue
        needs = db.needs_for(item.name)
        outputs = db.recycle_outputs(item.name)
        home_outputs = [o for o in outputs if o.location == HOME]
        needed = needed_parts(home_outputs, db.needs_for, is_needed)
        verdict = compute_verdict(
            item,
            needs,
            db.progress_map(),
            None,
            today,
            False,  # noqa: FBT003
            blueprint=db.blueprint_for(item.name),
            home_outputs=home_outputs,
        )
        cards.append(
            (item.name, build_advice_card(verdict, needs, None, outputs, needed))
        )
    return cards


def _out_dir(first: Path, out: str | None) -> Path:
    if out:
        return Path(out)
    attempt = first.parent / "attempt-1"
    return attempt if attempt.is_dir() else Path()


def _check(path: Path, cards, out_dir: Path, item_card: Rect | None) -> None:
    with Image.open(path) as img:
        frame = img.convert("RGB")
    screen = frame.size
    slots = find_slots(np.asarray(frame))
    guns = weapon_slots(slots)
    print(f"== {path.name} {screen[0]}x{screen[1]}, {len(slots)} slots")
    for gun in guns:
        print(f"weapon slot {tuple(gun)} width {gun.right - gun.left}")
    home = home_below_weapons(guns, screen)
    if home is None:
        home = fixed_home(screen)
        print(f"no two weapon slots: fixed home {tuple(home)}")
    else:
        print(f"home box {tuple(home)} (beneath the second gun)")

    tallest = None
    for name, card in cards:
        image = render_in_box(card, home)
        rect = Rect(
            home.left, home.top, home.left + image.width, home.top + image.height
        )
        print(f"card {name!r}: {image.width}x{image.height} at {tuple(rect)}")
        if tallest is None or rect.bottom > tallest[1].bottom:
            tallest = (name, rect, image)
    name, rect, image = tallest
    lower = max(guns, key=lambda r: r.bottom) if len(guns) == 2 else None
    overlaps = [s.rect for s in slots if rects_intersect(s.rect, rect)]
    print(f"tallest card {name!r} rect {tuple(rect)}")
    if lower is not None:
        print(
            f"  below the second gun: {rect.top >= lower.bottom} "
            f"(gap {rect.top - lower.bottom} px), inside its width: "
            f"{lower.left <= rect.left and rect.right <= lower.right}"
        )
    print(
        f"  bottom {rect.bottom} at or above the floor {home.bottom}: "
        f"{rect.bottom <= home.bottom}"
    )
    print(f"  overlaps slots: {[tuple(r) for r in overlaps] or 'none'}")

    copy = frame.copy()
    copy.paste(image.convert("RGB"), (rect.left, rect.top), image)
    draw = ImageDraw.Draw(copy)
    for gun in guns:
        draw.rectangle(tuple(gun), outline=(0, 160, 255), width=4)
    draw.rectangle(
        (rect.left, rect.top, rect.right - 1, rect.bottom - 1),
        outline=(255, 40, 40),
        width=6,
    )

    if item_card is not None:
        size = (rect.right - rect.left, rect.bottom - rect.top)
        labelled = {
            POPUP_HOME: rect,
            POPUP_CORNER: Rect(*CORNER, CORNER[0] + size[0], CORNER[1] + size[1]),
        }
        print(f"item card {tuple(item_card)}")
        if home_is_free(item_card, labelled):
            print("  advice card stays home: the item card leaves home free")
        spot = popup_escape_spot(item_card, labelled, screen)
        if spot is None:
            print("  no free spot: the advice card stays where it is")
        else:
            label, spot_rect = spot
            print(f"  card goes to {label} at {tuple(spot_rect)}")
        draw.rectangle(tuple(item_card), outline=(255, 220, 0), width=4)

    out = out_dir / f"home_{path.stem.split('-')[0]}.png"
    copy.save(out)
    print(f"drawn on {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--out")
    parser.add_argument("--card", help="item card rect LEFT,TOP,RIGHT,BOTTOM")
    args = parser.parse_args()
    out_dir = _out_dir(args.images[0], args.out)
    cards = _cards(Database())
    for path in args.images:
        if args.card:
            item_card = Rect(*(int(v) for v in args.card.split(",")))
        elif path.name.startswith("base-stiletto"):
            item_card = STILETTO_CARD
        else:
            item_card = None
        _check(path, cards, out_dir, item_card)


if __name__ == "__main__":
    main()
