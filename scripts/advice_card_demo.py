# ruff: noqa: INP001
"""Render the advice card for five real catalog items to PNGs.

Usage: uv run python scripts/advice_card_demo.py OUTPUT_DIR [SCALE]
       uv run --extra icons python scripts/advice_card_demo.py --ledger-demo --out DIR

Writes advice_<item>.png (owned unknown) and advice_<item>_owned2.png
(owned = 2) for Snitch Scanner, Bronze Statuette, Microscope,
ARC Powercell and Anvil I (a weapon), using the configured database.

--ledger-demo replays Jeffrey's 67 stash grid frames through a StashLedger on
a temporary user database (never arlo_user.db) and renders the card of
Durable Cloth, Metal Parts and Anvil I three ways: no ledger, the map after
the first 10 frames and the map after all 67. It writes real-data.txt and
cards/ into DIR and never starts ARLO.
"""

import argparse
import datetime
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from arc_helper.advice_card import build_advice_card
from arc_helper.advice_card import needed_parts
from arc_helper.advice_card import render_advice_card
from arc_helper.database import HOME
from arc_helper.database import Database
from arc_helper.verdict import KEEP
from arc_helper.verdict import KEEP_SOME
from arc_helper.verdict import compute_verdict

CAPTURES = Path("/home/jefflamb/workspace/ARLO/debug/captures")
LEDGER_ITEMS = ("Durable Cloth", "Metal Parts", "Anvil I")
ARLO_USER_DB = Path("/home/jefflamb/workspace/ARLO/arlo_user.db")
ITEMS = ("Snitch Scanner", "Bronze Statuette", "Microscope", "ARC Powercell", "Anvil I")


class _Clock:
    now = 1000.0

    def __call__(self):
        return self.now


class _Surface:
    def start(self):  # noqa: PLR6301
        return True

    def draw(self, badges):
        pass

    def translate(self, *a, **k):
        pass

    def clear(self):
        pass

    def stop(self):
        pass


class _Snapshot:
    """A ledger stand-in that returns one stored Owned snapshot."""

    def __init__(self, owned):
        self._owned = owned

    def owned(self):
        return self._owned


def _replay(db, snapshots_at):
    """Feed the grid frames to a StashLedger, return {frames: Owned snapshot}."""
    from arc_helper.config import get_settings
    from arc_helper.main import StashBadges
    from arc_helper.main import StashLedger
    from arc_helper.main import stash_badge_maker
    from arc_helper.main import stash_identifier

    frames = sorted(CAPTURES.glob("20261015-2245*_grid.png")) + sorted(
        CAPTURES.glob("20261015-2246*_grid.png")
    )
    settings = get_settings()
    clock = _Clock()
    ledger = StashLedger(db, clock=clock)
    maker = stash_badge_maker(db, lambda: settings.show_ended_projects, ledger.owned)
    stash = StashBadges(
        stash_identifier(db, settings.icon_cache_dir),
        maker,
        _Surface(),
        gap_s=0,
        threaded=False,
        clock=clock,
        ledger=ledger,
    )
    snaps = {}
    for i, path in enumerate(frames, 1):
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        for _ in range(3):
            clock.now += 0.5
            stash.feed(rgb, None, ())
            stash.step()
        if i in snapshots_at:
            snaps[i] = ledger.owned()
    snaps[len(frames)] = ledger.owned()
    return snaps, len(frames)


def ledger_demo(out: Path) -> None:
    from arc_helper.main import Scanner

    temp = Path(tempfile.mkdtemp(prefix="owned_card_"))
    user_db = temp / "arlo_user.db"
    assert user_db != ARLO_USER_DB
    db = Database(user_db_path=user_db)
    snaps, total = _replay(db, {10})
    cards = out / "cards"
    cards.mkdir(parents=True, exist_ok=True)
    lines = [f"temporary user database {user_db}", f"frames replayed: {total}"]
    ways = (
        ("no ledger", None),
        ("partial map (10 frames)", snaps[10]),
        (f"full map ({total} frames)", snaps[total]),
    )
    for label, owned in ways:
        lines.append(
            f"{label}: owned given={owned is not None}"
            + (f" sure={owned.sure} items={len(owned.counts)}" if owned else "")
        )
    timings = {"before": [], "after": []}
    for name in LEDGER_ITEMS:
        item = db.lookup(name)
        lines.append(f"== {name} ({item.category if item else 'not found'})")
        for label, owned in ways:
            scanner = Scanner(root=None, overlay=None, status=None, db=db, hotkey=None)
            scanner.ledger = _Snapshot(owned) if owned is not None else None
            hovered = SimpleNamespace(stack=None)
            count = scanner._owned_for(item, hovered)  # noqa: SLF001
            verdict = scanner._verdict_for(  # noqa: SLF001
                item,
                False,  # noqa: FBT003
                None,
                count.count if count else None,
            )
            card = scanner._advice_card_for(item, verdict, count)  # noqa: SLF001
            for key, arg in (("before", None), ("after", count)):
                start = time.perf_counter()
                for _ in range(50):
                    scanner._advice_card_for(item, verdict, arg)  # noqa: SLF001
                timings[key].append((time.perf_counter() - start) / 50 * 1000)
            shown = "none"
            if card.owned is not None:
                from arc_helper.advice_card import owned_text

                shown = owned_text(card.owned, card.owned_sure)
            line = f"  {label}: verdict={verdict.kind} title={card.title!r} owned line={shown}"
            if card.rows and card.owned is not None:
                total_n = card.needed_total or sum(q for _, _, q in card.rows)
                line += f" of {total_n}"
            lines.append(line)
            path = (
                cards
                / f"{name.replace(' ', '_').lower()}_{label.split(' (')[0].replace(' ', '_')}.png"
            )
            render_advice_card(card, 1.0).save(path)
            lines.append(f"    {path.name}")
    for key in ("before", "after"):
        values = timings[key]
        lines.append(
            f"_advice_card_for {key} (owned {'None' if key == 'before' else 'given'}): "
            f"mean {sum(values) / len(values):.3f} ms, max {max(values):.3f} ms per call over {len(values)} runs of 50"
        )
    (out / "real-data.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", nargs="?", type=Path)
    parser.add_argument("scale", nargs="?", type=float, default=1.0)
    parser.add_argument("--ledger-demo", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.ledger_demo:
        ledger_demo(args.out or args.output_dir)
        return
    out = args.output_dir
    scale = args.scale
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
