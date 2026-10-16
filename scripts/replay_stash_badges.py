# ruff: noqa: INP001
"""Replay saved captures through the live stash badge hand-off and pass.

    uv run --extra icons python scripts/replay_stash_badges.py <captures> <out folder>
    uv run --extra icons python scripts/replay_stash_badges.py --keep <capture> <out file>

The 3840x2160 captures are fed in name order to main.StashBadges, the same
object the scan thread feeds, built with the same identifier and badge
maker as the app (stash_identifier, stash_badge_maker), but run inline
(threaded False) with a surface that records draws instead of putting a
window on screen. The saved captures have no cursor, so no item card rect
is passed. The run stops at the first draw, after the first still pair.

Writes to the out folder: stash_badges_replay.png (that pair's newer
capture with the drawn badges laid over it, rendered by the badge helper's
own cairo code through badge_surface.render_png) and stash_badges_replay.txt
(the frames fed, the pass log lines and the badges drawn).

With --keep one capture is fed as a sequence of edited copies, on a clock
the script moves: the frame twice, a card over 6 backpack slots then those
slots darkened at alpha 0.5 with no card (a fading card) inside the card
linger, the frame again, the darkened frame with the linger lapsed, the
frame again, 3 backpack slots' interiors brightened (a hover) and the frame
shifted up by one slot row (a scroll). The out file gets, for each step,
how many badges are drawn, the badge rects drawn before and hidden now and the stash
log lines.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from arc_helper.badge_surface import render_png
from arc_helper.config import get_settings
from arc_helper.config import logger
from arc_helper.database import Database
from arc_helper.grid import find_slots
from arc_helper.main import CARD_LINGER_S
from arc_helper.main import StashBadges
from arc_helper.main import stash_badge_maker
from arc_helper.main import stash_identifier
from arc_helper.tooltip_layout import Rect

FULL_SIZE = (3840, 2160)
PNG_NAME = "stash_badges_replay.png"
LOG_NAME = "stash_badges_replay.txt"


class RecordingSurface:
    """A badge surface that keeps the last draw instead of showing it."""

    def __init__(self) -> None:
        self.drawn: list | None = None
        self.started = False

    def start(self) -> bool:
        self.started = True
        return True

    def draw(self, badges: list) -> None:
        self.drawn = list(badges)

    def clear(self) -> None:
        self.drawn = None

    def stop(self) -> None:
        pass


class _Lines(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage().startswith("stash"):
            self.lines.append(record.getMessage())


class _Clock:
    """The time the replay says it is."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _bounds(rects: list[Rect]) -> Rect:
    return Rect(
        min(r.left for r in rects),
        min(r.top for r in rects),
        max(r.right for r in rects),
        max(r.bottom for r in rects),
    )


def _keep_frames(rgb: np.ndarray) -> tuple[list, list[Rect]]:
    """The --keep steps as (label, frame, card, seconds since the last step)."""
    backpack = [s.rect for s in find_slots(rgb) if s.panel == "backpack"]
    faded, hovered = backpack[:3] + backpack[4:7], backpack[12:15]
    card = _bounds(faded)
    dark = rgb.copy()
    area = dark[card.top : card.bottom, card.left : card.right]
    area[:] = (area * 0.5).astype(np.uint8)
    hover = rgb.copy()
    inset = 16
    for r in hovered:
        part = hover[r.top + inset : r.bottom - inset, r.left + inset : r.right - inset]
        part[:] = (part * 0.75 + 255 * 0.25).astype(np.uint8)
    pitch = backpack[4].top - backpack[0].top
    scrolled = np.vstack([rgb[pitch:], rgb[-pitch:]])
    steps = [
        ("a frame", rgb, None, 0.1),
        ("a frame again", rgb, None, 0.1),
        ("b card over 6 backpack slots", rgb, card, 0.1),
        ("b fading card, 6 slots darkened, no card, linger active", dark, None, 0.1),
        ("c frame", rgb, None, 0.1),
        ("b 6 slots darkened, no card, linger lapsed", dark, None, CARD_LINGER_S + 1),
        ("c frame", rgb, None, 0.1),
        ("d 3 backpack slots brightened (hover)", hover, None, 0.1),
        ("c frame", rgb, None, 0.1),
        ("e frame shifted up one slot row (scroll)", scrolled, None, 0.1),
    ]
    return steps, [*faded, *hovered]


def keep_main(capture: Path, out: Path) -> int:
    """Replay the --keep sequence and write what each step keeps, hides and clears."""
    lines = _Lines()
    logger.addHandler(lines)
    logger.setLevel(logging.DEBUG)
    settings = get_settings()
    db = Database()
    surface = RecordingSurface()
    clock = _Clock()
    stash = StashBadges(
        stash_identifier(db, settings.icon_cache_dir),
        stash_badge_maker(db, lambda: settings.show_ended_projects),
        surface,
        keep_border=settings.overlay.stash_badge_keep_border,
        gap_s=0,
        threaded=False,
        clock=clock,
    )
    with Image.open(capture) as image:
        rgb = np.asarray(image.convert("RGB"))
    steps, named = _keep_frames(rgb)
    report = [f"capture: {capture.name}", f"slots edited: {named}", ""]
    every: set = set()
    for label, frame, card, wait in steps:
        clock.now += wait
        start = len(lines.lines)
        stash.feed(frame, card, ())
        stash.step()
        drawn = surface.drawn or []
        slots = {tuple(b.rect) for b in drawn}
        every |= slots
        report.append(
            f"{label}: {len(drawn)} badges drawn ({len(slots)} distinct rects)"
        )
        if drawn and every - slots:
            report.append(
                f"  badge rects drawn before, not now: {sorted(every - slots)}"
            )
        report += [f"  log: {line}" for line in lines.lines[start:]]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return 0


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--keep":
        return keep_main(Path(sys.argv[2]), Path(sys.argv[3]))
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    captures, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    lines = _Lines()
    logger.addHandler(lines)
    logger.setLevel(logging.DEBUG)

    settings = get_settings()
    db = Database()
    surface = RecordingSurface()
    stash = StashBadges(
        stash_identifier(db, settings.icon_cache_dir),
        stash_badge_maker(db, lambda: settings.show_ended_projects),
        surface,
        keep_border=settings.overlay.stash_badge_keep_border,
        gap_s=0,
        threaded=False,
    )
    fed: list[str] = []
    shown = None
    start = time.perf_counter()
    for path in sorted(captures.glob("*.png")):
        with Image.open(path) as image:
            if image.size != FULL_SIZE:
                continue
            rgb = image.convert("RGB")
        fed.append(path.name)
        stash.feed(rgb, None, ())
        stash.step()
        if surface.drawn:
            shown = (path, rgb)
            break
    elapsed = time.perf_counter() - start

    report = [f"frames fed: {len(fed)} in {elapsed:.1f} s", *fed, "", *lines.lines]
    if shown is None:
        report.append("no badges drawn: no still pair with identified slots")
        (out / LOG_NAME).write_text("\n".join(report) + "\n", encoding="utf-8")
        print("\n".join(report[-3:]))
        return 1
    path, rgb = shown
    layer = out / "stash_badges_layer.png"
    if not render_png(surface.drawn, rgb.size, layer):
        report.append("badge PNG not rendered (see the warning above)")
        (out / LOG_NAME).write_text("\n".join(report) + "\n", encoding="utf-8")
        return 1
    with Image.open(layer) as badges:
        composite = Image.alpha_composite(rgb.convert("RGBA"), badges.convert("RGBA"))
    composite.convert("RGB").save(out / PNG_NAME)
    layer.unlink()
    report += ["", f"drawn on {path.name}: {len(surface.drawn)} badges"]
    report += [repr(b) for b in surface.drawn]
    (out / LOG_NAME).write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"drawn on {path.name}: {len(surface.drawn)} badges -> {out / PNG_NAME}")
    print("\n".join(lines.lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
