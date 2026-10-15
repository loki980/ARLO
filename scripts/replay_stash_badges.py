# ruff: noqa: INP001
"""Replay saved captures through the live stash badge hand-off and pass.

    uv run --extra icons python scripts/replay_stash_badges.py <captures> <out folder>

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
"""

import logging
import sys
import time
from pathlib import Path

from PIL import Image

from arc_helper.badge_surface import render_png
from arc_helper.config import get_settings
from arc_helper.config import logger
from arc_helper.database import Database
from arc_helper.main import StashBadges
from arc_helper.main import stash_badge_maker
from arc_helper.main import stash_identifier

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


def main() -> int:
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
