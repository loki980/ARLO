# ruff: noqa: INP001
"""Replay saved captures through the live stash badge hand-off and pass.

    uv run --extra icons python scripts/replay_stash_badges.py <captures> <out folder>
    uv run --extra icons python scripts/replay_stash_badges.py --keep <capture> <out file>
    uv run --extra icons python scripts/replay_stash_badges.py --latency --out <folder> <capture>
    uv run --extra icons python scripts/replay_stash_badges.py --latency --scroll-steps N --out <folder> <capture>

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

With --latency one capture is fed as a scroll on a clock the script moves,
frames LATENCY_FRAME_S apart: the slot cache is warmed by identifying the
capture first, then the capture twice, the capture shifted up by one slot
row (the scroll) and the shifted frame LATENCY_STILL_FRAMES times. Each
identify, peek and badge-making call moves the clock on by the time it
really took, so a pass costs its real time. The app's default pass gap is
used. The script measures, on that clock, the time from the scroll frame to
the first still frame (moved_to_still), the pass time (pass), from the
first still frame to the first draw after the scroll (still_to_drawn) and
in all (total), and counts the badges drawn before the pass finished.
real-data.txt in the out folder gets those, every draw, the stash log lines
and, when latency-before.txt is in the folder (a run on the code before),
that file first.

With --latency --scroll-steps N the backpack viewport of one capture is
scrolled up by one slot row in N equal steps, frames SCROLL_FRAME_S apart,
on the same clock and warm cache, then held still. The rows leaving the top
come back in at the bottom, as new rows would. It runs twice, once with the
scroll follow turned off (StashBadges as before, every scroll a clear) and
once as the app runs, and real-data.txt in the out folder gets, per frame,
the offset StashBadges measured against the true one, the badges on screen
and how far the ones in the viewport sit from their items, then the time
from the first and the last scroll frame to the final draw.
"""

import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from arc_helper.badge_helper import translate_badges
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
LATENCY_FRAME_S = 0.15
LATENCY_STILL_FRAMES = 10
SCROLL_FRAME_S = 0.1
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


class _TimedSurface(RecordingSurface):
    """A recording surface that also logs each draw's clock time and size."""

    def __init__(self, clock: _Clock, passes: list) -> None:
        super().__init__()
        self.clock = clock
        self.passes = passes
        self.draws: list[tuple[float, int, int]] = []

    def draw(self, badges: list) -> None:
        super().draw(badges)
        self.draws.append((self.clock.now, len(badges), len(self.passes)))


def _timed(clock: _Clock, call, record: list | None = None):
    """call, moving the clock on by the time each call really takes."""

    def run(*args):
        start = time.perf_counter()
        try:
            return call(*args)
        finally:
            spent = time.perf_counter() - start
            clock.now += spent
            if record is not None:
                record.append((clock.now, spent))

    return run


def _shifted(rgb: np.ndarray) -> np.ndarray:
    """The frame scrolled up by one backpack slot row, the last row repeated."""
    backpack = [s.rect for s in find_slots(rgb) if s.panel == "backpack"]
    pitch = backpack[4].top - backpack[0].top
    return np.vstack([rgb[pitch:], rgb[-pitch:]])


def latency_main(capture: Path, out: Path) -> int:
    """Replay a scroll on a warm cache and write how long the badges take back."""
    lines = _Lines()
    logger.addHandler(lines)
    logger.setLevel(logging.DEBUG)
    settings = get_settings()
    db = Database()
    clock = _Clock()
    identify = stash_identifier(db, settings.icon_cache_dir)
    with Image.open(capture) as image:
        rgb = np.asarray(image.convert("RGB"))
    warm_start = time.perf_counter()
    identify(rgb)
    identify(rgb)
    warm_ms = (time.perf_counter() - warm_start) * 1000
    scrolled = _shifted(rgb)
    passes: list = []
    peeks: list = []
    timed = _timed(clock, identify, passes)
    if hasattr(identify, "peek"):
        timed.peek = _timed(clock, identify.peek, peeks)
    make = stash_badge_maker(db, lambda: settings.show_ended_projects)
    makes: list = []
    surface = _TimedSurface(clock, passes)
    stash = StashBadges(
        timed,
        _timed(clock, make, makes),
        surface,
        keep_border=settings.overlay.stash_badge_keep_border,
        threaded=False,
        clock=clock,
    )
    frames = [("capture", rgb), ("capture", rgb), ("scroll", scrolled)]
    frames += [("still", scrolled)] * LATENCY_STILL_FRAMES
    start = clock.now
    fed: list[tuple[str, float]] = []
    report = [f"capture: {capture.name}", f"cache warmed in {warm_ms:.0f} ms", ""]
    for n, (label, frame) in enumerate(frames):
        clock.now = max(clock.now, start + n * LATENCY_FRAME_S)
        fed.append((label, clock.now))
        first = len(lines.lines)
        stash.feed(frame, None, ())
        stash.step()
        report.append(f"t={(fed[-1][1] - start) * 1000:6.0f} ms fed {label}")
        report += [f"  log: {line}" for line in lines.lines[first:]]
    moved = next(t for label, t in fed if label == "scroll")
    still = next(t for label, t in fed if label == "still")
    after = [d for d in surface.draws if d[0] >= still]
    scroll_pass = [p for p in passes if p[0] >= moved]
    report.append("")
    report += [
        f"draw at t={(t - start) * 1000:.0f} ms: {count} badges, passes so far {done}"
        for t, count, done in surface.draws
    ]
    if not after or not scroll_pass:
        report.append("no draw or no pass after the scroll")
    else:
        drawn, _count, _done = after[0]
        identified, spent = scroll_pass[0]
        spent += next((s for t, s in makes if t > identified), 0.0)
        before_pass = [d for d in after if d[2] == passes.index(scroll_pass[0])]
        report += [
            "",
            "measured by the script:",
            (
                f"stash latency moved_to_still={(still - moved) * 1000:.0f} "
                f"pass={spent * 1000:.0f} "
                f"still_to_drawn={(drawn - still) * 1000:.0f} "
                f"total={(drawn - moved) * 1000:.0f} ms"
            ),
            (
                "known-slot badges drawn before the pass: "
                f"{before_pass[0][1] if before_pass else 0}"
            ),
            f"peek calls: {len(peeks)}, "
            + ", ".join(f"{p * 1000:.0f} ms" for _t, p in peeks),
        ]
    report += ["", "the app's own latency line:"]
    report += [line for line in lines.lines if line.startswith("stash latency")] or [
        "none"
    ]
    out.mkdir(parents=True, exist_ok=True)
    before = out / "latency-before.txt"
    text = "\n".join(report) + "\n"
    if before.exists():
        text = (
            "=== before (latency-before.txt) ===\n"
            + before.read_text(encoding="utf-8")
            + "\n=== after ===\n"
            + text
        )
    (out / "real-data.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


class _ScrollSurface(_TimedSurface):
    """A timed surface that applies translates as the badge helper does."""

    def __init__(self, clock: _Clock, passes: list) -> None:
        super().__init__(clock, passes)
        self.shown: list[dict] = []
        self.translates: list[tuple[float, int]] = []

    def draw(self, badges: list) -> None:
        super().draw(badges)
        self.shown = [{**asdict(b), "rect": list(b.rect)} for b in badges]

    def clear(self) -> None:
        super().clear()
        self.shown = []

    def translate(self, region, dy: int) -> None:
        self.translates.append((self.clock.now, dy))
        self.shown = translate_badges(self.shown, region, dy)


def _scroll_frames(rgb: np.ndarray, steps: int) -> tuple[list, Rect, int]:
    """The capture with its backpack viewport rolled up by k / steps of a row."""
    backpack = [s.rect for s in find_slots(rgb) if s.panel == "backpack"]
    slots = _bounds(backpack)
    pitch = backpack[4].top - backpack[0].top
    # Whole rows of pitch, so a row rolled in lands on the slot grid
    rows = -(-slots.height // pitch)
    band = Rect(slots.left, slots.top, slots.right, slots.top + rows * pitch)
    frames = []
    for k in range(1, steps + 1):
        moved = round(k * pitch / steps)
        frame = rgb.copy()
        view = rgb[band.top : band.bottom, band.left : band.right]
        frame[band.top : band.bottom, band.left : band.right] = np.roll(
            view, -moved, axis=0
        )
        frames.append((frame, moved))
    return frames, band, pitch


def _badge_error(shown: list[dict], first: list[dict], band: Rect, moved: int):
    """The badges inside band and the most px one sits from its item."""
    inside = [
        b
        for b in shown
        if band.left <= b["rect"][0]
        and b["rect"][2] <= band.right
        and band.top <= b["rect"][1]
        and b["rect"][3] <= band.bottom
    ]
    worst = 0
    for b in inside:
        left, top = b["rect"][0], b["rect"][1]
        tops = [
            f["rect"][1] - moved + wrap
            for f in first
            if f["kind"] == b["kind"] and f["rect"][0] == left
            for wrap in (0, band.bottom - band.top)
        ]
        if tops:
            worst = max(worst, min(abs(top - t) for t in tops))
    return len(inside), worst


def _scroll_run(capture: Path, steps: int, *, follow: bool) -> list[str]:
    """One scroll replay, with the scroll follow on or off."""
    lines = _Lines()
    logger.addHandler(lines)
    settings = get_settings()
    db = Database()
    clock = _Clock()
    identify = stash_identifier(db, settings.icon_cache_dir)
    with Image.open(capture) as image:
        rgb = np.asarray(image.convert("RGB"))
    identify(rgb)
    scrolled, band, pitch = _scroll_frames(rgb, steps)
    passes: list = []
    timed = _timed(clock, identify, passes)
    if hasattr(identify, "peek"):
        timed.peek = _timed(clock, identify.peek)
    make = stash_badge_maker(db, lambda: settings.show_ended_projects)
    surface = _ScrollSurface(clock, passes)
    stash = StashBadges(
        timed,
        _timed(clock, make),
        surface,
        keep_border=settings.overlay.stash_badge_keep_border,
        threaded=False,
        clock=clock,
    )
    if not follow:
        stash._scroll_of = lambda *_args: None  # noqa: SLF001 - the code before
    frames = [("capture", rgb, 0), ("capture", rgb, 0)]
    frames += [(f"scroll {k + 1}/{steps}", f, m) for k, (f, m) in enumerate(scrolled)]
    frames += [("still", scrolled[-1][0], scrolled[-1][1])] * LATENCY_STILL_FRAMES
    report = [
        (
            f"scroll follow {'on' if follow else 'off'}: backpack viewport "
            f"{tuple(band)}, row pitch {pitch} px, {steps} steps, frames "
            f"{SCROLL_FRAME_S * 1000:.0f} ms apart"
        ),
    ]
    start = clock.now
    first: list[dict] = []
    fed: list[tuple[str, float]] = []
    last_moved = 0
    for n, (label, frame, moved) in enumerate(frames):
        clock.now = max(clock.now, start + n * SCROLL_FRAME_S)
        fed.append((label, clock.now))
        mark = len(lines.lines)
        began = time.perf_counter()
        stash.feed(frame, None, ())
        stash.step()
        spent = (time.perf_counter() - began) * 1000
        if label == "capture" and surface.shown:
            first = [dict(b) for b in surface.shown]
        true_dy = -(moved - last_moved) if label.startswith("scroll") else 0
        last_moved = moved
        measured = [
            line.split("dy=")[1].split()[0]
            for line in lines.lines[mark:]
            if "followed" in line
        ]
        inside, worst = _badge_error(surface.shown, first, band, moved)
        report.append(
            f"t={(fed[-1][1] - start) * 1000:5.0f} ms {label:<11} "
            f"true dy {true_dy:4d} measured {measured[0] if measured else '-':>4} "
            f"on screen {len(surface.shown):3d} (in viewport {inside:2d}, "
            f"worst {worst} px off), round {spent:.0f} ms wall"
        )
        report += [f"    log: {line}" for line in lines.lines[mark:]]
    logger.removeHandler(lines)
    scroll_at = [t for label, t in fed if label.startswith("scroll")]
    final = max(
        [t for t, _c, _p in surface.draws] + [t for t, _d in surface.translates]
    )
    report += [
        "",
        f"draws: {len(surface.draws)}, translates: {len(surface.translates)}",
        (
            f"final draw at t={(final - start) * 1000:.0f} ms: "
            f"{(final - scroll_at[0]) * 1000:.0f} ms after the first scroll frame, "
            f"{(final - scroll_at[-1]) * 1000:.0f} ms after the last"
        ),
    ]
    report += [line for line in lines.lines if line.startswith("stash latency")]
    return report


def scroll_main(capture: Path, out: Path, steps: int) -> int:
    """Replay a scroll in steps with the follow off and on, into real-data.txt."""
    logger.setLevel(logging.DEBUG)
    report = [f"capture: {capture.name}", "", "=== before (scroll follow off) ==="]
    report += _scroll_run(capture, steps, follow=False)
    report += ["", "=== after (scroll follow on) ==="]
    report += _scroll_run(capture, steps, follow=True)
    out.mkdir(parents=True, exist_ok=True)
    text = "\n".join(report) + "\n"
    (out / "real-data.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


def main() -> int:
    if (
        len(sys.argv) == 7
        and sys.argv[1] == "--latency"
        and sys.argv[2] == "--scroll-steps"
        and sys.argv[4] == "--out"
    ):
        return scroll_main(Path(sys.argv[6]), Path(sys.argv[5]), int(sys.argv[3]))
    if len(sys.argv) == 4 and sys.argv[1] == "--keep":
        return keep_main(Path(sys.argv[2]), Path(sys.argv[3]))
    if len(sys.argv) == 5 and sys.argv[1] == "--latency" and sys.argv[2] == "--out":
        return latency_main(Path(sys.argv[4]), Path(sys.argv[3]))
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
