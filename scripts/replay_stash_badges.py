# ruff: noqa: INP001
"""Replay saved captures through the live stash badge hand-off and pass.

    uv run --extra icons python scripts/replay_stash_badges.py <captures> <out folder>
    uv run --extra icons python scripts/replay_stash_badges.py --keep <capture> <out file>
    uv run --extra icons python scripts/replay_stash_badges.py --keep --out <folder> <capture>
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

With --keep --out the flicker cases of the flicker-why-s1 timeline are fed,
each to its own StashBadges on one warm slot cache: a card fading in (a
half-faded card rect over 16 backpack slots before the scanner locates it,
then the card located), a card fading out after its linger lapsed, the
cursor hovering three slots in turn (each brightened, the cursor passed when
StashBadges takes one) and a pass that finds fewer badges on a still grid
(the backpack viewport scrolled one row, then a pass that misses three
slots in it). real-data.txt in the out folder gets, per step, the badges
drawn and the clear and hide lines, and per case the clears and hides in
all and the fewest badges drawn after the first draw (steps with a located
card and the step after one left out, the card covers badges by design). When keep-before.txt is in the folder (a run on the code before), that
file comes first.

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

--latency then feeds three open and close cases on the same clock and warm
cache, each to its own StashBadges: close (the capture CLOSE_GRID_FRAMES
times, then CLOSE_GONE_FRAMES black frames with no grid, the inventory
closed), reopen (after those, the clear the scanner makes on leaving ACTIVE,
then the capture CLOSE_GRID_FRAMES times) and lone (the capture, then two
frames on which find_slots loses LONE_SLOTS filled slots of a still grid,
then the capture again). real-data.txt gets, per case, every frame fed with
the badges drawn, the stash log lines, and the time from the first black
frame to the clear (close), from the first frame of the reopened grid to
the first draw and whether a pass had run by then (reopen) and the clears
(lone).

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

import inspect
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

import arc_helper.main as main_module
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
from arc_helper.tooltip_layout import Point
from arc_helper.tooltip_layout import Rect

FULL_SIZE = (3840, 2160)
LATENCY_FRAME_S = 0.15
LATENCY_STILL_FRAMES = 10
SCROLL_FRAME_S = 0.1
CLOSE_GRID_FRAMES = 4
CLOSE_GONE_FRAMES = 3
LONE_SLOTS = 4
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


# Cream the game's item card is drawn in, blended over the slots at half
# strength for a card fading in or out
CARD_CREAM = (226, 218, 196)


class _FlickerSurface(RecordingSurface):
    """A recording surface that applies translates as the badge helper does."""

    def draw(self, badges: list) -> None:
        self.drawn = [{**asdict(b), "rect": list(b.rect)} for b in badges]

    def translate(self, region, dy: int) -> None:
        if self.drawn is not None:
            self.drawn = translate_badges(self.drawn, region, dy)


def _faded(rgb: np.ndarray, card: Rect) -> np.ndarray:
    """rgb with a card rect blended halfway to the card's cream."""
    out = rgb.copy()
    area = out[card.top : card.bottom, card.left : card.right]
    area[:] = (area * 0.5 + np.array(CARD_CREAM) * 0.5).astype(np.uint8)
    return out


def _lit(rgb: np.ndarray, slot: Rect) -> np.ndarray:
    """rgb with a slot's interior brightened, as the game lights a hovered slot."""
    out = rgb.copy()
    inset = 16
    part = out[
        slot.top + inset : slot.bottom - inset, slot.left + inset : slot.right - inset
    ]
    part[:] = (part * 0.75 + 255 * 0.25).astype(np.uint8)
    return out


def _centre(rect: Rect) -> tuple[int, int]:
    return ((rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2)


def _flicker_cases(rgb: np.ndarray) -> list[tuple[str, list]]:
    """The cases as (name, steps), each step (label, frame, card, cursor, wait, miss).

    miss is the slot rects the next pass leaves out, None for none.
    """

    backpack = [s.rect for s in find_slots(rgb) if s.panel == "backpack"]
    card = _bounds(backpack[:16])
    half = _faded(rgb, card)
    full = rgb.copy()
    full[card.top : card.bottom, card.left : card.right] = CARD_CREAM
    hovered = backpack[5:8]
    off = Point(rgb.shape[1] - 10, rgb.shape[0] - 10)
    frames, _band, _pitch = _scroll_frames(rgb, 1)
    scrolled = frames[0][0]
    viewport = [s.rect for s in find_slots(scrolled) if s.panel == "backpack"]
    missed = viewport[:3]
    start = [
        ("frame", rgb, None, off, 0.1, None),
        ("frame again", rgb, None, off, 0.1, None),
    ]
    return [
        (
            "card fade in (22:20:38.365, clear 0.07 s before 'card new')",
            [
                *start,
                (
                    "half-faded card over 16 slots, not located",
                    half,
                    None,
                    off,
                    0.1,
                    None,
                ),
                ("card located", full, card, off, 0.1, None),
                ("card located again", full, card, off, 0.1, None),
            ],
        ),
        (
            "card fade out after the linger (22:15:09.777, clear 0.06 s after 'card gone')",
            [
                *start,
                ("card located", full, card, off, 0.1, None),
                (
                    "half-faded card, linger lapsed",
                    half,
                    None,
                    off,
                    CARD_LINGER_S + 0.1,
                    None,
                ),
                ("card gone", rgb, None, off, 0.1, None),
                ("card gone again", rgb, None, off, 0.1, None),
            ],
        ),
        (
            "hover over three slots (22:20:35 to 22:20:47, hide then pass per move)",
            [
                *start,
                *(
                    (
                        f"slot {n} hovered and lit",
                        _lit(rgb, r),
                        None,
                        Point(*_centre(r)),
                        0.1,
                        None,
                    )
                    for n, r in enumerate(hovered, 1)
                    for _ in range(2)
                ),
                ("cursor off the grid", rgb, None, off, 0.1, None),
                ("cursor off the grid again", rgb, None, off, 0.1, None),
            ],
        ),
        (
            "fewer badges on a still grid (22:40:07.489, 76 then 61)",
            [
                *start,
                ("backpack scrolled one row", scrolled, None, off, 0.1, None),
                ("still, the pass misses 3 slots", scrolled, None, off, 0.1, missed),
                ("still again", scrolled, None, off, 0.1, None),
            ],
        ),
    ]


def flicker_main(capture: Path, out: Path) -> int:
    """Replay the flicker cases and write the clears and hides of each step."""
    lines = _Lines()
    logger.addHandler(lines)
    logger.setLevel(logging.DEBUG)
    settings = get_settings()
    db = Database()
    identify = stash_identifier(db, settings.icon_cache_dir)
    make = stash_badge_maker(db, lambda: settings.show_ended_projects)
    with Image.open(capture) as image:
        rgb = np.asarray(image.convert("RGB"))
    identify(rgb)
    takes_cursor = "cursor" in inspect.signature(StashBadges.feed).parameters
    report = [f"capture: {capture.name}", f"cursor passed: {takes_cursor}", ""]
    totals = []
    for name, steps in _flicker_cases(rgb):
        miss: list = []

        def passing(frame, miss=miss):
            found = identify(frame)
            if miss:
                gone = miss.pop()
                found = [s for s in found if s.rect not in gone]
            return found

        if hasattr(identify, "peek"):
            passing.peek = identify.peek
        surface = _FlickerSurface()
        clock = _Clock()
        stash = StashBadges(
            passing,
            make,
            surface,
            keep_border=settings.overlay.stash_badge_keep_border,
            gap_s=0,
            threaded=False,
            clock=clock,
        )
        report.append(f"== {name}")
        clears = hides = 0
        counts: list[int] = []
        for label, frame, card, cursor, wait, missed in steps:
            clock.now += wait
            if missed:
                miss.append(set(missed))
            first = len(lines.lines)
            if takes_cursor:
                stash.feed(frame, card, (), cursor)
            else:
                stash.feed(frame, card, ())
            stash.step()
            new = lines.lines[first:]
            clears += sum("badges cleared" in line for line in new)
            hides += sum("badges hidden" in line for line in new)
            counts.append(len(surface.drawn or []))
            report.append(f"{label}: {counts[-1]} badges drawn")
            report += [
                f"  log: {line}"
                for line in new
                if not line.startswith(("stash pass identified", "stash latency"))
            ]
        # The fewest drawn after the first draw, outside steps a card covers
        # and the step right after one (the badges under it stay off)
        cards = [step[2] for step in steps]
        shown = [
            n
            for i, n in enumerate(counts)
            if cards[i] is None and (i == 0 or cards[i - 1] is None)
        ]
        first = next((i for i, n in enumerate(shown) if n), len(shown))
        least = min(shown[first:], default=0)
        tally = (
            f"clears={clears} hides={hides} drawn={shown[first] if shown[first:] else 0}"
            f" fewest_after={least}"
        )
        report += [f"  in all: {tally}", ""]
        totals.append(f"{name}: {tally}")
    report += ["summary:", *totals]
    out.mkdir(parents=True, exist_ok=True)
    text = "\n".join(report) + "\n"
    before = out / "keep-before.txt"
    if before.exists():
        text = (
            "=== before (keep-before.txt, the code before this change) ===\n"
            + before.read_text(encoding="utf-8")
            + "\n=== after ===\n"
            + text
        )
    (out / "real-data.txt").write_text(text, encoding="utf-8")
    print(text)
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
    report += _open_close_cases(rgb, identify, make, settings, clock, lines)
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


class _ClearSurface(_TimedSurface):
    """A timed surface that also logs each clear's clock time."""

    def __init__(self, clock: _Clock, passes: list) -> None:
        super().__init__(clock, passes)
        self.clears: list[float] = []

    def clear(self) -> None:
        super().clear()
        self.clears.append(self.clock.now)


def _open_close_cases(rgb, identify, make, settings, clock, lines) -> list[str]:
    """The close, reopen and lone cases of --latency, as report lines."""
    real_find = main_module.find_slots
    lone = rgb.copy()
    dropped = [s for s in real_find(rgb) if s.filled][:LONE_SLOTS]

    def find(frame):
        found = real_find(frame)
        return [s for s in found if s not in dropped] if frame is lone else found

    main_module.find_slots = find
    try:
        cases = [
            (
                "close and reopen",
                [("capture", rgb)] * CLOSE_GRID_FRAMES
                + [("closed", np.zeros_like(rgb))] * CLOSE_GONE_FRAMES
                + [("left ACTIVE", None)]
                + [("reopened", rgb)] * CLOSE_GRID_FRAMES,
            ),
            (
                f"lone ({LONE_SLOTS} filled slots lost by find_slots)",
                [("capture", rgb)] * CLOSE_GRID_FRAMES
                + [("lone", lone)] * 2
                + [("capture", rgb)] * 2,
            ),
        ]
        report: list[str] = []
        for name, frames in cases:
            report += _case_run(name, frames, identify, make, settings, clock, lines)
    finally:
        main_module.find_slots = real_find
    return report


def _case_run(name, frames, identify, make, settings, clock, lines) -> list[str]:
    """Feed one case LATENCY_FRAME_S apart and report its draws and clears."""
    passes: list = []
    timed = _timed(clock, identify, passes)
    if hasattr(identify, "peek"):
        timed.peek = _timed(clock, identify.peek)
    surface = _ClearSurface(clock, passes)
    stash = StashBadges(
        timed,
        _timed(clock, make),
        surface,
        keep_border=settings.overlay.stash_badge_keep_border,
        threaded=False,
        clock=clock,
    )
    report = ["", f"=== case: {name} ==="]
    start = clock.now = clock.now + 10.0
    fed: list[tuple[str, float]] = []
    n = 0
    for label, frame in frames:
        if frame is None:
            # The scanner leaves ACTIVE and stops feeding
            first = len(lines.lines)
            stash.clear("inventory_closed")
            report.append(f"t={(clock.now - start) * 1000:6.0f} ms {label}")
            report += [f"  log: {line}" for line in lines.lines[first:]]
            continue
        clock.now = max(clock.now, start + n * LATENCY_FRAME_S)
        n += 1
        fed.append((label, clock.now))
        first, drawn = len(lines.lines), len(surface.draws)
        stash.feed(frame, None, ())
        stash.step()
        shown = surface.drawn
        report.append(
            f"t={(fed[-1][1] - start) * 1000:6.0f} ms fed {label}: "
            f"{len(shown) if shown else 0} badges on screen"
            + (", drawn" if len(surface.draws) > drawn else "")
        )
        report += [f"  log: {line}" for line in lines.lines[first:]]
    report.append("")
    report.append(f"clears at: {[round((t - start) * 1000) for t in surface.clears]}")

    def first_at(label: str) -> float | None:
        return next((t for got, t in fed if got == label), None)

    closed, reopened = first_at("closed"), first_at("reopened")
    if closed is not None:
        after = [t for t in surface.clears if t >= closed]
        report.append(
            "first closed frame to the clear: "
            + (f"{(after[0] - closed) * 1000:.0f} ms" if after else "no clear")
        )
    if reopened is not None:
        after = [d for d in surface.draws if d[0] >= reopened]
        if after:
            t, count, done = after[0]
            before = sum(1 for p in passes if p[0] < reopened)
            report.append(
                f"first reopened frame to the first draw: "
                f"{(t - reopened) * 1000:.0f} ms, {count} badges, "
                f"{'before' if done == before else 'after'} its pass"
            )
        else:
            report.append("first reopened frame to the first draw: no draw")
    lone_at = first_at("lone")
    if lone_at is not None:
        cleared = [t for t in surface.clears if t >= lone_at]
        report.append(f"clears after the lone frames: {len(cleared)}")
    return report


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
    if len(sys.argv) == 5 and sys.argv[1:3] == ["--keep", "--out"]:
        return flicker_main(Path(sys.argv[4]), Path(sys.argv[3]))
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
