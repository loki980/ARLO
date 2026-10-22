# ruff: noqa: INP001, SLF001, FURB113
"""Measure how long a hover takes, from the game's card to ARLO's advice card.

    uv run --extra icons python scripts/eval_hover_latency.py --labels CSV
        --captures DIR --video WEBM --out DIR [--before-ref REF]

Writes DIR/real-data.txt with:

- the scored rows (labels.csv rows with card=yes and an expected title that
  is not empty, unreadable or covered) and the excluded rows by reason
- a replay of every scored row through Scanner._scan_cycle, each card
  hovered twice (a cold title cache before the first hover), broken down
  into locate, title read, hovered slot and located to advice queued, with
  and without a stash pass running on another thread, then the queue to
  shown part (the Tk pump wait, measured on a withdrawn Tk root with the
  code's pump interval, plus rendering the advice card the replay built)
- the hide fixtures: a shown card that goes away (card gone to hide queued
  plus the measured pump wait) and a card missing from one sighting only
- the title look check: no two different titles share a look key, the
  nearest look distance between different titles, on the labels, and with
  --all-captures also on every capture in DIR with a card read by OCR
- the hovered slot strip check: the strip search against the whole frame's
  on every 3840x2160 capture with a hovered cell
- the video: every item card seen in it after ARLO went ACTIVE

Every measurement runs twice, on the code at --before-ref (extracted with
git archive into a temporary folder, by default the merge base with
worktree-overlay-roadmap) and on the working tree, each in its own process.
The capture stands in for the screen grab, so locate here leaves out the
grab itself (about 50 ms live). Chips are left out of the replay, they are
queued after the advice card.
"""

import argparse
import csv
import gc
import json
import os
import queue
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
EXCLUDED = ("unreadable", "covered")
# The popup home box and tk scale of the 21 Oct live log (3840x2160 at 175%)
HOME_BOX = (1384, 1631, 1992, 2009)
TK_SCALE = 1.75
# Locate times of the 21 Oct live log hovers (76, 77 and 124 ms), the grab
# included. The hide fixture's locate takes the median, then the worst
LIVE_LOCATE_S = (0.077, 0.124)
# Video: frames decoded per second and their size
VIDEO_FPS = 10
VIDEO_SIZE = (960, 540)
# The video starts at this local time and the log's ACTIVE at 21:47:58.348
VIDEO_START = "21:47:45"
LOG_ACTIVE_S = 13.348
PUMP_SAMPLES = 200
# Model threads of the concurrent stash pass
STASH_THREADS = 2
HIDE_TRIALS = 30
LOOK_GRID = (128, 16)


def ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


def stats(values: list[float]) -> str:
    if not values:
        return "n=0"
    return (
        f"median {statistics.median(values):6.1f}  worst {max(values):6.1f}  "
        f"n={len(values)}"
    )


def load_rows(labels: Path) -> tuple[list[dict], dict[str, int]]:
    """Scored rows and the excluded counts by reason."""
    with labels.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    scored, excluded = [], {"card=no": 0, "expected empty": 0}
    excluded.update(dict.fromkeys(EXCLUDED, 0))
    for row in rows:
        expected = (row.get("expected") or "").strip()
        if row.get("card") != "yes":
            excluded["card=no"] += 1
        elif not expected:
            excluded["expected empty"] += 1
        elif expected.casefold() in EXCLUDED:
            excluded[expected.casefold()] += 1
        else:
            scored.append({"file": row["file"], "expected": expected})
    return scored, excluded


# --- one code version, in its own process --------------------------------


class StashLoad:
    """A stash pass (identify_slots on a 4K frame) looping on a thread."""

    def __init__(self, captures: Path) -> None:
        import onnxruntime as ort

        from arc_helper import stash_scan
        from arc_helper.config import get_settings
        from arc_helper.database import Database
        from arc_helper.grid import find_slots
        from arc_helper.icon_embed import default_embedder
        from arc_helper.main import load_stash_gallery

        # Few model threads, so the pass shares the cores (and any CPU quota
        # the run is under) with the hover instead of stalling the process
        options = ort.SessionOptions

        def limited():
            made = options()
            made.intra_op_num_threads = STASH_THREADS
            made.inter_op_num_threads = 1
            return made

        ort.SessionOptions = limited
        self.gallery, self.twins = load_stash_gallery(get_settings().icon_cache_dir)
        self.embedder = default_embedder()
        self.db = Database()
        self.identify = stash_scan.identify_slots
        self.frame = None
        for path in sorted(captures.glob("*_grid.png")):
            try:
                with Image.open(path) as image:
                    if image.size != (3840, 2160):
                        continue
                    rgb = np.asarray(image.convert("RGB"))
            except OSError:
                continue
            if len(find_slots(rgb)) >= 40:
                self.frame, self.name = rgb, path.name
                break
        self.passes: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.identify(self.frame, self.gallery, self.db.lookup, embedder=self.embedder)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            start = time.monotonic()
            self.identify(
                self.frame,
                self.gallery,
                self.db.lookup,
                embedder=self.embedder,
                read_stacks=True,
                twins=self.twins,
            )
            self.passes.append(time.monotonic() - start)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


class TimedQueue(queue.Queue):
    """A ui_queue that keeps the monotonic time of every put."""

    def __init__(self) -> None:
        super().__init__()
        self.times: list[tuple[float, tuple]] = []

    def put(self, item, block=True, timeout=None):  # noqa: FBT002
        self.times.append((time.monotonic(), item))
        super().put(item, block, timeout)


def _patch(main, ocr_module, engine, size: tuple[int, int]) -> None:
    main.get_ocr_engine = lambda: engine
    main.get_screen_resolution = lambda: size
    ocr_module.get_screen_resolution = lambda: size


def _scanner(main, db):
    scanner = main.Scanner(
        root=None,
        overlay=None,
        status=None,
        db=db,
        hotkey=SimpleNamespace(is_held=lambda: False),
        advice_card=True,
    )
    scanner._running = True
    scanner.state = main.ScannerState.ACTIVE
    scanner._queue_chips = lambda *_, **__: None
    scanner.ui_queue = TimedQueue()
    return scanner


def _engine(ocr_module, screen_point):
    class ReplayEngine(ocr_module.OCREngine):
        """OCREngine whose search window capture is a saved capture."""

        image = None

        def capture_search_window(self):
            image = self.image
            self._last_capture = image
            self._last_origin = ocr_module.LayoutPoint(0, 0)
            self._last_card = None
            self._last_lines = None
            return image, screen_point(x=image.width // 2, y=image.height // 2)

    engine = ReplayEngine()
    engine.capture_library = False
    return engine


def replay(main, ocr_module, scored, captures: Path) -> dict:
    """Each scored row hovered twice through Scanner._scan_cycle."""
    from arc_helper.config import get_settings
    from arc_helper.database import Database
    from arc_helper.screen import Point

    settings = get_settings()
    db = Database()
    db.set_blueprint_owned = lambda *_, **__: None  # never write the user db
    engine = _engine(ocr_module, Point)
    has_cache = getattr(engine, "title_cache", None) is not None
    marks: dict[str, float] = {}

    locate = engine.locate_card_at_cursor

    def timed_locate():
        found = locate()
        marks["located"] = time.monotonic()
        return found

    engine.locate_card_at_cursor = timed_locate
    read_title = engine._read_title

    def timed_read(image, *, card_found):
        start = time.monotonic()
        try:
            return read_title(image, card_found=card_found)
        finally:
            marks["read"] = time.monotonic() - start

    engine._read_title = timed_read
    hovered = main.hovered_slot_on_screen

    def timed_hovered(*args, **kwargs):
        start = time.monotonic()
        try:
            return hovered(*args, **kwargs)
        finally:
            marks["hovered"] = time.monotonic() - start

    main.hovered_slot_on_screen = timed_hovered
    # The new code searches for the hovered slot on its own thread while the
    # title is read, then finishes it in hovered_slot_on_screen
    search = getattr(main, "search_hovered_slot", None)

    def timed_search(*args, **kwargs):
        start = time.monotonic()
        try:
            return search(*args, **kwargs)
        finally:
            marks["search"] = time.monotonic() - start

    if search is not None:
        main.search_hovered_slot = timed_search

    hovers = []
    for row in scored:
        with Image.open(captures / row["file"]) as image:
            rgb = image.convert("RGB")
        _patch(main, ocr_module, engine, rgb.size)
        engine.image = rgb
        if has_cache:
            engine.title_cache = type(engine.title_cache)()
        for n in (1, 2):
            scanner = _scanner(main, db)
            marks.clear()
            start = time.monotonic()
            scanner._scan_cycle(engine, settings)
            advice = [(t, m) for t, m in scanner.ui_queue.times if m[0] == "advice"]
            hover = {
                "file": row["file"],
                "expected": row["expected"],
                "hover": n,
                "locate": ms(marks.get("located", start) - start),
                "read": ms(marks["read"]) if "read" in marks else None,
                "hovered": ms(marks["hovered"]) if "hovered" in marks else None,
                "search": ms(marks["search"]) if "search" in marks else None,
                "cached": bool(getattr(engine, "last_read_cached", False)),
                "resolved": advice[0][1][1] if advice else None,
                "queued": ms(advice[0][0] - marks["located"]) if advice else None,
                "card": advice[0][1][2] if advice else None,
            }
            hovers.append(hover)
    main.hovered_slot_on_screen = hovered
    if search is not None:
        main.search_hovered_slot = search
    return {"hovers": hovers}


def pump_waits(interval_ms: int) -> list[float]:
    """Queue to applied waits on a withdrawn Tk root pumping every interval_ms."""
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    messages: queue.Queue[float] = queue.Queue()
    waits: list[float] = []
    rng = random.Random(7)

    def pump():
        try:
            while True:
                waits.append(ms(time.monotonic() - messages.get_nowait()))
        except queue.Empty:
            pass
        if len(waits) >= PUMP_SAMPLES:
            root.quit()
            return
        root.after(interval_ms, pump)

    def produce():
        for _ in range(PUMP_SAMPLES):
            time.sleep(rng.uniform(0.005, 0.04))
            messages.put(time.monotonic())

    thread = threading.Thread(target=produce, daemon=True)
    root.after(interval_ms, pump)
    thread.start()
    root.mainloop()
    thread.join()
    root.destroy()
    # The Tcl interpreter must be freed on the thread that made it
    root = None
    gc.collect()
    return waits


def render_times(hovers: list[dict]) -> None:
    """Add the Tk side render time of each hover's card, in place.

    The window state follows each code version: the old show skips the
    render when the card equals the last one shown, the new one also keeps
    the last RENDER_CACHE_SIZE renders. The render cache starts empty for
    each row, so a first hover always renders.
    """
    import tkinter as tk
    from collections import OrderedDict

    from arc_helper import advice_window
    from arc_helper.tooltip_layout import Rect

    root = tk.Tk()
    root.withdraw()
    advice_window.get_backend = lambda: SimpleNamespace(
        tk_scale=TK_SCALE, resolution=lambda: (3840, 2160)
    )
    advice_window.phys_to_tk = lambda value: round(value / TK_SCALE)
    box = Rect(*HOME_BOX)
    cached = hasattr(advice_window.AdviceWindow, "_render")
    window = SimpleNamespace(_renders=OrderedDict())
    last = None
    for hover in hovers:
        card = hover["card"]
        if card is None:
            hover["render"] = None
            continue
        if hover["hover"] == 1:
            window._renders.clear()
            last = None
        start = time.monotonic()
        if card != last:
            if cached:
                advice_window.AdviceWindow._render(window, card, box, TK_SCALE)
            else:
                image = advice_window.render_in_box(card, box)
                flat = advice_window.flatten_on_panel(image)
                size = (
                    round(image.size[0] / TK_SCALE),
                    round(image.size[1] / TK_SCALE),
                )
                flat = flat.resize(size, Image.LANCZOS)
                advice_window.ImageTk.PhotoImage(flat)
        hover["render"] = ms(time.monotonic() - start)
        last = card
    # Tk images must be freed on this thread, before the root goes
    window._renders.clear()
    root.destroy()
    root = None
    gc.collect()


def hide_fixtures(main, ocr_module, waits: list[float]) -> dict:
    """The gone fixture timed per live locate time, and the one miss fixture."""
    from arc_helper.config import get_settings
    from arc_helper.tooltip_layout import Rect

    settings = get_settings()
    signature = ocr_module.CardSignature(
        Rect(100, 100, 860, 900), np.zeros((8, 32), dtype=np.uint8)
    )

    class FixtureEngine:
        """Locates the fixture's card while present, after locate_s."""

        last_tier_vote = None
        _last_capture = None
        _last_origin = None
        _last_card = None
        capture_library = False
        search_width = 0
        search_height = 0

        def __init__(self, locate_s: float, sightings=None) -> None:
            self.locate_s = locate_s
            self.present = True
            self.sightings = list(sightings) if sightings is not None else None

        def locate_card_at_cursor(self):
            # The capture is taken at the start of the locate
            if self.sightings is not None:
                present = self.sightings.pop(0)
            else:
                present = self.present
            time.sleep(self.locate_s)
            return signature if present else None

        @staticmethod
        def read_located_card():
            return None

    def shown_scanner():
        scanner = _scanner(main, SimpleNamespace())
        scanner._card_shown = signature
        scanner._last_shown_item = "Fixture"
        scanner._card_state = "same"
        return scanner

    rng = random.Random(11)
    result: dict = {"gone": {}}
    for locate_s in LIVE_LOCATE_S:
        delays = []
        for _ in range(HIDE_TRIALS):
            engine = FixtureEngine(locate_s)
            scanner = shown_scanner()
            stop = threading.Event()

            def loop(scanner=scanner, engine=engine, stop=stop):
                while not stop.is_set():
                    scanner._scan_cycle(engine, settings)

            thread = threading.Thread(target=loop, daemon=True)
            thread.start()
            time.sleep(rng.uniform(0.3, 0.6))
            engine.present = False
            gone_at = time.monotonic()
            hidden_at = None
            deadline = gone_at + 3.0
            while hidden_at is None and time.monotonic() < deadline:
                time.sleep(0.002)
                hides = [t for t, m in scanner.ui_queue.times if m[0] == "hide"]
                hidden_at = hides[0] if hides else None
            stop.set()
            thread.join()
            if hidden_at is not None:
                delays.append(ms(hidden_at - gone_at) + rng.choice(waits))
        result["gone"][f"{locate_s * 1000:.0f}"] = delays

    # One missed sighting between two sightings, then two misses
    engine = FixtureEngine(0.0, [True, False, True, True, False, False])
    scanner = shown_scanner()
    hidden = []
    for _ in range(6):
        scanner._scan_cycle(engine, settings)
        hidden.append(any(m[0] == "hide" for _, m in scanner.ui_queue.times))
    result["one_miss_hidden"] = hidden[2]
    result["two_misses_hidden"] = hidden[5]
    result["hidden_by_cycle"] = hidden
    return result


def measure(args) -> dict:
    """The timings of one condition on the arc_helper this process imports."""
    import logging

    from arc_helper import main
    from arc_helper import ocr as ocr_module

    logging.getLogger("arc_helper").setLevel(logging.WARNING)
    scored, _ = load_rows(args.labels)
    interval = getattr(main, "UI_PUMP_MS", 50)
    data = {"code": str(Path(main.__file__).parent), "pump_ms": interval}
    load = None
    if args.condition == "stash":
        load = StashLoad(args.captures)
        data["stash_frame"] = load.name
        load.start()
    run = replay(main, ocr_module, scored, args.captures)
    waits = pump_waits(interval)
    render_times(run["hovers"])
    rng = random.Random(3)
    for hover in run["hovers"]:
        hover["wait"] = rng.choice(waits)
        hover.pop("card")
    run["waits"] = waits
    run["hide"] = hide_fixtures(main, ocr_module, waits)
    if load is not None:
        load.stop()
        run["stash_passes"] = [ms(s) for s in load.passes]
    data[args.condition] = run
    return data


# --- the title look check, on the working tree --------------------------


def look_grid(look) -> np.ndarray:
    """A look's lines on one fixed grid per line, for distances."""
    import cv2

    lines = [
        cv2.resize(bits.astype(np.uint8) * 255, LOOK_GRID, interpolation=cv2.INTER_AREA)
        >= 128
        for bits in look.bits
    ]
    return np.concatenate(lines, axis=0)


def look_check(args, scored) -> list[str]:
    import logging

    from arc_helper.ocr import OCREngine
    from arc_helper.ocr import crop_card
    from arc_helper.title_reader import title_look
    from arc_helper.tooltip_layout import Point

    logging.getLogger("arc_helper").setLevel(logging.WARNING)
    out = ["", "== Title look check (working tree) =="]
    looks = []
    for row in scored:
        with Image.open(args.captures / row["file"]) as image:
            rgb = image.convert("RGB")
        found = crop_card(rgb, Point(rgb.width // 2, rgb.height // 2))
        look = title_look(np.asarray(found[0])) if found else None
        again = title_look(np.asarray(found[0])) if found else None
        looks.append((row, look, again))
    no_look = [row["file"] for row, look, _ in looks if look is None]
    unstable = [row["file"] for row, a, b in looks if a and b and a.key != b.key]
    out.append(f"scored rows with a title look: {len(looks) - len(no_look)}")
    out.append(f"scored rows without one (no card or title lines): {no_look}")
    out.append(f"look of the same crop twice differs: {len(unstable)} {unstable}")
    by_key: dict[bytes, set[str]] = {}
    per_title: dict[str, set[bytes]] = {}
    for row, look, _ in looks:
        if look is None:
            continue
        by_key.setdefault(look.key, set()).add(row["expected"])
        per_title.setdefault(row["expected"], set()).add(look.key)
    shared = [titles for titles in by_key.values() if len(titles) > 1]
    out.append(f"look keys shared by different titles: {len(shared)} {shared}")
    out.append("distinct looks per title (fewer means more hits across captures):")
    for title, keys in sorted(per_title.items()):
        rows = sum(1 for row, look, _ in looks if look and row["expected"] == title)
        out.append(f"  {title:<14} rows {rows:>3}  looks {len(keys)}")
    out.extend(
        _nearest(
            [(row["expected"], row["file"], look) for row, look, _ in looks if look]
        )
    )

    if not args.all_captures:
        out.append("")
        out.append("all captures check: not run (--all-captures)")
        return out
    # Every capture with a card, labelled by its own OCR read
    engine = OCREngine()
    engine.title_cache = None
    reads = []
    for path in sorted(args.captures.glob("*.png")):
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
        except OSError:
            continue
        found = crop_card(rgb, Point(rgb.width // 2, rgb.height // 2))
        if found is None:
            continue
        look = title_look(np.asarray(found[0]))
        if look is None:
            continue
        text = engine._read_title(found[0], card_found=True)
        if text:
            reads.append((text, path.name, look))
    by_key = {}
    for text, _, look in reads:
        by_key.setdefault(look.key, set()).add(text)
    shared = [texts for texts in by_key.values() if len(texts) > 1]
    out.append("")
    out.append(
        f"all captures in the folder with a card and a read: {len(reads)}, "
        f"distinct reads {len({t for t, _, _ in reads})}, looks {len(by_key)}"
    )
    out.append(f"look keys shared by different reads: {len(shared)} {shared}")
    out.extend(_nearest(reads))
    return out


def _nearest(items) -> list[str]:
    """The nearest look distance between items with different titles."""
    grids = [(title, name, look, look_grid(look)) for title, name, look in items]
    best = None
    best_same_shape = None
    for i, (title_a, name_a, look_a, grid_a) in enumerate(grids):
        for title_b, name_b, look_b, grid_b in grids[i + 1 :]:
            if title_a == title_b or grid_a.shape != grid_b.shape:
                continue
            distance = int(np.count_nonzero(grid_a != grid_b))
            pair = (distance, title_a, name_a, title_b, name_b)
            if best is None or distance < best[0]:
                best = pair
            sizes_a = [(r - left, b - t) for left, t, r, b in look_a.boxes]
            sizes_b = [(r - left, b - t) for left, t, r, b in look_b.boxes]
            if sizes_a == sizes_b and (
                best_same_shape is None or distance < best_same_shape[0]
            ):
                best_same_shape = pair
    bits = LOOK_GRID[0] * LOOK_GRID[1]
    lines = []
    if best is None:
        lines.append("nearest different titles: no pair with the same line count")
    else:
        lines.append(
            f"nearest different titles on a {LOOK_GRID[0]}x{LOOK_GRID[1]} grid per "
            f"line: {best[0]} bits of {bits} per line differ, {best[1]} "
            f"({best[2]}) vs {best[3]} ({best[4]})"
        )
    if best_same_shape is None:
        lines.append(
            "different titles with the same line sizes in px: none, so their "
            "keys can never match"
        )
    else:
        lines.append(
            f"nearest different titles with the same line sizes: "
            f"{best_same_shape[0]} bits differ, {best_same_shape[1]} vs "
            f"{best_same_shape[3]}"
        )
    return lines


def strip_check(args) -> list[str]:
    """The strip slot search against the whole frame's, on every capture."""
    import logging

    from arc_helper import main
    from arc_helper.grid import find_slots
    from arc_helper.hovered_cell import find_hovered_cell
    from arc_helper.hovered_cell import snap_to_slot
    from arc_helper.tooltip_layout import Point
    from arc_helper.tooltip_layout import find_tooltip_card

    logging.getLogger("arc_helper").setLevel(logging.WARNING)
    same, differ, cells = 0, [], 0
    strip_ms: dict[str, list[float]] = {}
    full_ms: dict[str, list[float]] = {}
    for path in sorted(args.captures.glob("*.png")):
        try:
            with Image.open(path) as image:
                if image.size != (3840, 2160):
                    continue  # a strip only runs on a frame twice its width
                rgb = np.asarray(image.convert("RGB"))
        except OSError:
            continue
        card = find_tooltip_card(rgb, Point(rgb.shape[1] // 2, rgb.shape[0] // 2))
        cell = find_hovered_cell(rgb, card) if card is not None else None
        if cell is None:
            continue
        cells += 1
        size = f"{rgb.shape[1]}x{rgb.shape[0]}"
        start = time.perf_counter()
        strip = main._snap_hovered(rgb, cell)
        strip_ms.setdefault(size, []).append(ms(time.perf_counter() - start))
        start = time.perf_counter()
        full = snap_to_slot(cell, find_slots(rgb))
        full_ms.setdefault(size, []).append(ms(time.perf_counter() - start))
        if (strip and strip.rect) == (full and full.rect):
            same += 1
        else:
            differ.append(path.name)
    out = ["", "== Hovered slot strip check (working tree) =="]
    out.append(
        f"captures with a hovered cell: {cells}, strip snap equals the whole "
        f"frame's: {same}, differs: {len(differ)} {differ}"
    )
    out.extend(
        f"  {size}: strip ms {stats(strip_ms[size])} | whole frame ms "
        f"{stats(full_ms[size])}"
        for size in sorted(strip_ms)
    )
    return out


# --- the video ------------------------------------------------------------


def video_check(video: Path) -> list[str]:
    import cv2

    from arc_helper.tooltip_layout import CREAM_LOWER
    from arc_helper.tooltip_layout import CREAM_UPPER

    out = ["", f"== Video {video.name} =="]
    width, height = VIDEO_SIZE
    scale = 3840 / width
    min_side = round(150 / scale)
    process = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-threads", "4", "-i", str(video), "-vf",
            f"fps={VIDEO_FPS},scale={width}:{height}:flags=area",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        stdout=subprocess.PIPE,
    )  # fmt: skip
    frame_bytes = width * height * 3
    index = 0
    seen: list[tuple[float, tuple]] = []
    brightness: list[float] = []
    lower = tuple(int(v) for v in CREAM_LOWER)
    upper = tuple(int(v) for v in CREAM_UPPER)
    while True:
        data = process.stdout.read(frame_bytes)
        if len(data) < frame_bytes:
            break
        frame = np.frombuffer(data, np.uint8).reshape(height, width, 3)
        brightness.append(float(frame.mean()))
        mask = cv2.inRange(np.ascontiguousarray(frame), lower, upper)
        count, _, found, _ = cv2.connectedComponentsWithStats(mask)
        for label in range(1, count):
            left, top, w, h, area = (int(v) for v in found[label][:5])
            if w >= min_side and h >= min_side and area / (w * h) >= 0.6:
                seen.append((index / VIDEO_FPS, (left, top, w, h)))
        index += 1
    process.wait()
    out.append(
        f"decoded {index} frames at {VIDEO_FPS} fps, {width}x{height}, "
        f"started {VIDEO_START} local"
    )
    drops = np.diff(brightness)
    if drops.size:
        at = int(np.argmin(drops)) + 1
        out.append(
            f"alignment: the biggest brightness drop (lobby to the dark "
            f"inventory) is at t={at / VIDEO_FPS:.1f} s, the log's ACTIVE at "
            f"t={LOG_ACTIVE_S:.1f} s (the IDLE trigger check runs about once a "
            "second)"
        )
    out.append(
        f"item card candidates (cream, at least 150 px a side at 4K, 60 percent "
        f"filled): {len(seen)} frames"
    )
    times = sorted({t for t, _ in seen})
    if times:
        out.append(f"  at t = {times[0]:.1f} to {times[-1]:.1f} s: {times}")
    after = [t for t in times if t >= LOG_ACTIVE_S]
    out.append(
        f"hovers with ARLO ACTIVE (a card at or after t={LOG_ACTIVE_S:.1f} s): "
        f"{len(after)}"
    )
    if not after:
        out.append(
            "  no item card is on screen while ARLO is ACTIVE, so the video has "
            "no hover to time (the candidates before are menus and the desktop)"
        )
    return out


# --- the report -----------------------------------------------------------


def _section(name: str, data: dict) -> list[str]:
    out = [f"-- {name}: code {data['code']}, pump every {data['pump_ms']} ms --"]
    for condition in ("idle", "stash"):
        run = data.get(condition)
        if run is None:
            out.append(f"[{condition}] not run")
            continue
        label = (
            "no stash pass"
            if condition == "idle"
            else (
                f"stash pass looping on {data['stash_frame']}, "
                f"passes {stats(run.get('stash_passes', []))} ms"
            )
        )
        out.append(f"[{label}]")
        hovers = run["hovers"]
        for n in (1, 2):
            rows = [h for h in hovers if h["hover"] == n]
            shown = [h for h in rows if h["queued"] is not None]
            out.append(
                f" hover {n}{' (first, cold cache)' if n == 1 else ' (seen before)'}:"
            )
            out.append(
                f"  locate (no grab)        ms {stats([h['locate'] for h in rows])}"
            )
            out.append(
                f"  title read              ms "
                f"{stats([h['read'] for h in rows if h['read'] is not None])}"
            )
            out.append(
                f"  hovered slot (inline)   ms "
                f"{stats([h['hovered'] for h in rows if h['hovered'] is not None])}"
            )
            out.append(
                f"  hovered slot search, on its own thread during the read, ms "
                f"{stats([h['search'] for h in rows if h.get('search') is not None])}"
            )
            out.append(
                f"  located to queued       ms {stats([h['queued'] for h in shown])}"
            )
            out.append(
                f"  pump wait               ms {stats([h['wait'] for h in shown])}"
            )
            out.append(
                f"  render                  ms "
                f"{stats([h['render'] for h in shown if h['render'] is not None])}"
            )
            out.append(
                f"  queue to shown          ms "
                f"{stats([h['wait'] + h['render'] for h in shown])}"
            )
            out.append(
                f"  located to shown        ms "
                f"{stats([h['queued'] + h['wait'] + h['render'] for h in shown])}"
            )
            wrong = [
                h for h in rows
                if h["resolved"] and h["resolved"].casefold() != h["expected"].casefold()
            ]  # fmt: skip
            missing = [h["file"] for h in rows if not h["resolved"]]
            cached = sum(1 for h in rows if h["cached"])
            out.append(
                f"  title cache hits {cached}/{len(rows)}, wrong titles {len(wrong)}, "
                f"no advice {len(missing)} {missing}"
            )
            out.extend(
                f"   wrong: {h['file']} {h['resolved']} != {h['expected']}"
                for h in wrong
            )
        firsts = {h["file"]: h["resolved"] for h in hovers if h["hover"] == 1}
        differ = [
            h["file"] for h in hovers
            if h["hover"] == 2 and h["resolved"] != firsts[h["file"]]
        ]  # fmt: skip
        out.append(f"  second hover resolved differently from the first: {differ}")
        hide = run["hide"]
        for locate, delays in hide["gone"].items():
            out.append(
                f" hide, card gone to hidden with pump wait (locate {locate} ms): "
                f"ms {stats(delays)}"
            )
        out.append(
            f" one missed sighting hides: {hide['one_miss_hidden']}, two misses "
            f"hide: {hide['two_misses_hidden']} (hidden after each cycle "
            f"{hide['hidden_by_cycle']})"
        )
    return out


def run_version(args, src: Path | None, json_path: Path, condition: str) -> dict:
    env = dict(os.environ)
    if src is not None:
        env["PYTHONPATH"] = str(src)
    command = [
        sys.executable, str(Path(__file__).resolve()), "--part", "measure",
        "--labels", str(args.labels), "--captures", str(args.captures),
        "--json", str(json_path), "--condition", condition,
    ]  # fmt: skip
    subprocess.run(command, check=True, env=env, cwd=src.parent if src else REPO)
    return json.loads(json_path.read_text(encoding="utf-8"))


def before_tree(ref: str | None, folder: Path) -> tuple[str, Path]:
    """Extract src and items.db at the before ref into folder."""
    if ref is None:
        ref = subprocess.run(
            ["git", "-C", str(REPO), "merge-base", "HEAD", "worktree-overlay-roadmap"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()  # fmt: skip
    archive = subprocess.run(
        ["git", "-C", str(REPO), "archive", ref, "src", "items.db"],
        check=True, capture_output=True,
    ).stdout  # fmt: skip
    subprocess.run(["tar", "-x", "-C", str(folder)], input=archive, check=True)
    cache = REPO / "cache"
    if cache.exists():
        (folder / "cache").symlink_to(cache.resolve())
    return ref, folder / "src"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--before-ref")
    parser.add_argument("--no-stash", action="store_true")
    parser.add_argument(
        "--all-captures",
        action="store_true",
        help="also read every capture with a card for the look check (slow)",
    )
    parser.add_argument("--part", choices=["measure"])
    parser.add_argument("--json", type=Path)
    parser.add_argument("--condition", choices=["idle", "stash"], default="idle")
    args = parser.parse_args()

    if args.part == "measure":
        args.json.write_text(json.dumps(measure(args)), encoding="utf-8")
        return 0

    scored, excluded = load_rows(args.labels)
    load = ", ".join(f"{v:.1f}" for v in os.getloadavg())
    out = [f"load average at the start (1, 5, 15 min): {load} on {os.cpu_count()} cpus"]
    out.append("== Rows ==")
    out.append(f"scored rows: {len(scored)}")
    counts: dict[str, int] = {}
    for row in scored:
        counts[row["expected"]] = counts.get(row["expected"], 0) + 1
    out.append(f"  by title: {dict(sorted(counts.items()))}")
    out.append(f"excluded rows by reason: {excluded}")

    def save() -> None:
        if args.out is not None:
            args.out.mkdir(parents=True, exist_ok=True)
            text = "\n".join(out) + "\n"
            (args.out / "real-data.txt").write_text(text, encoding="utf-8")

    results = {}
    with tempfile.TemporaryDirectory() as folder:
        ref, src = before_tree(args.before_ref, Path(folder))
        # The versions take turns per condition, so both see a similar load
        conditions = ("idle",) if args.no_stash else ("idle", "stash")
        for condition in conditions:
            for name, tree in (("before", src), ("after", None)):
                json_path = Path(folder) / f"{name}-{condition}.json"
                data = run_version(args, tree, json_path, condition)
                results.setdefault(name, {}).update(data)
            if args.out is not None:
                args.out.mkdir(parents=True, exist_ok=True)
                (args.out / "hover-latency.json").write_text(
                    json.dumps(results), encoding="utf-8"
                )
    out.append("")
    out.append("== Hover replay, each scored card hovered twice ==")
    out.append(f"before: {ref}")
    out.extend(_section("before", results["before"]))
    out.append("")
    out.extend(_section("after", results["after"]))
    save()
    out.extend(look_check(args, scored))
    save()
    out.extend(strip_check(args))
    save()
    if args.video is not None:
        out.extend(video_check(args.video))
    save()
    text = "\n".join(out) + "\n"
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
