# ruff: noqa: INP001, SLF001
"""Why the stash map stopped short on Jeffrey's one scroll of 21 Oct.

    uv run --extra icons python scripts/eval_stash_gaps.py \\
        --frames '<glob>' --log <live log> --db <user db copy> --out <folder>
        [--items <items.db>]

Investigation only (Phase 5, step 7, step 1 of the fix). Writes
real-data.txt to the out folder with every number the report cites.

The user database given is never written: each run works on its own copy in
a temporary folder, with the stash map cleared (the map before the scroll
was dropped as an old version, the live map started empty on the top
frame). items.db comes from Jeffrey's checkout by default, as the live run
used it: its item count is part of the slot cache's fingerprint.

1. Pairing. Each 'stash map observe' line of the log goes with the 'stash
   pass ... duration=' line right after it. The frame time is that line's
   time less the duration, and the frame is the capture nearest that time
   within PAIR_S (the earlier one on a tie). The other reading is listed
   too: a grid capture is saved only when the grid moved, so the still
   frame a pass took is the latest capture at or before its frame time.
2. Pass time. The live pass path of StashBadges._run_pass runs on the
   captures paired by the first rule, in log order: stash_identifier
   (identify_slots with the slot cache, read_stack per miss),
   StashLedger.after_pass (STASH and LOADOUT header reads,
   StashMapStore.observe, the loadout tally) and stash_badge_maker. Parts
   are timed by wrapping the functions they call. Twice as bounds: from an
   empty slot cache, and from the copied database's final cache. Then every
   capture from an empty cache, whose identifications feed the replays.
3. Slot signatures of one cell seen on two frames, at the same screen place
   and after a scroll, and empty tiles drawn at two screen places.
4. Replays through StashMapStore.observe, as StashLedger._observe calls it,
   from an empty map: every capture, the captures of the first pairing and
   those of the second. Truth is the scrollbar thumb: its top moves a fixed
   share of a px per px scrolled (fitted on the placed frames), so every
   frame with a visible thumb has a true row 0. The calls of _place are
   recorded, so an unplaced frame shows the place its checks refused.
5. Frames refused at their true place, the wrong place of 21:12:49 read
   back from the copied database's final cells and judged on the replay's
   state, and a rule that refuses it, checked against every correct
   placement of the three replays.
"""

import argparse
import itertools
import logging
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import UTC
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from arc_helper import main as arlo_main
from arc_helper import stash_map
from arc_helper import stash_scan
from arc_helper.config import get_settings
from arc_helper.database import Database
from arc_helper.grid import STASH
from arc_helper.grid import Slot
from arc_helper.grid import find_slots
from arc_helper.icon_embed import default_embedder
from arc_helper.inventory_header import read_stash_header
from arc_helper.slot_marks import bar_geometry
from arc_helper.stash_map import FIRST
from arc_helper.stash_map import LOOKS
from arc_helper.stash_map import SCROLL
from arc_helper.stash_map import StashMapStore
from arc_helper.stash_map import top_at_rest
from arc_helper.stash_scan import is_wide
from arc_helper.stash_scan import look_signature
from arc_helper.stash_scan import slot_signature
from arc_helper.stash_stitch import LOOK_DIFFERENT
from arc_helper.stash_stitch import LOOK_SAME
from arc_helper.stash_stitch import LOOK_TOLERANCE
from arc_helper.stash_stitch import ROW_RESIDUAL
from arc_helper.stash_stitch import Shift
from arc_helper.stash_stitch import frame_scale
from arc_helper.stash_stitch import judge_offset
from arc_helper.stash_stitch import look_distance
from arc_helper.stash_stitch import look_pairs
from arc_helper.stash_stitch import shift_wrong
from arc_helper.stash_stitch import slot_observations
from arc_helper.stash_stitch import whole_slots

ITEMS_DB = Path("/home/jefflamb/workspace/ARLO/items.db")
# The second scroll, recorded on screen (VP9, 3840x2160, 60 fps), its log and
# the db after it. t 0 of the video is VIDEO_START local time
VIDEO_NAME = "scroll-21oct-2147.webm"
VIDEO_LOG_NAME = "live-log-21oct-2147.txt"
VIDEO_DB_NAME = "jeffrey-arlo_user-21oct-2150.db"
VIDEO_START = "2026-10-21 21:47:45"
# The user database this script must never open
FORBIDDEN = Path("/home/jefflamb/workspace/ARLO/arlo_user.db")
PAIR_S = 1.0
# The log and the capture names are in the local time of the machine
LOCAL = datetime.now(UTC).astimezone().tzinfo
# The observe line of the frame that emptied cells
WRONG_AT = "21:12:49.710"
PLACED = {FIRST, SCROLL, LOOKS}
# Scrollbar thumb: lit columns right of the grid, a run this long (px at
# 2160) is the thumb, anything else (a card or window over it) is not
THUMB_LEN = (165, 185)
THUMB_DX = (20, 90)
THUMB_LIT = stash_map.THUMB_LIT
ROW_0_REST = 544

LOG_TIME = r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3})"
OBSERVE = re.compile(
    LOG_TIME + r".*stash map observe placed=(\w+) changed=(\d+) filled=(\d+)"
    r" header=(\S+) sure=(\w+)"
)
PASS = re.compile(
    LOG_TIME + r".*stash pass identified=(\d+) unidentified=(\d+) badges=(\d+)"
    r" duration=(\d+) ms"
)
WHY = re.compile(LOG_TIME + r".*stash map: frame not placed \((.*)\), nothing changed")
STALE = re.compile(LOG_TIME + r".*stash pass dropped reason=stale")


def _time(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=LOCAL)


def _capture_time(path: Path) -> datetime:
    stamp = path.name.split("_")[0]
    return datetime.strptime(stamp + "000", "%Y%m%d-%H%M%S-%f").replace(tzinfo=LOCAL)


def _short(path) -> str:
    if isinstance(path, str):
        return path
    return path.name.split("_")[0][9:]


def _load(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


# --- the log and the pairing -------------------------------------------------


def parse_log(path: Path) -> list[dict]:
    """The observe lines, each with its pass line, why and stale drop."""
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[dict] = []
    why = None
    for line in lines:
        if m := WHY.match(line):
            why = m.group(2)
        elif m := OBSERVE.match(line):
            out.append(
                {
                    "at": _time(m.group(1)),
                    "placed": m.group(2),
                    "changed": int(m.group(3)),
                    "filled": int(m.group(4)),
                    "header": m.group(5),
                    "why": why or "",
                    "pass": None,
                    "stale": False,
                }
            )
            why = None
        elif (m := PASS.match(line)) and out and out[-1]["pass"] is None:
            out[-1]["pass"] = {
                "at": _time(m.group(1)),
                "identified": int(m.group(2)),
                "unidentified": int(m.group(3)),
                "badges": int(m.group(4)),
                "ms": int(m.group(5)),
            }
        elif STALE.match(line) and out and out[-1]["pass"] is not None:
            out[-1]["stale"] = True
    return out


def pair(observes: list[dict], paths: list[Path]) -> None:
    """Set each observe's frame time and capture (None when none is near)."""
    times = [_capture_time(p) for p in paths]
    for o in observes:
        o["capture"] = None
        o["dt"] = None
        if o["pass"] is None:
            o["frame_at"] = None
            continue
        frame_at = o["pass"]["at"].timestamp() - o["pass"]["ms"] / 1000
        o["frame_at"] = frame_at
        best = None
        for n, t in enumerate(times):
            dt = abs(t.timestamp() - frame_at)
            if dt <= PAIR_S and (best is None or dt < best[1]):
                best = (n, dt)
        if best is not None:
            o["capture"] = best[0]
            o["dt"] = times[best[0]].timestamp() - frame_at
        # The other reading: a capture is saved only when the grid moved, so
        # the still frame a pass took is the latest capture at or before it
        before = [n for n, t in enumerate(times) if t.timestamp() <= frame_at]
        o["before"] = before[-1] if before else None
        o["dt_before"] = times[before[-1]].timestamp() - frame_at if before else None


# --- timing ------------------------------------------------------------------


class Probe:
    """Wraps module functions: time spent per label while on, and records."""

    def __init__(self) -> None:
        self.on = False
        self.spent: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.recording = False
        self.events: list[tuple[str, tuple, object]] = []
        self.embedded = 0

    def reset(self) -> None:
        self.spent = defaultdict(float)
        self.calls = defaultdict(int)
        self.events = []
        self.embedded = 0

    def wrap(self, owner, attr: str, label: str, *, keep: bool = False) -> None:
        func = getattr(owner, attr)

        def wrapped(*args, **kwargs):
            if not self.on:
                return func(*args, **kwargs)
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            finally:
                self.spent[label] += time.perf_counter() - start
                self.calls[label] += 1
            if keep and self.recording:
                self.events.append((label, args, result))
            return result

        setattr(owner, attr, wrapped)


PROBE = Probe()


class TimedEmbedder:
    def __init__(self) -> None:
        self._inner = default_embedder()

    def embed(self, squares):
        start = time.perf_counter()
        try:
            return self._inner.embed(squares)
        finally:
            if PROBE.on:
                PROBE.spent["embed"] += time.perf_counter() - start
                PROBE.calls["embed"] += 1
                PROBE.embedded += len(squares)

    def warm(self):
        return self._inner.warm()


def instrument() -> None:
    for attr, label in (
        ("find_slots", "id find_slots"),
        ("unpatch", "id unpatch"),
        ("bar_geometry", "id bar_geometry"),
        ("slot_signature", "id signature"),
        ("look_signature", "id signature"),
        ("crop_square", "id crop_square"),
        ("_rank", "id rank"),
        ("read_tier", "id read_tier"),
        ("is_blueprint_tile", "id blueprint_tile"),
        ("read_stack", "id read_stack"),
    ):
        PROBE.wrap(stash_scan, attr, label)
    embedder = TimedEmbedder()
    stash_scan.default_embedder = lambda: embedder
    for attr, label in (
        ("read_stash_header", "ledger stash header OCR"),
        ("read_loadout_header", "ledger loadout header OCR"),
        ("loadout_tally", "ledger loadout tally"),
        ("verdicts_for", "badges verdicts_for"),
        ("badges_for", "badges badges_for"),
    ):
        PROBE.wrap(arlo_main, attr, label)
    PROBE.wrap(StashMapStore, "observe", "ledger observe")
    PROBE.wrap(stash_map, "find_slots", "observe find_slots")
    PROBE.wrap(stash_map, "slot_observations", "observe looks")
    PROBE.wrap(stash_map, "frame_view", "observe view")
    PROBE.wrap(stash_map, "top_at_rest", "observe top_at_rest")
    PROBE.wrap(stash_map, "grid_probes", "observe probes")
    PROBE.wrap(StashMapStore, "_place", "observe place")
    PROBE.wrap(StashMapStore, "_save", "observe save")
    PROBE.wrap(stash_map, "scroll_offset", "place scroll_offset", keep=True)
    PROBE.wrap(stash_map, "offset_checked", "place offset_checked", keep=True)
    PROBE.wrap(stash_map, "look_offset", "place look_offset", keep=True)
    PROBE.wrap(stash_map, "judge_offset", "place judge_offset", keep=True)


TIMED = [
    "id find_slots",
    "id unpatch",
    "id bar_geometry",
    "id signature",
    "id crop_square",
    "embed",
    "id rank",
    "id read_tier",
    "id blueprint_tile",
    "id read_stack",
    "ledger stash header OCR",
    "ledger loadout header OCR",
    "ledger loadout tally",
    "ledger observe",
    "observe find_slots",
    "observe looks",
    "observe view",
    "observe top_at_rest",
    "observe probes",
    "observe place",
    "observe save",
    "badges verdicts_for",
    "badges badges_for",
]


class Env:
    """Temporary copies: items.db with its CSVs, and user databases."""

    def __init__(self, temp: Path, items: Path, user_db: Path):
        self.temp = temp
        self.items = temp / "items" / "items.db"
        self.items.parent.mkdir()
        shutil.copy2(items, self.items)
        for csv in items.parent.glob("*.csv"):
            shutil.copy2(csv, self.items.parent / csv.name)
        self.user_db = user_db
        self.copies = 0

    def database(
        self, *, keep_cache: bool, keep_map: bool = False, source: Path | None = None
    ) -> Database:
        """A fresh copy of the user database (or source), its stash map cleared
        unless kept."""
        self.copies += 1
        path = self.temp / f"user-{self.copies}.db"
        source = source or self.user_db
        if FORBIDDEN.resolve() in {path.resolve(), source.resolve()}:
            msg = "refusing to open arlo_user.db"
            raise SystemExit(msg)
        shutil.copy2(source, path)
        conn = sqlite3.connect(path)
        with conn:
            if not keep_map:
                conn.execute("DELETE FROM stash_map_cells")
                conn.execute("DELETE FROM stash_map_view")
            if not keep_cache:
                conn.execute("DELETE FROM slot_cache")
        conn.close()
        return Database(db_path=self.items, user_db_path=path)


class Pass:
    """The live pass path of one session: identifier, ledger and badge maker."""

    def __init__(self, db: Database):
        self.db = db
        settings = get_settings()
        self.identify = arlo_main.stash_identifier(db, settings.icon_cache_dir)
        # Load the gallery and the slot cache, and warm the model, as the
        # live warm-up does before the first pass
        self.identify(np.zeros((2160, 3840, 3), np.uint8))
        default_embedder().warm()
        self.ledger = arlo_main.StashLedger(db)
        self.make = arlo_main.stash_badge_maker(
            db, lambda: settings.show_ended_projects, self.ledger.owned
        )

    def run(self, rgb: np.ndarray) -> dict:
        slots = find_slots(rgb)
        hits = self.identify.peek(rgb)
        PROBE.reset()
        PROBE.on = True
        start = time.perf_counter()
        items = self.identify(rgb)
        after_identify = time.perf_counter()
        self.ledger.after_pass(rgb, slots, items)
        after_ledger = time.perf_counter()
        self.make(items)
        end = time.perf_counter()
        PROBE.on = False
        filled = [s for s in slots if s.filled]
        return {
            "total": (end - start) * 1000,
            "identify": (after_identify - start) * 1000,
            "ledger": (after_ledger - after_identify) * 1000,
            "badges": (end - after_ledger) * 1000,
            "parts": {k: v * 1000 for k, v in PROBE.spent.items()},
            "filled": len(filled),
            "stash_filled": sum(s.panel == STASH for s in filled),
            "hits": len(hits),
            "stash_hits": sum(h.panel == STASH for h in hits),
            "embedded": PROBE.embedded,
            "stacks": PROBE.calls.get("id read_stack", 0),
            "identified": sum(i.item is not None for i in items),
            "items": items,
        }

    def identify_again(self, rgb: np.ndarray) -> float:
        start = time.perf_counter()
        self.identify(rgb)
        return (time.perf_counter() - start) * 1000


def _median(values) -> float:
    values = list(values)
    return statistics.median(values) if values else float("nan")


def timing_section(env: Env, paths, observes, out: list[str]) -> dict:
    out.extend(("== 2. Pass time (question 1) ==", ""))
    paired = [o for o in observes if o["capture"] is not None]
    runs = {}
    for name, keep in (("empty cache", False), ("final cache", True)):
        db = env.database(keep_cache=keep)
        session = Pass(db)
        rows = []
        for o in paired:
            rgb = _load(paths[o["capture"]])
            result = session.run(rgb)
            if keep:
                result["again"] = session.identify_again(rgb)
            result["observe"] = o
            rows.append(result)
        session.identify.flush()
        runs[name] = rows
    out.extend(
        (
            "The live pass path on the paired captures in log order, one session per run. ms per pass. live is the log's duration. hits/filled are the filled slots of the whole frame whose signature was in the cache before the pass (stash panel in brackets). emb is the slots embedded, stk the read_stack calls (one per identified miss).",
            "",
        )
    )
    for name, rows in runs.items():
        out.extend(
            (
                f"-- {name} --",
                "  live at        capture       live  total  ident ledger badges  hits/filled (stash)  emb stk  hdrOCR loadOCR observe embed",
            )
        )
        for r in rows:
            o = r["observe"]
            p = r["parts"]
            out.append(
                f"  {o['at'].strftime('%H:%M:%S.%f')[:12]}  "
                f"{_short(paths[o['capture']])}  {o['pass']['ms']:5d}  "
                f"{r['total']:5.0f}  {r['identify']:5.0f}  {r['ledger']:5.0f}  "
                f"{r['badges']:5.0f}  {r['hits']:3d}/{r['filled']:<3d} "
                f"({r['stash_hits']:2d}/{r['stash_filled']:2d})       "
                f"{r['embedded']:3d} {r['stacks']:3d}  "
                f"{p.get('ledger stash header OCR', 0):6.0f} "
                f"{p.get('ledger loadout header OCR', 0):7.0f} "
                f"{p.get('ledger observe', 0):7.0f} {p.get('embed', 0):5.0f}"
            )
        out.append("")
    out.extend(
        (
            "Medians over the paired passes (ms):",
            f"  {'part':28s} {'empty cache':>12s} {'final cache':>12s}",
        )
    )
    out.extend(
        f"  {label:28s} "
        + " ".join(f"{_median(r[label] for r in runs[n]):12.0f}" for n in runs)
        for label in ["total", "identify", "ledger", "badges"]
    )
    out.extend(
        f"  {label:28s} "
        + " ".join(
            f"{_median(r['parts'].get(label, 0.0) for r in runs[n]):12.1f}"
            for n in runs
        )
        for label in TIMED
    )
    out.extend(
        f"  {label + ' (count)':28s} "
        + " ".join(f"{_median(r[label] for r in runs[n]):12.1f}" for n in runs)
        for label in ("embedded", "stacks", "hits", "filled")
    )
    live = [r["observe"]["pass"]["ms"] for r in runs["final cache"]]
    out.append(f"  {'live duration':28s} {_median(live):12.0f}")
    again = [r["again"] for r in runs["final cache"]]
    out.append(
        f"  identify of the same frame again, every slot cached: median "
        f"{_median(again):.0f} ms, max {max(again):.0f} ms"
    )
    for name, rows in runs.items():
        total = [r["total"] for r in rows]
        diffs = [abs(r["total"] - r["observe"]["pass"]["ms"]) for r in rows]
        ratio = [r["observe"]["pass"]["ms"] / r["total"] for r in rows]
        out.append(
            f"  {name}: median |live - replay| {_median(diffs):.0f} ms, "
            f"median live / replay {_median(ratio):.2f}, replay range "
            f"{min(total):.0f} to {max(total):.0f} ms"
        )
    share = {}
    for name, rows in runs.items():
        sums = defaultdict(float)
        for r in rows:
            for k, v in r["parts"].items():
                sums[k] += v
        whole = sum(r["total"] for r in rows)
        share[name] = (sums, whole)
    out.extend(("", "Share of the summed pass time per part:"))
    for label in TIMED:
        if label.startswith(("observe ", "place ")):
            continue
        out.append(
            f"  {label:28s} "
            + " ".join(
                f"{100 * share[n][0].get(label, 0) / share[n][1]:11.1f}%" for n in runs
            )
        )
    out.append("")
    # Every capture passed, from an empty cache
    db = env.database(keep_cache=False)
    session = Pass(db)
    every = [session.run(_load(p)) for p in paths]
    session.identify.flush()
    out.extend(
        (
            f"Every capture passed in order from an empty cache ({len(every)} passes): median total {_median(r['total'] for r in every):.0f} ms, identify {_median(r['identify'] for r in every):.0f}, ledger {_median(r['ledger'] for r in every):.0f}, observe {_median(r['parts'].get('ledger observe', 0) for r in every):.0f}, embedded {_median(r['embedded'] for r in every):.0f} of {_median(r['filled'] for r in every):.0f} filled slots, stash hits {sum(r['stash_hits'] for r in every)} of {sum(r['stash_filled'] for r in every)} stash slots",
            "",
        )
    )
    return {"every": every}


# --- signatures --------------------------------------------------------------


def _signatures(rgb: np.ndarray, rect) -> tuple:
    crop = rgb[rect.top : rect.bottom, rect.left : rect.right]
    crop = crop if is_wide(rect) else stash_scan.unpatch(crop)
    bar = None if is_wide(rect) else bar_geometry(crop)
    box = bar.bar_rect if bar is not None else None
    return slot_signature(crop, box), look_signature(crop), crop.shape[:2], crop, box


def signature_section(frames: dict, replay: dict, out: list[str]) -> None:
    """Signatures of one cell on consecutive correctly placed frames."""
    out.extend(("== 3. Slot signatures of one cell on two frames ==", ""))
    placed = [
        r for r in replay["rows"] if r["how"] in PLACED and r.get("err") is not None
    ]
    placed = [r for r in placed if abs(r["err"]) < 0.5]
    groups = {"same place": [], "scrolled": []}
    for a, b in itertools.pairwise(placed):
        ka, kb = (
            frames[a["index"]]["keys"](a["origin"]),
            frames[b["index"]]["keys"](b["origin"]),
        )
        for key in set(ka) & set(kb):
            ra, rb = ka[key], kb[key]
            if not (ra[1] and rb[1]):
                continue
            moved = abs(ra[0].top - rb[0].top) > 2
            sa = _signatures(frames[a["index"]]["rgb"](), ra[0])
            sb = _signatures(frames[b["index"]]["rgb"](), rb[0])
            groups["scrolled" if moved else "same place"].append((sa, sb))
    for name, pairs in groups.items():
        if not pairs:
            out.append(f"  {name}: no pairs")
            continue
        sig = sum(a[0] == b[0] for a, b in pairs)
        # identify_slots keys a miss again on the bar box kept for its look
        again = sum(
            a[0] == b[0] or (a[4] is not None and slot_signature(b[3], a[4]) == a[0])
            for a, b in pairs
        )
        look = sum(a[1] == b[1] for a, b in pairs)
        size = sum(a[2] != b[2] for a, b in pairs)
        diffs = [
            float(
                np.abs(
                    stash_scan._unmarked(a[3]).astype(np.int16)
                    - stash_scan._unmarked(b[3]).astype(np.int16)
                ).mean()
            )
            for a, b in pairs
            if a[2] == b[2]
        ]
        out.append(
            f"  {name}: {len(pairs)} cell pairs, slot_signature equal {sig} "
            f"({100 * sig / len(pairs):.0f}%), equal with the kept bar box "
            f"{again} ({100 * again / len(pairs):.0f}%), look_signature equal "
            f"{look} ({100 * look / len(pairs):.0f}%), crop size differs {size}, "
            f"mean abs pixel difference of the signed area (same size pairs) "
            f"median {_median(diffs):.1f}, max {max(diffs, default=0):.1f}"
        )
    # Empty tiles carry no icon and no badge: the same tile drawn at two
    # screen places shows whether the tile's pixels depend on where it is
    tiles = []
    for n in frames.info:
        f = frames[n]
        rgb = None
        for slot in f["whole"]:
            if slot.filled:
                continue
            rgb = f["rgb"]() if rgb is None else rgb
            r = slot.rect
            tiles.append(
                (r.left, r.top, n, rgb[r.top : r.top + 194, r.left : r.left + 194])
            )
    moved, flips = [], []
    for i, a in enumerate(tiles):
        for b in tiles[i + 1 :]:
            if a[0] != b[0] or a[3].shape != b[3].shape or abs(a[1] - b[1]) < 4:
                continue
            ka = stash_scan._unmarked(a[3]) >> stash_scan.SIG_SHIFT
            kb = stash_scan._unmarked(b[3]) >> stash_scan.SIG_SHIFT
            moved.append(float(np.abs(a[3].astype(np.int16) - b[3]).mean()))
            flips.append(float((ka != kb).mean()))
    out.extend(
        (
            f"  empty tiles of one column drawn at two screen places ({len(moved)} pairs, {len(tiles)} empty whole slots): mean abs pixel difference median {_median(moved):.2f}, min {min(moved, default=0):.2f}, share of pixels whose signature value (pixel >> {stash_scan.SIG_SHIFT}) changes median {100 * _median(flips):.0f}%, min {100 * min(flips, default=0):.0f}%",
            "",
        )
    )


# --- the replays -------------------------------------------------------------


def thumb_top(rgb: np.ndarray, right: int) -> int | None:
    """The scrollbar thumb's top, None when no run of the thumb's length is
    seen right of the grid's right edge right."""
    scale = frame_scale(rgb)
    x0 = right + round(THUMB_DX[0] * scale)
    x1 = min(rgb.shape[1], right + round(THUMB_DX[1] * scale))
    lit = rgb[:, x0:x1].max(axis=2) > THUMB_LIT
    best = None
    for column in lit.T:
        edges = np.flatnonzero(
            np.diff(np.concatenate(([0], column.astype(np.int8), [0])))
        )
        for start, stop in zip(edges[::2], edges[1::2], strict=True):
            length = stop - start
            if THUMB_LEN[0] * scale <= length <= THUMB_LEN[1] * scale and (
                best is None or length > best[1]
            ):
                best = (int(start), int(length))
    return best[0] if best else None


class Frames:
    """The captures with their slots, identification and thumb, loaded lazily."""

    def __init__(self, paths: list[Path], every: list[dict]):
        self.paths = paths
        self.info: dict[int, dict] = {}
        wholes = {}
        for n, path in enumerate(paths):
            stash = [s for s in find_slots(_load(path)) if s.panel == STASH]
            wholes[n] = whole_slots(stash)
        # The grid's right edge, the same on every frame of one view (a
        # window over the last column must not move the thumb's search)
        self.right = round(
            statistics.median(
                max(s.rect.right for s in w) for w in wholes.values() if w
            )
        )
        for n, path in enumerate(paths):
            self.info[n] = self.entry(
                n, _load(path), every[n]["items"], wholes[n], lambda p=path: _load(p)
            )
        self.pitch = None
        self.left = None

    @classmethod
    def empty(cls, names: list[str], right: int) -> "Frames":
        """A set filled frame by frame with entry (the video's)."""
        frames = cls.__new__(cls)
        frames.paths = names
        frames.info = {}
        frames.right = right
        frames.pitch = None
        frames.left = None
        return frames

    def entry(self, n: int, rgb: np.ndarray, items: list, whole, loader) -> dict:
        identified = sum(i.item is not None for i in items)
        return {
            "items": items,
            "identified": identified,
            "unidentified": len(items) - identified,
            "whole": whole,
            "top": min(s.rect.top for s in whole) if whole else None,
            "thumb": thumb_top(rgb, self.right),
            "rest": top_at_rest(rgb, whole) if whole else None,
            "rgb": loader,
            "keys": (lambda origin, w=whole, n=n: self.keys(n, origin, w)),
        }

    def keys(self, n: int, origin: float, whole=None) -> dict:
        """The (row, column) cell of every whole slot of frame n with row 0 at
        origin, with its rect and fill."""
        out = {}
        for s in self.info[n]["whole"] if whole is None else whole:
            exact = (s.rect.top - origin) / self.pitch
            row = round(exact)
            if abs(exact - row) > ROW_RESIDUAL:
                continue
            key = (row, round((s.rect.left - self.left) / self.pitch))
            out.setdefault(key, (s.rect, s.filled))
        return out

    def __getitem__(self, n: int) -> dict:
        return self.info[n]


_READ = object()


def observe_one(
    store: StashMapStore, rgb: np.ndarray, items: list, header=_READ
) -> dict:
    """One observe as StashLedger._observe makes it, with _place's calls.
    header is read here unless given (None is a header that did not read)."""
    before_map = store.map
    before_seen = set(store._seen)
    before_last = store._last
    if header is _READ:
        stash = [s for s in find_slots(rgb) if s.panel == STASH]
        header = read_stash_header(rgb, stash)
    PROBE.reset()
    PROBE.on = True
    PROBE.recording = True
    seen = store.observe(rgb, [s for s in items if s.panel == STASH], header)
    PROBE.on = False
    PROBE.recording = False
    after = store.map
    emptied = 0
    if before_map is not None and after is not None:
        for key in seen.changed:
            old = before_map.cells.get(key)
            if old is not None and old.filled and not after.cells[key].filled:
                emptied += 1
    return {
        "how": seen.how,
        "why": seen.why,
        "origin": seen.origin,
        "rows": seen.rows,
        "changed": len(seen.changed),
        "emptied": emptied,
        "filled": sum(c.filled for c in after.cells.values()) if after else 0,
        "header": header,
        "events": PROBE.events,
        "before_map": before_map,
        "before_seen": before_seen,
        "before_last": before_last,
        "changed_keys": seen.changed,
    }


def replay(
    env: Env,
    frames: Frames,
    order: list[int],
    label: str,
    labels=None,
    *,
    keep_map: bool = False,
) -> dict:
    """Observe the frames in order on a fresh store, as StashLedger._observe
    does, from an empty map or (keep_map) the copied database's."""
    store = StashMapStore(env.database(keep_cache=False, keep_map=keep_map))
    rows = []
    for k, n in enumerate(order):
        row = observe_one(store, frames[n]["rgb"](), frames[n]["items"])
        row["index"] = n
        row["live_at"] = labels[k] if labels else ""
        rows.append(row)
    return {"label": label, "rows": rows, "store": store}


def fit_thumb(frames: Frames, runs: list[dict]) -> tuple[float, float]:
    """The thumb's top at rest and its px per px scrolled, from placed frames."""
    rest = [
        frames[r["index"]]["thumb"]
        for run in runs
        for r in run["rows"]
        if r["how"] == FIRST and frames[r["index"]]["thumb"] is not None
    ]
    thumb_rest = float(statistics.median(rest))
    ratios = []
    for run in runs:
        for r in run["rows"]:
            t = frames[r["index"]]["thumb"]
            if r["how"] in {SCROLL, LOOKS} and t is not None and t - thumb_rest > 50:
                ratios.append((t - thumb_rest) / (ROW_0_REST - r["origin"]))
    return thumb_rest, float(statistics.median(ratios))


def truth_origin(frames: Frames, n: int, fit: tuple[float, float]) -> float | None:
    t = frames[n]["thumb"]
    if t is None:
        return None
    return ROW_0_REST - (t - fit[0]) / fit[1]


def proposed(r: dict) -> float | None:
    """Row 0's frame y as _place found it before its checks, from its calls."""
    events = r["events"]
    last = r["before_last"]

    def results(label):
        return [e[2] for e in events if e[0] == label]

    scroll, checked = results("place scroll_offset"), results("place offset_checked")
    looks = results("place look_offset")
    if scroll and scroll[0] is not None and checked and checked[0]:
        return last.origin + scroll[0]
    if looks and looks[0] is not None:
        return float(looks[0].offset)
    if len(looks) > 1 and looks[1] is not None:
        return last.origin + looks[1].offset
    return None


def annotate(run: dict, frames: Frames, fit) -> None:
    last = None
    for r in run["rows"]:
        n = r["index"]
        truth = truth_origin(frames, n, fit)
        r["truth"] = truth
        top = frames[n]["top"]
        r["truth_rows"] = None
        if truth is not None and top is not None:
            first = round((top - truth) / frames.pitch)
            rows_seen = {k[0] for k in frames.keys(n, truth, frames[n]["whole"])}
            r["truth_rows"] = (min(rows_seen), max(rows_seen)) if rows_seen else None
            r["truth_first"] = first
        r["err"] = None
        if r["origin"] is not None and truth is not None:
            r["err"] = (truth - r["origin"]) / frames.pitch
        r["prop"] = proposed(r)
        r["prop_err"] = None
        if r["how"] not in PLACED and r["prop"] is not None and truth is not None:
            r["prop_err"] = (truth - r["prop"]) / frames.pitch
        r["dist"] = None
        if last is not None and truth is not None and last["truth"] is not None:
            r["dist"] = (last["truth"] - truth) / frames.pitch
        if r["how"] in PLACED:
            last = r


def _range(rows) -> str:
    return f"{rows[0]:>3d}-{rows[1]:<3d}" if rows else "   -   "


def table(
    run: dict, frames: Frames, paths, out: list[str], *, skip_lead: bool = False
) -> None:
    out.append(
        "  #  capture     live observe  how      placed  true     err  prop  dist  "
        "filled chg emp  thumb  why"
    )
    lead = 0
    if skip_lead:
        while lead < len(run["rows"]) and run["rows"][lead]["how"] in {
            stash_map.NO_SLOTS,
            stash_map.OTHER_VIEW,
            stash_map.NO_TOP,
            stash_map.NO_HEADER,
        }:
            lead += 1
        out.append(
            f"  (#0 to #{lead - 1}: before the All tab of the stash is open, "
            + ", ".join(sorted({r["how"] for r in run["rows"][:lead]}))
            + ")"
        )
    for k, r in enumerate(run["rows"]):
        if k < lead:
            continue
        n = r["index"]
        thumb = frames[n]["thumb"]
        out.append(
            f"  {k:2d} {_short(paths[n])}  {r['live_at']:12s}  "
            f"{r['how']:8s} {_range(r['rows'])} {_range(r.get('truth_rows'))} "
            f"{_num(r['err']):>5s} {_num(r['prop_err']):>5s} {_num(r['dist']):>5s}  "
            f"{r['filled']:4d} {r['changed']:3d} {r['emptied']:3d}  "
            f"{thumb if thumb is not None else '-':>5}  {r['why']}"
        )


def summary(run: dict, out: list[str]) -> None:
    rows = run["rows"]
    placed = [r for r in rows if r["how"] in PLACED]
    wrong = [
        (k, r)
        for k, r in enumerate(rows)
        if r["how"] in PLACED and r["err"] is not None and abs(r["err"]) >= 0.5
    ]
    hows = defaultdict(int)
    for r in rows:
        hows[r["how"]] += 1
    store = run["store"]
    stored = store.map
    filled = sum(c.filled for c in stored.cells.values()) if stored else 0
    cells = len(stored.cells) if stored else 0
    found_rows = sorted({k[0] for k in stored.cells}) if stored else []
    filled_rows = sorted({k[0] for k, c in stored.cells.items() if c.filled})
    missing = [
        r
        for r in range((found_rows[-1] + 1) if found_rows else 0)
        if r not in found_rows
    ]
    out.extend(
        (
            f"  frames {len(rows)}, placed {len(placed)} ("
            + ", ".join(f"{k} {v}" for k, v in sorted(hows.items()))
            + f"), wrong places {len(wrong)}"
            + (
                " (" + ", ".join(f"#{k} err {r['err']:+.1f}" for k, r in wrong) + ")"
                if wrong
                else ""
            ),
            f"  map: {cells} cells, {filled} filled, rows {found_rows[0] if found_rows else '-'} to {found_rows[-1] if found_rows else '-'}, filled rows to {filled_rows[-1] if filled_rows else '-'}, rows missing {_spans(missing) or 'none'}",
        )
    )


def _spans(values: list[int]) -> str:
    spans = []
    for v in values:
        if spans and v == spans[-1][1] + 1:
            spans[-1][1] = v
        else:
            spans.append([v, v])
    return ", ".join(f"{a}" if a == b else f"{a} to {b}" for a, b in spans)


# --- the wrong place ---------------------------------------------------------


def _num(value, spec: str = "+.1f") -> str:
    return "-" if value is None else format(value, spec)


def _name(slot) -> str:
    if slot is None:
        return "-"
    return slot.item.name if slot.item is not None else f"?{slot.guess}"


def emptied_seen(frames: Frames, r: dict) -> tuple[int, int]:
    """Whole empty slots of a placed frame on cells seen filled this session,
    and whole filled slots on cells seen empty."""
    stored = r["before_map"]
    if stored is None or r["origin"] is None:
        return 0, 0
    keys = frames.keys(r["index"], r["origin"], frames[r["index"]]["whole"])
    empty_on_filled = filled_on_empty = 0
    for key, (_, filled) in keys.items():
        cell = stored.cells.get(key)
        if cell is None or key not in r["before_seen"]:
            continue
        if cell.filled and not filled:
            empty_on_filled += 1
        if not cell.filled and filled:
            filled_on_empty += 1
    return empty_on_filled, filled_on_empty


def wrong_section(runs: dict, frames: Frames, paths, observes, out: list[str]) -> None:
    out.extend(("== 5. The wrong place (question 3) ==", ""))
    target = next(
        o for o in observes if o["at"].strftime("%H:%M:%S.%f")[:12] == WRONG_AT
    )
    frame_at = datetime.fromtimestamp(target["frame_at"], LOCAL)
    out.append(
        f"Observe {WRONG_AT}: pass end "
        f"{target['pass']['at'].strftime('%H:%M:%S.%f')[:12]}, duration "
        f"{target['pass']['ms']} ms, frame time "
        f"{frame_at.strftime('%H:%M:%S.%f')[:12]}, nearest capture "
        f"{_short(paths[target['capture']])} ({target['dt']:+.2f} s), latest capture "
        f"before it {_short(paths[target['before']])} ({target['dt_before']:+.2f} s), "
        f"live changed={target['changed']} filled={target['filled']}"
    )
    for name in ("live", "before", "every"):
        run = runs[name]
        n = target["capture"] if name != "before" else target["before"]
        if name == "every":
            found = [(k, r) for k, r in enumerate(run["rows"]) if r["index"] == n]
        else:
            found = [
                (k, r) for k, r in enumerate(run["rows"]) if r["live_at"] == WRONG_AT
            ]
        if not found:
            out.append(f"  {name} replay: the frame is not in it")
            continue
        k, r = found[0]
        out.append(
            f"  {name} replay frame #{k} {_short(paths[n])}: how {r['how']}, placed "
            f"rows {_range(r['rows'])}, true rows {_range(r['truth_rows'])}, err "
            f"{_num(r['err'])} rows, changed {r['changed']}, emptied {r['emptied']}, "
            f"filled after {r['filled']}{', ' + r['why'] if r['why'] else ''}"
        )
    found = [
        (k, r) for k, r in enumerate(runs["before"]["rows"]) if r["live_at"] == WRONG_AT
    ]
    if found:
        reconstruct(runs["before"], found[0][0], frames, paths, out)
    out.append("")
    rule_section(runs, frames, paths, out)


def _probes(r: dict):
    for label, args, _ in r["events"]:
        if label in {"place offset_checked", "place look_offset"}:
            return args[1]
    return None


def clashes(r: dict, frames: Frames, origin: float, keys=None) -> tuple:
    """Pairs of the frame's looks and stored cells at origin: matches, clashes,
    and clashes where the frame slot and the cell name the same item."""
    stored = r["before_map"]
    pitch, left = frames.pitch, frames.left
    probes = _probes(r)
    cells = stash_map._stored_observations(
        stored, pitch, left, r["before_seen"] if keys is None else keys
    )
    items = {s.rect: s for s in frames[r["index"]]["items"]}
    tol = LOOK_TOLERANCE
    match = clash = same = 0
    detail = []
    for b in probes or ():
        if b.look is None:
            continue
        for a in cells:
            if abs(a.rect.left - b.rect.left) >= pitch / 2:
                continue
            if abs((b.rect.top - a.rect.top) - origin) > tol:
                continue
            d = float(look_distance(b.look, a.look))
            key = (round(a.rect.top / pitch), round((a.rect.left - left) / pitch))
            cell = stored.cells[key]
            slot = items.get(b.rect)
            name = slot.item.name if slot is not None and slot.item else None
            if d < LOOK_SAME:
                match += 1
            elif d > LOOK_DIFFERENT:
                clash += 1
                same += name is not None and name == cell.name
            detail.append((b.rect.top, key, _name(slot), cell.name, d))
    return match, clash, same, detail


def refusal_section(runs: dict, frames: Frames, paths, out: list[str]) -> None:
    """The frames refused at their true place: what clashed."""
    out.append(
        "Frames whose proposed place was the true one (prop 0) but were refused "
        "by the cells seen this session: at that place, look pairs matching "
        f"(under {LOOK_SAME}), clashing (over {LOOK_DIFFERENT}) and clashing while "
        "the frame's slot and the stored cell name the same item:"
    )
    for name in runs:
        for k, r in enumerate(runs[name]["rows"]):
            if r["prop_err"] is None or abs(r["prop_err"]) >= 0.5:
                continue
            match, clash, same, _ = clashes(r, frames, r["prop"])
            out.append(
                f"  {name:6s} #{k:2d} {_short(paths[r['index']])}: matching {match}, "
                f"clashing {clash}, clashing with the same name {same}  ({r['why']})"
            )
    out.append("")


def reconstruct(run: dict, k: int, frames: Frames, paths, out: list[str]) -> None:
    """Where the live run placed a frame the replay does not place, from the
    copied database's final cells, and how the checks judge that place."""
    r = run["rows"][k]
    n = r["index"]
    f = frames[n]
    pitch, left = frames.pitch, frames.left
    final = frames.final_map
    rgb = f["rgb"]()
    obs = slot_observations(0, rgb, f["whole"], {})
    ranked = []
    for row in range(-2, 80):
        origin = f["top"] - row * pitch
        exact = total = empty = 0
        for o in obs:
            key = (
                round((o.rect.top - origin) / pitch),
                round((o.rect.left - left) / pitch),
            )
            cell = final.cells.get(key)
            if cell is None:
                continue
            if o.look is None:
                empty += not o.filled and not cell.filled
                continue
            if cell.look is None:
                continue
            total += 1
            exact += float(look_distance(o.look, cell.look)) < 1.0
        ranked.append((exact + empty, exact, empty, total, row))
    ranked.sort(reverse=True)
    best = ranked[0]
    origin = f["top"] - best[4] * pitch
    out.extend(
        (
            "",
            f"The live place of {_short(paths[n])}, read back from the copied database's final cells: with its top whole row on row {best[4]} (row 0 at frame y {origin:.0f}), {best[1]} of its {best[3]} textured slots equal a stored look (distance under 1.0, written by this frame) and {best[2]} empty slots sit on stored empty cells. Next best: "
            + ", ".join(f"row {x[4]} ({x[1]} + {x[2]})" for x in ranked[1:3]),
        )
    )
    probes = _probes(r)
    stored_all = stash_map._stored_observations(r["before_map"], pitch, left)
    tops, diff = look_pairs(stored_all, probes, pitch)
    tol = LOOK_TOLERANCE * frame_scale(rgb)
    judged = judge_offset(tops, diff, round(origin), tol)
    out.append(
        f"  replay state just before it (#{k}): judge_offset of that place against "
        f"the stored cells: {judged.support} for, {judged.contra} against, "
        f"shift_wrong {shift_wrong(judged)}"
    )
    proposals = sorted({int(t) for t in tops[diff < LOOK_SAME]})
    scored = []
    for p in proposals:
        j = judge_offset(tops, diff, p, tol)
        scored.append((j.support - j.contra, j.support, j.contra, j.offset, p))
    scored.sort(reverse=True)
    out.append(
        "  look_offset's proposals to the stored cells by score (support less "
        "contra), as offset px (top whole row): "
        + ", ".join(
            f"{x[3]} (row {round((f['top'] - x[3]) / pitch)}) {x[1]}-{x[2]}"
            for x in scored[:6]
        )
    )
    match, clash, _same, _detail = clashes(r, frames, origin)
    out.append(
        f"  against the cells seen this session at that place: {match} matching, "
        f"{clash} clashing (shift_wrong needs 2 and a third of the support)"
    )
    _match_all, _clash_all, _, detail_all = clashes(
        r, frames, origin, keys=set(r["before_map"].cells)
    )
    out.append("  every same-column pair at that place (frame slot -> stored cell):")
    out.extend(
        f"    frame y {top:4d} {fname:26s} -> cell {key!s:9s} {cname or '?':26s} "
        f"{d:5.1f}{' match' if d < LOOK_SAME else (' clash' if d > LOOK_DIFFERENT else '')}"
        for top, key, fname, cname, d in detail_all
    )
    probe = dict(r, origin=origin)
    e_f, f_e = emptied_seen(frames, probe)
    out.append(
        f"  whole empty slots of the frame on cells seen filled this session: {e_f}, "
        f"whole filled slots on cells seen empty: {f_e}. The empty slots have flat "
        f"looks, which look_pairs leaves out, so none of them counts against"
    )
    refused = shift_wrong(Shift(0, max(match, judged.support), clash + e_f))
    out.append(
        f"  the rule below at that place: contra {clash} + {e_f} against support "
        f"{max(match, judged.support)}: refused {'yes' if refused else 'no'}"
    )
    if r.get("truth_first") is not None:
        stash_items = sorted(
            (s for s in f["items"] if s.panel == STASH),
            key=lambda s: (s.rect.top, s.rect.left),
        )
        out.append(
            f"  where it belongs: top whole row {r['truth_first']} by the thumb "
            f"(thumb top {f['thumb']}), rows {_range(r['truth_rows'])}. The stash "
            f"has 272 used of 304, so rows 0 to 67 are filled and 68 to 75 empty. "
            f"Its identified slots, row by row: "
            + ", ".join(_name(x) for x in stash_items)
        )
    before = r["before_map"]
    rows = [
        f"row {row}: "
        + ", ".join(
            (before.cells[row, c].name or "?")
            if (row, c) in before.cells and before.cells[row, c].filled
            else ("empty" if (row, c) in before.cells else "none")
            for c in range(4)
        )
        for row in range(best[4], best[4] + 6)
    ]
    out.append(
        "  the cells it overwrote live, as the replay had them: " + " | ".join(rows)
    )


def rule_section(runs: dict, frames: Frames, paths, out: list[str]) -> None:
    out.extend(
        [
            (
                "Rule A: a whole empty slot of the frame landing on a cell seen "
                "filled this session counts as a contradiction, beside the textured "
                "clashes. The place is refused when shift_wrong(support, contra + "
                "empty_on_filled), i.e. that sum >= 2 and > support / 3 (support as "
                "_place takes it, contra from its judge_offset against the cells seen "
                "this session). Rule B: a place that puts a row of the frame above "
                "row 0 (the stash's first row) is refused. Checked on every SCROLL "
                "and LOOKS placement. correct: within half a row of the thumb's "
                "place, ? where the thumb is not seen."
            ),
            (
                "  run     #  capture     how     err   support contra emp_on_filled  "
                "rows     A    B    correct"
            ),
        ]
    )
    counts = defaultdict(int)
    for name in runs:
        for k, r in enumerate(runs[name]["rows"]):
            if r["how"] not in {SCROLL, LOOKS}:
                continue
            judged = [e[2] for e in r["events"] if e[0] == "place judge_offset"]
            support = max((j.support for j in judged), default=0)
            for e in r["events"]:
                if e[0] == "place look_offset" and e[2] is not None:
                    support = max(support, e[2].support)
                    break
            # The first judge_offset of _place is against the cells seen this
            # session, the one that can refuse
            contra = judged[0].contra if judged else 0
            e_f, _ = emptied_seen(frames, r)
            rule_a = shift_wrong(Shift(0, support, contra + e_f))
            rule_b = r["rows"] is not None and r["rows"][0] < 0
            unknown = r["err"] is None
            correct = not unknown and abs(r["err"]) < 0.5
            kind = "?" if unknown else ("correct" if correct else "wrong")
            counts[kind] += 1
            counts[kind, "A"] += rule_a
            counts[kind, "B"] += rule_b
            counts[kind, "A or B"] += rule_a or rule_b
            out.append(
                f"  {name:6s} {k:3d}  {_short(paths[r['index']])}  {r['how']:7s} "
                f"{_num(r['err']):>5s}  {support:7d} {contra:6d} {e_f:13d}  "
                f"{_range(r['rows'])}  {'yes' if rule_a else 'no':4s} "
                f"{'yes' if rule_b else 'no':4s} "
                f"{'?' if unknown else ('yes' if correct else 'NO')}"
            )
    out.extend(
        (
            "  refused, of the correct and of the wrong placements: "
            + ", ".join(
                f"rule {rule} {counts['correct', rule]} of {counts['correct']} "
                f"correct and {counts['wrong', rule]} of {counts['wrong']} wrong"
                for rule in ("A", "B", "A or B")
            )
            + f" ({counts['?']} placements without a thumb)",
            "",
        )
    )


# --- the video ---------------------------------------------------------------

# The part of the frame the timeline decodes, at half size: the stash grid
# and its scrollbar (x 280 to 1200, y 480 to 2010 of 3840x2160)
VIDEO_BOX = (280, 480, 1200, 2010)
TIMELINE_FPS = 60
REPLAY_FPS = 5
# A still: the thumb stays within STILL_PX for at least STILL_S
STILL_PX = 2
STILL_S = 0.5
# A slot carries a price patch when PATCH_SHARE of its patch zone (bar rows
# and left half, where badge_helper draws the patch) is bright and coloured
PATCH_ZONE = (0.74, 0.95, 0.04, 0.55)
PATCH_LIT = 150
PATCH_CHROMA = 80
PATCH_SHARE = 0.25
# A rise of this many patched slots is new badges
BADGE_RISE = 3
BADGE_EVENTS = re.compile(
    LOG_TIME + r".*(stash known slots drawn|stash scrolled in slots drawn|"
    r"stash changed slots redrawn|stash badges cleared|stash pass identified|"
    r"stash pass dropped|stash badges followed)(.*)"
)


def _ffmpeg(video: Path, args: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        ["ffmpeg", "-v", "error", *args[:-1], "-i", str(video), *args[-1].split()],
        stdout=subprocess.PIPE,
    )


def video_stream(video: Path, fps: int, *, half: bool, start: float = 0.0, length=None):
    """(t, rgb) at fps from start, the whole frame or VIDEO_BOX at half size."""
    x0, y0, x1, y1 = VIDEO_BOX
    if half:
        w, h = (x1 - x0) // 2, (y1 - y0) // 2
        vf = f"fps={fps},crop={x1 - x0}:{y1 - y0}:{x0}:{y0},scale={w}:{h}"
    else:
        w, h = 3840, 2160
        vf = f"fps={fps}"
    pre = ["-ss", f"{start:.3f}"] + (["-t", f"{length:.3f}"] if length else [])
    proc = _ffmpeg(video, [*pre, f"-vf {vf} -f rawvideo -pix_fmt rgb24 -"])
    size = w * h * 3
    n = 0
    try:
        while True:
            data = proc.stdout.read(size)
            if len(data) < size:
                break
            yield start + n / fps, np.frombuffer(data, np.uint8).reshape(h, w, 3)
            n += 1
    finally:
        proc.stdout.close()
        proc.wait()


def video_frame(video: Path, t: float) -> np.ndarray:
    """The full frame at t s (an exact seek)."""
    data = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{t:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(data, np.uint8).reshape(2160, 3840, 3)


def half_thumb(half: np.ndarray, right: int) -> int | None:
    """thumb_top on a half size VIDEO_BOX frame, in full frame px."""
    x0, y0 = VIDEO_BOX[0], VIDEO_BOX[1]
    lo = (right + THUMB_DX[0] - x0) // 2
    hi = min(half.shape[1], (right + THUMB_DX[1] - x0) // 2)
    lit = half[:, lo:hi].max(axis=2) > THUMB_LIT
    best = None
    for column in lit.T:
        edges = np.flatnonzero(
            np.diff(np.concatenate(([0], column.astype(np.int8), [0])))
        )
        for start, stop in zip(edges[::2], edges[1::2], strict=True):
            length = stop - start
            if THUMB_LEN[0] / 2 <= length <= THUMB_LEN[1] / 2 + 1 and (
                best is None or length > best[1]
            ):
                best = (int(start), int(length))
    return None if best is None else best[0] * 2 + y0


def stills_of(timeline: list[tuple[float, int | None]]) -> list[dict]:
    """Runs where the thumb is seen and stays within STILL_PX for STILL_S."""
    stills = []
    run = []
    for t, thumb in [*timeline, (timeline[-1][0] + 1, None)]:
        if thumb is not None and run and abs(thumb - run[0][1]) <= STILL_PX:
            run.append((t, thumb))
            continue
        if run and run[-1][0] - run[0][0] >= STILL_S:
            stills.append(
                {
                    "start": run[0][0],
                    "end": run[-1][0],
                    "thumb": int(statistics.median(x[1] for x in run)),
                }
            )
        run = [(t, thumb)] if thumb is not None else []
    return stills


def patched(half: np.ndarray, rects) -> int:
    """Slots of a half size VIDEO_BOX frame whose patch zone shows a patch."""
    x0, y0 = VIDEO_BOX[0], VIDEO_BOX[1]
    a, b, c, d = PATCH_ZONE
    count = 0
    for r in rects:
        h, w = r.bottom - r.top, r.right - r.left
        zone = half[
            (r.top + int(a * h) - y0) // 2 : (r.top + int(b * h) - y0) // 2,
            (r.left + int(c * w) - x0) // 2 : (r.left + int(d * w) - x0) // 2,
        ].astype(np.int16)
        top, low = zone.max(axis=2), zone.min(axis=2)
        count += (
            (top >= PATCH_LIT) & (top - low >= PATCH_CHROMA)
        ).mean() >= PATCH_SHARE
    return count


EVENT_NAMES = {
    "stash known slots drawn": "known drawn",
    "stash scrolled in slots drawn": "scrolled-in drawn",
    "stash changed slots redrawn": "redrawn",
    "stash badges cleared": "cleared",
    "stash pass identified": "pass end",
    "stash pass dropped": "stale",
    "stash badges followed": "followed",
}


def badge_events(log: Path, t0: float) -> list[tuple[float, str]]:
    """The badge and pass lines of a log as (video t, short name)."""
    return [
        (_time(m.group(1)).timestamp() - t0, EVENT_NAMES[m.group(2)])
        for line in log.read_text(encoding="utf-8").splitlines()
        if (m := BADGE_EVENTS.match(line))
    ]


def video_badges(video: Path, stills, events, out: list[str]) -> None:
    """How long after the grid stops the badges appear, still by still."""
    for still in stills:
        rgb = video_frame(video, min(still["end"], still["start"] + 0.2))
        whole = whole_slots([s for s in find_slots(rgb) if s.panel == STASH])
        still["rects"] = [s.rect for s in whole if s.filled]
        still["counts"] = []
    k = 0
    for t, half in video_stream(video, TIMELINE_FPS, half=True):
        while k < len(stills) and t > stills[k]["end"]:
            k += 1
        if k < len(stills) and t >= stills[k]["start"]:
            stills[k]["counts"].append((t, patched(half, stills[k]["rects"])))
    out.extend(
        (
            (
                "Stills of the grid (the thumb within "
                f"{STILL_PX} px for {STILL_S} s or more, at {TIMELINE_FPS} fps). "
                "patched counts the filled whole slots showing a price patch. rise is "
                f"the first frame with {BADGE_RISE} more patched slots than the still's "
                "first frame, most the first frame at the still's highest count, both in s after the grid stopped. log lists the badge and pass lines in the still (s after the stop)."
            ),
            "  stop (t)   end (t)  thumb  slots  patched first/max   rise   most  log",
        )
    )
    rises, mosts = [], []
    for still in stills:
        counts = still["counts"]
        if not counts:
            continue
        first = counts[0][1]
        top = max(c for _, c in counts)
        rise = next((t for t, c in counts if c >= first + BADGE_RISE), None)
        most = next(t for t, c in counts if c == top)
        rise = None if rise is None else rise - still["start"]
        most = most - still["start"] if top > first else None
        if rise is not None:
            rises.append(rise)
        if most is not None and top >= first + BADGE_RISE:
            mosts.append(most)
        logged = [
            f"{t - still['start']:+.2f} {text}"
            for t, text in events
            if still["start"] - 0.05 <= t <= still["end"]
        ]
        out.append(
            f"  {still['start']:8.2f}  {still['end']:8.2f}  {still['thumb']:5d}  "
            f"{len(still['rects']):5d}  {first:7d}/{top:<3d}        "
            f"{_num(rise, '5.2f'):>5s}  {_num(most, '5.2f'):>5s}  " + ", ".join(logged)
        )
    out.extend(
        (
            f"  badges rising after a stop: {len(rises)} of {len(stills)} stills, rise median {_median(rises):.2f} s (min {min(rises, default=0):.2f}, max {max(rises, default=0):.2f}), most median {_median(mosts):.2f} s (min {min(mosts, default=0):.2f}, max {max(mosts, default=0):.2f})",
            "",
        )
    )


def video_section(args, env: Env, out: list[str]) -> None:
    """Questions 1 to 3 on Jeffrey's screen video of the second scroll."""
    video = args.video
    t0 = (
        datetime.strptime(args.video_start, "%Y-%m-%d %H:%M:%S")
        .replace(tzinfo=LOCAL)
        .timestamp()
    )
    out.extend(
        [
            "",
            "== 6. The video of the second scroll (the primary data set) ==",
            "",
            (
                f"video {video.name}, t 0 at {args.video_start}, log {args.video_log.name}, "
                f"db after it {args.video_db.name}. The session starts from the map of "
                f"{args.db.name} (the db after the first scroll): its first observe says "
                "filled=80."
            ),
            "",
        ]
    )
    observes = parse_log(args.video_log)
    for o in observes:
        o["frame_at"] = o["pass"]["at"].timestamp() - o["pass"]["ms"] / 1000
        o["t"] = o["frame_at"] - t0
    durations = [o["pass"]["ms"] for o in observes]
    placed_live = sum(o["placed"] == "yes" for o in observes)
    out.extend(
        (
            f"log: {len(observes)} observe lines ({placed_live} placed=yes), filled in "
            "order: " + ", ".join(str(o["filled"]) for o in observes),
            f"pass durations median {_median(durations):.0f} ms, min {min(durations)}, max {max(durations)}, dropped stale {sum(o['stale'] for o in observes)}",
        )
    )
    right = 1120
    timeline = [
        (t, half_thumb(half, right))
        for t, half in video_stream(video, TIMELINE_FPS, half=True)
    ]
    stills = stills_of(timeline)
    seen = [t for t, thumb in timeline if thumb is not None]
    out.append(
        f"timeline: {len(timeline)} frames at {TIMELINE_FPS} fps, thumb seen from "
        f"t {seen[0]:.2f} to {seen[-1]:.2f} on {len(seen)} frames, {len(stills)} "
        "stills"
    )
    events = badge_events(args.video_log, t0)
    # Alignment: the first badges ARLO logs against the first patched frames
    first_known = next(t for t, text in events if text == "known drawn")
    first_pass = next(t for t, text in events if text == "pass end")
    out.extend(("", "-- Badges after the grid stops (question 1) --"))
    video_badges(video, stills, events, out)
    for label, at in (
        ("first known slots drawn", first_known),
        ("first pass end", first_pass),
    ):
        found = None
        for still in stills:
            prior = None
            for t, c in still.get("counts", []):
                if t >= at - 1.0 and prior is not None and c >= prior + BADGE_RISE:
                    found = t
                    break
                prior = c
            if found is not None:
                break
        out.append(
            f"  alignment check, {label}: log t {at:.2f}, the first patched rise "
            f"after it in the video at t {_num(found, '.2f')}, video minus log "
            f"{_num(None if found is None else found - at, '+.2f')} s"
        )
    out.append("")
    # The live observes on the video's frames (the ones after its end are
    # the inventory opened again after the recording stopped)
    length = timeline[-1][0]
    outside = [o for o in observes if not 0 <= o["t"] <= length]
    observes = [o for o in observes if 0 <= o["t"] <= length]
    durations = [o["pass"]["ms"] for o in observes]
    out.append(
        f"observe lines after the video's end (t {length:.2f}), left out: "
        f"{len(outside)} ("
        + ", ".join(o["at"].strftime("%H:%M:%S.%f")[:12] for o in outside)
        + ")"
    )
    names = [f"v{o['t']:06.2f}" for o in observes]
    frames = Frames.empty(names, right)
    session = Pass(env.database(keep_cache=False, keep_map=True))
    timing = []
    for n, o in enumerate(observes):
        rgb = video_frame(video, o["t"])
        result = session.run(rgb)
        timing.append(result)
        whole = whole_slots([s for s in find_slots(rgb) if s.panel == STASH])
        frames.info[n] = frames.entry(
            n, rgb, result["items"], whole, lambda t=o["t"]: video_frame(video, t)
        )
    session.identify.flush()
    out.extend(
        (
            "-- Pass time on the video's frames, empty slot cache (question 1) --",
            "  live at        frame     live  total  ident ledger badges  hits/filled  emb  hdrOCR loadOCR observe embed",
        )
    )
    for o, r, name in zip(observes, timing, names, strict=True):
        p = r["parts"]
        out.append(
            f"  {o['at'].strftime('%H:%M:%S.%f')[:12]}  {name}  {o['pass']['ms']:5d}  "
            f"{r['total']:5.0f}  {r['identify']:5.0f}  {r['ledger']:5.0f}  "
            f"{r['badges']:5.0f}  {r['hits']:3d}/{r['filled']:<3d}      "
            f"{r['embedded']:3d}  {p.get('ledger stash header OCR', 0):6.0f} "
            f"{p.get('ledger loadout header OCR', 0):7.0f} "
            f"{p.get('ledger observe', 0):7.0f} {p.get('embed', 0):5.0f}"
        )
    ratio = [
        o["pass"]["ms"] / r["total"] for o, r in zip(observes, timing, strict=True)
    ]
    out.append(
        f"  medians: live {_median(durations):.0f} ms, replay total "
        f"{_median(r['total'] for r in timing):.0f}, identify "
        f"{_median(r['identify'] for r in timing):.0f}, ledger "
        f"{_median(r['ledger'] for r in timing):.0f}, embedded "
        f"{_median(r['embedded'] for r in timing):.0f}, live / replay "
        f"{_median(ratio):.2f}"
    )
    final = Pass(env.database(keep_cache=True, keep_map=True, source=args.video_db))
    hits = [len(final.identify.peek(frames[n]["rgb"]())) for n in range(len(observes))]
    out.extend(
        (
            (
                f"  the session's final slot cache ({args.video_db.name}) on the video's frames: {sum(hits)} hits of {sum(r['filled'] for r in timing)} filled "
                "slots (the video is VP9, its pixels are not the capture's, so the lower bound comes from the capture set above)"
            ),
            "",
        )
    )
    # The replays: the live observes, then every frame at REPLAY_FPS
    live = replay(
        env,
        frames,
        list(range(len(observes))),
        "live",
        [o["at"].strftime("%H:%M:%S.%f")[:12] for o in observes],
        keep_map=True,
    )
    every_frames = Frames.empty([], right)
    stores = {
        "every": StashMapStore(env.database(keep_cache=False, keep_map=True)),
        "every-empty": StashMapStore(env.database(keep_cache=False)),
    }
    every = {
        name: {"label": name, "rows": [], "store": st} for name, st in stores.items()
    }
    prev_thumb = None
    for n, (t, rgb) in enumerate(video_stream(video, REPLAY_FPS, half=False)):
        every_frames.paths.append(f"v{t:06.2f}")
        stash = [s for s in find_slots(rgb) if s.panel == STASH]
        whole = whole_slots(stash)
        every_frames.info[n] = every_frames.entry(
            n, rgb, [], whole, lambda t=t: video_frame(video, t)
        )
        header = read_stash_header(rgb, stash) if stash else None
        thumb = every_frames[n]["thumb"]
        moving = (
            "?"
            if thumb is None or prev_thumb is None
            else ("still" if abs(thumb - prev_thumb) <= STILL_PX else "moving")
        )
        prev_thumb = thumb
        for name, store in stores.items():
            row = observe_one(store, rgb, [], header)
            row["index"] = n
            row["live_at"] = moving
            every[name]["rows"].append(row)
    runs = {"live": live, **every}
    for fs, rs in ((frames, [live]), (every_frames, list(every.values()))):
        stored = next(r["store"].map for r in rs if r["store"].map is not None)
        fs.pitch = stored.pitch * 2160
        fs.left = stored.column_left * 2160
    fit = fit_thumb(every_frames, list(every.values()))
    for fs, rs in ((frames, [live]), (every_frames, list(every.values()))):
        for run in rs:
            annotate(run, fs, fit)
    out.extend(
        [
            "-- Replays on the video (question 2) --",
            "",
            (
                f"Truth: thumb top at rest {fit[0]:.0f}, {fit[1]:.4f} thumb px per px "
                "scrolled (fitted on the every-frame runs). Columns as in section 4. "
                "In the every-frame tables the live column says whether the thumb moved "
                f"since the frame before ({REPLAY_FPS} fps)."
            ),
            "",
            (
                "-- The live observes, each on the video frame at its frame time, from "
                f"the map of {args.db.name} --"
            ),
        ]
    )
    table(live, frames, frames.paths, out)
    summary(live, out)
    same = sum(
        o["filled"] == r["filled"] for o, r in zip(observes, live["rows"], strict=True)
    )
    agree = sum(
        (o["placed"] == "yes") == (r["how"] in PLACED)
        for o, r in zip(observes, live["rows"], strict=True)
    )
    out.extend(
        (
            f"  against the live log: placed yes/no equal on {agree} of {len(observes)} lines, filled count equal on {same} of {len(observes)}",
            "",
        )
    )
    for name, title in (
        ("every", f"Every frame at {REPLAY_FPS} fps, from the map of {args.db.name}"),
        ("every-empty", f"Every frame at {REPLAY_FPS} fps, from an empty map"),
    ):
        out.append(f"-- {title} --")
        table(every[name], every_frames, every_frames.paths, out, skip_lead=True)
        summary(every[name], out)
        out.append("")
    gaps_rows = [
        r["dist"]
        for r in live["rows"]
        if r["how"] not in PLACED and r["dist"] is not None
    ]
    out.extend(
        (
            "live replay, unplaced frames' true distance from the last placed frame (rows): "
            + ", ".join(f"{d:+.1f}" for d in gaps_rows),
            "placed frames against the thumb's row 0: max |err| "
            + _num(
                max(
                    (
                        abs(r["err"])
                        for run in runs.values()
                        for r in run["rows"]
                        if r["how"] in PLACED and r["err"] is not None
                    ),
                    default=None,
                ),
                ".3f",
            )
            + " rows",
            "",
        )
    )
    rows_section(live, frames, out)
    wrong_video(every, every_frames, out)
    refusal_section(every, every_frames, every_frames.paths, out)
    out.append("-- The rule of section 5 on the video's placements --")
    rule_section({"live": live}, frames, frames.paths, out)
    rule_section(every, every_frames, every_frames.paths, out)


def wrong_video(every: dict, frames: Frames, out: list[str]) -> None:
    """The first placement of each every-frame run more than a row off."""
    for name, run in every.items():
        found = [
            (k, r)
            for k, r in enumerate(run["rows"])
            if r["how"] in PLACED and r["err"] is not None and abs(r["err"]) >= 1
        ]
        if not found:
            out.append(f"-- {name}: no placement a row or more off --")
            continue
        k, r = found[0]
        seen_rows = sorted({key[0] for key in r["before_seen"]})
        out.append(
            f"-- {name}: the first placement a row or more off, #{k} "
            f"{frames.paths[r['index']]} ({r['live_at']}): {r['how']}, placed rows "
            f"{_range(r['rows'])}, true {_range(r['truth_rows'])}, err "
            f"{_num(r['err'])}, {len(found)} such placements in the run. Cells seen "
            f"this session before it: rows {seen_rows[0]} to {seen_rows[-1]}. "
            f"Map before it: {len(r['before_map'].cells)} cells --"
        )
        for label, _args, result in r["events"]:
            if label.startswith("place "):
                out.append(f"  {label[6:]}: {result}")
        match, clash, _, detail = clashes(
            r, frames, r["origin"], keys=set(r["before_map"].cells)
        )
        out.append(
            f"  every same-column pair at that place against the stored cells: "
            f"{match} matching, {clash} clashing"
        )
        out.extend(
            f"    frame y {top:4d} -> cell {key!s:9s} {cname or '?':26s} {d:5.1f}"
            f"{' match' if d < LOOK_SAME else (' clash' if d > LOOK_DIFFERENT else '')}"
            f"{' (seen this session)' if key in r['before_seen'] else ''}"
            for top, key, _fname, cname, d in detail
        )
        out.append("")


def rows_section(live: dict, frames: Frames, out: list[str]) -> None:
    """The cells of rows 5 to 9 the live observes saw, before and after."""
    out.append(
        "-- Rows 5 to 9 in the live replay: the stored cell before each placed "
        "frame, the frame's slot there, their look distance and whether it changed --"
    )
    for k, r in enumerate(live["rows"]):
        if r["how"] not in PLACED or r["before_map"] is None:
            continue
        items = {s.rect: s for s in frames[r["index"]]["items"]}
        rgb = None
        lines = []
        for key, (rect, filled) in sorted(frames.keys(r["index"], r["origin"]).items()):
            if not 5 <= key[0] <= 9:
                continue
            old = r["before_map"].cells.get(key)
            rgb = frames[r["index"]]["rgb"]() if rgb is None else rgb
            look = slot_observations(0, rgb, [Slot(rect, STASH, filled)], {})[0].look
            d = (
                None
                if old is None or old.look is None or look is None
                else float(look_distance(old.look, look))
            )
            lines.append(
                f"    {key}: stored {(old.name or '?') if old and old.filled else ('empty' if old else 'none')}"
                f", frame {_name(items.get(rect)) if filled else 'empty'}, distance "
                f"{_num(d, '.1f')}, {'changed' if key in r['changed_keys'] else 'kept'}"
            )
        if lines:
            out.append(
                f"  #{k} {frames.paths[r['index']]} ({r['how']}, rows {_range(r['rows'])}, true {_range(r.get('truth_rows'))}):"
            )
            out.extend(lines)
    out.append("")


# --- main --------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--frames", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--items", type=Path, default=ITEMS_DB)
    # The video of the second scroll, its log and the db after it, by default
    # the files beside --log
    parser.add_argument("--video", type=Path)
    parser.add_argument("--video-log", type=Path)
    parser.add_argument("--video-db", type=Path)
    parser.add_argument("--video-start", default=VIDEO_START)
    args = parser.parse_args()
    folder = args.log.parent
    args.video = args.video or folder / VIDEO_NAME
    args.video_log = args.video_log or folder / VIDEO_LOG_NAME
    args.video_db = args.video_db or folder / VIDEO_DB_NAME
    if args.db.resolve() == FORBIDDEN.resolve():
        print("refusing to open arlo_user.db", file=sys.stderr)
        return 2
    logging.getLogger("arc_helper").setLevel(logging.WARNING)
    instrument()
    out: list[str] = [
        "Stash gaps on Jeffrey's scroll of 21 Oct (eval_stash_gaps.py)",
        "",
    ]
    pattern = Path(args.frames)
    candidates = sorted(pattern.parent.glob(pattern.name))
    paths = []
    unreadable = []
    for p in candidates:
        try:
            _load(p)
            paths.append(p)
        except OSError:
            unreadable.append(p)
    observes = parse_log(args.log)
    pair(observes, paths)
    out.extend(
        (
            "== 1. Inputs and the pairing ==",
            "",
            f"captures: {len(candidates)} matched, {len(paths)} readable, {len(unreadable)} unreadable"
            + (": " + ", ".join(p.name for p in unreadable) if unreadable else ""),
            f"first {_short(paths[0])}, last {_short(paths[-1])}, gaps between captures median {_median(np.diff([_capture_time(p).timestamp() for p in paths])):.2f} s",
        )
    )
    placed_live = sum(o["placed"] == "yes" for o in observes)
    out.append(
        f"observe lines: {len(observes)} ({placed_live} placed=yes, "
        f"{len(observes) - placed_live} placed=no), with a pass line "
        f"{sum(o['pass'] is not None for o in observes)}, dropped stale "
        f"{sum(o['stale'] for o in observes)}"
    )
    gaps = np.diff([o["at"].timestamp() for o in observes])
    durations = [o["pass"]["ms"] for o in observes if o["pass"]]
    out.append(
        f"observe gaps: median {_median(gaps):.1f} s, min {min(gaps):.1f}, max "
        f"{max(gaps):.1f}, pass durations median {_median(durations):.0f} ms, min "
        f"{min(durations)}, max {max(durations)}"
    )
    with tempfile.TemporaryDirectory(prefix="stash-gaps-") as folder:
        env = Env(Path(folder), args.items, args.db)
        timing_lines: list[str] = []
        timed = timing_section(env, paths, observes, timing_lines)
        frames = Frames(paths, timed["every"])
        out.extend(
            (
                "",
                "  #  observe at    placed chg filled  pass end      ms   frame at      capture      dt     live id/un  replay id/un  why (live)",
            )
        )
        for k, o in enumerate(observes):
            cap = o["capture"]
            info = frames[cap] if cap is not None else None
            out.append(
                f"  {k:2d} {o['at'].strftime('%H:%M:%S.%f')[:12]}  {o['placed']:3s}  "
                f"{o['changed']:3d} {o['filled']:4d}    "
                f"{o['pass']['at'].strftime('%H:%M:%S.%f')[:12]} {o['pass']['ms']:5d}  "
                f"{datetime.fromtimestamp(o['frame_at'], LOCAL).strftime('%H:%M:%S.%f')[:12]}  "
                f"{_short(paths[cap]) if cap is not None else 'none':11s}  "
                f"{o['dt'] if o['dt'] is not None else float('nan'):+5.2f}  "
                f"{o['pass']['identified']:5d}/{o['pass']['unidentified']:<3d}  "
                + (
                    f"{info['identified']:5d}/{info['unidentified']:<3d}"
                    if info
                    else "    -/-  "
                )
                + f"      {'stale ' if o['stale'] else ''}{o['why']}"
            )
        unmatched = [o for o in observes if o["capture"] is None]
        used = {o["capture"] for o in observes if o["capture"] is not None}
        out.append(
            f"unmatched observe lines: {len(unmatched)}"
            + (
                " ("
                + ", ".join(o["at"].strftime("%H:%M:%S.%f")[:12] for o in unmatched)
                + ")"
                if unmatched
                else ""
            )
        )
        dup = len([o for o in observes if o["capture"] is not None]) - len(used)
        out.append(
            f"captures paired: {len(used)} distinct ({dup} paired twice), unpaired "
            f"{len(paths) - len(used)}: "
            + ", ".join(_short(p) for n, p in enumerate(paths) if n not in used)
        )
        same = sum(
            1
            for o in observes
            if o["capture"] is not None
            and frames[o["capture"]]["identified"] == o["pass"]["identified"]
            and frames[o["capture"]]["unidentified"] == o["pass"]["unidentified"]
        )
        out.extend(
            (
                f"paired captures whose identified/unidentified equal the live pass's: {same} of {len(observes) - len(unmatched)}",
                "",
            )
        )
        out.extend(timing_lines)
        every_order = list(range(len(paths)))
        stamp = {o["at"]: o["at"].strftime("%H:%M:%S.%f")[:12] for o in observes}
        first_live = {}
        for o in observes:
            if o["capture"] is not None:
                first_live.setdefault(o["capture"], stamp[o["at"]])
        spec = [o for o in observes if o["capture"] is not None]
        before = [o for o in observes if o["before"] is not None]
        runs = {
            "every": replay(
                env,
                frames,
                every_order,
                "every",
                [first_live.get(n, "") for n in every_order],
            ),
            "live": replay(
                env,
                frames,
                [o["capture"] for o in spec],
                "live",
                [stamp[o["at"]] for o in spec],
            ),
            "before": replay(
                env,
                frames,
                [o["before"] for o in before],
                "before",
                [stamp[o["at"]] for o in before],
            ),
        }
        first_map = runs["every"]["store"].map
        frames.pitch = first_map.pitch * 2160
        frames.left = first_map.column_left * 2160
        fit = fit_thumb(frames, [runs["every"]])
        for run in runs.values():
            annotate(run, frames, fit)
        frames.final_map = StashMapStore(
            env.database(keep_cache=False, keep_map=True)
        ).map
        signature_section(frames, runs["every"], out)
        out.extend(
            [
                "== 4. Replays from an empty map (question 2) ==",
                "",
                (
                    f"Start: an empty map (the live map started empty: its first observe "
                    f"placed 24 cells, filled 24), the top frame first. pitch "
                    f"{frames.pitch:.1f} px, column 0 at x {frames.left:.0f}, grid right "
                    f"edge x {frames.right}. Truth: thumb top at rest {fit[0]:.0f}, "
                    f"{fit[1]:.4f} thumb px per px scrolled (median over the every-frame "
                    f"run's SCROLL and LOOKS frames). err is true row 0 less placed row "
                    f"0, in rows (0 is right). prop is the same for the place an "
                    f"unplaced frame's check refused. dist is the true rows scrolled "
                    f"since the last placed frame. chg cells changed, emp cells "
                    f"emptied."
                ),
                "",
            ]
        )
        for name, title in (
            ("every", "Every capture observed"),
            (
                "live",
                (
                    "Only the captures paired with a live observe line "
                    f"(nearest within {PAIR_S} s, the spec's rule)"
                ),
            ),
            (
                "before",
                (
                    "Only the live observe lines, each with the latest capture at or "
                    "before its frame time (a capture is saved only when the grid "
                    "moved, so the pass's still frame is the last one saved)"
                ),
            ),
        ):
            out.append(f"-- {title} --")
            run = runs[name]
            table(run, frames, paths, out)
            summary(run, out)
            if name != "every":
                live = [
                    o for o in observes if o in (spec if name == "live" else before)
                ]
                filled = [r["filled"] for r in run["rows"]]
                same = sum(o["filled"] == f for o, f in zip(live, filled, strict=True))
                placed = sum(
                    (o["placed"] == "yes") == (r["how"] in PLACED)
                    for o, r in zip(live, run["rows"], strict=True)
                )
                out.append(
                    f"  against the live log: placed yes/no equal on {placed} of "
                    f"{len(live)} lines, filled count equal on {same} of {len(live)}"
                )
            out.append("")
        for name in ("live", "before"):
            gaps_rows = [
                r["dist"]
                for r in runs[name]["rows"]
                if r["how"] not in PLACED and r["dist"] is not None
            ]
            out.append(
                f"{name} replay, unplaced frames' true distance from the last placed "
                "frame (rows): " + ", ".join(f"{d:+.1f}" for d in gaps_rows)
            )
        vis = [
            len(
                {
                    k[0]
                    for k in frames.keys(
                        r["index"], r["truth"], frames[r["index"]]["whole"]
                    )
                }
            )
            for r in runs["every"]["rows"]
            if r["truth"] is not None
        ]
        out.extend(
            (
                f"whole rows shown per frame: median {_median(vis):.0f}, min {min(vis)}",
                "placed frames against the thumb's row 0 (all three replays, true "
                "known): max |err| "
                + _num(
                    max(
                        abs(r["err"])
                        for run in runs.values()
                        for r in run["rows"]
                        if r["how"] in PLACED and r["err"] is not None
                    ),
                    ".3f",
                )
                + " rows, one thumb px is "
                + _num(1 / fit[1] / frames.pitch, ".3f")
                + " rows",
                "captures without a visible thumb (a card or a window over it): "
                + ", ".join(
                    _short(paths[n]) for n in frames.info if frames[n]["thumb"] is None
                ),
                "",
            )
        )
        try:
            refusal_section(runs, frames, paths, out)
            wrong_section(runs, frames, paths, observes, out)
            if args.video.exists():
                video_section(args, env, out)
        finally:
            text = "\n".join(out) + "\n"
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "real-data.txt").write_text(text, encoding="utf-8")
            print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
