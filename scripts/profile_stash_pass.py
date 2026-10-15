# ruff: noqa: INP001
"""Time each stage of the stash pass (stash_scan.identify_slots) on real frames.

    uv run --extra icons python scripts/profile_stash_pass.py <out.txt>
        [--captures DIR] [--names CSV] [--raid PNG] [--repeats N]

The frames are the captures named in the bar-patch sample CSV (its frame
column) and the 4K raid frame. Every pass reads stacks, as the live overlay
does. The gallery and twins come from main.load_stash_gallery and items
resolve through Database.lookup. The model is loaded and warmed before any
timing.

Stages are timed by wrapping the functions identify_slots calls:
find_slots, crop (icon_embed.crop_square), embed (the embedder), match
(gallery ranking), tier (read_tier), stack (read_stack), bar (bar_geometry)
and sig (the slot cache signature, when the code has one). total is the
whole identify_slots call, other is total minus the stages.

Modes, each a median over --repeats passes:

- no cache: identify_slots without a slot cache, the path before the cache.
- cold: the first pass with an empty slot cache.
- repeat: the same frame again with the cache the cold pass filled.
- scroll: after the cold pass, a frame where the busiest panel's slot rows
  moved up one row, the top row gone and a new bottom row taken from the
  next frame at the same place. Most slots are then cached.

Results are checked against the no cache pass of the same frame: every slot
must keep its rect, item, tier, stack, guess, confidence, lead and bar.
"""

import argparse
import csv
import inspect
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from arc_helper import stash_scan
from arc_helper.config import get_settings
from arc_helper.database import Database
from arc_helper.grid import find_slots
from arc_helper.icon_embed import default_embedder
from arc_helper.main import load_stash_gallery

HANDOFF = Path("/home/jefflamb/workspace/ARLO/.claude/worktrees/overlay-roadmap")
HANDOFF = HANDOFF / ".claude" / "handoff"
CAPTURES = Path("/home/jefflamb/workspace/ARLO/debug/captures")
NAMES = HANDOFF / "bar-patch-s1" / "attempt-1" / "stash_scan.csv"
RAID = HANDOFF / "raid-still-s1" / "raid-inventory-4k.png"
STAGES = ("find_slots", "crop", "embed", "match", "tier", "stack", "bar", "sig")
WRAPPED = {
    "find_slots": "find_slots",
    "crop": "crop_square",
    "match": "_rank",
    "tier": "read_tier",
    "stack": "read_stack",
    "bar": "bar_geometry",
    "sig": "slot_signature",
}
ROW_TOLERANCE = 6


class Clock:
    """Seconds spent per stage since the last reset."""

    def __init__(self) -> None:
        self.spent: dict[str, float] = defaultdict(float)
        self.embedded = 0

    def reset(self) -> None:
        self.spent = defaultdict(float)
        self.embedded = 0

    def wrap(self, stage: str, func):
        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                self.spent[stage] += time.perf_counter() - start

        return timed


class TimedEmbedder:
    """The shared embedder, timed as the embed stage."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._inner = default_embedder()

    def embed(self, squares):
        start = time.perf_counter()
        try:
            return self._inner.embed(squares)
        finally:
            self._clock.spent["embed"] += time.perf_counter() - start
            self._clock.embedded += len(squares)


def instrument(clock: Clock) -> None:
    for stage, name in WRAPPED.items():
        func = getattr(stash_scan, name, None)
        if func is not None:
            setattr(stash_scan, name, clock.wrap(stage, func))


def has_cache() -> bool:
    return "cache" in inspect.signature(stash_scan.identify_slots).parameters


def load(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def frame_paths(captures: Path, names: Path) -> list[Path]:
    with names.open(newline="", encoding="utf-8") as f:
        frames = sorted({row["frame"] for row in csv.DictReader(f)})
    return [captures / name for name in frames]


def _rows(tops: list[int]) -> list[int]:
    rows: list[int] = []
    for top in sorted(tops):
        if not rows or top - rows[-1] > ROW_TOLERANCE:
            rows.append(top)
    return rows


def scrolled(rgb: np.ndarray, filler: np.ndarray) -> np.ndarray | None:
    """rgb with its busiest panel's rows moved up one row, or None.

    The new bottom row comes from filler at the same place. None when the
    panel has fewer than two rows.
    """
    filled = [s for s in find_slots(rgb) if s.filled]
    if not filled:
        return None
    panels: dict[str, list] = defaultdict(list)
    for slot in filled:
        panels[slot.panel].append(slot.rect)
    rects = max(panels.values(), key=len)
    rows = _rows([r.top for r in rects])
    if len(rows) < 2:
        return None
    pitch = int(statistics.median(np.diff(rows)))
    height = int(statistics.median(r.bottom - r.top for r in rects))
    gap = max(0, pitch - height)
    top = max(0, rows[0] - gap // 2)
    bottom = min(rgb.shape[0], top + len(rows) * pitch)
    left = max(0, min(r.left for r in rects) - gap)
    right = min(rgb.shape[1], max(r.right for r in rects) + gap)
    out = rgb.copy()
    out[top : bottom - pitch, left:right] = rgb[top + pitch : bottom, left:right]
    out[bottom - pitch : bottom, left:right] = filler[top : top + pitch, left:right]
    return out


def key(slot) -> tuple:
    return (
        tuple(slot.rect),
        slot.item.name if slot.item else None,
        slot.tier,
        slot.stack,
        slot.guess,
        slot.confidence,
        slot.lead,
        slot.bar,
    )


def run(identify, rgb, clock: Clock, **kwargs) -> tuple[dict[str, float], list]:
    clock.reset()
    start = time.perf_counter()
    result = identify(rgb, **kwargs)
    total = time.perf_counter() - start
    stages = {s: clock.spent.get(s, 0.0) * 1000 for s in STAGES}
    stages["total"] = total * 1000
    stages["other"] = stages["total"] - sum(stages[s] for s in STAGES)
    stages["embedded"] = clock.embedded
    return stages, result


def median_row(runs: list[dict[str, float]]) -> dict[str, float]:
    return {k: statistics.median(r[k] for r in runs) for k in runs[0]}


COLUMNS = (*STAGES, "other", "total", "embedded")


def table(title: str, rows: list[tuple[str, dict[str, float]]], out: list[str]):
    out.append(f"\n{title} (ms per pass, embedded = slots embedded)")
    head = f"{'frame':<34}{'slots':>6}" + "".join(f"{c:>11}" for c in COLUMNS)
    out.append(head)
    for name, row in rows:
        cells = "".join(f"{row[c]:>11.1f}" for c in COLUMNS)
        out.append(f"{name[:33]:<34}{row['slots']:>6.0f}{cells}")
    if len(rows) > 1:
        overall = median_row([r for _, r in rows])
        cells = "".join(f"{overall[c]:>11.1f}" for c in COLUMNS)
        out.append(f"{'median over frames':<34}{overall['slots']:>6.0f}{cells}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--captures", type=Path, default=CAPTURES)
    parser.add_argument("--names", type=Path, default=NAMES)
    parser.add_argument("--raid", type=Path, default=RAID)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    out: list[str] = []
    start = time.perf_counter()
    gallery, twins = load_stash_gallery(get_settings().icon_cache_dir)
    db = Database()
    out.append(
        f"gallery {len(gallery)} entries loaded in "
        f"{(time.perf_counter() - start) * 1000:.0f} ms"
    )
    start = time.perf_counter()
    default_embedder().embed([np.zeros((224, 224, 3), dtype=np.uint8)])
    out.append(
        f"model load and first embed {(time.perf_counter() - start) * 1000:.0f} ms"
    )

    clock = Clock()
    instrument(clock)
    embedder = TimedEmbedder(clock)
    cached = has_cache()
    out.append(f"slot cache in identify_slots: {'yes' if cached else 'no'}")

    def identify(rgb, cache=None):
        kwargs = {"cache": cache} if cached else {}
        return stash_scan.identify_slots(
            rgb,
            gallery,
            db.lookup,
            embedder=embedder,
            read_stacks=True,
            twins=twins,
            **kwargs,
        )

    paths = [*frame_paths(args.captures, args.names), args.raid]
    frames = [(p.name, load(p)) for p in paths]
    modes: dict[str, list] = defaultdict(list)
    mismatches: list[str] = []
    hits: list[str] = []
    for n, (name, rgb) in enumerate(frames):
        filler = frames[(n + 1) % len(frames)][1]
        scroll = scrolled(rgb, filler)
        plain, reference = [], None
        for _ in range(args.repeats):
            stages, reference = run(identify, rgb, clock)
            plain.append(stages)
        slots = len(reference)
        modes["no cache"].append((name, {**median_row(plain), "slots": slots}))
        if not cached:
            continue
        scroll_reference = None
        if scroll is not None:
            _, scroll_reference = run(identify, scroll, clock)
        cold, repeat, after = [], [], []
        for _ in range(args.repeats):
            cache = stash_scan.SlotCache()
            stages, result = run(identify, rgb, clock, cache=cache)
            cold.append(stages)
            if [key(s) for s in result] != [key(s) for s in reference]:
                mismatches.append(f"{name} cold differs from no cache")
            if scroll is not None:
                stages, result = run(identify, scroll, clock, cache=cache)
                after.append(stages)
                if [key(s) for s in result] != [key(s) for s in scroll_reference]:
                    mismatches.append(f"{name} scroll differs from no cache")
            stages, result = run(identify, rgb, clock, cache=cache)
            repeat.append(stages)
            if [key(s) for s in result] != [key(s) for s in reference]:
                mismatches.append(f"{name} repeat differs from no cache")
        modes["cold"].append((name, {**median_row(cold), "slots": slots}))
        modes["repeat"].append((name, {**median_row(repeat), "slots": slots}))
        if after:
            scroll_slots = len(scroll_reference)
            row = {**median_row(after), "slots": scroll_slots}
            modes["scroll"].append((name, row))
            hits.append(
                f"{name}: scroll pass {scroll_slots} slots, "
                f"{scroll_slots - row['embedded']:.0f} from the cache, "
                f"{row['embedded']:.0f} embedded, "
                f"{100 * row['total'] / median_row(cold)['total']:.0f} percent of cold"
            )
        print(f"{name} done", file=sys.stderr, flush=True)

    for mode, rows in modes.items():
        table(mode, rows, out)
    if hits:
        out.append("\nscroll passes")
        out.extend(hits)
    if cached:
        out.append("\nresults against the no cache pass")
        unchanged = (
            "every slot unchanged (rect, item, tier, stack, guess,"
            " confidence, lead, bar) in cold, repeat and scroll"
        )
        out.extend(mismatches or [unchanged])
    text = "\n".join(out) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
