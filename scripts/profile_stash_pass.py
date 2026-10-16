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

- before: cold, repeat and scroll, identify_slots without a slot cache,
  the path before the cache, so each of them identifies every slot.
- after: cold, the first pass with an empty slot cache.
- after: repeat, the same frame again with the cache the cold pass filled.
- after: scroll, after the cold pass, a frame where the busiest panel's
  slot rows moved up one row, the top row gone and a new bottom row taken
  from the next frame at the same place. Most slots are then cached.

The before scroll pass uses the same scrolled frame. --old appends an
earlier output of this script verbatim, by default profile-before.txt, the
no cache table taken on the code before the slot cache.

Results are checked against the no cache pass of the same frame: every slot
must keep its rect, item, tier, stack, guess and bar exactly, and its
confidence and lead within SCORE_TOLERANCE. onnxruntime gives slightly
different vectors for a picture in a different batch, so a cached slot's
scores can differ from a fresh pass in the fourth decimal.

Last, each frame is passed clean on a new slot cache and then with its
badges drawn over it (see badged_passes), which must hit the cache on
every slot.
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
# This script's no cache table, run on the code before the slot cache
OLD = HANDOFF / "grid-fast-s1" / "attempt-1" / "profile-before.txt"
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
SCORE_TOLERANCE = 0.001


class Clock:
    """Seconds spent per stage since the last reset."""

    def __init__(self) -> None:
        self.spent: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.embedded = 0

    def reset(self) -> None:
        self.spent = defaultdict(float)
        self.calls = defaultdict(int)
        self.embedded = 0

    def wrap(self, stage: str, func):
        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                self.spent[stage] += time.perf_counter() - start
                self.calls[stage] += 1

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
        slot.bar,
    )


def same(found: list, reference: list) -> bool:
    """Whether two passes agree, scores within SCORE_TOLERANCE."""
    return len(found) == len(reference) and all(
        key(a) == key(b)
        and abs(a.confidence - b.confidence) <= SCORE_TOLERANCE
        and abs(a.lead - b.lead) <= SCORE_TOLERANCE
        for a, b in zip(found, reference, strict=True)
    )


def badged_passes(frames, identify, clock: Clock, out: list[str]) -> None:
    """Pass each frame clean, then with its badges drawn, on one slot cache.

    The badges are stash_scan.badges_for of the clean pass with every
    verdict short, enough and without needs, drawn with
    badge_helper.render_png and laid over the frame. Each badged pass must
    embed nothing, read no tier or stack and give the clean pass's slots.
    """
    import tempfile

    from PIL import Image as PILImage

    from arc_helper.badge_helper import render_png
    from arc_helper.verdict import Verdict

    verdicts = {
        "short": Verdict("keep", "", 9999, (("need", 9999),)),
        "enough": Verdict("keep", "", 1, (("need", 1),)),
        "no needs": Verdict("sell", ""),
    }
    passes = slots = 0
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "badges.png"
        for name, rgb in frames:
            cache = stash_scan.SlotCache()
            clean = identify(rgb, cache=cache)
            height, width = rgb.shape[:2]
            for label, verdict in verdicts.items():
                badges = stash_scan.badges_for(clean, lambda _s, v=verdict: v)
                render_png(badges, width, height, png)
                with PILImage.open(png) as layer:
                    base = PILImage.fromarray(rgb).convert("RGBA")
                    drawn = PILImage.alpha_composite(base, layer.convert("RGBA"))
                badged = np.asarray(drawn.convert("RGB"))
                _, result = run(identify, badged, clock, cache=cache)
                passes += 1
                slots += len(result)
                reads = clock.calls["tier"] + clock.calls["stack"]
                if clock.embedded or reads or not same(result, clean):
                    problems.append(
                        f"  {name[:33]} {label}: {clock.embedded} embedded,"
                        f" {reads} tier or stack reads,"
                        f" slots {'same' if same(result, clean) else 'differ'}"
                    )
    out.extend(
        [
            "\nbadged passes on the cache of the clean pass",
            (
                f"{passes} passes, {slots} slots, "
                f"{passes - len(problems)} with no embed, no tier or stack read"
                " and the clean pass's slots"
            ),
            *problems,
        ]
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
    parser.add_argument(
        "--old", type=Path, default=OLD, help="an earlier profile to append"
    )
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
    hits: dict[str, list[str]] = defaultdict(list)
    for n, (name, rgb) in enumerate(frames):
        filler = frames[(n + 1) % len(frames)][1]
        scroll = scrolled(rgb, filler)
        plain, again, plain_scroll = [], [], []
        reference = scroll_reference = None
        for _ in range(args.repeats):
            stages, reference = run(identify, rgb, clock)
            plain.append(stages)
            if scroll is not None:
                stages, scroll_reference = run(identify, scroll, clock)
                plain_scroll.append(stages)
            stages, _ = run(identify, rgb, clock)
            again.append(stages)
        slots = len(reference)
        modes["before: cold"].append((name, {**median_row(plain), "slots": slots}))
        modes["before: repeat"].append((name, {**median_row(again), "slots": slots}))
        if plain_scroll:
            row = {**median_row(plain_scroll), "slots": len(scroll_reference)}
            modes["before: scroll"].append((name, row))
            ratio = 100 * row["total"] / median_row(plain)["total"]
            hits["before"].append(
                f"{name}: scroll pass {row['slots']:.0f} slots, "
                f"{row['embedded']:.0f} embedded, {ratio:.0f} percent of cold"
            )
        if not cached:
            continue
        cold, repeat, after = [], [], []
        for _ in range(args.repeats):
            cache = stash_scan.SlotCache()
            stages, result = run(identify, rgb, clock, cache=cache)
            cold.append(stages)
            if not same(result, reference):
                mismatches.append(f"{name} cold differs from no cache")
            if scroll is not None:
                stages, result = run(identify, scroll, clock, cache=cache)
                after.append(stages)
                if not same(result, scroll_reference):
                    mismatches.append(f"{name} scroll differs from no cache")
            stages, result = run(identify, rgb, clock, cache=cache)
            repeat.append(stages)
            if not same(result, reference):
                mismatches.append(f"{name} repeat differs from no cache")
        modes["after: cold"].append((name, {**median_row(cold), "slots": slots}))
        modes["after: repeat"].append((name, {**median_row(repeat), "slots": slots}))
        if after:
            scroll_slots = len(scroll_reference)
            row = {**median_row(after), "slots": scroll_slots}
            modes["after: scroll"].append((name, row))
            hits["after"].append(
                f"{name}: scroll pass {scroll_slots} slots, "
                f"{scroll_slots - row['embedded']:.0f} from the cache, "
                f"{row['embedded']:.0f} embedded, "
                f"{100 * row['total'] / median_row(cold)['total']:.0f} percent of cold"
            )
        print(f"{name} done", file=sys.stderr, flush=True)

    for mode, rows in modes.items():
        table(mode, rows, out)
    for label, lines in hits.items():
        out.append(f"\nscroll passes {label} the slot cache")
        out.extend(lines)
    if cached:
        out.append("\nresults against the no cache pass")
        unchanged = (
            "every slot unchanged (rect, item, tier, stack, guess, bar, and"
            f" confidence and lead within {SCORE_TOLERANCE}) in cold, repeat"
            " and scroll"
        )
        out.extend(mismatches or [unchanged])
        badged_passes(frames, identify, clock, out)
    if args.old is not None:
        out.extend(
            [
                f"\nprofile of the code before the slot cache ({args.old.name})",
                args.old.read_text(encoding="utf-8").rstrip("\n"),
            ]
        )
    text = "\n".join(out) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
