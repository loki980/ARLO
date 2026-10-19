# ruff: noqa: INP001, PLC2701
"""Build the stored stash map from a capture run, frame by frame, and check it.

    uv run --extra icons --extra wayland python scripts/eval_stash_map.py \\
        --out <folder> <frame.png>...

The frames go in name order through stash_map.StashMapStore.observe on a
user database in a temporary folder (never arlo_user.db), identified as
eval_stash_stitch does. Then stash_stitch.stitch runs on the same frames
for comparison. Then the last frame is shown again to a new store on the
saved database, once as it is and once with one filled slot painted over
with an empty slot of the same frame. Writes real-data.txt to the out
folder: one line per frame (how it was placed, the rows it showed, the
cells it changed), the map against the stitch's (filled cells, rows, names
and stacks), the simulated change, the status, and a reload of the saved
map compared with the map in memory.

    uv run --extra icons python scripts/eval_stash_map.py --restart-replay \
        --db <user db copy> --frames '<glob>' --out <folder> [--before REV]

The restart replay: the frames are identified once, then drawn over with
ARLO's badges by the badge helper's PNG mode (as worth_colour_demo does)
three ways. old is the overlay the stored map of 18 Oct was built under
(have/need tags N/M, bar coloured price patches, the red short-stock
outline). grey is step 21's grey patches with tags N/M. today is the overlay
since (tags N+/M, patches coloured by worth, and the gold hover frame outside
one tile). Each run starts a fresh StashMapStore, as after a
restart, on a copy of a database (the given one is never written) and
observes every frame in name order: the code before the fix (stash_map and
stash_stitch at REV, loaded from git) and the code as it is now, from the
copied database, from an empty one and from a map begun mid-stash under the
old overlay (frames MID_FIRST to MID_LAST on an empty database with the code
before the fix), on clean frames, with today's overlay, and with the overlay
changing between frames (old and today, or grey and today). With
--captured only the frames as captured are used. Writes the report: the
look distances of every whole slot clean against each overlay (looks
before and after the fix),
the per-frame placement table with the check that stopped each unplaced
frame, and per run the frames placed, the map against the header and
against the reference build (the code before the fix on clean frames from
an empty database, as on 15 Oct), and the missing rows.
"""

import argparse
import importlib.util
import logging
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from eval_stash_stitch import _Identify
from PIL import Image
from worth_colour_demo import GREY_CODES
from worth_colour_demo import composite
from worth_colour_demo import grey_scale
from worth_colour_demo import old_style
from worth_colour_demo import scale

from arc_helper import stash_map
from arc_helper import stash_stitch
from arc_helper.cell_border import HOVER_GOLD
from arc_helper.database import Database
from arc_helper.grid import STASH
from arc_helper.grid import find_slots
from arc_helper.grid import scroll_offset
from arc_helper.stash_map import StashMapStore
from arc_helper.stash_scan import SHORT_RED
from arc_helper.stash_scan import badges_for
from arc_helper.stash_scan import unpatch
from arc_helper.stash_scan import verdicts_for
from arc_helper.stash_stitch import stitch
from arc_helper.stash_stitch import whole_slots

# The code before the fix: the merge the fix branched from
BEFORE_REV = "d5a3bff"
# The map begun mid-stash: these frames, under the old overlay, on an empty
# database with the code before the fix (frame 10's top whole row is the
# stash's row 7, as Jeffrey's map of 18 Oct began at a row below the top)
MID_FIRST, MID_LAST = 10, 22
# The old overlay's short-stock outline, and the hover frame's width at 4K
# (from Jeffrey's log, "border show ... width=11")
OUTLINE_WIDTH = 5
HOVER_WIDTH = 11
# A composite's slot is a clean frame's slot when their corners are this many
# px apart at most (worth_colour_demo.MATCH_PX)
MATCH_PX = 6
PLACED = {stash_map.FIRST, stash_map.SCROLL, stash_map.LOOKS}


def _load(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _items(identify, index, rgb):
    stash = [s for s in find_slots(rgb) if s.panel == STASH]
    filled = [s for s in whole_slots(stash) if s.filled]
    return identify(index, rgb, filled) if filled else []


def _same_cell(a, b) -> bool:
    if a is None or b is None:
        return a is b
    looks = (a.look is None and b.look is None) or (
        a.look is not None and b.look is not None and np.array_equal(a.look, b.look)
    )
    return a[:7] == b[:7] and a.last_seen == b.last_seen and looks


def _same_maps(a, b) -> list[str]:
    """Where two stored maps differ, empty when they are the same."""
    out = []
    if a[:6] != b[:6]:
        out.append(f"view or header: {a[:6]} against {b[:6]}")
    if set(a.cells) != set(b.cells):
        out.append(f"cells {sorted(set(a.cells) ^ set(b.cells))}")
    for key in sorted(set(a.cells) & set(b.cells)):
        ca, cb = a.cells[key], b.cells[key]
        if not _same_cell(ca, cb):
            out.append(
                f"cell {key}: {ca[:7]} {ca.last_seen} against {cb[:7]} {cb.last_seen}"
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--restart-replay", action="store_true")
    parser.add_argument("--db", type=Path, help="restart replay: the user db copy")
    parser.add_argument("--frames", help="restart replay: a glob of the frames")
    parser.add_argument("--before", default=BEFORE_REV, help="the code before the fix")
    parser.add_argument(
        "--captured",
        action="store_true",
        help="restart replay on the frames as captured only, no badges drawn over",
    )
    parser.add_argument("--report", default="real-data.txt", help="the report's name")
    parser.add_argument("frames_list", nargs="*", metavar="frames")
    args = parser.parse_args()
    if args.restart_replay:
        return restart_replay(args)
    paths = sorted(Path(p) for p in args.frames_list)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logging.getLogger("arc_helper").setLevel(logging.ERROR)
    identify = _Identify()
    temp = Path(tempfile.mkdtemp(prefix="stash_map_"))
    db = Database(user_db_path=temp / "arlo_user.db")
    lines = [
        f"{len(paths)} frames, {paths[0].name} to {paths[-1].name}",
        f"user database: {db.user_db_path}",
        "",
        "build, frame by frame: index name | how | row 0 top y | rows shown | cells changed",
    ]
    store = StashMapStore(db)
    for index, path in enumerate(paths):
        rgb = _load(path)
        seen = store.observe(rgb, _items(identify, index, rgb))
        origin = f"{seen.origin:.0f}" if seen.origin is not None else "-"
        lines.append(
            f"  {index:2d} {path.name} | {seen.how} | {origin} | {seen.rows} | "
            f"{len(seen.changed)}"
        )
    built = store.map
    filled = {k for k, c in built.cells.items() if c.filled}
    rows = sorted({r for r, _ in built.cells})
    lines += [
        "",
        (
            f"map: view {built.view}, pitch {built.pitch * 2160:.1f} px at 2160, "
            f"{len(built.cells)} cells, {len(filled)} filled, rows {rows[0]} to {rows[-1]}"
        ),
        f"status: {store.status()}",
    ]

    result = stitch((_load(p) for p in paths), identify)
    cells = result.cells
    stitched = {k for k, c in cells.items() if c.filled}
    names = [
        (k, built.cells[k].name, c.slot.item.name if c.slot and c.slot.item else None)
        for k, c in cells.items()
        if k in built.cells
        and c.filled
        and built.cells[k].name
        != (c.slot.item.name if c.slot and c.slot.item else None)
    ]
    stacks = [
        (k, built.cells[k].stack, c.slot.stack)
        for k, c in cells.items()
        if k in built.cells
        and c.filled
        and c.slot is not None
        and built.cells[k].stack != c.slot.stack
    ]
    srows = sorted({r for r, _ in cells})
    lines += [
        "",
        f"stitch: {len(cells)} cells, {len(stitched)} filled, rows {srows[0]} to {srows[-1]}",
        (
            f"same cells as the stitch: {set(cells) == set(built.cells)}, "
            f"same filled cells: {stitched == filled}"
        ),
        f"  filled in the map only: {sorted(filled - stitched)}",
        f"  filled in the stitch only: {sorted(stitched - filled)}",
        f"cells whose name differs from the stitch's ({len(names)}): {names}",
        f"cells whose stack differs from the stitch's ({len(stacks)}): {stacks}",
    ]

    last = len(paths) - 1
    rgb = _load(paths[last])
    again = StashMapStore(db)
    reload = _same_maps(built, again.map)
    lines += [
        "",
        f"reload: a new store on the saved database, same map as in memory: {not reload}",
        *(f"  {d}" for d in reload),
    ]
    seen = again.observe(rgb, _items(identify, last, rgb))
    lines.append(
        f"frame {last} as it is, to the new store: {seen.how}, rows {seen.rows}, "
        f"changed {list(seen.changed)}"
    )
    whole = whole_slots([s for s in find_slots(rgb) if s.panel == STASH])
    empty = next(s for s in whole if not s.filled)
    full = max((s for s in whole if s.filled), key=lambda s: s.rect.top)
    blanked = rgb.copy()
    e, f = empty.rect, full.rect
    blanked[f.top : f.bottom, f.left : f.right] = rgb[
        e.top : e.bottom, e.left : e.right
    ]
    before = again.map
    seen = again.observe(blanked, _items(identify, last, blanked))
    after = again.map
    differs = [
        k for k in after.cells if not _same_cell(after.cells[k], before.cells.get(k))
    ]
    lines.append(
        f"frame {last} with the slot at {tuple(f)} painted over with the empty "
        f"slot at {tuple(e)}: {seen.how}, changed {list(seen.changed)}, cells that "
        f"differ from before {differs}"
    )
    for k in differs:
        b, a = before.cells[k], after.cells[k]
        lines.append(
            f"  cell {k}: filled {b.filled} {b.name} x{b.stack} seen {b.last_seen} "
            f"-> filled {a.filled} {a.name} x{a.stack} seen {a.last_seen}"
        )
    lines.append(f"status: {again.status()}")
    third = StashMapStore(db)
    reload = _same_maps(after, third.map)
    lines += [
        f"reload after the change: same map as in memory: {not reload}",
        *(f"  {d}" for d in reload),
    ]
    text = "\n".join(lines) + "\n"
    (out / "real-data.txt").write_text(text)
    print(text)
    return 0


# The restart replay (see the module docstring)

# One letter per frame and run: placed, or why not
CODES = {
    stash_map.FIRST: "F",
    stash_map.SCROLL: "S",
    stash_map.LOOKS: "L",
    stash_map.NO_SLOTS: "n",
    stash_map.NO_HEADER: "h",
    stash_map.NO_TOP: "t",
    stash_map.OTHER_VIEW: "v",
}
LEGEND = [
    "F placed on its own (an empty store's first frame, after the fix any frame",
    "  at the top of the stash), S by its scroll offset from the last placed",
    "  frame, L by slot looks.",
    "Not placed: o no offset found (no scroll offset or no last frame, and no",
    "  look offset to the stored cells nor to the last frame), w the place",
    "  found contradicts the stored looks (shift_wrong), q the place found has",
    "  fewer matching looks than LOOK_VOTES, t an empty store waiting for a",
    "  frame at the top, h an empty store and no header, n no stash slots,",
    "  v another view.",
]


def _code(how: str, why: str) -> str:
    if how != stash_map.UNPLACED:
        return CODES.get(how, "?")
    if "contradicts" in why:
        return "w"
    if "matching looks of" in why:
        return "q"
    return "o"


def _module(name: str, source: str):
    """source run as the module arc_helper.<name>, for its relative imports."""
    spec = importlib.util.spec_from_loader(f"arc_helper.{name}", loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "arc_helper"
    sys.modules[spec.name] = module
    exec(compile(source, f"{name}.py", "exec"), module.__dict__)
    return module


def _before_place(store, old, rgb, stash, probes, stored, pitch, left, scale):
    """The checks of _place before the fix, in its order, as (origin, how) or
    (None, the check that stopped the frame). old is stash_map before the fix."""
    cells = old._stored_observations(stored, pitch, left)  # noqa: SLF001 - the code before the fix
    last = store._last  # noqa: SLF001 - the code before the fix
    found = None
    tried = []
    if last is None or last.rgb.shape != rgb.shape:
        tried.append("no last frame")
    else:
        measured = scroll_offset(
            last.rgb,
            rgb,
            old.scroll_band(last.stash, stash),
            [s.rect for s in last.stash if s.filled],
            [s.rect for s in stash if s.filled],
        )
        if measured is None:
            tried.append("no scroll offset")
        elif not old.offset_checked(last.probes, probes, measured, pitch, scale):
            tried.append(f"scroll offset {measured} against the looks")
        else:
            found = (last.origin + measured, stash_map.SCROLL)
    if found is None:
        shift = old.look_offset(cells, probes, pitch, scale)
        if shift is not None:
            found = (float(shift.offset), stash_map.LOOKS)
        else:
            tried.append("no look offset to the stored cells")
    if found is None and last is not None:
        shift = old.look_offset(last.probes, probes, pitch, scale)
        if shift is not None:
            found = (last.origin + shift.offset, stash_map.LOOKS)
        else:
            tried.append("none to the last frame")
    if found is None:
        return None, ", ".join(tried)
    origin = found[0]
    tol = old.LOOK_TOLERANCE * scale
    support = 0
    tops, diff = old.look_pairs(cells, probes, pitch)
    if tops.size:
        judged = old.judge_offset(tops, diff, round(origin), tol)
        if old.shift_wrong(judged):
            return None, (
                f"{found[1]} place contradicts the stored looks "
                f"({judged.contra} against, {judged.support} for)"
            )
        support = judged.support
    if support < old.LOOK_VOTES and last is not None:
        tops, diff = old.look_pairs(last.probes, probes, pitch)
        if tops.size:
            judged = old.judge_offset(tops, diff, round(origin - last.origin), tol)
            if not old.shift_wrong(judged):
                support = max(support, judged.support)
    if support < old.LOOK_VOTES:
        return None, f"{found[1]} place with {support} matching looks of 3"
    return found


def _before_code(rev: str):
    """StashMapStore before the fix, from stash_map and stash_stitch at rev.

    Its _place runs _before_place, which keeps the check that stopped a
    frame in .why.
    """

    def show(path: str) -> str:
        return subprocess.run(
            ["git", "show", f"{rev}:{path}"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout

    old_stitch = _module("stash_stitch_before", show("src/arc_helper/stash_stitch.py"))
    old_map = _module(
        "stash_map_before",
        show("src/arc_helper/stash_map.py").replace(
            "from .stash_stitch import", "from .stash_stitch_before import"
        ),
    )

    class Before(old_map.StashMapStore):
        why = ""

        def _place(self, *place):
            found, how = _before_place(self, old_map, *place)
            self.why = how if found is None else ""
            return None if found is None else (found, how)

    return old_stitch, Before


def _stored_counts(db: Database) -> dict[str, int]:
    """Item name to count over a database's stored map, of any version."""
    counts: dict[str, int] = {}
    for row in db.load_stash_map()[1]:
        if row["filled"] and row["name"]:
            counts[row["name"]] = counts.get(row["name"], 0) + (row["stack"] or 1)
    return counts


def _matched(rgb: np.ndarray, found: list) -> list:
    """The clean frame's SlotItems moved onto the composite's slot rects."""
    stash = whole_slots([s for s in find_slots(rgb) if s.panel == STASH])
    out = []
    for slot in stash:
        near = [
            f
            for f in found
            if all(
                abs(a - b) <= MATCH_PX for a, b in zip(f.rect, slot.rect, strict=True)
            )
        ]
        if near:
            out.append(near[0]._replace(rect=slot.rect))
    return out


class _Frames:
    """The frames clean and under both overlays, kept as PNGs in a folder."""

    def __init__(
        self, paths: list[Path], temp: Path, db: Database, owned, *, drawn: bool
    ):
        self.paths = paths
        self.folder = temp / "frames"
        self.folder.mkdir()
        identify = _Identify()
        identify.db = db
        verdict_of = verdicts_for(db)
        self.items: dict[tuple[str, int], list] = {}
        scratch = temp / "layer.png"
        for index, path in enumerate(paths):
            rgb = _load(path)
            found = _items(identify, index, rgb)
            self.items["clean", index] = found
            if not drawn:
                continue
            old = old_style(badges_for(found, verdict_of, owned, sure=True), found)
            old += [
                {
                    "rect": b["rect"],
                    "color": SHORT_RED,
                    "width": OUTLINE_WIDTH,
                    "radius": 4,
                    "kind": "border",
                }
                for b in old
                if b["kind"] == "tag" and b.get("fg") == SHORT_RED
            ]
            today = badges_for(found, verdict_of, owned, sure=False)
            with scale(grey_scale, GREY_CODES):
                grey = badges_for(found, verdict_of, owned, sure=True)
            named = [f for f in found if f.item is not None]
            if named:
                left, top, right, bottom = named[index % len(named)].rect
                w = HOVER_WIDTH
                today.append(
                    {
                        "rect": [left - w, top - w, right + w, bottom + w],
                        "color": HOVER_GOLD,
                        "width": w,
                        "radius": 4,
                        "kind": "border",
                    }
                )
            for style, badges in (("old", old), ("grey", grey), ("today", today)):
                shown = composite(rgb, badges, scratch)
                Image.fromarray(shown).save(self.folder / f"{style}-{index:03d}.png")
                self.items[style, index] = _matched(shown, found)

    @staticmethod
    def style_of(style: str, index: int) -> str:
        if style == "changing":
            return ("old", "today")[index % 2]
        if style == "grey-colour":
            return ("grey", "today")[index % 2]
        return style

    def get(self, style: str, index: int) -> tuple[np.ndarray, list]:
        style = self.style_of(style, index)
        if style == "clean":
            rgb = _load(self.paths[index])
        else:
            rgb = _load(self.folder / f"{style}-{index:03d}.png")
        return rgb, self.items[style, index]


PAIRS = (
    ("clean", "old"),
    ("clean", "grey"),
    ("clean", "today"),
    ("old", "today"),
    ("grey", "today"),
)


def _look_lines(frames: _Frames, old_stitch) -> list[str]:
    """Look distances of every whole slot of the clean frames against both
    overlays, with the looks before and after the fix."""
    pairs = {(look, a, b): [] for look in ("before", "after") for a, b in PAIRS}
    for index in range(len(frames.paths)):
        shown = {s: frames.get(s, index)[0] for s in ("clean", "old", "grey", "today")}
        slots = whole_slots([s for s in find_slots(shown["clean"]) if s.panel == STASH])
        for slot in slots:
            r = slot.rect
            for look, fn in (
                ("before", old_stitch.slot_look),
                ("after", stash_stitch.slot_look),
            ):
                looks = {
                    k: fn(v[r.top : r.bottom, r.left : r.right])
                    for k, v in shown.items()
                }
                for a, b in PAIRS:
                    la, lb = looks[a], looks[b]
                    if la is None or lb is None:
                        pairs[look, a, b].append((None, None))
                        continue
                    trimmed = float(stash_stitch.look_distance(la, lb))
                    pairs[look, a, b].append((trimmed, float(np.abs(la - lb).mean())))
    lines = [
        "Look distances: every whole slot of the clean frames, its look on the clean",
        "frame against the same place under each overlay. trimmed is look_distance",
        f"(the mean of the smallest {stash_stitch.LOOK_KEEP:.0%} of the pixel differences, what",
        f"the matching uses, under LOOK_SAME {stash_stitch.LOOK_SAME} is one item), full the mean",
        "of all of them, changed the slots whose full distance is over 0.5.",
    ]
    for (look, a, b), found in pairs.items():
        textured = [p for p in found if p[0] is not None]
        flat = len(found) - len(textured)
        trimmed = np.array([p[0] for p in textured])
        full = np.array([p[1] for p in textured])
        lines.append(
            f"  looks {look} the fix, {a} against {b}: {len(textured)} textured pairs "
            f"({flat} flat on a side), trimmed max {trimmed.max():.2f} and "
            f"{int((trimmed >= stash_stitch.LOOK_SAME).sum())} at LOOK_SAME or more, "
            f"full median {np.median(full):.2f} max {full.max():.2f}, changed "
            f"{int((full > 0.5).sum())}"
        )
    return lines


class _Run:
    def __init__(self, key, code, start, style):
        self.key, self.code, self.start, self.style = key, code, start, style
        self.frames: list[tuple[str, float | None, tuple | None, int, str]] = []
        self.map = None
        self.status = None
        self.reload: list[str] = []

    @property
    def label(self) -> str:
        return f"{self.key}: code {self.code}, from {self.start}, frames {self.style}"

    def placed(self) -> int:
        return sum(f[0] in PLACED for f in self.frames)


def _observe_all(store, frames: _Frames, style: str, indices) -> list:
    out = []
    for index in indices:
        rgb, found = frames.get(style, index)
        seen = store.observe(rgb, found)
        why = getattr(seen, "why", "") or getattr(store, "why", "")
        if seen.how != stash_map.UNPLACED:
            why = ""
        out.append((seen.how, seen.origin, seen.rows, len(seen.changed), why))
    return out


def _shift(run: _Run, reference: _Run, pitch: float) -> tuple[int | None, list[int]]:
    """The reference row of the run's row 0, and the frames placed elsewhere."""
    deltas = {
        i: (f[1] - r[1]) / pitch
        for i, (f, r) in enumerate(zip(run.frames, reference.frames, strict=True))
        if f[1] is not None and r[1] is not None
    }
    if not deltas:
        return None, []
    k = Counter(round(d) for d in deltas.values()).most_common(1)[0][0]
    return k, sorted(i for i, d in deltas.items() if abs(d - k) > 0.25)


def _compare(run: _Run, reference: _Run, pitch: float) -> list[str]:
    k, wrong = _shift(run, reference, pitch)
    cells = run.map.cells if run.map else {}
    filled = sum(c.filled for c in cells.values())
    ref = reference.map.cells
    lines = [
        f"{run.label}",
        f"  placed {run.placed()} of {len(run.frames)}, "
        + ", ".join(
            f"{n} {c}"
            for c, n in sorted(Counter(_code(f[0], f[4]) for f in run.frames).items())
        ),
        f"  map: {len(cells)} cells, {filled} filled, rows "
        + (
            f"{min(r for r, _ in cells)} to {max(r for r, _ in cells)}"
            if cells
            else "none"
        )
        + f", {run.status.text}",
    ]
    if k is None:
        lines.append("  no frame placed where the reference placed one")
        return lines
    lines.append(
        f"  its row 0 is the reference's row {k}, frames placed one row or more "
        f"away from that: {wrong or 'none'}"
    )
    moved = {(r + k, c): cell for (r, c), cell in cells.items()}
    ref_filled = {key for key, c in ref.items() if c.filled}
    run_filled = {key for key, c in moved.items() if c.filled}
    names = sum(
        1 for key in ref_filled & run_filled if moved[key].name != ref[key].name
    )
    missing = sorted({r for r, _ in ref} - {r for r, _ in moved})
    extra = sorted({r for r, _ in moved} - {r for r, _ in ref})
    lines += [
        (
            f"  against the reference ({len(ref_filled)} filled): same filled cells "
            f"{run_filled == ref_filled}, filled only here "
            f"{len(run_filled - ref_filled)}, only in the reference "
            f"{len(ref_filled - run_filled)}, names differing {names}"
        ),
        f"  reference rows missing here: {missing or 'none'}"
        + (f", rows here outside the reference: {extra}" if extra else ""),
    ]
    if run.reload:
        lines.append(f"  reload from the database: {run.reload[0]}")
    return lines


def _why_lines(run: _Run) -> list[str]:
    """The unplaced frames of a run with their reasons, a run of frames with
    the same reason on one line."""
    out = []
    group: list[int] = []
    last = None
    for index, f in enumerate([*run.frames, (None, None, None, 0, None)]):
        why = f[4] if f[0] == stash_map.UNPLACED else None
        if why != last and group:
            span = f"{group[0]}" if len(group) == 1 else f"{group[0]}-{group[-1]}"
            out.append(f"    {span}: {last}")
            group = []
        if why is not None:
            group.append(index)
        last = why
    return out


def _readable(path: Path) -> bool:
    try:
        _load(path)
    except OSError:
        return False
    return True


def _patched(frames: _Frames) -> list[str]:
    """The frames as captured with ARLO's price patches on them (unpatch
    changes a whole slot), the overlay the stash pass reads."""
    counts = []
    for index in range(len(frames.paths)):
        rgb = frames.get("clean", index)[0]
        slots = whole_slots([s for s in find_slots(rgb) if s.panel == STASH])
        crops = (
            rgb[s.rect.top : s.rect.bottom, s.rect.left : s.rect.right] for s in slots
        )
        counts.append(sum(not np.array_equal(unpatch(c), c) for c in crops))
    shown = [f"{i}:{n}" for i, n in enumerate(counts) if n]
    return [
        "ARLO's overlay on the frames as captured (the capture is the grab the",
        "stash pass reads): slots with a price patch, by frame, frames without",
        f"left out ({len(shown)} of {len(counts)} frames have some):",
        "  " + " ".join(shown),
    ]


PLAN = [
    ("b1", "before", "empty", "clean"),
    ("b2", "before", "empty", "today"),
    ("b3", "before", "copy", "clean"),
    ("b4", "before", "copy", "today"),
    ("b5", "before", "mid", "clean"),
    ("b6", "before", "mid", "today"),
    ("a1", "after", "empty", "clean"),
    ("a2", "after", "empty", "today"),
    ("a3", "after", "empty", "changing"),
    ("a4", "after", "copy", "clean"),
    ("a5", "after", "copy", "today"),
    ("a6", "after", "copy", "changing"),
    ("a7", "after", "mid", "today"),
    ("b7", "before", "empty", "grey-colour"),
    ("a8", "after", "empty", "grey-colour"),
    ("a9", "after", "copy", "grey-colour"),
]
# On frames as captured: no badges drawn over, no map begun mid-stash
CAPTURED_PLAN = [
    ("b1", "before", "empty", "clean"),
    ("b2", "before", "copy", "clean"),
    ("a1", "after", "empty", "clean"),
    ("a2", "after", "copy", "clean"),
]


def _mid(before, frames: _Frames, mid_db: Path) -> list[str]:
    """Build the map begun mid-stash in mid_db, with the code before the fix."""
    store = before(Database(user_db_path=mid_db))
    seen = _observe_all(store, frames, "old", range(MID_FIRST, MID_LAST + 1))
    cells = store.map.cells
    return [
        (
            f"map begun mid-stash (code before the fix, empty database, frames "
            f"{MID_FIRST} to {MID_LAST} under the old overlay): placed "
            f"{sum(f[0] in PLACED for f in seen)} of {len(seen)}, "
            f"{len(cells)} cells, rows {min(r for r, _ in cells)} to "
            f"{max(r for r, _ in cells)}"
        ),
        "",
    ]


def restart_replay(args) -> int:
    pattern = Path(args.frames)
    found = sorted(pattern.parent.glob(pattern.name))
    paths = [p for p in found if _readable(p)]
    skipped = [p.name for p in found if p not in paths]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logging.getLogger("arc_helper").setLevel(logging.ERROR)
    temp = Path(tempfile.mkdtemp(prefix="stash_restart_"))
    given = temp / "given.db"
    shutil.copy(args.db, given)
    verdict_db = temp / "verdicts.db"
    shutil.copy(args.db, verdict_db)
    owned = _stored_counts(Database(user_db_path=given))
    old_stitch, before = _before_code(args.before)
    lines = [
        f"{len(paths)} frames, {paths[0].name} to {paths[-1].name}",
        f"database: copies of {args.db} in {temp} (the given file is never opened for writing)",
        f"code before the fix: stash_map and stash_stitch at {args.before}",
        f"frames that do not load, left out: {skipped or 'none'}",
        "",
    ]
    copy_view, copy_rows = Database(user_db_path=given).load_stash_map()
    lines.append(
        f"copied map: version {copy_view['version']}, "
        f"{len(copy_rows)} cells, {sum(r['filled'] for r in copy_rows)} filled, rows "
        f"{min(r['row'] for r in copy_rows)} to {max(r['row'] for r in copy_rows)}, header "
        f"{copy_view['header_used']}/{copy_view['header_capacity']}, row 0 starts with "
        f"{next(r['name'] for r in copy_rows if r['row'] == 0)}, its top row with "
        f"{copy_rows[0]['name']}"
    )
    frames = _Frames(
        paths, temp, Database(user_db_path=verdict_db), owned, drawn=not args.captured
    )
    if args.captured:
        lines += ["", *_patched(frames), ""]
    else:
        lines += ["", *_look_lines(frames, old_stitch), ""]

    def database(start: str, key: str) -> Database:
        path = temp / f"{key}.db"
        if start == "copy":
            shutil.copy(given, path)
        elif start == "mid":
            shutil.copy(mid_db, path)
        return Database(user_db_path=path)

    mid_db = temp / "mid.db"
    plan = CAPTURED_PLAN
    if not args.captured:
        lines += _mid(before, frames, mid_db)
        plan = PLAN
    runs = []
    for key, code, start, style in plan:
        run = _Run(key, code, start, style)
        db = database(start, key)
        store = (before if code == "before" else StashMapStore)(db)
        run.frames = _observe_all(store, frames, style, range(len(paths)))
        run.map = store.map
        run.status = store.status()
        if code == "after":
            differs = _same_maps(run.map, StashMapStore(db).map)
            run.reload = [
                "same map as in memory" if not differs else f"differs: {differs[:3]}"
            ]
        runs.append(run)
    # The reference: the code before the fix from an empty database, as on
    # 15 Oct, or the first run with a map when that one built none
    reference = next(r for r in runs if r.map is not None and r.map.cells)
    pitch = reference.map.pitch * frames.get("clean", 0)[0].shape[0]
    lines += [
        "Runs (each a fresh store on its own copy, frames observed in name order).",
        "clean: the frames as captured. today: today's overlay. changing: old",
        "overlay on even frames, today's on odd ones. grey-colour: the grey",
        "overlay (step 21's grey patches, tags N/M) on even frames, today's",
        "(colour patches, tags N+/M) on odd ones. empty: an empty database.",
        "copy: the copied database. mid: the map begun mid-stash above.",
        f"{reference.key} is the reference the rows are counted in.",
        *(f"  {r.label}" for r in runs),
        "",
        *LEGEND,
        "",
        "frame  ref rows   " + " ".join(r.key for r in runs),
    ]
    for index in range(len(paths)):
        ref = reference.frames[index]
        rows = f"{ref[2][0]:>2}-{ref[2][1]:<2}" if ref[2] else "  -  "
        codes = "  ".join(_code(r.frames[index][0], r.frames[index][4]) for r in runs)
        lines.append(f"  {index:3d}  {rows}      {codes}")
    lines += ["", "Why each unplaced frame was not placed:"]
    for run in runs:
        lines.append(f"  {run.key}:")
        lines += _why_lines(run) or ["    none"]
    lines += ["", "Per run:"]
    for run in runs:
        lines += _compare(run, reference, pitch)
    text = "\n".join(lines) + "\n"
    (out / args.report).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
