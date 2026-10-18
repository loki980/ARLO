"""
Stitch the stash across scrolling: every stash slot of a run of frames gets
its absolute place in the whole stash.

stitch walks full-screen RGB frames in order. On each frame grid.find_slots
finds the stash slots, and only whole slots (not cut by the panel's top or
bottom edge) are kept. A frame is placed by its scroll: how far the stash
content has moved up since the first frame, in frame px. The offset from the
previous frame of its view comes from grid.scroll_offset, checked against
the slot looks (an offset that lands slots on clearly different looks is
dropped). When that gives nothing, the offset comes from matching the slot
looks of the two frames alone (look_offset). The looks are taken at every
grid place in the viewport, also where find_slots missed a slot under
ARLO's badges (grid_probes), and compared without the part a stale badge covers
(look_distance). A frame that neither places
starts a new segment. After the last frame, segments are joined where a
frame of one matches a frame of another by looks. A segment that matches
none stays apart and is reported, never placed by a guess.

The absolute row of a slot is its content top (its top plus the frame's
scroll) less the first frame's top whole row, over the row pitch, so that
row is row 0 and rows above it are negative. Columns count from the
leftmost slot column, with the same pitch (the grid is square). Every
(row, column) is one Cell, merged over every frame that showed it whole:
filled when most frames saw it filled (on a tie, when one identified an
item in it), with the item most frames agree on.

Frames are grouped by view: the selected category tab in the tab bar left of
the grid (tab_index), the slot size and the grid's left edge. Frames of
another view go in a map of their own, so a filtered tab or another layout
is never mixed into the whole stash.

The STASH header is read on every frame (inventory_header.read_stash_header).
It anchors on the top slot found, but the header stays put while the slots
scroll, so once the top whole row sits away from its resting place the read
misses the header. Then it is read again with the slots of the last frame
where it read.

Items come from an identify callable, usually a wrapper of
stash_scan.identify_slots. Without one the map has geometry and fill only.
"""

from collections import Counter
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from itertools import combinations
from itertools import pairwise
from itertools import product
from typing import NamedTuple

import cv2
import numpy as np

from .grid import STASH
from .grid import Slot
from .grid import find_slots
from .grid import scroll_offset
from .inventory_header import StashHeader
from .inventory_header import read_stash_header
from .stash_scan import SlotItem
from .stash_scan import owned_totals
from .tooltip_layout import Rect

# Lengths in px on a 2160 px tall frame, scaled to the frame's height
REFERENCE_HEIGHT = 2160

# A slot is whole when its height is within this share of the frame's
# tallest stash slot
WHOLE_SHARE = 0.04

# The slot look: a grey LOOK_SIDE square of the slot's middle, without what
# ARLO draws on a tile, so a change in its badges never changes a look.
# - The border band, LOOK_BORDER of the side: the old red short-stock outline
#   (5 px inside the rect at 4K, 0.026 of a 194 px slot). The gold hover
#   frame (cell_border) lies outside the rect, in the gap between slots.
# - The top band, LOOK_TOP of the height: the have/need tag, or a wide
#   slot's price pill, in a top corner. badge_helper draws its text at
#   TAG_TEXT_SHARE (0.17) of the slot height, inset 0.3 of the text size,
#   the pill the text's ink (at most the text size) plus 0.3 of it above and
#   below, and a 2 px outline. Its bottom is at most 1.9 x 0.17 = 0.323 of
#   the height plus 1 px, and about 1 px more from rounding the text size,
#   for slots 65 to 235 px tall (where the text size is not clamped): under
#   0.34 from 100 px up. Measured on composites of the 15 Oct captures:
#   0.286.
# - The bottom bar, from LOOK_BOTTOM of the height: the price patch is drawn
#   on the bar, whose top is at 0.734 to 0.758, and the count.
LOOK_SIDE = 16
LOOK_BORDER = 0.07
LOOK_TOP = 0.34
LOOK_BOTTOM = 0.72
# A look whose grey spread is under this is flat: an empty slot or an icon
# not drawn yet, which matches anything and is left out of the matching
LOOK_TEXTURE = 10.0
# look_distance keeps this share of the pixel differences. Under LOOK_SAME
# two looks show one item, over LOOK_DIFFERENT clearly two
LOOK_KEEP = 0.75
LOOK_SAME = 4.0
LOOK_DIFFERENT = 10.0
# Fewest matching slots that place a frame by looks, and how far apart (in
# px) the tops of matching slots may disagree on the offset
LOOK_VOTES = 3
LOOK_TOLERANCE = 8
# A shift whose contradictions (textured slots landing on clearly different
# looks) reach this many, and more than CONTRA_SHARE of its matches, is wrong
CONTRA_MIN = 2
CONTRA_SHARE = 1 / 3

# An observation further than this share of the pitch from its row's line is
# misplaced and dropped
ROW_RESIDUAL = 0.25

# The tab bar: icons centred TAB_DX px left of the grid's left edge, each
# TAB_MIN to TAB_MAX px tall. The selected one is a white disc, where every
# channel is over TAB_WHITE (see tab_index)
TAB_DX = 104
TAB_HALF_WIDTH = 8
TAB_INK = 60
TAB_HALF_DISC = 30
TAB_MIN = 60
TAB_MAX = 130
TAB_WHITE = 200
TAB_WHITE_SHARE = 0.15

# The slot size and grid left edge in a view's key are rounded to this share
# of the frame height
VIEW_ROUND = 0.01

# Placement kinds
FIRST = "first"
SCROLL = "scroll"
LOOKS = "looks"
NEW_SEGMENT = "new segment"
NO_SLOTS = "no slots"

Identify = Callable[[int, np.ndarray, list[Slot]], list[SlotItem]]


class View(NamedTuple):
    """What a frame shows: the selected tab (None when not seen), the slot
    size and the grid's left edge, both in VIEW_ROUND shares of the height."""

    tab: int | None
    side: int
    left: int


class Observation(NamedTuple):
    """One whole stash slot on one frame."""

    frame: int
    rect: Rect
    filled: bool
    look: np.ndarray | None
    slot: SlotItem | None


class Cell(NamedTuple):
    """One absolute stash slot, merged over the frames that showed it whole.

    slot is the chosen identification (None when empty or never identified),
    frame the frame it comes from. seen counts the frames that showed it,
    filled_seen those that saw it filled, names the distinct item names they
    identified (more than one is a disagreement).
    """

    row: int
    column: int
    filled: bool
    slot: SlotItem | None
    frame: int | None
    seen: int
    filled_seen: int
    names: tuple[str, ...]


class Overlap(NamedTuple):
    """The cells two consecutive frames of a segment both showed, compared by
    look: agree (one look), disagree (clearly different looks) and unsure (a
    flat look against a textured one, or in between)."""

    a: int
    b: int
    shared: int
    agree: int
    disagree: int
    unsure: int


class FrameNote(NamedTuple):
    """How one frame went: its view, header, slot counts and placement.

    how is FIRST, SCROLL, LOOKS, NEW_SEGMENT or NO_SLOTS (a NEW_SEGMENT
    frame may be joined later, see ViewMap.joins). offset is
    the measured offset from the previous frame of its view (content moved
    down, px), scroll the frame's final scroll in its segment, dropped the
    observations too far from any row.
    """

    index: int
    view: View | None
    header: StashHeader | None
    slots: int
    whole: int
    filled: int
    how: str
    offset: int | None
    segment: int | None
    scroll: int | None
    dropped: int


class Join(NamedTuple):
    """Frame b's segment joined frame a's, b's content moved down by offset."""

    a: int
    b: int
    offset: int


class Segment(NamedTuple):
    """Frames placed together, with their cells keyed by (row, column)."""

    frames: tuple[int, ...]
    cells: dict[tuple[int, int], Cell]
    overlaps: tuple[Overlap, ...]


class ViewMap(NamedTuple):
    """The stitched stash of one view. segments[0] starts at the view's first
    frame, the others matched it nowhere. joins are the segments joined by
    looks after the last frame."""

    view: View
    header: StashHeader | None
    pitch: float
    segments: tuple[Segment, ...]
    joins: tuple[Join, ...]

    @property
    def cells(self) -> dict[tuple[int, int], Cell]:
        return self.segments[0].cells


class StashMap(NamedTuple):
    """Every view's map (maps[0] is the first frame's) and a note per frame."""

    maps: tuple[ViewMap, ...]
    frames: tuple[FrameNote, ...]

    @property
    def cells(self) -> dict[tuple[int, int], Cell]:
        return self.maps[0].cells if self.maps else {}


def frame_scale(rgb: np.ndarray) -> float:
    return rgb.shape[0] / REFERENCE_HEIGHT


def slot_look(crop: np.ndarray) -> np.ndarray | None:
    """The grey look of a slot crop, None when it is flat (see LOOK_TEXTURE)."""
    height, width = crop.shape[:2]
    band = max(1, round(LOOK_BORDER * min(height, width)))
    top, bottom = round(LOOK_TOP * height), round(LOOK_BOTTOM * height)
    inner = np.ascontiguousarray(crop[top:bottom, band : width - band])
    if inner.size == 0:
        return None
    grey = cv2.cvtColor(inner, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(grey, (LOOK_SIDE, LOOK_SIDE), interpolation=cv2.INTER_AREA)
    look = small.astype(np.float32).ravel()
    if float(look.std()) < LOOK_TEXTURE:
        return None
    return look


def _runs(rows: np.ndarray) -> list[tuple[int, int]]:
    """Start and stop of every run of true rows."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], rows.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def look_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """How far two looks (or stacks of them, on the last axis) differ.

    The mean of the smallest LOOK_KEEP share of the pixel differences, so a
    badge drawn over part of one look does not hide that the rest matches.
    """
    diff = np.sort(np.abs(a - b), axis=-1)
    keep = max(1, round(LOOK_KEEP * diff.shape[-1]))
    return diff[..., :keep].mean(axis=-1)


def tab_index(rgb: np.ndarray, grid_left: int) -> int | None:
    """Which tab of the bar left of the stash grid is selected, from the top.

    The selected tab is a white disc: the rows where TAB_WHITE_SHARE of a
    TAB_HALF_DISC wide strip is white, one run TAB_MIN to TAB_MAX tall. Its
    index is the number of other tab icons above it, each a run of rows
    whose TAB_HALF_WIDTH strip is lit on average (the disc's own dark glyph
    splits its run, so it is not counted). None without a white disc.
    """
    scale = frame_scale(rgb)
    x = grid_left - round(TAB_DX * scale)
    half = max(1, round(TAB_HALF_WIDTH * scale))
    disc = max(1, round(TAB_HALF_DISC * scale))
    if x - disc < 0:
        return None
    lo, hi = TAB_MIN * scale, TAB_MAX * scale
    wide = rgb[:, x - disc : x + disc].astype(np.int16)
    white = (wide.min(axis=2) > TAB_WHITE).mean(axis=1) >= TAB_WHITE_SHARE
    discs = [r for r in _runs(white) if lo <= r[1] - r[0] <= hi]
    if len(discs) != 1:
        return None
    top = discs[0][0]
    narrow = rgb[:, x - half : x + half].astype(np.int16)
    ink = narrow.max(axis=2).mean(axis=1) > TAB_INK
    icons = [r for r in _runs(ink) if lo <= r[1] - r[0] <= hi]
    return sum((a + b) / 2 < top for a, b in icons)


def slot_observations(
    index: int, rgb: np.ndarray, slots: Sequence[Slot], items: dict[Rect, SlotItem]
) -> tuple[Observation, ...]:
    """An Observation of every slot of frame index, with its look and its
    SlotItem from items (keyed by rect) when there is one."""
    return tuple(
        Observation(
            index,
            s.rect,
            s.filled,
            slot_look(rgb[s.rect.top : s.rect.bottom, s.rect.left : s.rect.right]),
            items.get(s.rect),
        )
        for s in slots
    )


def whole_slots(slots: list[Slot]) -> list[Slot]:
    if not slots:
        return []
    tallest = max(s.rect.height for s in slots)
    return [s for s in slots if s.rect.height >= (1 - WHOLE_SHARE) * tallest]


def frame_view(rgb: np.ndarray, slots: list[Slot]) -> View:
    unit = VIEW_ROUND * rgb.shape[0]
    left = min(s.rect.left for s in slots)
    side = float(np.median([s.rect.width for s in slots]))
    return View(tab_index(rgb, left), round(side / unit), round(left / unit))


class Shift(NamedTuple):
    offset: int
    support: int
    contra: int


def look_pairs(
    a: Sequence[Observation], b: Sequence[Observation], pitch: float
) -> tuple[np.ndarray, np.ndarray]:
    """Top differences (b less a) and look differences of same-column textured pairs."""
    ta = [o for o in a if o.look is not None]
    tb = [o for o in b if o.look is not None]
    if not ta or not tb:
        return np.empty((0, 0)), np.empty((0, 0))
    looks_a = np.stack([o.look for o in ta])
    looks_b = np.stack([o.look for o in tb])
    diff = look_distance(looks_b[:, None, :], looks_a[None, :, :])
    cols_a = np.array([o.rect.left for o in ta])
    cols_b = np.array([o.rect.left for o in tb])
    same_col = np.abs(cols_b[:, None] - cols_a[None, :]) < pitch / 2
    tops = (
        np.array([o.rect.top for o in tb])[:, None]
        - np.array([o.rect.top for o in ta])[None, :]
    )
    diff[~same_col] = np.inf
    return tops, diff


def judge_offset(tops: np.ndarray, diff: np.ndarray, offset: int, tol: float) -> Shift:
    """Support and contradictions of one offset over the pairs."""
    near = np.abs(tops - offset) <= tol
    matched = near & (diff < LOOK_SAME)
    clash = near & (diff > LOOK_DIFFERENT) & np.isfinite(diff)
    support = int(matched.any(axis=1).sum())
    contra = int((clash.any(axis=1) & ~matched.any(axis=1)).sum())
    found = tops[matched]
    middle = int(np.median(found)) if found.size else offset
    return Shift(middle, support, contra)


def shift_wrong(shift: Shift) -> bool:
    return shift.contra >= CONTRA_MIN and shift.contra > CONTRA_SHARE * shift.support


def look_offset(
    a: Sequence[Observation], b: Sequence[Observation], pitch: float, scale: float
) -> Shift | None:
    """The offset (content moved down, a to b, px) the slot looks agree on.

    Every same-column pair of textured looks within LOOK_SAME proposes the
    difference of their tops. The proposal with the most supporting slots
    less contradicting ones wins when it has LOOK_VOTES supporters, is not
    shift_wrong, and beats every proposal a half pitch or more away. None
    otherwise.
    """
    tops, diff = look_pairs(a, b, pitch)
    if tops.size == 0:
        return None
    tol = LOOK_TOLERANCE * scale
    proposals = sorted({int(t) for t in tops[diff < LOOK_SAME]})
    if not proposals:
        return None
    judged = [judge_offset(tops, diff, p, tol) for p in proposals]
    score = [j.support - j.contra for j in judged]
    best = int(np.argmax(score))
    win = judged[best]
    if win.support < LOOK_VOTES or shift_wrong(win):
        return None
    for j, s in zip(judged, score, strict=True):
        if abs(j.offset - win.offset) >= pitch / 2 and s >= score[best]:
            return None
    return win


def offset_checked(
    a: Sequence[Observation],
    b: Sequence[Observation],
    offset: int,
    pitch: float,
    scale: float,
) -> bool:
    tops, diff = look_pairs(a, b, pitch)
    if tops.size == 0:
        return True
    return not shift_wrong(judge_offset(tops, diff, offset, LOOK_TOLERANCE * scale))


class _Frame(NamedTuple):
    index: int
    view: View | None
    header: StashHeader | None
    slots: int
    observations: tuple[Observation, ...]
    probes: tuple[Observation, ...]


class _Track:
    """The running state of one view: placed frames and their scrolls."""

    def __init__(self, view: View, scale: float):
        self.view = view
        self.scale = scale
        self.frames: list[_Frame] = []
        self.segment: dict[int, int] = {}
        self.scroll: dict[int, int] = {}
        self.how: dict[int, str] = {}
        self.offset: dict[int, int | None] = {}
        self.pitches: list[float] = []
        self.segments = 0
        self.joins: list[Join] = []
        # The panel's viewport so far: the top and bottom of its slots
        self.top = 1 << 30
        self.bottom = 0


def row_pitches(observations: Sequence[Observation]) -> list[float]:
    """Adjacent same-column top differences of whole slots, one row apart."""
    if not observations:
        return []
    side = float(np.median([o.rect.height for o in observations]))
    found = []
    for column in _columns(observations):
        tops = [o.rect.top for o in column]
        tops.sort()
        found.extend(
            float(b - a) for a, b in pairwise(tops) if side < b - a < 1.4 * side
        )
    return found


def _columns(observations: Sequence[Observation]) -> list[list[Observation]]:
    """Observations grouped by column, left to right."""
    if not observations:
        return []
    side = float(np.median([o.rect.width for o in observations]))
    columns: list[list[Observation]] = []
    for o in sorted(observations, key=lambda o: o.rect.left):
        if not columns or o.rect.left - columns[-1][-1].rect.left > side / 2:
            columns.append([])
        columns[-1].append(o)
    return columns


def grid_probes(
    rgb: np.ndarray,
    observations: tuple[Observation, ...],
    top: int,
    bottom: int,
    pitch: float,
    scale: float,
) -> tuple[Observation, ...]:
    """The whole slots plus a look at every grid place inside the viewport.

    find_slots misses slots under ARLO's badges now and then, but the grid
    is regular: in every column with a whole slot, a probe sits every pitch
    up and down from the frame's top whole slot while it fits between top
    and bottom. Probes only feed the look matching, never the map.
    """
    if not observations or pitch <= 0:
        return observations
    width = round(np.median([o.rect.width for o in observations]))
    height = round(np.median([o.rect.height for o in observations]))
    base = min(o.rect.top for o in observations)
    first = -int((base - top) // pitch)
    last = int((bottom - height - base) // pitch)
    tol = LOOK_TOLERANCE * scale
    index = observations[0].frame
    out = list(observations)
    for column in _columns(observations):
        left = column[0].rect.left
        for k in range(first, last + 1):
            place = round(base + k * pitch)
            if any(abs(o.rect.top - place) <= tol for o in column):
                continue
            rect = Rect(left, place, left + width, place + height)
            crop = rgb[rect.top : rect.bottom, rect.left : rect.right]
            if crop.shape[:2] != (height, width):
                continue
            out.append(
                Observation(index, rect, filled=False, look=slot_look(crop), slot=None)
            )
    return tuple(out)


def scroll_band(slots_a: list[Slot], slots_b: list[Slot]) -> Rect:
    rects = [s.rect for s in (*slots_a, *slots_b)]
    return Rect(
        min(r.left for r in rects),
        min(r.top for r in rects),
        max(r.right for r in rects),
        max(r.bottom for r in rects),
    )


def stitch(
    frames: Iterable[np.ndarray],
    identify: Identify | None = None,
) -> StashMap:
    """The stitched stash of a run of full-screen RGB frames taken while scrolling.

    frames may be a generator: only the previous frame of each view is kept.
    identify(index, rgb, slots) is called once for each frame with stash
    slots, with the frame's whole filled stash slots, and returns their
    SlotItems in that order (stash_scan.identify_slots with slots=slots).
    """
    tracks: dict[View, _Track] = {}
    order: list[View] = []
    notes: list[tuple[_Frame, View | None]] = []
    previous: dict[View, tuple[np.ndarray, list[Slot], tuple[Observation, ...]]] = {}
    header_slots: list[Slot] | None = None
    for index, rgb in enumerate(frames):
        scale = frame_scale(rgb)
        stash = [s for s in find_slots(rgb) if s.panel == STASH]
        if not stash:
            notes.append((_Frame(index, None, None, 0, (), ()), None))
            continue
        header = read_stash_header(rgb, stash)
        if header is not None:
            header_slots = stash
        elif header_slots is not None:
            header = read_stash_header(rgb, header_slots)
        whole = whole_slots(stash)
        filled = [s for s in whole if s.filled]
        items = {}
        if identify is not None:
            for s in identify(index, rgb, filled):
                items[s.rect] = s
        observations = slot_observations(index, rgb, whole, items)
        view = frame_view(rgb, whole)
        track = tracks.get(view)
        if track is None:
            track = tracks[view] = _Track(view, scale)
            order.append(view)
        track.pitches.extend(row_pitches(observations))
        pitch = float(np.median(track.pitches)) if track.pitches else 0.0
        track.top = min(track.top, *(s.rect.top for s in stash))
        track.bottom = max(track.bottom, *(s.rect.bottom for s in stash))
        probes = grid_probes(rgb, observations, track.top, track.bottom, pitch, scale)
        frame = _Frame(index, view, header, len(stash), observations, probes)
        notes.append((frame, view))
        track.frames.append(frame)
        prior = previous.get(view)
        offset = None
        how = NEW_SEGMENT
        if prior is None:
            how = FIRST
        else:
            prgb, pslots, pprobes = prior
            measured = scroll_offset(
                prgb,
                rgb,
                scroll_band(pslots, stash),
                [s.rect for s in pslots if s.filled],
                [s.rect for s in stash if s.filled],
            )
            if measured is not None and offset_checked(
                pprobes, probes, measured, pitch, scale
            ):
                offset, how = measured, SCROLL
            else:
                shift = look_offset(pprobes, probes, pitch, scale)
                if shift is not None:
                    offset, how = shift.offset, LOOKS
        if offset is None:
            track.segment[index] = track.segments
            track.segments += 1
            track.scroll[index] = 0
        else:
            last = track.frames[-2].index
            track.segment[index] = track.segment[last]
            track.scroll[index] = track.scroll[last] - offset
        track.how[index] = how
        track.offset[index] = offset
        previous[view] = (rgb, stash, probes)
    maps = []
    finished: dict[int, FrameNote] = {}
    for view in order:
        view_map, frame_notes = _finish(tracks[view])
        maps.append(view_map)
        finished.update(frame_notes)
    out = []
    for frame, view in notes:
        if frame.index in finished:
            out.append(finished[frame.index])
        else:
            out.append(
                FrameNote(
                    frame.index, view, None, 0, 0, 0, NO_SLOTS, None, None, None, 0
                )
            )
    return StashMap(tuple(maps), tuple(out))


def _join(track: _Track, pitch: float) -> None:
    """Join segments by looks while some frame of one matches a frame of another."""
    by_index = {f.index: f for f in track.frames}
    while True:
        groups: dict[int, list[int]] = {}
        for index, seg in track.segment.items():
            groups.setdefault(seg, []).append(index)
        if len(groups) < 2:
            return
        best = None
        for seg_a, seg_b in combinations(sorted(groups), 2):
            for ia, ib in product(groups[seg_a], groups[seg_b]):
                shift = look_offset(
                    by_index[ia].probes,
                    by_index[ib].probes,
                    pitch,
                    track.scale,
                )
                if shift is None:
                    continue
                score = shift.support - shift.contra
                if best is None or score > best[0]:
                    best = (score, seg_a, seg_b, ia, ib, shift.offset)
        if best is None:
            return
        _, seg_a, seg_b, ia, ib, offset = best
        delta = track.scroll[ia] - offset - track.scroll[ib]
        for index in groups[seg_b]:
            track.segment[index] = seg_a
            track.scroll[index] += delta
        track.joins.append(Join(ia, ib, offset))


def _merge(row: int, column: int, seen: list[Observation]) -> Cell:
    filled_seen = [o for o in seen if o.filled]
    named = [o for o in filled_seen if o.slot is not None and o.slot.item is not None]
    names = Counter(o.slot.item.name for o in named)
    chosen = None
    if named:
        top = max(
            names,
            key=lambda name: (
                names[name],
                max(o.slot.confidence for o in named if o.slot.item.name == name),
            ),
        )
        mine = [o for o in named if o.slot.item.name == top]
        stacks = Counter(o.slot.stack for o in mine)
        chosen = max(mine, key=lambda o: (stacks[o.slot.stack], o.slot.confidence))
    elif filled_seen:
        guessed = [o for o in filled_seen if o.slot is not None]
        if guessed:
            chosen = max(guessed, key=lambda o: o.slot.confidence)
    filled = 2 * len(filled_seen) > len(seen) or (
        2 * len(filled_seen) == len(seen) and bool(named)
    )
    if not filled:
        chosen = None
    elif chosen is None and filled_seen:
        chosen = filled_seen[0]
    return Cell(
        row,
        column,
        filled,
        chosen.slot if chosen is not None else None,
        chosen.frame if chosen is not None else None,
        len(seen),
        len(filled_seen),
        tuple(sorted(names)),
    )


def _overlap(a: int, b: int, cells_a: dict, cells_b: dict) -> Overlap:
    shared = sorted(set(cells_a) & set(cells_b))
    agree = disagree = unsure = 0
    for key in shared:
        la, lb = cells_a[key].look, cells_b[key].look
        if la is None and lb is None:
            if cells_a[key].filled == cells_b[key].filled:
                agree += 1
            else:
                unsure += 1
            continue
        if la is None or lb is None:
            unsure += 1
            continue
        diff = float(look_distance(la, lb))
        if diff < LOOK_SAME:
            agree += 1
        elif diff > LOOK_DIFFERENT:
            disagree += 1
        else:
            unsure += 1
    return Overlap(a, b, len(shared), agree, disagree, unsure)


def _finish(track: _Track) -> tuple[ViewMap, dict[int, FrameNote]]:
    pitch = float(np.median(track.pitches)) if track.pitches else 1.0
    _join(track, pitch)
    by_index = {f.index: f for f in track.frames}
    first = {}
    for f in track.frames:
        first.setdefault(track.segment[f.index], f.index)
    left = min(o.rect.left for f in track.frames for o in f.observations)
    segments = []
    dropped: dict[int, int] = {}
    seg_ids = sorted(set(track.segment.values()), key=lambda s: first[s])
    for seg in seg_ids:
        indices = [f.index for f in track.frames if track.segment[f.index] == seg]
        origin_frame = by_index[first[seg]]
        origin = (
            min(o.rect.top for o in origin_frame.observations)
            + track.scroll[origin_frame.index]
        )
        seen: dict[tuple[int, int], list[Observation]] = {}
        placed: dict[int, dict[tuple[int, int], Observation]] = {}
        for index in indices:
            placed[index] = {}
            dropped[index] = 0
            for o in by_index[index].observations:
                exact = (o.rect.top + track.scroll[index] - origin) / pitch
                row = round(exact)
                if abs(exact - row) > ROW_RESIDUAL:
                    dropped[index] += 1
                    continue
                column = round((o.rect.left - left) / pitch)
                key = (row, column)
                if key in placed[index]:
                    dropped[index] += 1
                    continue
                placed[index][key] = o
                seen.setdefault(key, []).append(o)
        cells = {key: _merge(*key, obs) for key, obs in sorted(seen.items())}
        overlaps = tuple(
            _overlap(a, b, placed[a], placed[b]) for a, b in pairwise(indices)
        )
        segments.append(Segment(tuple(indices), cells, overlaps))
    headers = Counter(f.header for f in track.frames if f.header is not None)
    header = headers.most_common(1)[0][0] if headers else None
    notes = {}
    seg_number = {seg: n for n, seg in enumerate(seg_ids)}
    for f in track.frames:
        notes[f.index] = FrameNote(
            f.index,
            track.view,
            f.header,
            f.slots,
            len(f.observations),
            sum(o.filled for o in f.observations),
            track.how[f.index],
            track.offset[f.index],
            seg_number[track.segment[f.index]],
            track.scroll[f.index],
            dropped[f.index],
        )
    return ViewMap(
        track.view, header, pitch, tuple(segments), tuple(track.joins)
    ), notes


def filled_cells(cells: dict[tuple[int, int], Cell]) -> int:
    return sum(c.filled for c in cells.values())


def tally(cells: dict[tuple[int, int], Cell]) -> dict[str, int]:
    """Item name to total count over the cells (stash_scan.owned_totals)."""
    return owned_totals(c.slot for c in cells.values() if c.filled and c.slot)
