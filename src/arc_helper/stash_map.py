"""
The stored stash map: every slot of the whole stash, kept in the user
database between sessions and updated whenever part of the stash is seen.

StashMapStore.observe takes one full-screen RGB frame. Row 0 is always the
stash's first row. An empty store starts its map on a frame scrolled to the
top of the stash (see top_at_rest) of the whole stash, not a filtered tab:
its STASH header reads (a filtered tab shows no capacity, so its header
does not read) or, as the header's OCR can fail on every frame of a run,
its selected tab is ALL_TAB. It takes that frame's view (the selected tab,
slot size and grid left edge, see stash_stitch.frame_view). Frames of
another view are ignored after that.

A frame scrolled to the top places on its own, its top whole row as row 0,
with no last frame and no stored looks needed (FIRST). Any other frame is
placed by the stitch's method: its offset from the last placed frame of
this session (grid.scroll_offset checked by the slot looks), otherwise its
slot looks matched to the stored cells' looks (stash_stitch.look_offset,
with the stored cells standing in as the other frame), otherwise its looks
matched to the last placed frame's. The place found must then agree with
the cells seen this session (not stash_stitch.shift_wrong) and be borne
out by LOOK_VOTES matching looks. A cell not seen since the restart may be
stale, the stash changed since it was saved: it can find a place but never
stop one. A frame that does not place changes nothing, and Observed.why
says which check stopped it. The last placed frame stays the reference for
the next frame, so a later frame still places by its offset from that one,
by the stored looks, or on its own at the top.

The map is saved with MAP_VERSION. A map of another version (one saved
before the version, with row 0 wherever its first frame was and looks taken
from higher up the slot) is not loaded: the store starts empty and its
first save replaces the old map's cells.

Every whole slot of a placed frame lands on a (row, column) cell. A cell
not seen before is added. A stored cell changes only on clear evidence
(see _updated): another icon, an empty slot where an item was, an item
where the slot was empty, a better identification of the same icon, or
another stack count or tier. A frame drawn half way (icons not painted
yet) can clear cells, and the next sighting with icons fills them again.
Changed cells get the frame's time as last_seen, and only they are written.
"""

from collections.abc import Callable
from datetime import UTC
from datetime import datetime
from typing import TYPE_CHECKING
from typing import NamedTuple

import numpy as np

from .config import logger
from .grid import STASH
from .grid import Slot
from .grid import find_slots
from .grid import scroll_offset
from .inventory_header import STASH_REST_TOPS
from .inventory_header import StashHeader
from .inventory_header import read_stash_header
from .stash_scan import SlotItem
from .stash_stitch import LOOK_DIFFERENT
from .stash_stitch import LOOK_SAME
from .stash_stitch import LOOK_TOLERANCE
from .stash_stitch import LOOK_VOTES
from .stash_stitch import ROW_RESIDUAL
from .stash_stitch import Observation
from .stash_stitch import View
from .stash_stitch import frame_scale
from .stash_stitch import frame_view
from .stash_stitch import grid_probes
from .stash_stitch import judge_offset
from .stash_stitch import look_distance
from .stash_stitch import look_offset
from .stash_stitch import look_pairs
from .stash_stitch import offset_checked
from .stash_stitch import row_pitches
from .stash_stitch import scroll_band
from .stash_stitch import shift_wrong
from .stash_stitch import slot_observations
from .stash_stitch import whole_slots
from .tooltip_layout import Rect

if TYPE_CHECKING:
    from .database import Database

# How a frame went
FIRST = "first"
SCROLL = "scroll"
LOOKS = "looks"
NO_SLOTS = "no slots"
NO_HEADER = "no header"
NO_TOP = "no top"
OTHER_VIEW = "other view"
UNPLACED = "unplaced"

# The stored map's format. 2: row 0 is the stash's first row and the looks
# leave out ARLO's tag band and patch (stash_stitch.LOOK_TOP 0.34 and
# LOOK_BOTTOM 0.7). A map of another version is dropped and rebuilt
MAP_VERSION = 2

# The stash rests at its top when its top whole row is within REST_TOLERANCE
# px of a resting top (inventory_header.STASH_REST_TOPS) and the scrollbar's
# thumb starts there too. A row at the resting top alone also happens one
# whole number of rows down, the thumb tells the two apart. Lengths in px on
# a 2160 px tall frame (stash_stitch.REFERENCE_HEIGHT). Measured on the 15 Oct
# captures: the thumb is 12 px wide, 44 to 55 px right of the grid's right
# edge, grey 118 on a 25 background, and its top is at y 543 with the first
# row at rest (544). Scrolled by one row (208 px) it starts about 18 px lower
REST_TOLERANCE = 6
THUMB_DX_MIN = 20
THUMB_DX_MAX = 90
THUMB_LIT = 70
# The thumb's columns are lit from THUMB_FROM to THUMB_TO px below the
# resting top, and dark from CLEAR_FROM to CLEAR_TO px above it
THUMB_FROM = 3
THUMB_TO = 40
CLEAR_FROM = 30
CLEAR_TO = 8
THUMB_MIN_WIDTH = 6
# The tab bar's first tab shows the whole stash (stash_stitch.tab_index 0)
ALL_TAB = 0

Clock = Callable[[], datetime]


class StoredCell(NamedTuple):
    """One slot of the stored map.

    name is the identified item (None when empty or not identified), tier
    and stack as read, confidence the identification's similarity, look the
    slot's look fingerprint (stash_stitch.slot_look, None when flat), and
    last_seen when this content was seen (ISO time).
    """

    row: int
    column: int
    filled: bool
    name: str | None
    tier: str | None
    stack: int | None
    confidence: float
    look: np.ndarray | None
    last_seen: str


class StoredMap(NamedTuple):
    """The stored map: its view, the row pitch and column 0's left edge in
    shares of the frame height, the STASH header when last read and the cells."""

    view: View
    pitch: float
    column_left: float
    header_used: int | None
    header_capacity: int | None
    header_seen_at: str | None
    cells: dict[tuple[int, int], StoredCell]


class Observed(NamedTuple):
    """What observe did with a frame: how it went (FIRST, SCROLL, LOOKS or
    one of the reasons it changed nothing), the frame y of row 0's top,
    the cells it changed, the rows it showed, and for an UNPLACED frame
    which check stopped it."""

    how: str
    origin: float | None
    changed: tuple[tuple[int, int], ...]
    rows: tuple[int, int] | None
    why: str = ""


class StashStatus(NamedTuple):
    """The map's filled count against the header's used count. sure when they
    are equal, text says it ('map 273 filled, header says 275')."""

    filled: int
    used: int | None
    capacity: int | None
    sure: bool
    text: str


class _Last(NamedTuple):
    rgb: np.ndarray
    stash: list[Slot]
    probes: tuple[Observation, ...]
    origin: float


def _now() -> datetime:
    return datetime.now(UTC)


def _look_bytes(look: np.ndarray | None) -> bytes | None:
    return None if look is None else look.astype(np.float32).tobytes()


def _look_array(blob: bytes | None) -> np.ndarray | None:
    return None if blob is None else np.frombuffer(blob, np.float32).copy()


def _distance(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(look_distance(a, b))


def _from_observation(row: int, column: int, o: Observation, seen: str) -> StoredCell:
    slot = o.slot if o.filled else None
    name = slot.item.name if slot is not None and slot.item is not None else None
    return StoredCell(
        row,
        column,
        o.filled,
        name,
        slot.tier if slot is not None else None,
        slot.stack if slot is not None else None,
        slot.confidence if slot is not None else 0.0,
        o.look,
        seen,
    )


def _updated(old: StoredCell, new: StoredCell) -> tuple[StoredCell, bool] | None:
    """The stored cell after a sighting, and whether its content changed.

    None when the sighting changes nothing. The look fingerprint alone (a
    textured look where the stored one was flat) is taken as no content
    change, keeping last_seen.
    """
    distance = _distance(old.look, new.look)
    same = distance is not None and distance < LOOK_SAME
    different = distance is not None and distance > LOOK_DIFFERENT
    if not new.filled:
        if not old.filled or same:
            # Still empty, or the same icon (the fill test wavered)
            return None
        return new, True
    if not old.filled:
        # A stale badge can make an empty slot read filled: take it only
        # with an icon or an identification
        if new.look is None and new.name is None:
            return None
        return new, True
    if different:
        return new, True
    # The same icon, or looks that cannot tell: take what the sighting read
    # and keep what it did not
    cell = old
    if new.name is not None and new.name != old.name:
        if old.name is not None and new.confidence <= old.confidence:
            # A weaker identification: only the stack read still counts
            new = new._replace(tier=None)
        else:
            cell = cell._replace(name=new.name, confidence=new.confidence)
    if new.tier is not None and new.tier != cell.tier:
        cell = cell._replace(tier=new.tier)
    if new.stack is not None and new.stack != cell.stack:
        cell = cell._replace(stack=new.stack)
    if cell[:7] != old[:7]:
        look = new.look if new.look is not None else old.look
        return cell._replace(look=look, last_seen=new.last_seen), True
    if old.look is None and new.look is not None:
        return old._replace(look=new.look), False
    return None


def _nothing(how: str, why: str = "") -> Observed:
    return Observed(how, None, (), None, why)


def _thumb_at(rgb: np.ndarray, x0: int, x1: int, rest: int, scale: float) -> bool:
    """Whether the scrollbar's thumb starts at the resting top rest, between x0 and x1."""
    above = rest - round(CLEAR_FROM * scale)
    below = rest + round(THUMB_TO * scale)
    if above < 0 or below > rgb.shape[0]:
        return False
    lit = rgb[above:below, x0:x1].max(axis=2) > THUMB_LIT
    clear = ~lit[: rest - round(CLEAR_TO * scale) - above].any(axis=0)
    body = lit[rest + max(1, round(THUMB_FROM * scale)) - above :].all(axis=0)
    return int((clear & body).sum()) >= max(1, round(THUMB_MIN_WIDTH * scale))


def top_at_rest(rgb: np.ndarray, whole: list[Slot]) -> int | None:
    """The frame y of the top whole row when the stash is scrolled to its top.

    None when it is not, or when the scrollbar's thumb is not seen (see
    REST_TOLERANCE). whole are the frame's whole stash slots.
    """
    if not whole:
        return None
    scale = frame_scale(rgb)
    top = min(s.rect.top for s in whole)
    right = max(s.rect.right for s in whole)
    x0 = right + round(THUMB_DX_MIN * scale)
    x1 = min(rgb.shape[1], right + round(THUMB_DX_MAX * scale))
    if x1 <= x0:
        return None
    for share in STASH_REST_TOPS:
        rest = round(share * rgb.shape[0])
        if abs(top - rest) <= REST_TOLERANCE * scale and _thumb_at(
            rgb, x0, x1, rest, scale
        ):
            return top
    return None


def _stored_observations(
    stored: StoredMap, pitch: float, left: float, keys=None
) -> tuple[Observation, ...]:
    """The stored cells with a look (of keys only, when given) as
    observations of one tall frame, row r's top at r * pitch."""
    # Only the tops and left edges take part in the matching
    side = round(pitch)
    out = []
    for key, c in stored.cells.items():
        if c.look is None or (keys is not None and key not in keys):
            continue
        x = round(left + c.column * pitch)
        y = round(c.row * pitch)
        out.append(
            Observation(-1, Rect(x, y, x + side, y + side), c.filled, c.look, None)
        )
    return tuple(out)


class StashMapStore:
    """The stored stash map of one user database (see the module docstring)."""

    def __init__(self, db: "Database", clock: Clock = _now):
        self.db = db
        self.clock = clock
        self._map = self.load()
        # A store that starts without a map replaces whatever cells an old
        # map left on its first save
        self._replace = self._map is None
        self._last: _Last | None = None
        # The cells a placed frame of this session showed
        self._seen: set[tuple[int, int]] = set()
        self._top = 1 << 30
        self._bottom = 0

    def load(self) -> StoredMap | None:
        """The map as saved in the user database, None when there is none or
        it is of another MAP_VERSION."""
        view, rows = self.db.load_stash_map()
        if view is None:
            return None
        if view["version"] != MAP_VERSION:
            logger.info(
                "stash map: stored map of version %s dropped, it is rebuilt "
                "from the top of the stash",
                view["version"],
            )
            return None
        cells = {
            (r["row"], r["col"]): StoredCell(
                r["row"],
                r["col"],
                bool(r["filled"]),
                r["name"],
                r["tier"],
                r["stack"],
                r["confidence"] or 0.0,
                _look_array(r["look"]),
                r["last_seen"],
            )
            for r in rows
        }
        return StoredMap(
            View(view["tab"], view["side"], view["grid_left"]),
            view["pitch"],
            view["column_left"],
            view["header_used"],
            view["header_capacity"],
            view["header_seen_at"],
            cells,
        )

    @property
    def map(self) -> StoredMap | None:
        """The map as this store holds it now (the same as load() after a save)."""
        return self._map

    def status(self) -> StashStatus:
        """The filled count against the header's used count, when known."""
        stored = self._map
        filled = sum(c.filled for c in stored.cells.values()) if stored else 0
        used = stored.header_used if stored else None
        capacity = stored.header_capacity if stored else None
        if used is None:
            return StashStatus(
                filled,
                None,
                capacity,
                sure=False,
                text=f"map {filled} filled, header not read",
            )
        return StashStatus(
            filled,
            used,
            capacity,
            sure=filled == used,
            text=f"map {filled} filled, header says {used}",
        )

    def observe(
        self,
        rgb: np.ndarray,
        slot_items: list[SlotItem],
        header: StashHeader | None = None,
    ) -> Observed:
        """Place one frame against the stored map, update its cells and save.

        slot_items are the frame's identified stash slots (from
        stash_scan.identify_slots), matched to its slots by rect. header is
        the frame's STASH header, read here when not given.
        """
        stash = [s for s in find_slots(rgb) if s.panel == STASH]
        whole = whole_slots(stash)
        if not whole:
            return _nothing(NO_SLOTS)
        if header is None:
            header = read_stash_header(rgb, stash)
        scale = frame_scale(rgb)
        height = rgb.shape[0]
        items = {s.rect: s for s in slot_items}
        observations = slot_observations(0, rgb, whole, items)
        view = frame_view(rgb, whole)
        rest = top_at_rest(rgb, whole)
        stored = self._map
        if stored is None:
            if header is None and view.tab != ALL_TAB:
                logger.info("stash map: first frame without a STASH header, ignored")
                return _nothing(NO_HEADER)
            if rest is None:
                logger.info("stash map: waiting for a frame at the top of the stash")
                return _nothing(NO_TOP)
            pitches = row_pitches(observations)
            if not pitches:
                return _nothing(NO_SLOTS)
            pitch = float(np.median(pitches))
            left = min(o.rect.left for o in observations)
            stored = StoredMap(
                view, pitch / height, left / height, None, None, None, {}
            )
        elif view != stored.view:
            return _nothing(OTHER_VIEW)
        pitch = stored.pitch * height
        left = stored.column_left * height
        self._top = min(self._top, *(s.rect.top for s in stash))
        self._bottom = max(self._bottom, *(s.rect.bottom for s in stash))
        probes = grid_probes(rgb, observations, self._top, self._bottom, pitch, scale)
        if rest is not None:
            origin, how = float(rest), FIRST
        else:
            placed, how = self._place(rgb, stash, probes, stored, pitch, left, scale)
            if placed is None:
                logger.info("stash map: frame not placed (%s), nothing changed", how)
                return _nothing(UNPLACED, how)
            origin = placed
        now = self.clock().isoformat(timespec="seconds")
        seen: dict[tuple[int, int], StoredCell] = {}
        for o in observations:
            exact = (o.rect.top - origin) / pitch
            row = round(exact)
            if abs(exact - row) > ROW_RESIDUAL:
                continue
            key = (row, round((o.rect.left - left) / pitch))
            if key not in seen:
                seen[key] = _from_observation(*key, o, now)
        writes: dict[tuple[int, int], StoredCell] = {}
        changed = []
        for key, new in seen.items():
            old = stored.cells.get(key)
            update = (new, True) if old is None else _updated(old, new)
            if update is None:
                continue
            writes[key], content = update
            if content:
                changed.append(key)
        if header is not None and (header.used, header.capacity) != (
            stored.header_used,
            stored.header_capacity,
        ):
            stored = stored._replace(
                header_used=header.used,
                header_capacity=header.capacity,
                header_seen_at=now,
            )
        cells = {**stored.cells, **writes}
        new_map = stored._replace(cells=cells)
        if writes or new_map[:6] != (self._map[:6] if self._map else None):
            self._save(new_map, writes.values(), replace=self._replace)
            self._replace = False
        self._map = new_map
        self._last = _Last(rgb, stash, probes, origin)
        self._seen.update(seen)
        rows = sorted(r for r, _ in seen)
        return Observed(
            how, origin, tuple(sorted(changed)), (rows[0], rows[-1]) if rows else None
        )

    def _place(
        self,
        rgb: np.ndarray,
        stash: list[Slot],
        probes: tuple[Observation, ...],
        stored: StoredMap,
        pitch: float,
        left: float,
        scale: float,
    ) -> tuple[float | None, str]:
        """The frame y of row 0's top and how it was found (SCROLL or LOOKS),
        or None and which check stopped the frame."""
        cells = _stored_observations(stored, pitch, left)
        last = self._last
        found = None
        tried = []
        if last is None or last.rgb.shape != rgb.shape:
            tried.append("no last frame")
        else:
            measured = scroll_offset(
                last.rgb,
                rgb,
                scroll_band(last.stash, stash),
                [s.rect for s in last.stash if s.filled],
                [s.rect for s in stash if s.filled],
            )
            if measured is None:
                tried.append("no scroll offset")
            elif not offset_checked(last.probes, probes, measured, pitch, scale):
                tried.append(f"scroll offset {measured} against the looks")
            else:
                found = (last.origin + measured, SCROLL)
        support = 0
        if found is None:
            shift = look_offset(cells, probes, pitch, scale)
            if shift is not None:
                found = (float(shift.offset), LOOKS)
                support = shift.support
            else:
                tried.append("no look offset to the stored cells")
        if found is None and last is not None:
            shift = look_offset(last.probes, probes, pitch, scale)
            if shift is not None:
                found = (last.origin + shift.offset, LOOKS)
            else:
                tried.append("none to the last frame")
        if found is None:
            return None, ", ".join(tried)
        # The place must not contradict the cells seen this session, and
        # must be borne out by LOOK_VOTES matching looks: of the stored cells
        # that found it, of the cells seen this session, or of the last
        # placed frame (whose probes reach rows find_slots missed). A cell
        # not seen this session may be stale, the stash changed since it was
        # saved: it can find a place but never stop one
        origin = found[0]
        tol = LOOK_TOLERANCE * scale
        fresh = _stored_observations(stored, pitch, left, self._seen)
        tops, diff = look_pairs(fresh, probes, pitch)
        if tops.size:
            judged = judge_offset(tops, diff, round(origin), tol)
            if shift_wrong(judged):
                return None, (
                    f"{found[1]} place contradicts the cells seen this session "
                    f"({judged.contra} against, {judged.support} for)"
                )
            support = max(support, judged.support)
        if support < LOOK_VOTES and last is not None:
            tops, diff = look_pairs(last.probes, probes, pitch)
            if tops.size:
                judged = judge_offset(tops, diff, round(origin - last.origin), tol)
                if not shift_wrong(judged):
                    support = max(support, judged.support)
        if support < LOOK_VOTES:
            return (
                None,
                f"{found[1]} place with {support} matching looks of {LOOK_VOTES}",
            )
        return found

    def _save(self, stored: StoredMap, cells, *, replace: bool = False) -> None:
        view = {
            "tab": stored.view.tab,
            "side": stored.view.side,
            "grid_left": stored.view.left,
            "pitch": stored.pitch,
            "column_left": stored.column_left,
            "header_used": stored.header_used,
            "header_capacity": stored.header_capacity,
            "header_seen_at": stored.header_seen_at,
            "version": MAP_VERSION,
        }
        rows = [
            {
                "row": c.row,
                "col": c.column,
                "filled": int(c.filled),
                "name": c.name,
                "tier": c.tier,
                "stack": c.stack,
                "confidence": c.confidence,
                "look": _look_bytes(c.look),
                "last_seen": c.last_seen,
            }
            for c in cells
        ]
        self.db.save_stash_map(view, rows, replace=replace)
