"""
The stored stash map: every slot of the whole stash, kept in the user
database between sessions and updated whenever part of the stash is seen.

StashMapStore.observe takes one full-screen RGB frame. An empty store takes
the frame's view (the selected tab, slot size and grid left edge, see
stash_stitch.frame_view) when its STASH header reads (a filtered tab shows
no capacity, so its header does not read), and its top whole row is row 0.
Frames of another view are ignored after that.

A frame is placed by the stitch's method: its offset from the last placed
frame of this session (grid.scroll_offset checked by the slot looks),
otherwise its slot looks matched to the stored cells' looks
(stash_stitch.look_offset, with the stored cells standing in as the other
frame), otherwise its looks matched to the last placed frame's. The place
found must then agree with the stored looks: LOOK_VOTES textured slots
matching, not stash_stitch.shift_wrong. A frame that does not place changes
nothing.

Every whole slot of a placed frame lands on a (row, column) cell. A cell
not seen before is added. A stored cell changes only on clear evidence
(see _updated): another icon, an empty slot where an item was, an item
where the slot was empty, a better identification of the same icon, or
another stack count. A frame that shows most of the filled cells it covers
as empty is drawn half way (icons not painted yet) and changes nothing.
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
OTHER_VIEW = "other view"
UNPLACED = "unplaced"
HALF_DRAWN = "half drawn"

# A frame is half drawn when more than this share of the stored filled cells
# it shows whole, and at least HALF_DRAWN_MIN of them, come out empty
HALF_DRAWN_SHARE = 0.5
HALF_DRAWN_MIN = 3

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
    the cells it changed and the rows it showed."""

    how: str
    origin: float | None
    changed: tuple[tuple[int, int], ...]
    rows: tuple[int, int] | None


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


def _updated(old: StoredCell, new: StoredCell) -> StoredCell | None:
    """The stored cell after a sighting, None when the sighting changes nothing.

    The look fingerprint alone (a textured look where the stored one was
    flat) is taken without changing last_seen.
    """
    distance = _distance(old.look, new.look)
    same = distance is not None and distance < LOOK_SAME
    different = distance is not None and distance > LOOK_DIFFERENT
    if not new.filled:
        if not old.filled or same:
            # Still empty, or the same icon (the fill test wavered)
            return None
        return new
    if not old.filled:
        # A stale badge can make an empty slot read filled: take it only
        # with an icon or an identification
        if new.look is None and new.name is None:
            return None
        return new
    if different:
        return new
    if new.name is not None and new.name != old.name:
        if old.name is None or new.confidence > old.confidence:
            return new
    elif new.name is not None and new.stack is not None and new.stack != old.stack:
        return old._replace(stack=new.stack, last_seen=new.last_seen)
    if old.look is None and new.look is not None:
        return old._replace(look=new.look)
    return None


def _nothing(how: str) -> Observed:
    return Observed(how, None, (), None)


def _stored_observations(
    stored: StoredMap, pitch: float, left: float
) -> tuple[Observation, ...]:
    """The stored cells with a look as observations of one tall frame,
    row r's top at r * pitch."""
    # Only the tops and left edges take part in the matching
    side = round(pitch)
    out = []
    for c in stored.cells.values():
        if c.look is None:
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
        self._last: _Last | None = None
        self._top = 1 << 30
        self._bottom = 0

    def load(self) -> StoredMap | None:
        """The map as saved in the user database, None when there is none."""
        view, rows = self.db.load_stash_map()
        if view is None:
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
        stored = self._map
        if stored is None:
            if header is None:
                logger.info("stash map: first frame without a STASH header, ignored")
                return _nothing(NO_HEADER)
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
        if not stored.cells:
            origin, how = float(min(o.rect.top for o in observations)), FIRST
        else:
            placed = self._place(rgb, stash, probes, stored, pitch, left, scale)
            if placed is None:
                logger.info("stash map: frame not placed, nothing changed")
                return _nothing(UNPLACED)
            origin, how = placed
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
        covered = [stored.cells[k] for k in seen if k in stored.cells]
        emptied = sum(c.filled and not seen[c.row, c.column].filled for c in covered)
        filled_before = sum(c.filled for c in covered)
        if emptied >= HALF_DRAWN_MIN and emptied > HALF_DRAWN_SHARE * filled_before:
            logger.info(
                "stash map: %d of %d filled cells empty, frame half drawn",
                emptied,
                filled_before,
            )
            return _nothing(HALF_DRAWN)
        writes: dict[tuple[int, int], StoredCell] = {}
        changed = []
        for key, new in seen.items():
            old = stored.cells.get(key)
            cell = new if old is None else _updated(old, new)
            if cell is None:
                continue
            writes[key] = cell
            if old is None or cell.last_seen != old.last_seen:
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
            self._save(new_map, writes.values())
        self._map = new_map
        self._last = _Last(rgb, stash, probes, origin)
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
    ) -> tuple[float, str] | None:
        """The frame y of row 0's top and how it was found, None when the
        frame does not place."""
        cells = _stored_observations(stored, pitch, left)
        last = self._last
        found = None
        if last is not None and last.rgb.shape == rgb.shape:
            measured = scroll_offset(
                last.rgb,
                rgb,
                scroll_band(last.stash, stash),
                [s.rect for s in last.stash if s.filled],
                [s.rect for s in stash if s.filled],
            )
            if measured is not None and offset_checked(
                last.probes, probes, measured, pitch, scale
            ):
                found = (last.origin + measured, SCROLL)
        if found is None:
            shift = look_offset(cells, probes, pitch, scale)
            if shift is not None:
                found = (float(shift.offset), LOOKS)
        if found is None and last is not None:
            shift = look_offset(last.probes, probes, pitch, scale)
            if shift is not None:
                found = (last.origin + shift.offset, LOOKS)
        if found is None:
            return None
        # The place must not contradict the stored looks, and must be borne
        # out by LOOK_VOTES matching looks, of the stored cells or of the
        # last placed frame (whose probes reach rows find_slots missed)
        origin = found[0]
        tol = LOOK_TOLERANCE * scale
        support = 0
        tops, diff = look_pairs(cells, probes, pitch)
        if tops.size:
            judged = judge_offset(tops, diff, round(origin), tol)
            if shift_wrong(judged):
                return None
            support = judged.support
        if support < LOOK_VOTES and last is not None:
            tops, diff = look_pairs(last.probes, probes, pitch)
            if tops.size:
                judged = judge_offset(tops, diff, round(origin - last.origin), tol)
                if not shift_wrong(judged):
                    support = max(support, judged.support)
        if support < LOOK_VOTES:
            return None
        return found

    def _save(self, stored: StoredMap, cells) -> None:
        view = {
            "tab": stored.view.tab,
            "side": stored.view.side,
            "grid_left": stored.view.left,
            "pitch": stored.pitch,
            "column_left": stored.column_left,
            "header_used": stored.header_used,
            "header_capacity": stored.header_capacity,
            "header_seen_at": stored.header_seen_at,
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
        self.db.save_stash_map(view, rows)
