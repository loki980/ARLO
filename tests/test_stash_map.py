"""Tests for stash_map: synthetic frames of one stash panel observed into a stored map."""

from datetime import UTC
from datetime import datetime

import numpy as np
import pytest

from arc_helper.database import Database
from arc_helper.inventory_header import StashHeader
from arc_helper.stash_map import FIRST
from arc_helper.stash_map import OTHER_VIEW
from arc_helper.stash_map import UNPLACED
from arc_helper.stash_map import StashMapStore

# The stash panel of test_stash_stitch: 4 columns of 192 px slots on a 208 px
# pitch from x 304, shown between y 510 and 1900 of a 4K frame, row 0 of the
# content at 34 px
FRAME = (2160, 3840)
VIEW_TOP, VIEW_BOTTOM = 510, 1900
LEFT = 304
SIDE = 192
PITCH = 208
OUTLINE = 4
FIRST_ROW = 34
COLUMNS = 4
BACKGROUND = (27, 27, 35)
ROWS = 11
EMPTY = {(9, 2), (9, 3), (10, 0), (10, 1), (10, 2), (10, 3)}
SCROLLS = (0, 300, 650, 1040)


def _paint(image: np.ndarray, row: int, col: int, rng) -> None:
    """Draw slot (row, col): its frame, and an icon from rng unless rng is None."""
    top, left = FIRST_ROW + row * PITCH, col * PITCH
    image[top : top + SIDE, left : left + SIDE] = (160, 160, 165)
    inner = slice(top + OUTLINE, top + SIDE - OUTLINE)
    across = slice(left + OUTLINE, left + SIDE - OUTLINE)
    image[inner, across] = (35, 38, 52)
    if rng is None:
        return
    cells = rng.integers(0, 256, (13, 18, 3), dtype=np.uint8)
    icon = np.repeat(np.repeat(cells, 8, 0), 8, 1)
    image[top + 40 : top + 144, left + 24 : left + 168] = icon


def _content(seed: int = 1, rows: int = ROWS) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height = FIRST_ROW + rows * PITCH + VIEW_BOTTOM
    image = np.full((height, COLUMNS * PITCH, 3), BACKGROUND, np.uint8)
    for row in range(rows):
        for col in range(COLUMNS):
            _paint(image, row, col, None if (row, col) in EMPTY else rng)
    return image


def _frame(content: np.ndarray, scroll: int, left: int = LEFT) -> np.ndarray:
    frame = np.full((*FRAME, 3), BACKGROUND, np.uint8)
    window = content[scroll : scroll + VIEW_BOTTOM - VIEW_TOP]
    frame[VIEW_TOP:VIEW_BOTTOM, left : left + window.shape[1]] = window
    return frame


def _header(used: int) -> StashHeader:
    return StashHeader(used, 304, 0)


FILLED = ROWS * COLUMNS - len(EMPTY)


class _Clock:
    """A clock one minute later on every call."""

    def __init__(self):
        self.minute = 0

    def __call__(self) -> datetime:
        self.minute += 1
        return datetime(2026, 10, 17, 12, self.minute, tzinfo=UTC)


@pytest.fixture
def db(tmp_path) -> Database:
    return Database(db_path=tmp_path / "items.db")


def _built(db: Database, content: np.ndarray | None = None) -> StashMapStore:
    content = _content() if content is None else content
    store = StashMapStore(db, clock=_Clock())
    for scroll in SCROLLS:
        store.observe(_frame(content, scroll), [], _header(FILLED))
    return store


def _cells_except(a, b, keys) -> bool:
    """Whether the maps a and b have the same cells, but for keys."""
    if set(a.cells) != set(b.cells):
        return False
    for key in set(a.cells) - set(keys):
        ca, cb = a.cells[key], b.cells[key]
        if ca[:7] != cb[:7] or ca.last_seen != cb.last_seen:
            return False
        if (ca.look is None) != (cb.look is None):
            return False
        if ca.look is not None and not np.array_equal(ca.look, cb.look):
            return False
    return True


def test_a_scroll_builds_the_whole_map(db):
    store = StashMapStore(db, clock=_Clock())
    content = _content()
    seen = [store.observe(_frame(content, s), [], _header(FILLED)) for s in SCROLLS]
    assert seen[0].how == FIRST
    assert all(s.how != UNPLACED for s in seen)
    cells = store.map.cells
    assert sorted({r for r, _ in cells}) == list(range(ROWS))
    assert len(cells) == ROWS * COLUMNS
    assert {k for k, c in cells.items() if not c.filled} == EMPTY


def test_a_changed_row_updates_only_its_cells(db):
    store = _built(db)
    before = store.map
    changed = _content()
    # Row 3: another item in column 1, column 2 emptied
    _paint(changed, 3, 1, np.random.default_rng(99))
    _paint(changed, 3, 2, None)
    seen = store.observe(_frame(changed, 300), [], _header(FILLED - 1))
    assert seen.changed == ((3, 1), (3, 2))
    after = store.map
    assert _cells_except(before, after, {(3, 1), (3, 2)})
    assert after.cells[3, 1].filled
    assert not after.cells[3, 2].filled
    for key in ((3, 1), (3, 2)):
        assert after.cells[key].last_seen > before.cells[key].last_seen
    assert after.header_used == FILLED - 1


def test_a_frame_that_does_not_place_changes_nothing(db):
    store = _built(db)
    before = store.map
    seen = store.observe(_frame(_content(seed=7), 300), [], _header(3))
    assert seen.how == UNPLACED
    assert seen.changed == ()
    assert _cells_except(before, store.map, set())
    assert store.map.header_used == FILLED
    assert _cells_except(before, StashMapStore(db).map, set())


def test_a_frame_of_another_view_is_ignored(db):
    store = _built(db)
    before = store.map
    seen = store.observe(_frame(_content(), 300, left=LEFT + 400), [], _header(3))
    assert seen.how == OTHER_VIEW
    assert _cells_except(before, store.map, set())


def test_status_is_sure_only_when_the_header_agrees(db):
    store = _built(db)
    status = store.status()
    assert status.sure
    assert status.text == f"map {FILLED} filled, header says {FILLED}"
    store.observe(_frame(_content(), 650), [], _header(FILLED + 2))
    status = store.status()
    assert not status.sure
    assert status.text == f"map {FILLED} filled, header says {FILLED + 2}"


def test_the_map_round_trips_through_the_user_database(db, tmp_path):
    store = _built(db)
    loaded = StashMapStore(Database(db_path=tmp_path / "items.db")).map
    assert loaded[:6] == store.map[:6]
    assert _cells_except(store.map, loaded, set())
    assert db.user_db_path == tmp_path / "arlo_user.db"


def test_a_new_store_places_a_frame_and_updates_one_cell(db):
    _built(db)
    store = StashMapStore(db, clock=_Clock())
    before = store.map
    changed = _content()
    _paint(changed, 7, 0, np.random.default_rng(42))
    seen = store.observe(_frame(changed, 1040), [], _header(FILLED))
    assert seen.how != UNPLACED
    assert seen.changed == ((7, 0),)
    assert _cells_except(before, store.map, {(7, 0)})
    assert _cells_except(store.map, StashMapStore(db).map, set())
