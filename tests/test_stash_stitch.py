"""Tests for stash_stitch: synthetic frames of one stash panel scrolled by known offsets."""

import numpy as np

from arc_helper.stash_stitch import LOOKS
from arc_helper.stash_stitch import NEW_SEGMENT
from arc_helper.stash_stitch import SCROLL
from arc_helper.stash_stitch import filled_cells
from arc_helper.stash_stitch import slot_look
from arc_helper.stash_stitch import stitch

# A 4K frame with a stash viewport like the real one: 4 columns of 192 px
# slots on a 208 px pitch from x 304, shown between y 510 and 1900. The
# content above the viewport's top is the whole stash, its row 0 at 34 px,
# so a frame scrolled by 0 shows row 0 at y 544, as on Jeffrey's run.
FRAME = (2160, 3840)
VIEW_TOP, VIEW_BOTTOM = 510, 1900
LEFT = 304
SIDE = 192
PITCH = 208
OUTLINE = 4
FIRST_ROW = 34
COLUMNS = 4
BACKGROUND = (27, 27, 35)


def _content(rows: int, seed: int, empty: set[tuple[int, int]]) -> np.ndarray:
    """The whole stash as one tall image: every filled slot its own random icon."""
    rng = np.random.default_rng(seed)
    height = FIRST_ROW + rows * PITCH + VIEW_BOTTOM
    width = COLUMNS * PITCH
    image = np.full((height, width, 3), BACKGROUND, np.uint8)
    for row in range(rows):
        for col in range(COLUMNS):
            top, left = FIRST_ROW + row * PITCH, col * PITCH
            image[top : top + SIDE, left : left + SIDE] = (160, 160, 165)
            inner = slice(top + OUTLINE, top + SIDE - OUTLINE)
            across = slice(left + OUTLINE, left + SIDE - OUTLINE)
            image[inner, across] = (35, 38, 52)
            if (row, col) in empty:
                continue
            cells = rng.integers(0, 256, (13, 18, 3), dtype=np.uint8)
            icon = np.repeat(np.repeat(cells, 8, 0), 8, 1)
            image[top + 40 : top + 144, left + 24 : left + 168] = icon
    return image


def _frame(content: np.ndarray, scroll: int) -> np.ndarray:
    """A full frame showing the content scrolled up by scroll px."""
    frame = np.full((*FRAME, 3), BACKGROUND, np.uint8)
    window = content[scroll : scroll + VIEW_BOTTOM - VIEW_TOP]
    frame[VIEW_TOP:VIEW_BOTTOM, LEFT : LEFT + window.shape[1]] = window
    return frame


EMPTY = {(9, 2), (9, 3), (10, 0), (10, 1), (10, 2), (10, 3)}


def test_scrolled_frames_get_absolute_rows():
    content = _content(11, 1, EMPTY)
    scrolls = [0, 300, 650, 1040]
    result = stitch(_frame(content, s) for s in scrolls)
    assert len(result.maps) == 1
    view_map = result.maps[0]
    assert len(view_map.segments) == 1
    assert [n.how for n in result.frames[1:]] == [SCROLL] * 3
    for note, scroll in zip(result.frames, scrolls, strict=True):
        assert abs(note.scroll - scroll) <= 2
    cells = result.cells
    # Frame 3 shows rows 5 to 10 whole: rows 0 to 10 are all in the map
    assert sorted({r for r, _ in cells}) == list(range(11))
    assert len(cells) == 11 * COLUMNS
    assert {key for key, c in cells.items() if not c.filled} == EMPTY
    assert filled_cells(cells) == 11 * COLUMNS - len(EMPTY)


def test_an_overlap_is_counted_once():
    content = _content(10, 2, set())
    # The second frame shows rows 3 to 8, the first rows 0 to 5: rows 3 to 5
    # are on both
    result = stitch(_frame(content, s) for s in (0, 3 * PITCH))
    cells = result.cells
    assert len(cells) == 9 * COLUMNS
    shared = {key for key, c in cells.items() if c.seen == 2}
    assert shared == {(r, c) for r in (3, 4, 5) for c in range(COLUMNS)}
    overlap = result.maps[0].segments[0].overlaps[0]
    assert overlap.shared == 12
    assert overlap.agree == 12
    assert overlap.disagree == 0


def test_a_long_scroll_is_placed_by_slot_looks():
    content = _content(10, 5, set())
    # Two rows in common out of six: too far for the phase correlation
    jump = 4 * PITCH + 20
    result = stitch(_frame(content, s) for s in (0, jump))
    note = result.frames[1]
    assert note.how == LOOKS
    assert abs(note.scroll - jump) <= 2
    assert sorted({r for r, _ in result.cells}) == list(range(10))


def test_an_unmeasurable_offset_starts_a_new_segment():
    content = _content(10, 3, set())
    other = _content(10, 4, set())
    frames = [
        _frame(content, 0),
        _frame(other, 2 * PITCH),
        _frame(other, 2 * PITCH + 120),
    ]
    result = stitch(frames)
    notes = result.frames
    assert notes[1].how == NEW_SEGMENT
    assert notes[1].segment == 1
    assert notes[2].how == SCROLL
    assert notes[2].segment == 1
    segments = result.maps[0].segments
    assert len(segments) == 2
    assert segments[0].frames == (0,)
    assert segments[1].frames == (1, 2)
    # The first segment has only the first frame's six rows, nothing guessed
    assert sorted({r for r, _ in result.cells}) == list(range(6))


def test_arlo_drawings_on_a_slot_leave_its_look_alone():
    content = _content(1, seed=3, empty=set())
    slot = content[FIRST_ROW : FIRST_ROW + SIDE, 0:SIDE]
    rng = np.random.default_rng(4)
    looks = []
    for _ in range(2):
        drawn = slot.copy()
        # The have/need tag at its largest (badge_helper: text 0.17 of the
        # height, the pill down to 1.9 of that plus 1 px), across the top
        text = round(0.17 * SIDE)
        tag = slice(round(0.3 * text), round(1.9 * text) + 1)
        drawn[tag, 8 : SIDE - 8] = rng.integers(0, 256, (1, SIDE - 16, 3), np.uint8)
        # The old red short-stock outline, 5 px inside the edge
        red = (255, 92, 77)
        drawn[:5] = drawn[-5:] = drawn[:, :5] = drawn[:, -5:] = red
        # A price patch on the bar, from 0.734 of the height
        drawn[round(0.734 * SIDE) :, 8:90] = rng.integers(0, 256, 3, np.uint8)
        looks.append(slot_look(drawn))
    assert looks[0] is not None
    np.testing.assert_array_equal(looks[0], looks[1])
    np.testing.assert_array_equal(looks[0], slot_look(slot))
