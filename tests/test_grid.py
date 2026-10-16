"""Tests for the slot lattice and the frame stillness check."""

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageDraw

from arc_helper.cell_border import HOVER_GOLD
from arc_helper.cell_border import frame_bars
from arc_helper.grid import AUGMENTED
from arc_helper.grid import BACKPACK
from arc_helper.grid import EQUIPMENT
from arc_helper.grid import QUICK_USE
from arc_helper.grid import SAFE_POCKET
from arc_helper.grid import STASH
from arc_helper.grid import find_slots
from arc_helper.grid import frames_still
from arc_helper.stash_scan import BORDER_RADIUS
from arc_helper.stash_scan import BORDER_WIDTH
from arc_helper.stash_scan import SHORT_RED

FIXTURES = Path(__file__).parent / "fixtures"

# 20261014-210150-139_ZIPLINE.png scaled to 1920x1080 (64 colours): the
# Zipline card lies over most of the backpack, so only its left column of
# four slots shows. Counts per panel are by eye on the full capture.
BACKPACK_CARD_FIXTURE = FIXTURES / "grid_backpack_card_capture.png"
BACKPACK_CARD_COUNTS = {
    STASH: 21,
    EQUIPMENT: 4,
    BACKPACK: 4,
    QUICK_USE: 5,
    AUGMENTED: 1,
    SAFE_POCKET: 3,
}

# 20261012-001825-426_none.png scaled the same way: the Anvil I card lies
# over the stash, leaving its left column and three slots below the card
# (the fourth lies under ARLO's HOME ONLY chip).
STASH_CARD_FIXTURE = FIXTURES / "grid_stash_card_capture.png"
STASH_CARD_COUNTS = {
    STASH: 7,
    EQUIPMENT: 4,
    BACKPACK: 20,
    QUICK_USE: 5,
    AUGMENTED: 1,
    SAFE_POCKET: 3,
}


def _load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _assert_counts(path: Path, expected: dict[str, int]) -> None:
    counts = Counter(slot.panel for slot in find_slots(_load(path)))
    for panel, count in expected.items():
        assert abs(counts[panel] - count) <= 1, (panel, counts[panel], count)


def test_find_slots_with_card_over_backpack() -> None:
    _assert_counts(BACKPACK_CARD_FIXTURE, BACKPACK_CARD_COUNTS)


def test_find_slots_with_card_over_stash() -> None:
    _assert_counts(STASH_CARD_FIXTURE, STASH_CARD_COUNTS)


def test_slots_are_square_grid_cells_or_equipment_shapes() -> None:
    slots = find_slots(_load(BACKPACK_CARD_FIXTURE))
    stash = [slot.rect for slot in slots if slot.panel == STASH]
    assert all(abs(rect.width - rect.height) <= 2 for rect in stash)
    assert all(90 <= rect.width <= 100 for rect in stash)


def test_frames_still_on_equal_frames() -> None:
    frame = _load(BACKPACK_CARD_FIXTURE)
    slots = find_slots(frame)
    assert frames_still(frame, frame.copy(), slots)
    assert frames_still(frame, frame.copy())


def test_frames_still_false_on_shifted_frames() -> None:
    frame = _load(BACKPACK_CARD_FIXTURE)
    slots = find_slots(frame)
    # The grid scrolled by half a slot: every row moved 48 px up
    shifted = np.roll(frame, -48, axis=0)
    assert not frames_still(frame, shifted, slots)


def test_frames_still_ignores_small_noise() -> None:
    frame = _load(BACKPACK_CARD_FIXTURE)
    noise = np.random.default_rng(1).integers(-2, 3, frame.shape)
    noisy = np.clip(frame.astype(int) + noise, 0, 255).astype(np.uint8)
    assert frames_still(frame, noisy, find_slots(frame))


def test_frames_still_false_on_size_change() -> None:
    frame = _load(BACKPACK_CARD_FIXTURE)
    assert not frames_still(frame, frame[:-8])


# A synthetic 4K grid: 4 by 4 slots of 192 px with a 4 px light outline and
# a 16 px gap, in the stash column below the panel headings. A light line
# along the top row's top edge joins its outlines into one component too wide
# to be a slot, so that row is found as neighbours of the row below, where
# the thicker red outline moved the rects by 2 px.
SYNTH_SIZE = (3840, 2160)
SYNTH_ORIGIN = (400, 640)
SYNTH_SIDE = 192
SYNTH_PITCH = 208
SYNTH_OUTLINE = 4
SYNTH_GOLD_WIDTH = 6


def _synthetic_grid() -> np.ndarray:
    width, height = SYNTH_SIZE
    frame = np.full((height, width, 3), (27, 27, 35), np.uint8)
    for row in range(4):
        for col in range(4):
            left = SYNTH_ORIGIN[0] + col * SYNTH_PITCH
            top = SYNTH_ORIGIN[1] + row * SYNTH_PITCH
            frame[top : top + SYNTH_SIDE, left : left + SYNTH_SIDE] = (160, 160, 165)
            inner = slice(top + SYNTH_OUTLINE, top + SYNTH_SIDE - SYNTH_OUTLINE)
            across = slice(left + SYNTH_OUTLINE, left + SYNTH_SIDE - SYNTH_OUTLINE)
            frame[inner, across] = (35, 38, 52)
    top = SYNTH_ORIGIN[1]
    frame[top : top + SYNTH_OUTLINE, SYNTH_ORIGIN[0] - 100 : 1400] = (160, 160, 165)
    return frame


def _rgb(colour: str) -> tuple[int, int, int]:
    return tuple(int(colour[i : i + 2], 16) for i in (1, 3, 5))


def _red_outline(frame: np.ndarray, rect) -> np.ndarray:
    """frame with ARLO's short-stock outline drawn inside rect."""
    image = Image.fromarray(frame)
    ImageDraw.Draw(image).rounded_rectangle(
        (rect.left, rect.top, rect.right - 1, rect.bottom - 1),
        radius=BORDER_RADIUS,
        outline=_rgb(SHORT_RED),
        width=BORDER_WIDTH,
    )
    return np.asarray(image)


def _gold_frame(frame: np.ndarray, rect) -> np.ndarray:
    """frame with the hovered gold frame drawn outside rect, as CellBorder does."""
    out = frame.copy()
    for bar in frame_bars(rect, SYNTH_GOLD_WIDTH):
        out[bar.top : bar.bottom, bar.left : bar.right] = _rgb(HOVER_GOLD)
    return out


def test_synthetic_grid_slots() -> None:
    rects = [slot.rect for slot in find_slots(_synthetic_grid())]
    assert len(rects) == 16
    assert all(abs(rect.width - SYNTH_SIDE) <= 2 for rect in rects)


def test_red_outline_keeps_slot_rects() -> None:
    frame = _synthetic_grid()
    clean = [slot.rect for slot in find_slots(frame)]
    for rect in (clean[0], clean[3], clean[5], clean[15]):
        assert [slot.rect for slot in find_slots(_red_outline(frame, rect))] == clean


def test_red_outline_on_every_slot_keeps_slot_rects() -> None:
    frame = _synthetic_grid()
    clean = [slot.rect for slot in find_slots(frame)]
    badged = frame
    for rect in clean:
        badged = _red_outline(badged, rect)
    assert [slot.rect for slot in find_slots(badged)] == clean


def test_gold_frame_keeps_slot_rects() -> None:
    # The top row's rects lie 2 px inside its outlines (they are found as
    # neighbours), so a gold frame there would cover the outline itself
    frame = _synthetic_grid()
    clean = [slot.rect for slot in find_slots(frame)]
    for rect in (clean[4], clean[5], clean[11], clean[15]):
        assert [slot.rect for slot in find_slots(_gold_frame(frame, rect))] == clean
