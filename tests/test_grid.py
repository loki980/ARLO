"""Tests for the slot lattice and the frame stillness check."""

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from arc_helper.grid import AUGMENTED
from arc_helper.grid import BACKPACK
from arc_helper.grid import EQUIPMENT
from arc_helper.grid import QUICK_USE
from arc_helper.grid import SAFE_POCKET
from arc_helper.grid import STASH
from arc_helper.grid import find_slots
from arc_helper.grid import frames_still

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
