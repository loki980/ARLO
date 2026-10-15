"""Tests for the stash and loadout header readers."""

from pathlib import Path

import numpy as np
from PIL import Image

from arc_helper.grid import EQUIPMENT
from arc_helper.grid import STASH
from arc_helper.grid import Slot
from arc_helper.inventory_header import LoadoutHeader
from arc_helper.inventory_header import StashHeader
from arc_helper.inventory_header import read_loadout_header
from arc_helper.inventory_header import read_stash_header
from arc_helper.tooltip_layout import Rect

FIXTURES = Path(__file__).parent / "fixtures"
BAND_TOP = 250
# Both fixtures are rows 250 to 450 of a 3840x2160 frame (Jeffrey's stash and
# loadout screenshot upscaled from 2000x1125, and his in-raid CONTAINER
# screenshot the same way). Left edges of the first slot columns as
# find_slots reports them on the full frames.
FULL = FIXTURES / "inventory_header_full.png"
CONTAINER = FIXTURES / "inventory_header_container.png"
FULL_STASH_LEFT = 302
FULL_EQUIPMENT_LEFT = 1384
CONTAINER_STASH_LEFT = 314
CONTAINER_EQUIPMENT_LEFT = 1292


def _frame(fixture: Path) -> np.ndarray:
    band = np.array(Image.open(fixture).convert("RGB"))
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    frame[BAND_TOP : BAND_TOP + band.shape[0]] = band
    return frame


def _slots(panel: str, left: int) -> list[Slot]:
    return [Slot(Rect(left, 544, left + 196, 740), panel, filled=True)]


def _cover(frame: np.ndarray, left: int, right: int) -> np.ndarray:
    """Paint a light card over the counts row between two x positions."""
    out = frame.copy()
    out[335:400, left:right] = 200
    return out


def test_stash_header_visible():
    result = read_stash_header(_frame(FULL), _slots(STASH, FULL_STASH_LEFT))
    assert result == StashHeader(used=270, capacity=280, value=849247)


def test_stash_header_absent_in_container_view():
    result = read_stash_header(_frame(CONTAINER), _slots(STASH, CONTAINER_STASH_LEFT))
    assert result is None


def test_stash_header_none_without_slots():
    assert read_stash_header(_frame(FULL), []) is None


def test_stash_header_unreadable_when_value_is_covered():
    # The stash value spans about x 565 to 690 (the screenshot's 293 to 358
    # times 1.92), the card covers all of it
    frame = _cover(_frame(FULL), 540, 720)
    assert read_stash_header(frame, _slots(STASH, FULL_STASH_LEFT)) is None


def test_loadout_header_visible():
    result = read_loadout_header(_frame(FULL), _slots(EQUIPMENT, FULL_EQUIPMENT_LEFT))
    assert result == LoadoutHeader(weight=34.6, weight_cap=90.0, value=75629)


def test_loadout_header_absent_in_container_view():
    result = read_loadout_header(
        _frame(CONTAINER), _slots(EQUIPMENT, CONTAINER_EQUIPMENT_LEFT)
    )
    assert result is None


def test_loadout_header_none_without_slots():
    assert read_loadout_header(_frame(FULL), []) is None


def test_loadout_header_unreadable_when_value_is_covered():
    # The loadout value spans about x 1795 to 1945, the card covers all of it
    frame = _cover(_frame(FULL), 1780, 1960)
    assert read_loadout_header(frame, _slots(EQUIPMENT, FULL_EQUIPMENT_LEFT)) is None


def test_stash_header_none_when_trailing_digits_are_covered():
    # Black over x 600 to 800 hides the end of 849,247, which would read as 84
    frame = _frame(FULL)
    frame[335:400, 600:800] = 0
    assert read_stash_header(frame, _slots(STASH, FULL_STASH_LEFT)) is None


def test_loadout_header_none_when_trailing_digits_are_covered():
    # Black over x 1850 to 2000 hides the end of 75,629, which would read as 7
    frame = _frame(FULL)
    frame[335:400, 1850:2000] = 0
    assert read_loadout_header(frame, _slots(EQUIPMENT, FULL_EQUIPMENT_LEFT)) is None
