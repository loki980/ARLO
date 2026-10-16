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
# In-raid frame (Jeffrey's screenshot upscaled the same way), rows 250 to 480.
# Its layout is 48 px lower than the base one, so the first slots start at
# y 592 where the base ones start at 544.
RAID = FIXTURES / "inventory_header_raid.png"
RAID_EQUIPMENT_LEFT = 1292
RAID_SLOT_TOP = 592
BASE_SLOT_TOP = 544
FULL_STASH_LEFT = 302
FULL_EQUIPMENT_LEFT = 1384
CONTAINER_STASH_LEFT = 314
CONTAINER_EQUIPMENT_LEFT = 1292


def _frame(fixture: Path) -> np.ndarray:
    band = np.array(Image.open(fixture).convert("RGB"))
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    frame[BAND_TOP : BAND_TOP + band.shape[0]] = band
    return frame


def _slots(panel: str, left: int, top: int = BASE_SLOT_TOP) -> list[Slot]:
    return [Slot(Rect(left, top, left + 196, top + 196), panel, filled=True)]


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


def test_loadout_header_read_in_raid_layout():
    # The counts row is placed 209 px above the first slot row, so slots at
    # 592 put it 48 px lower than on the base screen
    slots = _slots(EQUIPMENT, RAID_EQUIPMENT_LEFT, RAID_SLOT_TOP)
    result = read_loadout_header(_frame(RAID), slots)
    assert result == LoadoutHeader(weight=39.5, weight_cap=90.0, value=116685)


def test_loadout_header_ignores_row_at_the_old_fixed_height():
    # The same frame with slots reported at the base height looks at the
    # wrong row, which shows the row follows the slots
    slots = _slots(EQUIPMENT, RAID_EQUIPMENT_LEFT, BASE_SLOT_TOP)
    assert read_loadout_header(_frame(RAID), slots) is None


def test_loadout_header_read_with_card_over_right_edge():
    # The value ends at about x 1945, the card from 2000 on only covers the
    # blank end of the crop, which is trimmed off
    frame = _cover(_frame(FULL), 2000, 2100)
    result = read_loadout_header(frame, _slots(EQUIPMENT, FULL_EQUIPMENT_LEFT))
    assert result == LoadoutHeader(weight=34.6, weight_cap=90.0, value=75629)


def test_loadout_header_read_with_card_just_after_the_last_digit():
    # The value ends at about x 1945, the card starts at 1950: five blank
    # columns are enough, the number is whole
    frame = _cover(_frame(FULL), 1950, 2100)
    result = read_loadout_header(frame, _slots(EQUIPMENT, FULL_EQUIPMENT_LEFT))
    assert result == LoadoutHeader(weight=34.6, weight_cap=90.0, value=75629)


def test_loadout_header_none_when_card_starts_between_value_digits():
    # A card from x 1910 hides the last digits behind a clean cut, which
    # would otherwise read as a number of its own
    frame = _cover(_frame(FULL), 1910, 2100)
    assert read_loadout_header(frame, _slots(EQUIPMENT, FULL_EQUIPMENT_LEFT)) is None


def test_loadout_header_none_when_whole_row_is_covered():
    frame = _cover(_frame(FULL), 1500, 2200)
    assert read_loadout_header(frame, _slots(EQUIPMENT, FULL_EQUIPMENT_LEFT)) is None
