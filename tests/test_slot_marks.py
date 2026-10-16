"""Tests for reading tier numerals and stack counts on slot crops."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from arc_helper.slot_marks import PATCH_FALLBACK
from arc_helper.slot_marks import bar_geometry
from arc_helper.slot_marks import learned_check
from arc_helper.slot_marks import read_stack
from arc_helper.slot_marks import read_stacks
from arc_helper.slot_marks import read_tier
from arc_helper.tooltip_layout import Rect

FIXTURES = Path(__file__).parent / "fixtures" / "slot_marks"
# Real hovered cell crops from the 12 and 14 Oct captures
NUMERALS = {
    "torrente_i": "I",
    "torrente_ii": "II",
    "rattler_iii": "III",
    "burletta_iv": "IV",
    "wide_anvil_iv": "IV",
}
STACKS = {"medium_ammo_x80": 80, "zipline_x1": 1}
# Slots cut from the 12 and 14 Oct captures: x5 at (720, 1376, 912, 1570) of
# 20261012-001659-705_LIGHTAMMO.png, x53 at (512, 544, 704, 736) of the same
# frame, and x3 at (3008, 752, 3200, 944) of 20261014-210206-580_STILETTO-I.png,
# where the bar ends three columns into the 3 (Tesseract read it as 2)
REAL_STACKS = {"stack_x5": 5, "stack_x3_clipped": 3, "stack_x53": 53}
NO_MARK = "grappling_hook_no_mark"


def _slot(name: str) -> np.ndarray:
    with Image.open(FIXTURES / f"{name}.png") as image:
        return np.asarray(image.convert("RGB"))


@pytest.mark.parametrize(("name", "tier"), NUMERALS.items())
def test_read_tier_reads_the_numeral(name: str, tier: str) -> None:
    read = read_tier(_slot(name))
    assert read is not None
    assert read.tier == tier
    assert 0.5 <= read.confidence <= 1.0


@pytest.mark.parametrize("name", [*STACKS, NO_MARK])
def test_read_tier_is_none_without_a_numeral(name: str) -> None:
    # The x1 stack ends in a 1, a single stroke like I, but white and serifless
    assert read_tier(_slot(name)) is None


@pytest.mark.parametrize(("name", "count"), (STACKS | REAL_STACKS).items())
def test_read_stack_reads_the_count(name: str, count: int) -> None:
    assert read_stack(_slot(name)) == count


def test_read_stacks_gives_one_result_per_slot_in_order() -> None:
    crops = [_slot("stack_x3_clipped"), _slot(NO_MARK), _slot("stack_x53")]
    assert read_stacks(crops) == [3, None, 53]


def test_read_stacks_of_no_slots_is_empty() -> None:
    assert read_stacks([]) == []


@pytest.mark.parametrize("name", [*NUMERALS, NO_MARK])
def test_read_stack_is_none_without_a_count(name: str) -> None:
    assert read_stack(_slot(name)) is None


def test_a_slot_without_a_dark_bar_reads_nothing() -> None:
    plain = np.full((200, 200, 3), 120, dtype=np.uint8)
    assert read_tier(plain) is None
    assert read_stack(plain) is None
    assert bar_geometry(plain) is None


# Blocked slots from 20261014-210939-441_252-X22.png, cut at (3216, 752,
# 3408, 946) and (3424, 1424, 3616, 1618): dark tile bodies with a red
# no-entry sign and no bottom bar
@pytest.mark.parametrize("name", ["blocked_slot_a", "blocked_slot_b"])
def test_bar_geometry_is_none_on_a_blocked_slot(name: str) -> None:
    assert bar_geometry(_slot(name)) is None


def test_bar_geometry_is_none_on_a_uniform_dark_crop() -> None:
    dark = np.zeros((190, 190, 3), dtype=np.uint8)
    dark[:] = (9, 12, 25)
    assert bar_geometry(dark) is None


# A synthetic 190 px tile: bright frame, tile body, then a dark bar with a
# white type icon at the left and a white count block at the right
TILE_BODY = (30, 40, 60)
BAR_RGB = (9, 12, 25)
BAR = Rect(6, 150, 184, 186)
ICON = Rect(14, 158, 34, 176)
COUNT = Rect(150, 160, 172, 178)


def _tile(*, count: bool = True) -> np.ndarray:
    tile = np.zeros((190, 190, 3), dtype=np.uint8)
    tile[:] = (90, 160, 220)  # the frame
    tile[3:187, 3:187] = TILE_BODY
    tile[BAR.top : BAR.bottom, BAR.left : BAR.right] = BAR_RGB
    marks = (ICON, COUNT) if count else (ICON,)
    for box in marks:
        tile[box.top : box.bottom, box.left : box.right] = 255
    return tile


def test_bar_geometry_finds_the_bar_its_colour_and_marks() -> None:
    geometry = bar_geometry(_tile())
    assert geometry is not None
    assert geometry.bar_rect == BAR
    assert geometry.colour == BAR_RGB
    assert geometry.icon_box == ICON
    assert geometry.right_mark_box == COUNT
    assert ICON.right < geometry.patch_right < COUNT.left


def test_bar_geometry_without_a_right_mark_falls_back_to_55_percent() -> None:
    geometry = bar_geometry(_tile(count=False))
    assert geometry is not None
    assert geometry.right_mark_box is None
    assert geometry.colour == BAR_RGB
    assert geometry.patch_right == BAR.left + round(PATCH_FALLBACK * BAR.width)


# Real square tile cut from 20261014-211002-366_ARC-MOTION-CORE.png at
# (2088, 544, 2280, 736): medium ammo x90 in the backpack. The reference bar
# region sits between the type icon and the x90, inside the bar.
REAL_TILE = "ammo_x90_tile"
REAL_REFERENCE = Rect(60, 160, 120, 176)


def test_bar_geometry_colour_matches_a_real_bar() -> None:
    tile = _slot(REAL_TILE)
    geometry = bar_geometry(tile)
    assert geometry is not None
    region = tile[
        REAL_REFERENCE.top : REAL_REFERENCE.bottom,
        REAL_REFERENCE.left : REAL_REFERENCE.right,
    ]
    reference = np.median(region.reshape(-1, 3), axis=0)
    assert np.all(np.abs(np.array(geometry.colour) - reference) <= 6)
    bar = geometry.bar_rect
    assert bar.left <= REAL_REFERENCE.left
    assert REAL_REFERENCE.right <= bar.right
    assert bar.top <= REAL_REFERENCE.top
    assert REAL_REFERENCE.bottom <= bar.bottom
    assert geometry.icon_box is not None
    assert geometry.right_mark_box is not None
    assert geometry.icon_box.right < geometry.right_mark_box.left


# The slot at (720, 752, 912, 944) of 20261012-001728-258_BURLETTAIV.png,
# clean and with its price patch drawn by badge_helper.render_png. The
# patch's white worth text lifts the median of rows 170 and 172 above the
# bar threshold, which split the bar before
PATCH_CLEAN = "price_patch_slot_clean"
PATCH_DRAWN = "price_patch_slot_patched"


def test_bar_geometry_is_the_same_with_the_price_patch_drawn() -> None:
    clean = bar_geometry(_slot(PATCH_CLEAN))
    patched = bar_geometry(_slot(PATCH_DRAWN))
    assert clean is not None
    assert patched is not None
    assert patched.bar_rect == clean.bar_rect
    assert patched.right_mark_box == clean.right_mark_box
    assert patched.patch_right == clean.patch_right
    assert np.all(np.abs(np.array(patched.colour) - clean.colour) <= 3)


# Slot crops from Jeffrey's 15 Oct loadout screenshot, stretched to 3840x2160
# (blueprint-checks-4k.png): the Vertical Grip III Blueprint shows the game's
# learned check, the Bettina Blueprint does not
CHECKED_BLUEPRINT = "vertical_grip_iii_blueprint_checked"
UNCHECKED_BLUEPRINT = "bettina_blueprint_unchecked"


def _whole(rgb):
    return Rect(0, 0, rgb.shape[1], rgb.shape[0])


def test_learned_check_reads_the_check_mark():
    rgb = _slot(CHECKED_BLUEPRINT)
    assert learned_check(rgb, _whole(rgb)) is True


def test_learned_check_without_the_mark_is_false():
    rgb = _slot(UNCHECKED_BLUEPRINT)
    assert learned_check(rgb, _whole(rgb)) is False


def test_learned_check_cannot_judge_a_corner_off_the_frame_or_covered():
    rgb = _slot(CHECKED_BLUEPRINT)
    height, width = rgb.shape[:2]
    # The slot reaches past the frame's right edge
    assert learned_check(rgb, Rect(width // 2, 0, width + width // 2, height)) is None
    covered = rgb.copy()
    covered[:, width // 2 :] = (240, 230, 210)  # the cream item card over it
    assert learned_check(covered, _whole(covered)) is None
