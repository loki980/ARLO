"""Tests for the hovered cell search and the gold frame geometry."""

from pathlib import Path

import numpy as np
from PIL import Image

from arc_helper.cell_border import FrameBars
from arc_helper.cell_border import frame_bars
from arc_helper.hovered_cell import find_hovered_cell
from arc_helper.tooltip_layout import Rect

FIXTURES = Path(__file__).parent / "fixtures"

# Crop (860, 890)-(1200, 1220) of the full-screen capture
# 20261012-001700-899_MEDIUMAMMO.png: a hovered Medium Ammo stash cell with
# the left 48 px of its card. Card and cell rects are in crop coordinates.
AMMO_FIXTURE = FIXTURES / "hovered_cell_ammo_capture.png"
AMMO_CARD = Rect(292, -24, 1051, 361)
AMMO_CELL = Rect(60, 62, 268, 271)

# Crop (1240, 1150)-(2060, 1680) of 20261012-001817-403_ANVILIV.png: the
# hovered Anvil IV weapon slot with ARLO's own HOME ONLY chip lying across
# its left edge, which joins the ring and widens the box by 72 px.
CHIP_FIXTURE = FIXTURES / "hovered_cell_chip_capture.png"
CHIP_CARD = Rect(784, -477, 1544, 752)
CHIP_RECT = Rect(46, 163, 473, 217)
CHIP_CELL = Rect(136, 58, 760, 475)

# Crop (3330, 480)-(3700, 800) of 20261014-210134-998_SHIELD-RECHARGER.png:
# the hovered Shield Recharger quick use slot, with its card opened to the
# left of the cell (only the card's right 62 px are in the crop).
LEFT_OPEN_FIXTURE = FIXTURES / "hovered_cell_left_open_capture.png"
LEFT_OPEN_CARD = Rect(-698, -111, 62, 674)
LEFT_OPEN_CELL = Rect(86, 56, 294, 265)

# Crop (60, 80)-(700, 820) of
# 20261014-210223-142_U-AMMO-TURE-N-HEAVY-AMMO-MAGAZINE-SIZE-6.png: the
# hovered Anvil I stash cell with ARLO's corner popup over its top 83 px.
# The cell is the by-eye label of the whole ring, hidden top included.
POPUP_FIXTURE = FIXTURES / "hovered_cell_popup_capture.png"
POPUP_CARD = Rect(468, 234, 1227, 1202)
POPUP_CELL = Rect(236, 456, 444, 665)

# Crop (2080, 960)-(3230, 1640) of 20261014-211019-500_GRAPPLING-HOOK.png:
# the right-click ACTIONS menu for the Grappling Hook in the Safe Pocket,
# opened left of the cell over the backpack. Left of the card, the slanted
# grip of a backpack icon cut off by the search window passes the line test.
MENU_FIXTURE = FIXTURES / "hovered_cell_menu_capture.png"
MENU_CARD = Rect(136, 120, 895, 801)
MENU_CELL = Rect(920, 456, 1128, 665)

# A detection matches a by-eye label when every edge is within this many px
MATCH_PX = 12

# A loadout crop with no card open, so no cell carries the glow ring
NO_RING_FIXTURE = FIXTURES / "loadout_no_card_capture.png"


def _load(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def test_finds_ring_left_of_card() -> None:
    assert find_hovered_cell(_load(AMMO_FIXTURE), AMMO_CARD) == AMMO_CELL


def _matches(found: Rect | None, label: Rect) -> bool:
    return found is not None and all(
        abs(a - b) <= MATCH_PX for a, b in zip(found, label, strict=True)
    )


def test_chip_stays_out_of_the_box_with_or_without_its_ignore_rect() -> None:
    rgb = _load(CHIP_FIXTURE)

    # the chip is found by its flat colours even when no ignore rect names it
    assert find_hovered_cell(rgb, CHIP_CARD) == CHIP_CELL
    assert find_hovered_cell(rgb, CHIP_CARD, ignore=[CHIP_RECT]) == CHIP_CELL


def test_finds_ring_right_of_a_card_opened_left() -> None:
    assert find_hovered_cell(_load(LEFT_OPEN_FIXTURE), LEFT_OPEN_CARD) == LEFT_OPEN_CELL


def test_slanted_icon_left_of_the_actions_menu_is_not_a_seed() -> None:
    assert find_hovered_cell(_load(MENU_FIXTURE), MENU_CARD) == MENU_CELL


def test_ring_under_the_corner_popup_is_squared_up() -> None:
    found = find_hovered_cell(_load(POPUP_FIXTURE), POPUP_CARD)

    assert _matches(found, POPUP_CELL)


def test_none_without_a_ring() -> None:
    rgb = _load(NO_RING_FIXTURE)

    assert find_hovered_cell(rgb, Rect(600, 300, 1031, 900)) is None


def test_none_for_an_empty_image() -> None:
    assert find_hovered_cell(np.zeros((0, 0, 3), np.uint8), AMMO_CARD) is None


def test_frame_bars_sit_wholly_outside_the_rect() -> None:
    assert frame_bars(Rect(100, 200, 300, 350), 6) == FrameBars(
        top=Rect(94, 194, 306, 200),
        bottom=Rect(94, 350, 306, 356),
        left=Rect(94, 200, 100, 350),
        right=Rect(300, 200, 306, 350),
    )
