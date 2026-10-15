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

# A loadout crop with no card open, so no cell carries the glow ring
NO_RING_FIXTURE = FIXTURES / "loadout_no_card_capture.png"


def _load(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def test_finds_ring_left_of_card() -> None:
    assert find_hovered_cell(_load(AMMO_FIXTURE), AMMO_CARD) == AMMO_CELL


def test_ignore_rect_keeps_chip_out_of_the_box() -> None:
    rgb = _load(CHIP_FIXTURE)

    widened = find_hovered_cell(rgb, CHIP_CARD)
    assert widened == Rect(64, 58, 760, 475)

    assert find_hovered_cell(rgb, CHIP_CARD, ignore=[CHIP_RECT]) == CHIP_CELL


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
