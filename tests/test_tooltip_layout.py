"""Tests for the pure tooltip geometry used to place the overlay chips."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from arc_helper.ocr import CARD_CROP_MARGIN
from arc_helper.ocr import crop_card
from arc_helper.tooltip_layout import FOUND_IN
from arc_helper.tooltip_layout import PRICE_CHIP
from arc_helper.tooltip_layout import RECYCLE_CHIP
from arc_helper.tooltip_layout import RECYCLES_INTO
from arc_helper.tooltip_layout import CreamRegion
from arc_helper.tooltip_layout import Edges
from arc_helper.tooltip_layout import Point
from arc_helper.tooltip_layout import Rect
from arc_helper.tooltip_layout import Size
from arc_helper.tooltip_layout import Word
from arc_helper.tooltip_layout import find_cream_rect
from arc_helper.tooltip_layout import find_label_rows
from arc_helper.tooltip_layout import find_tooltip_card
from arc_helper.tooltip_layout import place_chips
from arc_helper.tooltip_layout import to_screen

CREAM = (249, 238, 223)
FIXTURE = Path(__file__).parent / "fixtures" / "battery_tooltip_capture.png"
# Real 1031x1174 captures from the loadout screen. In the first the Anvil IV
# card sits right of the loadout column over the backpack grid, cut off on
# the right and bottom. The second has no card at all.
ANVIL_FIXTURE = Path(__file__).parent / "fixtures" / "anvil_iv_loadout_capture.png"
NO_CARD_FIXTURE = Path(__file__).parent / "fixtures" / "loadout_no_card_capture.png"


def _line(text: str, line: int, left: int, top: int) -> list[Word]:
    """Word boxes for one line, 20 px per character, 30 px tall."""
    words, x = [], left
    for part in text.split():
        words.append(Word(part, x, top, 20 * len(part), 30, line))
        x += 20 * (len(part) + 1)
    return words


def test_cream_rect_on_dark_frame():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    frame[40:160, 100:280] = CREAM
    frame[40:160, 96:100] = (40, 200, 60)  # colored border, not cream
    assert find_cream_rect(frame) == CreamRegion(Rect(100, 40, 280, 160), Edges())


def test_cream_touching_right_edge_is_flagged_clipped():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    frame[40:160, 100:300] = CREAM  # runs into the last column
    region = find_cream_rect(frame)
    assert region == CreamRegion(Rect(100, 40, 300, 160), Edges(right=True))


def test_no_cream_returns_none():
    assert find_cream_rect(np.full((200, 300, 3), 30, dtype=np.uint8)) is None


def _fixture_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def test_card_found_in_real_loadout_capture():
    # Cursor over the loadout weapon, left of the card
    card = find_tooltip_card(_fixture_rgb(ANVIL_FIXTURE), Point(300, 700))
    assert card is not None
    assert 450 <= card.left <= 520
    # The darker ACTIONS tab is outside the cream bounds, so the card starts
    # below it, at the cream body
    assert 440 <= card.top <= 480
    # The ANVIL IV name sits at about x 530 to 740, y 565 to 605
    assert card.left < 530
    assert card.right > 740
    assert card.top < 565
    assert card.bottom > 605


def test_no_card_in_real_loadout_capture():
    assert find_tooltip_card(_fixture_rgb(NO_CARD_FIXTURE), Point(300, 700)) is None


def test_crop_card_adds_margin_and_clamps_to_image():
    with Image.open(ANVIL_FIXTURE) as image:
        rgb = image.convert("RGB")
    # Cursor at the image centre, as eval_captures assumes
    found = crop_card(rgb, Point(rgb.width // 2, rgb.height // 2))
    assert found is not None
    crop, card = found
    # The card is cut off on the right and bottom, so the margin only
    # applies on the left and top
    assert crop.size == (
        rgb.width - card.left + CARD_CROP_MARGIN,
        rgb.height - card.top + CARD_CROP_MARGIN,
    )
    with Image.open(NO_CARD_FIXTURE) as image:
        assert crop_card(image.convert("RGB"), Point(515, 587)) is None


def test_card_nearest_cursor_wins():
    frame = np.zeros((600, 1200, 3), dtype=np.uint8)
    frame[50:550, 700:1150] = CREAM  # large panel, far from the cursor
    frame[100:400, 100:400] = CREAM  # card beside the cursor
    assert find_tooltip_card(frame, Point(450, 250)) == Rect(100, 100, 400, 400)


def test_cream_outline_is_not_a_card():
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    frame[50:350, 50:350] = CREAM
    frame[60:340, 60:340] = 0  # only a 10 px frame stays cream
    assert find_tooltip_card(frame, Point(200, 200)) is None


def test_label_rows_tolerate_ocr_typo_and_divide_by_scale():
    words = (
        _line("USED TO CRAFT", 0, 40, 100)
        + _line("CAN BE FOUND IN", 1, 40, 200)
        + _line("RECYCLES INT0", 2, 40, 400)
    )
    labels = find_label_rows(words, scale=2)
    assert labels == {
        FOUND_IN: Rect(20, 100, 170, 115),
        RECYCLES_INTO: Rect(20, 200, 150, 215),
    }


def test_unrelated_line_matches_nothing():
    assert find_label_rows(_line("USED TO CRAFT", 0, 40, 100), scale=2) == {}


def test_to_screen_adds_origin():
    assert to_screen(Rect(10, 20, 110, 70), Point(2400, 900)) == Rect(
        2410, 920, 2510, 970
    )


def test_place_chips_beside_labels():
    tooltip = Rect(1000, 500, 1600, 1500)
    labels = {
        FOUND_IN: Rect(1040, 800, 1300, 840),
        RECYCLES_INTO: Rect(1040, 1100, 1260, 1140),
    }
    sizes = {PRICE_CHIP: Size(200, 60), RECYCLE_CHIP: Size(250, 40)}
    positions = place_chips(tooltip, labels, sizes, Size(3840, 2160), pad=10)
    assert positions == {
        PRICE_CHIP: Point(1600 - 10 - 200, 800),
        RECYCLE_CHIP: Point(1000 + 270, 1140 + 10),
    }


def test_price_chip_clamped_at_right_screen_edge():
    tooltip = Rect(3500, 500, 4100, 1500)  # right edge past the screen
    labels = {FOUND_IN: Rect(3540, 800, 3800, 840)}
    positions = place_chips(
        tooltip, labels, {PRICE_CHIP: Size(400, 60)}, Size(3840, 2160), pad=10
    )
    assert positions[PRICE_CHIP] == Point(3840 - 400, 800)


def test_wide_price_chip_never_covers_the_label():
    tooltip = Rect(1000, 500, 1600, 1500)
    labels = {FOUND_IN: Rect(1040, 800, 1300, 840)}
    # Right-aligned, a 450 px chip would start at 1140, over the label
    positions = place_chips(
        tooltip, labels, {PRICE_CHIP: Size(450, 60)}, Size(3840, 2160), pad=10
    )
    assert positions[PRICE_CHIP] == Point(1300 + 10, 800)


def test_clipped_tooltip_anchors_price_chip_after_label():
    tooltip = Rect(1000, 500, 1600, 1500)
    labels = {FOUND_IN: Rect(1040, 800, 1300, 840)}
    positions = place_chips(
        tooltip,
        labels,
        {PRICE_CHIP: Size(200, 60)},
        Size(3840, 2160),
        pad=10,
        clipped=Edges(right=True),
    )
    assert positions[PRICE_CHIP] == Point(1300 + 10, 800)


def test_missing_label_yields_no_chip():
    tooltip = Rect(1000, 500, 1600, 1500)
    labels = {FOUND_IN: Rect(1040, 800, 1300, 840)}
    sizes = {PRICE_CHIP: Size(200, 60), RECYCLE_CHIP: Size(250, 40)}
    positions = place_chips(tooltip, labels, sizes, Size(3840, 2160), pad=10)
    assert set(positions) == {PRICE_CHIP}


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="needs tesseract")
def test_real_tooltip_capture_finds_both_labels():
    from arc_helper.ocr import OCREngine

    # The fixture is a capture region from a 4K screenshot downscaled to
    # 1877 px wide. Scale it back to physical pixels like a real grab.
    shot = Image.open(FIXTURE).convert("RGB")
    factor = 3840 / 1877
    capture = shot.resize(
        (round(shot.width * factor), round(shot.height * factor)),
        Image.Resampling.BICUBIC,
    )
    engine = OCREngine()
    engine._last_capture = capture  # noqa: SLF001 - test seam
    engine._last_origin = Point(2394, 900)  # noqa: SLF001 - test seam

    layout = engine.layout_from_last_capture()

    assert layout is not None
    assert layout.tooltip.left > 2394  # border columns trimmed
    assert set(layout.labels) == {FOUND_IN, RECYCLES_INTO}
    assert layout.clipped.right  # the capture is narrower than the tooltip
    assert layout.labels[FOUND_IN].bottom < layout.labels[RECYCLES_INTO].top
