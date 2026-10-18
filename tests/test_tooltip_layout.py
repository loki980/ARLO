"""Tests for the pure tooltip geometry used to place the overlay chips."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import arc_helper.ocr as ocr_module
from arc_helper.ocr import CARD_CROP_MARGIN
from arc_helper.ocr import OCREngine
from arc_helper.ocr import crop_card
from arc_helper.ocr import search_window_bbox
from arc_helper.title_reader import TextLine
from arc_helper.tooltip_layout import FOUND_IN
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
from arc_helper.tooltip_layout import find_recycle_icons
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


def test_search_window_near_screen_edge_matches_debug_box(monkeypatch):
    grabbed = []

    def grab(bbox):
        grabbed.append(bbox)
        return Image.new("RGB", (bbox[2] - bbox[0], bbox[3] - bbox[1]))

    cursor = Point(1000, 500)
    monkeypatch.setattr(ocr_module, "get_screen_resolution", lambda: (3840, 2160))
    monkeypatch.setattr(ocr_module, "get_cursor_position", lambda: cursor)
    monkeypatch.setattr(ocr_module, "grab_screen", grab)
    engine = OCREngine()
    engine.search_width, engine.search_height = 2400, 1600
    engine.capture_search_window()
    # The window is cut at the left and top edges, never shifted inwards
    assert grabbed == [search_window_bbox(cursor, 2400, 1600)]
    assert grabbed == [(0, 0, 2200, 1300)]


def test_search_window_of_zero_spans_the_full_screen(monkeypatch):
    grabbed = []

    def grab(bbox):
        grabbed.append(bbox)
        return Image.new("RGB", (bbox[2] - bbox[0], bbox[3] - bbox[1]))

    cursor = Point(1000, 500)
    monkeypatch.setattr(ocr_module, "get_screen_resolution", lambda: (3840, 2160))
    monkeypatch.setattr(ocr_module, "get_cursor_position", lambda: cursor)
    monkeypatch.setattr(ocr_module, "grab_screen", grab)
    engine = OCREngine()
    engine.search_width, engine.search_height = 0, 0
    engine.capture_search_window()
    assert grabbed == [search_window_bbox(cursor, 0, 0)]
    assert grabbed == [(0, 0, 3840, 2160)]


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


SCREEN = Size(3840, 2160)
TOOLTIP = Rect(1000, 500, 1600, 1500)
RECYCLES_LABEL = Rect(1040, 1100, 1260, 1140)
# Two icons of 60 px from x=1040 to 1170, 10 px below the label
ICONS = Rect(1040, 1150, 1170, 1210)
HOME_ONLY = Size(250, 40)


def test_no_price_or_verdict_chip_is_placed():
    labels = {FOUND_IN: Rect(1040, 800, 1300, 840), RECYCLES_INTO: RECYCLES_LABEL}
    sizes = {"price": Size(200, 60), "verdict": Size(180, 70), RECYCLE_CHIP: HOME_ONLY}
    positions = place_chips(TOOLTIP, labels, sizes, SCREEN, 10, ICONS)
    assert set(positions) == {RECYCLE_CHIP}


def test_recycle_chip_centred_right_of_the_icons():
    labels = {RECYCLES_INTO: RECYCLES_LABEL}
    positions = place_chips(
        TOOLTIP, labels, {RECYCLE_CHIP: HOME_ONLY}, SCREEN, 10, ICONS
    )
    # Centre of the row is y=1180, so a 40 px chip starts at 1160
    assert positions == {RECYCLE_CHIP: Point(1170 + 10, 1160)}


def test_recycle_chip_right_aligned_when_it_would_leave_the_card():
    labels = {RECYCLES_INTO: RECYCLES_LABEL}
    # Right of the icons a 450 px chip would end at 1630, past 1600 - 10
    wide = Size(450, 40)
    positions = place_chips(TOOLTIP, labels, {RECYCLE_CHIP: wide}, SCREEN, 10, ICONS)
    assert positions == {RECYCLE_CHIP: Point(1600 - 10 - 450, 1160)}


def test_recycle_chip_wider_than_the_card_is_not_placed():
    labels = {RECYCLES_INTO: RECYCLES_LABEL}
    # The card is 600 px, less two pads leaves 580
    too_wide = Size(581, 40)
    positions = place_chips(
        TOOLTIP, labels, {RECYCLE_CHIP: too_wide}, SCREEN, 10, ICONS
    )
    assert positions == {}


def test_recycle_chip_falls_back_without_icons():
    labels = {RECYCLES_INTO: RECYCLES_LABEL}
    positions = place_chips(TOOLTIP, labels, {RECYCLE_CHIP: HOME_ONLY}, SCREEN, 10)
    assert positions == {RECYCLE_CHIP: Point(1000 + 270, 1140 + 10)}


def test_missing_label_yields_no_chip():
    labels = {FOUND_IN: Rect(1040, 800, 1300, 840)}
    positions = place_chips(TOOLTIP, labels, {RECYCLE_CHIP: HOME_ONLY}, SCREEN, 10)
    assert positions == {}


def test_recycle_chip_clamped_at_the_screen_edge():
    tooltip = Rect(3500, 500, 4100, 1500)
    labels = {RECYCLES_INTO: Rect(3540, 3000, 3700, 3040)}
    positions = place_chips(tooltip, labels, {RECYCLE_CHIP: HOME_ONLY}, SCREEN, 10)
    assert positions[RECYCLE_CHIP] == Point(3840 - 250, 2160 - 40)


# Synthetic card: cream background, a dark label bar and two dark icons of
# 0.06 of the card height, 0.01 of it apart, just under the label
CARD_W, CARD_H = 760, 900
ICON_SIDE = round(0.06 * CARD_H)  # 54 px, 0.071 of the card width
ICON_GAP = round(0.01 * CARD_H)
CARD_LABEL = Rect(50, 600, 300, 625)


def _synthetic_card() -> np.ndarray:
    card = np.full((CARD_H, CARD_W, 3), (240, 232, 216), dtype=np.uint8)
    card[CARD_LABEL.top : CARD_LABEL.bottom, CARD_LABEL.left : CARD_LABEL.right] = (
        30,
        34,
        40,
    )
    top = CARD_LABEL.bottom + 12
    for i in range(2):
        left = 50 + i * (ICON_SIDE + ICON_GAP)
        card[top : top + ICON_SIDE, left : left + ICON_SIDE] = (30, 34, 40)
    return card


def test_find_recycle_icons_on_a_synthetic_card():
    icons = find_recycle_icons(_synthetic_card(), CARD_LABEL)
    top = CARD_LABEL.bottom + 12
    assert icons == Rect(50, top, 50 + 2 * ICON_SIDE + ICON_GAP, top + ICON_SIDE)


def test_find_recycle_icons_none_without_icons():
    card = np.full((CARD_H, CARD_W, 3), (240, 232, 216), dtype=np.uint8)
    assert find_recycle_icons(card, CARD_LABEL) is None


def test_layout_carries_the_icon_row_in_screen_pixels():
    capture = Image.new("RGB", (CARD_W + 100, CARD_H + 100), (20, 20, 30))
    capture.paste(Image.fromarray(_synthetic_card()), (40, 30))
    engine = OCREngine()
    engine._last_capture = capture  # noqa: SLF001 - test seam
    engine._last_origin = Point(2000, 500)  # noqa: SLF001 - test seam
    engine._last_card = Rect(40, 30, 40 + CARD_W, 30 + CARD_H)  # noqa: SLF001
    box = (CARD_LABEL.left, CARD_LABEL.top, CARD_LABEL.right, CARD_LABEL.bottom)
    engine._last_lines = [TextLine("RECYCLES INTO", box)]  # noqa: SLF001

    layout = engine.layout_from_last_capture()

    assert layout is not None
    top = 500 + 30 + CARD_LABEL.bottom + 12
    left = 2000 + 40 + 50
    assert layout.recycle_icons == Rect(
        left, top, left + 2 * ICON_SIDE + ICON_GAP, top + ICON_SIDE
    )
    chip = place_chips(
        layout.tooltip,
        layout.labels,
        {RECYCLE_CHIP: HOME_ONLY},
        SCREEN,
        10,
        layout.recycle_icons,
    )[RECYCLE_CHIP]
    assert chip.x == layout.recycle_icons.right + 10
    assert (
        chip.y + HOME_ONLY.height / 2
        == (layout.recycle_icons.top + layout.recycle_icons.bottom) / 2
    )


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
