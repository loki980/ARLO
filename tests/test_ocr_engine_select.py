"""Tests for the OCR_ENGINE reader choice and the paddle layout lines."""

import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest
from PIL import Image

from arc_helper import ocr as ocr_module
from arc_helper.config import get_settings
from arc_helper.ocr import OCREngine
from arc_helper.screen import Point
from arc_helper.tooltip_layout import FOUND_IN
from arc_helper.tooltip_layout import RECYCLES_INTO
from arc_helper.tooltip_layout import Rect

FIXTURES = Path(__file__).parent / "fixtures"


class FakeRapidOCR:
    """Stands in for rapidocr_onnxruntime.RapidOCR, returning set lines.

    results is a list of what each call returns, in order. A line is
    (points, text, score) with points in the image it was given.
    """

    results: ClassVar[list[list | None]] = []

    def __call__(self, _image, **_kwargs):
        return FakeRapidOCR.results.pop(0), [0.0, 0.0, 0.0]


@pytest.fixture
def fake_paddle(monkeypatch):
    module = types.ModuleType("rapidocr_onnxruntime")
    module.RapidOCR = FakeRapidOCR
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", module)
    monkeypatch.setattr(FakeRapidOCR, "results", [])
    return FakeRapidOCR


def _engine(monkeypatch, engine: str) -> OCREngine:
    monkeypatch.setattr(get_settings(), "ocr_engine", engine)
    return OCREngine()


def _warnings(monkeypatch) -> list[str]:
    seen: list[str] = []
    monkeypatch.setattr(ocr_module.logger, "warning", seen.append)
    return seen


def test_default_engine_is_tesseract_title():
    assert type(get_settings()).model_fields["ocr_engine"].default == "tesseract_title"


@pytest.mark.parametrize("engine", ["tesseract", "tesseract_title", "paddle"])
def test_engine_setting_picks_reader(monkeypatch, fake_paddle, engine):  # noqa: ARG001
    warnings = _warnings(monkeypatch)
    assert _engine(monkeypatch, engine).reader.name == engine
    assert warnings == []


def test_missing_paddle_falls_back_to_tesseract_title(monkeypatch):
    # None in sys.modules makes the import raise ImportError
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", None)
    warnings = _warnings(monkeypatch)
    assert _engine(monkeypatch, "paddle").reader.name == "tesseract_title"
    assert len(warnings) == 1
    assert "uv sync --extra paddle" in warnings[0]


def test_unknown_engine_falls_back_to_tesseract_title(monkeypatch):
    warnings = _warnings(monkeypatch)
    assert _engine(monkeypatch, "easyocr").reader.name == "tesseract_title"
    assert len(warnings) == 1


# The Anvil capture pasted into a dark 4K frame, as in test_ocr_name. The
# card sits at (482, 462) in the capture, so far from the capture origin, and
# the search window holds it whole, so the OCR crop keeps its full 8 px margin.
FRAME_AT = (1500, 400)
CURSOR = Point(x=1460, y=1000)
ANVIL_CARD = Rect(482, 462, 1031, 1174)
CARD_ON_SCREEN = (FRAME_AT[0] + ANVIL_CARD.left, FRAME_AT[1] + ANVIL_CARD.top)
MARGIN = ocr_module.CARD_CROP_MARGIN


def _box(left, top, right, bottom):
    """Paddle's four corner points for a box in crop coordinates."""
    return [[left, top], [right, top], [right, bottom], [left, bottom]]


def _paddle_on_frame(monkeypatch) -> OCREngine:
    with Image.open(FIXTURES / "anvil_iv_loadout_capture.png") as capture:
        frame = Image.new("RGB", (3840, 2160), (14, 16, 24))
        frame.paste(capture.convert("RGB"), FRAME_AT)
    monkeypatch.setattr(ocr_module, "grab_screen", frame.crop)
    monkeypatch.setattr(ocr_module, "get_cursor_position", lambda: CURSOR)
    monkeypatch.setattr(ocr_module, "get_screen_resolution", lambda: (3840, 2160))

    def no_tesseract(*_args, **_kwargs):
        msg = "the paddle layout must not run Tesseract"
        raise AssertionError(msg)

    monkeypatch.setattr(ocr_module.pytesseract, "image_to_data", no_tesseract)
    engine = _engine(monkeypatch, "paddle")
    engine.debug_mode = False
    engine.capture_library = False
    engine.search_width, engine.search_height = 2400, 1600
    return engine


def test_paddle_lines_give_label_rows_without_tesseract(monkeypatch, fake_paddle):
    engine = _paddle_on_frame(monkeypatch)
    # Boxes in the OCR crop, which starts MARGIN px above and left of the card
    fake_paddle.results = [
        [
            [_box(58, 115, 266, 149), "ANVIL IV", 0.99],
            [_box(30, 400, 260, 424), "CAN BE FOUND IN", 0.95],
            [_box(30, 600, 230, 624), "RECYCLES INTO", 0.95],
        ],
        None,  # the next scan detects nothing
    ]

    assert engine.extract_item_name_at_cursor() == "ANVIL IV"
    assert engine.last_tier_vote is not None
    layout = engine.layout_from_last_capture()

    def on_screen(left, top, right, bottom):
        dx = CARD_ON_SCREEN[0] - MARGIN
        dy = CARD_ON_SCREEN[1] - MARGIN
        return Rect(left + dx, top + dy, right + dx, bottom + dy)

    assert layout.labels == {
        FOUND_IN: on_screen(30, 400, 260, 424),
        RECYCLES_INTO: on_screen(30, 600, 230, 624),
    }

    # A scan that reads nothing leaves no lines from the scan before
    assert engine.extract_item_name_at_cursor() is None
    assert engine.layout_from_last_capture().labels == {}
