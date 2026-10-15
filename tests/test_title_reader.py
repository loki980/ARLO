"""Tests for the card title readers."""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from arc_helper.ocr import OCREngine
from arc_helper.ocr import crop_card
from arc_helper.title_reader import TesseractReader
from arc_helper.title_reader import TesseractTitleReader
from arc_helper.title_reader import TitleReaderUnavailableError
from arc_helper.title_reader import binarize_card
from arc_helper.title_reader import find_title_lines
from arc_helper.title_reader import fix_trailing_token
from arc_helper.title_reader import make_reader
from arc_helper.tooltip_layout import Point

FIXTURES = Path(__file__).parent / "fixtures"
CARD_FIXTURES = ["anvil_iv_loadout_capture.png", "battery_tooltip_capture.png"]
needs_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="needs tesseract"
)


def _card(fixture: str) -> Image.Image:
    """The card crop the scanner would read from a fixture capture."""
    with Image.open(FIXTURES / fixture) as capture:
        rgb = capture.convert("RGB")
    found = crop_card(rgb, Point(rgb.width // 2, rgb.height // 2))
    assert found is not None
    return found[0]


@pytest.mark.parametrize("fixture", CARD_FIXTURES)
def test_binarize_card_matches_preprocess_tooltip(fixture):
    card = _card(fixture)
    engine = OCREngine()
    engine.debug_mode = False
    expected = np.array(engine.preprocess_tooltip(card))
    assert np.array_equal(np.array(binarize_card(np.array(card)).image), expected)


@needs_tesseract
@pytest.mark.parametrize("fixture", CARD_FIXTURES)
def test_tesseract_reader_reads_as_ocr_engine(fixture):
    card = _card(fixture)
    engine = OCREngine()
    engine.debug_mode = False
    expected_text = engine._read_item_name(engine.preprocess_tooltip(card))  # noqa: SLF001
    read = TesseractReader().read(np.array(card))
    assert expected_text
    assert read.text == expected_text
    assert read.tier_vote == engine.last_tier_vote


def test_find_title_lines_finds_anvil_title_below_tags():
    lines = find_title_lines(np.array(_card("anvil_iv_loadout_capture.png")))
    assert len(lines) == 1
    left, top, right, bottom = lines[0]
    # One title line about 34 px tall in the top quarter of the card
    assert 25 <= bottom - top <= 45
    assert top < 200
    assert right > left


@needs_tesseract
def test_tesseract_title_reader_reads_anvil_iv():
    read = TesseractTitleReader().read(np.array(_card("anvil_iv_loadout_capture.png")))
    assert read.text == "ANVIL IV"
    assert read.tier_vote is not None
    assert read.tier_vote.tier == "IV"


@pytest.mark.parametrize(
    ("raw", "fixed"),
    [
        ("Rattler |l", "RATTLER II"),
        ("anvil 1V", "ANVIL IV"),
        ("Torrente l", "TORRENTE I"),
        ("Hullcracker", "HULLCRACKER"),
        ("Ball", "BALL"),  # a lone word keeps its trailing l
        ("", ""),
    ],
)
def test_fix_trailing_token(raw, fixed):
    assert fix_trailing_token(raw) == fixed


def test_make_reader_rejects_unknown_engine():
    with pytest.raises(ValueError, match="Unknown OCR_ENGINE"):
        make_reader("easyocr")


def test_paddle_reader_without_package_is_unavailable(monkeypatch):
    # None in sys.modules makes the import raise ImportError
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", None)
    with pytest.raises(TitleReaderUnavailableError, match="rapidocr-onnxruntime"):
        make_reader("paddle")
