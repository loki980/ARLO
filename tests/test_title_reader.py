"""Tests for the card title readers."""

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from arc_helper.ocr import OCREngine
from arc_helper.ocr import crop_card
from arc_helper.title_reader import CachedTitle
from arc_helper.title_reader import TesseractReader
from arc_helper.title_reader import TesseractTitleReader
from arc_helper.title_reader import TitleCache
from arc_helper.title_reader import TitleReaderUnavailableError
from arc_helper.title_reader import binarize_card
from arc_helper.title_reader import find_title_lines
from arc_helper.title_reader import fix_trailing_token
from arc_helper.title_reader import make_reader
from arc_helper.title_reader import title_look
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


@needs_tesseract
@pytest.mark.parametrize(
    "fixture", ["garnet_stencil_capture_a.png", "garnet_stencil_capture_b.png"]
)
def test_tesseract_title_reader_stops_at_light_grey_description(fixture):
    # The Garnet description is light grey, lighter than title ink, and the
    # black REQUIRED RESOURCES heading further down must not join the title
    card = np.array(_card(fixture))
    assert len(find_title_lines(card)) == 1
    assert TesseractTitleReader().read(card).text == "GARNET"


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


def _bakeoff_module():
    path = Path(__file__).parents[1] / "scripts" / "bakeoff_title_readers.py"
    spec = importlib.util.spec_from_file_location("bakeoff_title_readers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bakeoff_exact_title_ignores_case_only():
    is_exact_title = _bakeoff_module().is_exact_title
    assert is_exact_title("ANVIL IV", "Anvil IV")
    assert not is_exact_title("ANVILIV", "Anvil IV")
    assert not is_exact_title("ANVIL IV.", "Anvil IV")
    assert not is_exact_title(None, "Anvil IV")


def test_bakeoff_counts_covered_rows_apart_from_scored():
    labels = [
        {"file": "a.png", "expected": "Anvil I", "card": "yes"},
        {"file": "b.png", "expected": "covered", "card": "yes"},
        {"file": "c.png", "expected": "unreadable", "card": "yes"},
        {"file": "d.png", "expected": "", "card": "no"},
    ]
    scored, unreadable, covered = _bakeoff_module().split_rows(labels)
    assert [r["file"] for r in scored] == ["a.png"]
    assert [r["file"] for r in unreadable] == ["c.png"]
    assert [r["file"] for r in covered] == ["b.png"]


# Card crops from the labelled capture library (capture-labels), the closest
# titles in it: one stroke more or less in the tier numeral
NEAR_TITLES = {
    "TORRENTE I": "title_torrente_i_card.png",
    "TORRENTE II": "title_torrente_ii_card.png",
    "STILETTO I": "title_stiletto_i_card.png",
    "STILETTO IV": "title_stiletto_iv_card.png",
}


def _crop(fixture: str) -> np.ndarray:
    with Image.open(FIXTURES / fixture) as image:
        return np.array(image.convert("RGB"))


def test_title_look_is_stable_for_the_same_card_crop():
    card = np.array(_card("anvil_iv_loadout_capture.png"))
    first = title_look(card)
    assert first is not None
    assert title_look(card.copy()).key == first.key
    # Only the title rows count: pixels below them leave the look alone
    changed = card.copy()
    changed[-40:] = 0
    assert title_look(changed).key == first.key


def test_title_look_differs_for_different_titles():
    looks = {title: title_look(_crop(name)) for title, name in NEAR_TITLES.items()}
    assert all(look is not None for look in looks.values())
    assert len({look.key for look in looks.values()}) == len(NEAR_TITLES)
    assert looks["TORRENTE I"].key != looks["TORRENTE II"].key
    assert looks["STILETTO I"].key != looks["STILETTO IV"].key


def test_title_look_is_none_without_title_lines():
    assert title_look(np.full((300, 400, 3), 245, np.uint8)) is None


def test_title_cache_drops_the_least_recently_used():
    cache = TitleCache(size=2)
    cache.put(b"a", CachedTitle("A", None))
    cache.put(b"b", CachedTitle("B", None))
    assert cache.get(b"a").text == "A"
    cache.put(b"c", CachedTitle("C", None))
    assert cache.get(b"b") is None
    assert cache.get(b"a").text == "A"
    assert len(cache) == 2
