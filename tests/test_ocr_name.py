"""Tests for item-name line detection used by the two-pass OCR reader."""

import numpy as np
from PIL import Image

from arc_helper import ocr as ocr_module
from arc_helper.ocr import OCREngine


def test_upper_ratio():
    assert OCREngine._upper_ratio("RASCAL") == 1.0  # noqa: SLF001
    assert OCREngine._upper_ratio("Fires explosive") < 0.3  # noqa: SLF001
    assert OCREngine._upper_ratio("") == 0.0  # noqa: SLF001


def test_single_line_name():
    text = "RASCALII\nFires explosive projectiles that only"
    assert OCREngine._count_name_lines(text) == 1  # noqa: SLF001


def test_tolerates_miscased_numeral():
    # "II" misread as "Il" is still mostly uppercase -> counts as the name line
    text = "BETTINA Il\nHas slow fire rate and high damage"
    assert OCREngine._count_name_lines(text) == 1  # noqa: SLF001


def test_multiword_multiline_name():
    text = "ADVANCED ELECTRICAL\nCOMPONENTS\nA crafting material"
    assert OCREngine._count_name_lines(text) == 2  # noqa: SLF001


def test_zero_when_description_leads():
    text = "the quick brown fox\nmore lowercase text"
    assert OCREngine._count_name_lines(text) == 0  # noqa: SLF001


def test_skips_short_leading_noise():
    # A stray 1-char artifact above the name is ignored
    text = "RASCALII\nFires explosive projectiles"
    assert OCREngine._count_name_lines("x\n" + text) == 1  # noqa: SLF001


def test_last_name_word_finds_tier_token():
    lines = [["x"], ["ANVIL", "Il"], ["Fires", "explosive"]]
    assert OCREngine._last_name_word(lines, 1) == 1  # noqa: SLF001
    assert OCREngine._last_name_word([["ARPEGGIO", "|V"]], 1) == 0  # noqa: SLF001
    assert OCREngine._last_name_word([["BATTERY"]], 1) is None  # noqa: SLF001
    assert OCREngine._last_name_word([["ANVILIV"]], 1) is None  # noqa: SLF001


def _fake_tesseract(monkeypatch, plain, whitelisted, words):
    """Stub the Tesseract calls. words are (text, line, left, top, w, h)."""
    calls = []

    def image_to_string(_image, config=""):
        calls.append("string")
        return whitelisted if "whitelist" in config else plain

    def image_to_data(_image, config="", output_type=None):  # noqa: ARG001
        calls.append("data")
        keys = ("text", "line_num", "left", "top", "width", "height")
        data = {k: [w[i] for w in words] for i, k in enumerate(keys)}
        data["block_num"] = [1] * len(words)
        data["par_num"] = [1] * len(words)
        return data

    monkeypatch.setattr(ocr_module.pytesseract, "image_to_string", image_to_string)
    monkeypatch.setattr(ocr_module.pytesseract, "image_to_data", image_to_data)
    return calls


def test_tier_vote_reads_strokes_and_leaves_name_unchanged(monkeypatch):
    # Two black bars inside the box Tesseract reports for "Il" (box 100,10 40x40)
    pixels = np.full((80, 200), 255, dtype=np.uint8)
    pixels[10:50, 105:113] = 0
    pixels[10:50, 125:133] = 0
    calls = _fake_tesseract(
        monkeypatch,
        plain="ANVIL Il\nFires explosive rounds\n",
        whitelisted="ANVILII\nFIRESEXPLOSIVEROUNDS\n",
        words=[
            ("ANVIL", 1, 10, 10, 80, 40),
            ("Il", 1, 100, 10, 40, 40),
            ("Fires", 2, 10, 60, 40, 15),
        ],
    )
    engine = OCREngine()
    name = engine._read_item_name(Image.fromarray(pixels, mode="L"))  # noqa: SLF001
    assert name == "ANVILII"
    assert engine.last_tier_vote.tier == "II"
    assert calls == ["string", "string", "data"]


def test_no_word_boxes_for_untiered_name(monkeypatch):
    calls = _fake_tesseract(
        monkeypatch,
        plain="BATTERY\nPowers things\n",
        whitelisted="BATTERY\nPOWERSTHINGS\n",
        words=[],
    )
    engine = OCREngine()
    image = Image.new("L", (100, 50), 255)
    assert engine._read_item_name(image) == "BATTERY"  # noqa: SLF001
    assert engine.last_tier_vote is None
    assert calls == ["string", "string"]


def test_capture_library_throttles_per_name_and_never_raises(tmp_path):
    engine = OCREngine()
    engine.capture_library_dir = tmp_path / "captures"
    image = Image.new("RGB", (20, 20))
    engine._save_to_library(image, "ANVIL II")  # noqa: SLF001
    engine._save_to_library(image, "ANVIL II")  # noqa: SLF001 - within 2 seconds
    engine._save_to_library(image, None)  # noqa: SLF001
    saved = sorted(
        p.name.split("_", 1)[1] for p in engine.capture_library_dir.iterdir()
    )
    assert saved == ["ANVIL-II.png", "none.png"]

    engine.capture_library_dir = tmp_path / "blocked"
    engine.capture_library_dir.write_text("a file, not a folder")
    engine._save_to_library(image, "OSPREY I")  # noqa: SLF001 - logged, not raised
