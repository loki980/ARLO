import tkinter as tk
from pathlib import Path

import pytest
from PIL import ImageFont

from arc_helper import overlay as overlay_module
from arc_helper import textrender
from arc_helper.textrender import TextStyle
from arc_helper.textrender import families_scalable
from arc_helper.textrender import find_font_file
from arc_helper.textrender import has_scalable_fonts
from arc_helper.textrender import render_text
from arc_helper.textrender import wrap_text

STYLE = TextStyle(size=13, bold=False, fg="#ffffff", bg="#1a1a2e")


@pytest.fixture
def font_path() -> str:
    path = find_font_file(bold=False)
    if path is None:
        pytest.skip("no TrueType font on this machine")
    return path


def test_find_font_file_returns_an_existing_ttf(font_path):
    assert font_path.lower().endswith(".ttf")
    assert Path(font_path).is_file()


def test_families_scalable_needs_more_than_fixed():
    assert not families_scalable(("fixed",))
    assert families_scalable(("fixed", "Noto Sans"))


def test_bigger_points_make_a_taller_image(font_path):
    small = render_text("KEEP", STYLE, font_path, 1.334)
    big = render_text(
        "KEEP", TextStyle(size=44, bold=True, fg="#fff", bg="#000"), font_path, 1.334
    )
    assert big.height > small.height
    assert big.width > small.width


def test_image_is_as_wide_as_the_text_needs(font_path):
    pixels = round(13 * 1.334)
    needed = ImageFont.truetype(font_path, pixels).getlength("Fireball Burner")
    image = render_text("Fireball Burner", STYLE, font_path, 1.334)
    assert image.width >= needed


def test_wrapping_splits_a_long_line_under_a_narrow_wraplength(font_path):
    font = ImageFont.truetype(font_path, 17)
    text = "8 for Refiner II, 1 for Test Case, then recycle"
    assert len(wrap_text(text, font, 0)) == 1
    assert len(wrap_text(text, font, 120)) > 1
    wide = render_text(text, STYLE, font_path, 1.334)
    narrow = render_text(text, STYLE, font_path, 1.334, wraplength=120)
    assert narrow.height > wide.height
    assert narrow.width < wide.width


def test_newlines_make_lines(font_path):
    font = ImageFont.truetype(font_path, 17)
    assert wrap_text("a\nb", font, 0) == ["a", "b"]


def _root():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    return root


def test_label_shows_an_image_on_the_pillow_path(monkeypatch, font_path):  # noqa: ARG001
    root = _root()
    try:
        monkeypatch.setattr(overlay_module, "has_scalable_fonts", lambda _r: False)
        label = overlay_module._make_label(  # noqa: SLF001
            root, root, 13, fg="white", bg="#1a1a2e"
        )
        overlay_module._set_text(label, "Sell: 640", fg="#ffd700")  # noqa: SLF001
        assert label.cget("image")
        assert label._arlo_photo.width() > 0  # noqa: SLF001
    finally:
        root.destroy()


def test_label_keeps_tk_text_when_fonts_scale(monkeypatch):
    root = _root()
    try:
        monkeypatch.setattr(overlay_module, "has_scalable_fonts", lambda _r: True)
        label = overlay_module._make_label(  # noqa: SLF001
            root, root, 13, fg="white", bg="#1a1a2e"
        )
        overlay_module._set_text(label, "Sell: 640")  # noqa: SLF001
        assert label.cget("text") == "Sell: 640"
        assert not label.cget("image")
    finally:
        root.destroy()


class _FakeFont:
    def __init__(self, actual_family):
        self._family = actual_family

    def actual(self, _option):
        return self._family


def _stub_tk(monkeypatch, families, actual_family):
    calls = []

    def fake_families(_root):
        calls.append(1)
        return families

    monkeypatch.setattr(textrender.tkfont, "families", fake_families)
    monkeypatch.setattr(
        textrender.tkfont, "Font", lambda *_a, **_k: _FakeFont(actual_family)
    )
    return calls


def test_has_scalable_fonts_false_for_fixed_only(monkeypatch):
    _stub_tk(monkeypatch, ("fixed",), "fixed")
    assert has_scalable_fonts(type("Root", (), {})()) is False


def test_has_scalable_fonts_true_for_a_real_family(monkeypatch):
    _stub_tk(monkeypatch, ("fixed", "Noto Sans"), "Noto Sans")
    assert has_scalable_fonts(type("Root", (), {})()) is True


def test_has_scalable_fonts_false_when_request_resolves_to_fixed(monkeypatch):
    _stub_tk(monkeypatch, ("fixed", "Noto Sans"), "fixed")
    assert has_scalable_fonts(type("Root", (), {})()) is False


def test_has_scalable_fonts_is_cached_per_root(monkeypatch):
    calls = _stub_tk(monkeypatch, ("fixed", "Noto Sans"), "Noto Sans")
    root = type("Root", (), {})()
    has_scalable_fonts(root)
    has_scalable_fonts(root)
    assert len(calls) == 1


def test_condensed_font_is_the_bundled_barlow():
    assert find_font_file(bold=False, condensed=True).endswith(
        "BarlowSemiCondensed-SemiBold.ttf"
    )
    assert find_font_file(bold=True, condensed=True).endswith(
        "BarlowSemiCondensed-Bold.ttf"
    )


def test_credit_sign_missing_from_barlow_still_takes_width(font_path):
    barlow = find_font_file(bold=False, condensed=True)
    plain = render_text("640 ₡", STYLE, barlow, 1.334)
    with_fallback = render_text("640 ₡", STYLE, barlow, 1.334, 0, font_path)
    assert with_fallback.width != plain.width
