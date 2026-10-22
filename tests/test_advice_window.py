"""Tests for the advice window placement, scale and show caching."""

from types import SimpleNamespace

import pytest
from PIL import Image

from arc_helper import advice_window
from arc_helper.advice_card import AdviceCard
from arc_helper.advice_window import AdviceWindow
from arc_helper.advice_window import fit_scale
from arc_helper.advice_window import flatten_on_panel
from arc_helper.advice_window import place_card
from arc_helper.advice_window import render_in_box
from arc_helper.tooltip_layout import Rect

BOX_4K = Rect(1367, 1635, 2045, 1927)
BOX_QHD = Rect(911, 1090, 1364, 1284)


def test_fit_scale_4k_box():
    assert fit_scale(678) == pytest.approx(1.21, abs=0.005)


def test_fit_scale_qhd_box():
    assert fit_scale(453) == pytest.approx(0.809, abs=0.005)


def test_place_card_4k_inside_floor():
    assert place_card(BOX_4K, (678, 300), 2160) == (1367, 1635)


def test_place_card_qhd_inside_floor():
    assert place_card(BOX_QHD, (453, 150), 1440) == (911, 1090)


def test_place_card_never_raises_a_tall_card():
    # The card is scaled down to fit instead, see render_in_box
    assert place_card(BOX_4K, (678, 500), 2160) == (1367, 1635)


def test_render_in_box_scales_a_tall_card_down_to_the_box(monkeypatch):
    # A card 400 px tall at scale 1.0, home beneath the second gun on the
    # base stash screen: 610 wide, 1631 down to the floor at 2009
    renders = []

    def fake_render(_card, scale):
        renders.append(scale)
        return Image.new("RGBA", (round(560 * scale), round(400 * scale)))

    monkeypatch.setattr(advice_window, "render_advice_card", fake_render)
    box = Rect(1384, 1631, 1994, 2009)

    image = render_in_box(AdviceCard("keep", "Keep 7", None, 7, None, ()), box)

    assert image.size[1] <= 378
    assert image.size[0] <= 610
    assert renders[0] == pytest.approx(610 / 560)
    assert len(renders) >= 2


def test_render_in_box_keeps_the_width_scale_when_the_card_fits(monkeypatch):
    monkeypatch.setattr(
        advice_window,
        "render_advice_card",
        lambda _card, scale: Image.new("RGBA", (round(560 * scale), 200)),
    )
    box = Rect(1384, 1631, 1994, 2009)
    image = render_in_box(AdviceCard("keep", "Keep 7", None, 7, None, ()), box)
    assert image.size == (610, 200)


def test_flatten_puts_panel_colour_in_transparent_corner():
    image = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    image.putpixel((2, 2), (255, 0, 0, 255))
    flat = flatten_on_panel(image)
    assert flat.getpixel((0, 0)) == advice_window.PANEL_RGB
    assert flat.getpixel((2, 2)) == (255, 0, 0)


class _FakeWidget:
    def __init__(self, *_args, **_kwargs):
        self.calls = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return call


@pytest.fixture
def window(monkeypatch):
    backend = SimpleNamespace(tk_scale=1.0, resolution=lambda: (3840, 2160))
    renders = []

    def fake_render(card, scale):
        renders.append((card, scale))
        return Image.new("RGBA", (678, 200), (0, 0, 0, 0))

    monkeypatch.setattr(advice_window.tk, "Toplevel", _FakeWidget)
    monkeypatch.setattr(advice_window.tk, "Label", _FakeWidget)
    monkeypatch.setattr(advice_window.ImageTk, "PhotoImage", lambda image: image)
    monkeypatch.setattr(advice_window, "make_click_through", lambda _w: None)
    monkeypatch.setattr(advice_window, "get_backend", lambda: backend)
    monkeypatch.setattr(advice_window, "phys_to_tk", round)
    monkeypatch.setattr(advice_window, "render_advice_card", fake_render)
    win = AdviceWindow(None)
    win.renders = renders
    return win


def _card(title="Keep 7"):
    return AdviceCard("keep", title, None, 7, None, ())


def _count(win, name):
    return [c for c in win.window.calls if c[0] == name]


def test_show_equal_card_same_box_does_not_rerender_or_remap(window):
    window.show(_card(), BOX_4K)
    window.show(_card(), BOX_4K)
    assert len(window.renders) == 1
    assert len(_count(window, "deiconify")) == 1
    assert window.renders[0][1] == pytest.approx(678 / 560)


def test_show_different_card_swaps_in_place(window):
    window.show(_card(), BOX_4K)
    window.show(_card("Keep 8"), BOX_4K)
    assert len(window.renders) == 2
    assert len(_count(window, "deiconify")) == 1


def test_card_shown_before_is_not_rendered_again(window):
    window.show(_card(), BOX_4K)
    window.show(_card("Keep 8"), BOX_4K)
    window.hide()
    window.show(_card(), BOX_4K)
    assert len(window.renders) == 2
    assert window._card == _card()  # noqa: SLF001
    assert window.rect_phys() == Rect(1367, 1635, 2045, 1835)


def test_render_cache_keeps_the_newest_cards(window, monkeypatch):
    monkeypatch.setattr(advice_window, "RENDER_CACHE_SIZE", 2)
    for title in ("A", "B", "C", "A"):
        window.show(_card(title), BOX_4K)
    # A was dropped when C came in, so it renders again
    assert [card.title for card, _ in window.renders] == ["A", "B", "C", "A"]
    window.show(_card("C"), BOX_4K)
    assert len(window.renders) == 4


def test_hide_then_show_remaps(window):
    window.show(_card(), BOX_4K)
    window.hide()
    assert window.rect_phys() is None
    window.show(_card(), BOX_4K)
    assert len(_count(window, "deiconify")) == 2
    assert len(window.renders) == 1


def test_rect_phys_matches_placement(window):
    window.show(_card(), BOX_4K)
    assert window.rect_phys() == Rect(1367, 1635, 2045, 1835)
