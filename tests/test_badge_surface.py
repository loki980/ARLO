"""BadgeSurface messages, its no-op fallback and the helper's PNG mode."""

import logging

import pytest
from PIL import Image

from arc_helper import badge_surface
from arc_helper.badge_helper import checked_physical
from arc_helper.badge_helper import draw_scale
from arc_helper.badge_helper import parse_message
from arc_helper.badge_surface import BadgeSurface
from arc_helper.badge_surface import Border
from arc_helper.badge_surface import Tag
from arc_helper.badge_surface import encode_message
from arc_helper.badge_surface import find_system_python
from arc_helper.badge_surface import render_png

GOLD = (0xF2, 0xB1, 0x34)


def test_draw_message_round_trips_through_the_helper_parser():
    badges = [
        Border((10, 20, 110, 120)),
        Tag((10, 20, 110, 120), "Φ 4k", corner="tl", fg="#7ee07e"),
    ]
    op, parsed = parse_message(encode_message("draw", badges))
    assert op == "draw"
    assert parsed == [
        {
            "kind": "border",
            "rect": [10, 20, 110, 120],
            "color": "#f2b134",
            "width": 3,
            "radius": 4,
        },
        {
            "kind": "tag",
            "rect": [10, 20, 110, 120],
            "text": "Φ 4k",
            "corner": "tl",
            "fg": "#7ee07e",
            "bg": "#141414d9",
        },
    ]
    assert parse_message(encode_message("clear")) == ("clear", [])


def test_surface_is_a_no_op_when_the_system_python_lacks_gi(monkeypatch, caplog):
    # Every candidate Python fails the gi, cairo and Xlib import check
    monkeypatch.setattr(badge_surface, "_has_imports", lambda _python: False)
    popen_calls = []
    monkeypatch.setattr(
        badge_surface.subprocess, "Popen", lambda *a, **_k: popen_calls.append(a)
    )
    surface = BadgeSurface()
    with caplog.at_level(logging.WARNING):
        assert surface.start() is False
        assert surface.start() is False
        surface.draw([Border((0, 0, 10, 10))])
        surface.clear()
        surface.stop()
    assert popen_calls == []
    assert not surface.available
    warnings = [r for r in caplog.records if "Badge surface off" in r.getMessage()]
    assert len(warnings) == 1


def test_png_mode_draws_a_gold_border_with_a_clear_centre(tmp_path):
    python = find_system_python()
    if python is None:
        pytest.skip("no system Python with gi, cairo and Xlib")
    out = tmp_path / "badges.png"
    rect = (40, 30, 140, 130)
    assert render_png([Border(rect)], (320, 200), out, python=python)
    image = Image.open(out)
    assert image.size == (320, 200)
    assert image.mode == "RGBA"
    # Left edge, halfway down: inside the 3 px stroke
    assert image.getpixel((41, 80)) == (*GOLD, 255)
    assert image.getpixel((90, 80))[3] == 0
    assert image.getpixel((5, 5))[3] == 0


def test_scale_maps_physical_px_onto_the_window():
    sx, sy = draw_scale((2194, 1234), (3840, 2160), 1)
    assert 1750 * sx == pytest.approx(1000, abs=1)
    assert 1000 * sy == pytest.approx(571, abs=1)


def test_scale_is_one_to_one_when_sizes_match():
    assert draw_scale((1920, 1080), (1920, 1080), 1) == (1, 1)


@pytest.mark.parametrize("physical", [None, (0, 1080), (1920, -1)])
def test_scale_falls_back_to_the_gtk_factor(physical):
    assert draw_scale((1920, 1080), physical, 2) == (0.5, 0.5)


def _start_args(monkeypatch, resolution):
    monkeypatch.setattr(badge_surface, "find_system_python", lambda: "py")
    if isinstance(resolution, Exception):

        def fail():
            raise resolution

        monkeypatch.setattr(badge_surface, "get_screen_resolution", fail)
    else:
        monkeypatch.setattr(badge_surface, "get_screen_resolution", lambda: resolution)
    calls = []
    monkeypatch.setattr(
        badge_surface.subprocess, "Popen", lambda cmd, **_k: calls.append(cmd)
    )
    BadgeSurface().start()
    return calls[0]


def test_start_sends_the_physical_size(monkeypatch):
    cmd = _start_args(monkeypatch, (3840, 2160))
    assert cmd[-3:] == ["--physical", "3840", "2160"]


@pytest.mark.parametrize("resolution", [RuntimeError("no display"), (0, 2160)])
def test_start_omits_the_size_when_it_is_unknown(monkeypatch, resolution):
    assert "--physical" not in _start_args(monkeypatch, resolution)


@pytest.mark.parametrize("physical", [None, (0, 1080)])
def test_helper_warns_once_without_a_usable_size(capsys, physical):
    assert checked_physical(physical) is None
    assert capsys.readouterr().err.count("no physical screen size") == 1


def test_helper_is_quiet_with_a_usable_size(capsys):
    assert checked_physical((3840, 2160)) == (3840, 2160)
    assert not capsys.readouterr().err
