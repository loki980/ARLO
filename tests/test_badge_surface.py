"""BadgeSurface messages, its no-op fallback and the helper's PNG mode."""

import inspect
import logging
import subprocess
import threading

import pytest
from PIL import Image

from arc_helper import badge_helper
from arc_helper import badge_surface
from arc_helper.badge_helper import apply_message
from arc_helper.badge_helper import checked_physical
from arc_helper.badge_helper import draw_scale
from arc_helper.badge_helper import parse_message
from arc_helper.badge_surface import BadgeSurface
from arc_helper.badge_surface import Border
from arc_helper.badge_surface import Patch
from arc_helper.badge_surface import Tag
from arc_helper.badge_surface import encode_message
from arc_helper.badge_surface import find_system_python
from arc_helper.badge_surface import render_png

GOLD = (0xFF, 0xC8, 0x3D)


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
            "color": "#ffc83d",
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
            "check": False,
            "outline": None,
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


# A 190 px slot at the origin with a 38 px bottom bar, the type icon's right
# edge at x 40, and the right mark starting at x 120
SLOT = (0, 0, 190, 190)
BAR = (0, 152, 190, 190)
BAR_COLOUR = (20, 25, 35)
WHITE = (255, 255, 255, 255)


def _render(tmp_path, badges, size=(200, 200)):
    python = find_system_python()
    if python is None:
        pytest.skip("no system Python with gi, cairo and Xlib")
    out = tmp_path / "badges.png"
    assert render_png(badges, size, out, python=python)
    return Image.open(out)


def test_patch_message_round_trips_through_the_helper_parser():
    patch = Patch(BAR, icon_right=40, patch_right=120, colour=BAR_COLOUR, text="1.3k")
    _op, parsed = parse_message(encode_message("draw", [patch]))
    assert parsed == [
        {
            "kind": "patch",
            "rect": [0, 152, 190, 190],
            "icon_right": 40,
            "patch_right": 120,
            "colour": [20, 25, 35],
            "text": "1.3k",
            "radius": 6,
            "fg": None,
        }
    ]


def test_patch_fills_to_its_text_and_stops_at_the_computed_edge(tmp_path):
    patch = Patch(BAR, icon_right=40, patch_right=120, colour=BAR_COLOUR, text="1.3k")
    image = _render(tmp_path, [patch])
    # Row 154 is above the text: fill from the left edge, then nothing
    row = [image.getpixel((x, 154)) for x in range(200)]
    edge = next(x for x, p in enumerate(row) if p != (*BAR_COLOUR, 255))
    assert all(p[3] == 0 for p in row[edge:])
    cream = [
        x
        for x in range(200)
        for y in range(152, 190)
        if image.getpixel((x, y)) == (0xF9, 0xEE, 0xDF, 255)
    ]
    assert cream, "the worth text is drawn in the game's cream count colour"
    assert min(cream) >= 8
    # '1.3k' is wider than the icon gap, so the edge is the text's end plus
    # 8 px padding (a pixel or two more for the glyph's side bearing)
    assert 44 < edge <= 120
    assert 8 <= edge - (max(cream) + 1) <= 11
    # Nothing above the bar
    assert image.getpixel((20, 150))[3] == 0


def test_patch_text_is_drawn_in_fg_when_given(tmp_path):
    patch = Patch(
        BAR,
        icon_right=40,
        patch_right=120,
        colour=(255, 255, 255),
        text="6k",
        fg="#111111",
    )
    image = _render(tmp_path, [patch])
    pixels = {image.getpixel((x, y)) for x in range(200) for y in range(152, 190)}
    assert (0x11, 0x11, 0x11, 255) in pixels
    assert (0xF9, 0xEE, 0xDF, 255) not in pixels
    assert image.getpixel((2, 154)) == (255, 255, 255, 255)


def test_short_patch_text_fills_to_the_icon_gap(tmp_path):
    patch = Patch(BAR, icon_right=40, patch_right=120, colour=BAR_COLOUR, text="5")
    image = _render(tmp_path, [patch])
    assert image.getpixel((43, 154)) == (*BAR_COLOUR, 255)
    assert image.getpixel((44, 154))[3] == 0


def test_patch_that_cannot_fit_its_text_draws_nothing(tmp_path):
    patch = Patch(BAR, icon_right=40, patch_right=30, colour=BAR_COLOUR, text="1.3k")
    image = _render(tmp_path, [patch])
    assert image.getbbox() is None


def _text_rows(image):
    """First and last row (last exclusive) of the bright text pixels on the bar."""
    rows = [
        y
        for y in range(152, 190)
        for x in range(200)
        if (p := image.getpixel((x, y)))[3] == 255 and min(p[:3]) > 128
    ]
    return min(rows), max(rows) + 1


def _patch_text_height(tmp_path, text, patch_right):
    patch = Patch(
        BAR, icon_right=40, patch_right=patch_right, colour=BAR_COLOUR, text=text
    )
    first, last = _text_rows(_render(tmp_path, [patch]))
    return first, last, last - first


def test_patch_uses_semibold_and_tags_keep_bold():
    assert badge_helper.PATCH_FONT_FAMILY == "Barlow Semi Condensed"
    assert badge_helper.PATCH_FONT_WEIGHT == "Semi-Bold"
    assert badge_helper.PATCH_FONT_FILE.is_file()
    # A tag, which never passes a weight, is drawn in Bold
    assert badge_helper.TAG_FONT_WEIGHT == "Bold"
    weight = inspect.signature(badge_helper._Text).parameters["weight"]  # noqa: SLF001
    assert weight.default == "Bold"


def test_patch_digits_are_the_game_counts_size_on_its_baseline(tmp_path):
    # 0.8 of a 38 px bar is a 30.4 px font, Barlow's digits are 0.7 of it
    _first, last, height = _patch_text_height(tmp_path, "88", 120)
    assert abs(height - 0.8 * 38 * 0.7) <= 1.5
    # The digits' bottom is 0.216 of the bar height (8.2 px) above the bar bottom
    assert abs((190 - last) - 0.216 * 38) <= 1.5


def test_long_patch_value_fits_only_after_shrinking(tmp_path):
    _, normal_last, normal = _patch_text_height(tmp_path, "88", 120)
    # '88888' is too wide for 78 px at the normal size but fits at a smaller one
    _, last, height = _patch_text_height(tmp_path, "88888", 78)
    assert 0.57 * 38 * 0.7 - 1.5 <= height < normal - 2
    assert last == normal_last or abs(last - normal_last) <= 1.5


def test_patch_value_too_wide_at_the_minimum_size_draws_nothing(tmp_path):
    patch = Patch(
        BAR, icon_right=40, patch_right=60, colour=BAR_COLOUR, text="88888888"
    )
    assert _render(tmp_path, [patch]).getbbox() is None


def test_check_tag_draws_green_check_pixels(tmp_path):
    image = _render(tmp_path, [Tag(SLOT, "3/5", corner="tl", check=True)])
    pixels = [image.getpixel((x, y)) for x in range(60) for y in range(40)]
    assert any(g > 150 and g > r + 60 and g > b + 60 for r, g, b, _a in pixels)
    plain = _render(tmp_path, [Tag(SLOT, "3/5", corner="tl")])
    pixels = [plain.getpixel((x, y)) for x in range(60) for y in range(40)]
    assert not any(g > 150 and g > r + 60 and g > b + 60 for r, g, b, _a in pixels)


def test_border_stays_on_top_of_a_patch_sent_after_it(tmp_path):
    patch = Patch(BAR, icon_right=40, patch_right=120, colour=BAR_COLOUR, text="1.3k")
    # The border comes first in the message but is drawn after the patch
    image = _render(tmp_path, [Border(SLOT), patch])
    # Left edge inside the 3 px stroke, halfway down the bar
    assert image.getpixel((1, 170)) == (*GOLD, 255)
    # Just inside the stroke, the patch fill shows
    assert image.getpixel((5, 170)) == (*BAR_COLOUR, 255)


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


class _FakeStdin:
    """A helper's stdin that records lines and can block every write."""

    def __init__(self, blocking):
        self.lines = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.killed = False
        self.received = threading.Condition()
        if not blocking:
            self.release.set()

    def write(self, line):
        self.entered.set()
        self.release.wait()
        if self.killed:
            message = "killed"
            raise BrokenPipeError(message)
        with self.received:
            self.lines.append(line)
            self.received.notify_all()

    def flush(self):
        pass

    def close(self):
        pass

    def wait_for(self, count):
        with self.received:
            return self.received.wait_for(lambda: len(self.lines) >= count, 5)


class _FakeProc:
    def __init__(self, blocking):
        self.stdin = _FakeStdin(blocking)
        self.kill_called = threading.Event()

    def poll(self):
        return -9 if self.kill_called.is_set() else None

    def kill(self):
        self.stdin.killed = True
        self.kill_called.set()
        self.stdin.release.set()

    def wait(self, timeout=None):
        del timeout
        return self.poll()


def _writers():
    return {t for t in threading.enumerate() if t.name == "badge-writer"}


def _surface_with(monkeypatch, blocking):
    """A started surface on a fake helper, and the writer thread it started."""
    proc = _FakeProc(blocking)
    monkeypatch.setattr(badge_surface, "find_system_python", lambda: "py")
    monkeypatch.setattr(badge_surface, "get_screen_resolution", lambda: (100, 100))
    monkeypatch.setattr(badge_surface.subprocess, "Popen", lambda *_a, **_k: proc)
    surface = BadgeSurface()
    before = _writers()
    assert surface.start()
    (writer,) = _writers() - before
    return surface, proc, writer


def _ended(writer):
    writer.join(5)
    return not writer.is_alive()


def _off_warnings(caplog):
    return [r for r in caplog.records if "Badge surface off" in r.getMessage()]


DRAW_A = [Border((0, 0, 10, 10))]
DRAW_B = [Border((0, 0, 20, 20))]
DRAW_C = [Border((0, 0, 30, 30))]
DRAW_D = [Border((0, 0, 40, 40))]


def test_draw_and_clear_return_while_the_helper_stdin_is_blocked(monkeypatch):
    surface, proc, _writer = _surface_with(monkeypatch, blocking=True)
    surface.draw(DRAW_A)
    assert proc.stdin.entered.wait(5), "the writer is blocked on the first draw"
    returned = threading.Event()

    def caller():
        surface.draw(DRAW_B)
        surface.clear()
        returned.set()

    threading.Thread(target=caller, daemon=True).start()
    assert returned.wait(5)
    # The fake stdin was never released, so nothing waited for it
    assert not proc.stdin.release.is_set()
    assert proc.stdin.lines == []
    proc.stdin.release.set()
    surface.stop()


def test_latest_message_wins_while_a_write_is_blocked(monkeypatch):
    surface, proc, _writer = _surface_with(monkeypatch, blocking=True)
    surface.draw(DRAW_A)
    assert proc.stdin.entered.wait(5)
    surface.draw(DRAW_B)
    surface.draw(DRAW_C)
    surface.draw(DRAW_D)
    proc.stdin.release.set()
    assert proc.stdin.wait_for(2)
    surface.stop()
    assert proc.stdin.lines == [
        encode_message("draw", DRAW_A),
        encode_message("draw", DRAW_D),
    ]


def test_a_stalled_write_disables_badges_with_one_warning(monkeypatch, caplog):
    monkeypatch.setattr(badge_surface, "HANDOFF_STALL_S", 0.05)
    surface, proc, writer = _surface_with(monkeypatch, blocking=True)
    with caplog.at_level(logging.WARNING):
        surface.draw(DRAW_A)
        assert proc.kill_called.wait(5), "the stalled helper is killed"
        assert _ended(writer)
        proc.stdin.entered.clear()
        surface.draw(DRAW_B)
        surface.clear()
        assert surface.start() is False
        surface.stop()
    assert not proc.stdin.entered.is_set(), "later draws send nothing"
    assert proc.stdin.lines == []
    assert not surface.available
    warnings = _off_warnings(caplog)
    assert len(warnings) == 1
    assert "stalled" in warnings[0].getMessage()


def test_stop_does_not_hang_on_a_stalled_helper(monkeypatch, caplog):
    monkeypatch.setattr(badge_surface, "HANDOFF_STALL_S", 60)
    monkeypatch.setattr(badge_surface, "STOP_TIMEOUT_S", 0.05)
    surface, proc, writer = _surface_with(monkeypatch, blocking=True)
    surface.draw(DRAW_A)
    assert proc.stdin.entered.wait(5)
    stopper = threading.Thread(target=surface.stop, daemon=True)
    with caplog.at_level(logging.WARNING):
        stopper.start()
        stopper.join(5)
    assert not stopper.is_alive(), "stop returned"
    assert proc.kill_called.is_set()
    assert _ended(writer)
    assert _off_warnings(caplog) == []


def test_a_broken_pipe_still_disables_badges_once(monkeypatch, caplog):
    surface, proc, writer = _surface_with(monkeypatch, blocking=False)
    # The helper went away: every write fails as on a closed pipe
    proc.stdin.killed = True
    with caplog.at_level(logging.WARNING):
        surface.draw(DRAW_A)
        assert _ended(writer)
        surface.draw(DRAW_B)
        surface.clear()
        surface.stop()
    assert proc.stdin.lines == []
    warnings = _off_warnings(caplog)
    assert len(warnings) == 1
    assert "stopped" in warnings[0].getMessage()


def test_a_normal_helper_receives_draw_then_clear_in_order(monkeypatch, caplog):
    surface, proc, _writer = _surface_with(monkeypatch, blocking=False)
    with caplog.at_level(logging.WARNING):
        surface.draw(DRAW_A)
        assert proc.stdin.wait_for(1)
        surface.clear()
        assert proc.stdin.wait_for(2)
        surface.stop()
    assert proc.stdin.lines == [
        encode_message("draw", DRAW_A),
        encode_message("clear"),
    ]
    assert _off_warnings(caplog) == []


def test_translate_moves_the_badges_inside_its_region_and_drops_those_left():
    region = (0, 100, 200, 400)
    drawn = apply_message(
        [],
        encode_message(
            "draw",
            [
                Border((0, 120, 50, 170)),
                Tag((0, 300, 50, 350), "9"),
                Border((300, 120, 350, 170)),
            ],
        ),
    )
    moved = apply_message(drawn, encode_message("translate", region=region, dy=-40))
    assert [b["rect"] for b in moved] == [
        [0, 80, 50, 130],
        [0, 260, 50, 310],
        [300, 120, 350, 170],
    ]
    # Moved badges are clipped to the region from then on, the other is not
    assert [b.get("clip") for b in moved] == [list(region), list(region), None]
    gone = apply_message(moved, encode_message("translate", region=region, dy=-40))
    assert [b["rect"] for b in gone] == [[0, 220, 50, 270], [300, 120, 350, 170]]


def test_a_clipped_badge_keeps_following_until_it_leaves_the_region():
    # After the first move its centre (y 105) is out of the region, but it
    # was moved there, so it moves on and is dropped once it no longer meets it
    region = (0, 100, 200, 400)
    move = encode_message("translate", region=region, dy=-40)
    drawn = apply_message([], encode_message("draw", [Border((0, 145, 50, 195))]))
    once = apply_message(drawn, move)
    assert [b["rect"] for b in once] == [[0, 105, 50, 155]]
    twice = apply_message(once, move)
    assert [b["rect"] for b in twice] == [[0, 65, 50, 115]]
    assert apply_message(twice, move) == []


def test_png_mode_draws_a_translated_border_clipped_to_its_region(tmp_path):
    python = find_system_python()
    if python is None:
        pytest.skip("no system Python with gi, cairo and Xlib")
    out = tmp_path / "badges.png"
    lines = encode_message("draw", [Border((40, 60, 140, 160))]) + encode_message(
        "translate", region=(0, 50, 320, 200), dy=-30
    )
    result = subprocess.run(
        [python, str(badge_surface.HELPER), "--png", str(out), "--size", "320", "200"],
        input=lines,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    image = Image.open(out)
    # The border moved up from 60 to 160 to 30 to 130, nothing left below
    assert image.getpixel((41, 100)) == (*GOLD, 255)
    assert image.getpixel((90, 128)) == (*GOLD, 255)
    assert image.getpixel((41, 145))[3] == 0
    # Its top edge and the left edge above y 50 are clipped to the region
    assert image.getpixel((90, 31))[3] == 0
    assert image.getpixel((41, 40))[3] == 0


def test_a_translate_follows_a_waiting_draw_and_a_draw_replaces_it(monkeypatch):
    surface, proc, _writer = _surface_with(monkeypatch, blocking=True)
    surface.draw(DRAW_A)
    assert proc.stdin.entered.wait(5)
    surface.draw(DRAW_B)
    surface.translate((0, 0, 50, 50), -5)
    surface.translate((0, 0, 50, 50), -5)
    proc.stdin.release.set()
    assert proc.stdin.wait_for(2)
    proc.stdin.release.clear()
    proc.stdin.entered.clear()
    surface.translate((0, 0, 50, 50), -5)
    assert proc.stdin.entered.wait(5)
    surface.translate((0, 0, 50, 50), -7)
    surface.draw(DRAW_C)
    proc.stdin.release.set()
    assert proc.stdin.wait_for(4)
    surface.stop()
    move = encode_message("translate", region=(0, 0, 50, 50), dy=-5)
    assert proc.stdin.lines == [
        encode_message("draw", DRAW_A),
        encode_message("draw", DRAW_B) + move + move,
        move,
        encode_message("draw", DRAW_C),
    ]
