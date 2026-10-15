"""Tests for the gold cell border wiring: rects, hide hooks and messages."""

import arc_helper.main as main_module
from arc_helper.cell_border import CellBorder
from arc_helper.cell_border import frame_bars
from arc_helper.main import Application
from arc_helper.main import OverlayRects
from arc_helper.main import screen_to_capture
from arc_helper.overlay import TooltipChips
from arc_helper.overlay import tk_box_to_phys
from arc_helper.overlay import window_rect_phys
from arc_helper.tooltip_layout import Point
from arc_helper.tooltip_layout import Rect

CELL = Rect(712, 744, 920, 953)


class _StubBorder:
    def __init__(self):
        self.calls = []

    def show(self, rect, width_px):
        self.calls.append(("show", rect, width_px))

    def hide(self, reason="hide"):
        self.calls.append(("hide", reason))


class _StubChips:
    dpi_scale = 1.0


def _app(monkeypatch):
    monkeypatch.setattr(
        main_module, "get_backend", lambda: type("B", (), {"tk_scale": 1.75})
    )
    app = Application.__new__(Application)
    app.border = _StubBorder()
    app.chips = _StubChips()
    return app


def test_border_shows_beside_placed_chips(monkeypatch):
    app = _app(monkeypatch)
    app._apply_border(CELL, placed=True)  # noqa: SLF001
    # 3 Tk px at a 1.75 Tk scale
    assert app.border.calls == [("show", CELL, 5)]


def test_border_hides_when_the_chips_message_placed_no_chip(monkeypatch):
    app = _app(monkeypatch)
    app._apply_border(CELL, placed=False)  # noqa: SLF001
    assert app.border.calls == [("hide", "no_chips")]


def test_border_hides_without_a_cell(monkeypatch):
    app = _app(monkeypatch)
    app._apply_border(None, placed=True)  # noqa: SLF001
    assert app.border.calls == [("hide", "no_cell")]


class _StubRoot:
    def after_cancel(self, job):
        pass


class _StubWindow:
    def withdraw(self):
        pass


def test_chips_hide_calls_the_border_hook_with_the_reason():
    chips = TooltipChips.__new__(TooltipChips)
    chips.root = _StubRoot()
    chips.windows = {"price": _StubWindow()}
    chips._hide_after_id = "job"  # noqa: SLF001
    chips._current_item = "Battery"  # noqa: SLF001
    chips._shown = {"price"}  # noqa: SLF001
    chips._positions = {}  # noqa: SLF001
    reasons = []
    chips.on_hide = reasons.append

    chips.hide(reason="timer")
    chips.hide(reason="chips_hide")

    assert reasons == ["timer", "chips_hide"]


def test_border_bar_rects_follow_the_shown_frame():
    border = CellBorder.__new__(CellBorder)
    border._shown = None  # noqa: SLF001
    assert border.bar_rects() == ()
    border._shown = (CELL, 5, "#f2b134")  # noqa: SLF001
    assert border.bar_rects() == tuple(frame_bars(CELL, 5))


def test_screen_rects_shift_into_capture_coordinates():
    rects = [Rect(150, 60, 200, 100), Rect(0, 0, 10, 10)]
    assert screen_to_capture(rects, Point(100, 50)) == [
        Rect(50, 10, 100, 50),
        Rect(-100, -50, -90, -40),
    ]


def test_tk_window_box_converts_to_physical_pixels():
    assert tk_box_to_phys(10, 20, 100, 40, 1.75) == Rect(18, 35, 192, 105)


class _TkWindow:
    def __init__(self, viewable):
        self.viewable = viewable

    def winfo_viewable(self):
        return self.viewable

    def winfo_rootx(self):  # noqa: PLR6301
        return 10

    def winfo_rooty(self):  # noqa: PLR6301
        return 20

    def winfo_width(self):  # noqa: PLR6301
        return 100

    def winfo_height(self):  # noqa: PLR6301
        return 40


def test_only_mapped_windows_have_a_rect():
    assert window_rect_phys(_TkWindow(viewable=True), 1.0) == Rect(10, 20, 110, 60)
    assert window_rect_phys(_TkWindow(viewable=False), 1.0) is None


def test_overlay_rects_hand_over_a_snapshot():
    shared = OverlayRects()
    assert shared.snapshot() == ()
    shared.publish([CELL])
    snapshot = shared.snapshot()
    shared.publish([])
    assert snapshot == (CELL,)
