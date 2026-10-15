"""Tests for the gold cell border wiring: rects, hide hooks and messages."""

import queue
import threading

import arc_helper.main as main_module
from arc_helper.cell_border import CellBorder
from arc_helper.cell_border import frame_bars
from arc_helper.database import Item
from arc_helper.main import Application
from arc_helper.main import OverlayRects
from arc_helper.main import Scanner
from arc_helper.main import ScannerState
from arc_helper.main import screen_to_capture
from arc_helper.overlay import TooltipChips
from arc_helper.overlay import tk_box_to_phys
from arc_helper.overlay import window_rect_phys
from arc_helper.tooltip_layout import Point
from arc_helper.tooltip_layout import Rect
from arc_helper.verdict import Verdict

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
    assert app.border.calls == [("show", CELL, 10)]


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
    border._shown = (CELL, 10, "#ffc83d")  # noqa: SLF001
    assert border.bar_rects() == tuple(frame_bars(CELL, 10))


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


def test_a_patch_marks_its_whole_rect_for_the_stillness_compare():
    patch = {"rect": [10, 80, 100, 100], "kind": "patch"}
    assert main_module.badge_marks(patch) == [Rect(10, 80, 100, 100)]


def test_surface_badges_make_patches_and_outlined_tags():
    made = main_module.surface_badges(
        [
            {
                "rect": [1, 2, 3, 4],
                "icon_right": 2,
                "patch_right": 3,
                "colour": [1, 2, 3],
                "text": "9",
                "radius": 6,
                "kind": "patch",
            },
            {
                "rect": [1, 2, 3, 4],
                "text": "6/5",
                "corner": "tl",
                "check": True,
                "outline": [1, 2, 3],
                "kind": "tag",
            },
        ]
    )
    assert made[0].colour == (1, 2, 3)
    assert made[1].outline == (1, 2, 3)
    assert made[1].check


def test_hovered_gold_frame_shows_for_a_sell_verdict(monkeypatch):
    monkeypatch.setattr(main_module, "get_ocr_engine", _StubEngine)
    monkeypatch.setattr(main_module, "chip_payload", lambda *_a, **_k: {})
    monkeypatch.setattr(main_module, "hovered_cell_on_screen", lambda *_a: CELL)
    scanner = Scanner.__new__(Scanner)
    scanner._running = True  # noqa: SLF001
    scanner.state = ScannerState.ACTIVE
    scanner._chips_lock = threading.Lock()  # noqa: SLF001
    scanner._hold_ms = 1000  # noqa: SLF001
    scanner.ui_queue = queue.Queue()
    scanner.db = _StubDb()
    scanner.overlay_rects = OverlayRects()
    sell = Verdict("sell", "Sell it")
    scanner._queue_chips(  # noqa: SLF001
        Item(name="Rubber Parts", action="Sell"), verdict=sell
    )
    message = scanner.ui_queue.get_nowait()
    assert message[0] == "chips"
    assert message[4] == CELL
    # The app shows the frame beside placed chips
    app = _app(monkeypatch)
    app._apply_border(message[4], placed=True)  # noqa: SLF001
    assert app.border.calls == [("show", CELL, 10)]


class _StubEngine:
    _last_capture = None
    _last_origin = None
    _last_card = None

    def layout_from_last_capture(self):  # noqa: PLR6301
        return object()


class _StubDb:
    def recycle_outputs(self, _name):  # noqa: PLR6301
        return []

    def home_only_outputs(self, _name):  # noqa: PLR6301
        return []
