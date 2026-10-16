"""Tests that the advice card takes the corner popup's place (ADVICE_CARD)."""

import queue
from types import SimpleNamespace

import pytest
from PIL import Image

import arc_helper.main as main_module
from arc_helper import advice_window
from arc_helper.advice_card import AdviceCard
from arc_helper.advice_window import AdviceWindow
from arc_helper.config import OverlaySettings
from arc_helper.database import HOME
from arc_helper.database import Item
from arc_helper.database import Need
from arc_helper.database import RecycleOutput
from arc_helper.database import get_database
from arc_helper.layout_home import fixed_home
from arc_helper.main import POPUP
from arc_helper.main import POPUP_CORNER
from arc_helper.main import POPUP_HOME
from arc_helper.main import POPUP_RIGHT
from arc_helper.main import Application
from arc_helper.main import OverlayRects
from arc_helper.main import Scanner
from arc_helper.main import ScannerState
from arc_helper.tooltip_layout import Point
from arc_helper.tooltip_layout import Rect
from arc_helper.verdict import KEEP
from arc_helper.verdict import Verdict

SCREEN = (3840, 2160)
CORNER = Point(100, 100)
# The fixed home, the LOADOUT box down to the floor at 2009
BOX = fixed_home(SCREEN)
BOX_WIDTH = BOX.right - BOX.left


def _card(title="Keep 7"):
    return AdviceCard(KEEP, title, None, 7, None, ())


class _FakeWidget:
    def __init__(self, *_args, **_kwargs):
        self.calls = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return call


class _FakeRoot:
    """Tk root stub that records after timers without running them."""

    def __init__(self):
        self.timers = {}
        self.cancelled = []
        self._next = 0

    def after(self, ms, func):
        self._next += 1
        timer = f"after#{self._next}"
        self.timers[timer] = (ms, func)
        return timer

    def after_cancel(self, timer):
        self.cancelled.append(timer)
        self.timers.pop(timer, None)


class _Recorder:
    """Stands in for the corner popup and records every call."""

    def __init__(self):
        self.calls = []

    def is_visible(self):  # noqa: PLR6301
        return False

    def __getattr__(self, name):
        def call(*args, **_kwargs):
            self.calls.append((name, args))

        return call


@pytest.fixture
def advice(monkeypatch):
    """A real AdviceWindow on stub Tk whose card is 200 px tall at box width.

    The fake card's height scales with the render scale like a real one.
    """
    backend = SimpleNamespace(tk_scale=1.0, resolution=lambda: SCREEN)
    renders = []

    def fake_render(card, scale):
        renders.append((card, scale))
        height = round(fake_render.height * scale * 560 / BOX_WIDTH)
        return Image.new("RGBA", (round(560 * scale), height))

    fake_render.height = 200
    monkeypatch.setattr(advice_window.tk, "Toplevel", _FakeWidget)
    monkeypatch.setattr(advice_window.tk, "Label", _FakeWidget)
    monkeypatch.setattr(advice_window.ImageTk, "PhotoImage", lambda image: image)
    monkeypatch.setattr(advice_window, "make_click_through", lambda _w: None)
    monkeypatch.setattr(advice_window, "get_backend", lambda: backend)
    monkeypatch.setattr(advice_window, "phys_to_tk", round)
    monkeypatch.setattr(advice_window, "render_advice_card", fake_render)
    window = AdviceWindow(None)
    window.renders = renders
    window.render = fake_render
    return window


def _app(monkeypatch, advice):
    monkeypatch.setattr(
        main_module, "get_backend", lambda: SimpleNamespace(tk_scale=1.0)
    )
    monkeypatch.setattr(main_module, "get_screen_resolution", lambda: SCREEN)
    monkeypatch.setattr(main_module, "window_rect_phys", lambda *_: None)
    app = Application.__new__(Application)
    app.settings = SimpleNamespace(overlay=SimpleNamespace(display_time=4.0))
    app.root = _FakeRoot()
    app.overlay = _Recorder()
    app.advice = advice
    app.status = SimpleNamespace(window=None)
    app.chips = None
    app.border = None
    app.debug_overlay = None
    app.ui_queue = queue.Queue()
    app._last_status = None  # noqa: SLF001
    app.overlay_rects = OverlayRects()
    app._popup_corner = CORNER  # noqa: SLF001
    app._popup_spot = POPUP_HOME  # noqa: SLF001
    app._home_card = None  # noqa: SLF001
    return app


def _pump(app, *messages):
    for message in messages:
        app.ui_queue.put(message)
    app._pump_ui_queue()  # noqa: SLF001


def _deiconifies(window):
    return [c for c in window.window.calls if c[0] == "deiconify"]


def test_show_same_and_gone_drive_the_advice_card(monkeypatch, advice):
    app = _app(monkeypatch, advice)

    _pump(app, ("advice", "Battery", _card(), 3000))
    assert advice.is_visible()
    assert len(advice.renders) == 1
    assert advice.renders[0][1] == pytest.approx(BOX_WIDTH / 560)
    timer = app._advice_hide_id  # noqa: SLF001
    assert app.root.timers[timer][0] == 3000

    # Same card touched, then the same item read from a new card: the hold
    # restarts with no re-render and no re-map
    _pump(app, ("touch", "Battery", 3500), ("advice", "Battery", _card(), 3500))
    assert len(advice.renders) == 1
    assert len(_deiconifies(advice)) == 1
    assert timer in app.root.cancelled
    assert app.root.timers[app._advice_hide_id][0] == 3500  # noqa: SLF001

    # A touch for another item does not keep this card up
    held = app._advice_hide_id  # noqa: SLF001
    _pump(app, ("touch", "Anvil I", 3500))
    assert app._advice_hide_id == held  # noqa: SLF001

    # Card gone: hidden at once, the hold timer stopped
    _pump(app, ("hide",))
    assert not advice.is_visible()
    assert held in app.root.cancelled
    assert app._advice_hide_id is None  # noqa: SLF001
    assert held not in app.root.timers

    # The corner popup was never shown, touched or moved
    assert [c for c in app.overlay.calls if c[0] != "hide"] == []


def test_hold_ending_hides_the_advice_card(monkeypatch, advice):
    app = _app(monkeypatch, advice)
    _pump(app, ("advice", "Battery", _card(), None))
    ms, func = app.root.timers[app._advice_hide_id]  # noqa: SLF001
    assert ms == 4000  # display_time without a hold

    func()

    assert not advice.is_visible()
    assert app._advice_item is None  # noqa: SLF001


def test_advice_card_rect_is_published_as_the_popup(monkeypatch, advice):
    app = _app(monkeypatch, advice)
    _pump(app, ("advice", "Battery", _card(), 3000))

    card_rect = Rect(BOX.left, BOX.top, BOX.right, BOX.top + 200)
    assert advice.rect_phys() == card_rect
    assert app.overlay_rects.labelled()[POPUP] == card_rect
    assert card_rect in app.overlay_rects.snapshot()

    _pump(app, ("hide",))
    assert POPUP not in app.overlay_rects.labelled()
    assert app.overlay_rects.snapshot() == ()


def test_spots_use_the_card_size(monkeypatch, advice):
    app = _app(monkeypatch, advice)
    # Before the first show the spots are the size of the home box
    home, corner = app._popup_spots()  # noqa: SLF001
    assert home == BOX
    assert corner == Rect(100, 100, 100 + BOX_WIDTH, 100 + BOX.bottom - BOX.top)

    # A card too tall for the box is scaled down, never raised above home
    advice.render.height = 500
    _pump(app, ("advice", "Battery", _card(), 3000))
    rect = advice.rect_phys()
    assert (rect.left, rect.top) == (BOX.left, BOX.top)
    assert rect.bottom <= BOX.bottom
    assert rect.right - rect.left < BOX_WIDTH
    home, corner = app._popup_spots()  # noqa: SLF001
    assert home == rect
    labelled = app.overlay_rects.labelled()
    assert (labelled[POPUP_HOME], labelled[POPUP_CORNER]) == (home, corner)


def test_card_moves_to_escape_spots_and_back_home(monkeypatch, advice):
    app = _app(monkeypatch, advice)
    _pump(app, ("advice", "Battery", _card(), 3000))

    # A game card under the advice card sends it aside at once, same size
    _pump(app, ("popup_spot", POPUP_RIGHT, Rect(2200, BOX.top, 2880, BOX.top + 200)))
    assert advice.rect_phys() == Rect(2200, BOX.top, 2880, BOX.top + 200)
    assert app.overlay_rects.popup_away()

    _pump(app, ("popup_spot", POPUP_CORNER, Rect(100, 100, 780, 300)))
    assert advice.rect_phys() == Rect(100, 100, 100 + BOX_WIDTH, 300)
    assert app.overlay_rects.popup_away()

    # The next card shown while away keeps the size it has at home
    _pump(app, ("advice", "Anvil I", _card("Keep 1"), 3000))
    assert advice.rect_phys() == Rect(100, 100, 100 + BOX_WIDTH, 300)

    _pump(app, ("popup_spot", POPUP_HOME, None))
    assert advice.rect_phys() == Rect(BOX.left, BOX.top, BOX.right, BOX.top + 200)
    assert not app.overlay_rects.popup_away()


def test_found_home_moves_the_card_beneath_the_second_gun(monkeypatch, advice):
    app = _app(monkeypatch, advice)
    _pump(app, ("advice", "Battery", _card(), 3000))

    # Home beneath the second gun on the base stash screen: as wide as the
    # gun, 13 px below it
    gun_home = Rect(1384, 1631, 1994, 2009)
    _pump(app, ("popup_home", gun_home))
    rect = advice.rect_phys()
    assert (rect.left, rect.top) == (1384, 1631)
    assert rect.right - rect.left <= 610
    assert app.overlay_rects.labelled()[POPUP_HOME] == rect


def test_advice_card_false_keeps_the_popup_path(monkeypatch):
    monkeypatch.setattr(
        main_module, "get_backend", lambda: SimpleNamespace(tk_scale=1.0)
    )
    app = Application.__new__(Application)
    app.overlay = _Recorder()
    app.chips = None
    app.ui_queue = queue.Queue()
    app._last_status = None  # noqa: SLF001
    monkeypatch.setattr(app, "_after_popup_show", lambda: None)
    monkeypatch.setattr(app, "_publish_overlay_rects", lambda: None)
    app.root = _FakeRoot()
    app.db = SimpleNamespace(recycle_outputs=lambda _name: [])
    battery = Item(name="Battery", action="Keep")

    _pump(
        app,
        ("overlay", "Battery", battery, "verdict", 3000),
        ("touch", "Battery", 3000),
        ("hide",),
    )

    assert [c[0] for c in app.overlay.calls] == ["show", "touch"]
    assert app.advice is None


def test_advice_card_setting_defaults_on_and_reads_advice_card(monkeypatch):
    monkeypatch.delenv("ADVICE_CARD", raising=False)
    assert OverlaySettings(_env_file=None).advice_card is True
    monkeypatch.setenv("ADVICE_CARD", "false")
    assert OverlaySettings(_env_file=None).advice_card is False


class _FakeHotkey:
    def stop(self):
        pass


class _FakeDb:
    """Battery needs 3 for a quest, recycles at home into Wires and Rubber."""

    def lookup(self, name, tier_vote=None):  # noqa: ARG002, PLR6301
        return {"Battery": Item(name="Battery", action="Keep")}.get(name)

    def needs_for(self, name):  # noqa: PLR6301
        if name == "Battery":
            return [Need("Battery", "quest", "Power Up", None, 3, None, None)]
        return []

    def recycle_outputs(self, name):  # noqa: PLR6301
        if name != "Battery":
            return []
        return [
            RecycleOutput("Wires", 2, HOME),
            RecycleOutput("Rubber", 1, "field"),
        ]

    def log_missing_item(self, name):
        pass


class _NoCardOcr:
    last_tier_vote = None
    _last_card = None
    _last_origin = None


def _scanner(monkeypatch, *, advice_card, db=None):
    scanner = Scanner(
        root=None,
        overlay=None,
        status=None,
        db=db or _FakeDb(),
        hotkey=_FakeHotkey(),
        advice_card=advice_card,
    )
    scanner._running = True  # noqa: SLF001
    scanner.state = ScannerState.ACTIVE
    scanner._hold_ms = 3000  # noqa: SLF001
    monkeypatch.setattr(main_module, "get_ocr_engine", _NoCardOcr)
    monkeypatch.setattr(scanner, "_queue_chips", lambda *_, **__: None)
    return scanner


def _drain(scanner):
    messages = []
    while not scanner.ui_queue.empty():
        messages.append(scanner.ui_queue.get_nowait())
    return messages


def test_scanner_builds_the_card_on_the_show_path(monkeypatch):
    scanner = _scanner(monkeypatch, advice_card=True)
    verdict = Verdict(KEEP, "3 for Power Up", 3, (("Power Up", 3),))
    monkeypatch.setattr(scanner, "_verdict_for", lambda *_: verdict)

    scanner._handle_detected_item("Battery")  # noqa: SLF001

    messages = _drain(scanner)
    assert [m[0] for m in messages] == ["advice"]
    _, name, card, hold_ms = messages[0]
    assert (name, hold_ms) == ("Battery", 3000)
    assert card == AdviceCard(
        KEEP, "Keep 3", "For Power Up", 3, None, (("Quest", "Power Up", 3),)
    )


def test_scanner_without_advice_card_queues_the_popup(monkeypatch):
    scanner = _scanner(monkeypatch, advice_card=False)
    monkeypatch.setattr(scanner, "_verdict_for", lambda *_: "verdict")

    scanner._handle_detected_item("Battery")  # noqa: SLF001

    messages = _drain(scanner)
    assert [m[0] for m in messages] == ["overlay"]
    assert messages[0][3] == "verdict"


def test_real_database_cards_for_snitch_scanner_and_microscope(monkeypatch):
    """Show path with the real items.db (stub Tk), owned unknown."""
    scanner = _scanner(monkeypatch, advice_card=True, db=get_database())

    scanner._handle_detected_item("Snitch Scanner")  # noqa: SLF001
    scanner._handle_detected_item("Microscope")  # noqa: SLF001

    (_, snitch_name, snitch, _), (_, scope_name, scope, _) = _drain(scanner)
    assert snitch_name == "Snitch Scanner"
    assert snitch.title == "Keep 7"
    assert snitch.subtitle == "For Utility Station level 2 and Wires Crossed"
    assert snitch.rows == (
        ("Workshop", "Utility Station level 2", 6),
        ("Quest", "Wires Crossed", 1),
    )
    assert scope_name == "Microscope"
    assert scope.title == "Recycle it"
    assert scope.subtitle == (
        "Recycles into Advanced Mechanical Components, 3 Magnet, which you still need"
    )
    assert scope.rows == ()


def test_unmatched_read_shows_nothing_and_hides_the_previous_item(monkeypatch):
    scanner = _scanner(monkeypatch, advice_card=True)
    missing = []
    monkeypatch.setattr(scanner.db, "log_missing_item", missing.append)
    monkeypatch.setattr(scanner, "_verdict_for", lambda *_: Verdict(KEEP, "r"))
    scanner._handle_detected_item("Battery")  # noqa: SLF001
    scanner._card_shown = object()  # noqa: SLF001 - Battery's card is up
    _drain(scanner)

    assert scanner._handle_detected_item("IUST'", new_card=True)  # noqa: SLF001

    assert _drain(scanner) == [("hide",)]
    assert missing == ["IUST'"]
    assert not scanner._last_shown_item  # noqa: SLF001


def test_unmatched_read_with_nothing_shown_queues_no_windows(monkeypatch):
    scanner = _scanner(monkeypatch, advice_card=True)
    scanner._handle_detected_item("IUST'", new_card=True)  # noqa: SLF001
    assert _drain(scanner) == []
