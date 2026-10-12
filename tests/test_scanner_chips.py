"""Tests that leaving ACTIVE hides the tooltip chips and keeps them hidden."""

import threading
import time
from typing import ClassVar

import pytest

import arc_helper.main as main_module
import arc_helper.overlay as overlay_module
from arc_helper.config import get_settings
from arc_helper.database import Item
from arc_helper.main import Scanner
from arc_helper.main import ScannerState
from arc_helper.overlay import TooltipChips
from arc_helper.tooltip_layout import PRICE_CHIP
from arc_helper.tooltip_layout import RECYCLE_CHIP
from arc_helper.tooltip_layout import Point
from arc_helper.tooltip_layout import Rect
from arc_helper.tooltip_layout import TooltipLayout

LAYOUT = TooltipLayout(Rect(0, 0, 600, 900), {})
ITEM = Item(name="Battery", action="Keep", sell_price=250)


class _FakeHotkey:
    def stop(self):
        pass


class _FakeDb:
    # Raw OCR reads that resolve to a catalog item
    items: ClassVar[dict[str, Item]] = {"Battery": ITEM}

    def lookup(self, name, tier_vote=None):  # noqa: ARG002
        return self.items.get(name)

    def recycle_outputs(self, name):  # noqa: ARG002, PLR6301
        return []

    def home_only_outputs(self, name):  # noqa: ARG002, PLR6301
        return []


class _BlockingOcr:
    """layout_from_last_capture blocks until released, like a slow OCR pass."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def layout_from_last_capture(self):
        self.started.set()
        self.release.wait(timeout=5)
        return LAYOUT


def _active_scanner():
    scanner = Scanner(
        root=None, overlay=None, status=None, db=_FakeDb(), hotkey=_FakeHotkey()
    )
    scanner._running = True  # noqa: SLF001 - test seam
    scanner.state = ScannerState.ACTIVE
    return scanner


def _drain(scanner):
    messages = []
    while not scanner.ui_queue.empty():
        messages.append(scanner.ui_queue.get_nowait())
    return messages


def test_pause_from_active_hides_chips():
    scanner = _active_scanner()
    scanner.pause()
    assert scanner.state == ScannerState.PAUSED
    assert _drain(scanner) == [("chips_hide",)]


def test_chips_queued_while_active(monkeypatch):
    scanner = _active_scanner()
    ocr = _BlockingOcr()
    ocr.release.set()
    monkeypatch.setattr(main_module, "get_ocr_engine", lambda: ocr)
    scanner._queue_chips(ITEM)  # noqa: SLF001
    assert [m[0] for m in _drain(scanner)] == ["chips"]


@pytest.mark.parametrize("leave", ["stop", "pause"])
def test_no_chips_after_leaving_active_during_layout_ocr(monkeypatch, leave):
    scanner = _active_scanner()
    ocr = _BlockingOcr()
    monkeypatch.setattr(main_module, "get_ocr_engine", lambda: ocr)

    worker = threading.Thread(target=scanner._queue_chips, args=(ITEM,))  # noqa: SLF001
    worker.start()
    assert ocr.started.wait(timeout=5)
    getattr(scanner, leave)()  # the main thread leaves ACTIVE mid-OCR
    ocr.release.set()
    worker.join(timeout=5)

    assert _drain(scanner) == [("chips_hide",)]


class _NoVoteOcr:
    last_tier_vote = None


def test_repeat_detection_in_cooldown_only_touches_the_windows(monkeypatch):
    monkeypatch.setattr(main_module, "get_ocr_engine", _NoVoteOcr)
    scanner = _active_scanner()
    # The same item was shown a moment ago, well inside the 2 s cooldown
    scanner._last_shown_item = "Battery"  # noqa: SLF001
    scanner._last_shown_time = time.time()  # noqa: SLF001
    scanner._handle_detected_item("Battery")  # noqa: SLF001
    assert _drain(scanner) == [("touch", "Battery")]


def test_raw_read_variants_of_one_item_show_once_then_touch(monkeypatch):
    monkeypatch.setattr(main_module, "get_ocr_engine", _NoVoteOcr)
    anvil = Item(name="Anvil IV", action="Keep")
    scanner = _active_scanner()
    # Three raw reads of the one hovered card, as Tesseract varies them
    scanner.db.items = {"ANVIL IV": anvil, "ANVILIV": anvil, "ANVILLV": anvil}
    monkeypatch.setattr(scanner, "_verdict_for", lambda *_: "verdict")
    chips = []
    monkeypatch.setattr(scanner, "_queue_chips", lambda rec, **_: chips.append(rec))
    for raw in ("ANVIL IV", "ANVILIV", "ANVILLV"):
        scanner._handle_detected_item(raw)  # noqa: SLF001
    assert _drain(scanner) == [
        ("overlay", "Anvil IV", anvil, "verdict"),
        ("touch", "Anvil IV"),
        ("touch", "Anvil IV"),
    ]
    # Chips are only queued with the first show
    assert len(chips) == int(get_settings().overlay.tooltip_chips)


class _StubWindow:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name in {"winfo_reqwidth", "winfo_reqheight"}:
            return lambda: 100
        return lambda *_, **__: self.calls.append(name)


class _StubRoot:
    def after(self, ms, func):  # noqa: ARG002, PLR6301
        return "job"

    def after_cancel(self, job):
        pass


def test_chips_for_the_same_item_move_without_withdraw(monkeypatch):
    positions = {PRICE_CHIP: Point(100, 200)}
    monkeypatch.setattr(overlay_module, "place_chips", lambda *_: dict(positions))
    monkeypatch.setattr(overlay_module, "get_screen_resolution", lambda: (3840, 2160))
    monkeypatch.setattr(overlay_module, "phys_to_tk", lambda v: v)
    monkeypatch.setattr(
        overlay_module, "get_backend", lambda: type("B", (), {"tk_scale": 1.0})
    )
    chips = TooltipChips.__new__(TooltipChips)
    chips.root = _StubRoot()
    chips.settings = get_settings()
    chips.recycle_accent = "#000"
    for label in ("price_label", "stack_label", "recycle_title", "recycle_text"):
        setattr(chips, label, _StubWindow())
    chips.windows = {PRICE_CHIP: _StubWindow(), RECYCLE_CHIP: _StubWindow()}
    chips._hide_after_id = None  # noqa: SLF001
    chips._current_item = None  # noqa: SLF001
    chips._shown = set()  # noqa: SLF001
    chips._positions = {}  # noqa: SLF001
    text = {"price_text": "SELLS FOR 250", "stack_text": None}
    text |= {"recycle_label": None, "recycle_text": None}

    chips.show("Battery", LAYOUT, **text)
    price = chips.windows[PRICE_CHIP]
    for window in chips.windows.values():
        window.calls.clear()
    # Same item: the tooltip moved, then this layout read placed nothing
    positions[PRICE_CHIP] = Point(140, 260)
    chips.show("Battery", LAYOUT, **text)
    positions.clear()
    chips.show("Battery", LAYOUT, **text)

    assert "withdraw" not in price.calls
    assert "withdraw" not in chips.windows[RECYCLE_CHIP].calls
    assert "deiconify" not in price.calls
    assert price.calls.count("geometry") == 1
