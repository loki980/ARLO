"""Tests that leaving ACTIVE hides the tooltip chips and keeps them hidden."""

import threading
import time
from typing import ClassVar

import pytest
from PIL import Image

import arc_helper.main as main_module
import arc_helper.overlay as overlay_module
from arc_helper.config import get_settings
from arc_helper.database import Item
from arc_helper.main import Scanner
from arc_helper.main import ScannerState
from arc_helper.main import chip_payload
from arc_helper.overlay import TooltipChips
from arc_helper.tooltip_layout import PRICE_CHIP
from arc_helper.tooltip_layout import RECYCLE_CHIP
from arc_helper.tooltip_layout import Point
from arc_helper.tooltip_layout import Rect
from arc_helper.tooltip_layout import TooltipLayout
from arc_helper.verdict import Verdict

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


def test_pause_from_active_hides_the_card_windows():
    scanner = _active_scanner()
    scanner.pause()
    assert scanner.state == ScannerState.PAUSED
    assert _drain(scanner) == [("hide",)]


def test_chips_queued_while_active(monkeypatch):
    scanner = _active_scanner()
    ocr = _BlockingOcr()
    ocr.release.set()
    monkeypatch.setattr(main_module, "get_ocr_engine", lambda: ocr)
    scanner._queue_chips(ITEM)  # noqa: SLF001
    assert [m[0] for m in _drain(scanner)] == ["chips"]


def test_chip_payload_takes_word_and_kind_from_the_verdict_not_the_item():
    # ITEM.action is "Keep" but the verdict says sell
    payload = chip_payload(ITEM, [], [], Verdict("sell", "no needs"))
    assert payload["verdict_word"] == "SELL"
    assert payload["verdict_kind"] == "sell"
    assert chip_payload(ITEM, [], [])["verdict_word"] is None
    assert chip_payload(ITEM, [], [], Verdict("unknown", ""))["verdict_kind"] is None


def test_queued_chips_message_carries_the_verdict(monkeypatch):
    scanner = _active_scanner()
    ocr = _BlockingOcr()
    ocr.release.set()
    monkeypatch.setattr(main_module, "get_ocr_engine", lambda: ocr)
    scanner._queue_chips(ITEM, verdict=Verdict("recycle", "r"))  # noqa: SLF001
    (message,) = _drain(scanner)
    assert message[3]["verdict_word"] == "RECYCLE"
    assert message[3]["verdict_kind"] == "recycle"


def test_detected_item_passes_its_verdict_to_the_chips(monkeypatch):
    monkeypatch.setattr(main_module, "get_ocr_engine", _NoVoteOcr)
    scanner = _active_scanner()
    verdict = Verdict("recycle", "r")
    monkeypatch.setattr(scanner, "_verdict_for", lambda *_: verdict)
    seen = []
    monkeypatch.setattr(
        scanner, "_queue_chips", lambda _rec, **kw: seen.append(kw.get("verdict"))
    )
    scanner._handle_detected_item("Battery")  # noqa: SLF001
    assert seen == [verdict] * int(get_settings().overlay.tooltip_chips)


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

    assert _drain(scanner) == [("hide",)]


class _NoVoteOcr:
    last_tier_vote = None


def test_repeat_detection_in_cooldown_only_touches_the_windows(monkeypatch):
    monkeypatch.setattr(main_module, "get_ocr_engine", _NoVoteOcr)
    scanner = _active_scanner()
    # The same item was shown a moment ago, well inside the 2 s cooldown
    scanner._last_shown_item = "Battery"  # noqa: SLF001
    scanner._last_shown_time = time.time()  # noqa: SLF001
    scanner._handle_detected_item("Battery")  # noqa: SLF001
    assert _drain(scanner) == [("touch", "Battery", None)]


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
        ("overlay", "Anvil IV", anvil, "verdict", None),
        ("touch", "Anvil IV", None),
        ("touch", "Anvil IV", None),
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
    chips.verdict_label = _StubWindow()
    chips.verdict_stripe = _StubWindow()
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


class _CaptureOcr:
    """A layout read plus the last capture, origin and card the border uses."""

    def __init__(self):
        self._last_capture = Image.new("RGB", (400, 300))
        self._last_origin = Point(100, 50)
        self._last_card = Rect(200, 0, 400, 300)

    def layout_from_last_capture(self):  # noqa: PLR6301
        return LAYOUT


def _stub_search(monkeypatch, result):
    calls = []

    def search(rgb, card, ignore=()):
        calls.append((rgb.shape, card, list(ignore)))
        return result

    monkeypatch.setattr(main_module, "find_hovered_cell", search)
    monkeypatch.setattr(main_module, "get_ocr_engine", _CaptureOcr)
    return calls


def test_keep_with_a_found_cell_queues_the_border_on_screen(monkeypatch):
    calls = _stub_search(monkeypatch, Rect(10, 20, 30, 40))
    scanner = _active_scanner()
    # An ARLO chip on screen, which the search must see in capture coordinates
    scanner.overlay_rects.publish([Rect(150, 60, 200, 100)])

    scanner._queue_chips(ITEM, verdict=Verdict("keep", "k", 1, (("Power Up", 1),)))  # noqa: SLF001

    (message,) = _drain(scanner)
    assert message[0] == "chips"
    assert message[4] == Rect(110, 70, 130, 90)
    assert calls == [((300, 400, 3), Rect(200, 0, 400, 300), [Rect(50, 10, 100, 50)])]


def test_keep_without_active_needs_queues_the_border(monkeypatch):
    calls = _stub_search(monkeypatch, Rect(10, 20, 30, 40))
    scanner = _active_scanner()

    # the weapon rule: keep with no sources
    scanner._queue_chips(ITEM, verdict=Verdict("keep", "Weapon"))  # noqa: SLF001

    (message,) = _drain(scanner)
    assert message[4] == Rect(110, 70, 130, 90)
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["sell", "recycle"])
def test_other_verdicts_queue_the_border(monkeypatch, kind):
    calls = _stub_search(monkeypatch, Rect(10, 20, 30, 40))
    scanner = _active_scanner()

    scanner._queue_chips(ITEM, verdict=Verdict(kind, "x"))  # noqa: SLF001

    (message,) = _drain(scanner)
    assert message[4] == Rect(110, 70, 130, 90)
    assert len(calls) == 1


def test_keep_without_a_found_cell_queues_a_border_hide(monkeypatch):
    calls = _stub_search(monkeypatch, None)
    scanner = _active_scanner()

    scanner._queue_chips(ITEM, verdict=Verdict("keep", "k", 1, (("Power Up", 1),)))  # noqa: SLF001

    (message,) = _drain(scanner)
    assert message[4] is None
    assert len(calls) == 1
