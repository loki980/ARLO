"""Tests that leaving ACTIVE hides the tooltip chips and keeps them hidden."""

import threading

import pytest

import arc_helper.main as main_module
from arc_helper.database import Item
from arc_helper.main import Scanner
from arc_helper.main import ScannerState
from arc_helper.tooltip_layout import Rect
from arc_helper.tooltip_layout import TooltipLayout

LAYOUT = TooltipLayout(Rect(0, 0, 600, 900), {})
ITEM = Item(name="Battery", action="Keep", sell_price=250)


class _FakeHotkey:
    def stop(self):
        pass


class _FakeDb:
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
