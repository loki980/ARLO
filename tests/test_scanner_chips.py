"""Tests that leaving ACTIVE hides the tooltip chips and keeps them hidden."""

from arc_helper.main import Scanner
from arc_helper.main import ScannerState


class _FakeHotkey:
    def stop(self):
        pass


class _FakeDb:
    def recycle_outputs(self, name):  # noqa: ARG002, PLR6301
        return []

    def home_only_outputs(self, name):  # noqa: ARG002, PLR6301
        return []


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
