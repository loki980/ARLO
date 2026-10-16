"""Tests for the IDLE look at the Collections screen in the scan loop."""

import numpy as np
import pytest

import arc_helper.main as main_module
from arc_helper.collections_screen import CollectionsCell
from arc_helper.collections_screen import CollectionsRead
from arc_helper.config import get_settings
from arc_helper.main import COLLECTIONS_CHECK_S
from arc_helper.main import Scanner
from arc_helper.main import ScannerState
from arc_helper.tooltip_layout import Rect

OTHER, TOOLTIP, CLEAN = 0, 1, 2


def _read(*, consistent: bool) -> CollectionsRead:
    found = [True, False, True]
    cells = [
        CollectionsCell(i + 1, Rect(0, 0, 1, 1), flag) for i, flag in enumerate(found)
    ]
    return CollectionsRead(sum(found), len(found), cells, consistent)


# What read_collections makes of each fake frame, keyed by its first pixel
READS = {
    OTHER: None,
    TOOLTIP: _read(consistent=False),
    CLEAN: _read(consistent=True),
}


class _FakeDb:
    def __init__(self):
        self.owned = {}

    def blueprint_at(self, grid_index):  # noqa: PLR6301
        return {1: "Anvil Blueprint", 2: "Banjo Blueprint"}.get(grid_index)

    def set_blueprint_owned(self, name, *, owned):
        self.owned[name] = owned


class _FakeHotkey:
    held = False

    def is_held(self):
        return self.held

    def stop(self):
        pass


def _frame(kind):
    return np.full((2, 2, 3), kind, dtype=np.uint8)


@pytest.fixture
def scanner(monkeypatch):
    scanner = Scanner(
        root=None, overlay=None, status=None, db=_FakeDb(), hotkey=_FakeHotkey()
    )
    scanner.frames = []
    scanner.grabs = 0

    def grab():
        scanner.grabs += 1
        return _frame(scanner.frames.pop(0))

    monkeypatch.setattr(scanner, "_grab_full_screen", grab)
    monkeypatch.setattr(
        main_module, "read_collections", lambda rgb: READS[int(rgb[0, 0, 0])]
    )
    scanner.records = 0
    real_record = main_module.record_collections

    def record(db, read):
        scanner.records += 1
        return real_record(db, read)

    monkeypatch.setattr(main_module, "record_collections", record)
    return scanner


def test_one_record_per_visit(scanner, caplog):
    sequence = [OTHER, TOOLTIP, CLEAN, CLEAN, OTHER, CLEAN]
    scanner.frames = list(sequence)
    with caplog.at_level("INFO", logger=main_module.logger.name):
        for i in range(len(sequence)):
            scanner._check_collections(i * COLLECTIONS_CHECK_S)  # noqa: SLF001
    assert scanner.grabs == len(sequence)
    assert scanner.records == 2
    assert scanner.db.owned == {"Anvil Blueprint": True, "Banjo Blueprint": False}
    messages = [r.getMessage() for r in caplog.records]
    assert messages.count("collections read inconsistent") == 1
    assert messages.count("collections read found=2/3 recorded=2 unknown=[3]") == 2


def test_inconsistent_read_logged_once_and_retried(scanner, caplog):
    scanner.frames = [TOOLTIP, TOOLTIP, TOOLTIP, CLEAN]
    with caplog.at_level("INFO", logger=main_module.logger.name):
        for i in range(4):
            scanner._check_collections(i * COLLECTIONS_CHECK_S)  # noqa: SLF001
    messages = [r.getMessage() for r in caplog.records]
    assert messages.count("collections read inconsistent") == 1
    assert scanner.records == 1


def test_check_interval_holds_on_a_fake_clock(scanner):
    scanner.frames = [OTHER] * 10
    for now in (100.0, 100.5, 101.0, 101.9, 102.0, 103.99, 104.0):
        scanner._check_collections(now)  # noqa: SLF001
    # Looks at 100.0, 102.0 and 104.0 only
    assert scanner.grabs == 3


class _FakeOcr:
    def __init__(self, scanner, hits):
        self.scanner = scanner
        self.hits = list(hits)
        self.calls = 0

    def check_trigger_any(self, regions):  # noqa: ARG002
        self.calls += 1
        hit = self.hits.pop(0)
        if hit:
            # Stop the loop right after the IDLE round that saw INVENTORY
            self.scanner._running = False  # noqa: SLF001
        return hit


def test_trigger_path_still_fires_on_inventory(scanner, monkeypatch):
    ocr = _FakeOcr(scanner, [False, False, True])
    monkeypatch.setattr(main_module, "get_ocr_engine", lambda: ocr)
    sleeps = []
    monkeypatch.setattr(main_module.time, "sleep", sleeps.append)
    scanner.frames = [OTHER] * 5
    scanner._running = True  # noqa: SLF001
    scanner._scan_loop()  # noqa: SLF001
    assert scanner.state == ScannerState.ACTIVE
    assert ocr.calls == 3
    # Two idle waits, each at most the trigger interval, with one look at the
    # Collections screen between them (the second is under COLLECTIONS_CHECK_S)
    interval = get_settings().scan.trigger_scan_interval
    assert len(sleeps) == 2
    assert all(0.0 <= s <= interval for s in sleeps)
    assert scanner.grabs >= 1


def test_hotkey_path_skips_the_collections_look(scanner, monkeypatch):
    ocr = _FakeOcr(scanner, [])
    monkeypatch.setattr(main_module, "get_ocr_engine", lambda: ocr)
    scanner.hotkey.held = True

    def stop_after_one(*args, **kwargs):  # noqa: ARG001
        scanner._running = False  # noqa: SLF001

    monkeypatch.setattr(scanner, "_reset_scan_timing", stop_after_one)
    scanner._running = True  # noqa: SLF001
    scanner._scan_loop()  # noqa: SLF001
    assert scanner.state == ScannerState.ACTIVE
    assert ocr.calls == 0
    assert scanner.grabs == 0
