"""Tests for the IDLE look at the Collections screen in the scan loop."""

import numpy as np
import pytest

import arc_helper.main as main_module
from arc_helper.badge_surface import Icon
from arc_helper.collections_screen import CollectionsCell
from arc_helper.collections_screen import CollectionsRead
from arc_helper.config import get_settings
from arc_helper.database import Blueprint
from arc_helper.main import COLLECTIONS_CHECK_S
from arc_helper.main import Scanner
from arc_helper.main import ScannerState
from arc_helper.tooltip_layout import Point
from arc_helper.tooltip_layout import Rect

OTHER, TOOLTIP, CLEAN, GRID = 0, 1, 2, 3


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
    # A row of three 100 px cells 8 px apart: Anvil empty, Banjo found and an
    # unnamed slot empty
    GRID: CollectionsRead(
        1,
        3,
        [
            CollectionsCell(1, Rect(0, 0, 100, 100), found=False),
            CollectionsCell(2, Rect(108, 0, 208, 100), found=True),
            CollectionsCell(3, Rect(216, 0, 316, 100), found=False),
        ],
        consistent=True,
    ),
}
ANVIL, BANJO, UNNAMED, GAP = (
    Point(50, 50),
    Point(150, 50),
    Point(250, 50),
    Point(104, 50),
)


class _FakeDb:
    def __init__(self):
        self.owned = {}

    def blueprint_at(self, grid_index):  # noqa: PLR6301
        return {1: "Anvil Blueprint", 2: "Banjo Blueprint"}.get(grid_index)

    def set_blueprint_owned(self, name, *, owned):
        self.owned[name] = owned

    def blueprint_catalog(self):  # noqa: PLR6301
        where = ("All", "Any", "Yes", "Raider Containers", "No", "Yes")
        return [Blueprint("Anvil Blueprint", None, None, None, *where)]


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


@pytest.fixture
def hover(scanner, monkeypatch):
    """The scanner with the advice card on and a cursor the test moves."""
    scanner.advice_card = True
    scanner.cursor = ANVIL

    class _Backend:
        def cursor_position(self):  # noqa: PLR6301
            return scanner.cursor

    monkeypatch.setattr(main_module, "get_backend", _Backend)
    return scanner


def _messages(scanner):
    out = []
    while not scanner.ui_queue.empty():
        out.append(scanner.ui_queue.get_nowait())
    return out


def _tick(scanner, now, cursor):
    """One IDLE round: the screen look (when due), then the hover check."""
    scanner.cursor = cursor
    scanner._check_collections(now)  # noqa: SLF001
    scanner._check_collections_hover()  # noqa: SLF001
    return _messages(scanner)


def test_hover_on_an_empty_cell_shows_its_card_and_touches_it(hover):
    hover.frames = [GRID]
    shown = _tick(hover, 0.0, ANVIL)
    assert [m[0] for m in shown] == ["advice"]
    card = shown[0][2]
    assert (card.title, card.subtitle) == ("Anvil Blueprint", "Not found yet")
    assert ("Map", "All maps") in card.rows
    assert ("Containers", "Raider Containers") in card.rows
    # Later rounds inside the read interval touch the card without a grab
    assert _tick(hover, 0.5, ANVIL) == [("touch", "Anvil Blueprint", None)]
    assert _tick(hover, 1.0, Point(99, 99)) == [("touch", "Anvil Blueprint", None)]
    assert hover.grabs == 1


def test_hover_on_an_unnamed_slot_says_so(hover):
    hover.frames = [GRID]
    (message,) = _tick(hover, 0.0, UNNAMED)
    assert message[1] == "Blueprint slot 3"
    assert message[2].subtitle == "Unnamed slot, not in the public data"


def test_no_card_on_a_found_cell_or_between_cells(hover):
    hover.frames = [GRID]
    assert _tick(hover, 0.0, BANJO) == []
    assert _tick(hover, 0.5, GAP) == []
    assert _tick(hover, 1.0, Point(400, 50)) == []


def test_card_hides_on_leaving_the_cell(hover):
    hover.frames = [GRID]
    assert [m[0] for m in _tick(hover, 0.0, ANVIL)] == ["advice"]
    # Onto a found cell, then off the grid: one hide, then nothing
    assert _tick(hover, 0.5, BANJO) == [("hide",)]
    assert _tick(hover, 1.0, Point(400, 50)) == []
    # Off the grid straight from an empty cell
    assert [m[0] for m in _tick(hover, 1.5, UNNAMED)] == ["advice"]
    assert _tick(hover, 1.6, Point(400, 50)) == [("hide",)]
    # From one empty cell to another: the new card replaces the old one
    _tick(hover, 1.7, ANVIL)
    assert [m[1] for m in _tick(hover, 1.8, UNNAMED)] == ["Blueprint slot 3"]


def test_card_hides_on_leaving_the_screen(hover):
    hover.frames = [GRID, OTHER]
    assert [m[0] for m in _tick(hover, 0.0, ANVIL)] == ["advice"]
    # The next look finds another screen: the card hides, the cursor still
    # over where the cell was shows nothing
    assert _tick(hover, COLLECTIONS_CHECK_S, ANVIL) == [("hide",)]
    assert _tick(hover, COLLECTIONS_CHECK_S + 0.5, ANVIL) == []


def test_hover_keeps_the_last_consistent_cells(hover):
    # A game tooltip over the grid makes the next read inconsistent (its
    # cells all 1 px, found at 1 and 3): the consistent cells stay in use
    hover.frames = [GRID, TOOLTIP]
    _tick(hover, 0.0, BANJO)
    assert _tick(hover, COLLECTIONS_CHECK_S, BANJO) == []
    assert [m[1] for m in _tick(hover, COLLECTIONS_CHECK_S + 0.5, ANVIL)] == [
        "Anvil Blueprint"
    ]


def test_hover_needs_the_advice_card(hover):
    hover.advice_card = False
    hover.frames = [GRID]
    assert _tick(hover, 0.0, ANVIL) == []


class _FakeSurface:
    """The badge surface, recording what the scanner sends."""

    def __init__(self):
        self.sent = []
        self.starts = 0

    def start(self):
        self.starts += 1
        return True

    def draw(self, badges):
        self.sent.append(("draw", badges))

    def clear(self):
        self.sent.append(("clear",))


@pytest.fixture
def icons(scanner, monkeypatch, tmp_path):
    """The scanner with a badge surface and Anvil's and Banjo's icons cached.

    Anvil (slot 1) is empty, Banjo (slot 2) found and slot 3 unnamed in GRID.
    """
    scanner.badge_surface = _FakeSurface()
    scanner.icon_cache_dir = tmp_path
    for item_id in ("anvil_blueprint", "banjo_blueprint"):
        (tmp_path / f"{item_id}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    where = ("All", "Any", "Yes", "Raider Containers", "No", "Yes")
    catalog = [
        Blueprint("Anvil Blueprint", "anvil_blueprint", None, None, *where),
        Blueprint("Banjo Blueprint", "banjo_blueprint", None, None, *where),
    ]
    monkeypatch.setattr(scanner.db, "blueprint_catalog", lambda: catalog)
    return scanner


def _looks(scanner, frames):
    scanner.frames = list(frames)
    for i in range(len(frames)):
        scanner._check_collections(i * COLLECTIONS_CHECK_S)  # noqa: SLF001


def test_icons_only_on_empty_named_cells(icons, tmp_path):
    _looks(icons, [GRID])
    assert icons.badge_surface.starts == 1
    assert icons.badge_surface.sent == [
        (
            "draw",
            [
                Icon(
                    (0, 0, 100, 100),
                    str(tmp_path / "anvil_blueprint.png"),
                    alpha=0.35,
                    inset=0.12,
                )
            ],
        )
    ]


def test_no_icon_without_a_cached_file(icons, tmp_path):
    (tmp_path / "anvil_blueprint.png").unlink()
    _looks(icons, [GRID])
    assert icons.badge_surface.sent == []
    assert icons.badge_surface.starts == 0


def test_icons_wait_for_a_consistent_read(icons):
    # The tooltip read draws nothing, the clean one draws Banjo in empty slot 2
    icons.frames = [TOOLTIP, CLEAN]
    icons._check_collections(0.0)  # noqa: SLF001
    assert icons.badge_surface.sent == []
    icons._check_collections(COLLECTIONS_CHECK_S)  # noqa: SLF001
    [(op, badges)] = icons.badge_surface.sent
    assert op == "draw"
    assert [b.path.endswith("banjo_blueprint.png") for b in badges] == [True]


def test_icons_sent_once_per_read_not_every_round(icons):
    icons.frames = [GRID, GRID]
    # IDLE rounds every 0.5 s: looks at 0 and 2 s only, the second read
    # finds the same icons and sends nothing
    for now in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
        icons._check_collections(now)  # noqa: SLF001
    assert icons.grabs == 2
    assert [m[0] for m in icons.badge_surface.sent] == ["draw"]


def test_icons_cleared_on_leaving_the_screen(icons):
    _looks(icons, [GRID, OTHER, OTHER])
    assert [m[0] for m in icons.badge_surface.sent] == ["draw", "clear"]


def test_icons_cleared_on_entering_active(icons, monkeypatch):
    ocr = _FakeOcr(icons, [False, True])
    monkeypatch.setattr(main_module, "get_ocr_engine", lambda: ocr)
    monkeypatch.setattr(main_module.time, "sleep", [].append)
    icons.frames = [GRID]
    icons._running = True  # noqa: SLF001
    icons._scan_loop()  # noqa: SLF001
    assert icons.state == ScannerState.ACTIVE
    assert [m[0] for m in icons.badge_surface.sent] == ["draw", "clear"]


def test_icons_never_drawn_outside_idle(icons):
    # Outside IDLE the stash badges own the surface
    icons.state = ScannerState.ACTIVE
    icons._show_collections_icons(READS[GRID].cells)  # noqa: SLF001
    assert icons.badge_surface.sent == []
