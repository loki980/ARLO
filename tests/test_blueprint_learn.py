"""Tests for the hovered slot lookup and the blueprint learn verdict wiring."""

from types import SimpleNamespace
from typing import ClassVar

from PIL import Image

import arc_helper.main as main_module
from arc_helper.database import Item
from arc_helper.grid import Slot
from arc_helper.main import HoveredSlot
from arc_helper.main import Scanner
from arc_helper.main import ScannerState
from arc_helper.main import hovered_slot_on_screen
from arc_helper.tooltip_layout import Point
from arc_helper.tooltip_layout import Rect
from arc_helper.verdict import LEARN
from arc_helper.verdict import SELL

BLUEPRINT = Item(
    name="Bettina Blueprint", action="Sell", sell_price=5000, category="Blueprint"
)
FRAME = Rect(100, 200, 300, 400)


class _Db:
    items: ClassVar[dict[str, Item]] = {"Bettina Blueprint": BLUEPRINT}

    def __init__(self, owned=()):
        self.owned = set(owned)
        self.writes = []

    def lookup(self, name, tier_vote=None):  # noqa: ARG002
        return self.items.get(name)

    def recycle_outputs(self, name):  # noqa: ARG002, PLR6301
        return []

    def home_only_outputs(self, name):  # noqa: ARG002, PLR6301
        return []

    def needs_for(self, name):  # noqa: ARG002, PLR6301
        return []

    def progress_map(self):  # noqa: PLR6301
        return {}

    def blueprint_for(self, name):  # noqa: ARG002, PLR6301
        return None

    def owned_blueprints(self):
        return set(self.owned)

    def set_blueprint_owned(self, name, *, owned):
        self.writes.append((name, owned))


class _Ocr:
    last_tier_vote = None
    _last_capture = None
    _last_origin = None
    _last_card = None


def _scanner(monkeypatch, db, learned):
    """A scanner with tooltip chips off whose hovered slot reads learned."""
    settings = SimpleNamespace(
        overlay=SimpleNamespace(display_time=1.0, cooldown=60.0, tooltip_chips=False),
        show_ended_projects=False,
    )
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    monkeypatch.setattr(main_module, "get_ocr_engine", _Ocr)
    located = []

    def locate(*_args, read_check=False):
        located.append(read_check)
        return HoveredSlot(FRAME, learned)

    monkeypatch.setattr(main_module, "hovered_slot_on_screen", locate)
    scanner = Scanner(root=None, overlay=None, status=None, db=db, hotkey=None)
    scanner._running = True  # noqa: SLF001 - test seam
    scanner.state = ScannerState.ACTIVE
    shown = []
    monkeypatch.setattr(
        scanner, "_show_overlay", lambda _name, _rec, verdict: shown.append(verdict)
    )
    chips = []
    monkeypatch.setattr(scanner, "_queue_chips", lambda *_a, **k: chips.append(k))
    return scanner, located, shown, chips


def test_unchecked_blueprint_with_chips_off_learns_and_records_it(monkeypatch):
    db = _Db(owned={"Bettina Blueprint"})
    scanner, located, shown, chips = _scanner(monkeypatch, db, learned=False)
    scanner._handle_detected_item("Bettina Blueprint")  # noqa: SLF001
    assert located == [True]
    assert [v.kind for v in shown] == [LEARN]
    assert shown[0].reason == "You have not learned this blueprint yet"
    assert db.writes == [("Bettina Blueprint", False)]
    assert chips == []


def test_checked_blueprint_sells_and_records_it(monkeypatch):
    db = _Db()
    scanner, _, shown, _ = _scanner(monkeypatch, db, learned=True)
    scanner._handle_detected_item("Bettina Blueprint")  # noqa: SLF001
    assert [v.kind for v in shown] == [SELL]
    assert shown[0].reason.startswith("You already know this blueprint")
    assert db.writes == [("Bettina Blueprint", True)]


def test_unread_check_writes_nothing_and_falls_back_to_the_record(monkeypatch):
    db = _Db(owned={"bettina blueprint"})
    scanner, _, shown, _ = _scanner(monkeypatch, db, learned=None)
    scanner._handle_detected_item("Bettina Blueprint")  # noqa: SLF001
    assert [v.kind for v in shown] == [SELL]
    assert db.writes == []


def test_chips_on_get_the_frame_located_before_the_verdict(monkeypatch):
    db = _Db()
    scanner, located, _, chips = _scanner(monkeypatch, db, learned=None)
    main_module.get_settings().overlay.tooltip_chips = True
    scanner._handle_detected_item("Bettina Blueprint")  # noqa: SLF001
    assert len(located) == 1
    assert [k["frame"] for k in chips] == [FRAME]


def test_hovered_slot_frames_the_whole_wide_slot_on_screen(monkeypatch):
    ring = Rect(198, 0, 400, 200)
    wide = Slot(Rect(0, 4, 390, 196), "equipment", filled=True)
    monkeypatch.setattr(main_module, "find_hovered_cell", lambda *_a: ring)
    monkeypatch.setattr(main_module, "find_slots", lambda _rgb: [wide])
    read = []

    def check(_rgb, rect):
        read.append(rect)
        return True

    monkeypatch.setattr(main_module, "learned_check", check)
    capture = Image.new("RGB", (500, 300))
    found = hovered_slot_on_screen(
        capture, Point(1000, 500), Rect(410, 0, 500, 300), [], read_check=True
    )
    assert found == HoveredSlot(Rect(1000, 504, 1390, 696), learned=True)
    assert read == [wide.rect]
    # Without a slot under the ring box the ring box stays the frame
    monkeypatch.setattr(main_module, "find_slots", lambda _rgb: [])
    found = hovered_slot_on_screen(
        capture, Point(1000, 500), Rect(410, 0, 500, 300), [], read_check=True
    )
    assert found == HoveredSlot(Rect(1198, 500, 1400, 700), None)
