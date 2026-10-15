"""The scan loop acts on card changes: show at once, hide at once, no re-reads."""

import logging
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from PIL import Image

import arc_helper.main as main_module
from arc_helper.database import Item
from arc_helper.main import NO_CARD_SLEEP_CAP_S
from arc_helper.main import TRIGGER_RECHECK_S
from arc_helper.main import Application
from arc_helper.main import Scanner
from arc_helper.main import ScannerState
from arc_helper.ocr import CardSignature
from arc_helper.ocr import card_signature
from arc_helper.ocr import same_card
from arc_helper.tooltip_layout import Point
from arc_helper.tooltip_layout import Rect
from arc_helper.tooltip_layout import TooltipLayout
from arc_helper.verdict import Verdict


def _sig(rect=(100, 100, 500, 700), shade=200):
    return CardSignature(Rect(*rect), np.full((8, 32), shade, np.uint8))


# Two sightings of one card, a few px and grey levels apart, and two others
CARD_A = _sig()
CARD_A_AGAIN = _sig((102, 99, 501, 702), 205)
CARD_B = _sig((600, 100, 1000, 700))
CARD_A_OTHER_TITLE = _sig(shade=120)


class _Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now


class _Ocr:
    """Each cycle sees the next card of a sequence (None for no card).

    A search takes search_s and a read read_s on the fake clock. Reads
    return the name of the card last located.
    """

    last_tier_vote = None

    def __init__(self, clock, cards, *, search_s=0.15, read_s=0.2, trigger=True):
        self.clock = clock
        self.cards = list(cards)
        self.search_s = search_s
        self.read_s = read_s
        self.trigger = trigger
        self.reads = 0
        self.trigger_checks = []
        self.located = None

    def locate_card_at_cursor(self):
        self.clock.now += self.search_s
        self.located = self.cards.pop(0)
        return self.located

    def read_located_card(self):
        self.clock.now += self.read_s
        self.reads += 1
        return "Battery" if self.located is not CARD_B else "Anvil I"

    def check_trigger_any(self, regions):  # noqa: ARG002
        self.trigger_checks.append(self.clock.now)
        return self.trigger

    @staticmethod
    def layout_from_last_capture():
        return TooltipLayout(Rect(0, 0, 600, 900), {})


class _Db:
    items: ClassVar[dict[str, Item]] = {
        "Battery": Item(name="Battery", action="Keep"),
        "Anvil I": Item(name="Anvil I", action="Keep"),
    }

    def lookup(self, name, tier_vote=None):  # noqa: ARG002
        return self.items.get(name)

    def recycle_outputs(self, name):  # noqa: ARG002, PLR6301
        return []

    def home_only_outputs(self, name):  # noqa: ARG002, PLR6301
        return []


class _Hotkey:
    """Hotkey mode (held) or inventory mode, for as many cycles as cards."""

    def __init__(self, ocr, scanner_ref, *, held):
        self.ocr = ocr
        self.scanner_ref = scanner_ref
        self.held = held

    def is_held(self):
        if not self.ocr.cards:
            self.scanner_ref[0]._running = False  # noqa: SLF001 - test seam
            return False
        return self.held

    def stop(self):
        pass


def _settings(interval=0.3):
    return SimpleNamespace(
        scan=SimpleNamespace(tooltip_scan_interval=interval, trigger_scan_interval=0.5),
        overlay=SimpleNamespace(display_time=1.0, cooldown=60.0, tooltip_chips=True),
        show_ended_projects=False,
        trigger_region=None,
        trigger_region2=None,
    )


def _run(monkeypatch, cards, *, held=True, interval=0.3, **ocr_kwargs):
    """Run _scan_loop over a card sequence, returning scanner, ocr, sleeps."""
    clock = _Clock()
    ocr = _Ocr(clock, cards, **ocr_kwargs)
    ref = []
    scanner = Scanner(root=None, overlay=None, status=None, db=_Db(), hotkey=None)
    scanner.hotkey = _Hotkey(ocr, ref, held=held)
    ref.append(scanner)
    scanner._running = True  # noqa: SLF001 - test seam
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(main_module, "get_settings", lambda: _settings(interval))
    monkeypatch.setattr(main_module, "get_ocr_engine", lambda: ocr)
    monkeypatch.setattr(main_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(main_module.time, "sleep", sleep)
    monkeypatch.setattr(scanner, "_verdict_for", lambda *_: Verdict("sell", "r"))
    scanner._scan_loop()  # noqa: SLF001
    return scanner, ocr, sleeps


def _kinds(scanner, *, skip=("status",)):
    messages = []
    while not scanner.ui_queue.empty():
        message = scanner.ui_queue.get_nowait()
        if message[0] not in skip:
            messages.append(message[0])
    return messages


def test_same_card_twice_reads_once(monkeypatch):
    scanner, ocr, _ = _run(monkeypatch, [CARD_A, CARD_A_AGAIN, CARD_A])
    assert ocr.reads == 1
    # Shown once with its chips, then only touched (at most once a second)
    kinds = _kinds(scanner)
    assert kinds[:2] == ["overlay", "chips"]
    assert set(kinds[2:]) <= {"touch", "chips_hide"}


def test_card_gone_twice_queues_one_hide(monkeypatch):
    scanner, _, _ = _run(monkeypatch, [CARD_A, None, None, None, None])
    kinds = _kinds(scanner)
    assert kinds.count("hide") == 1
    assert kinds[:3] == ["overlay", "chips", "hide"]


def test_one_missed_cycle_keeps_the_card_without_a_read(monkeypatch):
    scanner, ocr, _ = _run(monkeypatch, [CARD_A, None, CARD_A])
    assert ocr.reads == 1
    assert "hide" not in _kinds(scanner)


def test_same_item_after_a_hide_shows_again(monkeypatch):
    scanner, ocr, _ = _run(monkeypatch, [CARD_A, None, None, CARD_A])
    assert ocr.reads == 2
    assert _kinds(scanner)[:5] == ["overlay", "chips", "hide", "overlay", "chips"]


def test_new_card_is_read_at_once(monkeypatch):
    scanner, ocr, sleeps = _run(monkeypatch, [CARD_A, CARD_B, CARD_A_OTHER_TITLE])
    assert ocr.reads == 3
    assert _kinds(scanner).count("overlay") == 3
    # Cycles that found a card never sleep
    assert sleeps == []


def test_no_card_sleep_is_bounded(monkeypatch):
    _, _, sleeps = _run(monkeypatch, [None, None], interval=0.3)
    assert sleeps == [NO_CARD_SLEEP_CAP_S, NO_CARD_SLEEP_CAP_S]
    _, _, sleeps = _run(monkeypatch, [None, None], interval=0.05)
    assert sleeps == [0.05, 0.05]


def test_trigger_recheck_is_time_based(monkeypatch):
    # Inventory mode: the trigger opens ACTIVE, then 20 cycles of 0.15 s
    cards = [CARD_A] * 20
    _, ocr, _ = _run(monkeypatch, cards, held=False, read_s=0.0)
    entry, *rechecks = ocr.trigger_checks
    # Every third scan would be 6 rechecks, the clock gives one a second
    assert len(rechecks) == 2
    gaps = np.diff([entry, *rechecks])
    assert all(gap >= TRIGGER_RECHECK_S for gap in gaps)


class _ClosingOcr(_Ocr):
    """INVENTORY is seen on entry, then gone at the first recheck."""

    def check_trigger_any(self, regions):
        found = not self.trigger_checks
        super().check_trigger_any(regions)
        if not found:
            self.cards.clear()
        return found


def test_trigger_lost_hides_the_card_windows(monkeypatch):
    clock = _Clock()
    ocr = _ClosingOcr(clock, [CARD_A] * 20, read_s=0.0)
    monkeypatch.setattr(main_module, "get_ocr_engine", lambda: ocr)
    ref = []
    scanner = Scanner(root=None, overlay=None, status=None, db=_Db(), hotkey=None)
    scanner.hotkey = _Hotkey(ocr, ref, held=False)
    ref.append(scanner)
    scanner._running = True  # noqa: SLF001 - test seam
    monkeypatch.setattr(main_module, "get_settings", _settings)
    monkeypatch.setattr(main_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(main_module.time, "sleep", lambda _: None)
    monkeypatch.setattr(scanner, "_verdict_for", lambda *_: Verdict("sell", "r"))
    scanner._scan_loop()  # noqa: SLF001

    assert scanner.state == ScannerState.IDLE
    kinds = _kinds(scanner)
    assert kinds[:2] == ["overlay", "chips"]
    assert kinds[-1] == "hide"


def test_debug_log_names_state_changes_only(monkeypatch, caplog):
    cards = [CARD_A, CARD_A, CARD_A, CARD_A, None, None, None, None]
    with caplog.at_level(logging.DEBUG, logger=main_module.logger.name):
        _run(monkeypatch, cards)
    states = [
        r.getMessage().split()[1]
        for r in caplog.records
        if r.getMessage().startswith("card ")
    ]
    assert states == ["new", "same", "gone", "hide"]


def test_scanner_state_after_the_run(monkeypatch):
    scanner, _, _ = _run(monkeypatch, [CARD_A])
    assert scanner.state == ScannerState.IDLE


def test_same_card_tolerates_small_shifts_and_noise():
    assert same_card(CARD_A, CARD_A_AGAIN)
    assert not same_card(CARD_A, CARD_B)
    assert not same_card(CARD_A, CARD_A_OTHER_TITLE)
    # Five px off on one edge is another card
    assert not same_card(CARD_A, _sig((105, 100, 500, 700)))


def test_card_signature_reads_the_top_band_in_screen_px():
    image = Image.new("RGB", (400, 400), "white")
    card = Rect(50, 50, 250, 350)
    # A dark title band at the top of the card, a dark mark lower down
    image.paste((20, 20, 20), (60, 60, 240, 100))
    first = card_signature(image, card, Point(1000, 500))
    assert first.rect == Rect(1050, 550, 1250, 850)
    assert first.thumb.shape == (8, 32)

    lower = image.copy()
    lower.paste((20, 20, 20), (60, 300, 240, 340))
    assert same_card(first, card_signature(lower, card, Point(1000, 500)))

    retitled = image.copy()
    retitled.paste((255, 255, 255), (60, 60, 240, 100))
    assert not same_card(first, card_signature(retitled, card, Point(1000, 500)))


class _Recorder:
    def __init__(self, *, visible=True):
        self.visible = visible
        self.calls = []

    def is_visible(self):
        return self.visible

    def hide(self, reason=None):
        self.calls.append(("hide", reason))


@pytest.mark.parametrize("visible", [True, False])
def test_hide_message_hides_popup_and_chips_at_once(visible):
    app = Application.__new__(Application)
    app.overlay = _Recorder(visible=visible)
    app.chips = _Recorder()
    app._hide_card_windows()  # noqa: SLF001
    assert app.overlay.calls == ([("hide", None)] if visible else [])
    assert app.chips.calls == [("hide", "card_gone")]
