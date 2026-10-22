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
from arc_helper.main import TRIGGER_LOST_S
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
    ocr = _Ocr(_Clock(), cards, **ocr_kwargs)
    sleeps = []
    scanner = _loop(monkeypatch, ocr, held=held, interval=interval, sleeps=sleeps)
    return scanner, ocr, sleeps


def _loop(monkeypatch, ocr, *, held, interval=0.3, sleeps=None):
    """Run _scan_loop with an engine stub, returning the scanner."""
    clock = ocr.clock
    sleeps = [] if sleeps is None else sleeps
    ref = []
    scanner = Scanner(root=None, overlay=None, status=None, db=_Db(), hotkey=None)
    scanner.hotkey = _Hotkey(ocr, ref, held=held)
    ref.append(scanner)
    scanner._running = True  # noqa: SLF001 - test seam

    def sleep(seconds):
        sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(main_module, "get_settings", lambda: _settings(interval))
    monkeypatch.setattr(main_module, "get_ocr_engine", lambda: ocr)
    monkeypatch.setattr(main_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(main_module.time, "sleep", sleep)
    monkeypatch.setattr(scanner, "_verdict_for", lambda *_: Verdict("sell", "r"))
    scanner._scan_loop()  # noqa: SLF001
    return scanner


def _kinds(scanner, *, skip=("status",)):
    messages = []
    while not scanner.ui_queue.empty():
        message = scanner.ui_queue.get_nowait()
        if message[0] not in skip:
            messages.append(message[0])
    return messages


def test_same_card_twice_reads_once(monkeypatch):
    scanner, ocr, _ = _run(monkeypatch, [CARD_A, CARD_A_AGAIN, CARD_A], held=False)
    assert ocr.reads == 1
    # Shown once with its chips, then only touched (at most once a second)
    kinds = _kinds(scanner)
    assert kinds[:2] == ["overlay", "chips"]
    assert set(kinds[2:]) <= {"touch"}


def test_card_gone_twice_queues_one_hide(monkeypatch):
    scanner, _, _ = _run(monkeypatch, [CARD_A, None, None, None, None], held=False)
    kinds = _kinds(scanner)
    assert kinds.count("hide") == 1
    assert kinds[:3] == ["overlay", "chips", "hide"]


def test_one_missed_cycle_keeps_the_card_without_a_read(monkeypatch):
    scanner, ocr, sleeps = _run(monkeypatch, [CARD_A, None, CARD_A], held=False)
    assert ocr.reads == 1
    assert "hide" not in _kinds(scanner)
    # The first miss of a shown card looks again at once
    assert NO_CARD_SLEEP_CAP_S not in sleeps


class _TimedOcr(_Ocr):
    """_Ocr that keeps the fake time each locate (its capture) started."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.starts = []

    def locate_card_at_cursor(self):
        self.starts.append(self.clock.now)
        return super().locate_card_at_cursor()


@pytest.mark.parametrize("locate_s", [0.077, 0.124])
def test_card_gone_hides_two_locates_later_without_a_sleep(monkeypatch, locate_s):
    # 77 and 124 ms are the median and worst locate of the 21 Oct live log.
    # The card goes between the shown cycle and the next capture
    ocr = _TimedOcr(_Clock(), [CARD_A, None, None, None], search_s=locate_s)
    hides = []
    hide = Scanner._hide_card_windows  # noqa: SLF001

    def timed_hide(self):
        hides.append(ocr.clock.now)
        hide(self)

    monkeypatch.setattr(Scanner, "_hide_card_windows", timed_hide)
    sleeps = []
    _loop(monkeypatch, ocr, held=False, sleeps=sleeps)
    gone_at = ocr.starts[1]
    assert hides[0] - gone_at == pytest.approx(2 * locate_s)
    # Of the three cycles without a card, the first miss does not sleep
    assert sleeps.count(NO_CARD_SLEEP_CAP_S) == 2


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


def test_failed_read_of_a_new_card_hides_the_old_windows(monkeypatch):
    clock = _Clock()

    class _FailOnB(_Ocr):
        def read_located_card(self):
            name = super().read_located_card()
            return None if self.located is CARD_B else name

    ocr = _FailOnB(clock, [CARD_A, CARD_B, CARD_B, CARD_A])
    scanner = _loop(monkeypatch, ocr, held=False)
    # Hidden once at the failed read, read again next cycle, then CARD_A
    # shows again
    assert ocr.reads == 4
    assert _kinds(scanner) == ["overlay", "chips", "hide", "overlay", "chips"]


def test_hotkey_release_hides_the_card_windows(monkeypatch):
    # In hotkey mode the run ends with the hotkey released
    scanner, _, _ = _run(monkeypatch, [CARD_A, CARD_A])
    assert _kinds(scanner)[-1] == "hide"


@pytest.mark.parametrize("leave", ["pause", "stop"])
def test_pause_and_stop_hide_the_card_windows(leave):
    scanner = Scanner(root=None, overlay=None, status=None, db=_Db(), hotkey=None)
    scanner.hotkey = SimpleNamespace(stop=lambda: None)
    scanner.state = ScannerState.ACTIVE
    scanner._card_shown = CARD_A  # noqa: SLF001
    getattr(scanner, leave)()
    assert _kinds(scanner) == ["hide"]
    assert scanner._card_shown is None  # noqa: SLF001


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
    """INVENTORY is seen on entry, then gone at every recheck.

    The run ends at the first check made back in IDLE.
    """

    scanner = None

    def check_trigger_any(self, regions):
        found = not self.trigger_checks
        super().check_trigger_any(regions)
        if self.scanner.state == ScannerState.IDLE and not found:
            self.cards.clear()
        return found


def test_trigger_lost_hides_the_card_windows(monkeypatch, caplog):
    clock = _Clock()
    ocr = _ClosingOcr(clock, [CARD_A] * 40, read_s=0.0)
    monkeypatch.setattr(main_module, "get_ocr_engine", lambda: ocr)
    ref = []
    scanner = Scanner(root=None, overlay=None, status=None, db=_Db(), hotkey=None)
    ocr.scanner = scanner
    scanner.hotkey = _Hotkey(ocr, ref, held=False)
    ref.append(scanner)
    scanner._running = True  # noqa: SLF001 - test seam
    monkeypatch.setattr(main_module, "get_settings", _settings)
    monkeypatch.setattr(main_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(main_module.time, "sleep", lambda _: None)
    monkeypatch.setattr(scanner, "_verdict_for", lambda *_: Verdict("sell", "r"))
    with caplog.at_level(logging.DEBUG, logger=main_module.logger.name):
        scanner._scan_loop()  # noqa: SLF001

    assert scanner.state == ScannerState.IDLE
    kinds = _kinds(scanner)
    assert kinds[:2] == ["overlay", "chips"]
    assert kinds[-1] == "hide"
    # Rechecks about 1.05 s apart: the third miss spans 2.1 s and leaves
    leave = [
        r.getMessage()
        for r in caplog.records
        if "reason=trigger_lost" in r.getMessage()
    ]
    assert leave == ["state ACTIVE -> IDLE reason=trigger_lost misses=3 span=2100 ms"]


class _TriggerOcr:
    """Answers each trigger check from a list, and records what it read."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.last_trigger_reads = ()

    def check_trigger_any(self, regions):  # noqa: ARG002
        found = self.answers.pop(0)
        self.last_trigger_reads = (("region1", ""), ("region2", "INVENT"))
        if not found:
            self.last_trigger_reads += (("band", "MAP LOGBOOK"),)
        return found


def _rechecks(monkeypatch, answers, times):
    """Enter ACTIVE at t=100, then call _trigger_lost at each time.

    Returns the scanner and what each call answered.
    """
    clock = _Clock()
    monkeypatch.setattr(main_module.time, "monotonic", clock.monotonic)
    scanner = Scanner(root=None, overlay=None, status=None, db=_Db(), hotkey=None)
    scanner._reset_scan_timing()  # noqa: SLF001
    ocr = _TriggerOcr(answers)
    settings = SimpleNamespace(trigger_region=None, trigger_region2=None)
    lost = []
    for t in times:
        clock.now = t
        lost.append(scanner._trigger_lost(ocr, settings))  # noqa: SLF001
    return scanner, lost


def test_one_trigger_miss_stays_active(monkeypatch):
    _, lost = _rechecks(monkeypatch, [False, True], [101.0, 102.0])
    assert lost == [False, False]


def test_trigger_misses_spanning_lost_s_on_two_rechecks_leave(monkeypatch):
    # A slow cycle puts the two rechecks TRIGGER_LOST_S apart
    _, lost = _rechecks(monkeypatch, [False, False], [101.0, 101.0 + TRIGGER_LOST_S])
    assert lost == [False, True]
    # Two misses a second apart are not enough yet, the third one is
    _, lost = _rechecks(monkeypatch, [False] * 3, [101.0, 102.0, 103.0])
    assert lost == [False, False, True]


def test_trigger_hit_between_misses_restarts_the_span(monkeypatch):
    answers = [False, True, False, False]
    _, lost = _rechecks(monkeypatch, answers, [101.0, 102.0, 103.0, 104.0])
    # The misses at 101 and 103 would span 2 s, but the hit at 102 broke them
    assert lost == [False, False, False, False]


def test_reentry_starts_a_fresh_trigger_miss_span(monkeypatch):
    scanner, lost = _rechecks(monkeypatch, [False, False], [101.0, 102.0])
    assert lost == [False, False]
    # Back in ACTIVE (say the hotkey) at 102.5: the old misses are forgotten
    monkeypatch.setattr(main_module.time, "monotonic", lambda: 102.5)
    scanner._reset_scan_timing()  # noqa: SLF001
    assert scanner._trigger_misses == 0  # noqa: SLF001
    ocr = _TriggerOcr([False])
    settings = SimpleNamespace(trigger_region=None, trigger_region2=None)
    monkeypatch.setattr(main_module.time, "monotonic", lambda: 103.5)
    assert not scanner._trigger_lost(ocr, settings)  # noqa: SLF001
    assert scanner._trigger_miss_since == 103.5  # noqa: SLF001


def test_trigger_miss_logs_what_each_place_read(monkeypatch, caplog):
    with caplog.at_level(logging.DEBUG, logger=main_module.logger.name):
        _rechecks(monkeypatch, [False, False], [101.0, 102.0])
    misses = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("trigger miss")
    ]
    reads = "region1=none region2='INVENT' band='MAP LOGBOOK'"
    assert misses == [
        f"trigger miss {reads} check=0 ms misses=1 span=0 ms",
        f"trigger miss {reads} check=0 ms misses=2 span=1000 ms",
    ]


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


GRID_FRAME_SLOTS = [Rect(0, 0, 16, 16), Rect(44, 44, 60, 60)]


class _GridClosingOcr(_ClosingOcr):
    """_ClosingOcr whose full-screen captures show the grid for grid_cycles.

    Every capture after those has no grid: the inventory closed.
    """

    search_width = 0
    search_height = 0
    _last_card = None

    def __init__(self, clock, cards, grid_cycles, **kwargs):
        super().__init__(clock, cards, **kwargs)
        self.grid_cycles = grid_cycles
        self.grid = np.zeros((64, 64, 3), np.uint8)
        self._last_origin = Point(0, 0)
        self._last_capture = None
        self.gone_at = None

    def locate_card_at_cursor(self):
        found = super().locate_card_at_cursor()
        self.grid_cycles -= 1
        if self.grid_cycles >= 0:
            self._last_capture = self.grid
        else:
            self._last_capture = np.zeros((64, 64, 3), np.uint8)
            if self.gone_at is None:
                self.gone_at = self.clock.now
        return found


def test_the_grid_gone_clears_at_once_and_the_state_waits_for_the_debounce(
    monkeypatch, caplog
):
    clock = _Clock()
    ocr = _GridClosingOcr(clock, [None] * 40, 4)
    monkeypatch.setattr(main_module, "get_ocr_engine", lambda: ocr)
    monkeypatch.setattr(
        main_module,
        "find_slots",
        lambda rgb: [SimpleNamespace(rect=r) for r in GRID_FRAME_SLOTS]
        if rgb is ocr.grid
        else [],
    )
    monkeypatch.setattr(main_module, "get_screen_resolution", lambda: (1, 1))

    def no_backend():
        raise RuntimeError

    monkeypatch.setattr(main_module, "get_backend", no_backend)
    ref = []
    scanner = Scanner(root=None, overlay=None, status=None, db=_Db(), hotkey=None)
    clears = []

    class _Surface:
        @staticmethod
        def start():
            return True

        @staticmethod
        def draw(_badges):
            pass

        @staticmethod
        def clear():
            clears.append((scanner.state, clock.now))

    def badges(slot_items):
        return [{"rect": tuple(s.rect), "kind": "border"} for s in slot_items]

    stash = main_module.StashBadges(
        lambda _rgb: [
            SimpleNamespace(rect=r, item=None, tier=None) for r in GRID_FRAME_SLOTS
        ],
        badges,
        _Surface(),
        threaded=False,
        gap_s=0,
        clock=clock.monotonic,
    )
    feed = stash.feed

    def feed_and_step(*args):
        feed(*args)
        stash.step()

    stash.feed = feed_and_step
    scanner.stash = stash
    ocr.scanner = scanner
    scanner.hotkey = _Hotkey(ocr, ref, held=False)
    ref.append(scanner)
    scanner._running = True  # noqa: SLF001 - test seam
    monkeypatch.setattr(main_module, "get_settings", _settings)
    monkeypatch.setattr(main_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(main_module.time, "sleep", lambda _: None)
    with caplog.at_level(logging.DEBUG, logger=main_module.logger.name):
        scanner._scan_loop()  # noqa: SLF001

    # Cleared on the first cycle without a grid, while still ACTIVE
    assert clears == [(ScannerState.ACTIVE, ocr.gone_at)]
    assert "stash badges cleared reason=grid_gone" in caplog.text
    leave = [
        r.getMessage()
        for r in caplog.records
        if "reason=trigger_lost" in r.getMessage()
    ]
    assert leave == ["state ACTIVE -> IDLE reason=trigger_lost misses=3 span=2100 ms"]
    assert scanner.state == ScannerState.IDLE
