"""The popup and chips hold follows the measured scan period."""

from itertools import pairwise
from types import SimpleNamespace
from typing import ClassVar

import pytest

import arc_helper.main as main_module
from arc_helper.database import Item
from arc_helper.main import HOLD_CAP_S
from arc_helper.main import Scanner
from arc_helper.main import ScannerState
from arc_helper.main import scan_hold_s
from arc_helper.tooltip_layout import Rect
from arc_helper.tooltip_layout import TooltipLayout
from arc_helper.verdict import Verdict

INTERVAL_S = 0.3


@pytest.mark.parametrize(
    ("periods", "display_time", "expected"),
    [
        ([2.465, 2.344, 2.456], 1.0, 4.93),
        ([1.122, 1.370, 1.516], 1.0, 3.032),
        ([5.0], 1.0, HOLD_CAP_S),
        ([5.0], 8.0, 8.0),
        # Only the last three periods count
        ([5.0, 1.0, 1.0, 1.0], 1.0, 2.0),
    ],
)
def test_hold_from_log_periods(periods, display_time, expected):
    assert scan_hold_s(display_time, periods, 0.1) == pytest.approx(expected)


def test_hold_before_any_period_uses_the_estimate():
    # A 1.669 s first read plus the 0.3 s interval
    assert scan_hold_s(1.0, [], 1.669 + INTERVAL_S) == pytest.approx(3.938)


def test_slow_scan_estimate_wins_over_fast_periods():
    # The first Bantam II read in the log: two fast empty scans before it
    hold = scan_hold_s(1.0, [0.621, 0.706], 1.669 + INTERVAL_S)
    assert hold == pytest.approx(3.938)


class _Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class _Hotkey:
    """Feeds is_held() from a list, then stops the scanner."""

    def __init__(self, held, scanner_ref):
        self.held = list(held)
        self.scanner_ref = scanner_ref

    def is_held(self):
        if not self.held:
            self.scanner_ref[0]._running = False  # noqa: SLF001 - test seam
            return False
        return self.held.pop(0)

    def stop(self):
        pass


class _Ocr:
    """Each read takes the next duration on the fake clock."""

    last_tier_vote = None

    def __init__(self, clock, durations, read):
        self.clock = clock
        self.durations = list(durations)
        self.read = read

    def extract_item_name_at_cursor(self):
        self.clock.now += self.durations.pop(0)
        return self.read

    @staticmethod
    def layout_from_last_capture():
        return TooltipLayout(Rect(0, 0, 600, 900), {})


class _Db:
    items: ClassVar[dict[str, Item]] = {"Battery": Item(name="Battery", action="Keep")}

    def lookup(self, name, tier_vote=None):  # noqa: ARG002
        return self.items.get(name)

    def recycle_outputs(self, name):  # noqa: ARG002, PLR6301
        return []

    def home_only_outputs(self, name):  # noqa: ARG002, PLR6301
        return []


def _settings():
    return SimpleNamespace(
        scan=SimpleNamespace(
            tooltip_scan_interval=INTERVAL_S, trigger_scan_interval=0.5
        ),
        overlay=SimpleNamespace(display_time=1.0, cooldown=60.0, tooltip_chips=True),
        show_ended_projects=False,
    )


def _run_loop(monkeypatch, held, durations, read=None):
    """Run _scan_loop on a fake clock, returning the scanner and the holds."""
    clock = _Clock()
    ref = []
    scanner = Scanner(root=None, overlay=None, status=None, db=_Db(), hotkey=None)
    scanner.hotkey = _Hotkey(held, ref)
    ref.append(scanner)
    scanner._running = True  # noqa: SLF001 - test seam
    ocr = _Ocr(clock, durations, read)
    holds = []

    def sleep(seconds):
        holds.append(scanner._hold_ms)  # noqa: SLF001
        clock.sleep(seconds)

    monkeypatch.setattr(main_module, "get_settings", _settings)
    monkeypatch.setattr(main_module, "get_ocr_engine", lambda: ocr)
    monkeypatch.setattr(main_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(main_module.time, "sleep", sleep)
    monkeypatch.setattr(scanner, "_verdict_for", lambda *_: Verdict("keep", "r"))
    scanner._scan_loop()  # noqa: SLF001
    return scanner, holds


def test_period_history_resets_on_entry_to_active(monkeypatch):
    # Enter, scan twice, release, enter again, scan once
    _, holds = _run_loop(
        monkeypatch,
        held=[True, True, True, False, True, True],
        durations=[0.5, 1.0, 0.6],
    )
    # Scan 1: estimate 0.5 + 0.3 s. Scan 2: period 0.8 s and estimate
    # 1.0 + 0.3 s. Scan 3 after re-entry: estimate 0.6 + 0.3 s alone, not
    # the 1.3 s period since scan 2 started
    assert holds == [1600, 2600, 1800]


def test_hold_reaches_queued_messages(monkeypatch):
    scanner, holds = _run_loop(
        monkeypatch,
        held=[True, True, True],
        durations=[0.5, 1.2],
        read="Battery",
    )
    messages = []
    while not scanner.ui_queue.empty():
        messages.append(scanner.ui_queue.get_nowait())
    shaped = [m for m in messages if m[0] in {"overlay", "chips", "touch"}]
    assert [m[0] for m in shaped] == ["overlay", "chips", "touch"]
    # Scan 2: estimate 1.2 + 0.3 s beats the 0.8 s period
    assert holds == [1600, 3000]
    assert [m[-1] for m in shaped] == [1600, 1600, 3000]
    assert scanner.state == ScannerState.IDLE


# jeffrey-hover-2026-10-14.log, one ACTIVE stretch from 19:37:47.029. Times
# are seconds past 19:37:00. Every scan start, and for each read of an item
# the scan end, the popup hide timer armed time and the chips hide timer
# armed time.
LOG_SCAN_STARTS = [
    47.029, 47.650, 48.356, 50.822, 53.167, 55.624, 57.933,
    59.870, 61.611, 62.734, 64.104, 65.621, 67.013, 68.341,
]  # fmt: skip
LOG_READS = [
    # (scan index, item, scan end, popup armed, chips armed)
    (2, "Bantam II", 50.026, 50.036, 50.544),
    (3, "Bantam II", 52.379, 52.429, 52.875),
    (4, "Bantam II", 54.769, 54.797, 55.249),
    (5, "Bantam II", 57.176, 57.234, 57.669),
    (6, "Burletta II", 59.199, 59.216, 59.605),
    (7, "Burletta II", 60.876, 60.930, 61.235),
    (8, "Simple Gun Parts", 62.150, 62.196, 62.458),
    (9, "Simple Gun Parts", 63.532, 63.569, 63.833),
    (10, "Simple Gun Parts", 64.927, 64.990, 65.239),
    (11, "Simple Gun Parts", 66.422, 66.468, 66.730),
    (12, "Simple Gun Parts", 67.764, 67.789, 68.047),
    (13, "Simple Gun Parts", 69.158, 69.190, 69.451),
]


def test_jeffrey_log_same_item_reads_arrive_before_hold():
    for read, after in pairwise(LOG_READS):
        index, item, scan_end, popup_armed, chips_armed = read
        if after[1] != item:
            continue
        # The periods known when this scan ended
        starts = LOG_SCAN_STARTS[: index + 1]
        periods = [b - a for a, b in pairwise(starts)]
        estimate = scan_end - starts[-1] + INTERVAL_S
        hold = scan_hold_s(1.0, periods, estimate)
        assert after[3] < popup_armed + hold, (item, scan_end, hold)
        assert after[4] < chips_armed + hold, (item, scan_end, hold)
