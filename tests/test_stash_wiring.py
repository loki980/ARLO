"""Stash badges: still frames start a pass, motion clears, stale results drop."""

import logging
import threading
import time
from types import SimpleNamespace

import numpy as np

from arc_helper.main import StashBadges
from arc_helper.stash_scan import SlotItem
from arc_helper.tooltip_layout import Point
from arc_helper.tooltip_layout import Rect

SLOT_A = Rect(0, 0, 16, 16)
SLOT_B = Rect(44, 44, 60, 60)
# The game's item card over slot A only (with its margin it ends at 36)
CARD_OVER_A = Rect(0, 0, 20, 20)
ORIGIN = Point(0, 0)


def _frame(shade: int) -> np.ndarray:
    return np.full((64, 64, 3), shade, np.uint8)


class _Surface:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def start(self) -> bool:
        self.calls.append(("start",))
        return True

    def draw(self, badges) -> None:
        self.calls.append(("draw", sorted(tuple(b.rect) for b in badges)))

    def clear(self) -> None:
        self.calls.append(("clear",))

    def stop(self) -> None:
        self.calls.append(("stop",))

    def draws(self) -> list:
        return [c[1] for c in self.calls if c[0] == "draw"]


def _slot(rect: Rect) -> SlotItem:
    item = SimpleNamespace(name="x", sell_price=100)
    return SlotItem(rect, "stash", item, None, None, 0.9, 0.2, "x")


def _badges(slot_items) -> list[dict]:
    """A border and a tag per slot, as badges_for makes them."""
    made = []
    for s in slot_items:
        rect = (s.rect.left, s.rect.top, s.rect.right, s.rect.bottom)
        made.extend(
            (
                {"rect": rect, "kind": "border"},
                {"rect": rect, "text": "100", "corner": "bl", "kind": "tag"},
            )
        )
    return made


def _stash(identify, surface, **kwargs) -> StashBadges:
    return StashBadges(identify, _badges, surface, threaded=False, gap_s=0, **kwargs)


def test_still_frames_start_one_pass_and_draw():
    calls = []

    def identify(rgb):
        calls.append(rgb)
        return [_slot(SLOT_A), _slot(SLOT_B)]

    surface = _Surface()
    stash = _stash(identify, surface)
    for _ in range(4):
        stash.feed(_frame(0), None, ())
        stash.step()
    assert len(calls) == 1
    assert surface.calls[0] == ("start",)
    assert surface.draws() == [[tuple(SLOT_A)] * 2 + [tuple(SLOT_B)] * 2]


def test_moving_frames_clear_at_once(caplog):
    surface = _Surface()
    stash = _stash(lambda _rgb: [_slot(SLOT_A)], surface)
    stash.feed(_frame(0), None, ())
    stash.step()
    stash.feed(_frame(0), None, ())
    stash.step()
    assert surface.draws()
    with caplog.at_level(logging.DEBUG, logger="arc_helper"):
        stash.feed(_frame(120), None, ())
        stash.step()
    assert surface.calls[-1] == ("clear",)
    assert "stash badges cleared reason=grid_moved" in caplog.text


def test_stale_pass_result_is_dropped(caplog):
    # Each step is still (mean diff 2 under 3) but 0 against 4 is not, so
    # the pass made on frame 0 no longer matches the newest grid
    surface = _Surface()
    stash = None

    def identify(_rgb):
        for shade in (2, 4):
            stash.feed(_frame(shade), None, ())
            stash._watch_once()  # noqa: SLF001
        return [_slot(SLOT_A)]

    stash = _stash(identify, surface)
    stash.feed(_frame(0), None, ())
    stash.step()
    with caplog.at_level(logging.DEBUG, logger="arc_helper"):
        stash.feed(_frame(0), None, ())
        stash.step()
    assert surface.draws() == []
    assert "stash pass dropped reason=stale" in caplog.text


def test_card_rect_suppresses_its_slot_until_it_goes():
    surface = _Surface()
    stash = _stash(lambda _rgb: [_slot(SLOT_A), _slot(SLOT_B)], surface)
    stash.feed(_frame(0), CARD_OVER_A, ())
    stash.step()
    stash.feed(_frame(0), CARD_OVER_A, ())
    stash.step()
    assert surface.draws()[-1] == [tuple(SLOT_B)] * 2
    stash.feed(_frame(0), None, ())
    stash.step()
    assert surface.draws()[-1] == [tuple(SLOT_A)] * 2 + [tuple(SLOT_B)] * 2
    # A card over slot A again hides its badges without a new pass
    stash.feed(_frame(0), CARD_OVER_A, ())
    stash.step()
    assert surface.draws()[-1] == [tuple(SLOT_B)] * 2


def test_keep_border_off_keeps_only_prices():
    surface = _Surface()
    stash = _stash(lambda _rgb: [_slot(SLOT_A)], surface, keep_border=False)
    for _ in range(2):
        stash.feed(_frame(0), None, ())
        stash.step()
    assert surface.draws() == [[tuple(SLOT_A)]]


def test_hover_path_never_waits_on_the_pass():
    entered, release = threading.Event(), threading.Event()

    def identify(_rgb):
        entered.set()
        release.wait(5)
        return [_slot(SLOT_A)]

    surface = _Surface()
    stash = StashBadges(identify, _badges, surface, tick_s=0.01)
    try:
        stash.feed(_frame(0), None, ())
        deadline = time.monotonic() + 5
        while not entered.is_set() and time.monotonic() < deadline:
            stash.feed(_frame(0), None, ())
            time.sleep(0.01)
        assert entered.is_set()
        start = time.perf_counter()
        for _ in range(20):
            stash.feed(_frame(0), None, ())
        assert time.perf_counter() - start < 0.05
        threads = {t.name for t in threading.enumerate()}
        assert {"stash-watch", "stash-pass"} <= threads
        assert threading.current_thread().name not in {"stash-watch", "stash-pass"}
    finally:
        release.set()
        stash.stop()
    assert surface.calls[-1] == ("stop",)


def _scanner_ocr(origin=ORIGIN):
    return SimpleNamespace(
        _last_capture=_frame(0),
        _last_origin=origin,
        _last_card=None,
        search_width=0,
        search_height=0,
    )


def test_scanner_feeds_only_full_screen_captures_outside_the_hotkey():
    from arc_helper.main import Scanner

    fed = []
    stash = SimpleNamespace(feed=lambda *a: fed.append(a), clear=lambda _reason: None)
    scanner = Scanner(root=None, overlay=None, status=None, db=None, hotkey=None)
    scanner.stash = stash
    scanner._feed_stash(_scanner_ocr())  # noqa: SLF001
    assert len(fed) == 1
    scanner._feed_stash(_scanner_ocr(Point(10, 0)))  # noqa: SLF001
    assert len(fed) == 1
    scanner._hotkey_override = True  # noqa: SLF001
    scanner._feed_stash(_scanner_ocr())  # noqa: SLF001
    assert len(fed) == 1
