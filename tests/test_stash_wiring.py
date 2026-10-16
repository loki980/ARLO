"""Stash badges: still frames start a pass, motion clears, stale results drop."""

import logging
import threading
import time
from types import SimpleNamespace

import numpy as np

from arc_helper import main
from arc_helper.grid import STILL_MEAN
from arc_helper.main import StashBadges
from arc_helper.main import stash_still
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


class _Clock:
    """A clock for the card linger that moves only when told to."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


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
    clock = _Clock()
    stash = _stash(lambda _rgb: [_slot(SLOT_A), _slot(SLOT_B)], surface, clock=clock)
    stash.feed(_frame(0), CARD_OVER_A, ())
    stash.step()
    stash.feed(_frame(0), CARD_OVER_A, ())
    stash.step()
    assert surface.draws()[-1] == [tuple(SLOT_B)] * 2
    # The second pass waits out the card's linger, then fills slot A in
    stash.feed(_frame(0), None, ())
    stash.step()
    assert surface.draws()[-1] == [tuple(SLOT_B)] * 2
    clock.now += main.CARD_LINGER_S + 0.1
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


def _fake_slots(monkeypatch, rects, calls=None):
    """main.find_slots replaced by one that finds these rects on any frame."""

    def find(rgb):
        if calls is not None:
            calls.append(id(rgb))
        return [SimpleNamespace(rect=r) for r in rects]

    monkeypatch.setattr(main, "find_slots", find)


def test_stillness_ignores_changes_outside_the_slots(monkeypatch):
    # Synthetic frames, find_slots monkeypatched to slots A and B
    _fake_slots(monkeypatch, [SLOT_A, SLOT_B])
    a, b = _frame(0), _frame(200)
    for r in (SLOT_A, SLOT_B):
        b[r.top : r.bottom, r.left : r.right] = 0
    assert stash_still(a, b, [])


def test_a_changed_slot_interior_is_movement(monkeypatch):
    _fake_slots(monkeypatch, [SLOT_A, SLOT_B])
    a, b = _frame(0), _frame(0)
    # Slot A is 4 of the 5 compared 8 px cells (B is unaligned, so only its
    # one wholly inside cell counts), so the mean is 4 * 4 / 5 = 3.2
    b[SLOT_A.top : SLOT_A.bottom, SLOT_A.left : SLOT_A.right] = 4
    assert STILL_MEAN <= 4 * 4 / 5
    assert not stash_still(a, b, [])


def test_without_slots_a_change_anywhere_is_movement(monkeypatch):
    _fake_slots(monkeypatch, [])
    a, b = _frame(0), _frame(0)
    b[40:64, 0:24] = 200
    assert not stash_still(a, b, [])


def test_wholly_ignored_slots_are_still(monkeypatch):
    _fake_slots(monkeypatch, [SLOT_A, SLOT_B])
    assert stash_still(_frame(0), _frame(200), [SLOT_A, SLOT_B])


def test_find_slots_runs_once_per_held_frame(monkeypatch):
    calls = []
    _fake_slots(monkeypatch, [SLOT_A], calls)
    stash = _stash(lambda _rgb: [_slot(SLOT_A)], _Surface())
    # Four still frames, then two of a moved grid, each a new array
    frames = [_frame(0) for _ in range(4)] + [_frame(120), _frame(120)]
    for i, rgb in enumerate(frames):
        stash.feed(rgb, None, ())
        stash.step()
        # Cached exactly the frames _prev, _newest and _job refer to, once
        # the first compare has run (the first frame alone is not looked at)
        held = {id(r) for r in stash._held()}  # noqa: SLF001
        assert set(stash._slot_cache) == (held if i else set())  # noqa: SLF001
    # Each frame once: the first as the frame before, the rest as the newest
    assert sorted(calls) == sorted(id(f) for f in frames)
    stash.clear("test")
    assert not stash._slot_cache  # noqa: SLF001


def test_cache_drops_replaced_newest_frames(monkeypatch):
    _fake_slots(monkeypatch, [SLOT_A])
    stash = _stash(lambda _rgb: [_slot(SLOT_A)], _Surface())
    frames = [_frame(0) for _ in range(5)]
    for rgb in frames:
        stash.feed(rgb, None, ())
        stash.step()
    # The last frame is both _prev and _newest, the pass frame (the second)
    # is _ref, the others are gone
    assert set(stash._slot_cache) == {id(frames[1]), id(frames[-1])}  # noqa: SLF001


def test_two_argument_still_works_with_slots(monkeypatch):
    _fake_slots(monkeypatch, [SLOT_A])
    seen = []

    def still(a, b):
        seen.append((a, b))
        return True

    assert stash_still(_frame(0), _frame(200), [], still)
    assert stash_still(_frame(0), _frame(200), [SLOT_B], still)
    surface = _Surface()
    stash = _stash(lambda _rgb: [_slot(SLOT_A)], surface, still=still)
    for _ in range(2):
        stash.feed(_frame(0), None, ())
        stash.step()
    assert len(seen) == 3
    assert surface.draws()


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


# A 8 by 5 grid of 48 px slots at a 56 px pitch, so a slot is 6 by 6 still
# cells and its middle stays compared outside the drawn badges' marks
GRID_SLOTS = [
    Rect(56 * c, 56 * r, 56 * c + 48, 56 * r + 48) for r in range(5) for c in range(8)
]
GRID_SHAPE = (280, 448, 3)


def _grid(changed=(), shade=40) -> np.ndarray:
    """The grid frame with the slots at these indexes brightened by shade."""
    rgb = np.zeros(GRID_SHAPE, np.uint8)
    for i in changed:
        r = GRID_SLOTS[i]
        rgb[r.top : r.bottom, r.left : r.right] = shade
    return rgb


def _grid_stash(monkeypatch, surface, identify=None, slots=None, **kwargs):
    """A stash with badges drawn on all 40 slots of the grid frame."""

    def find(rgb):
        rects = slots(rgb) if slots else GRID_SLOTS
        return [SimpleNamespace(rect=r) for r in rects]

    monkeypatch.setattr(main, "find_slots", find)
    calls = []

    def default(rgb):
        calls.append(rgb)
        return [_slot(r) for r in GRID_SLOTS]

    stash = _stash(identify or default, surface, **kwargs)
    for _ in range(2):
        stash.feed(_grid(), None, ())
        stash.step()
    assert len(surface.draws()[-1]) == 80
    return stash, calls


def _drawn_slots(surface) -> set:
    return set(surface.draws()[-1])


def test_a_fading_card_hides_nothing(monkeypatch, caplog):
    # The card over slots 0 to 2 and 8 to 10, then gone while its fading
    # shadow still darkens those 6 slots: the linger keeps every badge
    surface, clock = _Surface(), _Clock()
    stash, calls = _grid_stash(monkeypatch, surface, clock=clock)
    card = Rect(0, 0, 160, 104)
    fading = (0, 1, 2, 8, 9, 10)
    with caplog.at_level(logging.DEBUG, logger="arc_helper"):
        stash.feed(_grid(), card, ())
        stash.step()
        for _ in range(3):
            clock.now += 0.1
            stash.feed(_grid(fading), None, ())
            stash.step()
    assert len(surface.draws()[-1]) == 80
    assert ("clear",) not in surface.calls
    assert "hidden" not in caplog.text
    assert len(calls) == 1


def test_a_hovered_slot_hides_only_its_badges_until_it_reverts(monkeypatch, caplog):
    surface = _Surface()
    stash, calls = _grid_stash(monkeypatch, surface)
    with caplog.at_level(logging.DEBUG, logger="arc_helper"):
        stash.feed(_grid([5]), None, ())
        stash.step()
    assert _drawn_slots(surface) == {tuple(r) for r in GRID_SLOTS} - {
        tuple(GRID_SLOTS[5])
    }
    assert len(surface.draws()[-1]) == 78
    assert "stash badges hidden slots=1 reason=slot_changed" in caplog.text
    stash.feed(_grid(), None, ())
    stash.step()
    assert len(surface.draws()[-1]) == 80
    assert len(calls) == 1


def test_half_the_slots_changing_clears_all(monkeypatch, caplog):
    surface = _Surface()
    stash, _calls = _grid_stash(monkeypatch, surface)
    with caplog.at_level(logging.DEBUG, logger="arc_helper"):
        stash.feed(_grid(range(20)), None, ())
        stash.step()
    assert surface.calls[-1] == ("clear",)
    assert "stash badges cleared reason=grid_moved" in caplog.text


def test_a_scroll_clears_all(monkeypatch, caplog):
    # The scrolled frame's slots sit half a pitch lower, none match
    scrolled = _grid()
    shifted = [Rect(r.left, r.top + 28, r.right, r.bottom + 28) for r in GRID_SLOTS]
    surface = _Surface()
    stash, _calls = _grid_stash(
        monkeypatch,
        surface,
        slots=lambda rgb: shifted if rgb is scrolled else GRID_SLOTS,
    )
    with caplog.at_level(logging.DEBUG, logger="arc_helper"):
        stash.feed(scrolled, None, ())
        stash.step()
    assert surface.calls[-1] == ("clear",)
    assert "stash badges cleared reason=grid_moved" in caplog.text


def test_a_stale_pass_keeps_the_unchanged_slots(monkeypatch, caplog):
    # Slot 7 changes while the pass runs: its badges are dropped, the rest kept
    surface = _Surface()
    stash = None

    def identify(_rgb):
        stash.feed(_grid([7]), None, ())
        stash._watch_once()  # noqa: SLF001
        return [_slot(r) for r in GRID_SLOTS]

    def find(_rgb):
        return [SimpleNamespace(rect=r) for r in GRID_SLOTS]

    monkeypatch.setattr(main, "find_slots", find)
    stash = _stash(identify, surface)
    stash.feed(_grid(), None, ())
    stash.step()
    with caplog.at_level(logging.DEBUG, logger="arc_helper"):
        stash.feed(_grid(), None, ())
        stash.step()
    assert "stash pass partial kept=78 dropped=2" in caplog.text
    assert _drawn_slots(surface) == {tuple(r) for r in GRID_SLOTS} - {
        tuple(GRID_SLOTS[7])
    }


def _one_slot_off(monkeypatch, caplog, before, after):
    """Feed a frame whose find_slots finds after, with badges from before."""
    changed = _grid()
    surface = _Surface()
    stash, _calls = _grid_stash(
        monkeypatch,
        surface,
        slots=lambda rgb: after if rgb is changed else before,
    )
    with caplog.at_level(logging.DEBUG, logger="arc_helper"):
        stash.feed(changed, None, ())
        stash.step()
    assert surface.calls[-1] == ("clear",)
    assert "stash badges cleared reason=grid_moved" in caplog.text


def test_one_slot_removed_clears_all(monkeypatch, caplog):
    _one_slot_off(monkeypatch, caplog, GRID_SLOTS, GRID_SLOTS[:-1])


def test_one_slot_added_clears_all(monkeypatch, caplog):
    # The last slot found on the newer frame only
    _one_slot_off(monkeypatch, caplog, GRID_SLOTS[:-1], GRID_SLOTS)


def test_a_slot_missed_because_its_look_changed_only_hides(monkeypatch):
    # find_slots loses slot 3 once it is darkened: a changed slot, not a new set
    dark = _grid([3])
    surface = _Surface()
    stash, _calls = _grid_stash(
        monkeypatch,
        surface,
        slots=lambda rgb: GRID_SLOTS[:3] + GRID_SLOTS[4:]
        if rgb is dark
        else GRID_SLOTS,
    )
    stash.feed(dark, None, ())
    stash.step()
    assert ("clear",) not in surface.calls
    assert len(surface.draws()[-1]) == 78


def test_a_retry_restores_badges_a_partial_pass_dropped(monkeypatch):
    # Slot 7 changes during the first pass and reverts before the retry
    surface = _Surface()
    stash = None
    passes = []

    def identify(_rgb):
        passes.append(1)
        if len(passes) == 1:
            stash.feed(_grid([7]), None, ())
            stash._watch_once()  # noqa: SLF001
        return [_slot(r) for r in GRID_SLOTS]

    def find(_rgb):
        return [SimpleNamespace(rect=r) for r in GRID_SLOTS]

    monkeypatch.setattr(main, "find_slots", find)
    stash = _stash(identify, surface)
    for _ in range(2):
        stash.feed(_grid(), None, ())
        stash.step()
    assert len(surface.draws()[-1]) == 78
    for _ in range(3):
        stash.feed(_grid(), None, ())
        stash.step()
    assert len(passes) == 2
    assert len(surface.draws()[-1]) == 80
