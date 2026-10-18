"""Stash badges: still frames start a pass, motion clears, stale results drop."""

import logging
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from arc_helper import grid
from arc_helper import main
from arc_helper import stash_scan
from arc_helper.grid import STILL_MEAN
from arc_helper.grid import Slot
from arc_helper.icon_embed import EmbedGallery
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


def test_moving_frames_clear_on_the_next_frame(caplog):
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
        assert ("clear",) not in surface.calls
        stash.feed(_frame(120), None, ())
        stash.step()
    assert surface.calls[-1] == ("clear",)
    assert "stash badges cleared reason=grid_moved changed=0 slots=0 lone=0" in (
        caplog.text
    )


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


def test_scanner_feeds_only_full_screen_captures_outside_the_hotkey(monkeypatch):
    from arc_helper.main import Scanner

    backend = SimpleNamespace(cursor_position=lambda: Point(30, 40))
    monkeypatch.setattr(main, "get_backend", lambda: backend)
    fed = []
    stash = SimpleNamespace(feed=lambda *a: fed.append(a), clear=lambda _reason: None)
    scanner = Scanner(root=None, overlay=None, status=None, db=None, hotkey=None)
    scanner.stash = stash
    scanner._feed_stash(_scanner_ocr())  # noqa: SLF001
    assert len(fed) == 1
    # The cursor goes with the capture, in capture px
    assert fed[0][3] == Point(30, 40)
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


def test_a_changed_slot_hides_only_its_badges_until_it_reverts(monkeypatch, caplog):
    surface = _Surface()
    stash, calls = _grid_stash(monkeypatch, surface)
    with caplog.at_level(logging.DEBUG, logger="arc_helper"):
        # The first changed frame waits, the next one confirms the change
        # (still changing, so no pass starts)
        stash.feed(_grid([5]), None, ())
        stash.step()
        assert len(surface.draws()[-1]) == 80
        assert "hidden" not in caplog.text
        stash.feed(_grid([5], 60), None, ())
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


# A card fading in over slots 0 to 2, 8 to 10, 16 and 17 that the scanner
# has not located yet. find_slots also loses slot 3 at its edge, which looks
# the same, so the fading frame alone is a big change
FADING = (0, 1, 2, 8, 9, 10, 16, 17)
FADING_CARD = Rect(0, 0, 104, 160)


@pytest.mark.parametrize(
    ("after", "card"),
    [
        pytest.param(False, None, id="fade_gone_next_frame"),
        pytest.param(True, FADING_CARD, id="card_located_next_frame"),
    ],
)
def test_a_fading_card_with_no_located_card_clears_nothing(
    monkeypatch, caplog, after, card
):
    fading = _grid(FADING, shade=20)
    surface = _Surface()
    stash, calls = _grid_stash(
        monkeypatch,
        surface,
        slots=lambda rgb: GRID_SLOTS[:3] + GRID_SLOTS[13:]
        if rgb is fading
        else GRID_SLOTS,
    )
    with caplog.at_level(logging.DEBUG, logger="arc_helper"):
        stash.feed(fading, None, ())
        stash.step()
        stash.feed(fading if after else _grid(), card, ())
        stash.step()
    # find_slots loses the 10 slots 3 to 12 of 40 too, a quarter: big
    assert "stash big change waits changed=15 slots=40 lone=10" in caplog.text
    assert ("clear",) not in surface.calls
    assert "cleared" not in caplog.text
    assert "hidden" not in caplog.text
    assert len(calls) == 1


def _centre(i: int) -> Point:
    r = GRID_SLOTS[i]
    return Point((r.left + r.right) // 2, (r.top + r.bottom) // 2)


def test_the_hovered_slot_never_hides(monkeypatch, caplog):
    # The cursor moves over slot 5, then slot 6, then off the grid: the game
    # brightens the slot under it, the badges stay drawn and no pass runs
    surface = _Surface()
    stash, calls = _grid_stash(monkeypatch, surface)
    centre = _centre

    with caplog.at_level(logging.DEBUG, logger="arc_helper"):
        for frame, cursor in (
            (_grid([5]), centre(5)),
            (_grid([5]), centre(5)),
            (_grid([6]), centre(6)),
            (_grid(), Point(440, 270)),
        ):
            stash.feed(frame, None, (), cursor)
            stash.step()
            assert len(surface.draws()[-1]) == 80
    assert "hidden" not in caplog.text
    assert ("clear",) not in surface.calls
    assert len(calls) == 1


def test_a_slot_hovered_in_the_pass_frame_is_renewed_once_the_cursor_leaves(
    monkeypatch, caplog
):
    # The first pass runs while slot 5 is hovered, so slot 5 is never compared
    # with its frame: once the cursor moves to slot 6 a pass renews slot 5,
    # and no badge hides on the way
    monkeypatch.setattr(
        main, "find_slots", lambda _rgb: [SimpleNamespace(rect=r) for r in GRID_SLOTS]
    )
    calls = []

    def identify(rgb):
        calls.append(rgb)
        return [_slot(r) for r in GRID_SLOTS]

    surface = _Surface()
    stash = _stash(identify, surface)
    with caplog.at_level(logging.DEBUG, logger="arc_helper"):
        for frame, cursor in (
            (_grid([5]), _centre(5)),
            (_grid([5]), _centre(5)),
            (_grid([5]), _centre(5)),
            (_grid([6]), _centre(6)),
            (_grid([6]), _centre(6)),
        ):
            stash.feed(frame, None, (), cursor)
            stash.step()
            if calls:
                assert len(surface.draws()[-1]) == 80
    assert len(calls) == 2
    assert "hidden" not in caplog.text


def test_a_changed_slot_is_redrawn_from_the_slot_cache_before_the_pass(
    monkeypatch, caplog
):
    # Slot 5 takes a look the slot cache knows: hidden while it changes over
    # two frames, then redrawn from peek on the first still frame, before
    # that frame's pass
    surface = _Surface()
    draws_at_pass = []
    peeks = []

    def identify(_rgb):
        draws_at_pass.append(list(surface.draws()))
        return [_slot(r) for r in GRID_SLOTS]

    def peek(_rgb):
        peeks.append(1)
        return [_slot(GRID_SLOTS[5])]

    stash, _calls = _grid_stash(monkeypatch, surface, identify=identify, peek=peek)
    with caplog.at_level(logging.DEBUG, logger="arc_helper"):
        for shade in (40, 60):
            stash.feed(_grid([5], shade), None, ())
            stash.step()
        assert len(surface.draws()[-1]) == 78
        assert peeks == []
        stash.feed(_grid([5], 60), None, ())
        stash.step()
    assert peeks == [1]
    assert "stash changed slots redrawn slots=1 badges=2" in caplog.text
    # The second pass saw the redrawn slot already on screen
    assert len(draws_at_pass) == 2
    assert len(draws_at_pass[1][-1]) == 80
    assert len(surface.draws()[-1]) == 80


def test_half_the_slots_changing_clears_all(monkeypatch, caplog):
    surface = _Surface()
    stash, _calls = _grid_stash(monkeypatch, surface)
    with caplog.at_level(logging.DEBUG, logger="arc_helper"):
        for _ in range(2):
            stash.feed(_grid(range(20)), None, ())
            stash.step()
    assert surface.calls[-1] == ("clear",)
    assert "stash badges cleared reason=grid_moved changed=20 slots=40 lone=0" in (
        caplog.text
    )


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
        for _ in range(2):
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
    """Feed a frame whose find_slots finds after, with badges from before.

    The one lone slot hides only its own badges, nothing clears.
    """
    changed = _grid()
    surface = _Surface()
    stash, _calls = _grid_stash(
        monkeypatch,
        surface,
        slots=lambda rgb: after if rgb is changed else before,
    )
    with caplog.at_level(logging.DEBUG, logger="arc_helper"):
        for _ in range(2):
            stash.feed(changed, None, ())
            stash.step()
    assert ("clear",) not in surface.calls
    assert "cleared" not in caplog.text
    assert "stash badges hidden slots=1 reason=slot_changed" in caplog.text


def test_one_slot_removed_hides_only_its_badges(monkeypatch, caplog):
    _one_slot_off(monkeypatch, caplog, GRID_SLOTS, GRID_SLOTS[:-1])


def test_one_slot_added_hides_only_its_badges(monkeypatch, caplog):
    # The last slot found on the newer frame only
    _one_slot_off(monkeypatch, caplog, GRID_SLOTS[:-1], GRID_SLOTS)


def test_a_slot_missed_because_its_look_changed_only_hides(monkeypatch, caplog):
    # find_slots loses slot 3 once it is darkened: a changed slot, not a new set
    dark = [_grid([3]), _grid([3], 60)]
    surface = _Surface()
    stash, _calls = _grid_stash(
        monkeypatch,
        surface,
        slots=lambda rgb: GRID_SLOTS[:3] + GRID_SLOTS[4:]
        if any(rgb is d for d in dark)
        else GRID_SLOTS,
    )
    with caplog.at_level(logging.DEBUG, logger="arc_helper"):
        for frame in dark:
            stash.feed(frame, None, ())
            stash.step()
    # The second frame confirms the hide and, still, starts the pass at once
    assert ("clear",) not in surface.calls
    assert "stash badges hidden slots=1 reason=slot_changed" in caplog.text


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


def test_a_stale_pass_does_not_hold_the_next_back_by_the_gap():
    # The scroll lands while the first pass runs, 0.1 s after its start
    surface, clock = _Surface(), _Clock()
    calls = []
    stash = None

    def identify(rgb):
        calls.append(rgb)
        if len(calls) == 1:
            stash.feed(_frame(120), None, ())
            stash._watch_once()  # noqa: SLF001
        return [_slot(SLOT_A)]

    stash = StashBadges(identify, _badges, surface, threaded=False, clock=clock)
    assert main.STASH_PASS_GAP_S == 0.25
    stash.feed(_frame(0), None, ())
    stash.step()
    clock.now += 0.1
    stash.feed(_frame(0), None, ())
    stash.step()
    assert len(calls) == 1
    assert surface.draws() == []
    # The next frame confirms the scroll and clears, the one after is still
    clock.now += 0.1
    stash.feed(_frame(120), None, ())
    stash.step()
    clock.now += 0.1
    stash.feed(_frame(120), None, ())
    stash.step()
    assert len(calls) == 2
    assert surface.draws() == [[tuple(SLOT_A)] * 2]


def _known_stash(clock, pass_s=0.0, known=(SLOT_A,), found=(SLOT_A, SLOT_B)):
    """A stash whose identify knows the known slots already (identify.peek).

    Each pass finds found and moves clock on by pass_s.
    """
    order = []

    def identify(_rgb):
        order.append("pass")
        clock.now += pass_s
        return [_slot(r) for r in found]

    def peek(_rgb):
        order.append("peek")
        return [_slot(r) for r in known]

    identify.peek = peek
    surface = _Surface()
    stash = StashBadges(identify, _badges, surface, threaded=False, clock=clock)
    return stash, surface, order


def _feed_at(stash, clock, shades, start=100.0, apart=0.15) -> None:
    """Feed frames of these shades apart seconds on the clock, one step each."""
    for n, shade in enumerate(shades):
        clock.now = max(clock.now, round(start + n * apart, 6))
        stash.feed(_frame(shade), None, ())
        stash.step()


def test_known_slots_are_drawn_before_the_pass_and_replaced_by_it():
    clock = _Clock()
    stash, surface, order = _known_stash(clock, found=(SLOT_B,))
    _feed_at(stash, clock, (0, 0, 120, 120))
    assert order == ["pass"]
    assert surface.calls[-1] == ("clear",)
    order.clear()
    before = len(surface.draws())
    _feed_at(stash, clock, (120,), start=clock.now + 0.15)
    assert order == ["peek", "pass"]
    # SLOT_A's known badges first, then the pass's own, which have no SLOT_A
    assert surface.draws()[before:] == [[tuple(SLOT_A)] * 2, [tuple(SLOT_B)] * 2]


def test_a_peek_that_finds_nothing_draws_nothing():
    clock = _Clock()
    stash, surface, order = _known_stash(clock, known=(), found=(SLOT_B,))
    _feed_at(stash, clock, (0, 0, 120, 120))
    before = len(surface.draws())
    _feed_at(stash, clock, (120,), start=clock.now + 0.15)
    assert order == ["pass", "peek", "pass"]
    assert surface.draws()[before:] == [[tuple(SLOT_B)] * 2]


@pytest.mark.parametrize(
    ("known", "line"),
    [
        (
            (SLOT_A,),
            "stash latency moved_to_still=150 pass=50 still_to_drawn=0 total=150 ms",
        ),
        ((), "stash latency moved_to_still=150 pass=50 still_to_drawn=50 total=200 ms"),
    ],
)
def test_the_latency_line_after_a_scroll(caplog, known, line):
    # Frames 150 ms apart: two still ones, the moved one, the one that
    # confirms it and clears, then still again. Each pass takes 50 ms on the
    # clock, known slots are drawn at once
    clock = _Clock()
    stash, _surface, _order = _known_stash(clock, 0.05, known)
    with caplog.at_level(logging.DEBUG, logger="arc_helper"):
        _feed_at(stash, clock, (0, 0, 120, 120, 120))
    lines = [r.getMessage() for r in caplog.records if "latency" in r.getMessage()]
    assert lines == [line]


def test_peek_leaves_the_slot_cache_unchanged(monkeypatch):
    monkeypatch.setattr(stash_scan, "bar_geometry", lambda _crop: None)
    monkeypatch.setattr(stash_scan, "read_tier", lambda _crop: None)
    monkeypatch.setattr(stash_scan, "read_stack", lambda _crop: None)
    slots = [
        Slot(Rect(100 * n, 0, 100 * n + 90, 90), "stash", filled=True) for n in range(3)
    ]

    def looks(shades) -> np.ndarray:
        rgb = np.zeros((90, 300, 3), np.uint8)
        for slot, shade in zip(slots, shades, strict=True):
            r = slot.rect
            rgb[r.top : r.bottom, r.left : r.right] = shade
            rgb[r.top + 40 : r.top + 50, r.left + 40 : r.left + 50] = 255 - shade
        return rgb

    embedder = SimpleNamespace(
        embed=lambda squares: np.eye(3, dtype=np.float32)[: len(squares)]
    )
    gallery = EmbedGallery(
        ("a", "b", "c"), ("A", "B", "C"), np.eye(3, dtype=np.float32)
    )
    cache = stash_scan.SlotCache()
    stash_scan.identify_slots(
        looks([60, 120, 180]),
        gallery,
        lambda _name: None,
        embedder=embedder,
        slots=slots,
        read_stacks=True,
        cache=cache,
    )
    keys, bars = list(cache._entries), list(cache._bars)  # noqa: SLF001
    # The oldest look first, then a look the cache has never seen
    found = stash_scan.peek_slots(
        looks([60, 250, 120]), cache, read_stacks=True, slots=slots
    )
    assert [s.guess for s in found] == ["A", "B"]
    assert [s.rect for s in found] == [slots[0].rect, slots[2].rect]
    assert list(cache._entries) == keys  # noqa: SLF001
    assert list(cache._bars) == bars  # noqa: SLF001


# A textured frame with two panels of 48 px slots 56 px apart: equipment
# (2 x 4) on the left, which never scrolls, and backpack (3 x 5) on the
# right, whose viewport is SCROLL_VIEW. A scroll rolls the viewport's content
# up by SCROLL_DY, 2 grey cells, so the slot looks still vote on it
SCROLL_SHAPE = (320, 448, 3)
SCROLL_VIEW = Rect(224, 16, 392, 296)
SCROLL_DY = -16
EQUIPMENT_SLOTS = [
    Rect(16 + 56 * c, 16 + 56 * r, 64 + 56 * c, 64 + 56 * r)
    for r in range(4)
    for c in range(2)
]
BACKPACK_SLOTS = [
    Rect(224 + 56 * c, 16 + 56 * r, 272 + 56 * c, 64 + 56 * r)
    for r in range(5)
    for c in range(3)
]


def _textured(seed: int) -> np.ndarray:
    """Random grey 8 px cells over the whole frame, as RGB."""
    cells = np.random.default_rng(seed).integers(0, 256, (40, 56), dtype=np.uint8)
    grey = np.repeat(np.repeat(cells, 8, 0), 8, 1)
    return np.repeat(grey[:, :, None], 3, 2)


def _scrolled(rgb: np.ndarray) -> np.ndarray:
    """rgb with the backpack viewport's content rolled up by -SCROLL_DY px."""
    out = rgb.copy()
    v = SCROLL_VIEW
    out[v.top : v.bottom, v.left : v.right] = np.roll(
        rgb[v.top : v.bottom, v.left : v.right], SCROLL_DY, axis=0
    )
    return out


def _moved(rect: Rect, dy: int) -> Rect:
    return Rect(rect.left, rect.top + dy, rect.right, rect.bottom + dy)


# The backpack slots on the scrolled frame: the top row is partly out of the
# viewport and not found
SCROLLED_BACKPACK = [
    _moved(r, SCROLL_DY) for r in BACKPACK_SLOTS if r.top + SCROLL_DY >= 16
]


class _TranslatingSurface(_Surface):
    def translate(self, region, dy) -> None:
        self.calls.append(("translate", tuple(region), dy))


def _panel_stash(monkeypatch, surface, frames):
    """A stash on the textured frame with every slot's badges drawn.

    find_slots finds the backpack slots of SCROLLED_BACKPACK on the frames
    in frames (by identity) and BACKPACK_SLOTS on any other. On those
    frames the pass identifies every slot but SCROLLED_BACKPACK[0].
    """

    def find(rgb):
        backpack = (
            SCROLLED_BACKPACK if any(rgb is f for f in frames) else BACKPACK_SLOTS
        )
        return [
            *(Slot(r, "equipment", filled=True) for r in EQUIPMENT_SLOTS),
            *(Slot(r, "backpack", filled=True) for r in backpack),
        ]

    monkeypatch.setattr(main, "find_slots", find)
    calls = []

    def identify(rgb):
        calls.append(rgb)
        found = [s.rect for s in find(rgb)]
        if any(rgb is f for f in frames):
            found.remove(SCROLLED_BACKPACK[0])
        return [_slot(r) for r in found]

    stash = _stash(identify, surface)
    base = _textured(1)
    for _ in range(2):
        stash.feed(base, None, ())
        stash.step()
    assert len(surface.draws()[-1]) == 2 * (8 + 15)
    return stash, calls, base


def test_scroll_offset_measures_a_shifted_panel():
    base = _textured(1)
    assert main.scroll_offset(base, _scrolled(base), SCROLL_VIEW) == SCROLL_DY
    down = base.copy()
    v = SCROLL_VIEW
    down[v.top : v.bottom, v.left : v.right] = np.roll(
        base[v.top : v.bottom, v.left : v.right], 13, axis=0
    )
    # Off the 8 px cell grid: the grey copies estimate, the full size refines
    assert main.scroll_offset(base, down, SCROLL_VIEW) == 13


def test_scroll_offset_is_none_for_a_change_that_is_no_shift():
    base = _textured(1)
    other = base.copy()
    v = SCROLL_VIEW
    other[v.top : v.bottom, v.left : v.right] = _textured(2)[
        v.top : v.bottom, v.left : v.right
    ]
    assert main.scroll_offset(base, other, SCROLL_VIEW) is None


def test_scroll_offset_is_none_when_the_slot_looks_disagree():
    # The frame repeats its viewport's top 64 rows 64 px lower, so slot a's
    # look is found 88 px above where the content moved by 24
    base = _textured(1)
    v = SCROLL_VIEW
    base[v.top + 64 : v.top + 128, v.left : v.right] = base[
        v.top : v.top + 64, v.left : v.right
    ]
    moved = base.copy()
    moved[v.top : v.bottom, v.left : v.right] = np.roll(
        base[v.top : v.bottom, v.left : v.right], -24, axis=0
    )
    slot = Rect(v.left, v.top + 24, v.left + 48, v.top + 56)
    assert main.scroll_offset(base, moved, SCROLL_VIEW) == -24
    vote = main.scroll_offset(
        base, moved, SCROLL_VIEW, [_moved(slot, 64)], [_moved(slot, -24)]
    )
    assert vote is None


def test_the_slot_vote_checks_the_refined_offset(monkeypatch):
    # The vote sits a cell above the grey estimate and the refinement moves
    # the offset a cell below it, so the final offset is two cells off
    base = _textured(1)
    seen = {}

    def refined(_a, _b, _band, offset, by=-8):
        seen["coarse"] = offset
        return offset + by

    monkeypatch.setattr(grid, "_refined", refined)
    monkeypatch.setattr(grid, "_look_vote", lambda *_args: seen["coarse"] + 8)
    assert main.scroll_offset(base, _scrolled(base), SCROLL_VIEW) is None
    # Without the refinement's move the same vote is within a cell
    monkeypatch.setattr(grid, "_refined", lambda *args: refined(*args, by=0))
    assert main.scroll_offset(base, _scrolled(base), SCROLL_VIEW) == seen["coarse"]


def test_a_panel_scroll_moves_its_badges_and_drops_those_scrolled_out(
    monkeypatch, caplog
):
    surface = _TranslatingSurface()
    scrolled = _scrolled(_textured(1))
    stash, calls, _base = _panel_stash(monkeypatch, surface, [scrolled])
    with caplog.at_level(logging.DEBUG, logger="arc_helper"):
        stash.feed(scrolled, None, ())
        stash.step()
    region = Rect(224, 16, 384, 288)
    assert surface.calls[-1] == ("translate", tuple(region), SCROLL_DY)
    assert ("clear",) not in surface.calls
    assert "stash badges followed panel=backpack dy=-16 kept=40 dropped=6" in (
        caplog.text
    )
    # Equipment badges stay, backpack rows 1 to 4 move up, row 0 is gone
    kept = {tuple(int(v) for v in b["rect"]) for b in stash._badges}  # noqa: SLF001
    assert kept == {tuple(r) for r in (*EQUIPMENT_SLOTS, *SCROLLED_BACKPACK)}
    assert len(calls) == 1
    # The pass on the next still frame replaces them with its own, and the
    # slot it found nothing on, whose look did not change, keeps its badges
    with caplog.at_level(logging.DEBUG, logger="arc_helper"):
        stash.feed(scrolled, None, ())
        stash.step()
    assert len(calls) == 2
    assert "stash pass kept_drawn=2" in caplog.text
    assert _drawn_slots(surface) == kept


def test_a_pass_with_fewer_badges_keeps_the_drawn_ones(monkeypatch, caplog):
    # After the scroll the pass identifies none of the viewport's first row,
    # whose looks did not change: their followed badges stay drawn
    surface = _TranslatingSurface()
    scrolled = _scrolled(_textured(1))
    stash, calls, _base = _panel_stash(monkeypatch, surface, [scrolled])
    stash.feed(scrolled, None, ())
    stash.step()
    followed = {tuple(int(v) for v in b["rect"]) for b in stash._shown}  # noqa: SLF001
    missed = set(SCROLLED_BACKPACK[:3])
    monkeypatch.setattr(
        stash,
        "_identify",
        lambda rgb: [
            _slot(s.rect) for s in main.find_slots(rgb) if s.rect not in missed
        ],
    )
    with caplog.at_level(logging.DEBUG, logger="arc_helper"):
        stash.feed(scrolled, None, ())
        stash.step()
    assert "stash pass kept_drawn=6" in caplog.text
    assert _drawn_slots(surface) == followed
    assert len(calls) == 1


def test_a_tab_switch_still_clears_every_badge(monkeypatch, caplog):
    surface = _TranslatingSurface()
    stash, _calls, _base = _panel_stash(monkeypatch, surface, [])
    switched = _textured(2)
    with caplog.at_level(logging.DEBUG, logger="arc_helper"):
        for _ in range(2):
            stash.feed(switched, None, ())
            stash.step()
    assert surface.calls[-1] == ("clear",)
    assert not [c for c in surface.calls if c[0] == "translate"]
    assert "stash badges cleared reason=grid_moved" in caplog.text


# 59 slots, the stash and loadout of the 17 Oct log, on an 8 by 8 grid
LONE_SLOTS = [
    Rect(56 * c, 56 * r, 56 * c + 48, 56 * r + 48) for r in range(8) for c in range(8)
][:59]


def _lone_run(monkeypatch, caplog, lost: int) -> _Surface:
    """Badges on 59 slots, then two frames on which find_slots loses lost slots."""
    lone = np.zeros((448, 448, 3), np.uint8)

    def found(rgb):
        return LONE_SLOTS[lost:] if rgb is lone else LONE_SLOTS

    monkeypatch.setattr(
        main, "find_slots", lambda rgb: [SimpleNamespace(rect=r) for r in found(rgb)]
    )
    surface = _Surface()
    stash = _stash(lambda rgb: [_slot(r) for r in found(rgb)], surface)
    for _ in range(2):
        stash.feed(np.zeros((448, 448, 3), np.uint8), None, ())
        stash.step()
    assert len(surface.draws()[-1]) == 118
    with caplog.at_level(logging.DEBUG, logger="arc_helper"):
        for _ in range(2):
            stash.feed(lone, None, ())
            stash.step()
    return surface


def test_four_lone_slots_of_59_hide_only_their_badges(monkeypatch, caplog):
    surface = _lone_run(monkeypatch, caplog, 4)
    assert ("clear",) not in surface.calls
    assert "stash badges hidden slots=4 reason=slot_changed" in caplog.text
    assert _drawn_slots(surface) == {tuple(r) for r in LONE_SLOTS[4:]}


def test_twenty_lone_slots_of_59_clear_all(monkeypatch, caplog):
    # 20 of 59 is under BIG_CHANGE_FRACTION but over LONE_BIG_FRACTION
    surface = _lone_run(monkeypatch, caplog, 20)
    assert surface.calls[-1] == ("clear",)
    assert (
        "stash badges cleared reason=grid_moved changed=20 slots=59 lone=20"
        in caplog.text
    )
