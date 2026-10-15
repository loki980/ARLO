"""Tests for the advice card's home beneath the second gun, its escape spots
and its return home."""

from types import SimpleNamespace

import numpy as np

import arc_helper.main as main_module
from arc_helper.database import Item
from arc_helper.grid import EQUIPMENT
from arc_helper.grid import STASH
from arc_helper.grid import Slot
from arc_helper.layout_home import fixed_home
from arc_helper.layout_home import home_below_weapons
from arc_helper.layout_home import weapon_slots
from arc_helper.main import POPUP
from arc_helper.main import POPUP_BELOW
from arc_helper.main import POPUP_CORNER
from arc_helper.main import POPUP_HOME
from arc_helper.main import POPUP_LEFT
from arc_helper.main import POPUP_RIGHT
from arc_helper.main import Scanner
from arc_helper.main import ScannerState
from arc_helper.main import popup_escape_spot
from arc_helper.ocr import CardSignature
from arc_helper.tooltip_layout import Point
from arc_helper.tooltip_layout import Rect

SCREEN = (3840, 2160)
# HOME_GAP of 2160 is 13 px, FLOOR is 2009
GAP = 13
FLOOR_Y = 2009

# Weapon slots found on base-stiletto-4k.png and raid-inventory-4k.png
BASE_GUNS = (Rect(1384, 784, 1994, 1186), Rect(1384, 1216, 1994, 1618))
RAID_GUNS = (Rect(1292, 832, 1902, 1234), Rect(1292, 1264, 1902, 1666))


def _slots(guns, left):
    """find_slots output: two small equipment slots, the guns, stash slots."""
    small = [
        Slot(Rect(left, 544, left + 290, 738), EQUIPMENT, filled=True),
        Slot(Rect(left + 320, 544, left + 610, 738), EQUIPMENT, filled=True),
    ]
    grid = [
        Slot(Rect(300 + 208 * i, 600, 492 + 208 * i, 792), STASH, filled=False)
        for i in range(4)
    ]
    return grid + small + [Slot(g, EQUIPMENT, filled=True) for g in guns]


def test_home_is_beneath_the_second_gun_on_the_base_stash_screen():
    guns = weapon_slots(_slots(BASE_GUNS, 1384))
    assert guns == list(BASE_GUNS)
    assert home_below_weapons(guns, SCREEN) == Rect(1384, 1618 + GAP, 1994, FLOOR_Y)


def test_home_is_beneath_the_second_gun_in_raid():
    guns = weapon_slots(_slots(RAID_GUNS, 1292))
    assert home_below_weapons(guns, SCREEN) == Rect(1292, 1666 + GAP, 1902, FLOOR_Y)


def test_one_or_no_gun_found_gives_no_home():
    # The game's item card can hide the second gun, the first alone is not
    # trusted
    assert home_below_weapons(weapon_slots(_slots(BASE_GUNS[:1], 1384)), SCREEN) is None
    assert home_below_weapons(weapon_slots([]), SCREEN) is None


def test_fixed_home_is_the_loadout_box_down_to_the_floor():
    assert fixed_home(SCREEN) == Rect(1367, 1635, 2047, FLOOR_Y)


# Escape: a 600 x 300 popup at home, a 22 px gap (ESCAPE_GAP of 2160)
HOME = Rect(1384, 1631, 1984, 1931)
CORNER = Rect(57, 57, 657, 357)
ESCAPE_GAP_PX = 22


def _labelled(**extra):
    return {POPUP_HOME: HOME, POPUP_CORNER: CORNER, **extra}


def test_escape_prefers_home_when_the_card_leaves_it_free():
    card = Rect(2400, 300, 3100, 1300)
    assert popup_escape_spot(card, _labelled(), SCREEN) == (POPUP_HOME, HOME)


def test_escape_left_when_it_is_the_smaller_move():
    # Card from 1700 to 2400: left lands at 1078 (306 away), right at 2422
    card = Rect(1700, 1200, 2400, 1700)
    left = 1700 - ESCAPE_GAP_PX - 600
    assert popup_escape_spot(card, _labelled(), SCREEN) == (
        POPUP_LEFT,
        Rect(left, HOME.top, left + 600, HOME.bottom),
    )


def test_escape_right_when_it_is_the_smaller_move():
    # Card from 1000 to 1500: right lands at 1522 (138 away), left at 378
    card = Rect(1000, 1200, 1500, 1700)
    right = 1500 + ESCAPE_GAP_PX
    assert popup_escape_spot(card, _labelled(), SCREEN) == (
        POPUP_RIGHT,
        Rect(right, HOME.top, right + 600, HOME.bottom),
    )


def test_escape_left_on_a_tie():
    # Card centred on home: both moves are 622 + 22 px
    card = Rect(1084, 1200, 2284, 1700)
    label, _ = popup_escape_spot(card, _labelled(), SCREEN)
    assert label == POPUP_LEFT


def test_escape_right_when_left_leaves_the_screen():
    # Left is the smaller move (1506 against 1538) but starts at x -122
    card = Rect(500, 1200, 2900, 1700)
    label, rect = popup_escape_spot(card, _labelled(), SCREEN)
    assert label == POPUP_RIGHT
    assert rect.left == 2900 + ESCAPE_GAP_PX


def test_escape_below_when_both_sides_leave_the_screen():
    # A card nearly as wide as the screen, over home's top
    card = Rect(200, 1000, 3700, 1700)
    assert popup_escape_spot(card, _labelled(), SCREEN) == (
        POPUP_BELOW,
        Rect(HOME.left, 1722, HOME.right, 2022),
    )


def test_escape_corner_when_nothing_else_fits():
    card = Rect(200, 900, 3700, 2000)
    assert popup_escape_spot(card, _labelled(), SCREEN) == (POPUP_CORNER, CORNER)


def test_no_escape_when_the_card_covers_every_spot():
    card = Rect(0, 0, 3840, 2160)
    assert popup_escape_spot(card, _labelled(), SCREEN) is None


# Scanner: home refresh and the return home


class _Ocr:
    """OCR stub: each scan finds the next card (None for no card)."""

    last_tier_vote = None

    def __init__(self, cards, read="Battery"):
        self.cards = list(cards)
        self.read = read
        self._last_card = None
        self._last_origin = Point(0, 0)
        self._last_capture = None

    def locate_card_at_cursor(self):
        card = self.cards.pop(0)
        self._last_card = card
        if card is None:
            return None
        return CardSignature(card, np.full((8, 32), card.left % 251, np.uint8))

    def read_located_card(self):
        return self.read


class _Db:
    def lookup(self, name, tier_vote=None):  # noqa: ARG002, PLR6301
        return Item(name=name, action="Keep")

    def log_missing_item(self, name):
        pass


def _settings():
    return SimpleNamespace(
        scan=SimpleNamespace(tooltip_scan_interval=0.0),
        overlay=SimpleNamespace(display_time=1.0, cooldown=60.0, tooltip_chips=False),
        show_ended_projects=False,
    )


def _scanner(monkeypatch, ocr):
    scanner = Scanner(root=None, overlay=None, status=None, db=_Db(), hotkey=None)
    scanner._running = True  # noqa: SLF001 - test seam
    scanner.state = ScannerState.ACTIVE
    monkeypatch.setattr(main_module, "get_settings", _settings)
    monkeypatch.setattr(main_module, "get_ocr_engine", lambda: ocr)
    monkeypatch.setattr(main_module, "get_screen_resolution", lambda: SCREEN)
    monkeypatch.setattr(main_module.time, "sleep", lambda _s: None)
    monkeypatch.setattr(scanner, "_verdict_for", lambda *_: "verdict")
    return scanner


def _drain(scanner):
    messages = []
    while not scanner.ui_queue.empty():
        messages.append(scanner.ui_queue.get_nowait())
    return messages


def _cycle(scanner, ocr):
    scanner._scan_cycle(ocr, _settings())  # noqa: SLF001
    return _drain(scanner)


def test_home_refreshes_at_most_once_a_second_and_only_when_the_guns_move(
    monkeypatch,
):
    ocr = _Ocr([None] * 5)
    ocr._last_capture = np.zeros((SCREEN[1], SCREEN[0], 3), np.uint8)  # noqa: SLF001
    scanner = _scanner(monkeypatch, ocr)
    found = [_slots(BASE_GUNS, 1384)]
    calls = []

    def find_slots(_rgb):
        calls.append(1)
        return found[0]

    now = [100.0]
    monkeypatch.setattr(main_module, "find_slots", find_slots)
    monkeypatch.setattr(main_module.time, "monotonic", lambda: now[0])

    base_home = Rect(1384, 1618 + GAP, 1994, FLOOR_Y)
    assert ("popup_home", base_home) in _cycle(scanner, ocr)

    # Within the second: no look at all
    now[0] += 0.5
    assert _cycle(scanner, ocr) == []
    assert len(calls) == 1

    # A second later the guns are 2 px lower: jitter, nothing sent
    now[0] += 1.0
    moved = [Rect(r.left, r.top + 2, r.right, r.bottom + 2) for r in BASE_GUNS]
    found[0] = _slots(moved, 1384)
    assert _cycle(scanner, ocr) == []
    assert len(calls) == 2

    # The raid layout: home follows the second gun
    now[0] += 1.0
    found[0] = _slots(RAID_GUNS, 1292)
    assert ("popup_home", Rect(1292, 1666 + GAP, 1902, FLOOR_Y)) in _cycle(scanner, ocr)


def test_no_home_refresh_without_the_guns(monkeypatch):
    ocr = _Ocr([None])
    ocr._last_capture = np.zeros((SCREEN[1], SCREEN[0], 3), np.uint8)  # noqa: SLF001
    scanner = _scanner(monkeypatch, ocr)
    monkeypatch.setattr(main_module, "find_slots", lambda _rgb: [])

    assert [m for m in _cycle(scanner, ocr) if m[0] == "popup_home"] == []


# A card over home (HOME above), and the same item's card moved clear of it
OVER_HOME = Rect(1300, 900, 2000, 1800)
CLEAR = Rect(2400, 300, 3100, 1300)
# Right of it is the smaller move (638 px against 706 to the left)
AWAY_SPOT = Rect(2022, HOME.top, 2622, HOME.bottom)


def test_covered_then_closed_card_returns_home_then_hides_after_the_debounce(
    monkeypatch,
):
    ocr = _Ocr([OVER_HOME, OVER_HOME, None, None])
    scanner = _scanner(monkeypatch, ocr)
    scanner.overlay_rects.publish([HOME], _labelled(popup=HOME))

    # The advice card at home covers the game card: the read is discarded
    # and the card escapes to the right
    assert _cycle(scanner, ocr) == [("popup_spot", POPUP_RIGHT, AWAY_SPOT)]

    # Away, the game card reads whole and shows. Home is still covered
    scanner.overlay_rects.publish(
        [AWAY_SPOT], _labelled(popup=AWAY_SPOT), popup_away=True
    )
    assert [m[0] for m in _cycle(scanner, ocr)] == ["overlay"]

    # The game card closes: back home on the first empty scan, not hidden
    assert _cycle(scanner, ocr) == [("popup_spot", POPUP_HOME, None)]

    # The Tk thread moved it home, the second empty scan hides it
    scanner.overlay_rects.publish([HOME], _labelled(popup=HOME))
    assert _cycle(scanner, ocr) == [("hide",)]


def test_same_item_card_moving_off_home_brings_the_popup_back(monkeypatch):
    ocr = _Ocr([OVER_HOME, CLEAR])
    scanner = _scanner(monkeypatch, ocr)
    scanner.overlay_rects.publish(
        [AWAY_SPOT], _labelled(popup=AWAY_SPOT), popup_away=True
    )
    assert [m[0] for m in _cycle(scanner, ocr)] == ["overlay"]

    # The same item's card, now clear of home
    messages = _cycle(scanner, ocr)
    assert messages[0] == ("popup_spot", POPUP_HOME, None)
    assert [m[0] for m in messages] == ["popup_spot", "overlay"]


def test_popup_at_home_never_queues_a_return(monkeypatch):
    ocr = _Ocr([CLEAR, None])
    scanner = _scanner(monkeypatch, ocr)
    scanner.overlay_rects.publish([HOME], _labelled(popup=HOME))

    assert [m[0] for m in _cycle(scanner, ocr)] == ["overlay"]
    assert _cycle(scanner, ocr) == []


def test_popup_label_is_the_covering_window():
    assert POPUP in main_module.COVERING_WINDOWS
