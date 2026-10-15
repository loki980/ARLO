"""Tests that a card covered by ARLO's popup or status window is not read."""

import arc_helper.main as main_module
import arc_helper.overlay as overlay_module
from arc_helper.database import Item
from arc_helper.main import POPUP
from arc_helper.main import POPUP_CORNER
from arc_helper.main import POPUP_HOME
from arc_helper.main import POPUP_LEFT
from arc_helper.main import STATUS
from arc_helper.main import Application
from arc_helper.main import Scanner
from arc_helper.main import ScannerState
from arc_helper.main import popup_spot_rects
from arc_helper.overlay import OverlayWindow
from arc_helper.tooltip_layout import Point
from arc_helper.tooltip_layout import Rect

SCREEN = (3840, 2160)
# OVERLAY_X and OVERLAY_Y at their defaults: the top left corner spot
CORNER = Point(100, 100)
# Popup size in physical px, measured on the Anvil I evidence screenshot
POPUP_SIZE = (536, 516)
HOME_RECT, CORNER_RECT = popup_spot_rects(SCREEN, POPUP_SIZE, CORNER)

# find_tooltip_card on 20261014-210223-142_U-AMMO-TURE-N-HEAVY-AMMO-MAGAZINE-
# SIZE-6.png with cursor Point(400, 640), full screen capture at origin 0, 0
ANVIL_CARD = Rect(528, 314, 1287, 1282)
# A card in the upper middle of the screen, clear of both spots
CLEAR_CARD = Rect(1500, 300, 2200, 1200)
# A card reaching down over the LOADOUT home spot
HOME_CARD = Rect(1300, 900, 2000, 1800)

ANVIL = Item(name="Anvil I", action="Keep", sell_price=5000)
BATTERY = Item(name="Battery", action="Keep", sell_price=250)


class _FakeHotkey:
    def stop(self):
        pass


class _FakeDb:
    def __init__(self):
        self.missing = []

    def lookup(self, name, tier_vote=None):  # noqa: ARG002, PLR6301
        return {"Anvil I": ANVIL, "Battery": BATTERY}.get(name)

    def log_missing_item(self, name):
        self.missing.append(name)


class _CardOcr:
    """OCR engine stub whose last capture found card at origin."""

    last_tier_vote = None

    def __init__(self, card, origin=Point(0, 0)):  # noqa: B008
        self._last_card = card
        self._last_origin = origin


def _scanner(monkeypatch, card, labelled):
    scanner = Scanner(
        root=None, overlay=None, status=None, db=_FakeDb(), hotkey=_FakeHotkey()
    )
    scanner._running = True  # noqa: SLF001 - test seam
    scanner.state = ScannerState.ACTIVE
    scanner.overlay_rects.publish(list(labelled.values()), labelled)
    monkeypatch.setattr(main_module, "get_ocr_engine", lambda: _CardOcr(card))
    monkeypatch.setattr(scanner, "_verdict_for", lambda *_: "verdict")
    monkeypatch.setattr(scanner, "_queue_chips", lambda *_, **__: None)
    return scanner


def _set_card(monkeypatch, card):
    monkeypatch.setattr(main_module, "get_ocr_engine", lambda: _CardOcr(card))


def _drain(scanner):
    messages = []
    while not scanner.ui_queue.empty():
        messages.append(scanner.ui_queue.get_nowait())
    return messages


def _spots(**windows):
    return {POPUP_HOME: HOME_RECT, POPUP_CORNER: CORNER_RECT, **windows}


def test_card_under_the_popup_is_discarded_and_the_popup_moves(monkeypatch):
    card = HOME_CARD
    scanner = _scanner(monkeypatch, card, _spots(popup=HOME_RECT))

    scanner._handle_detected_item("Battery")  # noqa: SLF001

    # Left of the card is the smaller move (625 px against 655 to the right)
    assert _drain(scanner) == [("popup_spot", POPUP_LEFT, Rect(742, 1493, 1278, 2009))]
    assert not scanner._last_shown_item  # noqa: SLF001
    assert scanner.db.missing == []


def test_card_clear_of_the_popup_shows_as_before(monkeypatch):
    scanner = _scanner(monkeypatch, CLEAR_CARD, _spots(popup=HOME_RECT))

    scanner._handle_detected_item("Battery")  # noqa: SLF001

    messages = _drain(scanner)
    assert [m[0] for m in messages] == ["overlay"]
    assert messages[0][1] == "Battery"


def test_chips_and_frame_rects_never_discard_a_read(monkeypatch):
    # Unlabelled rects (chips, gold frame) sit on the card by design
    scanner = _scanner(monkeypatch, CLEAR_CARD, _spots())
    scanner.overlay_rects.publish(
        [Rect(1600, 800, 1700, 900), Rect(1500, 700, 1510, 1600)], _spots()
    )

    scanner._handle_detected_item("Battery")  # noqa: SLF001

    assert [m[0] for m in _drain(scanner)] == ["overlay"]


def test_card_under_the_status_window_is_discarded_without_a_move(monkeypatch):
    status = Rect(10, 10, 300, 60)
    card = Rect(200, 40, 900, 900)
    scanner = _scanner(monkeypatch, card, _spots(status=status))

    scanner._handle_detected_item("Anvil I")  # noqa: SLF001

    assert _drain(scanner) == []


def test_anvil_capture_under_the_corner_popup_is_discarded_and_the_popup_moved(
    monkeypatch,
):
    """Real capture 20261014-210223: the Anvil I card under the KEEP popup.

    Tonight the popup sat in the top left corner, where it now only goes
    while a card covers its home.
    """
    scanner = _scanner(monkeypatch, ANVIL_CARD, _spots(popup=CORNER_RECT))

    scanner._handle_detected_item("U AMMO TYPE N, HEAVY AMMO MAGAZINE SIZE 6")  # noqa: SLF001

    assert _drain(scanner) == [("popup_spot", POPUP_HOME, HOME_RECT)]
    assert scanner.db.missing == []
    # The home spot leaves the whole card visible
    assert not main_module.rects_intersect(ANVIL_CARD, HOME_RECT)


def test_anvil_card_reads_normally_with_the_popup_at_home(monkeypatch):
    scanner = _scanner(monkeypatch, ANVIL_CARD, _spots(popup=HOME_RECT))

    scanner._handle_detected_item("Anvil I")  # noqa: SLF001

    assert [m[0] for m in _drain(scanner)] == ["overlay"]


def test_home_spot_is_in_the_loadout_panel_above_the_floor():
    # 0.356 and 0.757 of a 4K screen, a short popup fits below the top
    home, corner = popup_spot_rects(SCREEN, (536, 200), CORNER)
    assert home == Rect(1367, 1635, 1903, 1835)
    assert corner == Rect(100, 100, 636, 300)
    # A tall popup shifts up so its bottom stays at 0.93 of the height
    home, _ = popup_spot_rects(SCREEN, POPUP_SIZE, CORNER)
    assert home == Rect(1367, 1493, 1903, 2009)


class _StubToplevel:
    def __init__(self, viewable):
        self.viewable = viewable
        self.geometries = []
        self.size = POPUP_SIZE

    def winfo_viewable(self):
        return self.viewable

    def winfo_width(self):
        return self.size[0]

    def winfo_height(self):
        return self.size[1]

    def geometry(self, spec):
        self.geometries.append(spec)


def _overlay(monkeypatch, viewable):
    monkeypatch.setattr(overlay_module, "phys_to_tk", lambda v: v)
    overlay = OverlayWindow.__new__(OverlayWindow)
    overlay.window = _StubToplevel(viewable)
    overlay.overlay_x, overlay.overlay_y = CORNER
    return overlay


def test_set_position_moves_a_visible_popup(monkeypatch):
    overlay = _overlay(monkeypatch, viewable=True)
    overlay.set_position(1367, 1493)
    assert overlay.window.geometries == ["+1367+1493"]
    assert (overlay.overlay_x, overlay.overlay_y) == (1367, 1493)


def test_set_position_only_stores_for_a_hidden_popup(monkeypatch):
    overlay = _overlay(monkeypatch, viewable=False)
    overlay.set_position(1367, 1493)
    assert overlay.window.geometries == []
    assert (overlay.overlay_x, overlay.overlay_y) == (1367, 1493)


def test_popup_spot_message_moves_the_popup(monkeypatch):
    monkeypatch.setattr(
        main_module, "get_backend", lambda: type("B", (), {"tk_scale": 1.0})
    )
    monkeypatch.setattr(main_module, "get_screen_resolution", lambda: SCREEN)
    app = Application.__new__(Application)
    app.overlay = _overlay(monkeypatch, viewable=True)
    app._popup_corner = CORNER  # noqa: SLF001
    app._popup_spot = POPUP_HOME  # noqa: SLF001

    app._place_popup()  # noqa: SLF001
    assert app.overlay.window.geometries == [f"+{HOME_RECT.left}+{HOME_RECT.top}"]

    # Already in place: no second move
    app._place_popup()  # noqa: SLF001
    assert len(app.overlay.window.geometries) == 1

    app._popup_spot = POPUP_CORNER  # noqa: SLF001
    app._place_popup()  # noqa: SLF001
    assert app.overlay.window.geometries[-1] == "+100+100"


def test_labels_name_the_covering_windows():
    assert {POPUP, STATUS} == set(main_module.COVERING_WINDOWS)


def test_side_spot_message_moves_the_popup_and_home_brings_it_back(monkeypatch):
    monkeypatch.setattr(
        main_module, "get_backend", lambda: type("B", (), {"tk_scale": 1.0})
    )
    monkeypatch.setattr(main_module, "get_screen_resolution", lambda: SCREEN)
    app = Application.__new__(Application)
    app.overlay = _overlay(monkeypatch, viewable=True)
    app._popup_corner = CORNER  # noqa: SLF001
    app._popup_spot = POPUP_HOME  # noqa: SLF001

    app._apply_popup_spot(POPUP_LEFT, Rect(742, 1493, 1278, 2009))  # noqa: SLF001
    assert app.overlay.window.geometries == ["+742+1493"]

    app._apply_popup_spot(POPUP_HOME, None)  # noqa: SLF001
    assert app.overlay.window.geometries[-1] == f"+{HOME_RECT.left}+{HOME_RECT.top}"
