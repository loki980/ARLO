"""
ARLO - Main Application.
Coordinates OCR scanning and overlay display.
"""

import argparse
import datetime
import inspect
import json
import queue
import sys
import threading
import time
import tkinter as tk
import traceback
from collections import deque
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from enum import auto
from pathlib import Path
from threading import Thread
from typing import NamedTuple

import numpy as np
from dotenv import load_dotenv

from arc_helper import layout_home
from arc_helper.advice_card import AdviceCard
from arc_helper.advice_card import build_advice_card
from arc_helper.advice_card import needed_parts
from arc_helper.advice_window import AdviceWindow
from arc_helper.badge_surface import Badge
from arc_helper.badge_surface import BadgeSurface
from arc_helper.badge_surface import Border
from arc_helper.badge_surface import Patch
from arc_helper.badge_surface import Tag
from arc_helper.cell_border import CellBorder
from arc_helper.clickthrough import make_click_through
from arc_helper.collections_screen import read_collections
from arc_helper.collections_screen import record_collections
from arc_helper.config import APP_DIR
from arc_helper.config import DEFAULT_CAPTURE_LIMIT
from arc_helper.config import SettingsManager
from arc_helper.config import capture_library_in_env
from arc_helper.config import get_screen_resolution
from arc_helper.config import get_settings
from arc_helper.config import logger
from arc_helper.database import FIELD
from arc_helper.database import HOME
from arc_helper.database import Database
from arc_helper.database import Item
from arc_helper.database import RecycleOutput
from arc_helper.database import get_database
from arc_helper.grid import BACKPACK
from arc_helper.grid import EQUIPMENT
from arc_helper.grid import STASH
from arc_helper.grid import STILL_MEAN
from arc_helper.grid import STILL_SCALE
from arc_helper.grid import find_slots
from arc_helper.grid import frames_still
from arc_helper.grid import scroll_offset
from arc_helper.grid import slot_diffs
from arc_helper.grid import still_gray
from arc_helper.hotkey import HotkeyMonitor
from arc_helper.hotkey import get_hotkey_monitor
from arc_helper.hovered_cell import find_hovered_cell
from arc_helper.hovered_cell import snap_to_slot
from arc_helper.icon_embed import EmbedGallery
from arc_helper.icon_embed import build_gallery
from arc_helper.icon_embed import default_embedder
from arc_helper.inventory_header import read_loadout_header
from arc_helper.inventory_header import read_stash_header
from arc_helper.inventory_ledger import Owned
from arc_helper.inventory_ledger import loadout_tally
from arc_helper.inventory_ledger import owned_counts
from arc_helper.ocr import CardSignature
from arc_helper.ocr import OCREngine
from arc_helper.ocr import OCREngineManager
from arc_helper.ocr import clamp_bbox
from arc_helper.ocr import get_ocr_engine
from arc_helper.ocr import grab_screen
from arc_helper.ocr import same_card
from arc_helper.overlay import RECYCLE_LABEL_HOME_ONLY
from arc_helper.overlay import RECYCLE_LABEL_HOME_RECYCLER
from arc_helper.overlay import OverlayWindow
from arc_helper.overlay import StatusWindow
from arc_helper.overlay import TooltipChips
from arc_helper.overlay import window_rect_phys
from arc_helper.resolution_profiles import get_profile_manager
from arc_helper.screen import ScreenCaptureUnavailable
from arc_helper.screen import get_backend
from arc_helper.slot_marks import learned_check
from arc_helper.slot_marks import read_stack
from arc_helper.stash_map import FIRST
from arc_helper.stash_map import LOOKS
from arc_helper.stash_map import SCROLL
from arc_helper.stash_map import StashMapStore
from arc_helper.stash_scan import SlotCacheSaver
from arc_helper.stash_scan import SlotItem
from arc_helper.stash_scan import badges_for
from arc_helper.stash_scan import distinct_gallery
from arc_helper.stash_scan import identify_slots
from arc_helper.stash_scan import load_slot_cache
from arc_helper.stash_scan import peek_slots
from arc_helper.stash_scan import slot_cache_fingerprint
from arc_helper.stash_scan import verdicts_for
from arc_helper.tooltip_layout import Point
from arc_helper.tooltip_layout import Rect
from arc_helper.tooltip_layout import to_screen
from arc_helper.verdict import KEEP
from arc_helper.verdict import KEEP_SOME
from arc_helper.verdict import Verdict
from arc_helper.verdict import compute_verdict
from arc_helper.verdict import is_blueprint

load_dotenv(Path(__file__).with_name(".env"), override=False)

# The popup and chips hide this long after the last show or touch of an
# item: twice the slowest of the last three scan periods and the scan just
# finished, capped, and never less than display_time, so a hovered item stays
# shown until the next scan sees its card.
HOLD_SCAN_PERIODS = 2
HOLD_HISTORY = 3
HOLD_CAP_S = 6.0
# The hold is only a safety net for a stalled scanner: the scan loop hides
# the windows itself once the card is gone, so the hold is never shorter
HOLD_FLOOR_S = 3.0

# A shown card missing from this many cycles in a row hides the popup, chips
# and frame at once
CARD_GONE_CYCLES = 2
# A cycle without a card sleeps at most this long before the next one. A
# cycle that found a card does not sleep at all
NO_CARD_SLEEP_CAP_S = 0.1
# Seconds between INVENTORY trigger rechecks while ACTIVE
TRIGGER_RECHECK_S = 1.0
# Seconds between looks for the Collections screen (BLUEPRINTS tab) while IDLE
COLLECTIONS_CHECK_S = 2.0
# ACTIVE ends on a lost trigger only once the rechecks have missed for this
# many seconds in a row (and on at least two rechecks). One hit resets it
TRIGGER_LOST_S = 2.0
# While the same card stays up its windows are touched at most this often,
# well inside HOLD_FLOOR_S
TOUCH_EVERY_S = 1.0

# Gold frame around the hovered cell of an identified item, in Tk units scaled the
# way the chips scale
BORDER_WIDTH_TK = 6


def scan_hold_s(
    display_time: float, periods: Sequence[float], estimate_s: float
) -> float:
    """Seconds the popup and chips stay up after a read.

    periods are the scan periods measured in this ACTIVE stretch, oldest
    first. estimate_s is the just-finished scan's duration plus the scan
    interval, which counts as a period too, so a slow read after fast empty
    scans, or the first scan of the stretch, still gets a long enough hold.
    """
    period = max([*list(periods)[-HOLD_HISTORY:], estimate_s])
    return max(display_time, min(HOLD_SCAN_PERIODS * period, HOLD_CAP_S))


class OverlayRects:
    """ARLO's own window rectangles, shared from the Tk thread to the scanner.

    The Tk thread publishes the physical screen rectangles of every mapped
    ARLO window, the scan thread reads a snapshot to keep them out of the
    hovered cell search. Neither side touches Tk through this object.

    The labelled rects name the windows a read must not be covered by
    (POPUP and STATUS, only while mapped) and the popup's two spots
    (POPUP_HOME and POPUP_CORNER, sized like the popup), so the scanner can
    tell them apart from the chips and the gold frame, which sit on the card
    by design. popup_away is True while the popup sits at any spot but home
    (beside or below the card, or the corner).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rects: tuple[Rect, ...] = ()
        self._labelled: dict[str, Rect] = {}
        self._popup_away = False

    def publish(
        self,
        rects: Sequence[Rect],
        labelled: Mapping[str, Rect] | None = None,
        *,
        popup_away: bool = False,
    ) -> None:
        with self._lock:
            self._rects = tuple(rects)
            self._labelled = dict(labelled or {})
            self._popup_away = popup_away

    def snapshot(self) -> tuple[Rect, ...]:
        with self._lock:
            return self._rects

    def labelled(self) -> dict[str, Rect]:
        with self._lock:
            return dict(self._labelled)

    def popup_away(self) -> bool:
        with self._lock:
            return self._popup_away


# Labels of the OverlayRects windows and spots
POPUP = "popup"
STATUS = "status"
POPUP_HOME = "popup_home"
POPUP_CORNER = "popup_corner"
# Spots beside or below the item card the popup moves to while the card
# covers home
POPUP_LEFT = "popup_left"
POPUP_RIGHT = "popup_right"
POPUP_BELOW = "popup_below"
# ARLO windows that make a read unusable when they cover the card
COVERING_WINDOWS = (POPUP, STATUS)
# Gap between the item card and an escape spot beside or below it, as a
# fraction of the screen height (22 px at 2160)
ESCAPE_GAP = 0.01
# Seconds between looks for the weapon slots that place home
HOME_REFRESH_S = 1.0


def rects_intersect(a: Rect, b: Rect) -> bool:
    """True when two rectangles share any pixel (right and bottom exclusive)."""
    return (
        a.left < b.right and b.left < a.right and a.top < b.bottom and b.top < a.bottom
    )


def covering_window(card: Rect, labelled: Mapping[str, Rect]) -> str | None:
    """Label of the first ARLO window covering the card, else None."""
    for label in COVERING_WINDOWS:
        rect = labelled.get(label)
        if rect is not None and rects_intersect(card, rect):
            return label
    return None


def _on_screen(rect: Rect, screen: tuple[int, int]) -> bool:
    return (
        rect.left >= 0
        and rect.top >= 0
        and rect.right <= screen[0]
        and rect.bottom <= screen[1]
    )


def popup_escape_spot(
    card: Rect, labelled: Mapping[str, Rect], screen: tuple[int, int]
) -> tuple[str, Rect] | None:
    """Spot and rect the popup moves to so it stops covering the card.

    Candidates are rects the size of the popup at home (the labelled home
    spot): home itself, left and right of the card at home's height, below
    the card at home's left, then the corner spot. Left and right go in
    order of the smaller horizontal move from home, left on a tie. The first
    candidate that stays on screen and clear of the card wins, None when
    none does. The gap to the card is ESCAPE_GAP of the screen height.
    """
    home = labelled.get(POPUP_HOME)
    if home is None:
        return None
    width = home.right - home.left
    height = home.bottom - home.top
    gap = round(screen[1] * ESCAPE_GAP)

    def at(x: int, y: int) -> Rect:
        return Rect(x, y, x + width, y + height)

    beside = [
        (POPUP_LEFT, at(card.left - gap - width, home.top)),
        (POPUP_RIGHT, at(card.right + gap, home.top)),
    ]
    beside.sort(key=lambda spot: abs(spot[1].left - home.left))
    candidates = [(POPUP_HOME, home), *beside]
    candidates.append((POPUP_BELOW, at(home.left, card.bottom + gap)))
    corner = labelled.get(POPUP_CORNER)
    if corner is not None:
        candidates.append((POPUP_CORNER, corner))
    for label, rect in candidates:
        if _on_screen(rect, screen) and not rects_intersect(rect, card):
            return label, rect
    return None


def home_is_free(card: Rect | None, labelled: Mapping[str, Rect]) -> bool:
    """True when the popup at home would leave the card (if any) uncovered."""
    if card is None:
        return True
    home = labelled.get(POPUP_HOME)
    return home is not None and not rects_intersect(card, home)


# The fixed home's fractions (see layout_home), kept for the demo scripts
POPUP_HOME_LEFT = layout_home.LEFT
POPUP_HOME_TOP = layout_home.TOP
POPUP_HOME_FLOOR = layout_home.FLOOR


# Home moves only when an edge moves by more than this fraction of the
# screen height (4 px at 2160), more than the slot finder's jitter
HOME_MOVE = 0.002


def home_moved(old: Rect | None, new: Rect, screen: tuple[int, int]) -> bool:
    """True when the new home box differs from the old one beyond jitter."""
    if old is None:
        return True
    limit = screen[1] * HOME_MOVE
    return any(abs(a - b) > limit for a, b in zip(old, new, strict=True))


def popup_spot_rects(
    screen: tuple[int, int],
    size: tuple[int, int],
    corner: Point,
    home: Rect | None = None,
) -> tuple[Rect, Rect]:
    """The popup's home and top left corner spots in physical px.

    screen is the screen size, size the popup's or advice card's size and
    corner the top left spot (OVERLAY_X and OVERLAY_Y). home is the home box
    (layout_home), the fixed LOADOUT box when None. The spot's top left is
    the box's, raised only to keep a taller popup's bottom on the floor.
    """
    if home is None:
        home = layout_home.fixed_home(screen)
    width, height = size
    floor = round(screen[1] * layout_home.FLOOR)
    x = home.left
    y = min(home.top, floor - height)
    return (
        Rect(x, y, x + width, y + height),
        Rect(corner.x, corner.y, corner.x + width, corner.y + height),
    )


def screen_to_capture(rects: Sequence[Rect], origin: Point) -> list[Rect]:
    """Shift physical screen rectangles into a capture's own coordinates."""
    return [
        Rect(
            r.left - origin.x,
            r.top - origin.y,
            r.right - origin.x,
            r.bottom - origin.y,
        )
        for r in rects
    ]


class HoveredSlot(NamedTuple):
    """The hovered slot's gold frame in physical screen px and its check read.

    frame is None when no hovered cell was found. learned is the learned
    blueprint check read on the snapped slot (slot_marks.learned_check),
    None when it was not read or could not be judged. stack is the xN count
    read from the snapped slot (slot_marks.read_stack), None when it was not
    read or the slot shows none.
    """

    frame: Rect | None
    learned: bool | None
    stack: int | None = None


NO_HOVERED_SLOT = HoveredSlot(None, None)


class OwnedCount(NamedTuple):
    """The owned count of one item for the verdict and the advice card.

    sure is False when the count is a lower bound (the stash map is partial).
    """

    count: int
    sure: bool


# _queue_chips locates the frame itself when no frame is passed
LOCATE_FRAME = object()


def hovered_slot_on_screen(
    capture,
    origin: Point | None,
    card: Rect | None,
    ignore_phys: Sequence[Rect],
    *,
    read_check: bool = False,
    read_count: bool = False,
) -> HoveredSlot:
    """Find the hovered slot in a cursor capture, framed in physical screen px.

    capture is the PIL image the card was found in, origin its top left on
    screen and card the item card in capture coordinates. ignore_phys are
    ARLO's own windows in physical screen px. The glow ring box is snapped
    to the slot under it among grid.find_slots in the same capture
    (hovered_cell.snap_to_slot), so the frame covers the whole slot, and
    the ring box stays without a matching slot. With read_check the
    snapped slot's learned blueprint check is read, and with read_count its
    stack count, from the capture already taken. Runs on the scan thread.
    """
    if capture is None or origin is None or card is None:
        return NO_HOVERED_SLOT
    rgb = np.asarray(capture if capture.mode == "RGB" else capture.convert("RGB"))
    start = time.perf_counter()
    cell = find_hovered_cell(rgb, card, screen_to_capture(ignore_phys, origin))
    slot = snap_to_slot(cell, find_slots(rgb)) if cell is not None else None
    learned = None
    if read_check and slot is not None:
        learned = learned_check(rgb, slot.rect)
    stack = None
    if read_count and slot is not None:
        r = slot.rect
        stack = read_stack(rgb[r.top : r.bottom, r.left : r.right])
    logger.debug(
        "border search found=%s slot=%s learned=%s duration=%.1f ms",
        "yes" if cell is not None else "no",
        "yes" if slot is not None else "no",
        learned,
        (time.perf_counter() - start) * 1000,
    )
    if cell is None:
        return NO_HOVERED_SLOT
    frame = slot.rect if slot is not None else cell
    return HoveredSlot(to_screen(frame, origin), learned, stack)


def hovered_cell_on_screen(
    capture, origin: Point | None, card: Rect | None, ignore_phys: Sequence[Rect]
) -> Rect | None:
    """The hovered slot's gold frame in physical screen px, None without one."""
    return hovered_slot_on_screen(capture, origin, card, ignore_phys).frame


# Stash badges: the stack's worth on every identified slot and a need tag and
# outline on needed items, drawn while the inventory grid is still. A pass starts at most once
# per STASH_PASS_GAP_S, unless the last one was dropped as stale. The watcher
# thread wakes at least every STASH_TICK_S.
STASH_PASS_GAP_S = 0.25
STASH_TICK_S = 0.1
# A loadout tally whose counts did not change is saved at most once per
# LOADOUT_SAVE_S, changed counts are saved at once
LOADOUT_SAVE_S = 5.0
# Share of the inventory slots that must change at once (or the slot set
# itself, a scroll or tab switch) to clear every badge. Fewer changed slots
# only hide the badges on them.
BIG_CHANGE_FRACTION = 0.4
# Share of the slots found on one frame only (lone) that clears every badge
# on its own, a changed slot set. Fewer lone slots only hide their own badges,
# find_slots loses a few slots on a still grid now and then
LONE_BIG_FRACTION = 0.25
# A frame with fewer slots than this share of the last frame with a grid has
# no inventory grid: the inventory closed, and the badges clear at once
GRID_GONE_FRACTION = 0.25
# The panels that scroll on their own. A big change inside one of them only,
# by a measured vertical offset, moves its badges with it instead of clearing
SCROLL_PANELS = (STASH, BACKPACK)
# Px around the game's item card the stillness compare skips: the ACTIONS tab
# above the card (about 98 px tall at 3840x2160, 51 px on the 2000 px wide
# jeffrey-popup-corner screenshot) and its shadow. The card stays skipped for
# CARD_LINGER_S after it is gone, while it fades out over a few frames on
# which no card is found.
CARD_MARGIN_PX = 120
CARD_LINGER_S = 0.6
# Px around the game's item card that the stillness compare and the badges
# keep clear of, for the card's shadow
STASH_CARD_MARGIN = 16
# The parts of a slot a drawn badge changes on screen, which the stillness
# compare skips: a tag's band (a share of the slot height, at its corner's
# edge), a patch's whole rect and a band along the edges for a border
STASH_TAG_BAND = 0.4
STASH_BORDER_BAND = 6
# Catalog item types that never sit in an inventory slot, left out of the
# stash gallery (the same list as scripts/eval_stash_scan.py)
STASH_EXCLUDED_TYPES = frozenset({"Outpost Furniture", "Outpost Room", "Design"})
STASH_ITEMS_FILE = "arctracker_items.json"
STASH_GALLERY_FILE = "stash_gallery.npz"


class StashFrame(NamedTuple):
    """One full-screen capture handed to the stash badges.

    image is the capture (a PIL image or an RGB array) and is never mutated
    after the hand off. card is the game's item card and overlays are ARLO's
    own windows, both in physical screen px. cursor is the pointer, in
    capture px, when the screen backend gave it.
    """

    image: object
    card: Rect | None
    overlays: tuple[Rect, ...]
    cursor: Point | None = None


def frame_hidden(frame: StashFrame | None) -> list[Rect]:
    """The parts of a frame no badge may cover: the padded card and ARLO's windows."""
    if frame is None:
        return []
    hidden = list(frame.overlays)
    if frame.card is not None:
        c, m = frame.card, STASH_CARD_MARGIN
        hidden.append(Rect(c.left - m, c.top - m, c.right + m, c.bottom + m))
    return hidden


def grown(rect: Rect, margin: int) -> Rect:
    """A rect grown by margin px on every side."""
    return Rect(
        rect.left - margin, rect.top - margin, rect.right + margin, rect.bottom + margin
    )


def frame_still_ignore(frame: StashFrame | None) -> list[Rect]:
    """The parts of a frame the stillness compare skips: ARLO's windows and the card.

    The card is grown by CARD_MARGIN_PX, for its ACTIONS tab and shadow.
    """
    if frame is None:
        return []
    ignore = list(frame.overlays)
    if frame.card is not None:
        ignore.append(grown(frame.card, CARD_MARGIN_PX))
    return ignore


def badge_marks(badge: Mapping) -> list[Rect]:
    """The parts of the screen a drawn badge changes."""
    left, top, right, bottom = (int(v) for v in badge["rect"])
    if badge["kind"] == "patch":
        # The patch rect is the bar itself, all of it changes on screen
        return [Rect(left, top, right, bottom)]
    if badge["kind"] == "border":
        band = STASH_BORDER_BAND
        return [
            Rect(left, top, right, top + band),
            Rect(left, bottom - band, right, bottom),
            Rect(left, top, left + band, bottom),
            Rect(right - band, top, right, bottom),
        ]
    band = round((bottom - top) * STASH_TAG_BAND)
    if str(badge.get("corner", "bl")).startswith("t"):
        return [Rect(left, top, right, top + band)]
    return [Rect(left, bottom - band, right, bottom)]


def _centre_in(rect: Rect, region: Rect) -> bool:
    x, y = (rect.left + rect.right) / 2, (rect.top + rect.bottom) / 2
    return region.left <= x < region.right and region.top <= y < region.bottom


def _near(a: Rect, b: Rect) -> bool:
    """True when two slot rects are the same slot, within half a STILL_SCALE cell."""
    reach = STILL_SCALE // 2
    return all(abs(p - q) <= reach for p, q in zip(a, b, strict=True))


def _within(rect: Rect, region: Rect) -> bool:
    return (
        region.left <= rect.left
        and rect.right <= region.right
        and region.top <= rect.top
        and rect.bottom <= region.bottom
    )


def followed_rect(rect: Rect, region: Rect, dy: int) -> Rect:
    """rect moved down by dy when its centre lies inside region, else rect."""
    if not _centre_in(rect, region):
        return rect
    return Rect(rect.left, rect.top + dy, rect.right, rect.bottom + dy)


def follow_badges(badges: Sequence[dict], region: Rect, dy: int) -> list[dict]:
    """The badges after the content of region scrolled down by dy px.

    A badge whose rect's centre lies inside region moves by dy, and is
    dropped when its moved rect is no longer wholly inside region (a full
    draw cannot clip it). Other badges stay as they are.
    """
    kept = []
    for b in badges:
        rect = Rect(*(int(v) for v in b["rect"]))
        moved = followed_rect(rect, region, dy)
        if moved is rect:
            kept.append(b)
        elif _within(moved, region):
            kept.append({**b, "rect": tuple(moved)})
    return kept


def badges_clear_of(badges: Sequence[dict], hidden: Sequence[Rect]) -> list[dict]:
    """The badges whose slot meets none of the hidden rects."""
    return [
        b
        for b in badges
        if not any(rects_intersect(Rect(*b["rect"]), h) for h in hidden)
    ]


def surface_badges(badges: Sequence[dict]) -> list[Badge]:
    """Badge dicts (stash_scan.badges_for) as the badge surface's Border, Tag and Patch."""
    made: list[Badge] = []
    for b in badges:
        fields = {**b, "rect": tuple(int(v) for v in b["rect"])}
        if b["kind"] == "border":
            made.append(Border(**fields))
        elif b["kind"] == "patch":
            made.append(Patch(**{**fields, "colour": tuple(b["colour"])}))
        else:
            if b.get("outline") is not None:
                fields["outline"] = tuple(b["outline"])
            made.append(Tag(**fields))
    return made


def free_rects(shape: Sequence[int], ignore: Sequence[Rect]) -> list[Rect]:
    """Rects covering a frame of this shape outside the ignore rects.

    The rects are runs within rows of STILL_SCALE px, the grain frames_still
    compares at, so passing them as its slots compares all of the frame but
    the ignored parts.
    """
    return _cell_runs(shape, None, ignore)


def _cell_runs(
    shape: Sequence[int], keep: Sequence[Rect] | None, ignore: Sequence[Rect]
) -> list[Rect]:
    """Row runs of STILL_SCALE cells inside keep (all when None), outside ignore.

    A cell partly in an ignore rect is dropped, as is a cell partly outside
    every keep rect.
    """
    height, width = shape[:2]
    scale = STILL_SCALE
    rows, cols = -(-height // scale), -(-width // scale)
    if keep is None:
        free = np.ones((rows, cols), bool)
    else:
        # Only cells wholly inside a rect, so no pixel outside it is compared
        free = np.zeros((rows, cols), bool)
        for r in keep:
            free[
                max(0, -(-r.top // scale)) : max(0, r.bottom // scale),
                max(0, -(-r.left // scale)) : max(0, r.right // scale),
            ] = True
    for r in ignore:
        free[
            max(0, r.top // scale) : max(0, -(-r.bottom // scale)),
            max(0, r.left // scale) : max(0, -(-r.right // scale)),
        ] = False
    rects = []
    for row in range(rows):
        line = free[row]
        if not line.any():
            continue
        top, bottom = row * scale, min(height, (row + 1) * scale)
        edges = np.flatnonzero(np.diff(np.concatenate(([0], line.view(np.int8), [0]))))
        rects.extend(
            Rect(int(start) * scale, top, min(width, int(end) * scale), bottom)
            for start, end in zip(edges[::2], edges[1::2], strict=True)
        )
    return rects


def stash_still(
    a: np.ndarray,
    b: np.ndarray,
    ignore: Sequence[Rect],
    still: Callable[..., bool] = frames_still,
    slots: Callable[[np.ndarray], Sequence[Rect]] | None = None,
) -> bool:
    """frames_still inside the inventory slots of either frame, outside ignore.

    The slots are those grid.find_slots finds on a and on b (slots, when
    given, returns their rects for a frame instead), so the game world and
    anything else outside the grid never count as movement. When every slot
    pixel is ignored the frames count as still. Without a slot on either
    frame the whole frame outside the ignore rects is compared, and a
    wholly ignored frame counts as still. An injected still that takes no
    rects is called as still(a, b) and compares as it likes.
    """
    if a.shape != b.shape:
        return False
    if slots is None:
        found = [s.rect for s in (*find_slots(a), *find_slots(b))]
    else:
        found = [*slots(a), *slots(b)]
    if found:
        rects = _cell_runs(a.shape, found, ignore)
    elif not ignore:
        return still(a, b)
    else:
        rects = free_rects(a.shape, ignore)
    if not rects:
        return True
    return still(a, b, rects) if _takes_rects(still) else still(a, b)


def _takes_rects(still: Callable[..., bool]) -> bool:
    """True when still can be called as still(a, b, rects)."""
    try:
        inspect.signature(still).bind(None, None, None)
    except TypeError:
        return False
    except ValueError:
        # No signature to inspect (a builtin), so assume the full form
        return True
    return True


class GridChange(NamedTuple):
    """How the inventory grid changed between two frames.

    big is a change that clears every badge. changed are the slot rects
    that changed on their own, slots the number of slots compared and lone
    the number found on one frame only.
    """

    big: bool
    changed: tuple[Rect, ...] = ()
    slots: int = 0
    lone: int = 0


def stash_changes(
    a: np.ndarray,
    b: np.ndarray,
    ignore: Sequence[Rect],
    still: Callable[..., bool] = frames_still,
    slots: Callable[[np.ndarray], Sequence[Rect]] | None = None,
    gray: Callable[[np.ndarray], np.ndarray] | None = None,
) -> GridChange:
    """The slots that changed between two frames, each compared on its own.

    The slots are those grid.find_slots finds on a and on b (slots, when
    given, returns their rects for a frame instead), matched by rect within
    half a STILL_SCALE cell. A matched slot changed when its own mean
    difference outside ignore reaches STILL_MEAN (grid.slot_diffs). A slot
    found on one frame only changed too, unless ignore covers all of it and
    it may only be hidden. The change is big when at least
    BIG_CHANGE_FRACTION of the slots changed (lone ones included), or when at
    least LONE_BIG_FRACTION of the slots are lone, so the slot set itself
    changed (a scroll, a tab switch, the inventory layout). Fewer lone slots
    are only changed slots. Without a slot on either frame, or with an
    injected still other than frames_still, the whole compare of
    stash_still decides: not still is big. gray, when given, returns the
    grey copy of a frame (a cached grid.still_gray).
    """
    if a.shape != b.shape:
        return GridChange(big=True)
    find = slots or (lambda rgb: [s.rect for s in find_slots(rgb)])
    if still is not frames_still:
        return GridChange(big=not stash_still(a, b, ignore, still, find))
    in_a, in_b = list(find(a)), list(find(b))
    if not in_a and not in_b:
        return GridChange(big=not stash_still(a, b, ignore, still, lambda _rgb: []))
    pairs, alone = _match_slots(in_a, in_b)
    if gray is not None:
        a, b = gray(a), gray(b)
    means = slot_diffs(a, b, pairs, ignore)
    changed = [
        r
        for r, m in zip(pairs, means, strict=True)
        if m is not None and m >= STILL_MEAN
    ]
    # A lone slot wholly under ignore rects may only be covered on one frame.
    # Others changed: find_slots lost a darkened or lit slot, or a few slots
    # on a still grid, or the slot set changed when many are lone
    seen = slot_diffs(a, b, alone, ignore)
    lone = [r for r, m in zip(alone, seen, strict=True) if m is not None]
    changed += lone
    total = len(pairs) + len(alone)
    return GridChange(
        big=len(changed) >= BIG_CHANGE_FRACTION * total
        or len(lone) >= LONE_BIG_FRACTION * total,
        changed=tuple(changed),
        slots=total,
        lone=len(lone),
    )


def _match_slots(a: Sequence[Rect], b: Sequence[Rect]) -> tuple[list[Rect], list[Rect]]:
    """The rects of a matched in b (as a has them) and those of either unmatched."""
    reach = STILL_SCALE // 2
    left = list(b)
    pairs, alone = [], []
    for r in a:
        hit = next(
            (
                i
                for i, s in enumerate(left)
                if all(abs(p - q) <= reach for p, q in zip(r, s, strict=True))
            ),
            None,
        )
        if hit is None:
            alone.append(r)
        else:
            pairs.append(r)
            del left[hit]
    return pairs, alone + left


def _frame_rgb(image) -> np.ndarray:
    """An RGB array of a capture, without copying one that already is."""
    if isinstance(image, np.ndarray):
        return image
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.asarray(image)


def load_stash_gallery(
    cache_dir: Path,
) -> tuple[EmbedGallery, dict[str, tuple[str, ...]]]:
    """The stash pass's gallery and twins from the icon cache.

    Every item in the cached arctracker items list outside
    STASH_EXCLUDED_TYPES whose icon is cached, embedded once and kept in
    STASH_GALLERY_FILE, cut to one entry per distinct icon. Raises when the
    list, the icons or the model are missing.
    """
    listing = json.loads((cache_dir / STASH_ITEMS_FILE).read_text(encoding="utf-8"))
    icons = {}
    for item in listing["items"]:
        if item.get("type") in STASH_EXCLUDED_TYPES:
            continue
        path = cache_dir / f"{item['id']}.png"
        if path.is_file():
            icons[item["id"]] = (item["name"]["en"], path)
    if not icons:
        msg = f"no item icons in {cache_dir}"
        raise FileNotFoundError(msg)
    gallery = build_gallery(icons, default_embedder(), cache_dir / STASH_GALLERY_FILE)
    return distinct_gallery(gallery)


def stash_identifier(
    db: Database, cache_dir: Path, *, warm: bool = False
) -> Callable[[np.ndarray], list[SlotItem]]:
    """identify_slots on the catalog gallery, with a slot cache as long lived.

    The gallery loads on the first call, or with warm on a background
    thread started here that also runs one blank embed, so the first pass
    does not wait for the model. A call made while that thread runs waits
    for it. The SlotCache is loaded with the gallery from db's user database
    (stash_scan.load_slot_cache), for the fingerprint of this gallery, its
    embedding file and the catalog, and kept with it. After a pass that kept
    new entries a stash_scan.SlotCacheSaver saves it in the background, at
    once the first time, then at most every SLOT_CACHE_SAVE_S.

    The callable has a peek(rgb) attribute: the SlotItems of the slots whose
    look is in the cache already (stash_scan.peek_slots), misses left out.
    peek never embeds, reads tiers or stacks or changes the cache, and finds
    nothing before the gallery has loaded. Its flush() attribute saves what
    is unsaved and waits a bounded time for the save (StashBadges.stop).
    """
    loaded: list = []
    lock = threading.Lock()

    def load() -> tuple:
        with lock:
            if not loaded:
                gallery, twins = load_stash_gallery(cache_dir)
                try:
                    size = (cache_dir / STASH_GALLERY_FILE).stat().st_size
                except OSError:
                    size = 0
                fingerprint = slot_cache_fingerprint(gallery.names, size, db.count())
                cache = load_slot_cache(db, fingerprint)
                logger.info("slot cache: loaded %d entries", len(cache))
                saver = SlotCacheSaver(
                    lambda rows: db.save_slot_cache(fingerprint, rows)
                )
                loaded.extend((gallery, twins, cache, saver))
            return tuple(loaded)

    def warm_up() -> None:
        start = time.perf_counter()
        try:
            load()
            default_embedder().warm()
        except Exception:  # noqa: BLE001 - the first pass loads and reports it
            logger.warning("stash model warm-up failed", exc_info=True)
            return
        logger.info(
            "stash model warmed in %.0f ms", (time.perf_counter() - start) * 1000
        )

    def identify(rgb: np.ndarray) -> list[SlotItem]:
        gallery, twins, cache, saver = load()
        found = identify_slots(
            rgb, gallery, db.lookup, twins=twins, read_stacks=True, cache=cache
        )
        saver.after_pass(cache)
        return found

    def peek(rgb: np.ndarray) -> list[SlotItem]:
        if len(loaded) < 4:
            return []
        return peek_slots(rgb, loaded[2], read_stacks=True)

    def flush() -> bool:
        if len(loaded) < 4:
            return True
        return loaded[3].flush()

    identify.peek = peek
    identify.flush = flush
    if warm:
        Thread(target=warm_up, name="stash-warm", daemon=True).start()
    return identify


def stash_badge_maker(
    db: Database,
    show_ended: Callable[[], bool],
    owned: Callable[[], Owned | None] | None = None,
) -> Callable[[list[SlotItem]], list[dict]]:
    """badges_for with the verdicts of the moment, as the popup decides them.

    owned gives the owned totals of the moment and whether they are sure
    (StashLedger.owned), None to count the slots on screen.
    """

    def make(slot_items: list[SlotItem]) -> list[dict]:
        verdicts = verdicts_for(db, show_ended=show_ended())
        totals = owned() if owned else None
        if totals is None:
            return badges_for(slot_items, verdicts)
        return badges_for(slot_items, verdicts, totals.counts, sure=totals.sure)

    return make


class StashLedger:
    """The stored stash map and the latest loadout, kept current from the passes.

    after_pass takes each pass's frame, its find_slots slots and its slot
    items. A frame with stash slots goes to the StashMapStore with its STASH
    header, one DEBUG line per observe. A frame with loadout slots and a
    readable LOADOUT header gives the latest loadout tally, saved with
    db.save_loadout at once when its counts changed, else at most once per
    LOADOUT_SAVE_S on clock. An error is logged and never raised.

    owned is the stored map's counts plus the latest loadout's, with whether
    the map is sure (inventory_ledger.owned_counts), None while the map is
    missing. The map and the loadout saved before are used from the start.
    """

    def __init__(
        self,
        db: Database,
        *,
        store: StashMapStore | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._db = db
        self._clock = clock
        self._lock = threading.Lock()
        self._store = store if store is not None else StashMapStore(db)
        try:
            self._loadout: dict[str, int] = db.loadout_counts()
        except Exception:  # noqa: BLE001 - a missing loadout only drops it from the totals
            logger.warning("loadout counts not loaded", exc_info=True)
            self._loadout = {}
        # The counts last saved and when, on clock
        self._saved: tuple[dict[str, int], float] | None = None
        self._totals = self._combined()

    def owned(self) -> Owned | None:
        """The combined owned totals and their sureness, None to fall back to the slots on screen."""
        with self._lock:
            return self._totals

    def after_pass(
        self, rgb: np.ndarray, slots: Sequence, slot_items: list[SlotItem]
    ) -> None:
        """Observe the stash part and tally the loadout part of a pass's frame."""
        stash = [s for s in slots if s.panel == STASH]
        if stash:
            try:
                self._observe(rgb, stash, slot_items)
            except Exception:  # noqa: BLE001 - the ledger must never stop a pass
                logger.warning("stash map observe failed", exc_info=True)
        if any(s.panel != STASH for s in slots):
            try:
                self._tally(rgb, slots, slot_items)
            except Exception:  # noqa: BLE001 - the ledger must never stop a pass
                logger.warning("loadout tally failed", exc_info=True)
        totals = self._combined()
        with self._lock:
            self._totals = totals

    def _combined(self) -> Owned | None:
        try:
            sure = self._store.status().sure
            return owned_counts(self._store.map, self._loadout, sure=sure)
        except Exception:  # noqa: BLE001 - fall back to the slots on screen
            logger.warning("owned totals failed", exc_info=True)
            return None

    def _observe(
        self, rgb: np.ndarray, stash: list, slot_items: list[SlotItem]
    ) -> None:
        header = read_stash_header(rgb, stash)
        items = [s for s in slot_items if s.panel == STASH]
        seen = self._store.observe(rgb, items, header)
        status = self._store.status()
        logger.debug(
            "stash map observe placed=%s changed=%d filled=%d header=%s sure=%s",
            "yes" if seen.how in {FIRST, SCROLL, LOOKS} else "no",
            len(seen.changed),
            status.filled,
            f"{header.used}/{header.capacity}" if header else "none",
            "yes" if status.sure else "no",
        )

    def _tally(
        self, rgb: np.ndarray, slots: Sequence, slot_items: list[SlotItem]
    ) -> None:
        header = read_loadout_header(rgb, [s for s in slots if s.panel == EQUIPMENT])
        if header is None:
            return
        tally = loadout_tally(slot_items, header)
        self._loadout = tally.totals
        now = self._clock()
        saved = self._saved
        if (
            saved is None
            or saved[0] != tally.totals
            or now - saved[1] >= LOADOUT_SAVE_S
        ):
            self._db.save_loadout(tally)
            self._saved = (tally.totals, now)


def _latency_from(moved: float) -> dict[str, float | None]:
    """A new StashBadges latency record from the last changed frame's feed time."""
    return {"moved": moved, "still": None, "drawn": None, "pass": None}


class StashBadges:
    """Badges on every stash slot, drawn while the inventory grid is still.

    The scan thread hands each full-screen capture to feed(), which only
    keeps a reference, so the hover path never waits here. A watcher daemon
    thread compares each capture with the one before and with the frame of
    the drawn badges, slot by slot (stash_changes), outside the game's card,
    ARLO's windows, the drawn badges and the slot under the cursor (the
    game brightens it on hover, its badges stay drawn). A big change
    (BIG_CHANGE_FRACTION of the slots, or a changed slot set) waits one
    frame: the badges clear only when the next frame is still big against
    the frame before the change, with any card located on either frame or
    lingering skipped, so a card fading in before the scanner locates it
    clears nothing. A smaller one hides only the badges on the changed
    slots, once the next frame shows them changed too, which come back when their slots look as they did in the badges'
    frame again, or as soon as peek knows a changed slot's new look, from
    the slot cache before the pass. The card, grown by CARD_MARGIN_PX, stays
    skipped for CARD_LINGER_S after it is gone, while it fades out.

    A change that is big inside one of SCROLL_PANELS only, whose content
    moved by an offset grid.scroll_offset measures, is a scroll and clears
    nothing: that panel's badges, their marks and the parts the last pass
    could not see move with it (follow_badges), those scrolled out of its
    viewport are dropped, and the surface is told to translate them, one
    small message instead of a full draw. The slots peek knows among those
    scrolled in are drawn at once, and the next pass renews the whole
    viewport. Any other big change still clears.

    A still pair starts a pass on the pass daemon thread when one is needed
    and none is running, at most once per gap_s. A finished pass keeps its
    badges for the slots unchanged since its frame and drops those on
    changed slots, or all of them after a big change. A pass that finds no
    badge for a drawn slot whose look did not change keeps the drawn ones
    there. Badges never cover the card or ARLO's windows, and come back
    when those move away.

    A pass made while the card covered part of the grid has no badges
    there. Once the card is gone a second pass fills that part in and keeps
    the rest. A pass dropped as stale does not hold the next one back by
    gap_s.

    On the first still frame after a clear, before its pass runs, peek
    (identify.peek when not given) returns the slots already in the slot
    cache and their badges are drawn at once. The pass then replaces them
    with its own.

    A frame with no inventory grid (find_slots finds fewer than
    GRID_GONE_FRACTION of the slots of the last frame with one) clears at
    once, without the one-frame wait ("grid_gone"), and the frames after it
    are skipped until the grid is back. On the first frame with a grid at
    the start, after a grid_gone clear or after a clear from outside (the
    inventory closed), the known slots are drawn from peek on that frame
    itself, without waiting for a still pair. The pass on the next still
    pair confirms them as after any clear.

    After the pass that follows a clear or a slot hide, one DEBUG line
    "stash latency" gives, on clock, the time from the last changed frame
    fed to the first still one (moved_to_still), the pass's identify and
    badge making (pass), from the first still frame to the first draw
    (still_to_drawn, known slots or pass) and from the last changed frame to
    that draw (total).

    identify turns an RGB frame into SlotItems (stash_scan.identify_slots),
    make_badges turns those into badge dicts (stash_scan.badges_for).
    surface is a BadgeSurface, started on the first draw. clock gives the
    time in seconds for the pass gap and the card linger. With threaded
    False no thread starts and step() does one watcher round, running a
    started pass inline (for tests and the replay script).
    """

    def __init__(
        self,
        identify: Callable[[np.ndarray], list[SlotItem]],
        make_badges: Callable[[list[SlotItem]], list[dict]],
        surface,
        *,
        keep_border: bool = True,
        still: Callable[..., bool] = frames_still,
        gap_s: float = STASH_PASS_GAP_S,
        tick_s: float = STASH_TICK_S,
        threaded: bool = True,
        clock: Callable[[], float] = time.monotonic,
        peek: Callable[[np.ndarray], list[SlotItem]] | None = None,
        ledger: StashLedger | None = None,
    ) -> None:
        self._identify = identify
        self._ledger = ledger
        self._peek = peek if peek is not None else getattr(identify, "peek", None)
        self._make_badges = make_badges
        self._surface = surface
        self._keep_border = keep_border
        self._still = still
        self._gap_s = gap_s
        self._tick_s = tick_s
        self._threaded = threaded
        self._clock = clock
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._job_ready = threading.Event()
        self._threads: list[Thread] = []
        self._stopping = False
        self._disabled = False
        self._surface_started = False
        # Bumped by every clear, so a pass or compare from before is stale
        self._generation = 0
        # Newest capture not yet compared, the one before it (RGB and frame)
        # and the newest compared one, for the stale check and the badge hiding
        self._latest: StashFrame | None = None
        self._latest_at = 0.0
        self._prev: tuple[np.ndarray, StashFrame] | None = None
        self._newest: tuple[np.ndarray, StashFrame] | None = None
        # The last card fed and when, kept as an ignore rect while it fades
        self._card_seen: tuple[Rect, float] | None = None
        # The pass handed over and not yet taken, and the pass state
        self._job: tuple[int, np.ndarray, StashFrame] | None = None
        # find_slots slots and still_gray copies of the RGB frames held
        # above, by id (with the array, so a reused id never hits)
        self._slot_cache: dict[int, tuple[np.ndarray, list]] = {}
        self._gray_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._pass_running = False
        self._last_pass_start: float | None = None
        self._need_pass = True
        # The accepted badges, the frame of their pass, the parts of it that
        # were hidden, the screen parts they change, the slots changed since
        # that frame (their badges hidden) and the badges drawn now
        self._badges: list[dict] = []
        self._have_result = False
        self._ref: tuple[np.ndarray, StashFrame] | None = None
        self._pass_hidden: list[Rect] = []
        self._marks: list[Rect] = []
        self._changed: tuple[Rect, ...] = ()
        # Slots whose badges a partial pass dropped, renewed by the next pass
        # even when they look as in the last pass's frame again
        self._omitted: tuple[Rect, ...] = ()
        self._shown: list[dict] = []
        self._drawn = False
        # Set by a clear: the next still frame draws the known slots first
        self._peek_due = False
        # Set at the start and by a clear other than grid_moved: the next
        # frame with a grid draws the known slots at once
        self._open_due = True
        # The slot count of the last frame with a grid (0 for none since the
        # last such clear), and whether the grid has left since
        self._grid_slots = 0
        self._grid_left = False
        # The viewport of each panel followed since the last clear
        self._viewports: dict[str, Rect] = {}
        # Slots a peek after a scroll looked at since the last pass, moved with it
        self._peeked: list[Rect] = []
        # The frame before a big change and that change's card, while the
        # change waits for the next frame to confirm it
        self._pending: tuple[tuple[np.ndarray, StashFrame], Rect | None] | None = None
        # Badges of changed slots whose new look peek knew, by changed slot
        self._renew: dict[Rect, list[dict]] = {}
        # The slots changed since the badges' frame on the last compare,
        # hidden or waiting for this compare to confirm them
        self._raw_changed: tuple[Rect, ...] = ()
        # The latency since the last clear or slot hide, as clock times
        # (moved, still, drawn) and the pass's seconds, until it is logged
        self._lat: dict[str, float | None] | None = None

    def feed(
        self,
        image,
        card: Rect | None,
        overlays: Sequence[Rect],
        cursor: Point | None = None,
    ) -> None:
        """Hand over the newest full-screen capture. Never waits on a pass.

        cursor is the pointer in capture px, None when unknown.
        """
        if self._disabled or self._stopping:
            return
        with self._lock:
            self._latest = StashFrame(image, card, tuple(overlays), cursor)
            self._latest_at = self._clock()
            if card is not None:
                self._card_seen = (card, self._clock())
            if self._threaded and not self._threads:
                self._threads = [
                    Thread(target=self._watch_loop, name="stash-watch", daemon=True),
                    Thread(target=self._pass_loop, name="stash-pass", daemon=True),
                ]
                for thread in self._threads:
                    thread.start()
        self._wake.set()

    def clear(self, reason: str, detail: str = "", *, opened: bool = True) -> None:
        """Remove the badges at once and drop every pass and compare so far.

        detail is added to the log line after the reason. With opened (a
        clear from outside, or the grid gone) the next frame with a grid
        draws the known slots at once. Without it (a big change of a grid
        still in view) they wait for a still pair.
        """
        with self._lock:
            if opened:
                self._open_due = True
                self._grid_slots = 0
                self._grid_left = False
            busy = self._drawn or self._have_result or self._pass_running
            self._generation += 1
            self._latest = self._prev = self._newest = self._ref = None
            self._slot_cache.clear()
            self._gray_cache.clear()
            self._badges, self._shown, self._marks, self._pass_hidden = [], [], [], []
            self._changed = self._omitted = self._raw_changed = ()
            self._viewports = {}
            self._peeked = []
            self._pending = None
            self._renew = {}
            self._have_result = False
            self._need_pass = True
            self._peek_due = True
            if self._drawn:
                self._drawn = False
                self._surface.clear()
        if busy:
            logger.debug("stash badges cleared reason=%s%s", reason, detail)

    def stop(self) -> None:
        """Clear, end both threads, save the slot cache and stop the surface.

        The save is identify's flush(), when it has one, which waits a
        bounded time for the last queued save.
        """
        self.clear("stopped")
        self._stopping = True
        self._wake.set()
        self._job_ready.set()
        for thread in self._threads:
            thread.join(timeout=STASH_PASS_GAP_S)
        flush = getattr(self._identify, "flush", None)
        if flush is not None:
            flush()
        self._surface.stop()

    def step(self) -> None:
        """One watcher round, with a pass it starts run inline (threaded False)."""
        self._watch_once()
        with self._lock:
            job, self._job = self._job, None
        if job is not None:
            self._run_pass(job)
            self._refresh()

    def _watch_loop(self) -> None:
        while not self._stopping:
            self._wake.wait(self._tick_s)
            self._wake.clear()
            try:
                self._watch_once()
            except Exception as e:  # noqa: BLE001 - badges must never stop ARLO
                logger.error(f"Stash watcher error: {e}")

    def _pass_loop(self) -> None:
        while not self._stopping:
            if not self._job_ready.wait(self._tick_s):
                continue
            self._job_ready.clear()
            with self._lock:
                job, self._job = self._job, None
            if job is not None:
                self._run_pass(job)
                self._wake.set()

    def _watch_once(self) -> None:
        """One watcher round, then the caches keep only held frames."""
        try:
            self._watch_round()
        finally:
            with self._lock:
                self._evict_slots()

    def _lingering(self) -> list[Rect]:
        """The last card, grown, while it is less than CARD_LINGER_S gone (under the lock)."""
        if self._card_seen is None:
            return []
        card, seen = self._card_seen
        if self._clock() - seen >= CARD_LINGER_S:
            return []
        return [grown(card, CARD_MARGIN_PX)]

    def _changes(
        self,
        a: tuple[np.ndarray, StashFrame],
        b: tuple[np.ndarray, StashFrame],
        extra: Sequence[Rect],
    ) -> GridChange:
        """stash_changes of two held frames, outside both cards and windows and extra.

        The slots under either frame's cursor are skipped too.
        """
        ignore = [*self._ignore_of(a, b), *extra]
        return stash_changes(
            a[0], b[0], ignore, self._still, self._slots_of, self._gray_of
        )

    def _ignore_of(
        self, a: tuple[np.ndarray, StashFrame], b: tuple[np.ndarray, StashFrame]
    ) -> list[Rect]:
        """What a compare of two held frames skips: cards, windows, hovered slots."""
        return [
            *frame_still_ignore(a[1]),
            *frame_still_ignore(b[1]),
            *self._hovered(a, b),
        ]

    def _hovered(
        self, a: tuple[np.ndarray, StashFrame], b: tuple[np.ndarray, StashFrame]
    ) -> list[Rect]:
        """The slots of a and b under the cursor of either frame."""
        points = [f[1].cursor for f in (a, b) if f[1].cursor is not None]
        if not points:
            return []
        return [
            r
            for f in (a, b)
            for r in self._slots_of(f[0])
            if any(r.left <= p.x < r.right and r.top <= p.y < r.bottom for p in points)
        ]

    def _hover_left(
        self, ref: tuple[np.ndarray, StashFrame], b: tuple[np.ndarray, StashFrame]
    ) -> bool:
        """True when the cursor of b is off a slot hovered in the badges' frame.

        That slot is never compared with the badges' frame, so a pass renews it.
        """
        was = self._hovered(ref, ref)
        if not was:
            return False
        now = self._hovered(b, b)
        return any(not any(_near(r, c) for c in now) for r in was)

    def _renew_marks(self) -> list[Rect]:
        """The screen parts the badges of renewed slots change (under the lock)."""
        return [m for v in self._renew.values() for b in v for m in badge_marks(b)]

    def _watch_round(self) -> None:
        """Compare the newest capture with the one before and the badges' frame."""
        with self._lock:
            frame, self._latest = self._latest, None
            fed_at = self._latest_at
            generation = self._generation
            prev = self._prev
            ref = self._ref
            pending = self._pending
            extra = [*self._lingering(), *self._marks, *self._renew_marks()]
        if frame is None:
            self._refresh()
            return
        rgb = _frame_rgb(frame.image)
        with self._lock:
            if generation != self._generation:
                return
            self._prev = (rgb, frame)
        if self._grid_round(generation, rgb, frame):
            return
        if prev is None:
            return
        if pending is not None:
            # A big change waits on this frame: compare it with the frame
            # before the change, skipping the card the change came with
            prev, card = pending
            if card is not None:
                extra.append(grown(card, CARD_MARGIN_PX))
        pair = self._changes(prev, (rgb, frame), extra)
        changed: tuple[Rect, ...] = ()
        if (
            not pair.big
            and ref is not None
            and ref[0] is not rgb
            and self._still is frames_still
        ):
            since = self._changes(ref, (rgb, frame), extra)
            if since.big:
                pair = since
            changed = since.changed
        if pair.big or pair.changed:
            scroll = self._scroll_of(prev, (rgb, frame), extra)
            if scroll is not None:
                self._follow(generation, rgb, frame, fed_at, *scroll)
                return
        if pair.big and pending is None:
            # A card fading in or out is gone or located by the next frame
            with self._lock:
                if generation == self._generation:
                    self._pending = (prev, frame.card)
                    # Kept for the compare with the next frame
                    self._prev = prev
            logger.debug(
                "stash big change waits changed=%d slots=%d lone=%d",
                len(pair.changed),
                pair.slots,
                pair.lone,
            )
            return
        if pair.big:
            with self._lock:
                kept = self._slot_cache.get(id(rgb))
            self.clear(
                "grid_moved",
                f" changed={len(pair.changed)} slots={pair.slots} lone={pair.lone}",
                opened=False,
            )
            with self._lock:
                self._prev = (rgb, frame)
                if kept is not None:
                    self._slot_cache[id(rgb)] = kept
                self._lat = _latency_from(fed_at)
            return
        left = ref is not None and self._hover_left(ref, (rgb, frame))
        with self._lock:
            if generation != self._generation:
                return
            if pending is not None:
                self._pending = None
            # A slot newly changed on this frame only hides once the next
            # frame shows it changed too: a card fading in is located by then
            raw = changed
            changed = tuple(
                r for r in raw if r in self._raw_changed or r in self._changed
            )
            self._raw_changed = raw
            if changed and set(changed) != set(self._changed):
                self._lat = _latency_from(fed_at)
            elif self._lat is not None and self._lat["drawn"] is None:
                if pair.changed:
                    self._lat["moved"], self._lat["still"] = fed_at, None
                elif self._lat["still"] is None:
                    self._lat["still"] = fed_at
            peek = not pair.changed and self._peek_due and not self._have_result
            if peek:
                self._peek_due = False
        if peek:
            self._draw_known(generation, rgb, frame)
        start = False
        with self._lock:
            if generation != self._generation:
                return
            self._newest = (rgb, frame)
            if changed and set(changed) != set(self._changed):
                logger.debug(
                    "stash badges hidden slots=%d reason=slot_changed", len(changed)
                )
            self._changed = changed
            # A renewed slot whose look moved on, or that looks as in the
            # badges' frame again, waits for a new peek
            self._renew = {
                r: v
                for r, v in self._renew.items()
                if r in changed and not any(_near(r, c) for c in pair.changed)
            }
            unknown = [
                r
                for r in changed
                if r not in self._renew and not pair.changed and self._peek
            ]
            if self._uncovered(frame) or left:
                self._need_pass = True
            now = self._clock()
            if not pair.changed and self._pass_due(frame, now):
                self._need_pass = False
                self._pass_running = True
                self._last_pass_start = now
                self._job = (generation, rgb, frame)
                start = True
            if (
                self._lat is not None
                and self._lat["still"] is not None
                and not (self._need_pass or self._changed or self._pass_running)
            ):
                # Nothing left to wait for, no pass follows
                self._lat = None
        if start and self._threaded:
            self._job_ready.set()
        if unknown:
            self._renew_slots(generation, rgb, frame, unknown)
        self._refresh()

    def _grid_round(self, generation: int, rgb: np.ndarray, frame: StashFrame) -> bool:
        """Clear when the grid left, draw the known slots when it is back.

        A frame with fewer than GRID_GONE_FRACTION of the slots of the last
        frame with a grid clears at once, the first time, and is not
        compared: True, the round ends here. The first frame with a grid
        after a clear that opened draws the known slots. Only with
        frames_still, an injected still compares whole frames.
        """
        if self._still is not frames_still:
            return False
        found = len(self._found_of(rgb))
        with self._lock:
            if generation != self._generation:
                return True
            had = self._grid_slots
            gone = had > 0 and found < GRID_GONE_FRACTION * had
            left = self._grid_left
            if gone:
                # The first frame with the grid back is not compared with this
                self._prev = None
            elif found:
                self._grid_slots = found
                self._grid_left = False
            opening = not gone and found > 0 and self._open_due
            if opening:
                self._open_due = self._peek_due = False
        if gone:
            if not left:
                self.clear("grid_gone", f" slots={found} last={had}")
                with self._lock:
                    self._grid_left = True
                    self._grid_slots = had
            return True
        if opening:
            self._draw_known(generation, rgb, frame)
        return False

    def _renew_slots(
        self, generation: int, rgb: np.ndarray, frame: StashFrame, slots: list[Rect]
    ) -> None:
        """Badges for changed slots whose new look peek knows, before the pass.

        A slot peek does not know gets none and stays hidden until the pass.
        """
        try:
            slot_items = [
                s for s in self._peek(rgb) if any(_near(s.rect, r) for r in slots)
            ]
            badges = self._make_badges(slot_items) if slot_items else []
        except Exception as e:  # noqa: BLE001 - the pass draws them instead
            logger.warning("stash changed slots not redrawn (%s)", e)
            return
        if not self._keep_border:
            badges = [b for b in badges if b["kind"] != "border"]
        badges = badges_clear_of(badges, frame_hidden(frame))
        renew = {
            r: [b for b in badges if rects_intersect(Rect(*b["rect"]), r)]
            for r in slots
        }
        with self._lock:
            if generation != self._generation:
                return
            for r, made in renew.items():
                if r in self._changed:
                    self._renew[r] = made
        known = [r for r, made in renew.items() if made]
        if known:
            logger.debug(
                "stash changed slots redrawn slots=%d badges=%d",
                len(known),
                sum(len(renew[r]) for r in known),
            )

    def _scroll_of(
        self,
        a: tuple[np.ndarray, StashFrame],
        b: tuple[np.ndarray, StashFrame],
        extra: Sequence[Rect],
    ) -> tuple[str, Rect, int] | None:
        """The panel that scrolled from a to b alone, its viewport and offset.

        The change must be big within the panel's own slots, not big within
        all other slots, and scroll_offset must measure a nonzero offset
        inside the viewport. The viewport is the bounds of the panel's slots
        on both frames of the first scroll step since the last clear, kept
        for the steps that follow: mid scroll find_slots can report a partly
        shown row at its full size, past the viewport's edge. None otherwise.
        """
        if self._still is not frames_still:
            return None
        ignore = [*self._ignore_of(a, b), *extra]
        found_a, found_b = self._found_of(a[0]), self._found_of(b[0])

        def changes(keep: Callable[[object], bool]) -> GridChange | None:
            rects_a = [s.rect for s in found_a if keep(s)]
            rects_b = [s.rect for s in found_b if keep(s)]
            if not rects_a and not rects_b:
                return None
            return stash_changes(
                a[0],
                b[0],
                ignore,
                frames_still,
                lambda rgb: rects_a if rgb is a[0] else rects_b,
                self._gray_of,
            )

        for panel in SCROLL_PANELS:
            mine_a = [s.rect for s in found_a if getattr(s, "panel", None) == panel]
            mine_b = [s.rect for s in found_b if getattr(s, "panel", None) == panel]
            if not mine_a or not mine_b:
                continue
            inside = changes(lambda s, p=panel: getattr(s, "panel", None) == p)
            if not inside.big:
                continue
            outside = changes(lambda s, p=panel: getattr(s, "panel", None) != p)
            if outside is not None and outside.big:
                return None
            with self._lock:
                seen = self._viewports.get(panel)
            rects = [*mine_a, *mine_b]
            region = seen or Rect(
                min(r.left for r in rects),
                min(r.top for r in rects),
                max(r.right for r in rects),
                max(r.bottom for r in rects),
            )

            def filled(found: list, p: str = panel) -> list[Rect]:
                return [
                    s.rect
                    for s in found
                    if getattr(s, "panel", None) == p and getattr(s, "filled", False)
                ]

            dy = scroll_offset(
                a[0],
                b[0],
                region,
                filled(found_a),
                filled(found_b),
                grays=(self._gray_of(a[0]), self._gray_of(b[0])),
            )
            if not dy:
                return None
            return panel, region, dy
        return None

    def _follow(
        self,
        generation: int,
        rgb: np.ndarray,
        frame: StashFrame,
        fed_at: float,
        panel: str,
        region: Rect,
        dy: int,
    ) -> None:
        """Move the badges of a scrolled panel with it instead of clearing them."""
        translate = getattr(self._surface, "translate", None)
        with self._lock:
            if generation != self._generation:
                return
            self._newest = (rgb, frame)
            self._viewports[panel] = region
            # Badges hidden on changed slots stay off, the next pass renews them
            before = badges_clear_of(self._badges, self._changed)
            self._badges = follow_badges(before, region, dy)
            self._marks = [m for b in self._badges for m in badge_marks(b)]
            self._pass_hidden = [
                followed_rect(r, region, dy) for r in self._pass_hidden
            ]
            self._peeked = [
                m
                for m in (followed_rect(r, region, dy) for r in self._peeked)
                if _within(m, region)
            ]
            if region not in self._omitted:
                self._omitted = (*self._omitted, region)
            self._changed = self._raw_changed = ()
            self._renew = {}
            self._pending = None
            if self._ref is not None:
                # The badges now sit where this frame shows their slots
                self._ref = (rgb, frame)
            self._need_pass = True
            self._lat = _latency_from(fed_at)
            if self._drawn and translate is not None:
                translate(tuple(region), dy)
                self._shown = follow_badges(self._shown, region, dy)
            kept = len(self._badges)
        logger.debug(
            "stash badges followed panel=%s dy=%d kept=%d dropped=%d",
            panel,
            dy,
            kept,
            len(before) - kept,
        )
        self._draw_scrolled_in(generation, rgb, frame, region)
        self._refresh()

    def _draw_scrolled_in(
        self, generation: int, rgb: np.ndarray, frame: StashFrame, region: Rect
    ) -> None:
        """Add the badges of the known slots in region that have none, before the pass.

        peek runs only when a filled slot in region has no badge and was not
        looked at by an earlier peek since the scroll began.
        """
        if self._peek is None:
            return
        with self._lock:
            have = [Rect(*(int(v) for v in b["rect"])) for b in self._badges]
            peeked = list(self._peeked)
        bare = [
            s.rect
            for s in self._found_of(rgb)
            if getattr(s, "filled", False)
            and _within(s.rect, region)
            and not any(rects_intersect(s.rect, h) for h in have)
        ]
        _seen, alone = _match_slots(bare, peeked)
        new = [r for r in alone if r in bare]
        if not new:
            return
        with self._lock:
            if generation != self._generation:
                return
            self._peeked += new
        try:
            slot_items = [
                s
                for s in self._peek(rgb)
                if _within(s.rect, region)
                and not any(rects_intersect(s.rect, h) for h in have)
            ]
            badges = self._make_badges(slot_items) if slot_items else []
        except Exception as e:  # noqa: BLE001 - the pass draws them instead
            logger.warning("stash scrolled in slots not drawn (%s)", e)
            return
        if not self._keep_border:
            badges = [b for b in badges if b["kind"] != "border"]
        badges = badges_clear_of(badges, frame_hidden(frame))
        if not badges:
            return
        with self._lock:
            if generation != self._generation:
                return
            self._badges = [*self._badges, *badges]
            self._marks += [m for b in badges for m in badge_marks(b)]
        logger.debug("stash scrolled in slots drawn badges=%d", len(badges))

    def _draw_known(self, generation: int, rgb: np.ndarray, frame: StashFrame) -> None:
        """Draw the badges of the slots peek already knows, before the pass."""
        if self._peek is None:
            return
        try:
            slot_items = self._peek(rgb)
            badges = self._make_badges(slot_items) if slot_items else []
        except Exception as e:  # noqa: BLE001 - the pass draws them instead
            logger.warning("stash known slots not drawn (%s)", e)
            return
        if not self._keep_border:
            badges = [b for b in badges if b["kind"] != "border"]
        badges = badges_clear_of(badges, frame_hidden(frame))
        if not badges:
            return
        with self._lock:
            if generation != self._generation or self._have_result:
                return
            self._badges = badges
            self._marks = [m for b in badges for m in badge_marks(b)]
        logger.debug("stash known slots drawn badges=%d", len(badges))
        self._refresh()

    def _latency_check(self) -> None:
        """Log the latency line once its pass and first draw are in (under the lock)."""
        lat = self._lat
        if lat is None or lat["pass"] is None or lat["drawn"] is None:
            return
        self._lat = None
        logger.debug(
            "stash latency moved_to_still=%.0f pass=%.0f still_to_drawn=%.0f "
            "total=%.0f ms",
            (lat["still"] - lat["moved"]) * 1000,
            lat["pass"] * 1000,
            (lat["drawn"] - lat["still"]) * 1000,
            (lat["drawn"] - lat["moved"]) * 1000,
        )

    def _pass_due(self, frame: StashFrame, now: float) -> bool:
        """True when a still frame should start a pass now (under the lock).

        A pass is due when one is needed or slots changed since the badges'
        frame, none is running, gap_s has passed since the last one and no
        card is fading out over the grid.
        """
        if not (self._need_pass or self._changed) or self._pass_running:
            return False
        if frame.card is None and self._lingering():
            return False
        return (
            self._last_pass_start is None or now - self._last_pass_start >= self._gap_s
        )

    def _found_of(self, rgb: np.ndarray) -> list:
        """find_slots of a frame, found once while StashBadges holds it."""
        return self._cached(self._slot_cache, rgb, find_slots)

    def _slots_of(self, rgb: np.ndarray) -> list[Rect]:
        """find_slots rects of a frame, found once while StashBadges holds it."""
        return [s.rect for s in self._found_of(rgb)]

    def _gray_of(self, rgb: np.ndarray) -> np.ndarray:
        """The still_gray copy of a frame, made once while StashBadges keeps it."""
        return self._cached(self._gray_cache, rgb, still_gray)

    def _cached(self, cache: dict, rgb: np.ndarray, make: Callable):
        with self._lock:
            hit = cache.get(id(rgb))
        if hit is not None and hit[0] is rgb:
            return hit[1]
        value = make(rgb)
        with self._lock:
            if any(r is rgb for r in self._held()):
                cache[id(rgb)] = (rgb, value)
        return value

    def _held(self) -> list[np.ndarray]:
        """The RGB frames _prev, _newest, _job and _ref refer to (under the lock)."""
        return [
            r
            for r in (
                self._prev[0] if self._prev else None,
                self._newest[0] if self._newest else None,
                self._job[1] if self._job else None,
                self._ref[0] if self._ref else None,
            )
            if r is not None
        ]

    def _evict_slots(self) -> None:
        """Drop the cached slots and greys of frames no longer held (under the lock).

        Runs at the end of every watcher round and pass. Within a round the
        frame _prev referred to before stays cached until its compare is done.
        """
        held = {id(r): r for r in self._held()}
        for cache in (self._slot_cache, self._gray_cache):
            for key, (rgb, _value) in list(cache.items()):
                if held.get(key) is not rgb:
                    del cache[key]

    def _uncovered(self, frame: StashFrame) -> bool:
        """True when the parts the last pass could not see are in view now."""
        if not self._have_result or not self._pass_hidden or frame.card is not None:
            return False
        return not any(
            rects_intersect(o, h) for o in frame.overlays for h in self._pass_hidden
        )

    def _run_pass(self, job: tuple[int, np.ndarray, StashFrame]) -> None:
        """Identify the slots of a still frame and keep its badges on unchanged slots."""
        generation, rgb, frame = job
        start = time.perf_counter()
        began = self._clock()
        try:
            slot_items = self._identify(rgb)
            if self._ledger is not None:
                self._feed_ledger(rgb, slot_items)
            badges = self._make_badges(slot_items)
            spent = self._clock() - began
        except Exception as e:  # noqa: BLE001 - badges must never stop ARLO
            logger.warning("Stash badges off: the stash pass failed (%s)", e)
            self._disabled = True
            with self._lock:
                self._pass_running = False
            self.clear("pass_failed")
            return
        if not self._keep_border:
            badges = [b for b in badges if b["kind"] != "border"]
        hidden = frame_hidden(frame)
        badges = badges_clear_of(badges, hidden)
        identified = sum(s.item is not None for s in slot_items)
        logger.debug(
            "stash pass identified=%d unidentified=%d badges=%d duration=%.0f ms",
            identified,
            len(slot_items) - identified,
            len(badges),
            (time.perf_counter() - start) * 1000,
        )
        with self._lock:
            newest = self._newest
            ref = self._ref
            extra = [*self._lingering(), *self._marks, *self._renew_marks()]
            # A big change waiting for its next frame may be about to clear
            waiting = self._pending is not None
        current = generation == self._generation and not waiting
        # Slots under the cursor in the last pass's frame, never compared with it
        was_hovered = self._hovered(ref, ref) if ref is not None else []
        since = GridChange(big=False)
        if current and newest is not None and newest[1] is not frame:
            since = self._changes((rgb, frame), newest, extra)
        # Slots that changed between the last pass's frame and this one
        moved = GridChange(big=True)
        if current and ref is not None and self._still is frames_still:
            moved = self._changes(ref, (rgb, frame), extra)
        with self._lock:
            self._pass_running = False
            if not current or generation != self._generation or since.big:
                self._evict_slots()
                if generation == self._generation:
                    self._need_pass = True
                # A stale pass does not hold the next one back by the gap
                self._last_pass_start = None
                logger.debug("stash pass dropped reason=stale")
                return
            if since.changed:
                kept = badges_clear_of(badges, since.changed)
                logger.debug(
                    "stash pass partial kept=%d dropped=%d",
                    len(kept),
                    len(badges) - len(kept),
                )
                badges = kept
                self._need_pass = True
            if self._have_result:
                # A later pass: new badges where the last pass could not see,
                # dropped badges or the slot changed, the last pass's elsewhere
                renew = [
                    *self._pass_hidden,
                    *self._omitted,
                    *(() if moved.big else (*moved.changed, *was_hovered)),
                ]
                if moved.big:
                    old = []
                else:
                    old = badges_clear_of(self._badges, renew)
                    badges = [
                        b
                        for b in badges
                        if any(rects_intersect(Rect(*b["rect"]), h) for h in renew)
                    ]
                badges = [*old, *badges]
                if not moved.big:
                    badges += self._kept_drawn(
                        badges, [*since.changed, *moved.changed, *was_hovered]
                    )
            self._badges = badges
            self._have_result = True
            self._ref = (rgb, frame)
            self._pass_hidden = hidden
            self._changed = self._raw_changed = since.changed
            self._omitted = since.changed
            self._peeked = []
            self._renew = {}
            self._marks = [m for b in badges for m in badge_marks(b)]
            if (
                self._lat is not None
                and self._lat["still"] is not None
                and self._lat["pass"] is None
                and began >= self._lat["still"]
            ):
                self._lat["pass"] = spent
                self._latency_check()
            self._evict_slots()

    def _feed_ledger(self, rgb: np.ndarray, slot_items: list[SlotItem]) -> None:
        """Hand a pass's frame to the ledger before its badges are made. Never raises."""
        try:
            self._ledger.after_pass(rgb, self._found_of(rgb), slot_items)
        except Exception:  # noqa: BLE001 - the ledger must never stop a pass
            logger.warning("stash ledger failed", exc_info=True)

    def _kept_drawn(self, badges: list[dict], looks: Sequence[Rect]) -> list[dict]:
        """The drawn badges a pass's badges have nothing for, on unchanged slots.

        A badge's rect is its slot's, so a drawn badge is kept when no badge
        of the pass and no slot in looks (changed since the badges' frame)
        meets it. Under the lock.
        """
        have = [Rect(*b["rect"]) for b in badges]
        kept = [
            b
            for b in self._shown
            if not any(rects_intersect(Rect(*b["rect"]), r) for r in (*have, *looks))
        ]
        if kept:
            logger.debug("stash pass kept_drawn=%d", len(kept))
        return kept

    def _refresh(self) -> None:
        """Draw the accepted badges clear of the newest card, windows and changed slots."""
        with self._lock:
            hidden = frame_hidden(self._newest[1]) if self._newest else []
            renewed = [b for r in self._changed for b in self._renew.get(r, ())]
            visible = [
                *badges_clear_of(self._badges, [*hidden, *self._changed]),
                *badges_clear_of(renewed, hidden),
            ]
            if visible == self._shown:
                return
            generation = self._generation
        if visible and not self._surface_started:
            # Outside the lock: the first start looks for a system Python
            self._surface_started = True
            self._surface.start()
        with self._lock:
            if generation != self._generation:
                return
            self._shown = visible
            if visible:
                self._surface.draw(surface_badges(visible))
                self._drawn = True
                lat = self._lat
                if (
                    lat is not None
                    and lat["still"] is not None
                    and lat["drawn"] is None
                ):
                    lat["drawn"] = self._clock()
                    self._latency_check()
            elif self._drawn:
                self._surface.clear()
                self._drawn = False


class ScannerState(Enum):
    """State machine for the scanner."""

    IDLE = auto()  # Scanning for trigger (INVENTORY)
    ACTIVE = auto()  # Inventory detected, scanning tooltip
    PAUSED = auto()  # Temporarily paused
    STOPPED = auto()  # Fully stopped


class DebugOverlay:
    """Semi-transparent overlay showing the card search window."""

    def __init__(self, root: tk.Tk, settings):
        self.root = root
        self.settings = settings

        self.window = tk.Toplevel(root)
        self.window.title("Capture Area")
        self.window.attributes("-topmost", True)  # noqa: FBT003
        self.window.attributes("-alpha", 0.3)  # Semi-transparent
        self.window.overrideredirect(boolean=True)
        self.window.config(bg="red")
        make_click_through(self.window)

        # Update position periodically
        self._update_position()

    def _update_position(self):
        """Update overlay position to follow cursor."""
        try:
            from arc_helper.ocr import get_cursor_position
            from arc_helper.ocr import search_window_bbox
            from arc_helper.screen import phys_to_tk

            cursor = get_cursor_position()
            left, top, right, bottom = search_window_bbox(
                cursor,
                self.settings.tooltip_search.width,
                self.settings.tooltip_search.height,
            )

            x = phys_to_tk(left)
            y = phys_to_tk(top)
            w = phys_to_tk(right - left)
            h = phys_to_tk(bottom - top)

            self.window.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:  # noqa: BLE001
            pass

        # Schedule next update (every 50ms for smooth following)
        self.root.after(50, self._update_position)

    def destroy(self):
        self.window.destroy()


def chip_payload(
    outputs: list[RecycleOutput],
    home_only: list[RecycleOutput],
) -> dict[str, str | None]:
    """Recycle chip texts for TooltipChips.show. None hides the chip.

    Without field rows the recycle chip lists every home output. With field
    rows it lists only the home extras, and hides when there are none. The
    game card shows the sell price and the advice card the verdict, so the
    chips show neither.
    """
    recycle_label = recycle_text = None
    home = [o for o in outputs if o.location == HOME]
    has_field = any(o.location == FIELD for o in outputs)
    if home and not has_field:
        recycle_label = RECYCLE_LABEL_HOME_RECYCLER
        recycle_text = "  ".join(f"{o.quantity} {o.component}" for o in home)
    elif home and home_only:
        recycle_label = RECYCLE_LABEL_HOME_ONLY
        recycle_text = "  ".join(f"+{o.quantity} {o.component}" for o in home_only)
    return {"recycle_label": recycle_label, "recycle_text": recycle_text}


@dataclass
class ScannerStats:
    """Statistics for the scanner."""

    trigger_scans: int = 0
    tooltip_scans: int = 0
    items_detected: int = 0
    items_found_in_db: int = 0
    last_item: str | None = None
    last_item_time: float = 0


@dataclass
class Scanner:
    """
    Main scanner that coordinates OCR and overlay.

    Uses a two-phase approach:
    1. Low-frequency scan for "INVENTORY" text (trigger)
    2. When triggered, high-frequency scan for item names
    """

    root: tk.Tk
    overlay: OverlayWindow
    # None unless SHOW_STATUS_WINDOW is on
    status: StatusWindow | None
    db: Database = field(default_factory=get_database)
    state: ScannerState = ScannerState.IDLE
    stats: ScannerStats = field(default_factory=ScannerStats)
    hotkey: HotkeyMonitor = field(default_factory=get_hotkey_monitor)

    # UI updates are posted here for the main (GUI) thread to apply. tkinter
    # is not thread-safe, and on Linux/Tcl root.after() called from this
    # background thread silently never fires - so we marshal via a queue that
    # Application drains on a main-thread timer.
    ui_queue: "queue.Queue[tuple]" = field(default_factory=queue.Queue)

    # ARLO's own window rectangles, published by the Tk thread, kept out of
    # the hovered cell search
    overlay_rects: OverlayRects = field(default_factory=OverlayRects)

    # True when the advice card replaces the corner popup (ADVICE_CARD). The
    # Application passes the setting, a bare Scanner keeps the popup path
    advice_card: bool = False

    # Badges on every stash slot (STASH_BADGES), fed each full-screen capture
    # while ACTIVE through the inventory trigger. None turns them off
    stash: StashBadges | None = None

    # The stash ledger behind the stash badges, read for the owned count of
    # the hovered item. None without STASH_BADGES, the owned count stays None
    ledger: StashLedger | None = None

    # Cooldown tracking
    _last_shown_item: str = ""
    _last_shown_time: float = 0
    # Monotonic time of the last INVENTORY trigger check while ACTIVE
    _last_trigger_check: float = 0.0
    # Rechecks missed in a row while ACTIVE, and the monotonic time of the
    # first of them (None after a hit or on entry to ACTIVE)
    _trigger_misses: int = 0
    _trigger_miss_since: float | None = None

    # Monotonic start of the previous tooltip scan, for the scan period.
    # Reset on every entry to ACTIVE.
    _last_scan_start: float | None = None

    # Signature of the card the shown windows belong to, None while nothing
    # is shown. Cycles in a row without a card, the last card state logged
    # (none, new, same, gone, hide) and when the windows were last touched.
    _card_shown: CardSignature | None = None
    _card_missed: int = 0
    _card_state: str = "none"
    _last_touch: float = 0.0

    # Monotonic time of the last look for the weapon slots and the home box
    # last sent to the Tk thread, None until one was found
    _home_checked: float = float("-inf")
    _home_sent: Rect | None = None

    # Scan periods measured in this ACTIVE stretch (seconds) and the hold
    # computed after the latest scan (ms), sent with every overlay, touch and
    # chips message. None until the first scan, which leaves display_time.
    _scan_periods: deque = field(default_factory=lambda: deque(maxlen=HOLD_HISTORY))
    _hold_ms: int | None = None

    # Hotkey override (Ctrl+Shift to force tooltip scanning)
    _hotkey_override: bool = False

    # Monotonic time of the last look for the Collections screen while IDLE,
    # whether the screen showed then, and whether this visit has recorded a
    # consistent read or logged an inconsistent one
    _collections_checked: float = float("-inf")
    _collections_open: bool = False
    _collections_recorded: bool = False
    _collections_warned: bool = False

    # Thread control
    _running: bool = False
    _scan_thread: Thread | None = None

    # Held by stop() and pause() while they leave ACTIVE and queue the chip
    # hide, and by _queue_chips around its final state check and enqueue, so
    # a layout OCR pass still running during the transition cannot queue
    # chips after the hide. The OCR pass itself runs outside the lock.
    _chips_lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self) -> None:
        """Start the scanner in a background thread."""
        if self._running:
            return

        self._running = True
        self.state = ScannerState.IDLE
        self.hotkey.start()
        self._scan_thread = Thread(target=self._scan_loop, daemon=True)
        self._scan_thread.start()
        logger.info("Scanner started")

    def stop(self) -> None:
        """Stop the scanner."""
        with self._chips_lock:
            self._running = False
            self.state = ScannerState.STOPPED
            self._hide_card_windows()
        self._clear_stash("stopped")
        if self._scan_thread:
            self._scan_thread.join(timeout=2.0)
        self.hotkey.stop()
        logger.info("Scanner stopped")

    def pause(self) -> None:
        """Pause scanning."""
        with self._chips_lock:
            was_active = self.state == ScannerState.ACTIVE
            self.state = ScannerState.PAUSED
            if was_active:
                self._hide_card_windows()
        self._clear_stash("paused")

    def resume(self) -> None:
        """Resume scanning."""
        self.state = ScannerState.IDLE

    def _scan_loop(self) -> None:
        """Main scanning loop running in background thread."""
        settings = get_settings()
        ocr = get_ocr_engine()

        while self._running:
            try:
                if self.state == ScannerState.PAUSED:
                    time.sleep(0.1)
                    continue

                if self.state == ScannerState.STOPPED:
                    break

                # Check hotkey override (Ctrl+Shift)
                hotkey_now = self.hotkey.is_held()

                # Phase 1: Check for trigger (INVENTORY) or hotkey
                if self.state == ScannerState.IDLE:
                    self._update_status("scanning")

                    if hotkey_now:
                        # Hotkey pressed - enter active mode via override
                        self._hotkey_override = True
                        self.state = ScannerState.ACTIVE
                        self._reset_scan_timing()
                        logger.debug("state IDLE -> ACTIVE reason=hotkey")
                        self._update_status("hotkey")
                        logger.info("Hotkey override - activating tooltip scanner")
                    elif ocr.check_trigger_any(
                        [settings.trigger_region, settings.trigger_region2]
                    ):
                        # Trigger detected! Switch to active mode
                        self._hotkey_override = False
                        self.state = ScannerState.ACTIVE
                        self._reset_scan_timing()
                        logger.debug("state IDLE -> ACTIVE reason=inventory_trigger")
                        self._update_status("active")
                        logger.info("INVENTORY detected - activating tooltip scanner")
                    else:
                        # Look for the Collections screen at most every
                        # COLLECTIONS_CHECK_S, inside the wait before the next
                        # trigger scan so the trigger keeps its timing
                        began = time.monotonic()
                        self._check_collections(began)
                        spent = time.monotonic() - began
                        time.sleep(
                            max(0.0, settings.scan.trigger_scan_interval - spent)
                        )

                    self.stats.trigger_scans += 1

                # Phase 2: Active mode - scan tooltip
                elif self.state == ScannerState.ACTIVE:
                    if self._hotkey_override:
                        # Hotkey mode: stay active as long as hotkey is held
                        if not hotkey_now:
                            self._hotkey_override = False
                            self.state = ScannerState.IDLE
                            logger.debug("state ACTIVE -> IDLE reason=hotkey_released")
                            self._hide_card_windows()
                            self._update_status("scanning")
                            logger.info("Hotkey released - returning to idle")
                            continue
                    elif self._trigger_lost(ocr, settings):
                        # Inventory closed: the card went with it
                        self.state = ScannerState.IDLE
                        logger.debug(
                            "state ACTIVE -> IDLE reason=trigger_lost"
                            " misses=%d span=%d ms",
                            self._trigger_misses,
                            self._trigger_miss_span_ms(self._last_trigger_check),
                        )
                        self._hide_card_windows()
                        self._clear_stash("inventory_closed")
                        self._update_status("scanning")
                        logger.info("INVENTORY closed - returning to idle")
                        continue

                    self._scan_cycle(ocr, settings)

            except ScreenCaptureUnavailable as e:
                logger.error(f"Screen capture lost permanently: {e}")
                self._update_status("error")
                self.state = ScannerState.STOPPED
                self._hide_chips()
                self._clear_stash("capture_lost")
                break  # latch: no point scanning without capture
            except Exception as e:  # noqa: BLE001
                logger.error(f"Scanner error: {e}")
                self._update_status("error")
                time.sleep(1.0)  # Back off on error

    def _grab_full_screen(self):  # noqa: PLR6301 - test seam
        """One capture of the whole primary screen."""
        width, height = get_screen_resolution()
        return grab_screen(clamp_bbox(0, 0, width, height))

    def _check_collections(self, now: float) -> None:
        """Record blueprint ownership from the Collections screen, once a visit.

        Runs at most every COLLECTIONS_CHECK_S. A visit lasts while each look
        finds the screen on its BLUEPRINTS tab. Its first consistent read is
        recorded and nothing more is read until the screen has been left. An
        inconsistent read (a tooltip over the grid) is retried on the next
        look and logged once a visit. ARLO's windows and badges are untouched.
        """
        if now - self._collections_checked < COLLECTIONS_CHECK_S:
            return
        self._collections_checked = now
        read = read_collections(_frame_rgb(self._grab_full_screen()))
        if read is None:
            if self._collections_open:
                logger.debug("collections screen left")
            self._collections_open = False
            self._collections_recorded = False
            self._collections_warned = False
            return
        self._collections_open = True
        if self._collections_recorded:
            return
        if not read.consistent:
            if not self._collections_warned:
                self._collections_warned = True
                logger.info("collections read inconsistent")
            return
        unknown = record_collections(self.db, read)
        self._collections_recorded = True
        recorded = sum(
            1
            for cell in read.cells
            if self.db.blueprint_at(cell.grid_index) is not None
        )
        logger.info(
            "collections read found=%d/%d recorded=%d unknown=%s",
            read.found_count,
            read.total,
            recorded,
            unknown,
        )

    def _reset_scan_timing(self) -> None:
        """Forget scan timing and the shown card on entry to ACTIVE.

        The trigger was just checked (or the hotkey is held), so the next
        trigger recheck is TRIGGER_RECHECK_S away.
        """
        self._last_scan_start = None
        self._scan_periods.clear()
        self._last_trigger_check = time.monotonic()
        self._trigger_misses = 0
        self._trigger_miss_since = None
        self._card_shown = None
        self._card_missed = 0
        self._card_state = "none"

    def _trigger_lost(self, ocr: OCREngine, settings) -> bool:
        """Recheck INVENTORY about once a second, True once it is gone.

        The recheck runs on elapsed time, not on a cycle count, so fast
        cycles do not check it more often. It is gone only when the rechecks
        have missed for TRIGGER_LOST_S in a row and at least twice, so one
        slow or blocked read does not leave ACTIVE. One hit resets the count.
        Each miss logs what every place read.
        """
        now = time.monotonic()
        if now - self._last_trigger_check < TRIGGER_RECHECK_S:
            return False
        self._last_trigger_check = now
        found = ocr.check_trigger_any(
            [settings.trigger_region, settings.trigger_region2]
        )
        if found:
            self._trigger_misses = 0
            self._trigger_miss_since = None
            return False
        if self._trigger_miss_since is None:
            self._trigger_miss_since = now
        self._trigger_misses += 1
        reads = " ".join(
            f"{place}={text!r}" if text else f"{place}=none"
            for place, text in getattr(ocr, "last_trigger_reads", ())
        )
        logger.debug(
            "trigger miss %s check=%d ms misses=%d span=%d ms",
            reads or "reads=none",
            (time.monotonic() - now) * 1000,
            self._trigger_misses,
            self._trigger_miss_span_ms(now),
        )
        return (
            self._trigger_misses >= 2
            and now - self._trigger_miss_since >= TRIGGER_LOST_S
        )

    def _trigger_miss_span_ms(self, now: float) -> int:
        """Milliseconds from the first missed recheck in a row to now."""
        if self._trigger_miss_since is None:
            return 0
        return round((now - self._trigger_miss_since) * 1000)

    def _scan_cycle(self, ocr: OCREngine, settings) -> None:
        """One ACTIVE cycle: locate the card, then act on what changed.

        The same card as the one shown is not read again, only touched. A
        new card is read and shown at once. A shown card missing for
        CARD_GONE_CYCLES cycles hides the popup, chips and frame at once.
        Only a cycle without a card sleeps, and briefly.
        """
        start = time.monotonic()
        period_s = None
        if self._last_scan_start is not None:
            period_s = start - self._last_scan_start
            self._scan_periods.append(period_s)
        self._last_scan_start = start
        sighting = ocr.locate_card_at_cursor()
        locate_s = time.monotonic() - start
        self.stats.tooltip_scans += 1
        self._feed_stash(ocr)
        self._refresh_home(ocr)
        self._return_home(None if sighting is None else self._card_rect(ocr))

        if sighting is None:
            sleep_s = min(settings.scan.tooltip_scan_interval, NO_CARD_SLEEP_CAP_S)
            self._set_hold(settings, locate_s + sleep_s)
            self._card_absent()
            time.sleep(sleep_s)
            return

        self._card_missed = 0
        if self._card_shown is not None and same_card(self._card_shown, sighting):
            self._set_hold(settings, locate_s)
            self._log_card_state("same")
            now = time.monotonic()
            if now - self._last_touch >= TOUCH_EVERY_S:
                self._last_touch = now
                self.ui_queue.put(("touch", self._last_shown_item, self._hold_ms))
            return

        item_name = ocr.read_located_card()
        scan_s = time.monotonic() - start
        self._set_hold(settings, scan_s)
        self._log_card_state(
            "new",
            "rect=%s read=%s locate=%d ms read=%d ms period=%s",
            tuple(sighting.rect),
            item_name or "none",
            locate_s * 1000,
            (scan_s - locate_s) * 1000,
            "none" if period_s is None else f"{period_s * 1000:.0f} ms",
        )
        # A new card with no name hides the old item's windows at once, and
        # the next cycle reads this card again
        if not item_name:
            if self._card_shown is not None:
                self._log_card_state("hide", "reason=no_read")
                self._hide_card_windows()
            return
        # A discarded read (card under ARLO's popup) keeps the shown card, so
        # the next cycle reads this card again once the popup moved aside
        if self._handle_detected_item(item_name, new_card=True):
            self._card_shown = sighting
            self._last_touch = time.monotonic()

    def _feed_stash(self, ocr: OCREngine) -> None:
        """Hand this cycle's capture to the stash badges and the capture run.

        Only in inventory mode (not the hotkey) and only for a full-screen
        capture. Only a reference changes hands, so the cycle does not wait.
        In a capture run the capture is also offered as a grid frame, also
        with the stash badges off. The first cycle of an ACTIVE stretch (no
        scan period yet) starts a new stretch of grid frames. The stash gets
        the card this cycle's locate found and the cursor from the screen
        backend, in capture px, or None when the backend cannot say.
        """
        grid = getattr(ocr, "capture_library", False) is True
        if (self.stash is None and not grid) or self._hotkey_override:
            return
        capture = getattr(ocr, "_last_capture", None)
        origin = getattr(ocr, "_last_origin", None)
        full_screen = (
            getattr(ocr, "search_width", 0) <= 0
            and getattr(ocr, "search_height", 0) <= 0
        )
        if capture is None or origin is None or not full_screen:
            return
        if (origin.x, origin.y) != (0, 0):
            return
        card = getattr(ocr, "_last_card", None)
        overlays = self.overlay_rects.snapshot()
        if grid:
            ocr.offer_grid_frame(
                capture,
                frame_still_ignore(StashFrame(capture, card, overlays)),
                lambda a, b, ignore: stash_still(_frame_rgb(a), _frame_rgb(b), ignore),
                new_stretch=not self._scan_periods,
            )
        if self.stash is not None:
            try:
                at = get_backend().cursor_position()
                cursor = Point(at.x - origin.x, at.y - origin.y)
            except Exception:  # noqa: BLE001 - the hovered slot is then compared
                cursor = None
            self.stash.feed(capture, card, overlays, cursor)

    def _refresh_home(self, ocr: OCREngine) -> None:
        """Send home beneath the second gun when the weapon slots moved.

        At most once every HOME_REFRESH_S, find_slots looks at this cycle's
        capture when it is the full screen. Home is sent only when both
        weapon slots are found and it moved by more than HOME_MOVE of the
        screen height, so the slot finder's pixel jitter does not re-render
        the card.
        """
        now = time.monotonic()
        if now - self._home_checked < HOME_REFRESH_S:
            return
        self._home_checked = now
        capture = getattr(ocr, "_last_capture", None)
        origin = getattr(ocr, "_last_origin", None)
        if capture is None or origin is None or (origin.x, origin.y) != (0, 0):
            return
        rgb = _frame_rgb(capture)
        screen = (rgb.shape[1], rgb.shape[0])
        if screen != get_screen_resolution():
            return
        home = layout_home.home_below_weapons(
            layout_home.weapon_slots(find_slots(rgb)), screen
        )
        if home is None or not home_moved(self._home_sent, home, screen):
            return
        logger.debug("popup home box=%s", tuple(home))
        self._home_sent = home
        self.ui_queue.put(("popup_home", home))

    def _return_home(self, card: Rect | None) -> None:
        """Queue the popup's return home while it is away and home is free.

        Runs on every scan, so the popup goes home on the first scan whose
        card (the same item or not) leaves home free, or that has no card.
        Hiding the windows after CARD_GONE_CYCLES is a separate event.
        """
        if not self.overlay_rects.popup_away():
            return
        if home_is_free(card, self.overlay_rects.labelled()):
            self._queue_popup_spot((POPUP_HOME, None))

    def _clear_stash(self, reason: str) -> None:
        if self.stash is not None:
            self.stash.clear(reason)

    def _set_hold(self, settings, estimate_s: float) -> None:
        """Set the hold sent with the next messages from this cycle's estimate."""
        hold_s = scan_hold_s(
            settings.overlay.display_time, self._scan_periods, estimate_s
        )
        self._hold_ms = round(max(HOLD_FLOOR_S, hold_s) * 1000)

    def _card_absent(self) -> None:
        """A cycle found no card: hide the shown windows on the second miss."""
        self._card_missed += 1
        if self._card_shown is None:
            self._card_state = "none"
            return
        if self._card_missed < CARD_GONE_CYCLES:
            self._log_card_state("gone")
            return
        self._log_card_state("hide")
        self._hide_card_windows()

    def _hide_card_windows(self) -> None:
        """Queue an immediate hide of the popup, chips and frame.

        The shown item is forgotten too, so the same item read again later
        shows instead of only touching hidden windows.
        """
        self._card_shown = None
        self._last_shown_item = ""
        self.ui_queue.put(("hide",))

    def _log_card_state(self, state: str, detail: str = "", *args) -> None:
        """Log a card state once per change. A new card always logs."""
        if state == self._card_state and state != "new":
            return
        self._card_state = state
        message = f"card {state} {detail}".rstrip()
        logger.debug(message, *args)

    def _handle_detected_item(self, item_name: str, *, new_card: bool = False) -> bool:
        """Handle a detected item name.

        new_card is True when the scan loop read a card it had not shown,
        which always takes the show path, so the popup, chips and frame
        follow the card even for the same item inside the cooldown. Returns
        whether the windows now show (or keep showing) this item, False
        when the read was discarded.
        """
        settings = get_settings()
        current_time = time.time()

        # A card under ARLO's own popup or status window was read without
        # its covered part, so the read is dropped. The popup moves aside so
        # the next scan sees the whole card.
        card = self._card_on_screen()
        labelled = self.overlay_rects.labelled()
        if card is not None:
            covering = covering_window(card, labelled)
            if covering is not None:
                logger.debug("read discarded: card covered by %s", covering)
                if covering == POPUP:
                    self._queue_popup_spot(
                        popup_escape_spot(card, labelled, get_screen_resolution())
                    )
                return False

        # Look up first: raw reads of one item vary between scans (ANVIL IV,
        # ANVILIV), so the cooldown and the popup key on the catalog name
        recommendation = self.db.lookup(
            item_name, tier_vote=get_ocr_engine().last_tier_vote
        )
        shown_name = recommendation.name if recommendation else item_name
        same_item = shown_name == self._last_shown_item

        # Check cooldown - don't spam the same item
        if (
            same_item
            and not new_card
            and current_time - self._last_shown_time < settings.overlay.cooldown
        ):
            # Still hovered: keep the popup and chips from auto-hiding
            logger.debug(
                "detect read=%s item=%s path=touch since_shown=%d ms",
                item_name,
                shown_name,
                (current_time - self._last_shown_time) * 1000,
            )
            self.ui_queue.put(("touch", shown_name, self._hold_ms))
            return True

        logger.debug(
            "detect read=%s item=%s path=show same_item=%s",
            item_name,
            shown_name,
            "yes" if same_item else "no",
        )

        self.stats.items_detected += 1
        self.stats.last_item = item_name
        self.stats.last_item_time = current_time

        if recommendation:
            self.stats.items_found_in_db += 1
            logger.debug(f"Found: {item_name} → {recommendation.action}")
        else:
            logger.debug("read unmatched read=%s", item_name)
            # Log to missing items file for easier database updates
            self.db.log_missing_item(item_name)
            # A read matching nothing shows no card. An older item's windows
            # hide, as for a read with no name
            if self._card_shown is not None:
                self._log_card_state("hide", "reason=unmatched")
                self._hide_card_windows()
            # Handled: the card counts as read so the next cycle does not
            # read and log it again
            return True

        # The slot under the cursor is found once per shown card. Its frame
        # goes to the chips and its learned check to the verdict
        hovered = self._locate_hovered_slot(recommendation)
        owned = self._owned_for(recommendation, hovered)
        verdict = self._verdict_for(
            recommendation,
            settings.show_ended_projects,
            hovered.learned,
            owned.count if owned else None,
        )

        # Show overlay (must be done on main thread)
        if owned is None:
            self._show_overlay(shown_name, recommendation, verdict)
        else:
            self._show_overlay(shown_name, recommendation, verdict, owned)

        if settings.overlay.tooltip_chips:
            self._queue_chips(
                recommendation,
                same_item=same_item,
                frame=hovered.frame,
            )

        # Update cooldown tracking
        self._last_shown_item = shown_name
        self._last_shown_time = current_time
        return True

    @staticmethod
    def _card_on_screen() -> Rect | None:
        """The last scan's item card in physical screen px, None without one."""
        return Scanner._card_rect(get_ocr_engine())

    @staticmethod
    def _card_rect(engine) -> Rect | None:
        """The engine's last item card in physical screen px, None without one."""
        origin = getattr(engine, "_last_origin", None)
        card = getattr(engine, "_last_card", None)
        if origin is None or card is None:
            return None
        return to_screen(card, origin)

    def _queue_popup_spot(self, spot: tuple[str, Rect | None] | None) -> None:
        """Queue the popup's move to a spot and its rect. None queues nothing.

        Home goes without a rect: the Tk thread knows where home is.
        """
        if spot is None:
            return
        self.ui_queue.put(("popup_spot", *spot))

    def _locate_hovered_slot(self, recommendation: Item | None) -> HoveredSlot:
        """Find the hovered slot of a recognized item in the last capture.

        For a blueprint the slot's learned check is read too, and a True or
        False read is recorded in the user database. Unknown items are not
        located.
        """
        if recommendation is None:
            return NO_HOVERED_SLOT
        engine = get_ocr_engine()
        hovered = hovered_slot_on_screen(
            getattr(engine, "_last_capture", None),
            getattr(engine, "_last_origin", None),
            getattr(engine, "_last_card", None),
            self.overlay_rects.snapshot(),
            read_check=is_blueprint(recommendation),
            read_count=self.ledger is not None,
        )
        if hovered.learned is not None:
            self.db.set_blueprint_owned(recommendation.name, owned=hovered.learned)
        return hovered

    def _owned_for(
        self, recommendation: Item | None, hovered: HoveredSlot
    ) -> OwnedCount | None:
        """The owned count of the item for its verdict and card, one ledger read.

        None without a ledger, a stored stash map or an item. The count is
        the stored map plus the loadout by item name. While the map is not
        sure the count is a lower bound, and the stack read under the cursor
        raises it when larger (the same rule as the stash tags).
        """
        if recommendation is None or self.ledger is None:
            return None
        totals = self.ledger.owned()
        if totals is None:
            return None
        count = totals.counts.get(recommendation.name, 0)
        if not totals.sure and hovered.stack is not None:
            count = max(count, hovered.stack)
        return OwnedCount(count, totals.sure)

    def _verdict_for(
        self,
        recommendation: Item | None,
        show_ended: bool,  # noqa: FBT001
        learned: bool | None = None,  # noqa: FBT001
        owned: int | None = None,
    ) -> Verdict:
        """Advice for the item from its needs and the recorded progress.

        owned is the item's owned count from the stash ledger, None when
        unknown. Project end dates compare against the local date. learned is
        the hovered slot's learned blueprint check, and without a read a
        blueprint falls back to the blueprints recorded as owned.
        """
        today = datetime.datetime.now().astimezone().date()
        if recommendation is None:
            return compute_verdict(None, [], {}, None, today, show_ended)
        recorded = False
        if learned is None and is_blueprint(recommendation):
            name = recommendation.name.casefold()
            recorded = any(n.casefold() == name for n in self.db.owned_blueprints())
        home = [
            o
            for o in self.db.recycle_outputs(recommendation.name)
            if o.location == HOME
        ]
        return compute_verdict(
            recommendation,
            self.db.needs_for(recommendation.name),
            self.db.progress_map(),
            owned,
            today,
            show_ended,
            blueprint=self.db.blueprint_for(recommendation.name),
            home_outputs=home,
            learned=learned,
            recorded_learned=recorded,
            owned_blueprints=self.db.owned_blueprints(),
        )

    def _show_overlay(
        self,
        item_name: str,
        recommendation: Item | None,
        verdict: Verdict,
        owned: OwnedCount | None = None,
    ) -> None:
        """Queue an overlay update for the main (GUI) thread to apply.

        With the advice card on, the card is built here on the scan thread
        and only rendered and shown on the Tk thread.
        """
        if self.advice_card:
            card = self._advice_card_for(recommendation, verdict, owned)
            self.ui_queue.put(("advice", item_name, card, self._hold_ms))
            return
        self.ui_queue.put(
            ("overlay", item_name, recommendation, verdict, self._hold_ms)
        )

    def _advice_card_for(
        self,
        recommendation: Item | None,
        verdict: Verdict,
        owned: OwnedCount | None = None,
    ) -> AdviceCard:
        """The advice card for an item and its verdict (scan thread).

        A home recycle output counts as still needed when its own verdict is
        keep or keep_some. owned is the item's count from the stash ledger,
        None leaves the card without one.
        """
        if recommendation is None:
            return build_advice_card(verdict, [])
        show_ended = get_settings().show_ended_projects
        home = [
            o
            for o in self.db.recycle_outputs(recommendation.name)
            if o.location == HOME
        ]

        def is_needed(component: str, _needs) -> bool:
            part = self.db.lookup(component)
            return self._verdict_for(part, show_ended).kind in {KEEP, KEEP_SOME}

        return build_advice_card(
            verdict,
            self.db.needs_for(recommendation.name),
            owned.count if owned else None,
            home,
            needed_parts(home, self.db.needs_for, is_needed),
            owned_sure=owned.sure if owned else True,
        )

    def _queue_chips(
        self,
        recommendation: Item | None,
        *,
        verdict: Verdict | None = None,  # noqa: ARG002
        same_item: bool = False,
        frame: Rect | object | None = LOCATE_FRAME,
    ) -> None:
        """Locate the tooltip and queue the chips for the current item.

        verdict is accepted but no longer reaches the chips, the advice card
        shows it.

        frame is the hovered slot's gold frame from _locate_hovered_slot,
        None for no frame. Left out, the frame is located here.

        Runs the extra layout OCR pass, so it is only called once the
        cooldown check passed. Unknown items and missing tooltips hide the
        chips, so they never sit beside another item's tooltip. When the
        item is unchanged and only this scan's layout read failed, the
        chips stay where they are and their timer is restarted.
        """
        layout = None
        if recommendation is not None:
            layout = get_ocr_engine().layout_from_last_capture()
        if recommendation is not None and layout is None and same_item:
            self.ui_queue.put(("touch", recommendation.name, self._hold_ms))
            return
        if recommendation is None or layout is None:
            self._hide_chips()
            return

        payload = chip_payload(
            self.db.recycle_outputs(recommendation.name),
            self.db.home_only_outputs(recommendation.name),
        )
        # The gold frame marks the hovered cell of every identified item,
        # whatever the verdict
        border = frame
        if border is LOCATE_FRAME:
            engine = get_ocr_engine()
            border = hovered_cell_on_screen(
                getattr(engine, "_last_capture", None),
                getattr(engine, "_last_origin", None),
                getattr(engine, "_last_card", None),
                self.overlay_rects.snapshot(),
            )
        with self._chips_lock:
            # stop() or pause() may have hidden the chips during the OCR pass
            if self._running and self.state == ScannerState.ACTIVE:
                # border is the gold frame rect, None hides it. The hold
                # stays the last field, as in every held message
                self.ui_queue.put(
                    (
                        "chips",
                        recommendation.name,
                        layout,
                        payload,
                        border,
                        self._hold_ms,
                    )
                )

    def _hide_chips(self) -> None:
        """Queue hiding the tooltip chips for the main (GUI) thread."""
        self.ui_queue.put(("chips_hide",))

    def _update_status(self, status: str) -> None:
        """Queue a status update for the main (GUI) thread to apply."""
        self.ui_queue.put(("status", status))


def _ui_message_detail(msg: tuple) -> str:
    """Short text for a ui_queue message in the debug log."""
    if msg[0] == "status":
        return f"status={msg[1]}"
    if msg[0] in {"touch", "overlay", "advice", "chips"}:
        return f"item={msg[1]}"
    if msg[0] == "popup_spot":
        return f"spot={msg[1]}"
    if msg[0] == "popup_home":
        return f"box={tuple(msg[1])}"
    return ""


class Application:
    """Main application controller."""

    # The advice card window, None with ADVICE_CARD=false (the corner popup)
    advice: AdviceWindow | None = None
    # Item and card the advice window shows, its last size in physical px
    # and its hold timer
    _advice_item: str | None = None
    _advice_card: AdviceCard | None = None
    _advice_size: tuple[int, int] | None = None
    _advice_hide_id: str | None = None
    # Home box beneath the second gun from the scanner, None until the
    # weapon slots were found (the fixed LOADOUT box then), and the top left
    # of the popup's spot while away from home
    _home_box: Rect | None = None
    _popup_away_at: Point | None = None

    def __init__(self):
        """Initialize the application."""
        # Load settings
        self.settings = get_settings()

        # Initialize database
        self.db = get_database()

        # Create main Tk root (hidden)
        self.root = tk.Tk()
        self.root.withdraw()  # Hide main window

        # Create overlay and status windows
        self.overlay = OverlayWindow(self.root)
        # With ADVICE_CARD on the advice card takes the popup's place, the
        # popup itself stays hidden
        if self.settings.overlay.advice_card:
            self.advice = AdviceWindow(self.root)
        # OVERLAY_X and OVERLAY_Y now set the top left corner spot, the last
        # spot the popup moves to while a card covers its home
        self._popup_corner = Point(self.overlay.overlay_x, self.overlay.overlay_y)
        self._popup_spot = POPUP_HOME
        self._place_popup()
        self.status = self._make_status()
        self.chips = (
            TooltipChips(self.root) if self.settings.overlay.tooltip_chips else None
        )
        # Gold frame around the hovered item, hidden with the chips
        self.border = CellBorder(self.root) if self.chips is not None else None
        if self.chips is not None and self.border is not None:
            self.chips.on_hide = self.border.hide

        # Debug overlay for visualizing capture area (separate from debug_mode)
        self.debug_overlay = None
        if self.settings.show_capture_area:
            self.debug_overlay = DebugOverlay(self.root, self.settings)

        # Create scanner. The scanner thread posts UI updates to this queue;
        # the main thread drains it in _pump_ui_queue (tkinter is single-thread).
        self.ui_queue: queue.Queue[tuple] = queue.Queue()
        # Last status applied, so the debug log names only status changes
        self._last_status: str | None = None
        self.ledger: StashLedger | None = None
        self.stash = self._make_stash()
        self.scanner = Scanner(
            root=self.root,
            overlay=self.overlay,
            status=self.status,
            db=self.db,
            ui_queue=self.ui_queue,
            advice_card=self.advice is not None,
            stash=self.stash,
            ledger=self.ledger,
        )
        self.overlay_rects = self.scanner.overlay_rects

        # Bind close handler
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    def _make_status(self) -> StatusWindow | None:
        """The scanning and inventory indicator, None unless SHOW_STATUS_WINDOW.

        It is a debug aid, so it is off by default and nothing creates or
        maps it then.
        """
        if not self.settings.show_status_window:
            return None
        return StatusWindow(self.root, on_quit=self.quit)

    def _make_stash(self) -> StashBadges | None:
        """The stash badges per STASH_BADGES, None when turned off."""
        overlay = self.settings.overlay
        if not overlay.stash_badges:
            return None
        try:
            ledger = StashLedger(self.db)
            self.ledger = ledger
        except Exception:  # noqa: BLE001 - without it the tags count the slots on screen
            logger.warning("stash ledger off", exc_info=True)
            ledger = None
        return StashBadges(
            stash_identifier(self.db, self.settings.icon_cache_dir, warm=True),
            stash_badge_maker(
                self.db,
                lambda: get_settings().show_ended_projects,
                ledger.owned if ledger else None,
            ),
            BadgeSurface(),
            keep_border=overlay.stash_badge_keep_border,
            ledger=ledger,
        )

    def _apply_status(self, status: str) -> None:
        """Apply a status update to the StatusWindow (main thread only)."""
        if self.status is None:
            return
        if status == "scanning":
            self.status.set_scanning()
        elif status == "active":
            self.status.set_active()
        elif status == "hotkey":
            self.status.set_hotkey()
        elif status == "error":
            self.status.set_error("Error")

    def _pump_ui_queue(self) -> None:
        """Drain scanner UI messages on the main thread, then reschedule."""
        try:
            while True:
                msg = self.ui_queue.get_nowait()
                # The IDLE loop queues a status on every trigger scan, so a
                # status is logged only when it changes
                if msg[0] != "status" or msg[1] != self._last_status:
                    logger.debug("ui apply %s %s", msg[0], _ui_message_detail(msg))
                if msg[0] == "status":
                    self._last_status = msg[1]
                    self._apply_status(msg[1])
                elif msg[0] == "overlay":
                    _, item_name, recommendation, verdict, hold_ms = msg
                    home_outputs = (
                        self.db.recycle_outputs(recommendation.name)
                        if recommendation is not None
                        else None
                    )
                    self.overlay.show(
                        item_name, recommendation, verdict, home_outputs, hold_ms
                    )
                    self._after_popup_show()
                elif msg[0] == "advice":
                    self._show_advice(msg[1], msg[2], msg[3])
                    self._after_popup_show()
                elif msg[0] == "popup_spot":
                    self._apply_popup_spot(msg[1], msg[2])
                elif msg[0] == "popup_home":
                    self._home_box = msg[1]
                    self._place_popup()
                elif msg[0] == "touch":
                    if self.advice is not None:
                        self._touch_advice(msg[1], msg[2])
                    else:
                        self.overlay.touch(msg[1], msg[2])
                    if self.chips is not None:
                        self.chips.touch(msg[2])
                elif msg[0] == "chips" and self.chips is not None:
                    placed = self.chips.show(msg[1], msg[2], **msg[3], hold_ms=msg[5])
                    self._apply_border(msg[4], placed=placed)
                elif msg[0] == "chips_hide" and self.chips is not None:
                    self.chips.hide(reason="chips_hide")
                elif msg[0] == "hide":
                    self._hide_card_windows()
        except queue.Empty:
            pass
        self._publish_overlay_rects()
        self.root.after(50, self._pump_ui_queue)

    def _hide_card_windows(self) -> None:
        """Hide the popup, chips and frame at once (main thread only).

        The scanner queues this when the card is gone. The chips' hide hook
        hides the frame with them.
        """
        if self.overlay.is_visible():
            self.overlay.hide()
        if self.advice is not None:
            self._hide_advice()
        if self.chips is not None:
            self.chips.hide(reason="card_gone")

    def _show_advice(
        self, item_name: str, card: AdviceCard, hold_ms: int | None
    ) -> None:
        """Show the advice card at the popup's spot (main thread only).

        The same card at the same spot is neither rendered nor mapped again,
        only its hold timer restarts.
        """
        self._advice_item = item_name
        self._advice_card = card
        self.advice.show(card, self._advice_box())
        self._note_advice_size()
        self._arm_advice_hold(hold_ms)

    def _touch_advice(self, item_name: str, hold_ms: int | None) -> None:
        """Keep the advice card up while item_name is the item showing."""
        if self._advice_item == item_name and self.advice.is_visible():
            self._arm_advice_hold(hold_ms)

    def _arm_advice_hold(self, hold_ms: int | None) -> None:
        """Restart the advice card's auto-hide timer, display_time without hold."""
        if self._advice_hide_id is not None:
            self.root.after_cancel(self._advice_hide_id)
        if hold_ms is None:
            hold_ms = round(self.settings.overlay.display_time * 1000)
        self._advice_hide_id = self.root.after(hold_ms, self._advice_hold_ended)

    def _advice_hold_ended(self) -> None:
        """The hold ran out without a touch: hide the advice card."""
        self._advice_hide_id = None
        logger.debug("advice hide reason=timer item=%s", self._advice_item)
        self._hide_advice()

    def _hide_advice(self) -> None:
        """Hide the advice card and stop its hold timer (main thread only)."""
        if self._advice_hide_id is not None:
            self.root.after_cancel(self._advice_hide_id)
            self._advice_hide_id = None
        self._advice_item = None
        self.advice.hide()

    def _note_advice_size(self) -> None:
        """Remember the advice card's size after a show, for the spot logic."""
        rect = self.advice.rect_phys()
        if rect is not None:
            self._advice_size = (rect.right - rect.left, rect.bottom - rect.top)

    def _home_rect(self) -> Rect:
        """Home box: beneath the second gun once found, else the fixed one."""
        if self._home_box is not None:
            return self._home_box
        return layout_home.fixed_home(get_screen_resolution())

    def _advice_box(self) -> Rect:
        """The box the advice card is placed in for the current spot.

        Home is the home box. Away it is a box of home's size at the spot's
        top left, cut at the screen's bottom, so the card keeps its scale.
        """
        home = self._home_rect()
        if self._popup_spot == POPUP_HOME:
            return home
        at = self._popup_away_at or self._popup_corner
        screen_h = get_screen_resolution()[1]
        return Rect(
            at.x,
            at.y,
            at.x + home.right - home.left,
            min(screen_h, at.y + home.bottom - home.top),
        )

    def _border_width_px(self) -> int:
        """Border width in physical px, BORDER_WIDTH_TK scaled like the chips."""
        width_tk = max(1, int(BORDER_WIDTH_TK / self.chips.dpi_scale))
        return max(1, round(width_tk * get_backend().tk_scale))

    def _apply_border(self, rect: Rect | None, *, placed: bool) -> None:
        """Show the gold frame at rect, or hide it (main thread only).

        The frame only shows beside chips this layout placed, so it can
        never outlive them.
        """
        if self.border is None:
            return
        if rect is None:
            self.border.hide(reason="no_cell")
        elif not placed:
            self.border.hide(reason="no_chips")
        else:
            self.border.show(rect, self._border_width_px())

    def _popup_spots(self) -> tuple[Rect, Rect]:
        """The popup's home and corner spots at its current size (main thread).

        With the advice card on, the size is the card's: its last shown
        size, the home box before the first show.
        """
        screen = get_screen_resolution()
        home = self._home_rect()
        if self.advice is not None:
            size = self._advice_size
            if size is None:
                size = (home.right - home.left, home.bottom - home.top)
            return popup_spot_rects(screen, size, self._popup_corner, home)
        tk_scale = get_backend().tk_scale
        size = (
            round(self.overlay.window.winfo_width() * tk_scale),
            round(self.overlay.window.winfo_height() * tk_scale),
        )
        return popup_spot_rects(screen, size, self._popup_corner, home)

    def _apply_popup_spot(self, spot: str, rect: Rect | None) -> None:
        """Apply a popup_spot message (main thread only).

        Home needs no rect. Any other spot is the rect's top left, the
        corner spot without a rect is OVERLAY_X and OVERLAY_Y.
        """
        self._popup_spot = spot
        if spot == POPUP_HOME:
            self._popup_away_at = None
        elif rect is not None:
            self._popup_away_at = Point(rect.left, rect.top)
        else:
            self._popup_away_at = self._popup_corner
        self._place_popup()

    def _after_popup_show(self) -> None:
        """Place the popup after a show changed its size (main thread only).

        A taller popup shifts home up to stay above the floor.
        """
        self._place_popup()

    def _place_popup(self) -> None:
        """Move the popup to its current spot if it is elsewhere.

        With the advice card on, a showing card moves instead. A hidden one
        is placed at its spot by the next show.
        """
        if self.advice is not None:
            if self.advice.is_visible() and self._advice_card is not None:
                self.advice.show(self._advice_card, self._advice_box())
                self._note_advice_size()
            return
        home, _ = self._popup_spots()
        spot = (
            Point(home.left, home.top)
            if self._popup_spot == POPUP_HOME
            else self._popup_away_at or self._popup_corner
        )
        if (spot.x, spot.y) != (self.overlay.overlay_x, self.overlay.overlay_y):
            logger.debug(
                "popup move spot=%s x=%d y=%d", self._popup_spot, spot.x, spot.y
            )
            self.overlay.set_position(spot.x, spot.y)

    def _publish_overlay_rects(self) -> None:
        """Publish ARLO's mapped window rectangles for the scan thread."""
        tk_scale = get_backend().tk_scale
        rects: list[Rect] = []
        labelled: dict[str, Rect] = {}
        windows: list[tuple[str, tk.Toplevel]] = []
        if self.status is not None:
            windows.append((STATUS, self.status.window))
        if self.advice is None:
            windows.insert(0, (POPUP, self.overlay.window))
        else:
            # The advice card covers a card just like the popup does
            rect = self.advice.rect_phys()
            if rect is not None:
                rects.append(rect)
                labelled[POPUP] = rect
        if self.debug_overlay is not None:
            windows.append(("debug", self.debug_overlay.window))
        for label, window in windows:
            rect = window_rect_phys(window, tk_scale)
            if rect is not None:
                rects.append(rect)
                labelled[label] = rect
        labelled[POPUP_HOME], labelled[POPUP_CORNER] = self._popup_spots()
        if self.chips is not None:
            rects.extend(self.chips.shown_rects(tk_scale))
        if self.border is not None:
            rects.extend(self.border.bar_rects())
        self.overlay_rects.publish(
            rects, labelled, popup_away=self._popup_spot != POPUP_HOME
        )

    def run(self) -> None:
        """Start the application."""
        logger.info("=" * 50)
        logger.info("ARLO - Started")
        logger.info("=" * 50)
        logger.info(f"Database: {self.db.count()} items loaded")
        logger.info(f"Debug mode: {self.settings.debug_mode}")
        logger.info("=" * 50)
        logger.info("Trigger Region:")
        logger.info(
            f"  Position: ({self.settings.trigger_region.x}, {self.settings.trigger_region.y})"
        )
        logger.info(
            f"  Size: {self.settings.trigger_region.width}x{self.settings.trigger_region.height}"
        )
        logger.info("Trigger Region 2:")
        logger.info(
            f"  Position: ({self.settings.trigger_region2.x}, {self.settings.trigger_region2.y})"
        )
        logger.info(
            f"  Size: {self.settings.trigger_region2.width}x{self.settings.trigger_region2.height}"
        )
        logger.info("=" * 50)
        logger.info(
            f"Trigger scan interval: {self.settings.scan.trigger_scan_interval}s"
        )
        logger.info(
            f"Tooltip scan interval: {self.settings.scan.tooltip_scan_interval}s"
        )
        logger.info("=" * 50)
        logger.info("Looking for INVENTORY screen...")
        logger.info(
            "Hold the hotkey to force tooltip scanning (Ctrl+Shift; "
            f"{get_settings().hotkey_key} on Wayland)"
        )
        logger.info("Press Ctrl+C in terminal to quit")
        logger.info("=" * 50)

        # Start scanner
        self.scanner.start()

        # Start the main-thread UI pump (must be scheduled from the main thread)
        self._pump_ui_queue()

        # Run Tk mainloop
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.quit()

    def quit(self) -> None:
        """Clean shutdown."""
        logger.info("\nShutting down...")
        self.scanner.stop()
        if self.stash is not None:
            self.stash.stop()
        from arc_helper.screen import reset_backend

        reset_backend()  # release the portal session / PipeWire stream
        self.root.quit()
        self.root.destroy()


def check_first_run() -> bool:
    """
    Check if this is a first run or uncalibrated state.

    Returns True if ready to run, False if calibration needed.
    """
    profile_manager = get_profile_manager()
    resolution = profile_manager.get_resolution_key()

    if profile_manager.is_uncalibrated():
        logger.info(f"Detected resolution: {resolution}")
        logger.info("Settings are uncalibrated, checking for profile...")

        if profile_manager.has_profile():
            logger.info(f"Found profile for {resolution}, applying...")
            profile_manager.apply_profile()

            # Force complete reload of everything
            SettingsManager.reload()
            OCREngineManager.reset()

            # Verify the reload worked
            new_settings = SettingsManager.get()
            logger.info(
                f"After reload - Trigger region: ({new_settings.trigger_region.x}, {new_settings.trigger_region.y})"
            )

            logger.info("Profile applied successfully!")
            return True
        supported = profile_manager.get_supported_resolutions()
        logger.warning(f"No pre-configured profile found for {resolution}.")
        logger.info(
            f"Supported resolutions: {', '.join(supported) if supported else 'None yet'}"
        )
        logger.info("Please run the Calibration tool to configure screen regions.")
        return False

    return True


def _positive_int(value: str) -> int:
    """argparse type for a whole number of 1 or more."""
    try:
        number = int(value)
    except ValueError:
        number = 0
    if number < 1:
        msg = f"must be a whole number of 1 or more, got {value!r}"
        raise argparse.ArgumentTypeError(msg)
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line, exiting with status 2 on a bad argument."""
    parser = argparse.ArgumentParser(
        prog="arc-helper", description="Arc Raiders Loot Overlay"
    )
    parser.add_argument(
        "--capture",
        nargs="?",
        const=DEFAULT_CAPTURE_LIMIT,
        default=None,
        type=_positive_int,
        metavar="N",
        help=(
            "save raw item card captures for this run only, "
            f"at most N images (default {DEFAULT_CAPTURE_LIMIT})"
        ),
    )
    return parser.parse_args(argv)


def apply_capture_args(args: argparse.Namespace) -> None:
    """Turn the capture library on for this run when --capture was given."""
    if capture_library_in_env():
        logger.warning(
            "CAPTURE_LIBRARY in .env is ignored, start ARLO with --capture to collect captures"
        )
    if args.capture is None:
        return
    SettingsManager.enable_capture_library(args.capture)
    folder = get_settings().capture_library_dir
    logger.info(
        f"Capture library on for this run: up to {args.capture} images in {folder}"
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point for ARLO."""
    # Before calibration, the OCR engine and the window: --help and bad
    # arguments exit here, and the engine sees the capture switch at init
    apply_capture_args(parse_args(argv))

    # Log any unhandled exception before exit
    def exception_hook(exc_type, exc_value, exc_tb):
        logger.error("=" * 50)
        logger.error("UNHANDLED EXCEPTION - APP CRASHING")
        logger.error("=" * 50)
        logger.error("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
        # Also write to a crash log file
        crash_log = APP_DIR / "crash.log"
        Path(crash_log).write_text(
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
            encoding="utf-8",
        )
        logger.error(f"Crash log written to: {crash_log}")
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = exception_hook

    def thread_exception_hook(args):
        logger.error("=" * 50)
        logger.error(f"THREAD EXCEPTION in {args.thread.name}")
        logger.error("=" * 50)
        logger.error(
            "".join(
                traceback.format_exception(
                    args.exc_type, args.exc_value, args.exc_traceback
                )
            )
        )
        crash_log = APP_DIR / "crash.log"
        with Path(crash_log).open("a", encoding="utf-8") as f:
            f.write(f"\n\nTHREAD {args.thread.name}:\n")
            f.write(
                "".join(
                    traceback.format_exception(
                        args.exc_type, args.exc_value, args.exc_traceback
                    )
                )
            )

    threading.excepthook = thread_exception_hook

    try:
        # Check first run / calibration status
        if not check_first_run():
            input("\nPress Enter to exit...")
            return

        # Auto-update item database from wiki (throttled to once per 24h)
        try:
            sys.path.insert(0, str(APP_DIR))
            from update_db import auto_update

            auto_update()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Wiki auto-update unavailable: {e}")

        # Load recycle_outputs.csv and needs.csv into the database
        try:
            Database().sync_from_csvs()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not load the recycle and needs CSVs: {e}")

        settings = get_settings()
        logger.info(
            f"Settings loaded for resolution, trigger at ({settings.trigger_region.x}, {settings.trigger_region.y})"
        )

        _ = get_ocr_engine()
        app = Application()
        app.run()

    except Exception as e:  # noqa: BLE001
        logger.error(f"Fatal error in main: {e}")
        logger.error(traceback.format_exc())
        input("\nPress Enter to exit after error...")


if __name__ == "__main__":
    main()
