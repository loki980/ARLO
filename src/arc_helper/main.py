"""
ARLO - Main Application.
Coordinates OCR scanning and overlay display.
"""

import argparse
import datetime
import queue
import sys
import threading
import time
import tkinter as tk
import traceback
from collections import deque
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from enum import auto
from pathlib import Path
from threading import Thread

import numpy as np
from dotenv import load_dotenv

from arc_helper.cell_border import CellBorder
from arc_helper.clickthrough import make_click_through
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
from arc_helper.hotkey import HotkeyMonitor
from arc_helper.hotkey import get_hotkey_monitor
from arc_helper.hovered_cell import find_hovered_cell
from arc_helper.ocr import OCREngineManager
from arc_helper.ocr import CardSignature
from arc_helper.ocr import OCREngine
from arc_helper.ocr import get_ocr_engine
from arc_helper.ocr import same_card
from arc_helper.overlay import RECYCLE_LABEL_HOME_ONLY
from arc_helper.overlay import RECYCLE_LABEL_HOME_RECYCLER
from arc_helper.overlay import OverlayWindow
from arc_helper.overlay import StatusWindow
from arc_helper.overlay import TooltipChips
from arc_helper.overlay import action_word
from arc_helper.overlay import window_rect_phys
from arc_helper.resolution_profiles import get_profile_manager
from arc_helper.screen import ScreenCaptureUnavailable
from arc_helper.screen import get_backend
from arc_helper.tooltip_layout import Point
from arc_helper.tooltip_layout import Rect
from arc_helper.tooltip_layout import to_screen
from arc_helper.verdict import KEEP
from arc_helper.verdict import UNKNOWN
from arc_helper.verdict import Verdict
from arc_helper.verdict import compute_verdict

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
# While the same card stays up its windows are touched at most this often,
# well inside HOLD_FLOOR_S
TOUCH_EVERY_S = 1.0

# Gold frame around the hovered cell for keep items, in Tk units scaled the
# way the chips scale
BORDER_WIDTH_TK = 3


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
    by design. popup_away is True while the popup sits in its corner spot.
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
# ARLO windows that make a read unusable when they cover the card
COVERING_WINDOWS = (POPUP, STATUS)


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


def popup_escape_spot(card: Rect, labelled: Mapping[str, Rect]) -> str | None:
    """Spot the popup moves to so it stops covering the card.

    The other spot from the one the card overlaps, and only when the card
    leaves that one free. None when neither spot helps.
    """
    home = labelled.get(POPUP_HOME)
    corner = labelled.get(POPUP_CORNER)
    if home is None or corner is None:
        return None
    on_home = rects_intersect(card, home)
    on_corner = rects_intersect(card, corner)
    if on_home and not on_corner:
        return POPUP_CORNER
    if on_corner and not on_home:
        return POPUP_HOME
    return None


# The popup's home: the empty part of the LOADOUT panel below the second
# weapon slot, as fractions of the screen size. Its top left is the popup's
# top left. A taller popup grows downward but never below POPUP_HOME_FLOOR.
POPUP_HOME_LEFT = 0.356
POPUP_HOME_TOP = 0.757
POPUP_HOME_FLOOR = 0.93


def popup_spot_rects(
    screen: tuple[int, int], size: tuple[int, int], corner: Point
) -> tuple[Rect, Rect]:
    """The popup's home and top left corner spots in physical px.

    screen is the screen size, size the popup's size and corner the top
    left spot (OVERLAY_X and OVERLAY_Y), used while a card covers home.
    """
    screen_w, screen_h = screen
    width, height = size
    x = round(screen_w * POPUP_HOME_LEFT)
    y = min(
        round(screen_h * POPUP_HOME_TOP), round(screen_h * POPUP_HOME_FLOOR) - height
    )
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


def hovered_cell_on_screen(
    capture, origin: Point | None, card: Rect | None, ignore_phys: Sequence[Rect]
) -> Rect | None:
    """Find the hovered cell in a cursor capture, in physical screen px.

    capture is the PIL image the card was found in, origin its top left on
    screen and card the item card in capture coordinates. ignore_phys are
    ARLO's own windows in physical screen px. None when anything is missing
    or no glow ring is found. Runs on the scan thread.
    """
    if capture is None or origin is None or card is None:
        return None
    rgb = np.asarray(capture.convert("RGB"))
    start = time.perf_counter()
    cell = find_hovered_cell(rgb, card, screen_to_capture(ignore_phys, origin))
    logger.debug(
        "border search found=%s duration=%.1f ms",
        "yes" if cell is not None else "no",
        (time.perf_counter() - start) * 1000,
    )
    return to_screen(cell, origin) if cell is not None else None


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
    item: Item,
    outputs: list[RecycleOutput],
    home_only: list[RecycleOutput],
    verdict: Verdict | None = None,
) -> dict[str, str | None]:
    """Chip texts for TooltipChips.show. None hides that line or chip.

    Without field rows the recycle chip lists every home output. With field
    rows it lists only the home extras, and hides when there are none. The
    verdict word and kind come from the verdict, never from item.action.
    """
    price_text = stack_text = None
    if item.sell_price is not None:
        price_text = f"SELLS FOR {item.sell_price:,} \u20a1"
        if item.stack_size is not None:
            stack_total = item.sell_price * item.stack_size
            stack_text = f"\u00d7{item.stack_size} = {stack_total:,} \u20a1"

    recycle_label = recycle_text = None
    home = [o for o in outputs if o.location == HOME]
    has_field = any(o.location == FIELD for o in outputs)
    if home and not has_field:
        recycle_label = RECYCLE_LABEL_HOME_RECYCLER
        recycle_text = "  ".join(f"{o.quantity} {o.component}" for o in home)
    elif home and home_only:
        recycle_label = RECYCLE_LABEL_HOME_ONLY
        recycle_text = "  ".join(f"+{o.quantity} {o.component}" for o in home_only)

    verdict_shown = verdict is not None and verdict.kind != UNKNOWN
    return {
        "price_text": price_text,
        "stack_text": stack_text,
        "recycle_label": recycle_label,
        "recycle_text": recycle_text,
        "verdict_word": action_word(verdict.kind) if verdict_shown else None,
        "verdict_kind": verdict.kind if verdict_shown else None,
    }


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
    status: StatusWindow
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

    # Cooldown tracking
    _last_shown_item: str = ""
    _last_shown_time: float = 0
    # Monotonic time of the last INVENTORY trigger check while ACTIVE
    _last_trigger_check: float = 0.0

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

    # Scan periods measured in this ACTIVE stretch (seconds) and the hold
    # computed after the latest scan (ms), sent with every overlay, touch and
    # chips message. None until the first scan, which leaves display_time.
    _scan_periods: deque = field(default_factory=lambda: deque(maxlen=HOLD_HISTORY))
    _hold_ms: int | None = None

    # Hotkey override (Ctrl+Shift to force tooltip scanning)
    _hotkey_override: bool = False

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
            self._hide_chips()
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
                self._hide_chips()

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
                        # Wait before next trigger scan
                        time.sleep(settings.scan.trigger_scan_interval)

                    self.stats.trigger_scans += 1

                # Phase 2: Active mode - scan tooltip
                elif self.state == ScannerState.ACTIVE:
                    if self._hotkey_override:
                        # Hotkey mode: stay active as long as hotkey is held
                        if not hotkey_now:
                            self._hotkey_override = False
                            self.state = ScannerState.IDLE
                            logger.debug("state ACTIVE -> IDLE reason=hotkey_released")
                            self._hide_chips()
                            self._update_status("scanning")
                            logger.info("Hotkey released - returning to idle")
                            continue
                    elif self._trigger_lost(ocr, settings):
                        # Inventory closed: the card went with it
                        self.state = ScannerState.IDLE
                        logger.debug("state ACTIVE -> IDLE reason=trigger_lost")
                        self._hide_card_windows()
                        self._update_status("scanning")
                        logger.info("INVENTORY closed - returning to idle")
                        continue

                    self._scan_cycle(ocr, settings)

            except ScreenCaptureUnavailable as e:
                logger.error(f"Screen capture lost permanently: {e}")
                self._update_status("error")
                self.state = ScannerState.STOPPED
                self._hide_chips()
                break  # latch: no point scanning without capture
            except Exception as e:  # noqa: BLE001
                logger.error(f"Scanner error: {e}")
                self._update_status("error")
                time.sleep(1.0)  # Back off on error

    def _reset_scan_timing(self) -> None:
        """Forget scan timing and the shown card on entry to ACTIVE.

        The trigger was just checked (or the hotkey is held), so the next
        trigger recheck is TRIGGER_RECHECK_S away.
        """
        self._last_scan_start = None
        self._scan_periods.clear()
        self._last_trigger_check = time.monotonic()
        self._card_shown = None
        self._card_missed = 0
        self._card_state = "none"

    def _trigger_lost(self, ocr: OCREngine, settings) -> bool:
        """Recheck INVENTORY about once a second, True when it is gone.

        The recheck runs on elapsed time, not on a cycle count, so fast
        cycles do not check it more often.
        """
        now = time.monotonic()
        if now - self._last_trigger_check < TRIGGER_RECHECK_S:
            return False
        self._last_trigger_check = now
        return not ocr.check_trigger_any(
            [settings.trigger_region, settings.trigger_region2]
        )

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
        # A failed or discarded read keeps the shown card, so the next cycle
        # reads this card again and the shown one is still known
        if item_name and self._handle_detected_item(item_name, new_card=True):
            self._card_shown = sighting
            self._last_touch = time.monotonic()

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
                    self._queue_popup_spot(popup_escape_spot(card, labelled))
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
            logger.debug(f"Unknown item: {item_name}")
            # Log to missing items file for easier database updates
            self.db.log_missing_item(item_name)

        verdict = self._verdict_for(recommendation, settings.show_ended_projects)

        # A new item may bring the popup back home. The same item never
        # moves it. Home depends on the new item's popup size (a taller
        # popup shifts it up, a shorter one down), so every candidate goes
        # to the Tk thread with its card, which moves the popup only when
        # the card leaves home free at the size after the show.
        if self.overlay_rects.popup_away() and not same_item and card is not None:
            self._queue_popup_spot(POPUP_HOME, card)

        # Show overlay (must be done on main thread)
        self._show_overlay(shown_name, recommendation, verdict)

        if settings.overlay.tooltip_chips:
            self._queue_chips(recommendation, verdict=verdict, same_item=same_item)

        # Update cooldown tracking
        self._last_shown_item = shown_name
        self._last_shown_time = current_time
        return True

    @staticmethod
    def _card_on_screen() -> Rect | None:
        """The last scan's item card in physical screen px, None without one."""
        engine = get_ocr_engine()
        origin = getattr(engine, "_last_origin", None)
        card = getattr(engine, "_last_card", None)
        if origin is None or card is None:
            return None
        return to_screen(card, origin)

    def _queue_popup_spot(self, spot: str | None, card: Rect | None = None) -> None:
        """Queue the popup's move to a spot. None queues nothing.

        With a card, the move home waits for the next popup show and only
        happens when the card leaves home free at the popup's new size.
        """
        if spot is None:
            return
        self.ui_queue.put(("popup_spot", spot, card))

    def _verdict_for(self, recommendation: Item | None, show_ended: bool) -> Verdict:  # noqa: FBT001
        """Advice for the item from its needs and the recorded progress.

        The owned count is unknown until the stash ledger exists. Project end
        dates compare against the local date.
        """
        today = datetime.datetime.now().astimezone().date()
        if recommendation is None:
            return compute_verdict(None, [], {}, None, today, show_ended)
        home = [
            o
            for o in self.db.recycle_outputs(recommendation.name)
            if o.location == HOME
        ]
        return compute_verdict(
            recommendation,
            self.db.needs_for(recommendation.name),
            self.db.progress_map(),
            None,
            today,
            show_ended,
            blueprint=self.db.blueprint_for(recommendation.name),
            home_outputs=home,
        )

    def _show_overlay(
        self, item_name: str, recommendation: Item | None, verdict: Verdict
    ) -> None:
        """Queue an overlay update for the main (GUI) thread to apply."""
        self.ui_queue.put(
            ("overlay", item_name, recommendation, verdict, self._hold_ms)
        )

    def _queue_chips(
        self,
        recommendation: Item | None,
        *,
        verdict: Verdict | None = None,
        same_item: bool = False,
    ) -> None:
        """Locate the tooltip and queue the chips for the current item.

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
            recommendation,
            self.db.recycle_outputs(recommendation.name),
            self.db.home_only_outputs(recommendation.name),
            verdict,
        )
        border = None
        if verdict is not None and verdict.kind == KEEP:
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
    if msg[0] in {"touch", "overlay", "chips"}:
        return f"item={msg[1]}"
    if msg[0] == "popup_spot":
        return f"spot={msg[1]}"
    return ""


class Application:
    """Main application controller."""

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
        # OVERLAY_X and OVERLAY_Y now set the top left corner spot the popup
        # moves to while a card covers its home in the LOADOUT panel
        self._popup_corner = Point(self.overlay.overlay_x, self.overlay.overlay_y)
        self._popup_spot = POPUP_HOME
        # Card of a queued return home, checked after the next popup show
        self._home_card: Rect | None = None
        self._place_popup()
        self.status = StatusWindow(self.root, on_quit=self.quit)
        self.chips = (
            TooltipChips(self.root) if self.settings.overlay.tooltip_chips else None
        )
        # Gold frame around the hovered keep item, hidden with the chips
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
        self.scanner = Scanner(
            root=self.root,
            overlay=self.overlay,
            status=self.status,
            db=self.db,
            ui_queue=self.ui_queue,
        )
        self.overlay_rects = self.scanner.overlay_rects

        # Bind close handler
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    def _apply_status(self, status: str) -> None:
        """Apply a status update to the StatusWindow (main thread only)."""
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
                elif msg[0] == "popup_spot":
                    self._apply_popup_spot(msg[1], msg[2])
                elif msg[0] == "touch":
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
        if self.chips is not None:
            self.chips.hide(reason="card_gone")

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
        """The popup's home and corner spots at its current size (main thread)."""
        tk_scale = get_backend().tk_scale
        size = (
            round(self.overlay.window.winfo_width() * tk_scale),
            round(self.overlay.window.winfo_height() * tk_scale),
        )
        return popup_spot_rects(get_screen_resolution(), size, self._popup_corner)

    def _apply_popup_spot(self, spot: str, card: Rect | None) -> None:
        """Apply a popup_spot message (main thread only).

        A move home with a card waits for the next popup show, see
        _after_popup_show. Any other move happens now.
        """
        if spot == POPUP_HOME and card is not None:
            self._home_card = card
            return
        self._home_card = None
        self._popup_spot = spot
        self._place_popup()

    def _after_popup_show(self) -> None:
        """Place the popup after a show changed its size (main thread only).

        A taller popup shifts home up to stay above the floor. A pending
        return home only happens when the card leaves home free at the new
        size, else the popup stays in its corner.
        """
        if self._home_card is not None:
            home, _ = self._popup_spots()
            if rects_intersect(self._home_card, home):
                logger.debug("popup stays in corner: card covers home at new size")
            else:
                self._popup_spot = POPUP_HOME
            self._home_card = None
        self._place_popup()

    def _place_popup(self) -> None:
        """Move the popup to its current spot if it is elsewhere."""
        home, corner = self._popup_spots()
        spot = corner if self._popup_spot == POPUP_CORNER else home
        if (spot.left, spot.top) != (self.overlay.overlay_x, self.overlay.overlay_y):
            logger.debug(
                "popup move spot=%s x=%d y=%d", self._popup_spot, spot.left, spot.top
            )
            self.overlay.set_position(spot.left, spot.top)

    def _publish_overlay_rects(self) -> None:
        """Publish ARLO's mapped window rectangles for the scan thread."""
        tk_scale = get_backend().tk_scale
        rects: list[Rect] = []
        labelled: dict[str, Rect] = {}
        windows = [(POPUP, self.overlay.window), (STATUS, self.status.window)]
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
            rects, labelled, popup_away=self._popup_spot == POPUP_CORNER
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
