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
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from enum import auto
from pathlib import Path
from threading import Thread

from dotenv import load_dotenv

from arc_helper.clickthrough import make_click_through
from arc_helper.config import APP_DIR
from arc_helper.config import DEFAULT_CAPTURE_LIMIT
from arc_helper.config import SettingsManager
from arc_helper.config import capture_library_in_env
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
from arc_helper.ocr import OCREngineManager
from arc_helper.ocr import get_ocr_engine
from arc_helper.overlay import RECYCLE_LABEL_HOME_ONLY
from arc_helper.overlay import RECYCLE_LABEL_HOME_RECYCLER
from arc_helper.overlay import OverlayWindow
from arc_helper.overlay import StatusWindow
from arc_helper.overlay import TooltipChips
from arc_helper.resolution_profiles import get_profile_manager
from arc_helper.screen import ScreenCaptureUnavailable
from arc_helper.verdict import Verdict
from arc_helper.verdict import compute_verdict

load_dotenv(Path(__file__).with_name(".env"), override=False)


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
    item: Item, outputs: list[RecycleOutput], home_only: list[RecycleOutput]
) -> dict[str, str | None]:
    """Chip texts for TooltipChips.show. None hides that line or chip.

    Without field rows the recycle chip lists every home output. With field
    rows it lists only the home extras, and hides when there are none.
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

    return {
        "price_text": price_text,
        "stack_text": stack_text,
        "recycle_label": recycle_label,
        "recycle_text": recycle_text,
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

    # Cooldown tracking
    _last_shown_item: str = ""
    _last_shown_time: float = 0
    _trigger_check_counter: int = 0

    # Monotonic start of the previous tooltip scan, for the scan period in
    # the debug log. Reset on every entry to ACTIVE.
    _last_scan_start: float | None = None

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
                        self._last_scan_start = None
                        logger.debug("state IDLE -> ACTIVE reason=hotkey")
                        self._update_status("hotkey")
                        logger.info("Hotkey override - activating tooltip scanner")
                    elif ocr.check_trigger_any(
                        [settings.trigger_region, settings.trigger_region2]
                    ):
                        # Trigger detected! Switch to active mode
                        self._hotkey_override = False
                        self.state = ScannerState.ACTIVE
                        self._last_scan_start = None
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
                    else:
                        # Inventory mode: check trigger every 3rd scan
                        self._trigger_check_counter += 1
                        should_check_trigger = self._trigger_check_counter % 3 == 0

                        if should_check_trigger and not ocr.check_trigger_any(
                            [settings.trigger_region, settings.trigger_region2]
                        ):
                            # Inventory closed, go back to idle
                            self.state = ScannerState.IDLE
                            logger.debug("state ACTIVE -> IDLE reason=trigger_lost")
                            self._hide_chips()
                            self._update_status("scanning")
                            logger.info("INVENTORY closed - returning to idle")
                            continue

                    # Scan tooltip at cursor position
                    scan_start = time.monotonic()
                    if self._last_scan_start is None:
                        logger.debug("scan start period=none")
                    else:
                        logger.debug(
                            "scan start period=%d ms",
                            (scan_start - self._last_scan_start) * 1000,
                        )
                    self._last_scan_start = scan_start
                    item_name = ocr.extract_item_name_at_cursor()
                    logger.debug(
                        "scan end %d ms read=%s",
                        (time.monotonic() - scan_start) * 1000,
                        item_name or "none",
                    )
                    self.stats.tooltip_scans += 1

                    if item_name:
                        self._handle_detected_item(item_name)

                    # Wait before next tooltip scan
                    time.sleep(settings.scan.tooltip_scan_interval)

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

    def _handle_detected_item(self, item_name: str) -> None:
        """Handle a detected item name."""
        settings = get_settings()
        current_time = time.time()

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
            and current_time - self._last_shown_time < settings.overlay.cooldown
        ):
            # Still hovered: keep the popup and chips from auto-hiding
            logger.debug(
                "detect read=%s item=%s path=touch since_shown=%d ms",
                item_name,
                shown_name,
                (current_time - self._last_shown_time) * 1000,
            )
            self.ui_queue.put(("touch", shown_name))
            return

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

        # Show overlay (must be done on main thread)
        self._show_overlay(shown_name, recommendation, verdict)

        if settings.overlay.tooltip_chips:
            self._queue_chips(recommendation, same_item=same_item)

        # Update cooldown tracking
        self._last_shown_item = shown_name
        self._last_shown_time = current_time

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
        self.ui_queue.put(("overlay", item_name, recommendation, verdict))

    def _queue_chips(
        self, recommendation: Item | None, *, same_item: bool = False
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
            self.ui_queue.put(("touch", recommendation.name))
            return
        if recommendation is None or layout is None:
            self._hide_chips()
            return

        payload = chip_payload(
            recommendation,
            self.db.recycle_outputs(recommendation.name),
            self.db.home_only_outputs(recommendation.name),
        )
        with self._chips_lock:
            # stop() or pause() may have hidden the chips during the OCR pass
            if self._running and self.state == ScannerState.ACTIVE:
                self.ui_queue.put(("chips", recommendation.name, layout, payload))

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
        self.status = StatusWindow(self.root, on_quit=self.quit)
        self.chips = (
            TooltipChips(self.root) if self.settings.overlay.tooltip_chips else None
        )

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
                    _, item_name, recommendation, verdict = msg
                    home_outputs = (
                        self.db.recycle_outputs(recommendation.name)
                        if recommendation is not None
                        else None
                    )
                    self.overlay.show(item_name, recommendation, verdict, home_outputs)
                elif msg[0] == "touch":
                    self.overlay.touch(msg[1])
                    if self.chips is not None:
                        self.chips.touch()
                elif msg[0] == "chips" and self.chips is not None:
                    self.chips.show(msg[1], msg[2], **msg[3])
                elif msg[0] == "chips_hide" and self.chips is not None:
                    self.chips.hide(reason="chips_hide")
        except queue.Empty:
            pass
        self.root.after(50, self._pump_ui_queue)

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
