"""
Overlay UI for ARLO (Arc Raiders Loot Overlay).
Displays item recommendations as a styled popup with action colors.
"""

import tkinter as tk
from typing import Callable

from .clickthrough import make_click_through
from .config import get_dpi_scale
from .config import get_screen_resolution
from .config import get_settings
from .database import HOME
from .database import Item
from .database import RecycleOutput
from .screen import get_backend
from .screen import phys_to_tk
from .tooltip_layout import PRICE_CHIP
from .tooltip_layout import RECYCLE_CHIP
from .tooltip_layout import Size
from .tooltip_layout import TooltipLayout
from .tooltip_layout import place_chips
from .verdict import Verdict
from .verdict import verdict_from_action

# Action colors
ACTION_COLORS: dict[str, str] = {
    "SELL": "#FFD700",
    "RECYCLE": "#00CED1",
    "KEEP": "#32CD32",
    "USE": "#FF69B4",
    "TRASH": "#FF4444",
    "UNKNOWN": "#888888",
}

# Background palette
BG_DARK = "#12121f"
BG_PANEL = "#1a1a2e"
BORDER_COLOR = "#3a3a5c"
TEXT_DIM = "#9999bb"
TEXT_INFO = "#c4a0ff"

# Recycle chip labels. HOME_ONLY draws in the accent color, HOME_RECYCLER
# in the neutral info color.
RECYCLE_LABEL_HOME_ONLY = "HOME ONLY"
RECYCLE_LABEL_HOME_RECYCLER = "HOME RECYCLER"


def popup_title(item_name: str, recommendation: Item | None) -> str:
    """The catalog name for a known item, the raw read for an unknown one."""
    return recommendation.name if recommendation is not None else item_name


def action_word(kind: str) -> str:
    """The popup's action line for a verdict kind: KEEP, KEEP SOME, SELL."""
    return kind.replace("_", " ").upper()


def action_color(kind: str) -> str:
    """ACTION_COLORS entry for a verdict kind. keep_some draws as KEEP."""
    word = "KEEP" if kind == "keep_some" else kind.upper()
    return ACTION_COLORS.get(word, ACTION_COLORS["UNKNOWN"])


def popup_notes(recommendation: Item, home_outputs: list[RecycleOutput] | None) -> str:
    """Home recycle outputs as 'Recycles into: N Name', else keep_for text."""
    home = [o for o in home_outputs or [] if o.location == HOME]
    if home:
        return "Recycles into: " + ", ".join(
            f"{o.quantity} {o.component}" for o in home
        )
    return recommendation.keep_for or ""


class OverlayWindow:
    """Transparent overlay window for showing recommendations."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.settings = get_settings()
        self.dpi_scale = get_dpi_scale()

        self.window = tk.Toplevel(root)
        self.window.title("ARLO")
        self.window.attributes("-topmost", True)  # noqa: FBT003
        self.window.attributes("-alpha", 0.92)
        self.window.overrideredirect(boolean=True)
        self.window.withdraw()

        self.overlay_x = self.settings.overlay.x
        self.overlay_y = self.settings.overlay.y

        self._setup_ui()
        # The popup must never swallow game clicks
        make_click_through(self.window)

        self._hide_after_id: str | None = None
        self._current_item: str | None = None

    def _scale(self, value: int) -> int:
        return max(1, int(value / self.dpi_scale))

    def _setup_ui(self) -> None:
        """Build the overlay layout."""
        padx = self._scale(18)
        pady = self._scale(12)
        border_w = max(1, self._scale(2))

        # Font sizes - bigger and bolder
        name_size = max(10, self._scale(18))
        action_size = max(12, self._scale(26))
        info_size = max(8, self._scale(13))
        detail_size = max(8, self._scale(12))

        # Outer frame with border
        self.outer_frame = tk.Frame(
            self.window,
            bg=BORDER_COLOR,
            padx=border_w,
            pady=border_w,
        )
        self.outer_frame.pack()

        # Inner content area
        self.frame = tk.Frame(
            self.outer_frame,
            bg=BG_PANEL,
            padx=padx,
            pady=pady,
        )
        self.frame.pack()

        # Colored accent bar at the top (changes color with action)
        self.accent_bar = tk.Frame(
            self.frame,
            bg="#32CD32",
            height=max(2, self._scale(4)),
        )
        self.accent_bar.pack(fill="x", pady=(0, self._scale(8)))

        # Item name
        self.name_label = tk.Label(
            self.frame,
            text="Item Name",
            font=("Segoe UI", name_size, "bold"),
            fg="white",
            bg=BG_PANEL,
            anchor="w",
        )
        self.name_label.pack(anchor="w")

        # Action label (large, colored)
        self.action_label = tk.Label(
            self.frame,
            text="ACTION",
            font=("Segoe UI", action_size, "bold"),
            fg="#FFD700",
            bg=BG_PANEL,
            anchor="w",
        )
        self.action_label.pack(anchor="w", pady=(self._scale(4), 0))

        # Reason line under the action (why keep, sell or recycle)
        self.reason_label = tk.Label(
            self.frame,
            text="",
            font=("Segoe UI", info_size),
            fg="white",
            bg=BG_PANEL,
            justify=tk.LEFT,
            anchor="w",
        )
        self.reason_label.pack(anchor="w")

        # Sell price / stack size info line
        self.info_label = tk.Label(
            self.frame,
            text="",
            font=("Segoe UI", info_size),
            fg=TEXT_INFO,
            bg=BG_PANEL,
            anchor="w",
        )
        self.info_label.pack(anchor="w", pady=(self._scale(4), 0))

        # Thin separator before details
        self.separator = tk.Frame(
            self.frame,
            bg=BORDER_COLOR,
            height=max(1, self._scale(1)),
        )
        # Only packed when there's detail text (see show())

        # Detail / notes label (recycle_for or keep_for)
        self.notes_label = tk.Label(
            self.frame,
            text="",
            font=("Segoe UI", detail_size),
            fg=TEXT_DIM,
            bg=BG_PANEL,
            justify=tk.LEFT,
            anchor="w",
        )
        # Only packed when there's detail text (see show())

    def show(
        self,
        item_name: str,
        recommendation: Item | None,
        verdict: Verdict | None = None,
        home_outputs: list[RecycleOutput] | None = None,
    ) -> None:
        """Show the overlay with item info.

        verdict picks the action word, its color and the reason line. When
        None it is derived from the item's action string. home_outputs are
        the item's recycle_outputs rows for the notes line.
        """
        # Skip redraw if same item already showing
        if self._current_item == item_name and self.is_visible():
            if self._hide_after_id:
                self.window.after_cancel(self._hide_after_id)
            hide_ms = int(self.settings.overlay.display_time * 1000)
            self._hide_after_id = self.window.after(hide_ms, self.hide)
            return

        self._current_item = item_name

        if self._hide_after_id:
            self.window.after_cancel(self._hide_after_id)

        if verdict is None:
            verdict = verdict_from_action(recommendation)

        # Update content
        self.name_label.config(text=popup_title(item_name, recommendation))

        color = action_color(verdict.kind)
        self.action_label.config(text=f"\u2192 {action_word(verdict.kind)}", fg=color)
        self.accent_bar.config(bg=color)
        self.reason_label.config(text=verdict.reason)

        info_text = detail_text = ""
        if recommendation:
            # Info line
            info_parts: list[str] = []
            if recommendation.sell_price is not None:
                info_parts.append(f"Sell: {recommendation.sell_price:,}\u20a1")
            if recommendation.stack_size is not None:
                info_parts.append(f"Stack: {recommendation.stack_size}")
            info_text = "    \u00b7    ".join(info_parts)
            detail_text = popup_notes(recommendation, home_outputs)
        self.info_label.config(text=info_text)

        # Show or hide separator and notes based on content
        self.separator.pack_forget()
        self.notes_label.pack_forget()
        if detail_text:
            self.separator.pack(fill="x", pady=(self._scale(8), self._scale(6)))
            self.notes_label.pack(anchor="w")
            self.notes_label.config(text=detail_text)

        # Position and show
        self.window.geometry(
            f"+{phys_to_tk(self.overlay_x)}+{phys_to_tk(self.overlay_y)}"
        )
        self.window.deiconify()
        self.window.lift()

        # Update wraplength
        self.window.update_idletasks()
        widths = [
            self.action_label.winfo_width(),
            self.name_label.winfo_width(),
            self.info_label.winfo_width(),
        ]
        max_width = max(*widths, self._scale(280))
        self.notes_label.config(wraplength=max_width)
        self.reason_label.config(wraplength=max_width)

        # Auto-hide
        hide_ms = int(self.settings.overlay.display_time * 1000)
        self._hide_after_id = self.window.after(hide_ms, self.hide)

    def hide(self) -> None:
        if self._hide_after_id:
            self.window.after_cancel(self._hide_after_id)
            self._hide_after_id = None
        self._current_item = None
        self.window.withdraw()

    def set_position(self, x: int, y: int) -> None:
        self.overlay_x = x
        self.overlay_y = y

    def is_visible(self) -> bool:
        try:
            return self.window.winfo_viewable()
        except tk.TclError:
            return False


class TooltipChips:
    """Two small click-through chips placed beside the game's own tooltip.

    The price chip sits on the "Can be found in" row, the recycle chip on
    the icon row under "Recycles into". Positions come from place_chips in
    physical pixels and are converted with phys_to_tk.
    """

    # Gap between chip and tooltip edge or label, in tk units
    PAD = 8

    def __init__(self, root: tk.Tk):
        self.root = root
        self.settings = get_settings()
        self.dpi_scale = get_dpi_scale()

        self.info_size = max(8, self._scale(13))
        self.detail_size = max(8, self._scale(12))

        self.price_window, price_body = self._make_chip(ACTION_COLORS["SELL"])
        self.price_label = self._label(price_body, "white", bold=True)
        self.price_label.pack(anchor="w")
        self.stack_label = self._label(price_body, TEXT_DIM, size=self.detail_size)
        # Only packed when there is stack text (see show())

        self.recycle_accent = ACTION_COLORS["RECYCLE"]
        self.recycle_window, recycle_body = self._make_chip(self.recycle_accent)
        self.recycle_title = self._label(recycle_body, TEXT_INFO, bold=True)
        self.recycle_title.pack(side="left")
        self.recycle_text = self._label(recycle_body, "white")
        self.recycle_text.pack(side="left", padx=(self._scale(10), 0))

        self.windows = {
            PRICE_CHIP: self.price_window,
            RECYCLE_CHIP: self.recycle_window,
        }
        self._hide_after_id: str | None = None

    def _scale(self, value: int) -> int:
        return max(1, int(value / self.dpi_scale))

    def _make_chip(self, accent: str) -> tuple[tk.Toplevel, tk.Frame]:
        """Build one hidden chip window, returning it and its content frame."""
        window = tk.Toplevel(self.root)
        window.title("ARLO chip")
        window.attributes("-topmost", True)  # noqa: FBT003
        window.attributes("-alpha", 0.92)
        window.overrideredirect(boolean=True)
        window.withdraw()

        border_w = max(1, self._scale(2))
        outer = tk.Frame(window, bg=BORDER_COLOR, padx=border_w, pady=border_w)
        outer.pack()
        inner = tk.Frame(outer, bg=BG_PANEL)
        inner.pack()
        tk.Frame(inner, bg=accent, width=max(2, self._scale(4))).pack(
            side="left", fill="y"
        )
        body = tk.Frame(inner, bg=BG_PANEL, padx=self._scale(10), pady=self._scale(5))
        body.pack(side="left")

        # The chips sit on top of the game tooltip and must not eat clicks
        make_click_through(window)
        return window, body

    def _label(
        self, parent: tk.Frame, fg: str, *, size: int | None = None, bold: bool = False
    ) -> tk.Label:
        font = ("Segoe UI", size or self.info_size, *(("bold",) if bold else ()))
        return tk.Label(parent, text="", font=font, fg=fg, bg=BG_PANEL, anchor="w")

    def show(
        self,
        layout: TooltipLayout,
        price_text: str | None,
        stack_text: str | None,
        recycle_label: str | None,
        recycle_text: str | None,
    ) -> None:
        """Fill, measure, place and show the chips that have content."""
        if self._hide_after_id:
            self.root.after_cancel(self._hide_after_id)
            self._hide_after_id = None

        active: list[str] = []
        if price_text:
            self.price_label.config(text=price_text)
            self.stack_label.pack_forget()
            if stack_text:
                self.stack_label.config(text=stack_text)
                self.stack_label.pack(anchor="w")
            active.append(PRICE_CHIP)
        if recycle_label and recycle_text:
            title_fg = (
                self.recycle_accent
                if recycle_label == RECYCLE_LABEL_HOME_ONLY
                else TEXT_INFO
            )
            self.recycle_title.config(text=recycle_label, fg=title_fg)
            self.recycle_text.config(text=recycle_text)
            active.append(RECYCLE_CHIP)

        # Sizes in physical pixels, measured before the chips are mapped
        tk_scale = get_backend().tk_scale
        sizes: dict[str, Size] = {}
        for key in active:
            window = self.windows[key]
            window.update_idletasks()
            sizes[key] = Size(
                round(window.winfo_reqwidth() * tk_scale),
                round(window.winfo_reqheight() * tk_scale),
            )

        screen_w, screen_h = get_screen_resolution()
        positions = place_chips(
            layout.tooltip,
            layout.labels,
            sizes,
            Size(screen_w, screen_h),
            round(self.PAD * tk_scale),
            layout.clipped,
        )

        for key, window in self.windows.items():
            point = positions.get(key)
            if point is None:
                window.withdraw()
                continue
            window.geometry(f"+{phys_to_tk(point.x)}+{phys_to_tk(point.y)}")
            window.deiconify()
            window.lift()

        if positions:
            hide_ms = int(self.settings.overlay.display_time * 1000)
            self._hide_after_id = self.root.after(hide_ms, self.hide)

    def hide(self) -> None:
        if self._hide_after_id:
            self.root.after_cancel(self._hide_after_id)
            self._hide_after_id = None
        for window in self.windows.values():
            window.withdraw()


class StatusWindow:
    """Small status indicator showing scanner state."""

    def __init__(self, root: tk.Tk, *, on_quit: Callable[[], None] | None = None):
        self.root = root
        self.on_quit = on_quit
        self.dpi_scale = get_dpi_scale()

        self.window = tk.Toplevel(root)
        self.window.title("ARLO Status")
        self.window.attributes("-topmost", True)  # noqa: FBT003
        self.window.attributes("-alpha", 0.8)
        self.window.overrideredirect(boolean=True)
        self.window.geometry("+10+10")
        if on_quit:
            self.window.bind("<Button-3>", lambda e: on_quit())  # Right-click to quit

        self._setup_ui()

    def _scale(self, value: int) -> int:
        return max(1, int(value / self.dpi_scale))

    def _setup_ui(self) -> None:
        padx = self._scale(10)
        pady = self._scale(5)
        font_size = max(7, self._scale(10))

        self.frame = tk.Frame(self.window, bg=BG_DARK, padx=padx, pady=pady)
        self.frame.pack()

        self.status_label = tk.Label(
            self.frame,
            text="\u25cf Scanning...",
            font=("Segoe UI", font_size),
            fg="#888888",
            bg=BG_DARK,
        )
        self.status_label.pack()
        if self.on_quit:
            self.status_label.bind("<Button-3>", lambda e: self.on_quit())
            self.frame.bind("<Button-3>", lambda e: self.on_quit())

    def set_scanning(self) -> None:
        self.status_label.config(text="\u25cf Scanning...", fg="#888888")

    def set_active(self) -> None:
        self.status_label.config(text="\u25cf INVENTORY", fg="#32CD32")

    def set_hotkey(self) -> None:
        self.status_label.config(text="\u25cf HOTKEY SCAN", fg="#FFD700")

    def set_error(self, message: str) -> None:
        self.status_label.config(text=f"\u25cf {message}", fg="#FF4444")

    def hide(self) -> None:
        self.window.withdraw()

    def show(self) -> None:
        self.window.deiconify()
