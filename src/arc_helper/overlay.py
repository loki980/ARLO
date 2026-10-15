"""
Overlay UI for ARLO (Arc Raiders Loot Overlay).
Displays item recommendations as a styled popup with action colors.
"""

import tkinter as tk
from dataclasses import replace
from tkinter import font as tkfont
from typing import Callable

from .clickthrough import make_click_through
from .config import get_dpi_scale
from .config import get_screen_resolution
from .config import get_settings
from .config import logger
from .database import HOME
from .database import Item
from .database import RecycleOutput
from .screen import get_backend
from .screen import phys_to_tk
from .textrender import CONDENSED_FAMILY
from .textrender import TextStyle
from .textrender import find_font_file
from .textrender import has_scalable_fonts
from .textrender import render_photo
from .tooltip_layout import PRICE_CHIP
from .tooltip_layout import RECYCLE_CHIP
from .tooltip_layout import VERDICT_CHIP
from .tooltip_layout import Rect
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

# Verdict chip: dark slate panel and the word size in points
VERDICT_BG = "#23272e"
VERDICT_SIZE = 32

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


def _make_label(
    root: tk.Misc,
    parent: tk.Misc,
    size: int,
    *,
    bold: bool = False,
    condensed: bool = False,
    fg: str,
    bg: str,
    **options,
) -> tk.Label:
    """Build a text label. Without scalable Tk fonts it shows Pillow images.

    condensed asks for Barlow Semi Condensed, used by the tooltip chips.
    """
    family = "Segoe UI"
    if condensed and has_scalable_fonts(root) and CONDENSED_FAMILY in _families(root):
        family = CONDENSED_FAMILY
    font = (family, size, *(("bold",) if bold else ()))
    label = tk.Label(parent, text="", font=font, fg=fg, bg=bg, **options)
    label._arlo_style = None  # noqa: SLF001
    if not has_scalable_fonts(root):
        path = find_font_file(bold=bold, condensed=condensed)
        if path:
            label._arlo_style = TextStyle(size, bold, fg, bg)  # noqa: SLF001
            label._arlo_fallback = find_font_file(bold=bold) if condensed else None  # noqa: SLF001
            label._arlo_font = path  # noqa: SLF001
            label._arlo_root = root  # noqa: SLF001
            label._arlo_text = ""  # noqa: SLF001
            label._arlo_wrap = 0  # noqa: SLF001
    return label


def _families(root: tk.Misc) -> tuple[str, ...]:
    try:
        return tuple(tkfont.families(root))
    except tk.TclError:
        return ()


def _set_text(
    label: tk.Label,
    text: str,
    *,
    fg: str | None = None,
    wraplength: int | None = None,
) -> None:
    """Set a label's text, colour and wrap width on the Tk or the image path.

    wraplength is in Tk pixels, left as it was when None.
    """
    style = getattr(label, "_arlo_style", None)
    if not isinstance(style, TextStyle):
        options: dict = {}
        if text is not None:
            options["text"] = text
        if fg is not None:
            options["fg"] = fg
        if wraplength is not None:
            options["wraplength"] = wraplength
        label.config(**options)
        return
    if fg is not None:
        style = replace(style, fg=fg)
        label._arlo_style = style  # noqa: SLF001
    if wraplength is not None:
        label._arlo_wrap = wraplength  # noqa: SLF001
    label._arlo_text = text  # noqa: SLF001
    if not text:
        label._arlo_photo = None  # noqa: SLF001
        label.config(image="", text="")
        return
    photo = render_photo(
        label._arlo_root,  # noqa: SLF001
        text,
        style,
        label._arlo_font,  # noqa: SLF001
        label._arlo_wrap,  # noqa: SLF001
        label._arlo_fallback,  # noqa: SLF001
    )
    # Keep a reference, Tk does not hold the image alive
    label._arlo_photo = photo  # noqa: SLF001
    label.config(image=photo, text="")


def _rewrap(label: tk.Label, wraplength: int) -> None:
    """Change the wrap width, keeping the text the label already shows."""
    style = getattr(label, "_arlo_style", None)
    text = label._arlo_text if isinstance(style, TextStyle) else None  # noqa: SLF001
    if text is None:
        label.config(wraplength=wraplength)
    else:
        _set_text(label, text, wraplength=wraplength)


def _hide_delay_ms(settings, hold_ms: int | None) -> int:
    """Auto-hide delay: the scanner's hold, or display_time without one."""
    if hold_ms is None:
        return int(settings.overlay.display_time * 1000)
    return hold_ms


def tk_box_to_phys(x: int, y: int, width: int, height: int, tk_scale: float) -> Rect:
    """Rectangle of a Tk window box (tk units) in physical screen pixels."""
    return Rect(
        round(x * tk_scale),
        round(y * tk_scale),
        round((x + width) * tk_scale),
        round((y + height) * tk_scale),
    )


def window_rect_phys(window: tk.Misc, tk_scale: float) -> Rect | None:
    """Screen rectangle of a mapped window in physical pixels, else None.

    Tk thread only.
    """
    try:
        if not window.winfo_viewable():
            return None
        return tk_box_to_phys(
            window.winfo_rootx(),
            window.winfo_rooty(),
            window.winfo_width(),
            window.winfo_height(),
            tk_scale,
        )
    except tk.TclError:
        return None


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
        # Delay the pending hide timer was armed with, for the debug log
        self._hide_delay_ms: int | None = None
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
        self.name_label = _make_label(
            self.root,
            self.frame,
            name_size,
            bold=True,
            fg="white",
            bg=BG_PANEL,
            anchor="w",
        )
        _set_text(self.name_label, "Item Name")
        self.name_label.pack(anchor="w")

        # Action label (large, colored)
        self.action_label = _make_label(
            self.root,
            self.frame,
            action_size,
            bold=True,
            fg="#FFD700",
            bg=BG_PANEL,
            anchor="w",
        )
        _set_text(self.action_label, "ACTION")
        self.action_label.pack(anchor="w", pady=(self._scale(4), 0))

        # Reason line under the action (why keep, sell or recycle)
        self.reason_label = _make_label(
            self.root,
            self.frame,
            info_size,
            fg="white",
            bg=BG_PANEL,
            justify=tk.LEFT,
            anchor="w",
        )
        self.reason_label.pack(anchor="w")

        # Sell price / stack size info line
        self.info_label = _make_label(
            self.root,
            self.frame,
            info_size,
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
        self.notes_label = _make_label(
            self.root,
            self.frame,
            detail_size,
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
        hold_ms: int | None = None,
    ) -> None:
        """Show the overlay with item info.

        verdict picks the action word, its color and the reason line. When
        None it is derived from the item's action string. home_outputs are
        the item's recycle_outputs rows for the notes line. hold_ms is the
        auto-hide delay, display_time when None.
        """
        # Skip redraw if same item already showing
        if self._current_item == item_name and self.is_visible():
            logger.debug("popup show item=%s redraw=no", item_name)
            self._reschedule_hide(hold_ms)
            return

        logger.debug("popup show item=%s redraw=yes", item_name)
        self._current_item = item_name

        if self._hide_after_id:
            self.window.after_cancel(self._hide_after_id)
            logger.debug("popup hide timer cancelled delay=%s ms", self._hide_delay_ms)

        if verdict is None:
            verdict = verdict_from_action(recommendation)

        # Update content
        _set_text(self.name_label, popup_title(item_name, recommendation))

        color = action_color(verdict.kind)
        _set_text(self.action_label, action_word(verdict.kind), fg=color)
        self.accent_bar.config(bg=color)
        _set_text(self.reason_label, verdict.reason)

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
        _set_text(self.info_label, info_text)

        # Show or hide separator and notes based on content
        self.separator.pack_forget()
        self.notes_label.pack_forget()
        if detail_text:
            self.separator.pack(fill="x", pady=(self._scale(8), self._scale(6)))
            self.notes_label.pack(anchor="w")
            _set_text(self.notes_label, detail_text)

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
        _rewrap(self.notes_label, max_width)
        _rewrap(self.reason_label, max_width)

        # Auto-hide
        hide_ms = _hide_delay_ms(self.settings, hold_ms)
        self._hide_after_id = self.window.after(hide_ms, self.hide)
        self._hide_delay_ms = hide_ms
        logger.debug("popup hide timer armed delay=%d ms", hide_ms)

    def _reschedule_hide(self, hold_ms: int | None = None) -> None:
        """Restart the auto-hide timer from now, display_time when no hold."""
        if self._hide_after_id:
            self.window.after_cancel(self._hide_after_id)
            logger.debug("popup hide timer cancelled delay=%s ms", self._hide_delay_ms)
        hide_ms = _hide_delay_ms(self.settings, hold_ms)
        self._hide_after_id = self.window.after(hide_ms, self.hide)
        self._hide_delay_ms = hide_ms
        logger.debug("popup hide timer armed delay=%d ms", hide_ms)

    def touch(self, item_name: str, hold_ms: int | None = None) -> None:
        """Keep the overlay up while item_name is the item showing."""
        if self._current_item == item_name and self.is_visible():
            logger.debug("popup touch item=%s restart=yes", item_name)
            self._reschedule_hide(hold_ms)
        else:
            logger.debug(
                "popup touch item=%s restart=no showing=%s",
                item_name,
                self._current_item,
            )

    def hide(self) -> None:
        """Withdraw the popup. Only its own auto-hide timer calls this."""
        logger.debug("popup hide reason=timer item=%s", self._current_item)
        # The id here is the timer that just fired, so this cancel is not logged
        if self._hide_after_id:
            self.window.after_cancel(self._hide_after_id)
            self._hide_after_id = None
        self._current_item = None
        self.window.withdraw()

    def set_position(self, x: int, y: int) -> None:
        """Move the popup to x, y in physical px, at once when it is showing."""
        self.overlay_x = x
        self.overlay_y = y
        if self.is_visible():
            self.window.geometry(f"+{phys_to_tk(x)}+{phys_to_tk(y)}")

    def is_visible(self) -> bool:
        try:
            return self.window.winfo_viewable()
        except tk.TclError:
            return False


class TooltipChips:
    """Three small click-through chips placed on the game's own tooltip.

    The price chip sits on the "Can be found in" row, the recycle chip on
    the icon row under "Recycles into", and the verdict chip above the
    recycle chip. Positions come from place_chips in
    physical pixels and are converted with phys_to_tk.
    """

    # Gap between chip and tooltip edge or label, in tk units
    PAD = 8

    # Called with the hide reason whenever the chips hide (timer or
    # chips_hide), so the gold cell border can hide with them
    on_hide: Callable[[str], None] | None = None

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

        self.verdict_window, self.verdict_stripe, verdict_body = (
            self._make_verdict_chip()
        )
        self.verdict_label = _make_label(
            self.root,
            verdict_body,
            VERDICT_SIZE,
            bold=True,
            condensed=True,
            fg="white",
            bg=VERDICT_BG,
            anchor="w",
        )
        self.verdict_label.pack(anchor="w")

        self.windows = {
            PRICE_CHIP: self.price_window,
            RECYCLE_CHIP: self.recycle_window,
            VERDICT_CHIP: self.verdict_window,
        }
        self._hide_after_id: str | None = None
        # Delay the pending hide timer was armed with, for the debug log
        self._hide_delay_ms: int | None = None
        # Item the chips belong to, the chips mapped and their positions,
        # so a repeat show for one item only moves windows that moved
        self._current_item: str | None = None
        self._shown: set[str] = set()
        self._positions: dict[str, tuple[int, int]] = {}

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

    def _make_verdict_chip(self) -> tuple[tk.Toplevel, tk.Frame, tk.Frame]:
        """Build the hidden verdict chip: dark slate panel, left stripe.

        Returns the window, the stripe frame (recoloured per verdict) and the
        content frame. Corners stay square, a Tk toplevel is rectangular.
        """
        window = tk.Toplevel(self.root)
        window.title("ARLO verdict")
        window.attributes("-topmost", True)  # noqa: FBT003
        window.attributes("-alpha", 0.96)
        window.overrideredirect(boolean=True)
        window.withdraw()

        panel = tk.Frame(window, bg=VERDICT_BG)
        panel.pack()
        # About a fifth of the text height, in tk units
        stripe_w = max(3, self._scale(round(VERDICT_SIZE * 1.334 / 5)))
        stripe = tk.Frame(panel, bg=ACTION_COLORS["UNKNOWN"], width=stripe_w)
        stripe.pack(side="left", fill="y")
        body = tk.Frame(panel, bg=VERDICT_BG, padx=self._scale(14), pady=self._scale(6))
        body.pack(side="left")
        make_click_through(window)
        return window, stripe, body

    def _label(
        self, parent: tk.Frame, fg: str, *, size: int | None = None, bold: bool = False
    ) -> tk.Label:
        return _make_label(
            self.root,
            parent,
            size or self.info_size,
            bold=bold,
            condensed=True,
            fg=fg,
            bg=BG_PANEL,
            anchor="w",
        )

    def show(
        self,
        item_name: str,
        layout: TooltipLayout,
        price_text: str | None,
        stack_text: str | None,
        recycle_label: str | None,
        recycle_text: str | None,
        hold_ms: int | None = None,
        verdict_word: str | None = None,
        verdict_kind: str | None = None,
    ) -> bool:
        """Fill, measure, place and show the chips that have content.

        verdict_word and verdict_kind (a Verdict.kind) fill the verdict chip.

        For the item already showing, chips are only moved with geometry,
        never withdrawn and mapped again, so the compositor does not fade
        them. A chip this layout could not place stays where it is. hold_ms
        is the auto-hide delay, display_time when None. Returns whether this
        layout placed any chip.
        """
        same_item = item_name == self._current_item
        logger.debug(
            "chips show item=%s same_item=%s", item_name, "yes" if same_item else "no"
        )
        self._current_item = item_name
        if self._hide_after_id:
            self.root.after_cancel(self._hide_after_id)
            logger.debug("chips hide timer cancelled delay=%s ms", self._hide_delay_ms)
            self._hide_after_id = None

        active: list[str] = []
        if price_text:
            _set_text(self.price_label, price_text)
            self.stack_label.pack_forget()
            if stack_text:
                _set_text(self.stack_label, stack_text)
                self.stack_label.pack(anchor="w")
            active.append(PRICE_CHIP)
        if recycle_label and recycle_text:
            title_fg = (
                self.recycle_accent
                if recycle_label == RECYCLE_LABEL_HOME_ONLY
                else TEXT_INFO
            )
            _set_text(self.recycle_title, recycle_label, fg=title_fg)
            _set_text(self.recycle_text, recycle_text)
            active.append(RECYCLE_CHIP)
        if verdict_word and verdict_kind:
            _set_text(self.verdict_label, verdict_word)
            self.verdict_stripe.config(bg=action_color(verdict_kind))
            active.append(VERDICT_CHIP)

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
                if not same_item:
                    window.withdraw()
                    self._shown.discard(key)
                    self._positions.pop(key, None)
                continue
            position = (phys_to_tk(point.x), phys_to_tk(point.y))
            if self._positions.get(key) != position:
                window.geometry(f"+{position[0]}+{position[1]}")
                self._positions[key] = position
            if key not in self._shown:
                window.deiconify()
                window.lift()
                self._shown.add(key)

        if self._shown:
            hide_ms = _hide_delay_ms(self.settings, hold_ms)
            self._hide_after_id = self.root.after(hide_ms, self.hide)
            self._hide_delay_ms = hide_ms
            logger.debug("chips hide timer armed delay=%d ms", hide_ms)
        return bool(positions)

    def shown_rects(self, tk_scale: float) -> list[Rect]:
        """Physical screen rectangles of the chips now mapped. Tk thread only."""
        rects = []
        for key in self._shown:
            rect = window_rect_phys(self.windows[key], tk_scale)
            if rect is not None:
                rects.append(rect)
        return rects

    def touch(self, hold_ms: int | None = None) -> None:
        """Restart the auto-hide timer when any chip is showing.

        hold_ms is the auto-hide delay, display_time when None.
        """
        try:
            showing = any(w.winfo_viewable() for w in self.windows.values())
        except tk.TclError:
            logger.debug("chips touch restart=no reason=tcl_error")
            return
        if not showing:
            logger.debug("chips touch restart=no reason=none_showing")
            return
        logger.debug("chips touch restart=yes")
        if self._hide_after_id:
            self.root.after_cancel(self._hide_after_id)
            logger.debug("chips hide timer cancelled delay=%s ms", self._hide_delay_ms)
        hide_ms = _hide_delay_ms(self.settings, hold_ms)
        self._hide_after_id = self.root.after(hide_ms, self.hide)
        self._hide_delay_ms = hide_ms
        logger.debug("chips hide timer armed delay=%d ms", hide_ms)

    def hide(self, reason: str = "timer") -> None:
        """Withdraw the chips. reason is timer from the auto-hide timer."""
        logger.debug("chips hide reason=%s item=%s", reason, self._current_item)
        if self._hide_after_id:
            self.root.after_cancel(self._hide_after_id)
            # From the timer the id is the one that just fired, not a cancel
            if reason != "timer":
                logger.debug(
                    "chips hide timer cancelled delay=%s ms", self._hide_delay_ms
                )
            self._hide_after_id = None
        self._current_item = None
        self._shown.clear()
        self._positions.clear()
        for window in self.windows.values():
            window.withdraw()
        if self.on_hide is not None:
            self.on_hide(reason)


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

        self.status_label = _make_label(
            self.root,
            self.frame,
            font_size,
            fg="#888888",
            bg=BG_DARK,
        )
        _set_text(self.status_label, "\u25cf Scanning...")
        self.status_label.pack()
        if self.on_quit:
            self.status_label.bind("<Button-3>", lambda e: self.on_quit())
            self.frame.bind("<Button-3>", lambda e: self.on_quit())

    def set_scanning(self) -> None:
        _set_text(self.status_label, "\u25cf Scanning...", fg="#888888")

    def set_active(self) -> None:
        _set_text(self.status_label, "\u25cf INVENTORY", fg="#32CD32")

    def set_hotkey(self) -> None:
        _set_text(self.status_label, "\u25cf HOTKEY SCAN", fg="#FFD700")

    def set_error(self, message: str) -> None:
        _set_text(self.status_label, f"\u25cf {message}", fg="#FF4444")

    def hide(self) -> None:
        self.window.withdraw()

    def show(self) -> None:
        self.window.deiconify()
