"""
Gold frame around the hovered inventory cell.

The frame is four thin bars, each its own topmost, overrideredirect and
click-through Toplevel, so it needs no window transparency. The bars sit
wholly outside the cell rectangle, so the game's own glow ring stays visible
inside the frame.
"""

import tkinter as tk
from typing import NamedTuple

from .clickthrough import make_click_through
from .config import logger
from .screen import phys_to_tk
from .tooltip_layout import Rect

# The hovered-cell gold, the one gold every ARLO frame and border uses
HOVER_GOLD = "#ffc83d"


class FrameBars(NamedTuple):
    """The four bars of a frame, in the same coordinates as the cell."""

    top: Rect
    bottom: Rect
    left: Rect
    right: Rect


def frame_bars(rect: Rect, width: int) -> FrameBars:
    """Return the four bars of a frame of the given width around rect.

    The top and bottom bars span the corners, the side bars only the cell's
    own height, so the bars never overlap and none covers the cell.
    """
    left, top, right, bottom = rect
    return FrameBars(
        top=Rect(left - width, top - width, right + width, top),
        bottom=Rect(left - width, bottom, right + width, bottom + width),
        left=Rect(left - width, top, left, bottom),
        right=Rect(right, top, right + width, bottom),
    )


def _tk_geometry(bar: Rect) -> str:
    """Tk geometry string for a bar in physical pixels.

    Both edges are converted, not the size, so neighbouring bars meet
    without a rounding gap.
    """
    left, top = phys_to_tk(bar.left), phys_to_tk(bar.top)
    width = max(1, phys_to_tk(bar.right) - left)
    height = max(1, phys_to_tk(bar.bottom) - top)
    return f"{width}x{height}+{left}+{top}"


class CellBorder:
    """A gold frame drawn around a screen rectangle in physical pixels."""

    def __init__(self, root: tk.Misc) -> None:
        self.windows: list[tk.Toplevel] = [self._make_bar(root) for _ in range(4)]
        self._shown: tuple[Rect, int, str] | None = None
        self._color: str | None = None

    @staticmethod
    def _make_bar(root: tk.Misc) -> tk.Toplevel:
        window = tk.Toplevel(root)
        window.title("ARLO cell border")
        window.attributes("-topmost", True)  # noqa: FBT003
        window.overrideredirect(boolean=True)
        window.withdraw()
        make_click_through(window)
        return window

    def show(self, rect_phys: Rect, width_px: int, color: str = HOVER_GOLD) -> None:
        """Frame rect_phys with bars width_px wide, outside the rect.

        Showing the same frame again does nothing, so the windows are not
        re-mapped (no compositor fade). A new rect only moves the bars.
        """
        state = (rect_phys, width_px, color)
        if self._shown == state:
            return
        if color != self._color:
            for window in self.windows:
                window.configure(bg=color)
            self._color = color
        bars = frame_bars(rect_phys, width_px)
        for window, bar in zip(self.windows, bars, strict=True):
            window.geometry(_tk_geometry(bar))
        if self._shown is None:
            for window in self.windows:
                window.deiconify()
                window.lift()
        self._shown = state
        logger.debug("border show rect=%d,%d,%d,%d width=%d", *rect_phys, width_px)

    def hide(self, reason: str = "hide") -> None:
        """Withdraw the frame when it is showing."""
        if self._shown is None:
            return
        for window in self.windows:
            window.withdraw()
        self._shown = None
        logger.debug("border hide reason=%s", reason)

    def bar_rects(self) -> tuple[Rect, ...]:
        """Physical screen rectangles of the bars now showing, else empty."""
        if self._shown is None:
            return ()
        rect, width, _ = self._shown
        return tuple(frame_bars(rect, width))
