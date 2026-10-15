"""Click-through window that shows the advice card at its home.

The card replaces the text popup that used to sit in the empty area below
the loadout grid. It is as wide as its box and, when that makes it taller
than the box, scaled down until it fits, so it never rises above the box's
top. fit_scale and fit_height_scale are pure so they can be tested without
a display.
"""

import tkinter as tk

from PIL import Image
from PIL import ImageTk

from .advice_card import AdviceCard
from .advice_card import render_advice_card
from .clickthrough import make_click_through
from .config import logger
from .screen import get_backend
from .screen import phys_to_tk
from .tooltip_layout import Rect

# Width of render_advice_card at scale 1.0, in px.
CARD_BASE_WIDTH = 560
# Renders tried while a card is still taller than its box (rounding can
# leave it a pixel over after the first scale down)
FIT_TRIES = 3
# LOADOUT panel colour sampled from the lower area of the box in a 3840x2160
# capture (debug/captures/20261014-211438-853_none.png), the most common
# pixel value, #0B0D19. The card's transparent corners are composited onto it.
PANEL_RGB = (11, 13, 25)


def fit_scale(box_width_phys: int) -> float:
    """Render scale at which the card is as wide as the box."""
    return box_width_phys / CARD_BASE_WIDTH


def fit_height_scale(scale: float, height_phys: int, box_height_phys: int) -> float:
    """Scale at which a card height_phys tall at scale fits box_height_phys.

    A card that already fits keeps its scale.
    """
    if height_phys <= box_height_phys:
        return scale
    return scale * box_height_phys / height_phys


def place_card(
    box_rect_phys: Rect,
    card_size_phys: tuple[int, int],  # noqa: ARG001
    screen_height: int,  # noqa: ARG001
) -> tuple[int, int]:
    """Top left of the card: the box's top left.

    A card too tall for the box is scaled down by render_in_box, never
    raised. The size and screen height stay for older callers.
    """
    return box_rect_phys.left, box_rect_phys.top


def render_in_box(card: AdviceCard, box_phys: Rect) -> Image.Image:
    """Render the card as wide as the box, scaled down to fit its height."""
    box_height = box_phys.bottom - box_phys.top
    scale = fit_scale(box_phys.right - box_phys.left)
    image = render_advice_card(card, scale)
    for _ in range(FIT_TRIES):
        if image.size[1] <= box_height:
            break
        scale = fit_height_scale(scale, image.size[1], box_height)
        image = render_advice_card(card, scale)
    return image


def flatten_on_panel(image: Image.Image) -> Image.Image:
    """Composite an RGBA image onto the panel colour, returning RGB."""
    background = Image.new("RGBA", image.size, (*PANEL_RGB, 255))
    return Image.alpha_composite(background, image).convert("RGB")


class AdviceWindow:
    """Topmost, click-through window showing a rendered advice card."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.window = tk.Toplevel(root)
        self.window.title("ARLO advice")
        self.window.attributes("-topmost", True)  # noqa: FBT003
        self.window.overrideredirect(boolean=True)
        self.window.withdraw()
        self._label = tk.Label(self.window, borderwidth=0, highlightthickness=0)
        self._label.pack()
        make_click_through(self.window)

        self._card: AdviceCard | None = None
        self._box_key: tuple[int, int, int, int] | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._pos: tuple[int, int] | None = None  # tk units
        self._size_tk: tuple[int, int] = (0, 0)
        self._visible = False

    def show(self, card: AdviceCard, home_rect_phys: Rect) -> None:
        """Show the card at the box's top left, no wider or taller than it.

        An equal card in the same box is a no-op.
        """
        box_key = tuple(home_rect_phys)
        if self._visible and card == self._card and box_key == self._box_key:
            return
        backend = get_backend()
        tk_scale = backend.tk_scale
        if card != self._card or box_key != self._box_key or self._photo is None:
            image = render_in_box(card, home_rect_phys)
            phys_size = image.size
            flat = flatten_on_panel(image)
            width_tk = max(1, round(phys_size[0] / tk_scale))
            height_tk = max(1, round(phys_size[1] / tk_scale))
            if (width_tk, height_tk) != phys_size:
                flat = flat.resize((width_tk, height_tk), Image.LANCZOS)
            self._photo = ImageTk.PhotoImage(flat)
            self._label.configure(image=self._photo)
            self._size_tk = (width_tk, height_tk)
            x, y = place_card(home_rect_phys, phys_size, backend.resolution()[1])
            self._pos = (phys_to_tk(x), phys_to_tk(y))
            self._card = card
            self._box_key = box_key
        width, height = self._size_tk
        x_tk, y_tk = self._pos
        self.window.geometry(f"{width}x{height}+{x_tk}+{y_tk}")
        if not self._visible:
            self.window.deiconify()
            self._visible = True
        self.window.lift()
        logger.debug(f"Advice card shown at tk {x_tk},{y_tk} size {width}x{height}")

    def hide(self) -> None:
        """Withdraw the window."""
        if self._visible:
            self.window.withdraw()
            self._visible = False

    def is_visible(self) -> bool:
        return self._visible

    def rect_phys(self) -> Rect | None:
        """Current rectangle in physical px, None while hidden.

        Part 2b adds it to the occlusion and border ignore lists.
        """
        if not self._visible or self._pos is None:
            return None
        tk_scale = get_backend().tk_scale
        x, y = self._pos
        width, height = self._size_tk
        return Rect(
            round(x * tk_scale),
            round(y * tk_scale),
            round((x + width) * tk_scale),
            round((y + height) * tk_scale),
        )
