"""Where ARLO's advice sits on screen: the LOADOUT panel box.

The box is the empty part of the LOADOUT panel below the second weapon
slot, given as fractions of the screen size so it scales with the
resolution. The card's or popup's top left is the box's top left. A taller
window grows downward but never below FLOOR.
"""

from .tooltip_layout import Rect

LEFT = 0.356
TOP = 0.757
RIGHT = 0.533
BOTTOM = 0.892
# Nothing ARLO places in the box reaches below this fraction of the height
FLOOR = 0.93


def loadout_box(screen: tuple[int, int]) -> Rect:
    """The LOADOUT box in physical px for a screen of that size."""
    width, height = screen
    return Rect(
        round(width * LEFT),
        round(height * TOP),
        round(width * RIGHT),
        round(height * BOTTOM),
    )
