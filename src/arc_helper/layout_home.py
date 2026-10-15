"""Where ARLO's advice sits on screen: beneath the second weapon slot.

Home is a box whose top left is the card's or popup's top left. When the
scanner has found the two wide weapon slots of the loadout on a recent
full-screen capture, the box starts HOME_GAP below the lower one, as wide as
it, and reaches down to FLOOR. Without them it is the fixed LOADOUT panel box
below, given as fractions of the screen size so it scales with the
resolution. Nothing ARLO places at home reaches below FLOOR.
"""

from collections.abc import Sequence
from statistics import median

from .grid import EQUIPMENT
from .grid import Slot
from .tooltip_layout import Rect

LEFT = 0.356
TOP = 0.757
RIGHT = 0.533
BOTTOM = 0.892
# Nothing ARLO places in the box reaches below this fraction of the height
FLOOR = 0.93
# Gap between the second weapon slot's bottom edge and home's top, as a
# fraction of the screen height (13 px at 2160)
HOME_GAP = 0.006
# A weapon slot is at least this many grid slot sides wide (3.15 measured,
# the small equipment slots are 1.49)
WEAPON_MIN_SIDES = 2.5


def loadout_box(screen: tuple[int, int]) -> Rect:
    """The LOADOUT box in physical px for a screen of that size."""
    width, height = screen
    return Rect(
        round(width * LEFT),
        round(height * TOP),
        round(width * RIGHT),
        round(height * BOTTOM),
    )


def fixed_home(screen: tuple[int, int]) -> Rect:
    """Home without found weapon slots: the LOADOUT box down to FLOOR."""
    box = loadout_box(screen)
    return Rect(box.left, box.top, box.right, round(screen[1] * FLOOR))


def weapon_slots(slots: Sequence[Slot]) -> list[Rect]:
    """The wide equipment weapon slots among find_slots results, top to bottom.

    The grid slot side is the median short side of all slots, a weapon slot
    is an equipment slot at least WEAPON_MIN_SIDES of them wide.
    """
    if not slots:
        return []
    side = median(
        min(s.rect.right - s.rect.left, s.rect.bottom - s.rect.top) for s in slots
    )
    wide = [
        s.rect
        for s in slots
        if s.panel == EQUIPMENT
        and s.rect.right - s.rect.left >= WEAPON_MIN_SIDES * side
    ]
    return sorted(wide, key=lambda r: (r.top, r.left))


def home_below_weapons(weapons: Sequence[Rect], screen: tuple[int, int]) -> Rect | None:
    """Home beneath the lower of exactly two weapon slots, else None.

    One slot alone may be the first gun with the second covered by the
    game's item card, so it is not trusted.
    """
    if len(weapons) != 2:
        return None
    lower = max(weapons, key=lambda r: r.bottom)
    top = lower.bottom + round(screen[1] * HOME_GAP)
    floor = round(screen[1] * FLOOR)
    if top >= floor:
        return None
    return Rect(lower.left, top, lower.right, floor)
