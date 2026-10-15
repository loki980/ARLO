"""
Find the hovered inventory cell beside the game's item card.

The game draws an animated glow ring (white, purple and cyan, partly dark)
about 4 px wide and 3 to 4 px outside the hovered cell's thin outline. Other
cells have only the thin outline. The card opens beside the hovered cell, so
the search runs in a window on each side of the card rectangle:

1. lit mask: max channel at least 70, or a colourful pixel (max minus min at
   least 50). The white parts of the ring are the card's cream colour, so
   cream is kept and the card is left out by the window instead. Pixels
   inside any ignore rect (ARLO's own chips and panels) count as not lit.
2. close with a 5 px square, which merges the glow ring and the outline
   beside it into one band about 10 px thick.
3. open with a 7 px square, which removes thin outlines, rarity borders and
   most text, and keeps that band. Whatever survives an opening with a
   21 px square (filled blobs such as icons and rarity wedges) is removed
   from it, so a wedge touching the outline does not swell the ring.
4. seeds: opened components that are line shaped (longest side at least
   80 px, area over width plus height at most 14) and reach within 80 px of
   the card edge.
5. the left side of the card is searched first, the right side only when
   the left has no seed. Seeds are tried largest first. The cell is the
   bounding box of the seed's component in the closed mask, which also
   follows the dim parts of the ring through the outline. When that box is
   not cell sized (120 to 720 px a side) the seed box is used, and when
   neither is, the next seed is tried.

Thresholds are in physical pixels of a 3840x2160 screen. Pure functions, no
Tk or screen access.
"""

from collections.abc import Iterable
from typing import NamedTuple

import cv2
import numpy as np

from .tooltip_layout import Rect

LIT_MIN = 70
COLOURFUL_SPREAD = 50
CLOSE_SIDE = 5
OPEN_SIDE = 7
BLOB_SIDE = 21
SEED_MIN_LENGTH = 80
SEED_MAX_THICKNESS = 14
EDGE_BAND = 80
CARD_GAP = 3
WINDOW_WIDTH = 720
WINDOW_ABOVE = 100
WINDOW_BELOW = 100
CELL_MIN_SIDE = 120
CELL_MAX_SIDE = 720

LEFT = "left"
RIGHT = "right"


class HoveredCell(NamedTuple):
    """The found cell, which card side it was on and its seed area in px."""

    rect: Rect
    side: str
    seed_area: int


def _lit_mask(window: np.ndarray) -> np.ndarray:
    red, green, blue = cv2.split(np.ascontiguousarray(window))
    high = cv2.max(cv2.max(red, green), blue)
    low = cv2.min(cv2.min(red, green), blue)
    bright = cv2.threshold(high, LIT_MIN - 1, 255, cv2.THRESH_BINARY)[1]
    spread = cv2.threshold(
        cv2.subtract(high, low), COLOURFUL_SPREAD - 1, 255, cv2.THRESH_BINARY
    )[1]
    return cv2.bitwise_or(bright, spread)


def _clear_ignored(
    mask: np.ndarray, origin: tuple[int, int], ignore: Iterable[Rect]
) -> None:
    """Zero the mask inside each ignore rect, given in image coordinates."""
    x0, y0 = origin
    height, width = mask.shape
    for rect in ignore:
        left, right = max(0, rect.left - x0), min(width, rect.right - x0)
        top, bottom = max(0, rect.top - y0), min(height, rect.bottom - y0)
        if left < right and top < bottom:
            mask[top:bottom, left:right] = 0


def _cell_sized(width: int, height: int) -> bool:
    return (
        CELL_MIN_SIDE <= width <= CELL_MAX_SIDE
        and CELL_MIN_SIDE <= height <= CELL_MAX_SIDE
    )


def _search_side(
    rgb: np.ndarray, card: Rect, side: str, ignore: tuple[Rect, ...]
) -> HoveredCell | None:
    """Search one side of the card, returning the cell in image coordinates."""
    height, width = rgb.shape[:2]
    if side == LEFT:
        x0, x1 = max(0, card.left - WINDOW_WIDTH), card.left - CARD_GAP
    else:
        x0, x1 = card.right + CARD_GAP, min(width, card.right + WINDOW_WIDTH)
    y0, y1 = max(0, card.top - WINDOW_ABOVE), min(height, card.bottom + WINDOW_BELOW)
    if x1 - x0 < CELL_MIN_SIDE or y1 - y0 < CELL_MIN_SIDE:
        return None

    lit = _lit_mask(rgb[y0:y1, x0:x1, :3])
    _clear_ignored(lit, (x0, y0), ignore)
    closed = cv2.morphologyEx(
        lit, cv2.MORPH_CLOSE, np.ones((CLOSE_SIDE, CLOSE_SIDE), np.uint8)
    )
    thick = cv2.morphologyEx(
        closed, cv2.MORPH_OPEN, np.ones((OPEN_SIDE, OPEN_SIDE), np.uint8)
    )
    blobs = cv2.morphologyEx(
        closed, cv2.MORPH_OPEN, np.ones((BLOB_SIDE, BLOB_SIDE), np.uint8)
    )
    thick[blobs > 0] = 0

    edge = card.left - x0 if side == LEFT else card.right - x0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(thick)
    seeds = []
    for label in range(1, count):
        left, _, w, h, area = (int(v) for v in stats[label][:5])
        if max(w, h) < SEED_MIN_LENGTH or area / (w + h) > SEED_MAX_THICKNESS:
            continue
        near = left + w >= edge - EDGE_BAND if side == LEFT else left <= EDGE_BAND
        if near:
            seeds.append((area, label))
    if not seeds:
        return None

    # Follow each seed into the closed mask to recover the dim ring parts,
    # largest seed first, and keep the first one with a cell sized box
    _, closed_labels, closed_stats, _ = cv2.connectedComponentsWithStats(closed)
    for area, label in sorted(seeds, reverse=True):
        ys, xs = np.nonzero(labels == label)
        owner = int(closed_labels[ys[0], xs[0]])
        cl, ct, cw, ch = (int(v) for v in closed_stats[owner][:4])
        if not _cell_sized(cw, ch):
            cl, ct, cw, ch = (int(v) for v in stats[label][:4])
            if not _cell_sized(cw, ch):
                continue
        rect = Rect(cl + x0, ct + y0, cl + cw + x0, ct + ch + y0)
        return HoveredCell(rect, side, area)
    return None


def locate_hovered_cell(
    rgb: np.ndarray, card: Rect, ignore: Iterable[Rect] = ()
) -> HoveredCell | None:
    """Return the hovered cell with its side and seed area, or None.

    The game opens the card to the right of the hovered cell, so the left
    side is searched first and the right side only when the left side has
    no seed (a card flipped left near the right screen edge).
    """
    if rgb.ndim != 3 or rgb.shape[0] == 0 or rgb.shape[1] == 0:
        return None
    pixels = np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)
    ignored = tuple(ignore)
    for side in (LEFT, RIGHT):
        found = _search_side(pixels, card, side, ignored)
        if found is not None:
            return found
    return None


def find_hovered_cell(
    rgb: np.ndarray, card: Rect, ignore: Iterable[Rect] = ()
) -> Rect | None:
    """Return the outer box of the hovered cell's glow ring, or None.

    rgb is an RGB array, card the item card rectangle (from
    find_tooltip_card) and ignore any rectangles whose pixels must not count
    as lit, all in the array's own coordinates.
    """
    found = locate_hovered_cell(rgb, card, ignore)
    return found.rect if found is not None else None
