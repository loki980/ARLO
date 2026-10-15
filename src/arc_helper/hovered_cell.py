"""
Find the hovered inventory cell beside the game's item card.

The game draws an animated glow ring (white, purple and cyan, partly dark)
about 4 px wide and 3 to 4 px outside the hovered cell's thin outline. Other
cells have only the thin outline. The card opens beside the hovered cell, so
the search runs in a window on each side of the card rectangle:

1. lit mask: max channel at least 70, or a colourful pixel (max minus min at
   least 50). The white parts of the ring are the card's cream colour, so
   cream is kept and the card is left out by the window instead. Warm
   pixels (gold, orange, yellow) count as not lit, since the ring has none
   and ARLO's gold frame and the game's orange rarity marks do. Pixels
   inside any ignore rect, and inside ARLO's own windows found by their
   flat colours (see occluders below), count as not lit.
2. close with a 5 px square, which merges the glow ring and the outline
   beside it into one band about 10 px thick.
3. open with a 7 px square, which removes thin outlines, rarity borders and
   most text, and keeps that band. Whatever survives an opening with a
   21 px square (filled blobs such as icons and rarity wedges) is removed
   from it, so a wedge touching the outline does not swell the ring.
4. seeds: opened components that are line shaped (longest side at least
   80 px, area over width plus height at most 14) and reach within 130 px
   of the card edge. The band is wider than the gap to the card because the
   ring's side next to the card can be dim or lie under ARLO's gold frame,
   leaving only the far sides as seeds. A seed is a straight run or an L or
   U of the ring's sides, so its pixels lie within 14 px of its own bounding
   box edges. A seed with more than 30 percent of its pixels further inside
   is a slanted part of an item icon (a gun's grip cut off by the window
   edge passes the line test) and is dropped.
5. the left side of the card is searched first, the right side only when
   the left has no seed. Seeds are tried largest first. The cell is the
   bounding box of the seed's component in the closed mask, which also
   follows the dim parts of the ring through the outline. When that box is
   not cell sized (120 to 720 px a side) the seed box is used, and when
   neither is, the next seed is tried.
6. occluders: ARLO's corner popup, chips and status panel are opaque
   windows painted in a few exact colours (backgrounds, border and the
   chips' verdict accent bars). Each large patch of those colours is an
   occluder, and its pixels count as not lit. The ignore rects passed by the app are taken
   when the scan ends, so a window that moved or was redrawn after the
   screen grab is not in them, and finding the windows in the grab itself
   covers that gap. When the ring box ends against an occluder and is
   shorter across that edge than along the other side, the hidden edge
   is moved into the occluder to make the box square, as the grid cells
   are.

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
EDGE_BAND = 130
SEED_INNER_MARGIN = 14
SEED_INNER_MAX = 0.3
CARD_GAP = 3
WINDOW_WIDTH = 720
WINDOW_ABOVE = 100
WINDOW_BELOW = 100
CELL_MIN_SIDE = 120
CELL_MAX_SIDE = 720
WARM_RED_OVER_BLUE = 60
WARM_GREEN_OVER_BLUE = 30
# ARLO's flat window colours (overlay.py BG_DARK, BG_PANEL, VERDICT_BG,
# BORDER_COLOR and the border's antialiased edge) and the verdict colours of
# the chips' accent bars, as RGB
ARLO_COLOURS = (
    (18, 18, 31),
    (26, 26, 46),
    (35, 39, 46),
    (58, 58, 92),
    (42, 42, 69),
    (255, 215, 0),
    (0, 206, 209),
    (50, 205, 50),
    (255, 105, 180),
    (255, 68, 68),
    (136, 136, 136),
)
# ARLO's windows are looked for on every second pixel of every second row,
# so the closing side and the minimum area are in those sampled pixels
ARLO_STEP = 2
ARLO_CLOSE_SIDE = 7
ARLO_MIN_AREA = 750
ARLO_PAD = 3
OCCLUDER_REACH = 8

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
    lit = cv2.bitwise_or(bright, spread)
    warm = (
        (red >= green)
        & (cv2.subtract(red, blue) >= WARM_RED_OVER_BLUE)
        & (cv2.subtract(green, blue) >= WARM_GREEN_OVER_BLUE)
    )
    lit[warm] = 0
    return lit


def _occluders(window: np.ndarray) -> list[Rect]:
    """Rects of ARLO's own opaque windows in the window, by their colours."""
    sample = np.ascontiguousarray(window[::ARLO_STEP, ::ARLO_STEP])
    flat = np.zeros(sample.shape[:2], np.uint8)
    for colour in ARLO_COLOURS:
        flat |= cv2.inRange(sample, colour, colour)
    flat = cv2.morphologyEx(
        flat, cv2.MORPH_CLOSE, np.ones((ARLO_CLOSE_SIDE, ARLO_CLOSE_SIDE), np.uint8)
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(flat)
    rects = []
    for label in range(1, count):
        left, top, w, h, area = (int(v) for v in stats[label][:5])
        if area >= ARLO_MIN_AREA:
            rects.append(
                Rect(
                    left * ARLO_STEP - ARLO_PAD,
                    top * ARLO_STEP - ARLO_PAD,
                    (left + w) * ARLO_STEP + ARLO_PAD,
                    (top + h) * ARLO_STEP + ARLO_PAD,
                )
            )
    return rects


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return min(a1, b1) - max(a0, b0)


def _complete(box: Rect, occluders: list[Rect]) -> Rect:
    """Square up a ring box whose short side ends against an occluder."""
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    for occ in occluders:
        across = _overlap(left, right, occ.left, occ.right) * 2 >= width
        down = _overlap(top, bottom, occ.top, occ.bottom) * 2 >= height
        if height < width and across:
            if occ.top < top and abs(occ.bottom - top) <= OCCLUDER_REACH:
                return Rect(left, bottom - width, right, bottom)
            if occ.bottom > bottom and abs(occ.top - bottom) <= OCCLUDER_REACH:
                return Rect(left, top, right, top + width)
        if width < height and down:
            if occ.left < left and abs(occ.right - left) <= OCCLUDER_REACH:
                return Rect(right - height, top, right, bottom)
            if occ.right > right and abs(occ.left - right) <= OCCLUDER_REACH:
                return Rect(left, top, left + height, bottom)
    return box


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


def _inner_share(mask: np.ndarray) -> float:
    """Share of a seed's pixels away from its bounding box edges."""
    inner = mask[
        SEED_INNER_MARGIN:-SEED_INNER_MARGIN, SEED_INNER_MARGIN:-SEED_INNER_MARGIN
    ]
    return int(np.count_nonzero(inner)) / max(1, int(np.count_nonzero(mask)))


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

    window = np.ascontiguousarray(rgb[y0:y1, x0:x1, :3])
    lit = _lit_mask(window)
    occluders = _occluders(window)
    _clear_ignored(lit, (x0, y0), ignore)
    _clear_ignored(lit, (0, 0), occluders)
    # ignore rects in window coordinates also count as occluders
    occluders += [
        Rect(r.left - x0, r.top - y0, r.right - x0, r.bottom - y0) for r in ignore
    ]
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
        left, top, w, h, area = (int(v) for v in stats[label][:5])
        if max(w, h) < SEED_MIN_LENGTH or area / (w + h) > SEED_MAX_THICKNESS:
            continue
        near = left + w >= edge - EDGE_BAND if side == LEFT else left <= EDGE_BAND
        if not near:
            continue
        own = labels[top : top + h, left : left + w] == label
        if _inner_share(own) <= SEED_INNER_MAX:
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
        box = _complete(Rect(cl, ct, cl + cw, ct + ch), occluders)
        rect = Rect(box.left + x0, box.top + y0, box.right + x0, box.bottom + y0)
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
