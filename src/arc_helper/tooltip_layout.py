"""
Tooltip geometry for ARLO.

Pure functions that locate the game's cream tooltip inside a cursor capture,
find its "CAN BE FOUND IN" and "RECYCLES INTO" label rows from Tesseract word
boxes, and place the overlay chips beside them. No Tk, Tesseract or screen
backend access happens here, so everything is unit-testable with synthetic
arrays and fake word boxes.
"""

import difflib
import math
from collections.abc import Iterable
from typing import NamedTuple

import cv2
import numpy as np

# Tooltip background is #f9eedf = RGB(249, 238, 223). Same bounds and
# percentages as OCREngine.preprocess_tooltip.
CREAM_LOWER = np.array([240, 225, 210])
CREAM_UPPER = np.array([255, 250, 240])
ROW_CREAM_MIN = 0.5
COL_CREAM_MIN = 0.3
TIGHT_COL_CREAM_MIN = 0.5

# A cream component counts as the item card only when its bounding box is at
# least this many pixels on each side and the cream fills this share of it.
# The card is a solid rectangle, the ACTIONS tab and stray highlights are not.
CARD_MIN_SIDE = 150
CARD_MIN_FILL = 0.6

# Label keys returned by find_label_rows, mapped to the text the game prints
FOUND_IN = "found_in"
RECYCLES_INTO = "recycles_into"
LABEL_TEXT: dict[str, str] = {
    FOUND_IN: "CAN BE FOUND IN",
    RECYCLES_INTO: "RECYCLES INTO",
}
LABEL_MATCH_RATIO = 0.8

# Chip keys used by place_chips
PRICE_CHIP = "price"
RECYCLE_CHIP = "recycle"
VERDICT_CHIP = "verdict"

# The recycle chip starts this far across the tooltip, right of the icons
RECYCLE_CHIP_X_FRACTION = 0.45


class Point(NamedTuple):
    x: int
    y: int


class Size(NamedTuple):
    width: int
    height: int


class Rect(NamedTuple):
    """Axis-aligned rectangle. right and bottom are exclusive."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


class Word(NamedTuple):
    """One Tesseract word box (pytesseract.image_to_data row)."""

    text: str
    left: int
    top: int
    width: int
    height: int
    line: int


class Edges(NamedTuple):
    """Which capture edges the cream region touches (so may be cut off)."""

    left: bool = False
    top: bool = False
    right: bool = False
    bottom: bool = False


NOT_CLIPPED = Edges()


class CreamRegion(NamedTuple):
    """The cream rectangle in capture coordinates and its clipped edges."""

    rect: Rect
    clipped: Edges


class TooltipLayout(NamedTuple):
    """Tooltip rectangle and found label rows, in physical screen pixels.

    clipped marks tooltip edges that coincide with the capture edge, where
    the real tooltip may extend further than the rectangle.
    """

    tooltip: Rect
    labels: dict[str, Rect]
    clipped: Edges = NOT_CLIPPED


def find_cream_rect(rgb: np.ndarray) -> CreamRegion | None:
    """Return the cream tooltip region in capture coordinates, or None.

    Rows count as tooltip when over 50 percent cream, columns when over 30
    percent cream. A second pass keeps only columns over 50 percent cream
    inside that box, which trims the colored borders. The result also says
    which capture edges the region touches, since a capture narrower than
    the tooltip cuts it off there.
    """
    if rgb.ndim != 3 or rgb.shape[0] == 0 or rgb.shape[1] == 0:
        return None

    pixels = rgb[:, :, :3]
    mask = np.all((pixels >= CREAM_LOWER) & (pixels <= CREAM_UPPER), axis=2)

    cream_rows = np.where(mask.mean(axis=1) > ROW_CREAM_MIN)[0]
    cream_cols = np.where(mask.mean(axis=0) > COL_CREAM_MIN)[0]
    if len(cream_rows) == 0 or len(cream_cols) == 0:
        return None

    top, bottom = int(cream_rows[0]), int(cream_rows[-1]) + 1
    left, right = int(cream_cols[0]), int(cream_cols[-1]) + 1

    tight_cols = np.where(
        mask[top:bottom, left:right].mean(axis=0) > TIGHT_COL_CREAM_MIN
    )[0]
    if len(tight_cols) == 0:
        return None

    rect = Rect(left + int(tight_cols[0]), top, left + int(tight_cols[-1]) + 1, bottom)
    height, width = mask.shape
    clipped = Edges(
        left=rect.left == 0,
        top=rect.top == 0,
        right=rect.right == width,
        bottom=rect.bottom == height,
    )
    return CreamRegion(rect, clipped)


def _distance_to_rect(point: Point, rect: Rect) -> float:
    """Distance from a point to a rectangle, zero when the point is inside."""
    dx = max(rect.left - point.x, point.x - (rect.right - 1), 0)
    dy = max(rect.top - point.y, point.y - (rect.bottom - 1), 0)
    return math.hypot(dx, dy)


def find_tooltip_card(rgb: np.ndarray, cursor: Point) -> Rect | None:
    """Return the item card nearest the cursor, or None.

    Builds a mask from the same cream bounds as find_cream_rect and splits it
    into connected components with OpenCV. Components under 150 px on either
    side, or whose cream fills under 60 percent of their bounding box, are
    discarded. Of the rest, the one whose bounding box is nearest the cursor
    wins (distance zero when the cursor is inside it), so a large cream panel
    elsewhere on screen loses to the card under the cursor. The cursor and
    the result are in the array's own coordinates.
    """
    if rgb.ndim != 3 or rgb.shape[0] == 0 or rgb.shape[1] == 0:
        return None

    pixels = np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)
    mask = cv2.inRange(
        pixels,
        tuple(int(v) for v in CREAM_LOWER),
        tuple(int(v) for v in CREAM_UPPER),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)

    best: tuple[float, Rect] | None = None
    for label in range(1, count):  # label 0 is the background
        left, top, width, height, area = (int(v) for v in stats[label][:5])
        if width < CARD_MIN_SIDE or height < CARD_MIN_SIDE:
            continue
        if area / (width * height) < CARD_MIN_FILL:
            continue
        rect = Rect(left, top, left + width, top + height)
        distance = _distance_to_rect(cursor, rect)
        if best is None or distance < best[0]:
            best = (distance, rect)

    return best[1] if best is not None else None


def _label_key(text: str) -> str:
    """Uppercase alphanumerics only, so spacing and stray marks do not matter."""
    return "".join(c for c in text.upper() if c.isalnum())


def _group_lines(words: Iterable[Word]) -> dict[int, list[Word]]:
    lines: dict[int, list[Word]] = {}
    for word in words:
        if word.text.strip():
            lines.setdefault(word.line, []).append(word)
    return lines


def find_label_rows(words: list[Word], scale: float) -> dict[str, Rect]:
    """Find the label rows among Tesseract word boxes.

    Words are grouped by line and each line's text is compared with the label
    text after _label_key, accepting a difflib ratio of at least 0.8 (so
    "RECYCLES INT0" still matches). Returns the label rectangles divided back
    by scale, with an entry only for labels that were found.
    """
    targets = {name: _label_key(text) for name, text in LABEL_TEXT.items()}
    best: dict[str, tuple[float, Rect]] = {}

    for line_words in _group_lines(words).values():
        key = _label_key(" ".join(w.text for w in line_words))
        if not key:
            continue
        for name, target in targets.items():
            ratio = difflib.SequenceMatcher(None, key, target).ratio()
            if ratio < LABEL_MATCH_RATIO or ratio <= best.get(name, (0.0,))[0]:
                continue
            rect = Rect(
                round(min(w.left for w in line_words) / scale),
                round(min(w.top for w in line_words) / scale),
                round(max(w.left + w.width for w in line_words) / scale),
                round(max(w.top + w.height for w in line_words) / scale),
            )
            best[name] = (ratio, rect)

    return {name: rect for name, (_, rect) in best.items()}


def to_screen(rect: Rect, origin: Point) -> Rect:
    """Shift a capture-space rectangle by the capture origin."""
    return Rect(
        rect.left + origin.x,
        rect.top + origin.y,
        rect.right + origin.x,
        rect.bottom + origin.y,
    )


def _clamp(value: int, size: int, limit: int) -> int:
    return max(0, min(value, limit - size))


def _right_aligned_x(
    tooltip: Rect, label: Rect, width: int, pad: int, clipped: Edges
) -> int:
    """Left edge of a chip right-aligned in the card and clear of the label."""
    after_label = label.right + pad
    if clipped.right:
        return after_label
    return max(tooltip.right - pad - width, after_label)


def place_chips(
    tooltip: Rect,
    labels: dict[str, Rect],
    chip_sizes: dict[str, Size],
    screen: Size,
    pad: int,
    clipped: Edges = NOT_CLIPPED,
) -> dict[str, Point]:
    """Top-left screen positions for the chips, keyed like chip_sizes.

    The price chip is right-aligned inside the tooltip on the found-in row,
    but never starts left of the label's right edge plus pad. When the
    tooltip's right edge is clipped by the capture, that edge is not the
    real one, so the price chip anchors just right of the label instead.
    The recycle chip sits on the icon row under the recycles-into label,
    45 percent across the tooltip. The verdict chip follows the same
    right-aligned and clipped-right rules. When the recycle chip is placed
    its bottom edge sits pad above the recycle chip's top, so the two never
    overlap. With the label but no recycle chip it sits at the label's top.
    Without that label it sits directly above the price
    chip with pad between, or on the found-in row itself when there is no
    price chip. Without either label it gets no entry. All are clamped to
    the screen last, so the screen edge wins over the label rule. A chip
    without its label, or without a size, gets no entry.
    """
    positions: dict[str, Point] = {}

    found_in = labels.get(FOUND_IN)
    price = chip_sizes.get(PRICE_CHIP)
    if found_in is not None and price is not None:
        x = _right_aligned_x(tooltip, found_in, price.width, pad, clipped)
        y = found_in.top
        positions[PRICE_CHIP] = Point(
            _clamp(x, price.width, screen.width),
            _clamp(y, price.height, screen.height),
        )

    recycles_into = labels.get(RECYCLES_INTO)
    recycle = chip_sizes.get(RECYCLE_CHIP)
    if recycles_into is not None and recycle is not None:
        x = tooltip.left + round(RECYCLE_CHIP_X_FRACTION * tooltip.width)
        y = recycles_into.bottom + pad
        positions[RECYCLE_CHIP] = Point(
            _clamp(x, recycle.width, screen.width),
            _clamp(y, recycle.height, screen.height),
        )

    verdict = chip_sizes.get(VERDICT_CHIP)
    if verdict is not None:
        anchor = recycles_into or found_in
        if anchor is not None:
            x = _right_aligned_x(tooltip, anchor, verdict.width, pad, clipped)
            y = anchor.top
            if RECYCLE_CHIP in positions:
                # Bottom edge pad above the recycle chip, never overlapping it
                y = positions[RECYCLE_CHIP].y - pad - verdict.height
            elif recycles_into is None and PRICE_CHIP in positions:
                y = positions[PRICE_CHIP].y - pad - verdict.height
            positions[VERDICT_CHIP] = Point(
                _clamp(x, verdict.width, screen.width),
                _clamp(y, verdict.height, screen.height),
            )

    return positions
