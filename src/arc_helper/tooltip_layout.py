"""
Tooltip geometry for ARLO.

Pure functions that locate the game's cream tooltip inside a cursor capture,
find its "CAN BE FOUND IN" and "RECYCLES INTO" label rows from Tesseract word
boxes, and place the overlay chips beside them. No Tk, Tesseract or screen
backend access happens here, so everything is unit-testable with synthetic
arrays and fake word boxes.
"""

import difflib
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np

# Tooltip background is #f9eedf = RGB(249, 238, 223). Same bounds and
# percentages as OCREngine.preprocess_tooltip.
CREAM_LOWER = np.array([240, 225, 210])
CREAM_UPPER = np.array([255, 250, 240])
ROW_CREAM_MIN = 0.5
COL_CREAM_MIN = 0.3
TIGHT_COL_CREAM_MIN = 0.5

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


class TooltipLayout(NamedTuple):
    """Tooltip rectangle and found label rows, in physical screen pixels."""

    tooltip: Rect
    labels: dict[str, Rect]


def find_cream_rect(rgb: np.ndarray) -> Rect | None:
    """Return the cream tooltip rectangle in capture coordinates, or None.

    Rows count as tooltip when over 50 percent cream, columns when over 30
    percent cream. A second pass keeps only columns over 50 percent cream
    inside that box, which trims the colored borders.
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

    return Rect(left + int(tight_cols[0]), top, left + int(tight_cols[-1]) + 1, bottom)


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


def place_chips(
    tooltip: Rect,
    labels: dict[str, Rect],
    chip_sizes: dict[str, Size],
    screen: Size,
    pad: int,
) -> dict[str, Point]:
    """Top-left screen positions for the chips, keyed like chip_sizes.

    The price chip is right-aligned inside the tooltip on the found-in row.
    The recycle chip sits on the icon row under the recycles-into label,
    45 percent across the tooltip. Both are clamped to the screen. A chip
    without its label, or without a size, gets no entry.
    """
    positions: dict[str, Point] = {}

    found_in = labels.get(FOUND_IN)
    price = chip_sizes.get(PRICE_CHIP)
    if found_in is not None and price is not None:
        x = tooltip.right - pad - price.width
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

    return positions
