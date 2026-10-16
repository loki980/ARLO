"""Read the numbers the inventory screen prints above the stash and loadout.

The stash header shows used and capacity slots and the total value, the
loadout header shows carried weight, weight capacity and value. They are
the anchors a running tally of the inventory can be checked against.

Each reader takes a full-screen RGB frame and the find_slots results of the
panel it anchors on. It returns every field or None, never a partial read,
because a wrong number is worse than no number.
"""

import re
from typing import NamedTuple

import cv2
import numpy as np
import pytesseract

from arc_helper.grid import Slot
from arc_helper.tooltip_layout import Rect

# Header text rows sit at a fixed distance above the top of the panel's first
# slot row, not at a fixed screen height: the in-raid layout is 48 px lower
# than the base one. Offsets in px on a 2160 px tall frame, measured on the
# base captures where the first slots start at y 544 and the counts row is
# y 335 to 400 and the panel title y 276 to 341. Scaled by frame height.
FRAME_HEIGHT_4K = 2160
COUNTS_ABOVE_TOP = 209 / FRAME_HEIGHT_4K
COUNTS_ABOVE_BOTTOM = 144 / FRAME_HEIGHT_4K
TITLE_ABOVE_TOP = 268 / FRAME_HEIGHT_4K
# Offsets in frame heights from the left edge of the first slot column: the
# stash counts start after the grid icon, the loadout weight after the bag
# icon (measured on 3840x2160, 40 and 215 px).
STASH_TEXT_LEFT = 40 / 2160
STASH_TEXT_RIGHT = 540 / 2160
LOADOUT_TEXT_LEFT = 215 / 2160
LOADOUT_TEXT_RIGHT = 680 / 2160
UPSCALE = 2
# The loadout counts row is read larger: at 2 the soft strokes of /90.0 in a
# base screenshot read as /190.0
LOADOUT_UPSCALE = 4
# The background behind the text is flat. Per column, the 20th percentile of
# the rows (which ignores the bright text) must stay within
# BACKGROUND_TOLERANCE gray levels of the crop's typical column for no more
# than COVER_RUN columns in a row. A longer run is something covering part of
# the row, such as an item card or popup. The crop is then cut at the start of
# the first such run and only the columns left of it are read. A truncated
# value could read as whole, so the cut crop must end in at least EDGE_CLEAR
# columns of plain background (wider than any gap between digits), else the
# text runs into the cover and the read is given up. Isolated dark columns
# beside glyph edges are normal.
BACKGROUND_TOLERANCE = 6
COVER_RUN = 6
EDGE_CLEAR = 24
TEXT_CONTRAST = 60
# One text line for the loadout, two (title and counts) for the stash
ONE_LINE = "--psm 7 --oem 1"
TWO_LINES = "--psm 6 --oem 1"

# Between the numbers sits the coin glyph, which Tesseract reads as junk
_STASH = re.compile(r"^\D*?(\d+)\s*/\s*(\d+)\D+?(\d[\d.,]*)\s*$")
_LOADOUT = re.compile(r"^\D*?(\d+\.\d)\s*/\s*(\d+\.\d)\D+?(\d[\d.,]*)\s*$")


class StashHeader(NamedTuple):
    used: int
    capacity: int
    value: int


class LoadoutHeader(NamedTuple):
    weight: float
    weight_cap: float
    value: int


def _anchor(slots: list[Slot]) -> tuple[int, int] | None:
    """Left edge of the first slot column and top of the first slot row."""
    if not slots:
        return None
    return min(s.rect.left for s in slots), min(s.rect.top for s in slots)


def _first_run(flags: np.ndarray, length: int) -> int | None:
    """Start of the first run of True values longer than length, if any."""
    run = 0
    for index, flag in enumerate(flags):
        run = run + 1 if flag else 0
        if run > length:
            return index - run + 1
    return None


def _read_row(rgb: np.ndarray, rect: Rect, config: str, upscale: int = UPSCALE) -> str:
    """OCR one text row: grayscale, upscale, Otsu, dark text on white.

    A cover over part of the row is cut off (see COVER_RUN) and the columns
    left of it are read.
    """
    height, width = rgb.shape[:2]
    left, top = max(0, rect.left), max(0, rect.top)
    crop = rgb[top : min(height, rect.bottom), left : min(width, rect.right)]
    if crop.size == 0:
        return ""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    columns = np.percentile(gray, 20, axis=0)
    # Reference is a low percentile, not the median, so a cover over more than
    # half of the crop is still flagged and not taken for the background
    off = np.abs(columns - np.percentile(columns, 10)) > BACKGROUND_TOLERANCE
    cover = _first_run(off, COVER_RUN)
    if cover is not None:
        background = np.median(gray)
        gray = gray[:, :cover]
        if gray.shape[1] <= EDGE_CLEAR:
            return ""
        if gray[:, -EDGE_CLEAR:].max() - background > TEXT_CONTRAST:
            return ""
    gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.mean() < 128:
        binary = 255 - binary
    binary = cv2.copyMakeBorder(binary, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    try:
        return pytesseract.image_to_string(binary, config=config).strip()
    except pytesseract.TesseractError:
        return ""


def _row_rect(
    frame_height: int,
    left: int,
    slot_top: int,
    above_top: float,
    above_bottom: float,
    x0: float,
    x1: float,
) -> Rect:
    """A text row above the slots: offsets above slot_top in frame heights."""
    return Rect(
        left + round(x0 * frame_height),
        slot_top - round(above_top * frame_height),
        left + round(x1 * frame_height),
        slot_top - round(above_bottom * frame_height),
    )


def _int(text: str) -> int:
    """Digits of a value such as 849,247 (Tesseract may read the comma as a dot)."""
    return int(re.sub(r"\D", "", text))


def read_stash_header(rgb: np.ndarray, stash_slots: list[Slot]) -> StashHeader | None:
    """Used and capacity slots and total value from the STASH header.

    None when no stash slots are given, the STASH title is not above them (a
    CONTAINER view), or any of the three numbers cannot be read. A stash
    filtered by category shows no capacity, which is also None.
    """
    anchor = _anchor(stash_slots)
    if anchor is None:
        return None
    left, top = anchor
    block = _row_rect(
        rgb.shape[0],
        left,
        top,
        TITLE_ABOVE_TOP,
        COUNTS_ABOVE_BOTTOM,
        0,
        STASH_TEXT_RIGHT,
    )
    lines = [ln for ln in _read_row(rgb, block, TWO_LINES).splitlines() if ln.strip()]
    if len(lines) < 2 or "STASH" not in lines[0].upper():
        return None
    match = _STASH.match(lines[-1].strip())
    if not match:
        return None
    used, capacity = int(match.group(1)), int(match.group(2))
    if capacity == 0 or used > capacity:
        return None
    return StashHeader(used, capacity, _int(match.group(3)))


def read_loadout_header(
    rgb: np.ndarray, equipment_slots: list[Slot]
) -> LoadoutHeader | None:
    """Weight, weight capacity and value from the LOADOUT header.

    None when no equipment slots are given or any field cannot be read.
    """
    anchor = _anchor(equipment_slots)
    if anchor is None:
        return None
    left, top = anchor
    counts = _row_rect(
        rgb.shape[0],
        left,
        top,
        COUNTS_ABOVE_TOP,
        COUNTS_ABOVE_BOTTOM,
        LOADOUT_TEXT_LEFT,
        LOADOUT_TEXT_RIGHT,
    )
    match = _LOADOUT.match(_read_row(rgb, counts, ONE_LINE, LOADOUT_UPSCALE).strip())
    if not match:
        return None
    weight_cap = float(match.group(2))
    if weight_cap <= 0:
        return None
    return LoadoutHeader(float(match.group(1)), weight_cap, _int(match.group(3)))
