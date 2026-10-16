"""Read the Collections screen: which blueprints the player has found.

On the BLUEPRINTS tab the game prints FOUND: n/91 above a lattice of cells in
rows of 17, one per blueprint in a fixed order. A found cell is a blue
blueprint tile with a white check mark in its top right corner, an empty cell
is a dark box with a thin grey border. read_collections reads one frame of
that screen, record_collections stores what it read in the user database.
Cells are numbered by grid index from 1, row by row, as blueprint_grid.csv
and Database.blueprint_at number them.

Geometry is measured on Jeffrey's screenshot of 15 Oct, 3840x2160 scaled to
2000x1125, and kept in frame heights so it scales to any frame size. Columns
are offsets from the left edge of the frame, scaled by height as well (the
screen is laid out from the left).
"""

import re
from typing import TYPE_CHECKING
from typing import NamedTuple

import cv2
import numpy as np
import pytesseract

from arc_helper.tooltip_layout import Rect

if TYPE_CHECKING:
    from arc_helper.database import Database

# Measured on the 1125 px tall fixture
_REF_HEIGHT = 1125
# Text rows (left, top, right, bottom)
TITLE_BOX = (135 / _REF_HEIGHT, 14 / _REF_HEIGHT, 300 / _REF_HEIGHT, 52 / _REF_HEIGHT)
TAB_BOX = (46 / _REF_HEIGHT, 144 / _REF_HEIGHT, 210 / _REF_HEIGHT, 192 / _REF_HEIGHT)
FOUND_BOX = (44 / _REF_HEIGHT, 220 / _REF_HEIGHT, 200 / _REF_HEIGHT, 248 / _REF_HEIGHT)
# The selected tab is outlined: its left edge is a bright vertical stroke at
# x 50 to 52 and its top edge a bright horizontal stroke at y 148 to 150
TAB_LEFT_EDGE = (50 / _REF_HEIGHT, 168 / _REF_HEIGHT)
TAB_TOP_EDGE = (127 / _REF_HEIGHT, 149 / _REF_HEIGHT)
EDGE_SEARCH = 6 / _REF_HEIGHT
OUTLINE_BRIGHT = 150
# Lattice: first cell's top left corner, pitch and cell size
GRID_LEFT = 50 / _REF_HEIGHT
GRID_TOP = 303 / _REF_HEIGHT
PITCH = 108.375 / _REF_HEIGHT
CELL = 100 / _REF_HEIGHT
COLUMNS = 17
# The first cell's corner is looked for within this distance of where it is
# expected, so a frame a few pixels off still reads
CORNER_SEARCH = 8 / _REF_HEIGHT
# Every cell, found or empty, has a border far brighter than the black
# background (gray about 60 to 90 against 3 on the fixture)
BORDER_BRIGHT = 25
# A found tile is dark blue (about 12, 26, 68 on the fixture): at least
# BLUE_FRACTION of the interior has blue over red by more than BLUE_MARGIN.
# The art covers up to half of a tile (0.48 to 0.73 blue on the fixture), an
# empty cell has none
BLUE_MARGIN = 20
BLUE_FRACTION = 0.25
# The check mark: bright pixels (every channel at least CHECK_WHITE) over at
# least CHECK_MIN_FRACTION of the top right corner of the cell, the corner in
# fractions of the cell size. The mark covers 0.034 to 0.11 of the corner on
# the fixture scaled from 720 to 2160 px tall, plain tile or background none
CHECK_BOX = (0.72, 0.03, 0.98, 0.25)
CHECK_WHITE = 160
CHECK_MIN_FRACTION = 0.015

ONE_LINE = "--psm 7 --oem 1"
_FOUND = re.compile(r"FOUND\W*(\d+)\s*/\s*(\d+)", re.IGNORECASE)


class CollectionsCell(NamedTuple):
    # Slot number as Database.blueprint_at takes it: from 1, row by row
    grid_index: int
    rect: Rect
    found: bool


class CollectionsRead(NamedTuple):
    found_count: int
    total: int
    cells: list[CollectionsCell]
    # True when the found cells number found_count and the cells number total
    consistent: bool

    @property
    def found_indices(self) -> list[int]:
        """Grid indices (from 1) of the found cells."""
        return [cell.grid_index for cell in self.cells if cell.found]


def _box(height: int, box: tuple[float, float, float, float]) -> Rect:
    left, top, right, bottom = box
    return Rect(
        round(left * height),
        round(top * height),
        round(right * height),
        round(bottom * height),
    )


def _read_text(gray: np.ndarray, rect: Rect) -> str:
    """OCR one bright text row on the dark background as dark on white."""
    crop = gray[rect.top : rect.bottom, rect.left : rect.right]
    if crop.size == 0:
        return ""
    # Text is read at the size of a 2160 px frame or larger
    scale = max(1.0, 2 * 1080 / gray.shape[0])
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = 255 - binary
    binary = cv2.copyMakeBorder(binary, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    try:
        return pytesseract.image_to_string(binary, config=ONE_LINE).strip()
    except pytesseract.TesseractError:
        return ""


def _letters(text: str) -> str:
    return re.sub(r"[^A-Z]", "", text.upper())


def _tab_outlined(gray: np.ndarray) -> bool:
    """The BLUEPRINTS tab's outline: a bright left edge and top edge."""
    height = gray.shape[0]
    reach = max(2, round(EDGE_SEARCH * height))
    x, y = round(TAB_LEFT_EDGE[0] * height), round(TAB_LEFT_EDGE[1] * height)
    left_edge = gray[y, max(0, x - reach) : x + reach + 1]
    x, y = round(TAB_TOP_EDGE[0] * height), round(TAB_TOP_EDGE[1] * height)
    top_edge = gray[max(0, y - reach) : y + reach + 1, x]
    return bool(
        left_edge.size > 0
        and top_edge.size > 0
        and left_edge.max() >= OUTLINE_BRIGHT
        and top_edge.max() >= OUTLINE_BRIGHT
    )


def _border_at(gray: np.ndarray, left: int, top: int, size: int) -> bool:
    """A cell border along the top and left edges of a cell at left, top."""
    height, width = gray.shape
    if left < 0 or top < 0 or left + size > width or top + size > height:
        return False
    band = max(2, size // 25)
    middle = slice(size // 4, size - size // 4)
    top_edge = gray[top : top + band, left : left + size][:, middle].max(axis=0)
    left_edge = gray[top : top + size, left : left + band][middle, :].max(axis=1)
    return bool(
        np.median(top_edge) > BORDER_BRIGHT and np.median(left_edge) > BORDER_BRIGHT
    )


def _first_corner(gray: np.ndarray) -> tuple[int, int] | None:
    """Top left corner of the first cell: its left and top border strokes."""
    height = gray.shape[0]
    x0, y0 = round(GRID_LEFT * height), round(GRID_TOP * height)
    reach = max(2, round(CORNER_SEARCH * height))
    size = round(CELL * height)
    middle_y = y0 + size // 2
    middle_x = x0 + size // 2
    row = gray[middle_y, max(0, x0 - reach) : x0 + reach + 1]
    column = gray[max(0, y0 - reach) : y0 + reach + 1, middle_x]
    bright_x = np.flatnonzero(row > BORDER_BRIGHT)
    bright_y = np.flatnonzero(column > BORDER_BRIGHT)
    if bright_x.size == 0 or bright_y.size == 0:
        return None
    return max(0, x0 - reach) + int(bright_x[0]), max(0, y0 - reach) + int(bright_y[0])


def _is_found(rgb: np.ndarray, rect: Rect) -> bool:
    """A blue blueprint tile with the white check mark in its top right."""
    size = rect.right - rect.left
    inset = size // 10
    interior = rgb[
        rect.top + inset : rect.bottom - inset, rect.left + inset : rect.right - inset
    ].astype(int)
    blue = (interior[..., 2] - interior[..., 0] > BLUE_MARGIN).mean()
    if blue < BLUE_FRACTION:
        return False
    left, top, right, bottom = CHECK_BOX
    corner = rgb[
        rect.top + round(top * size) : rect.top + round(bottom * size),
        rect.left + round(left * size) : rect.left + round(right * size),
    ]
    white = (corner >= CHECK_WHITE).all(axis=2)
    return bool(white.size > 0 and white.mean() >= CHECK_MIN_FRACTION)


def _cells(rgb: np.ndarray, gray: np.ndarray) -> list[CollectionsCell]:
    """Every cell in row-major order, up to the first slot without one."""
    corner = _first_corner(gray)
    if corner is None:
        return []
    height = gray.shape[0]
    pitch = PITCH * height
    size = round(CELL * height)
    cells: list[CollectionsCell] = []
    index = 0
    while True:
        row, column = divmod(index, COLUMNS)
        left = corner[0] + round(column * pitch)
        top = corner[1] + round(row * pitch)
        if not _border_at(gray, left, top, size):
            break
        rect = Rect(left, top, left + size, top + size)
        cells.append(CollectionsCell(index + 1, rect, _is_found(rgb, rect)))
        index += 1
    return cells


def read_collections(rgb: np.ndarray) -> CollectionsRead | None:
    """Read one RGB frame of the Collections screen on its BLUEPRINTS tab.

    None unless the frame shows the COLLECTIONS title, the BLUEPRINTS tab
    selected and a FOUND n/total count. The cells are read whatever the
    count says, and the read is consistent only when they agree with it.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height = gray.shape[0]
    # The cheap outline check first, so other screens cost no OCR
    if not _tab_outlined(gray):
        return None
    if _letters(_read_text(gray, _box(height, TAB_BOX))) != "BLUEPRINTS":
        return None
    if "COLLECTIONS" not in _letters(_read_text(gray, _box(height, TITLE_BOX))):
        return None
    match = _FOUND.search(_read_text(gray, _box(height, FOUND_BOX)))
    if match is None:
        return None
    found_count, total = int(match.group(1)), int(match.group(2))
    cells = _cells(rgb, gray)
    consistent = (
        len(cells) == total
        and len([cell for cell in cells if cell.found]) == found_count
    )
    return CollectionsRead(found_count, total, cells, consistent)


def record_collections(db: "Database", read: CollectionsRead) -> list[int]:
    """Store a consistent read's ownership in the user database.

    Each cell whose blueprint is known is marked owned when found and not
    owned when empty. An inconsistent read records nothing. Returns the
    grid indices of found cells whose blueprint is unknown (slots 26, 40 and
    56 in the current grid data), to be resolved by other means.
    """
    if not read.consistent:
        return []
    unknown_found = []
    for cell in read.cells:
        name = db.blueprint_at(cell.grid_index)
        if name is None:
            if cell.found:
                unknown_found.append(cell.grid_index)
            continue
        db.set_blueprint_owned(name, owned=cell.found)
    return unknown_found
