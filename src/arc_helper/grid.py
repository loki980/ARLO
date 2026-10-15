"""
Find every inventory slot on a full-screen frame and name its panel.

The inventory draws each slot as a rounded rectangle with a thin light
outline (about 2 px at 4K), on a dark blue fill, with a gap of plain
background between neighbours. The search works on a copy scaled to 1080
rows, where the outline is 1 to 2 px wide:

1. ridge mask: ARLO's own gold frame bars are painted over with the
   background beside them first, as they lie just outside a slot's
   outline. Then the brightest channel minus its opening with a 5 px square
   (a white top hat) keeps thin bright lines such as the outlines, and
   drops flat areas, rarity wedges, the large bright parts of icons and the
   game's item card. Pixels at least 18 above the opening are kept.
2. candidates: each 8-connected component of the ridge mask whose bounding
   box is slot sized and lies between the panel headings and the bottom of
   the stash. Each side is moved inward onto the outline (the hovered
   slot's glow ring can join it and widen the box).
3. outlined sides: a side counts when, on at least 85% of its length
   (rounded corners left out), the side is 18 lighter than the darkest
   pixel just outside it. Boxes with four such sides set the grid slot
   side, and every kept box must have one of the slot shapes (grid slot,
   small equipment slot, weapon slot). A box with three outlined sides is
   kept too, as the fourth can lie against the item card.
4. neighbours: the outline of a slot beside one of ARLO's windows can join
   the window into a component too large to be a slot, so each found
   slot's neighbours of the same size are tested directly, and each new
   slot in turn.
5. panels from layout: slots are grouped into columns by their left and
   right edges, and touching columns (gap under 0.3 of the slot side) form
   a column group. Each group is named by where its middle lies across the
   16:9 content area (stash, equipment, backpack, and the right column of
   quick use, augmented slots and safe pocket). In the right column the
   three blocks sit under their headings at fixed heights, so a slot is
   named by its top edge as a fraction of the frame height.
6. filled: a slot's inner area with at least 2% bright or colourful pixels
   (the icon, its rarity wedge and its count) is filled. Empty slots show
   only a dim placeholder glyph. A slot partly under a card or one of
   ARLO's windows can read as filled.

Slots wholly or mostly under an item card or ARLO's windows are missing.

frames_still compares two frames on a 1/8 scale copy, restricted to the
given slot rects, so later steps only read the grid when it is not
scrolling.

Pure functions, no Tk or screen access.
"""

from collections.abc import Iterable
from typing import NamedTuple

import cv2
import numpy as np

from .tooltip_layout import Rect

WORK_HEIGHT = 1080
TOPHAT_SIDE = 5
RIDGE_MIN = 18
# Slot sizes in work pixels (a 4K grid slot is 96 px here, the equipment
# slots 144 by 96 and the weapon slots 304 by 201)
SLOT_MIN_SIDE = 80
SLOT_MAX_WIDTH = 330
SLOT_MAX_HEIGHT = 230
ASPECT_MIN = 0.85
ASPECT_MAX = 1.65
SIDE_BAND = 3
SIDE_TRIM = 0.1
SIDE_COVER = 0.85
# Every slot has one of these sizes, in grid slot sides: grid slot, small
# equipment slot, weapon slot. The grid slot side is the median side of the
# square boxes outlined on all four sides. A box with only three outlined
# sides is kept too when it has one of these sizes, as its fourth side can
# lie against the edge of the game's item card, which is lighter than the
# outline. The tolerance is a share of the grid slot side for each shape.
SLOT_SHAPES = ((1.0, 1.0), (1.49, 1.0), (3.15, 2.08))
SHAPE_TOLERANCE = 0.025
# Gaps between neighbouring slots in work pixels (8 in the grids, 15 to 23
# between the equipment slots)
GAP_MIN = 4
GAP_MAX = 26
SNAP = 10
GROUP_GAP = 0.3
FILL_INSET = 0.15
FILL_BRIGHT = 80
FILL_COLOURFUL = 70
FILL_FRACTION = 0.02
# ARLO's gold frame colour (cell_border.GOLD, #f2b134) as RGB
ARLO_GOLD = (242, 177, 52)
GOLD_TOLERANCE = 14
GOLD_FILL = 7
RIM_RED_OVER_BLUE = 40
RIM_GREEN_OVER_BLUE = 15
STILL_SCALE = 8
STILL_MEAN = 3.0

STASH = "stash"
EQUIPMENT = "equipment"
BACKPACK = "backpack"
QUICK_USE = "quick use"
AUGMENTED = "augmented slots"
SAFE_POCKET = "safe pocket"
PANELS = (STASH, EQUIPMENT, BACKPACK, QUICK_USE, AUGMENTED, SAFE_POCKET)
RIGHT_COLUMN = "right column"

# Middle of each column group across the 16:9 content area, measured on the
# 4K captures (stash 152 to 560 of 1920, equipment 692 to 996, backpack
# 1044 to 1452, right column 1504 to 1808)
GROUP_MIDDLES = (
    (STASH, 0.185),
    (EQUIPMENT, 0.44),
    (BACKPACK, 0.65),
    (RIGHT_COLUMN, 0.865),
)
# Right column blocks by slot top over frame height: quick use rows start at
# 0.252 (two rows end at 0.437), augmented slots at 0.503, safe pocket at
# 0.659
QUICK_USE_END = 0.47
AUGMENTED_END = 0.625
# The slots lie between the panel headings (the top row starts at 0.252 of
# the frame height) and the stash's bottom edge (0.89), so slot shaped
# things above or below, such as the item icon in the game's upgrade popup,
# are not slots
GRID_TOP = 0.235
GRID_BOTTOM = 0.9


class Slot(NamedTuple):
    """One slot rect in frame pixels, its panel and whether it has an item."""

    rect: Rect
    panel: str
    filled: bool


def _work_frame(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = rgb.shape[:2]
    scale = height / WORK_HEIGHT
    if height == WORK_HEIGHT:
        return np.ascontiguousarray(rgb), 1.0
    size = (max(1, round(width / scale)), WORK_HEIGHT)
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA), scale


def _brightest(rgb: np.ndarray) -> np.ndarray:
    red, green, blue = cv2.split(rgb)
    return cv2.max(cv2.max(red, green), blue)


def _hide_gold(work: np.ndarray, high: np.ndarray) -> None:
    """Darken ARLO's gold frame bars in the brightest channel, in place.

    The bars lie just outside a slot's outline and would join it, so their
    pixels take the darkest value within GOLD_FILL pixels (the background
    beside the bar), and so does the 1 px rim around them where the scaling
    blended the gold with its neighbour, when that pixel is still warm. The
    outline beside a bar is not warm and stays.
    """
    low = tuple(max(0, c - GOLD_TOLERANCE) for c in ARLO_GOLD)
    high_bound = tuple(min(255, c + GOLD_TOLERANCE) for c in ARLO_GOLD)
    gold = cv2.inRange(work, low, high_bound)
    if not gold.any():
        return
    red, green, blue = cv2.split(work)
    warm = (cv2.subtract(red, blue) >= RIM_RED_OVER_BLUE) & (
        cv2.subtract(green, blue) >= RIM_GREEN_OVER_BLUE
    )
    rim = cv2.dilate(gold, np.ones((3, 3), np.uint8)) > 0
    hidden = (gold > 0) | (rim & warm)
    background = cv2.erode(high, np.ones((GOLD_FILL, GOLD_FILL), np.uint8))
    high[hidden] = background[hidden]


def _ridge_mask(high: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (TOPHAT_SIDE, TOPHAT_SIDE))
    top = cv2.morphologyEx(high, cv2.MORPH_TOPHAT, kernel)
    return cv2.threshold(top, RIDGE_MIN - 1, 1, cv2.THRESH_BINARY)[1]


def _slot_sized(width: int, height: int) -> bool:
    if width < SLOT_MIN_SIDE or height < SLOT_MIN_SIDE:
        return False
    if width > SLOT_MAX_WIDTH or height > SLOT_MAX_HEIGHT:
        return False
    return ASPECT_MIN <= width / height <= ASPECT_MAX


def _sides(high: np.ndarray, box: Rect) -> list[tuple[np.ndarray, int, slice]]:
    """Each side of the box as (view, line, span), turned to face up.

    In each view the side's outer line is row `line`, the inside of the box
    lies at higher rows and `span` is the side's extent along the row.
    """
    height, width = high.shape
    across = slice(box.left, box.right)
    down = slice(box.top, box.bottom)
    return [
        (high, box.top, across),
        (high[::-1], height - box.bottom, across),
        (high.T, box.left, down),
        (high.T[::-1], width - box.right, down),
    ]


def _side_cover(view: np.ndarray, line: int, span: slice, band: int) -> float:
    """Share of a side where its band is lighter than just outside it.

    The band is `band` rows from `line` inward, the outside SIDE_BAND rows
    beyond it. The darkest outside pixel is the reference, so the hovered
    slot's glow ring a few pixels beyond its outline does not hide the
    outline, while the game's item card lying over the edge does. The
    corners are left out, as the slots' corners are rounded.
    """
    if line < SIDE_BAND or line + band > view.shape[0]:
        return 0.0
    edge = view[line : line + band, span].max(axis=0).astype(np.int16)
    outside = view[line - SIDE_BAND : line, span].min(axis=0)
    lift = edge - outside
    trim = int(lift.size * SIDE_TRIM)
    middle = lift[trim : lift.size - trim]
    return float((middle >= RIDGE_MIN).mean()) if middle.size else 0.0


def _snap_line(view: np.ndarray, line: int, span: slice, reach: int) -> int:
    """The side's outline row within reach rows inward of line.

    The hovered slot's glow ring and ARLO's gold frame lie a few pixels
    outside the outline and can join it in one component, which widens the
    box. The side moves to the innermost run of rows that are lighter than
    just outside on at least SIDE_COVER of the side, to the run's outer row
    as the outline can be two rows thick. With no such row it stays.
    """
    good = [
        _side_cover(view, line + depth, span, 1) >= SIDE_COVER
        for depth in range(reach + 1)
    ]
    if not any(good):
        return line
    depth = len(good) - 1 - good[::-1].index(True)
    while depth > 0 and good[depth - 1]:
        depth -= 1
    return line + depth


def _snap(high: np.ndarray, box: Rect) -> Rect:
    """The box with each side moved onto the outline (see _snap_line)."""
    height, width = high.shape
    reach = min(SNAP, box.height // 4, box.width // 4)
    top, bottom, left, right = (
        _snap_line(view, line, span, reach) for view, line, span in _sides(high, box)
    )
    return Rect(left, top, width - right, height - bottom)


def _side_covers(high: np.ndarray, box: Rect) -> list[float]:
    """How much of each side (top, bottom, left, right) is lighter than outside."""
    return [
        _side_cover(view, line, span, SIDE_BAND)
        for view, line, span in _sides(high, box)
    ]


def _known_shape(box: Rect, side: float) -> bool:
    """True when the box has the size of one of the slot shapes."""
    return any(
        abs(box.width - wide * side) <= SHAPE_TOLERANCE * side
        and abs(box.height - tall * side) <= SHAPE_TOLERANCE * side
        for wide, tall in SLOT_SHAPES
    )


def _is_filled(work: np.ndarray, high: np.ndarray, box: Rect) -> bool:
    inset_x = int(box.width * FILL_INSET)
    inset_y = int(box.height * FILL_INSET)
    rows = slice(box.top + inset_y, box.bottom - inset_y)
    cols = slice(box.left + inset_x, box.right - inset_x)
    inner_high = high[rows, cols]
    if inner_high.size == 0:
        return False
    red, green, blue = cv2.split(np.ascontiguousarray(work[rows, cols]))
    low = cv2.min(cv2.min(red, green), blue)
    lit = (inner_high >= FILL_BRIGHT) | (
        cv2.subtract(inner_high, low) >= FILL_COLOURFUL
    )
    return float(lit.mean()) >= FILL_FRACTION


def _boxes(high: np.ndarray) -> list[Rect]:
    mask = _ridge_mask(high)
    _count, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    three_sided = []
    for left, top, width, height, _area in stats[1:].tolist():
        if not _slot_sized(width, height):
            continue
        if top < GRID_TOP * mask.shape[0] or top + height > GRID_BOTTOM * mask.shape[0]:
            continue
        box = _snap(high, Rect(left, top, left + width, top + height))
        covers = sorted(_side_covers(high, box))
        if covers[0] >= SIDE_COVER:
            boxes.append(box)
        elif covers[1] >= SIDE_COVER:
            three_sided.append(box)
    squares = [
        (box.width + box.height) / 2
        for box in boxes
        if abs(box.width - box.height) <= SHAPE_TOLERANCE * box.height
    ]
    if not squares:
        return []
    side = float(np.median(squares))
    boxes = [box for box in boxes + three_sided if _known_shape(box, side)]
    return _complete(high, _innermost(list(dict.fromkeys(boxes))))


def _overlaps(a: Rect, b: Rect) -> bool:
    return (
        a.left < b.right and b.left < a.right and a.top < b.bottom and b.top < a.bottom
    )


def _neighbour(high: np.ndarray, box: Rect, step_x: int, step_y: int) -> Rect | None:
    """The outlined slot of the box's size beside it in one direction, if any.

    Candidates are placed at every gap from GAP_MIN to GAP_MAX. Those whose
    side facing the box is outlined are compared on all four sides (those
    whose far side is outlined when none is, as the facing side can be
    covered), and the best counts when three of its sides are outlined, as
    it has the size of a slot already found.
    """
    height, width = high.shape
    facing, far = {(0, 1): (0, 1), (0, -1): (1, 0), (1, 0): (2, 3), (-1, 0): (3, 2)}[
        step_x, step_y
    ]
    candidates = []
    for gap in range(GAP_MIN, GAP_MAX + 1):
        left = box.left + step_x * (box.width + gap)
        top = box.top + step_y * (box.height + gap)
        candidate = Rect(left, top, left + box.width, top + box.height)
        if (
            candidate.left < SIDE_BAND
            or candidate.right + SIDE_BAND > width
            or candidate.top < GRID_TOP * height
            or candidate.bottom > GRID_BOTTOM * height
        ):
            continue
        sides = _sides(high, candidate)
        near_cover = _side_cover(*sides[facing], SIDE_BAND)
        far_cover = _side_cover(*sides[far], SIDE_BAND)
        candidates.append((near_cover, far_cover, candidate))
    pool = [c for near, _far, c in candidates if near >= SIDE_COVER]
    pool = pool or [c for _near, far, c in candidates if far >= SIDE_COVER]
    best = None
    best_covers: list[float] = []
    for candidate in pool:
        covers = sorted(_side_covers(high, candidate))
        if covers > best_covers:
            best, best_covers = candidate, covers
    if best is None or best_covers[1] < SIDE_COVER:
        return None
    return best


def _complete(high: np.ndarray, boxes: list[Rect]) -> list[Rect]:
    """Add the slots beside found ones that the component search missed.

    A slot's outline can join one of ARLO's windows or another bright shape
    beside it into a component too large to be a slot. Each found slot's
    four neighbours of the same size are tested directly, and each new slot
    is tested in turn.
    """
    found = list(boxes)
    queue = list(boxes)
    while queue:
        box = queue.pop()
        for step_x, step_y in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            reach = Rect(
                box.left + step_x * (box.width + GAP_MIN),
                box.top + step_y * (box.height + GAP_MIN),
                box.right + step_x * (box.width + GAP_MIN),
                box.bottom + step_y * (box.height + GAP_MIN),
            )
            if any(_overlaps(reach, other) for other in found):
                continue
            new = _neighbour(high, box, step_x, step_y)
            if new is not None and not any(_overlaps(new, other) for other in found):
                found.append(new)
                queue.append(new)
    return found


def _innermost(boxes: list[Rect]) -> list[Rect]:
    """Drop boxes that contain another box.

    Only the outline itself is slot sized inside a slot, so a box around
    another is the hovered slot's glow ring around its outline.
    """

    def inside(a: Rect, b: Rect) -> bool:
        return (
            a != b
            and b.left <= a.left
            and b.top <= a.top
            and a.right <= b.right
            and a.bottom <= b.bottom
        )

    return [b for b in boxes if not any(inside(a, b) for a in boxes)]


def _column_groups(boxes: list[Rect], side: float) -> list[list[Rect]]:
    """Boxes grouped by overlapping or nearly touching column spans."""
    order = sorted(boxes, key=lambda box: box.left)
    groups: list[list[Rect]] = []
    right = None
    for box in order:
        if right is None or box.left - right > GROUP_GAP * side:
            groups.append([])
            right = box.right
        groups[-1].append(box)
        right = max(right, box.right)
    return groups


def _group_name(group: list[Rect], width: int, height: int) -> str:
    content = min(width, height * 16 / 9)
    origin = (width - content) / 2
    left = min(box.left for box in group)
    right = max(box.right for box in group)
    middle = ((left + right) / 2 - origin) / content
    return min(GROUP_MIDDLES, key=lambda item: abs(item[1] - middle))[0]


def _right_column_panel(box: Rect, height: int) -> str:
    top = box.top / height
    if top < QUICK_USE_END:
        return QUICK_USE
    if top < AUGMENTED_END:
        return AUGMENTED
    return SAFE_POCKET


def _to_frame(box: Rect, scale: float, width: int, height: int) -> Rect:
    return Rect(
        max(0, round(box.left * scale)),
        max(0, round(box.top * scale)),
        min(width, round(box.right * scale)),
        min(height, round(box.bottom * scale)),
    )


def find_slots(rgb: np.ndarray) -> list[Slot]:
    """Every outlined inventory slot in an RGB frame, in frame pixels.

    Slots partly covered by the game's item card or ARLO's windows may be
    missing. The list is sorted by panel order, then row, then column.
    """
    work, scale = _work_frame(rgb)
    work_height, work_width = work.shape[:2]
    high = _brightest(work)
    _hide_gold(work, high)
    boxes = _boxes(high)
    if not boxes:
        return []
    side = float(np.median([min(box.width, box.height) for box in boxes]))
    height, width = rgb.shape[:2]
    slots = []
    for group in _column_groups(boxes, side):
        name = _group_name(group, work_width, work_height)
        for box in group:
            panel = name
            if name == RIGHT_COLUMN:
                panel = _right_column_panel(box, work_height)
            slots.append(
                Slot(
                    _to_frame(box, scale, width, height),
                    panel,
                    _is_filled(work, high, box),
                )
            )
    slots.sort(key=lambda s: (PANELS.index(s.panel), s.rect.top, s.rect.left))
    return slots


def _small_gray(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    size = (max(1, width // STILL_SCALE), max(1, height // STILL_SCALE))
    small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)


def frames_still(
    a: np.ndarray, b: np.ndarray, slots: Iterable[Rect | Slot] | None = None
) -> bool:
    """True when two RGB frames of the inventory barely differ.

    The frames are compared on a 1/8 scale grey copy by the mean absolute
    difference, inside the given slot rects when there are any and over the
    whole frame otherwise. Frames of different sizes are never still.
    """
    if a.shape != b.shape:
        return False
    diff = cv2.absdiff(_small_gray(a), _small_gray(b))
    rects = [s.rect if isinstance(s, Slot) else s for s in slots or ()]
    if rects:
        area = np.zeros(diff.shape, bool)
        for rect in rects:
            area[
                rect.top // STILL_SCALE : -(-rect.bottom // STILL_SCALE),
                rect.left // STILL_SCALE : -(-rect.right // STILL_SCALE),
            ] = True
        values = diff[area]
    else:
        values = diff.ravel()
    if values.size == 0:
        return True
    return float(values.mean()) < STILL_MEAN
