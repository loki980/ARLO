"""Read the tier numeral and the stack count printed on an inventory slot.

Every inventory slot has a dark bar along its bottom edge. The bar shows a
type icon at the left and, at the right, either a stack count such as x90
in white, or for weapons a Roman numeral I to IV in a grey serif font. Wide
weapon slots (EQUIPMENT) also show ammo and durability figures in the bar,
with the numeral still the rightmost mark.

read_tier and read_stack take an RGB slot crop (as from find_hovered_cell)
and look only at that bar. bar_geometry gives the bar's box, its colour and
the boxes of the type icon and the right mark, for painting over the icon:

- The bar is the longest run of near-black rows in the lower half of the
  slot, measured on the middle columns (rows between them crossed by wide
  bright text, such as ARLO's price patch, also count), and its horizontal
  extent is the longest run of columns that are near-black in the bar's
  top and bottom rows (this drops the coloured slot frame).
- Ink is anything clearly brighter than the bar. Ink columns are grouped
  into words, with gaps narrower than a fifth of the bar height bridged.
  The rightmost word is the mark.
- A tier numeral is grey ink with serifs: its top and bottom rows are inked
  across nearly its full width. Its middle band (serifs cut away) splits
  into one column run per stroke: one, two or three narrow runs are I, II
  or III, a narrow run then a wide one is IV.
- A stack count is white ink that starts with a short x glyph. Each digit
  after it is matched against templates of the game's digit glyphs, built
  by scripts/eval_stack_reads.py from counts labelled on 4K captures.

learned_check takes a whole frame and a slot rect, and reads the slot's top
right corner for the white check mark the game draws on a learned
blueprint (see the constants above it for the measured mark).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NamedTuple

import cv2
import numpy as np

from .tooltip_layout import Rect

if TYPE_CHECKING:
    from collections.abc import Sequence

TIER_TOKENS = ("I", "II", "III", "IV")

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# Bar location
_BAR_MAX = 24.0  # median brightness of a bar row (the bar is about 15)
_BAR_EDGE_MAX = 30.0  # a column inside the bar is this dark in its edge rows
_BAR_MIN_HEIGHT = 0.08  # of the slot height
_BAR_EDGE_ROWS = 2

# Ink and words
_INK_ABOVE = 45.0  # ink is this much brighter than the bar
_WORD_GAP = 0.2  # of the bar height, narrower gaps join one word
_MIN_INK = 6  # a word with fewer ink pixels is noise

# Tier numerals, sizes as fractions of the glyph height unless named
_GREY_MIN = 60.0  # the numeral ink is about 100
_GREY_MAX = 170.0  # the stack ink is about 220
_HEIGHT_MIN = 0.4  # glyph height over bar height, numerals are about 0.56
_HEIGHT_MAX = 0.75
_SERIF_ROWS = 0.12  # the top and bottom rows checked for serifs
_SERIF_MIN = 0.8  # share of the glyph width those rows cover (1.0 seen)
_MID_TOP, _MID_BOTTOM = 0.3, 0.7  # the band without serifs
_NARROW_MAX = 0.3  # an I stroke (about 0.16 to 0.2)
_WIDE_MIN, _WIDE_MAX = 0.4, 0.75  # the V of IV (about 0.56)

# Stack counts
_WHITE_MIN = 150.0  # median count ink, 218 on 4K captures, 170 upscaled
_TIMES_MAX_HEIGHT = 0.8  # the x glyph over the digit height
_RIGHT_HALF = 0.5  # a stack count ends in the right half of the bar
_EXTEND_MAX = 0.5  # of the digit height, ink followed past the bar's right edge
_CELL_WIDTH = 0.72  # a glyph is centred in a cell this wide, of the digit height
_GLYPH_W, _GLYPH_H = 10, 16  # the cell is scaled to this for matching
_INK_LEVEL = 90  # percentile of the ink luminance that counts as full ink
_GLYPH_MAX_WIDTH = 0.85  # of the digit height, a wider run is joined glyphs
_MATCH_BLUR = 0.8  # Gaussian sigma on the cells and templates before matching
_MATCH_MAX = 0.2  # 1 - correlation to the best template, 4K glyphs are under 0.14
_MATCH_GAP = 0.05  # the second best is this much further, 4K glyphs 0.098 or more

# Bar geometry
_ICON_START_MAX = 0.3  # the type icon starts in this share of the bar width
PATCH_FALLBACK = 0.55  # the patch boundary, of the bar width, without a mark
_PATCH_GAP = 0.1  # of the bar height, kept clear left of the right mark
_ROW_TRIM = 3.0  # an edge row whose 90th percentile is this much brighter
# A real bar sits under a lighter tile body (about 13 brighter, a blocked or
# all-dark slot about 1) and spans most of the slot (about 0.89 of the width,
# 0.23 of the height)
_BODY_ABOVE = 6.0
_BODY_ROWS = 5
_BAR_MIN_WIDTH = 0.7  # of the slot width
_BAR_MAX_HEIGHT = 0.35  # of the slot height


class TierRead(NamedTuple):
    """A tier read from the numeral on a slot."""

    tier: str
    confidence: float
    detail: str


class BarGeometry(NamedTuple):
    """Where a square slot's bottom bar and its marks sit, and its colour.

    Every rectangle is crop-local (in the slot crop's pixels), right and
    bottom exclusive. bar_rect is the bar inside the tile's rounded outline.
    colour is the bar's integer RGB, the median of its plain pixels between
    the type icon and the right mark. icon_box is the type icon at the bar's
    left, or None when the bar has no ink there. right_mark_box is the stack
    count or tier numeral at the right, or None when there is none.
    """

    bar_rect: Rect
    colour: tuple[int, int, int]
    icon_box: Rect | None
    right_mark_box: Rect | None

    @property
    def patch_right(self) -> int:
        """Right edge (exclusive) of the bar part that can be painted over.

        A small gap left of the right mark, or PATCH_FALLBACK of the bar
        width when there is no right mark.
        """
        return _patch_right(self.bar_rect, self.right_mark_box)


class _Bar(NamedTuple):
    lum: np.ndarray  # luminance of the bar, rows by columns
    ink: np.ndarray  # bool, same shape
    background: float
    rect: Rect  # the bar's place in the slot crop


def _luminance(slot_rgb: np.ndarray) -> np.ndarray:
    return np.asarray(slot_rgb)[..., :3].astype(np.float32) @ _LUMA


def _longest_run(mask: np.ndarray, start: int = 0) -> tuple[int, int]:
    """Return the longest run of True in mask[start:] as (begin, end)."""
    padded = np.concatenate(([False], mask[start:], [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    if edges.size == 0:
        return start, start
    begins, ends = edges[0::2], edges[1::2]
    best = int(np.argmax(ends - begins))
    return start + int(begins[best]), start + int(ends[best])


def _runs(mask: np.ndarray, bridge: int) -> list[tuple[int, int]]:
    """True runs of a 1D mask as (begin, end), joining gaps under bridge."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    runs: list[list[int]] = []
    for begin, end in zip(edges[0::2], edges[1::2], strict=True):
        if runs and begin - runs[-1][1] < bridge:
            runs[-1][1] = int(end)
        else:
            runs.append([int(begin), int(end)])
    return [(a, b) for a, b in runs]


def _bar_rows(columns: np.ndarray) -> np.ndarray:
    """Which rows of the slot's middle columns are bar rows.

    A bar row has a median at most _BAR_MAX. Wide bright text across the
    bar, such as the worth on ARLO's price patch, can lift a row's median
    above that. A run of such rows between two bar rows still counts when
    the median of each row's pixels other than ink (those lighter than
    _BAR_MAX + _INK_ABOVE) is bar dark, so the text does not split the bar.
    """
    rows = np.median(columns, axis=1) <= _BAR_MAX
    for begin, end in zip(*_gaps(rows), strict=True):
        between = columns[begin:end]
        plain = np.where(between <= _BAR_MAX + _INK_ABOVE, between, np.nan)
        if np.isnan(plain).all(axis=1).any():
            continue
        if (np.nanmedian(plain, axis=1) <= _BAR_MAX).all():
            rows[begin:end] = True
    return rows


def _gaps(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Begins and ends of the False runs of a 1D mask with True on both sides."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return edges[1:-1:2], edges[2::2]


def _find_bar(slot_rgb: np.ndarray) -> _Bar | None:
    lum = _luminance(slot_rgb)
    if lum.ndim != 2 or min(lum.shape) < 20:
        return None
    height, width = lum.shape
    columns = lum[:, int(width * 0.3) : int(width * 0.7)]
    top, bottom = _longest_run(_bar_rows(columns), height // 2)
    if bottom - top < max(_BAR_MIN_HEIGHT * height, 2 * _BAR_EDGE_ROWS + 4):
        return None
    edge_rows = np.concatenate(
        (lum[top : top + _BAR_EDGE_ROWS], lum[bottom - _BAR_EDGE_ROWS : bottom])
    )
    left, right = _longest_run(edge_rows.max(axis=0) <= _BAR_EDGE_MAX)
    if right - left < bottom - top:
        return None
    bar = lum[top:bottom, left:right]
    background = float(np.median(bar))
    return _Bar(
        bar, bar > background + _INK_ABOVE, background, Rect(left, top, right, bottom)
    )


def _last_word(bar: _Bar) -> tuple[int, int] | None:
    """Column span of the rightmost ink word in the bar."""
    words = _words(bar)
    return words[-1] if words else None


def _trim(ink: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(ink.any(axis=1))
    cols = np.flatnonzero(ink.any(axis=0))
    return ink[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _margin(value: float, threshold: float, full: float) -> float:
    """0 at the threshold, 1 at full, clipped. full may be above or below."""
    return float(min(1.0, max(0.0, (value - threshold) / (full - threshold))))


def _classify_numeral(glyph: np.ndarray) -> tuple[str | None, float, str]:
    """Return (tier or None, shape score 0 to 1, detail) for a trimmed glyph."""
    height, width = glyph.shape
    band = max(1, round(_SERIF_ROWS * height))
    top = glyph[:band].any(axis=0).sum() / width
    bottom = glyph[-band:].any(axis=0).sum() / width
    middle = glyph[int(_MID_TOP * height) : int(_MID_BOTTOM * height) + 1]
    strokes = [(b - a) / height for a, b in _runs(middle.any(axis=0), 2)]
    detail = f"serifs={top:.2f}/{bottom:.2f} strokes={[round(s, 2) for s in strokes]}"
    serif = min(top, bottom)
    if serif < _SERIF_MIN or not strokes:
        return None, 0.0, detail
    score = _margin(serif, _SERIF_MIN, 1.0)
    narrow = [s <= _NARROW_MAX for s in strokes]
    if all(narrow) and len(strokes) <= 3:
        score = min(
            [score] + [_margin(s, _NARROW_MAX, _NARROW_MAX / 2) for s in strokes]
        )
        return TIER_TOKENS[len(strokes) - 1], score, detail
    if len(strokes) == 2 and narrow[0] and _WIDE_MIN <= strokes[1] <= _WIDE_MAX:
        centre = (_WIDE_MIN + _WIDE_MAX) / 2
        wide = 1.0 - abs(strokes[1] - centre) / (centre - _WIDE_MIN)
        score = min(score, _margin(strokes[0], _NARROW_MAX, _NARROW_MAX / 2), wide)
        return "IV", score, detail
    return None, 0.0, detail


def read_tier(slot_rgb: np.ndarray) -> TierRead | None:
    """Read the Roman numeral tier mark at the right of a slot's bottom bar.

    Returns the tier ("I" to "IV") with a confidence between 0.5 and 1, or
    None when the slot has no grey serif numeral there.
    """
    bar = _find_bar(slot_rgb)
    if bar is None:
        return None
    word = _last_word(bar)
    if word is None:
        return None
    ink = bar.ink[:, word[0] : word[1]]
    brightness = float(np.median(bar.lum[:, word[0] : word[1]][ink]))
    if not _GREY_MIN <= brightness <= _GREY_MAX:
        return None
    glyph = _trim(ink)
    height_ratio = glyph.shape[0] / bar.ink.shape[0]
    if not _HEIGHT_MIN <= height_ratio <= _HEIGHT_MAX:
        return None
    tier, shape, detail = _classify_numeral(glyph)
    if tier is None:
        return None
    grey_mid = (_GREY_MIN + _GREY_MAX) / 2
    grey = 1.0 - abs(brightness - grey_mid) / (grey_mid - _GREY_MIN)
    confidence = 0.5 + 0.5 * min(shape, grey)
    return TierRead(
        tier,
        round(confidence, 3),
        f"ink={brightness:.0f} h={height_ratio:.2f} {detail}",
    )


def _stack_digits(slot_rgb: np.ndarray) -> tuple[np.ndarray, float] | None:
    """The digits of a white xN stack count as a luminance crop, or None.

    Returns the crop and the bar's background luminance. A count can run a
    few columns past the bar's right edge (where the slot's rounded corner
    ends the bar early), so ink touching that edge is followed into the slot.
    """
    bar = _find_bar(slot_rgb)
    if bar is None:
        return None
    word = _last_word(bar)
    if word is None or word[1] < _RIGHT_HALF * bar.ink.shape[1]:
        return None
    ink = bar.ink[:, word[0] : word[1]]
    if float(np.median(bar.lum[:, word[0] : word[1]][ink])) < _WHITE_MIN:
        return None
    glyphs = _runs(ink.any(axis=0), 1)
    if len(glyphs) < 2:
        return None
    rows = np.flatnonzero(ink.any(axis=1))
    digit_ink = ink[:, glyphs[1][0] :]
    digit_rows = np.flatnonzero(digit_ink.any(axis=1))
    times_rows = np.flatnonzero(ink[:, glyphs[0][0] : glyphs[0][1]].any(axis=1))
    digit_height = digit_rows[-1] - digit_rows[0] + 1
    times_height = times_rows[-1] - times_rows[0] + 1
    # The x sits lower and is shorter than the digits beside it
    if times_height > _TIMES_MAX_HEIGHT * digit_height or times_rows[0] <= rows[0]:
        return None
    top = bar.rect.top + int(digit_rows[0])
    bottom = bar.rect.top + int(digit_rows[-1]) + 1
    left = bar.rect.left + word[0] + glyphs[1][0]
    right = bar.rect.left + word[1]
    lum = _luminance(slot_rgb)
    if word[1] == bar.ink.shape[1]:
        limit = min(lum.shape[1], right + round(_EXTEND_MAX * digit_height))
        while right < limit and (
            lum[top:bottom, right].max() > bar.background + _INK_ABOVE
        ):
            right += 1
    return lum[top:bottom, left:right], bar.background


# Digit templates 0 to 9, _GLYPH_H rows of _GLYPH_W columns, one hex digit
# (0 bar to f full ink) per pixel, from scripts/eval_stack_reads.py --templates
_DIGIT_HEX = (
    # 0, 21 glyphs
    (
        "0048bb830004ceeeec301ce8349eb04eb1002ce25f80000ae46f800009e46f800009e46f800009e4"
        "6f800009e46f800009e46f800009e45f90000ae43ec2002de21bfa55afa003beeeea200037996200"
    ),
    # 1, 59 glyphs
    (
        "00026a8100002aefd200003daed20000141ed20000000ed20000000ed20000000ed20000000ed200"
        "00000ed20000000ed20000000ed20000000ed20000000ed20000000ed20000000ed2000000099100"
    ),
    # 2, 28 glyphs
    (
        "0159a9510007deeed7103ed746de406f81007f806e50005f902520008f70000002de30000009f800"
        "00005ec3000003de4000002bf71000019fa1000006ec2000003df84444406ffeeeeed14aaaaaaa91"
    ),
    # 3, 23 glyphs
    (
        "1abbbbbba11ceeeeffd1033335ef9000001afb2000018fd3000006ef6100001affb500000366ce50"
        "0000005eb00000001dd11640001ce22db1001ce11de4004ec008fb55cf7002aeeee9200026996200"
    ),
    # 4, 8 glyphs
    (
        "0001aa10000005fa0000001af40000003ec00000008f70000002dd20121006f9007d401ce3008f50"
        "4eb1008f509fdbbbdfc5aeeeeeefe62333339f710000008f500000008f500000008f500000004930"
    ),
    # 5, 19 glyphs
    (
        "5bbbbbbb306feeeeed406f833333106f600000006f600000006f758851006fdeffd7006fe957de40"
        "5c91018f801210004ea00100004eb04b60004ea04eb2019f701be958ed3004ceeed6000037984100"
    ),
    # 6, 4 glyphs
    (
        "0159ba610007deeee8103ed636de406f90008e607f600025207f758851007fcdfed7007fe957ee30"
        "7fc2019f707f80004ea07f70003eb06f80004ea04eb1008f701ce957ed3004ceeed6000037985100"
    ),
    # 7, 8 glyphs
    (
        "4bbbbbbba06feeeeefd06e53336ec03610007f80000000ce40000003ec10000007f80000000ce300"
        "00002fc10000007f70000001bf30000003ec10000007f80000001bf30000003ec000000039600000"
    ),
    # 8, 5 glyphs
    (
        "003798510004ceeed7101be957dd404eb2008f704ea1006e802cd523be5007edbcea2005dedef810"
        "1be735cd404ea1006e805f70004ea16f70004ea14ea1007e802ce845ce5005ceeed8100048aa6100"
    ),
    # 9, 11 glyphs
    (
        "0038bb820002beeeeb2009fa44af801dd3003dd12ec0000ce22ec0001ce21dd2003de309fa44afe3"
        "03cedddee30038a94ce30111110ce309a2000ce20ae5003dd106fc66bfa0019eeeeb300025897200"
    ),
)
_DIGIT_TEMPLATES = (
    np.array([[int(c, 16) for c in h] for h in _DIGIT_HEX], dtype=np.float32) / 15.0
).reshape(10, _GLYPH_H, _GLYPH_W)


def _match_vector(cell: np.ndarray) -> np.ndarray | None:
    """A cell blurred, flattened, zero mean and unit length, or None if flat."""
    vector = cv2.GaussianBlur(cell, (3, 3), _MATCH_BLUR).ravel()
    vector -= vector.mean()
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-6 else None


_TEMPLATE_VECTORS = np.array([_match_vector(t) for t in _DIGIT_TEMPLATES])


def _glyph_cells(digits: np.ndarray, background: float) -> list[np.ndarray] | None:
    """Each digit glyph of a stack count, centred and scaled for matching.

    A cell is the glyph's columns over the full digit height, its ink scaled
    to 0 (bar) to 1 (full ink), centred in a box _CELL_WIDTH of the height
    wide and resized to _GLYPH_W by _GLYPH_H. None when a glyph is too wide
    to be one digit.
    """
    ink = digits > background + _INK_ABOVE
    if not ink.any():
        return None
    height = digits.shape[0]
    full = float(np.percentile(digits[ink], _INK_LEVEL))
    scaled = np.clip((digits - background) / max(1.0, full - background), 0, 1)
    cells = []
    for begin, end in _runs(ink.any(axis=0), 1):
        width = end - begin
        if width > _GLYPH_MAX_WIDTH * height:
            return None
        cell_width = max(width, round(_CELL_WIDTH * height))
        cell = np.zeros((height, cell_width), dtype=np.float32)
        offset = (cell_width - width) // 2
        cell[:, offset : offset + width] = scaled[:, begin:end]
        cells.append(
            cv2.resize(cell, (_GLYPH_W, _GLYPH_H), interpolation=cv2.INTER_AREA)
        )
    return cells


def _match_digit(cell: np.ndarray) -> str | None:
    """The digit whose template best correlates with the cell, or None if unsure.

    Both are blurred a little first, which keeps upscaled (blurrier) glyphs
    close to the 4K templates.
    """
    vector = _match_vector(cell)
    if vector is None:
        return None
    distances = 1.0 - _TEMPLATE_VECTORS @ vector
    best, second = np.argsort(distances)[:2]
    if distances[best] > _MATCH_MAX or distances[second] - distances[best] < _MATCH_GAP:
        return None
    return str(int(best))


def read_stack(slot_rgb: np.ndarray) -> int | None:
    """Read the xN stack count at the right of a slot's bottom bar.

    Returns the count, or None when the slot shows no stack count or a
    digit does not clearly match one of the game's digit glyphs.
    """
    found = _stack_digits(slot_rgb)
    if found is None:
        return None
    cells = _glyph_cells(*found)
    if not cells:
        return None
    digits = [_match_digit(cell) for cell in cells]
    if None in digits:
        return None
    return int("".join(digits))


def read_stacks(slot_crops: Sequence[np.ndarray]) -> list[int | None]:
    """read_stack for each slot crop, one result per crop in order."""
    return [read_stack(crop) for crop in slot_crops]


def _words(bar: _Bar) -> list[tuple[int, int]]:
    """Column spans of the ink words in the bar, left to right."""
    height = bar.ink.shape[0]
    words = _runs(bar.ink.any(axis=0), max(2, round(_WORD_GAP * height)))
    return [w for w in words if bar.ink[:, w[0] : w[1]].sum() >= _MIN_INK]


def _word_box(bar: _Bar, word: tuple[int, int]) -> Rect:
    """The crop-local box around a word's ink."""
    rows = np.flatnonzero(bar.ink[:, word[0] : word[1]].any(axis=1))
    return Rect(
        bar.rect.left + word[0],
        bar.rect.top + int(rows[0]),
        bar.rect.left + word[1],
        bar.rect.top + int(rows[-1]) + 1,
    )


def _patch_right(bar_rect: Rect, right_mark: Rect | None) -> int:
    if right_mark is None:
        return bar_rect.left + round(PATCH_FALLBACK * bar_rect.width)
    gap = max(2, round(_PATCH_GAP * bar_rect.height))
    return max(bar_rect.left, right_mark.left - gap)


def _inner_bar(slot_rgb: np.ndarray) -> _Bar | None:
    """The bar inside the tile's rounded outline, for bar_geometry.

    _find_bar's rows can include a blend row into the coloured corner swoosh
    above, which also cuts its columns short under the swoosh. Here edge
    rows lighter than the bar are trimmed first, then the columns are the
    ones dark in the trimmed edge rows, which stops at the rounded outline.
    A dark region is only a bar when it spans most of the slot's width, is
    not too tall, and the tile body just above it is lighter than it.
    """
    found = _find_bar(slot_rgb)
    if found is None:
        return None
    lum = _luminance(slot_rgb)
    top, bottom = found.rect.top, found.rect.bottom
    left, right = found.rect.left, found.rect.right
    limit = found.background + _ROW_TRIM

    def light(row: int) -> bool:
        return float(np.percentile(lum[row, left:right], 90)) > limit

    while bottom - top > 2 * _BAR_EDGE_ROWS and light(top):
        top += 1
    while bottom - top > 2 * _BAR_EDGE_ROWS and light(bottom - 1):
        bottom -= 1
    edge_rows = np.concatenate(
        (lum[top : top + _BAR_EDGE_ROWS], lum[bottom - _BAR_EDGE_ROWS : bottom])
    )
    left, right = _longest_run(edge_rows.max(axis=0) <= _BAR_EDGE_MAX)
    height, width = lum.shape
    if right - left < _BAR_MIN_WIDTH * width or bottom - top > _BAR_MAX_HEIGHT * height:
        return None
    bar = lum[top:bottom, left:right]
    background = float(np.median(bar))
    body = lum[max(0, top - _BODY_ROWS - 1) : top - 1, left:right]
    if body.size == 0 or float(np.median(body)) < background + _BODY_ABOVE:
        return None
    return _Bar(
        bar, bar > background + _INK_ABOVE, background, Rect(left, top, right, bottom)
    )


def bar_geometry(slot_rgb: np.ndarray) -> BarGeometry | None:
    """The bottom bar of a square slot crop: its box, colour and mark boxes.

    The type icon is the leftmost ink word when it starts in the bar's left
    part. The right mark is the rightmost other word when it ends in the
    bar's right half. The colour is the per-channel median of the bar's
    pixels between the icon and the patch boundary (see
    BarGeometry.patch_right), leaving out ink and the pixel around it.
    Returns None when the slot has no dark bar.
    """
    bar = _inner_bar(slot_rgb)
    if bar is None:
        return None
    width = bar.ink.shape[1]
    words = _words(bar)
    icon = words[0] if words and words[0][0] < _ICON_START_MAX * width else None
    rest = words[1:] if icon else words
    mark = rest[-1] if rest and rest[-1][1] >= _RIGHT_HALF * width else None
    icon_box = _word_box(bar, icon) if icon else None
    mark_box = _word_box(bar, mark) if mark else None
    # Ink grown by one pixel, so anti-aliased glyph edges stay out of the median
    near_ink = bar.ink.copy()
    near_ink[1:] |= bar.ink[:-1]
    near_ink[:-1] |= bar.ink[1:]
    near_ink[:, 1:] |= near_ink[:, :-1].copy()
    near_ink[:, :-1] |= near_ink[:, 1:].copy()
    begin = icon[1] if icon else 0
    end = _patch_right(bar.rect, mark_box) - bar.rect.left
    plain = np.zeros_like(near_ink)
    plain[:, begin:end] = ~near_ink[:, begin:end]
    if not plain.any():
        plain = ~near_ink
    if not plain.any():
        plain = np.ones_like(near_ink)
    pixels = np.asarray(slot_rgb)[
        bar.rect.top : bar.rect.bottom, bar.rect.left : bar.rect.right, :3
    ][plain]
    colour = np.median(pixels, axis=0).round().astype(int)
    return BarGeometry(
        bar.rect, (int(colour[0]), int(colour[1]), int(colour[2])), icon_box, mark_box
    )


# Learned blueprint check: the game draws a small white check mark in the
# top right corner of a learned blueprint's tile. Measured on the Vertical
# Grip III Blueprint tile of Jeffrey's 15 Oct screenshot, scaled to a 194 px
# slot: the mark spans 0.79 to 0.91 of the slot width and 0.10 to 0.19 of
# its height (about 24 by 17 px), its ink is about (222, 213, 208) on the
# blueprint grid, whose lines have a lowest channel under 100, 27 percent of
# its box is inked and 4 to 5 percent of its ink lies in the box's top left
# quarter. The tracked item eye drawn in the same corner spans 0.78 to 0.93
# by 0.09 to 0.20, fills 39 to 41 percent of its box, has 23 to 25 percent
# of its ink in the top left quarter and encloses 31 to 33 percent of its
# box around the pupil. All shares are of the slot's own width and height,
# so a screenshot stretched on one axis reads the same.
CHECK_ZONE = (0.70, 0.04, 0.97, 0.27)  # left, top, right, bottom shares
# Where the mark itself sits. Item art from outside the zone reaching in
# here can hide a mark, so the corner cannot be judged
CHECK_CORE = (0.77, 0.08, 0.93, 0.21)
_CHECK_INK_MIN = 150.0  # lowest channel of mark ink
_CHECK_INK_SPREAD = 60.0  # highest minus lowest channel of mark ink
_CHECK_MIN_AREA = 0.0012  # of the slot area, the mark is 0.0029
_CHECK_WIDTH = (0.08, 0.17)  # box width over slot width, the mark is 0.12
_CHECK_HEIGHT = (0.05, 0.14)  # box height over slot height, the mark is 0.09
_CHECK_FILL_MAX = 0.34  # inked share of the box, the mark 0.27, the eye 0.4
_CHECK_QUARTER_MAX = 0.12  # ink share in the top left quarter, 0.05 vs 0.23
_CHECK_HOLE_MAX = 0.05  # enclosed unlit share of the box, 0 vs the eye's 0.31
# Slot width over height outside which the corner is not read. Blueprints
# sit in 1 by 1 slots, and Jeffrey's screenshot stretched to 16:9 has 1.25
_CHECK_ASPECT = (0.74, 1.35)
_COVER_LUMA = 170.0  # luminance that counts as covering the corner
_COVER_SHARE = 0.5  # zone share of covering pixels that hides the mark


class _Ink(NamedTuple):
    """One white ink component in the zone, in slot shares and zone px."""

    width: float
    height: float
    fill: float
    quarter: float
    hole: float
    box: Rect


def _hole_share(own: np.ndarray) -> float:
    """Share of a component's box that is unlit but enclosed by its ink."""
    padded = np.pad(own.astype(np.uint8), 1)
    mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(padded, mask, (0, 0), 1)
    return float(np.count_nonzero(padded == 0)) / own.size


def _inks(zone: np.ndarray, slot_w: int, slot_h: int) -> list[_Ink]:
    """The white ink components in the mark zone, small specks dropped."""
    pixels = zone.astype(np.float32)
    low = pixels.min(axis=2)
    spread = pixels.max(axis=2) - low
    ink = ((low >= _CHECK_INK_MIN) & (spread <= _CHECK_INK_SPREAD)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    found = []
    for label in range(1, count):
        left, top, width, height, area = (int(v) for v in stats[label][:5])
        if area < _CHECK_MIN_AREA * slot_w * slot_h:
            continue
        own = labels[top : top + height, left : left + width] == label
        quarter = own[: max(1, height // 2), : max(1, width // 2)]
        found.append(
            _Ink(
                width / slot_w,
                height / slot_h,
                area / (width * height),
                float(np.count_nonzero(quarter)) / area,
                _hole_share(own),
                Rect(left, top, left + width, top + height),
            )
        )
    return found


def _is_check(ink: _Ink, zone_w: int, zone_h: int) -> bool:
    inside = (
        ink.box.left > 0
        and ink.box.top > 0
        and ink.box.right < zone_w
        and ink.box.bottom < zone_h
    )
    return (
        inside
        and _CHECK_WIDTH[0] <= ink.width <= _CHECK_WIDTH[1]
        and _CHECK_HEIGHT[0] <= ink.height <= _CHECK_HEIGHT[1]
        and ink.fill <= _CHECK_FILL_MAX
        and ink.quarter <= _CHECK_QUARTER_MAX
        and ink.hole <= _CHECK_HOLE_MAX
    )


def _share_box(rect: Rect, shares: tuple[float, float, float, float]) -> Rect:
    width, height = rect.right - rect.left, rect.bottom - rect.top
    return Rect(
        rect.left + round(shares[0] * width),
        rect.top + round(shares[1] * height),
        rect.left + round(shares[2] * width),
        rect.top + round(shares[3] * height),
    )


def learned_check(rgb: np.ndarray, slot_rect: Rect) -> bool | None:
    """Whether a slot shows the game's learned blueprint check mark.

    rgb is an RGB frame and slot_rect a slot in its pixels (as from
    grid.find_slots). Reads the top right corner zone (CHECK_ZONE) for a
    white check shaped component: inside the zone, check sized, thin (low
    fill), with almost no ink in its top left quarter and no enclosed hole
    (which rules out the tracked item eye). True with the mark, False
    without. None when the corner cannot be judged: the zone is off the
    frame, the slot is not about square, at least half the zone is bright
    (the item card or another window over it) or white item art reaching in
    from outside the zone covers the mark's own spot (CHECK_CORE).
    """
    width = slot_rect.right - slot_rect.left
    height = slot_rect.bottom - slot_rect.top
    if width <= 0 or height <= 0:
        return None
    if not _CHECK_ASPECT[0] <= width / height <= _CHECK_ASPECT[1]:
        return None
    zone_rect = _share_box(slot_rect, CHECK_ZONE)
    frame_h, frame_w = rgb.shape[:2]
    if (
        zone_rect.left < 0
        or zone_rect.top < 0
        or zone_rect.right > frame_w
        or zone_rect.bottom > frame_h
    ):
        return None
    zone = np.asarray(
        rgb[zone_rect.top : zone_rect.bottom, zone_rect.left : zone_rect.right, :3]
    )
    if zone.size == 0:
        return None
    if float(np.mean(_luminance(zone) >= _COVER_LUMA)) >= _COVER_SHARE:
        return None
    zone_h, zone_w = zone.shape[:2]
    inks = _inks(zone, width, height)
    if any(_is_check(ink, zone_w, zone_h) for ink in inks):
        return True
    core = _share_box(slot_rect, CHECK_CORE)
    core = Rect(
        core.left - zone_rect.left,
        core.top - zone_rect.top,
        core.right - zone_rect.left,
        core.bottom - zone_rect.top,
    )
    for ink in inks:
        reaches_in = (
            ink.box.left == 0
            or ink.box.top == 0
            or ink.box.right >= zone_w
            or ink.box.bottom >= zone_h
        )
        overlaps_core = (
            ink.box.left < core.right
            and core.left < ink.box.right
            and ink.box.top < core.bottom
            and core.top < ink.box.bottom
        )
        if reaches_in and overlaps_core:
            return None
    return False
