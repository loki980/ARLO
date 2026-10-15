"""Read the tier numeral and the stack count printed on an inventory slot.

Every inventory slot has a dark bar along its bottom edge. The bar shows a
type icon at the left and, at the right, either a stack count such as x90
in white, or for weapons a Roman numeral I to IV in a grey serif font. Wide
weapon slots (EQUIPMENT) also show ammo and durability figures in the bar,
with the numeral still the rightmost mark.

read_tier and read_stack take an RGB slot crop (as from find_hovered_cell)
and look only at that bar:

- The bar is the longest run of near-black rows in the lower half of the
  slot, measured on the middle columns, and its horizontal extent is the
  longest run of columns that are near-black in the bar's top and bottom
  rows (this drops the coloured slot frame).
- Ink is anything clearly brighter than the bar. Ink columns are grouped
  into words, with gaps narrower than a fifth of the bar height bridged.
  The rightmost word is the mark.
- A tier numeral is grey ink with serifs: its top and bottom rows are inked
  across nearly its full width. Its middle band (serifs cut away) splits
  into one column run per stroke: one, two or three narrow runs are I, II
  or III, a narrow run then a wide one is IV.
- A stack count is white ink that starts with a short x glyph. The digits
  after it are read by Tesseract.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import pytesseract
from PIL import Image

log = logging.getLogger(__name__)

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
_WHITE_MIN = 180.0
_TIMES_MAX_HEIGHT = 0.8  # the x glyph over the digit height
_RIGHT_HALF = 0.5  # a stack count ends in the right half of the bar
_OCR_HEIGHT = 48  # digits are scaled to this height for Tesseract
_OCR_PAD = 12
_OCR_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789"


class TierRead(NamedTuple):
    """A tier read from the numeral on a slot."""

    tier: str
    confidence: float
    detail: str


class _Bar(NamedTuple):
    lum: np.ndarray  # luminance of the bar, rows by columns
    ink: np.ndarray  # bool, same shape
    background: float


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


def _find_bar(slot_rgb: np.ndarray) -> _Bar | None:
    lum = _luminance(slot_rgb)
    if lum.ndim != 2 or min(lum.shape) < 20:
        return None
    height, width = lum.shape
    middle = np.median(lum[:, int(width * 0.3) : int(width * 0.7)], axis=1)
    top, bottom = _longest_run(middle <= _BAR_MAX, height // 2)
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
    return _Bar(bar, bar > background + _INK_ABOVE, background)


def _last_word(bar: _Bar) -> tuple[int, int] | None:
    """Column span of the rightmost ink word in the bar."""
    height = bar.ink.shape[0]
    words = _runs(bar.ink.any(axis=0), max(2, round(_WORD_GAP * height)))
    words = [w for w in words if bar.ink[:, w[0] : w[1]].sum() >= _MIN_INK]
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


def _stack_digits(slot_rgb: np.ndarray) -> np.ndarray | None:
    """The digit part of a white xN stack count as a luminance crop, or None."""
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
    left = word[0] + glyphs[1][0]
    return bar.lum[digit_rows[0] : digit_rows[-1] + 1, left : word[1]]


def read_stack(slot_rgb: np.ndarray) -> int | None:
    """Read the xN stack count at the right of a slot's bottom bar.

    Returns the count, or None when the slot shows no stack count or
    Tesseract cannot read it.
    """
    digits = _stack_digits(slot_rgb)
    if digits is None:
        return None
    scale = _OCR_HEIGHT / digits.shape[0]
    image = Image.fromarray(np.clip(255 - digits, 0, 255).astype(np.uint8))
    image = image.resize(
        (max(1, round(digits.shape[1] * scale)), _OCR_HEIGHT), Image.Resampling.LANCZOS
    )
    padded = Image.new(
        "L", (image.width + 2 * _OCR_PAD, image.height + 2 * _OCR_PAD), 255
    )
    padded.paste(image, (_OCR_PAD, _OCR_PAD))
    try:
        text = pytesseract.image_to_string(padded, config=_OCR_CONFIG)
    except (pytesseract.TesseractError, OSError) as e:
        log.warning("stack count OCR failed: %s", e)
        return None
    text = text.strip()
    return int(text) if text.isdigit() else None
