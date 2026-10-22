"""
Card title readers for ARLO.

A title reader turns the located item card into the item name text. Three
readers share one interface so the engine is chosen by the OCR_ENGINE setting
and the next engine is a drop-in:

- TesseractReader, today's two-pass Tesseract read on the binarized card.
- TesseractTitleReader, which isolates the title line by geometry first and
  runs a single-line Tesseract pass on it.
- PaddleReader, PP-OCRv4 detection and recognition through
  rapidocr-onnxruntime, an optional dependency imported only when chosen.

Each reader returns a TitleRead with the text, the line boxes it read, a
confidence and, when the name ends in a tier-like token, the stroke-count
vote on that token.
"""

import hashlib
import re
import string
import time
from collections import OrderedDict
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import Protocol

import cv2
import numpy as np
import pytesseract
from PIL import Image

from .config import logger
from .tiers import TIER_LIKE
from .tiers import TIER_TOKENS
from .tiers import TierVote
from .tiers import count_strokes
from .tiers import tier_key

if TYPE_CHECKING:
    from collections.abc import Callable

TESSERACT = "tesseract"
TESSERACT_TITLE = "tesseract_title"
PADDLE = "paddle"
ENGINES = (TESSERACT, TESSERACT_TITLE, PADDLE)
PADDLE_INSTALL = "uv sync --extra paddle"

NAME_WHITELIST = string.ascii_uppercase

# Card background bounds, the same as tooltip_layout's cream
CREAM_LOWER = np.array([240, 225, 210])
CREAM_UPPER = np.array([255, 250, 240])

# Title geometry, as fractions of the card's inner width unless noted
_TAG_RUN_MIN = 0.06  # a tag chip row has a non-background run this long
_TAG_FILL_MIN = 0.12  # and at least this much of the row is chip
_CHIP_MIN = 60  # a chip pixel's brightest channel is at least this
_TAG_SEARCH = 0.4  # the tag row starts in the top 40 percent of the card
_INK_MAX = 100  # a pixel with every channel below this is text ink
_TEXT_MAX = 160  # below this is text of any shade, light grey description too
_LIGHT_MAX = 220  # a background pixel's brightest channel is at least this
_LIGHT_MIN = 180  # and its darkest channel at least this
_BLACK_MAX = 55  # title ink is near black, description ink is grey
_TITLE_BLACK_SHARE = 0.35  # a title line has at least this share of black ink
_LINE_MIN_INK = 3  # a text row has at least this many ink pixels
_MAX_TITLE_LINES = 3
_BAND_MARGIN = 0.3  # margin around a title line, as a fraction of its height
_TITLE_SCALE = 3

_TRAILING_ONE = str.maketrans({"|": "I", "l": "I", "1": "I"})

Box = tuple[int, int, int, int]  # left, top, right, bottom


class TextLine(NamedTuple):
    """One detected text line and its box in the reader's input image."""

    text: str
    box: Box


class TitleRead(NamedTuple):
    """What a reader read from one card.

    line_boxes are (left, top, right, bottom) in the coordinates of the image
    the reader was given: the card crop for read, the binarized card for
    TesseractReader.read_binary. confidence is 0 to 1, or 0 when the reader
    has no measure. tier_vote is the stroke-count vote on the tier token that
    ends the name, or None when the name has no tier-like last token. lines
    are every text line the reader detected on the whole card, in the same
    coordinates, or None when the reader reads the title only.
    """

    text: str | None
    line_boxes: list[Box]
    confidence: float
    tier_vote: TierVote | None = None
    lines: list[TextLine] | None = None


NO_READ = TitleRead(None, [], 0.0)


class TitleReader(Protocol):
    """Reads the item name from a card crop.

    wants_binary readers can also read the binarized card from
    binarize_card through read_binary, so the engine can preprocess once and
    keep its debug images.
    """

    name: str
    wants_binary: bool

    def read(self, card_rgb: np.ndarray) -> TitleRead:
        """Read the title from an RGB card crop."""
        ...


class TitleReaderUnavailableError(RuntimeError):
    """The configured reader cannot run, for example a missing package."""


# Text helpers shared with OCREngine


def clean_text(text: str) -> str:
    """Clean OCR output text."""
    # Remove common OCR artifacts
    text = re.sub(r"[|\\/_]", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Strip
    return text.strip()


def upper_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are uppercase."""
    alpha = [c for c in text if c.isalpha()]
    return sum(c.isupper() for c in alpha) / len(alpha) if alpha else 0.0


def count_name_lines(text: str) -> int:
    """Count the leading mostly-uppercase lines that form the item name.

    Stops at the first lower-case line (the description), tolerating a
    stray mis-cased character (e.g. "II" read as "Il") via a ratio.
    """
    count = 0
    started = False
    for raw_line in text.split("\n"):
        line = clean_text(raw_line)
        if len(line) >= 2 and upper_ratio(line) >= 0.6:
            count += 1
            started = True
        elif started or len(line) >= 2:
            break
    return count


def last_name_word(lines: list[list[str]], line_count: int) -> int | None:
    """Index of the tier-like last word of the name, or None.

    lines are the words of each OCR line in order. The name lines are
    found the way count_name_lines finds them: short leading lines are
    skipped, then line_count lines form the name.
    """
    start = 0
    while start < len(lines) and len(clean_text(" ".join(lines[start]))) < 2:
        start += 1
    last = start + line_count - 1
    if last >= len(lines) or not lines[last]:
        return None
    if TIER_LIKE.match(tier_key(lines[last][-1])):
        return last
    return None


# Preprocessing


class BinaryCard(NamedTuple):
    """The binarized card: dark text black on white, upscaled 2x.

    stages are the intermediate images by debug file stem, filled only when
    binarize_card is asked for them.
    """

    image: Image.Image
    stages: dict[str, np.ndarray]


def binarize_card(img_array: np.ndarray, *, stages: bool = False) -> BinaryCard:
    """Binarize a card for the two-pass Tesseract read.

    Masks the cream card, crops to it, drops rows with colored tag pixels,
    keeps dark pixels as black ink on white and upscales 2x bilinear. With
    no cream region the result is a blank white image twice the input size.
    """
    height, width = img_array.shape[:2]

    def blank() -> BinaryCard:
        return BinaryCard(Image.new("L", (width * 2, height * 2), 255), {})

    lower_bound = CREAM_LOWER
    upper_bound = CREAM_UPPER

    mask = np.all((img_array >= lower_bound) & (img_array <= upper_bound), axis=2)

    row_cream_percentage = np.sum(mask, axis=1) / mask.shape[1]
    col_cream_percentage = np.sum(mask, axis=0) / mask.shape[0]

    cream_rows = np.where(row_cream_percentage > 0.5)[0]
    cream_cols = np.where(col_cream_percentage > 0.3)[0]

    if len(cream_rows) == 0 or len(cream_cols) == 0:
        return blank()

    y_min, y_max = cream_rows[0], cream_rows[-1]
    x_min, x_max = cream_cols[0], cream_cols[-1]

    cropped = img_array[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]

    n_rows = cropped_mask.shape[0]
    if n_rows == 0:
        return blank()
    col_cream_pct = np.sum(cropped_mask, axis=0) / n_rows
    tight_cols = np.where(col_cream_pct > 0.5)[0]

    if len(tight_cols) == 0:
        return blank()

    tight_x_min, tight_x_max = tight_cols[0], tight_cols[-1]

    tight_cropped = cropped[:, tight_x_min:tight_x_max]

    r, g, b = tight_cropped[:, :, 0], tight_cropped[:, :, 1], tight_cropped[:, :, 2]

    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    color_diff = max_rgb - min_rgb

    is_colored = (color_diff > 30) & (max_rgb > 80) & (max_rgb < 240)

    row_color_percentage = np.sum(is_colored, axis=1) / is_colored.shape[1]
    row_has_significant_color = row_color_percentage > 0.05

    result = (
        np.ones((tight_cropped.shape[0], tight_cropped.shape[1]), dtype=np.uint8) * 255
    )

    is_dark = np.all(tight_cropped < 100, axis=2)

    mask = (~row_has_significant_color[:, np.newaxis]) & is_dark
    result[mask] = 0

    result_image = Image.fromarray(result, mode="L")

    scale = 2
    new_size = (result_image.width * scale, result_image.height * scale)
    result_image = result_image.resize(new_size, Image.Resampling.BILINEAR)

    debug: dict[str, np.ndarray] = {}
    if stages:
        debug = {
            "tooltip_mask": mask.astype(np.uint8) * 255,
            "tooltip_cropped": cropped,
            "tooltip_tight_cropped": tight_cropped,
            "tooltip_colored": is_colored.astype(np.uint8) * 255,
            "tooltip_colored_rows": row_has_significant_color[:, np.newaxis]
            .repeat(tight_cropped.shape[1], axis=1)
            .astype(np.uint8)
            * 255,
            "tooltip_text_mask": result,
        }
    return BinaryCard(result_image, debug)


# Title geometry


def _inner_card(rgb: np.ndarray) -> Box | None:
    """The cream card inside a crop, without the margin around it."""
    if rgb.dtype == np.uint8:
        cream = cv2.inRange(
            np.ascontiguousarray(rgb),
            tuple(int(v) for v in CREAM_LOWER),
            tuple(int(v) for v in CREAM_UPPER),
        ).astype(bool)
    else:
        cream = np.all((rgb >= CREAM_LOWER) & (rgb <= CREAM_UPPER), axis=2)
    rows = np.flatnonzero(cream.mean(axis=1) > 0.5)
    cols = np.flatnonzero(cream.mean(axis=0) > 0.5)
    if rows.size == 0 or cols.size == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _longest_runs(mask: np.ndarray) -> np.ndarray:
    """Length of the longest run of True in each row of a 2D mask."""
    padded = np.pad(mask.astype(np.int8), ((0, 0), (1, 1)))
    edges = np.diff(padded, axis=1)
    longest = np.zeros(mask.shape[0], dtype=int)
    for y in np.flatnonzero(mask.any(axis=1)):
        starts = np.flatnonzero(edges[y] == 1)
        ends = np.flatnonzero(edges[y] == -1)
        longest[y] = int((ends - starts).max())
    return longest


def _bands(rows: np.ndarray) -> list[tuple[int, int]]:
    """Group a boolean row mask into (top, bottom) bands, bottom exclusive."""
    padded = np.pad(rows.astype(np.int8), 1)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def find_title_lines(card_rgb: np.ndarray) -> list[Box]:
    """Locate the title lines of a card by geometry, in crop coordinates.

    The colored tag row (category and rarity chips) is found as the first
    rows in the top of the card with a long run of non-background pixels.
    Below it, rows with ink of any shade form text lines. The title is the
    run of lines whose ink is near black, up to the first line of grey
    sentence-case description, light grey included. Returns [] when there is no tag row, which
    is the case when the title is cut off above the capture.
    """
    rgb = np.asarray(card_rgb)[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.int16)
    inner = _inner_card(rgb)
    if inner is None:
        return []
    left, top, right, bottom = inner
    card = rgb[top:bottom, left:right]
    height, width = card.shape[:2]
    if height < 20 or width < 20:
        return []

    if card.dtype == np.uint8:
        red, green, blue = cv2.split(np.ascontiguousarray(card))
        max_rgb = cv2.max(cv2.max(red, green), blue)
        min_rgb = cv2.min(cv2.min(red, green), blue)
    else:
        max_rgb = card.max(axis=2)
        min_rgb = card.min(axis=2)
    # Light and unsaturated: the cream card, its row stripes and white text
    background = (max_rgb >= _LIGHT_MAX) & (min_rgb >= _LIGHT_MIN)
    fill = (~background).mean(axis=1)
    # Chip fill: colored or grey, not the near black ink of text
    chip = (~background & (max_rgb >= _CHIP_MIN)).mean(axis=1)

    search = int(height * _TAG_SEARCH)
    # Only the rows searched for the tag row need their longest run
    runs = _longest_runs(~background[:search])
    tag_rows = np.flatnonzero(
        (runs[:search] >= _TAG_RUN_MIN * width) & (chip[:search] >= _TAG_FILL_MIN)
    )
    if tag_rows.size == 0:
        return []
    tag_bottom = int(tag_rows[0])
    while tag_bottom < height and fill[tag_bottom] >= _TAG_FILL_MIN:
        tag_bottom += 1

    ink = max_rgb < _INK_MAX
    black = max_rgb < _BLACK_MAX
    # Rows of text of any shade, so a light grey description is a line too
    # and ends the title instead of being skipped as blank
    text = max_rgb < _TEXT_MAX
    text_rows = text.sum(axis=1) >= _LINE_MIN_INK
    text_rows[:tag_bottom] = False

    lines: list[Box] = []
    for line_top, line_bottom in _bands(text_rows):
        if line_bottom - line_top < 6:
            continue  # a stray rule or speck, not a text line
        band_ink = ink[line_top:line_bottom]
        band_text = text[line_top:line_bottom]
        share = black[line_top:line_bottom].sum() / max(1, band_text.sum())
        if share < _TITLE_BLACK_SHARE:
            break  # grey sentence-case description
        cols = np.flatnonzero(band_ink.any(axis=0))
        lines.append(
            (
                left + int(cols[0]),
                top + line_top,
                left + int(cols[-1]) + 1,
                top + line_bottom,
            )
        )
        if len(lines) == _MAX_TITLE_LINES:
            break
    return lines


# Title look: each title line's ink, shrunk to LOOK_HEIGHT rows (its width in
# proportion) and thresholded again, so a card whose title row looks the same
# as before can skip the read. A look keeps every line's box size in px, so
# only an exact match of sizes and bits counts as the same title.
LOOK_HEIGHT = 16
# Title looks remembered per run, oldest dropped first
TITLE_CACHE_SIZE = 512


class TitleLook(NamedTuple):
    """The title rows of one card sighting, from find_title_lines.

    boxes are the title line boxes in the crop, bits each line's ink at
    LOOK_HEIGHT rows, and key a digest of the line sizes and bits. Two
    sightings with the same key show the same title pixels at that scale.
    """

    key: bytes
    boxes: list[Box]
    bits: list[np.ndarray]


def line_look(rgb: np.ndarray, box: Box) -> np.ndarray:
    """One title line's ink (every channel below _INK_MAX) at LOOK_HEIGHT rows."""
    left, top, right, bottom = box
    band = rgb[top:bottom, left:right]
    ink = (band.max(axis=2) < _INK_MAX).astype(np.uint8) * 255
    width = max(1, round((right - left) * LOOK_HEIGHT / max(1, bottom - top)))
    small = cv2.resize(ink, (width, LOOK_HEIGHT), interpolation=cv2.INTER_AREA)
    return small >= 128


def title_look(card_rgb: np.ndarray) -> TitleLook | None:
    """The look of a card crop's title rows, None when no title line is found."""
    rgb = np.ascontiguousarray(np.asarray(card_rgb)[:, :, :3], dtype=np.uint8)
    boxes = find_title_lines(rgb)
    if not boxes:
        return None
    bits = [line_look(rgb, box) for box in boxes]
    digest = hashlib.blake2b(digest_size=16)
    for (left, top, right, bottom), line in zip(boxes, bits, strict=True):
        digest.update(f"{right - left}x{bottom - top};".encode())
        digest.update(np.packbits(line).tobytes())
    return TitleLook(digest.digest(), boxes, bits)


class CachedTitle(NamedTuple):
    """A title read remembered for its look: the text and its tier vote."""

    text: str
    tier_vote: TierVote | None


class TitleCache:
    """Title reads by look, in memory for the run, at most size of them.

    Only an exact look key is a hit. The least recently used look is
    dropped first.
    """

    def __init__(self, size: int = TITLE_CACHE_SIZE) -> None:
        self.size = size
        self._reads: OrderedDict[bytes, CachedTitle] = OrderedDict()

    def __len__(self) -> int:
        return len(self._reads)

    def get(self, key: bytes) -> CachedTitle | None:
        found = self._reads.get(key)
        if found is not None:
            self._reads.move_to_end(key)
        return found

    def put(self, key: bytes, read: CachedTitle) -> None:
        self._reads[key] = read
        self._reads.move_to_end(key)
        while len(self._reads) > self.size:
            self._reads.popitem(last=False)


def _band_crop(gray: np.ndarray, box: Box) -> tuple[np.ndarray, Box]:
    """Crop a line box from a grayscale card with margin, clamped."""
    left, top, right, bottom = box
    margin = max(4, round(_BAND_MARGIN * (bottom - top)))
    height, width = gray.shape
    padded = (
        max(0, left - margin),
        max(0, top - margin),
        min(width, right + margin),
        min(height, bottom + margin),
    )
    return gray[padded[1] : padded[3], padded[0] : padded[2]], padded


def _otsu_band(gray_band: np.ndarray, scale: int) -> np.ndarray:
    """Upscale a grayscale band with cubic interpolation and Otsu threshold.

    Ink comes out black (0) on white (255).
    """
    big = cv2.resize(gray_band, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def fix_trailing_token(text: str) -> str:
    """Uppercase a title read, mapping |, l and 1 to I in its last token.

    The mapping applies only when it turns the last token, without stray
    punctuation, into a Roman tier token, so a name that ends in a real L
    keeps it.
    """
    words = text.split()
    if not words:
        return ""
    token = re.sub(r"[^A-Za-z0-9|]", "", words[-1])
    mapped = token.translate(_TRAILING_ONE).upper()
    if len(words) > 1 and mapped in TIER_TOKENS:
        words[-1] = mapped
    return clean_text(" ".join(words).upper())


def trailing_token_box(ink: np.ndarray) -> Box | None:
    """Box of the last word in a binary line image, ink True, or None.

    Words are column runs of ink separated by gaps of at least a quarter of
    the line's ink height, the width of a word space in the card font.
    """
    rows = np.flatnonzero(ink.any(axis=1))
    cols = np.flatnonzero(ink.any(axis=0))
    if rows.size == 0:
        return None
    height = int(rows[-1] - rows[0] + 1)
    min_gap = max(2, height // 4)
    end = int(cols[-1]) + 1
    start = end - 1
    gap = 0
    x = start
    while x >= cols[0]:
        if ink[:, x].any():
            start = x
            gap = 0
        else:
            gap += 1
            if gap >= min_gap:
                break
        x -= 1
    return start, int(rows[0]), end, int(rows[-1]) + 1


def vote_on_box(ink: np.ndarray, box: Box) -> TierVote:
    """Count the strokes of the token inside box of a binary image."""
    left, top, right, bottom = box
    return count_strokes(ink[top:bottom, left:right])


# Readers


class TesseractReader:
    """Today's reader: two Tesseract passes on the binarized card.

    The plain pass locates the name block (the leading mostly-uppercase
    lines, set off from the lower-case description by case). A second
    uppercase-whitelisted pass supplies the text with Roman numerals
    intact - a plain pass reads "II" as "Il" or drops it to "I", which
    silently selects the wrong item tier.
    """

    name = TESSERACT
    wants_binary = True

    def read(self, card_rgb: np.ndarray) -> TitleRead:
        """Binarize the card crop, then read it as read_binary does."""
        return self.read_binary(binarize_card(np.asarray(card_rgb)).image)

    def read_binary(self, processed: Image.Image) -> TitleRead:
        """Read the item name from a binarized card via two OCR passes."""
        plain = pytesseract.image_to_string(processed, config="--psm 6")
        logger.debug(f"Raw tooltip OCR:\n{plain}")

        line_count = count_name_lines(plain)
        if line_count == 0:
            return NO_READ

        whitelisted = pytesseract.image_to_string(
            processed,
            config=f"--psm 6 -c tessedit_char_whitelist={NAME_WHITELIST}",
        )
        name_lines = [ln.strip() for ln in whitelisted.split("\n") if ln.strip()]

        # The word boxes cost a third Tesseract pass, so only fetch them
        # when the plain read already ends the name in a tier-like token
        vote = None
        boxes: list[Box] = []
        plain_lines = [ln.split() for ln in plain.split("\n")]
        if last_name_word(plain_lines, line_count) is not None:
            vote, boxes = self._vote_on_tier(processed, line_count)

        text = clean_text(" ".join(name_lines[:line_count])) or None
        return TitleRead(text, boxes, 0.0, vote)

    @staticmethod
    def _vote_on_tier(
        processed: Image.Image, line_count: int
    ) -> tuple[TierVote | None, list[Box]]:
        """Count the strokes of the tier token that ends the name.

        Reads word boxes from the plain pass, crops the last word of the
        last name line from the processed image (ink is black) and counts
        its strokes. Returns the vote, None when the token cannot be
        located, and the box of that word.
        """
        try:
            data = pytesseract.image_to_data(
                processed, config="--psm 6", output_type=pytesseract.Output.DICT
            )
        except pytesseract.TesseractError as e:
            logger.debug(f"Tier vote skipped, word boxes failed: {e}")
            return None, []

        # Tesseract numbers lines per block and paragraph
        lines: dict[tuple[int, int, int], list[int]] = {}
        for i, text in enumerate(data["text"]):
            if text.strip():
                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                lines.setdefault(key, []).append(i)
        boxes = list(lines.values())
        words = [[data["text"][i] for i in line] for line in boxes]

        last = last_name_word(words, line_count)
        if last is None:
            return None, []
        i = boxes[last][-1]
        left, top = data["left"][i], data["top"][i]
        box = (left, top, left + data["width"][i], top + data["height"][i])
        ink = np.array(processed.convert("L").crop(box)) < 128
        vote = count_strokes(ink)
        logger.debug(f"Tier vote on {data['text'][i]!r}: {vote}")
        return vote, [box]


class TesseractTitleReader:
    """Tesseract on the title line alone, isolated by card geometry.

    Each title line found by find_title_lines is cropped with margin,
    upscaled 3x with cubic interpolation, thresholded with Otsu and read
    with --psm 7 and no whitelist. The text is uppercased with |, l and 1
    read as I in the trailing tier token. The stroke vote runs on the
    Tesseract box of that token.
    """

    name = TESSERACT_TITLE
    wants_binary = False

    def read(self, card_rgb: np.ndarray) -> TitleRead:  # noqa: PLR6301 - protocol
        """Read the title lines of an RGB card crop."""
        rgb = np.ascontiguousarray(np.asarray(card_rgb)[:, :, :3], dtype=np.uint8)
        lines = find_title_lines(rgb)
        if not lines:
            return NO_READ
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        texts: list[str] = []
        confidences: list[float] = []
        last_word: tuple[np.ndarray, Box, str] | None = None
        for box in lines:
            band, _ = _band_crop(gray, box)
            binary = _otsu_band(band, _TITLE_SCALE)
            data = pytesseract.image_to_data(
                binary, config="--psm 7", output_type=pytesseract.Output.DICT
            )
            words = [
                i
                for i, text in enumerate(data["text"])
                if text.strip() and float(data["conf"][i]) >= 0
            ]
            texts.extend(data["text"][i].strip() for i in words)
            confidences.extend(float(data["conf"][i]) / 100 for i in words)
            if words:
                i = words[-1]
                left, top = data["left"][i], data["top"][i]
                word_box = (left, top, left + data["width"][i], top + data["height"][i])
                last_word = (binary < 128, word_box, data["text"][i])

        text = fix_trailing_token(" ".join(texts)) or None
        vote = None
        if text and last_word is not None:
            ink, word_box, raw = last_word
            # A tier-like last word votes on its own box. Otherwise the
            # numeral may have been dropped from the text, so the last ink
            # word of the line votes instead.
            box = word_box if TIER_LIKE.match(tier_key(raw)) else None
            box = box or trailing_token_box(ink)
            if box is not None:
                vote = vote_on_box(ink, box)
                logger.debug(f"Tier vote after {raw!r}: {vote}")
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return TitleRead(text, lines, confidence, vote)


class PaddleReader:
    """PP-OCRv4 detection and recognition through rapidocr-onnxruntime.

    One detection and recognition pass runs on the whole card crop. The
    title is the detected lines that sit on the title lines card geometry
    finds, or the tallest detected line when geometry finds none. The stroke
    vote runs on the last word of the last title line, found by column gaps
    in its Otsu binary. Every detected line comes back in TitleRead.lines so
    the label rows can be found without another OCR pass. The package is
    imported when the reader is built, so the app runs without it when
    another engine is chosen.
    """

    name = PADDLE
    wants_binary = False

    def __init__(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:
            msg = (
                "OCR_ENGINE=paddle needs the rapidocr-onnxruntime package. "
                f"Install it with: {PADDLE_INSTALL}"
            )
            raise TitleReaderUnavailableError(msg) from e
        self._ocr: Callable[..., Any] = RapidOCR()

    def read(self, card_rgb: np.ndarray) -> TitleRead:
        """Read every line of a card crop and pick the title among them."""
        rgb = np.ascontiguousarray(np.asarray(card_rgb)[:, :, :3], dtype=np.uint8)
        result, _ = self._ocr(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), use_cls=False)
        height, width = rgb.shape[:2]
        detected: list[tuple[TextLine, float]] = []
        for points, raw, score in result or []:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            box = (
                max(0, round(min(xs))),
                max(0, round(min(ys))),
                min(width, round(max(xs))),
                min(height, round(max(ys))),
            )
            detected.append((TextLine(str(raw), box), float(score)))
        detected.sort(key=lambda d: (d[0].box[1], d[0].box[0]))
        lines = [line for line, _ in detected]
        if not detected:
            return TitleRead(None, [], 0.0, None, lines)

        title = _title_lines(detected, find_title_lines(rgb))
        raw_text = " ".join(line.text for line, _ in title)
        score = sum(s for _, s in title) / len(title)
        boxes = [line.box for line, _ in title]

        text = fix_trailing_token(raw_text) or None
        vote = None
        left, top, right, bottom = boxes[-1]
        band = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[top:bottom, left:right]
        if text and band.size:
            # The line box covers the whole name, so the vote runs on its
            # last ink word, which is the numeral even when the text
            # dropped it
            ink = _otsu_band(band, _TITLE_SCALE) < 128
            token = trailing_token_box(ink)
            if token is not None:
                vote = vote_on_box(ink, token)
                logger.debug(f"Tier vote after {raw_text!r}: {vote}")
        return TitleRead(text, boxes, score, vote, lines)


def _title_lines(
    detected: list[tuple[TextLine, float]], geometry: list[Box]
) -> list[tuple[TextLine, float]]:
    """The detected lines that form the title, in reading order.

    A detected line is part of the title when its vertical centre falls
    inside one of the geometry title boxes. With no geometry, or no line on
    it, the tallest detected line is the title, as it is when in view.
    """
    title = [
        (line, score)
        for line, score in detected
        if any(
            top <= (line.box[1] + line.box[3]) / 2 <= bottom
            for _, top, _, bottom in geometry
        )
    ]
    if title:
        return title
    return [max(detected, key=lambda d: d[0].box[3] - d[0].box[1])]


def make_reader(engine: str) -> TitleReader:
    """Build the reader named by an OCR_ENGINE value.

    Raises ValueError for an unknown name and TitleReaderUnavailableError when
    the reader's optional package is missing.
    """
    readers: dict[str, type[Any]] = {
        TESSERACT: TesseractReader,
        TESSERACT_TITLE: TesseractTitleReader,
        PADDLE: PaddleReader,
    }
    key = engine.strip().lower()
    if key not in readers:
        msg = f"Unknown OCR_ENGINE {engine!r}, expected one of {', '.join(ENGINES)}"
        raise ValueError(msg)
    return readers[key]()


def timed_read(reader: TitleReader, card_rgb: np.ndarray) -> tuple[TitleRead, float]:
    """Run reader.read and return the read with its wall time in seconds."""
    start = time.perf_counter()
    read = reader.read(card_rgb)
    return read, time.perf_counter() - start
