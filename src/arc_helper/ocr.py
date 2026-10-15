"""
OCR module for ARLO.
Screen capture and text extraction using Tesseract.
"""

import queue
import re
import string
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher

import numpy as np
import pytesseract
from PIL import Image
from PIL import ImageOps
from pydantic import BaseModel

from .config import RegionMixin
from .config import get_screen_resolution
from .config import get_settings
from .config import logger
from .screen import Point
from .screen import ScreenCaptureUnavailable
from .screen import get_backend
from .tiers import TIER_LIKE
from .tiers import TierVote
from .tiers import count_strokes
from .tiers import tier_key
from .title_reader import PADDLE_INSTALL
from .title_reader import TESSERACT
from .title_reader import TESSERACT_TITLE
from .title_reader import TextLine
from .title_reader import TitleReader
from .title_reader import TitleReaderUnavailableError
from .title_reader import make_reader
from .tooltip_layout import Edges
from .tooltip_layout import Point as LayoutPoint
from .tooltip_layout import Rect
from .tooltip_layout import TooltipLayout
from .tooltip_layout import Word
from .tooltip_layout import find_cream_rect
from .tooltip_layout import find_label_rows
from .tooltip_layout import find_tooltip_card
from .tooltip_layout import to_screen

# Captures waiting for the writer thread before new ones are dropped
CAPTURE_QUEUE_SIZE = 4

# Margin kept around the located card when it is cropped for OCR
CARD_CROP_MARGIN = 8

# Card signature: the top band of the card (its title area) as a fraction of
# the card height, shrunk to a tiny grey thumbnail of SIGNATURE_SIZE (width,
# height). Two sightings are the same card when every rect edge is within
# SIGNATURE_RECT_TOL_PX and the thumbnails differ by at most
# SIGNATURE_THUMB_TOL grey levels on average.
SIGNATURE_BAND = 0.3
SIGNATURE_SIZE = (32, 8)
SIGNATURE_RECT_TOL_PX = 4
SIGNATURE_THUMB_TOL = 12.0


@dataclass(frozen=True, eq=False)
class CardSignature:
    """A cheap fingerprint of one sighting of the item card.

    rect is the card in physical screen px, thumb a tiny grey thumbnail of
    its top band (SIGNATURE_SIZE, uint8).
    """

    rect: Rect
    thumb: np.ndarray


def card_signature(
    image: Image.Image, card: Rect, origin: LayoutPoint
) -> CardSignature:
    """Signature of the card found at card (capture px) in a capture at origin."""
    band_bottom = card.top + max(1, round(card.height * SIGNATURE_BAND))
    band = image.crop((card.left, card.top, card.right, band_bottom))
    thumb = np.asarray(
        band.convert("L").resize(SIGNATURE_SIZE, Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    return CardSignature(to_screen(card, origin), thumb)


def same_card(a: CardSignature, b: CardSignature) -> bool:
    """True when two sightings show the same card at the same place."""
    if any(
        abs(p - q) > SIGNATURE_RECT_TOL_PX for p, q in zip(a.rect, b.rect, strict=True)
    ):
        return False
    diff = np.abs(a.thumb.astype(np.int16) - b.thumb.astype(np.int16))
    return float(diff.mean()) <= SIGNATURE_THUMB_TOL


def build_reader(engine: str) -> TitleReader:
    """Build the title reader named by OCR_ENGINE, never raising.

    An engine whose package is missing, or an unknown name, logs one
    warning and falls back to tesseract_title, second best in the bake-off.
    """
    try:
        return make_reader(engine)
    except TitleReaderUnavailableError as e:
        logger.warning(f"{e}. Falling back to {TESSERACT_TITLE}.")
    except ValueError as e:
        logger.warning(
            f"{e}. Falling back to {TESSERACT_TITLE}. "
            f"For paddle install it with: {PADDLE_INSTALL}"
        )
    return make_reader(TESSERACT_TITLE)


def clamp_bbox(
    left: int, top: int, width: int, height: int
) -> tuple[int, int, int, int]:
    """Clamp a capture box to the primary screen as (left, top, right, bottom)."""
    screen_width, screen_height = get_screen_resolution()
    right = left + width
    bottom = top + height
    left = max(0, min(left, screen_width - 1))
    top = max(0, min(top, screen_height - 1))
    right = max(left + 1, min(right, screen_width))
    bottom = max(top + 1, min(bottom, screen_height))
    return left, top, right, bottom


def search_window_bbox(
    cursor: Point, width: int, height: int
) -> tuple[int, int, int, int]:
    """The card search window centred on the cursor, clamped to the screen.

    A width or height of 0 (or less) spans the whole screen on that axis.
    """
    screen_width, screen_height = get_screen_resolution()
    if width <= 0:
        width, left = screen_width, 0
    else:
        left = cursor.x - width // 2
    if height <= 0:
        height, top = screen_height, 0
    else:
        top = cursor.y - height // 2
    return clamp_bbox(left, top, width, height)


def crop_card(rgb: Image.Image, cursor: LayoutPoint) -> tuple[Image.Image, Rect] | None:
    """Locate the item card in an RGB image and return its OCR crop and rect.

    The cursor is in image coordinates. The crop includes CARD_CROP_MARGIN
    px around the card, clamped to the image. None when there is no card.
    """
    card = find_tooltip_card(np.array(rgb), cursor)
    if card is None:
        return None
    crop = rgb.crop(
        (
            max(0, card.left - CARD_CROP_MARGIN),
            max(0, card.top - CARD_CROP_MARGIN),
            min(rgb.width, card.right + CARD_CROP_MARGIN),
            min(rgb.height, card.bottom + CARD_CROP_MARGIN),
        )
    )
    return crop, card


def grab_screen(bbox: tuple[int, int, int, int]) -> Image.Image:
    """Capture a screen region via the active screen backend."""
    return get_backend().grab(bbox)


class OCRResult(BaseModel):
    """Result of an OCR operation."""

    text: str | None
    confidence: float = 0.0
    raw_text: str = ""


def get_cursor_position() -> Point:
    """Get current cursor position on screen in physical pixels."""
    return get_backend().cursor_position()


@dataclass
class _BandRegion(RegionMixin):
    """A screen rectangle in physical pixels, for the trigger band capture."""

    x: int
    y: int
    width: int
    height: int


class OCREngine:
    """OCR engine for extracting text from screen regions."""

    # The trigger word we're looking for
    TRIGGER_WORD = "INVENTORY"

    # Band searched for the trigger word when the configured regions miss it,
    # as fractions of the screen. It spans the tab bar in the stash view and
    # in raid. A word counts when it is at least this similar to the trigger.
    TRIGGER_BAND_LEFT = 0.30
    TRIGGER_BAND_RIGHT = 0.70
    TRIGGER_BAND_BOTTOM = 0.07
    TRIGGER_BAND_RATIO = 0.8

    # What the last check_trigger_any read in each place it tried, as (place,
    # text) pairs: region1, region2 and band, in order, up to the first match
    last_trigger_reads: tuple[tuple[str, str], ...] = ()

    # Item names render as uppercase letters and Roman numerals. Restricting
    # OCR to this set stops "II" being misread as "Il"/"1l" and the trailing
    # numeral from collapsing to "I" (which silently picks the wrong tier).
    # No space: pytesseract splits the config on spaces, so a space in the
    # whitelist breaks it. Words run together ("HULLCRACKERIV"); the database
    # lookup is space-insensitive, so that still resolves correctly.
    NAME_WHITELIST = string.ascii_uppercase

    def __init__(self):
        """Initialize OCR engine."""
        settings = get_settings()

        # Set Tesseract path if configured
        if settings.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_path

        self.debug_mode = settings.debug_mode
        self.debug_dir = settings.debug_output_dir

        # Tooltip capture settings
        self.tooltip_width = settings.tooltip_capture.width
        self.tooltip_height = settings.tooltip_capture.height
        self.tooltip_offset_x = settings.tooltip_capture.offset_x
        self.tooltip_offset_y = settings.tooltip_capture.offset_y

        # Card search window, centred on the cursor
        self.search_width = settings.tooltip_search.width
        self.search_height = settings.tooltip_search.height

        # Last cursor capture and its top-left screen origin, kept so
        # layout_from_last_capture can locate the tooltip without a new grab.
        # _last_card is the item card located in that capture, if any.
        self._last_capture: Image.Image | None = None
        self._last_origin: LayoutPoint | None = None
        self._last_card: Rect | None = None
        # Every text line the last title read detected on the located card,
        # in that card's coordinates (CARD_CROP_MARGIN already taken off).
        # None when the reader does not return lines, so the layout runs its
        # own Tesseract pass.
        self._last_lines: list[TextLine] | None = None
        # Offset from the OCR crop to the located card, set by _locate_card
        self._crop_to_card = LayoutPoint(0, 0)
        # OCR crop of the card the last locate_card_at_cursor found, read by
        # read_located_card. None when that locate found no card.
        self._card_crop: Image.Image | None = None

        # Card title reader chosen by OCR_ENGINE
        self.reader = build_reader(settings.ocr_engine)

        if self.debug_mode:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        # Capture library: raw tooltip captures kept for offline evaluation
        self.capture_library = settings.capture_library
        self.capture_library_dir = settings.capture_library_dir
        self.capture_library_limit = settings.capture_library_limit
        self._capture_saved_at: dict[str, float] = {}
        # Captures queued this run, and the queue the writer thread drains so
        # a 2 MB PNG save never runs on the scan thread
        self._capture_count = 0
        self._capture_queue: queue.Queue[tuple[Image.Image, str, str]] = queue.Queue(
            maxsize=CAPTURE_QUEUE_SIZE
        )
        self._capture_writer: threading.Thread | None = None

        # Stroke-count vote on the tier token of the last name read
        self.last_tier_vote: TierVote | None = None

    @staticmethod
    def capture_region(region: RegionMixin) -> Image.Image:
        """Capture a screen region."""
        try:
            return grab_screen(region.bbox)
        except ScreenCaptureUnavailable:
            raise  # terminal: must reach the scanner's error handling, not become a black frame
        except OSError as e:
            logger.error(f"Screen grab failed for region {region.bbox}: {e}")
            # Return a dummy image to avoid crashing
            return Image.new("RGB", (region.width, region.height), color="black")

    def capture_around_cursor(self) -> tuple[Image.Image, Point]:
        """
        Capture a region around the current cursor position.
        Returns the captured image and the cursor position.

        If cursor is in the right 30% of the screen, the X offset is flipped
        to capture the tooltip on the left side instead.
        """

        cursor = get_cursor_position()

        # TODO: flipping the tooltip region on the right side of screen
        # to be implemented properly
        # # Check if cursor is in right 30% of screen
        # right_threshold = screen_width * 0.7

        # if cursor.x > right_threshold:
        #     # Flip X offset (and account for capture width)
        #     offset_x = -self.tooltip_offset_x - self.tooltip_width
        # else:
        #     offset_x = self.tooltip_offset_x
        offset_x = self.tooltip_offset_x

        # Calculate capture region around cursor
        left = cursor.x + offset_x
        top = cursor.y + self.tooltip_offset_y
        image = self._grab_clamped(
            cursor, left, top, self.tooltip_width, self.tooltip_height
        )
        return image, cursor

    def capture_search_window(self) -> tuple[Image.Image, Point]:
        """Capture the card search window centred on the cursor.

        The window is search_width by search_height physical pixels, clamped
        to the screen, where 0 means the full screen on that axis. It becomes
        the last capture with its origin. Returns the captured image and the
        cursor position.
        """
        cursor = get_cursor_position()
        left, top, right, bottom = search_window_bbox(
            cursor, self.search_width, self.search_height
        )
        # The box is already clamped, so its own size is passed on
        image = self._grab_clamped(cursor, left, top, right - left, bottom - top)
        return image, cursor

    def _grab_clamped(
        self, cursor: Point, left: int, top: int, width: int, height: int
    ) -> Image.Image:
        """Grab a box clamped to the screen and keep it as the last capture.

        Clears the last capture and located card first. When the grab fails
        a small black image is returned and no last capture is kept.
        """
        # Clamp to valid screen coordinates (handle multi-monitor)
        # Note: We clamp to primary monitor bounds for simplicity
        left, top, right, bottom = clamp_bbox(left, top, width, height)

        # Log if we had to clamp significantly
        if right - left < width // 2 or bottom - top < height // 2:
            logger.debug(
                f"Capture region clamped significantly: cursor=({cursor.x}, {cursor.y}), bbox=({left}, {top}, {right}, {bottom})"
            )

        self._last_capture = None
        self._last_origin = None
        self._last_card = None
        self._last_lines = None
        try:
            image = grab_screen((left, top, right, bottom))
        except ScreenCaptureUnavailable:
            raise  # terminal: must reach the scanner's error handling
        except OSError as e:
            logger.error(
                f"Screen grab failed at bbox ({left}, {top}, {right}, {bottom}): {e}"
            )
            # Return a dummy image to avoid crashing
            return Image.new("RGB", (100, 100), color="black")

        self._last_capture = image
        self._last_origin = LayoutPoint(left, top)
        return image

    def _locate_card(self, cursor: Point) -> Image.Image | None:
        """Find the item card in the last capture and return its OCR crop.

        The cursor is converted to capture coordinates for
        find_tooltip_card. On success the card is kept as _last_card and
        the crop includes CARD_CROP_MARGIN px around it, clamped to the
        capture. None when there is no capture or no card.
        """
        image, origin = self._last_capture, self._last_origin
        if image is None or origin is None:
            return None
        found = crop_card(
            image.convert("RGB"),
            LayoutPoint(cursor.x - origin.x, cursor.y - origin.y),
        )
        if found is None:
            return None
        crop, card = found
        self._last_card = card
        # The crop starts CARD_CROP_MARGIN px before the card, less where the
        # capture edge clamped it
        self._crop_to_card = LayoutPoint(
            max(0, card.left - CARD_CROP_MARGIN) - card.left,
            max(0, card.top - CARD_CROP_MARGIN) - card.top,
        )
        return crop

    def layout_from_last_capture(self) -> TooltipLayout | None:
        """Locate the tooltip and its label rows in the last cursor capture.

        When the last name read located the item card, that card is the
        cream region and its clipped edges are the ones touching the search
        window edge. Otherwise find_cream_rect locates it in the capture.
        The label rows come from the lines the title reader already detected
        on the card when it returned them (paddle), otherwise from a
        word-box Tesseract pass on the cream region upscaled 2x, which costs
        one extra pass, so callers run it once per shown item, not every
        scan. Returns the tooltip and label rectangles in physical screen
        pixels, or None when there is no capture or no cream region.
        """
        start = time.perf_counter()
        layout = self._layout_from_last_capture()
        logger.debug(
            f"Layout done in {1000 * (time.perf_counter() - start):.0f} ms, "
            f"labels {sorted(layout.labels) if layout else None}"
        )
        return layout

    def _layout_from_last_capture(self) -> TooltipLayout | None:
        """layout_from_last_capture without the timing log."""
        image, origin = self._last_capture, self._last_origin
        if image is None or origin is None:
            return None

        if self._last_card is not None:
            cream = self._last_card
            clipped = Edges(
                left=cream.left == 0,
                top=cream.top == 0,
                right=cream.right == image.width,
                bottom=cream.bottom == image.height,
            )
        else:
            region = find_cream_rect(np.array(image.convert("RGB")))
            if region is None:
                return None
            cream, clipped = region

        crop_origin = LayoutPoint(origin.x + cream.left, origin.y + cream.top)
        if self._last_card is not None and self._last_lines is not None:
            # One Word per detected line, already in card coordinates
            words = [
                Word(line.text, left, top, right - left, bottom - top, i)
                for i, line in enumerate(self._last_lines)
                for left, top, right, bottom in [line.box]
            ]
            labels = {
                name: to_screen(rect, crop_origin)
                for name, rect in find_label_rows(words, 1).items()
            }
            return TooltipLayout(
                tooltip=to_screen(cream, origin), labels=labels, clipped=clipped
            )

        scale = 2
        crop = image.crop((cream.left, cream.top, cream.right, cream.bottom))
        crop = crop.convert("L").resize(
            (crop.width * scale, crop.height * scale), Image.Resampling.BILINEAR
        )

        try:
            data = pytesseract.image_to_data(
                crop, config="--psm 6", output_type=pytesseract.Output.DICT
            )
        except pytesseract.TesseractError as e:
            logger.error(f"OCR Error: {e}")
            data = {"text": []}

        # Tesseract numbers lines per block and paragraph, so the three
        # together identify one line
        line_ids: dict[tuple[int, int, int], int] = {}
        words = [
            Word(
                text=data["text"][i],
                left=data["left"][i],
                top=data["top"][i],
                width=data["width"][i],
                height=data["height"][i],
                line=line_ids.setdefault(
                    (data["block_num"][i], data["par_num"][i], data["line_num"][i]),
                    len(line_ids),
                ),
            )
            for i in range(len(data["text"]))
            if data["text"][i].strip()
        ]

        labels = {
            name: to_screen(rect, crop_origin)
            for name, rect in find_label_rows(words, scale).items()
        }
        return TooltipLayout(
            tooltip=to_screen(cream, origin), labels=labels, clipped=clipped
        )

    def check_trigger_any(self, regions: list[RegionMixin]) -> bool:
        """
        Check if the trigger word (INVENTORY) is visible in any of the regions.

        The configured regions are tried first. Only when they all fail is the
        wide band across the top centre searched, which finds the word when the
        tab bar puts it elsewhere (the in-raid bar has an extra MAP tab).

        What each place read is kept in last_trigger_reads, as (place, text)
        pairs with an empty text for a blank read, so a miss can be logged.
        """
        self.last_trigger_reads = ()
        for number, region in enumerate(regions, start=1):
            found, text = self._read_trigger(region)
            self.last_trigger_reads += ((f"region{number}", text),)
            if found:
                return True
        found, words = self._read_trigger_band()
        self.last_trigger_reads += (("band", " ".join(words)),)
        return found

    def check_trigger_band(self) -> bool:
        """Search a band across the top centre of the screen for INVENTORY."""
        return self._read_trigger_band()[0]

    def _read_trigger_band(self) -> tuple[bool, list[str]]:
        """Search the top centre band, returning the match and the words read."""
        width, height = get_screen_resolution()
        left = int(width * self.TRIGGER_BAND_LEFT)
        right = int(width * self.TRIGGER_BAND_RIGHT)
        bottom = int(height * self.TRIGGER_BAND_BOTTOM)
        if right <= left or bottom <= 0:
            return False, []
        band = _BandRegion(left, 0, right - left, bottom)
        words = self.band_words(self.capture_region(band))
        return self._band_matches(words), words

    @classmethod
    def band_has_trigger(cls, image: Image.Image) -> bool:
        """
        Read the words of a band image and look for the trigger word in them.

        The band is read as a block of text at half size (the tab text is
        large, so this keeps Tesseract fast), and each word is matched on its
        own, so the neighbouring tabs do not hide it.
        """
        return cls._band_matches(cls.band_words(image))

    @classmethod
    def _band_matches(cls, words: list[str]) -> bool:
        """True when one of the band words is close to the trigger word."""
        return any(
            len(word) >= len(cls.TRIGGER_WORD) - 2
            and SequenceMatcher(None, word, cls.TRIGGER_WORD).ratio()
            >= cls.TRIGGER_BAND_RATIO
            for word in words
        )

    @staticmethod
    def band_words(image: Image.Image) -> list[str]:
        """The upper case words Tesseract reads in a band image."""
        gray = ImageOps.invert(image.convert("L"))
        gray = gray.resize(
            (max(1, gray.width // 2), max(1, gray.height // 2)),
            Image.Resampling.BILINEAR,
        )
        gray = gray.point(lambda x: 255 if x > 128 else 0, "1")
        try:
            text = pytesseract.image_to_string(gray, config="--psm 6 --oem 1")
        except pytesseract.TesseractError:
            return []
        return re.findall(r"[A-Z]+", text.upper())

    @staticmethod
    def preprocess_for_ocr(
        image: Image.Image, *, invert: bool = True, scale: int = 2
    ) -> Image.Image:
        """
        Preprocess image for better OCR accuracy.

        Args:
            image: Input image
            invert: Whether to invert colors (for white text on dark bg)
            scale: Scale factor for upscaling
        """
        # Convert to grayscale
        gray = image.convert("L")

        # Upscale for better OCR
        if scale > 1:
            new_size = (gray.width * scale, gray.height * scale)
            gray = gray.resize(new_size, Image.Resampling.BILINEAR)

        # Invert if needed (OCR prefers black text on white)
        if invert:
            gray = ImageOps.invert(gray)

        # Apply threshold for cleaner text
        threshold = 128
        gray = gray.point(lambda x: 255 if x > threshold else 0, "1")

        return gray

    def preprocess_tooltip(self, image: Image.Image) -> Image.Image:
        """
        Preprocess tooltip image by isolating the cream-colored tooltip background
        and extracting dark text from it.
        """
        img_array = np.array(image)

        # Tooltip background is #f9eedf = RGB(249, 238, 223)
        lower_bound = np.array([240, 225, 210])
        upper_bound = np.array([255, 250, 240])

        # Create mask for tooltip background
        mask = np.all((img_array >= lower_bound) & (img_array <= upper_bound), axis=2)

        # Find rows and columns that are predominantly cream (>50%)
        row_cream_percentage = np.sum(mask, axis=1) / mask.shape[1]
        col_cream_percentage = np.sum(mask, axis=0) / mask.shape[0]

        # Find first/last row with significant cream
        cream_rows = np.where(row_cream_percentage > 0.5)[0]
        cream_cols = np.where(col_cream_percentage > 0.3)[0]

        if len(cream_rows) == 0 or len(cream_cols) == 0:
            result = Image.new("L", (image.width * 2, image.height * 2), 255)
            return result

        y_min, y_max = cream_rows[0], cream_rows[-1]
        x_min, x_max = cream_cols[0], cream_cols[-1]

        # First crop - to the cream region
        cropped = img_array[y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]

        # Find columns that are mostly cream (guard against empty crop)
        n_rows = cropped_mask.shape[0]
        if n_rows == 0:
            result = Image.new("L", (image.width * 2, image.height * 2), 255)
            return result
        col_cream_pct = np.sum(cropped_mask, axis=0) / n_rows
        tight_cols = np.where(col_cream_pct > 0.5)[0]

        if len(tight_cols) == 0:
            result = Image.new("L", (image.width * 2, image.height * 2), 255)
            return result

        tight_x_min, tight_x_max = tight_cols[0], tight_cols[-1]

        # Tight crop - removes the colored borders
        tight_cropped = cropped[:, tight_x_min:tight_x_max]

        # Detect "colored" pixels - specifically looking for saturated colors (tags)
        # Tags are colorful (blue, green, etc.) - they have high saturation
        # Text is nearly black, cream is desaturated
        r, g, b = tight_cropped[:, :, 0], tight_cropped[:, :, 1], tight_cropped[:, :, 2]

        # Calculate a simple "colorfulness" metric
        # Colored pixels have larger differences between RGB channels
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        color_diff = max_rgb - min_rgb

        # Colored pixels: significant difference between channels AND not too dark AND not too bright
        is_colored = (color_diff > 30) & (max_rgb > 80) & (max_rgb < 240)

        # Count colored pixels per row - only exclude rows with SIGNIFICANT color (>5% of row)
        row_color_percentage = np.sum(is_colored, axis=1) / is_colored.shape[1]
        row_has_significant_color = row_color_percentage > 0.05

        # Create output - white background
        result = (
            np.ones((tight_cropped.shape[0], tight_cropped.shape[1]), dtype=np.uint8)
            * 255
        )

        # Only include dark text from rows WITHOUT significant colored pixels
        is_dark = np.all(tight_cropped < 100, axis=2)

        # Vectorized masking: rows without significant color AND dark pixels
        mask = (~row_has_significant_color[:, np.newaxis]) & is_dark
        result[mask] = 0

        # Convert to PIL
        result_image = Image.fromarray(result, mode="L")

        # Upscale for better OCR
        scale = 2
        new_size = (result_image.width * scale, result_image.height * scale)
        result_image = result_image.resize(new_size, Image.Resampling.BILINEAR)

        # Debug: save intermediate images
        if self.debug_mode:
            Image.fromarray(mask.astype(np.uint8) * 255).save(
                self.debug_dir / "tooltip_mask.png"
            )
            Image.fromarray(cropped).save(self.debug_dir / "tooltip_cropped.png")
            Image.fromarray(tight_cropped).save(
                self.debug_dir / "tooltip_tight_cropped.png"
            )
            Image.fromarray(is_colored.astype(np.uint8) * 255).save(
                self.debug_dir / "tooltip_colored.png"
            )
            Image.fromarray(
                row_has_significant_color[:, np.newaxis]
                .repeat(tight_cropped.shape[1], axis=1)
                .astype(np.uint8)
                * 255
            ).save(self.debug_dir / "tooltip_colored_rows.png")
            Image.fromarray(result).save(self.debug_dir / "tooltip_text_mask.png")

        return result_image

    def extract_text(
        self,
        image: Image.Image,
        *,
        single_line: bool = True,
        whitelist: str | None = None,
    ) -> OCRResult:
        """
        Extract text from an image.

        Args:
            image: Preprocessed image
            single_line: Whether to expect a single line of text
            whitelist: Character whitelist for OCR
        """
        # Build Tesseract config
        config_parts = []

        # Page segmentation mode
        if single_line:
            config_parts.append("--psm 7")  # Single line
        else:
            config_parts.append("--psm 6")  # Block of text

        # Character whitelist
        if whitelist:
            config_parts.append(f'-c tessedit_char_whitelist="{whitelist}"')

        config = " ".join(config_parts)

        try:
            # Get text with confidence data
            data = pytesseract.image_to_data(
                image, config=config, output_type=pytesseract.Output.DICT
            )

            # Extract text and calculate average confidence
            texts = []
            confidences = []

            for i, text in enumerate(data["text"]):
                conf = data["conf"][i]
                if text.strip() and conf > 0:
                    texts.append(text.strip())
                    confidences.append(conf)

            raw_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            # Clean up text
            cleaned = self._clean_text(raw_text)

            return OCRResult(
                text=cleaned or None,
                confidence=avg_confidence,
                raw_text=raw_text,
            )

        except pytesseract.TesseractError as e:
            if self.debug_mode:
                logger.error(f"OCR Error: {e}")
            return OCRResult(text=None, confidence=0, raw_text="")

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean OCR output text."""
        # Remove common OCR artifacts
        text = re.sub(r"[|\\/_]", "", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        # Strip
        return text.strip()

    def check_trigger(self, region: RegionMixin) -> bool:
        """
        Check if the trigger word (INVENTORY) is visible in the region.

        This is optimized for speed - we just need to detect the word,
        not extract it perfectly.
        """
        return self._read_trigger(region)[0]

    def _read_trigger(self, region: RegionMixin) -> tuple[bool, str]:
        """Read one trigger region, returning the match and the text read."""
        # Capture
        image = self.capture_region(region)

        # Preprocess
        processed = self.preprocess_for_ocr(image, invert=True, scale=2)

        # Save debug image if enabled
        if self.debug_mode:
            image.save(self.debug_dir / "trigger_raw.png")
            processed.save(self.debug_dir / "trigger_processed.png")

        # Extract with limited whitelist for speed using image_to_string (faster than image_to_data)
        config = f'--psm 7 -c tessedit_char_whitelist="{string.ascii_uppercase}"'
        try:
            text = pytesseract.image_to_string(processed, config=config)
        except pytesseract.TesseractError:
            return False, ""
        read = text.strip().upper()
        # Fuzzy match - allow for some OCR errors
        return self._fuzzy_match(read, self.TRIGGER_WORD), read

    def extract_item_name_at_cursor(self) -> str | None:
        """
        Extract item name from tooltip near the cursor.

        Captures the search window around the cursor and locates the item
        card in it, so a card drawn left of, below or further from the cursor
        is still read. The name passes then run on the card crop only. When
        no card is found it falls back to the cursor-relative capture box.
        Returns the cleaned item name or None if extraction fails.
        """
        self.last_tier_vote = None

        window, cursor = self.capture_search_window()
        card_crop = self._locate_card(cursor)
        if card_crop is not None:
            image = card_crop
            # The whole window goes to the library, so evaluation sees context
            library_image = window
        else:
            image, _ = self.capture_around_cursor()
            library_image = image
        return self._read_and_keep(
            image, library_image, card_found=card_crop is not None
        )

    def locate_card_at_cursor(self) -> CardSignature | None:
        """Capture the search window and locate the item card, without reading.

        Returns the card's signature, or None when no card is found. The
        capture, its origin and the card stay as the last capture for
        read_located_card and layout_from_last_capture.
        """
        self.last_tier_vote = None
        window, cursor = self.capture_search_window()
        self._card_crop = self._locate_card(cursor)
        if self._card_crop is None or self._last_card is None:
            return None
        return card_signature(window, self._last_card, self._last_origin)

    def read_located_card(self) -> str | None:
        """Read the title of the card the last locate_card_at_cursor found.

        Returns the cleaned item name, or None without a located card or a
        read. The search window goes to the capture library as for
        extract_item_name_at_cursor.
        """
        if self._card_crop is None or self._last_capture is None:
            return None
        return self._read_and_keep(self._card_crop, self._last_capture, card_found=True)

    def _read_and_keep(
        self, image: Image.Image, library_image: Image.Image, *, card_found: bool
    ) -> str | None:
        """Read the title from image, log its time and keep the capture."""
        # Save debug image if enabled
        if self.debug_mode:
            image.save(self.debug_dir / "tooltip_capture_raw.png")

        # The tooltip has a cream/beige background with dark text
        # We need to find the item name which is the large bold text
        # It appears after "ACTIONS" header and category tags

        # Strategy: Look for the item name area
        # The item name is typically in the upper portion of the tooltip
        # in large dark text on the cream background

        # Extract from the card crop (or the fallback capture) using
        # preprocessing for dark text on light background

        start = time.perf_counter()
        try:
            item_name = self._read_title(image, card_found=card_found)
        except pytesseract.TesseractError as e:
            logger.error(f"OCR Error: {e}")
            item_name = None
        else:
            if item_name:
                logger.debug(f"Extracted item name: '{item_name}'")
        logger.debug(
            f"Title read by {self.reader.name} done in "
            f"{1000 * (time.perf_counter() - start):.0f} ms"
        )

        if self.capture_library and self._last_capture is not None:
            self._save_to_library(library_image, item_name)
        return item_name

    def _save_to_library(self, image: Image.Image, item_name: str | None) -> None:
        """Queue a raw capture for the writer thread, never blocking or raising.

        At most one file per distinct read name every 2 seconds, so holding
        the cursor on one item does not fill the folder, and at most
        capture_library_limit files in one run. A full queue drops the
        capture.
        """
        if self._capture_count >= self.capture_library_limit:
            return
        label = re.sub(r"[^A-Za-z0-9]+", "-", item_name or "").strip("-") or "none"
        now = time.monotonic()
        last = self._capture_saved_at.get(label)
        if last is not None and now - last < 2.0:
            return
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        self._start_capture_writer()
        try:
            self._capture_queue.put_nowait((image, stamp, label))
        except queue.Full:
            logger.debug(f"Capture writer busy, dropped capture '{label}'")
            return
        self._capture_saved_at[label] = now
        self._capture_count += 1
        if self._capture_count >= self.capture_library_limit:
            logger.info(
                f"Capture library limit of {self.capture_library_limit} reached, "
                "no more captures this run"
            )

    def _start_capture_writer(self) -> None:
        """Start the capture writer thread on first use."""
        if self._capture_writer is None:
            self._capture_writer = threading.Thread(
                target=self._capture_writer_loop, name="capture-writer", daemon=True
            )
            self._capture_writer.start()

    def _capture_writer_loop(self) -> None:
        """Save queued captures one at a time, for the life of the process."""
        while True:
            image, stamp, label = self._capture_queue.get()
            try:
                self._write_capture(image, stamp, label)
            finally:
                self._capture_queue.task_done()

    def _write_capture(self, image: Image.Image, stamp: str, label: str) -> None:
        """Save one capture as <stamp>_<label>.png, never raising."""
        try:
            self.capture_library_dir.mkdir(parents=True, exist_ok=True)
            image.save(self.capture_library_dir / f"{stamp}_{label}.png")
        except Exception as e:  # noqa: BLE001 - a full disk must not stop scanning
            logger.warning(f"Could not save capture to library: {e}")

    def _read_title(self, image: Image.Image, *, card_found: bool) -> str | None:
        """Read the item name from a card crop with the configured reader.

        The tesseract engine keeps today's path, preprocess_tooltip and the
        two passes of _read_item_name, with its debug images. Other readers
        get the RGB crop of a located card only. Without a card they read
        nothing, since a full-screen read costs paddle about a second and
        reads game text such as LOADOUT as an item. When the reader returned
        every detected line, those lines are kept in card coordinates for
        layout_from_last_capture.
        """
        self.last_tier_vote = None
        self._last_lines = None
        if self.reader.name == TESSERACT:
            processed = self.preprocess_tooltip(image)
            if self.debug_mode:
                processed.save(self.debug_dir / "tooltip_capture_processed.png")
            return self._read_item_name(processed)
        if not card_found:
            return None

        read = self.reader.read(np.array(image.convert("RGB")))
        self.last_tier_vote = read.tier_vote
        if read.lines is not None:
            dx, dy = self._crop_to_card
            self._last_lines = [
                TextLine(line.text, (left + dx, top + dy, right + dx, bottom + dy))
                for line in read.lines
                for left, top, right, bottom in [line.box]
            ]
        return read.text

    def _read_item_name(self, processed: Image.Image) -> str | None:
        """Read the item name from a preprocessed tooltip via two OCR passes.

        The plain pass locates the name block (the leading mostly-uppercase
        lines, set off from the lower-case description by case). A second
        uppercase-whitelisted pass supplies the text with Roman numerals
        intact - a plain pass reads "II" as "Il" or drops it to "I", which
        silently selects the wrong item tier.
        """
        self.last_tier_vote = None
        plain = pytesseract.image_to_string(processed, config="--psm 6")
        logger.debug(f"Raw tooltip OCR:\n{plain}")

        line_count = self._count_name_lines(plain)
        if line_count == 0:
            return None

        whitelisted = pytesseract.image_to_string(
            processed,
            config=f"--psm 6 -c tessedit_char_whitelist={self.NAME_WHITELIST}",
        )
        name_lines = [ln.strip() for ln in whitelisted.split("\n") if ln.strip()]

        # The word boxes cost a third Tesseract pass, so only fetch them
        # when the plain read already ends the name in a tier-like token
        plain_lines = [ln.split() for ln in plain.split("\n")]
        if self._last_name_word(plain_lines, line_count) is not None:
            self.last_tier_vote = self._vote_on_tier(processed, line_count)

        return self._clean_text(" ".join(name_lines[:line_count])) or None

    @classmethod
    def _last_name_word(cls, lines: list[list[str]], line_count: int) -> int | None:
        """Index of the tier-like last word of the name, or None.

        lines are the words of each OCR line in order. The name lines are
        found the way _count_name_lines finds them: short leading lines are
        skipped, then line_count lines form the name.
        """
        start = 0
        while start < len(lines) and len(cls._clean_text(" ".join(lines[start]))) < 2:
            start += 1
        last = start + line_count - 1
        if last >= len(lines) or not lines[last]:
            return None
        if TIER_LIKE.match(tier_key(lines[last][-1])):
            return last
        return None

    def _vote_on_tier(self, processed: Image.Image, line_count: int) -> TierVote | None:
        """Count the strokes of the tier token that ends the name.

        Reads word boxes from the plain pass, crops the last word of the
        last name line from the processed image (ink is black) and counts
        its strokes. None when the token cannot be located.
        """
        try:
            data = pytesseract.image_to_data(
                processed, config="--psm 6", output_type=pytesseract.Output.DICT
            )
        except pytesseract.TesseractError as e:
            logger.debug(f"Tier vote skipped, word boxes failed: {e}")
            return None

        # Tesseract numbers lines per block and paragraph
        lines: dict[tuple[int, int, int], list[int]] = {}
        for i, text in enumerate(data["text"]):
            if text.strip():
                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                lines.setdefault(key, []).append(i)
        boxes = list(lines.values())
        words = [[data["text"][i] for i in line] for line in boxes]

        last = self._last_name_word(words, line_count)
        if last is None:
            return None
        i = boxes[last][-1]
        left, top = data["left"][i], data["top"][i]
        box = (left, top, left + data["width"][i], top + data["height"][i])
        ink = np.array(processed.convert("L").crop(box)) < 128
        vote = count_strokes(ink)
        logger.debug(f"Tier vote on {data['text'][i]!r}: {vote}")
        return vote

    @staticmethod
    def _upper_ratio(text: str) -> float:
        """Fraction of alphabetic characters that are uppercase."""
        alpha = [c for c in text if c.isalpha()]
        return sum(c.isupper() for c in alpha) / len(alpha) if alpha else 0.0

    @classmethod
    def _count_name_lines(cls, text: str) -> int:
        """Count the leading mostly-uppercase lines that form the item name.

        Stops at the first lower-case line (the description), tolerating a
        stray mis-cased character (e.g. "II" read as "Il") via a ratio.
        """
        count = 0
        started = False
        for raw_line in text.split("\n"):
            line = cls._clean_text(raw_line)
            if len(line) >= 2 and cls._upper_ratio(line) >= 0.6:
                count += 1
                started = True
            elif started or len(line) >= 2:
                break
        return count

    def parse_item_name_from_tooltip(self, text: str) -> str | None:
        """
        Parse item name from tooltip OCR text.

        Item names are in ALL CAPS (like "ADVANCED ELECTRICAL COMPONENTS")
        and may span multiple lines.
        """
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        if not lines:
            return None

        # Find consecutive ALL CAPS lines (the item name)
        name_lines = []
        found_name = False

        for line in lines:
            # Clean the line
            cleaned = self._clean_text(line)
            if not cleaned or len(cleaned) < 2:
                continue

            # Check if line is ALL CAPS (item name style)
            # Allow spaces, numbers, and punctuation
            alpha_chars = [c for c in cleaned if c.isalpha()]
            if not alpha_chars:
                continue

            is_all_caps = all(c.isupper() for c in alpha_chars)

            if is_all_caps and len(cleaned) >= 3:
                # This looks like an item name line
                name_lines.append(cleaned)
                found_name = True
            elif found_name:
                # We were collecting name lines but hit a non-caps line
                # This means the name is complete
                break

        if name_lines:
            # Join multi-line names with a space
            return " ".join(name_lines)

        return None

    def extract_item_name(self, region: RegionMixin) -> str | None:
        """
        Legacy method - extract item name from a fixed region.
        Kept for calibration tool compatibility.
        """
        image = self.capture_region(region)
        processed = self.preprocess_tooltip(image)

        if self.debug_mode:
            image.save(self.debug_dir / "tooltip_raw.png")
            processed.save(self.debug_dir / "tooltip_processed.png")

        try:
            text = pytesseract.image_to_string(processed, config="--psm 6")
            return self.parse_item_name_from_tooltip(text)
        except pytesseract.TesseractError as e:
            logger.error(f"OCR Error: {e}")
            return None

    @staticmethod
    def _fuzzy_match(text: str, target: str, threshold: float = 0.7) -> bool:
        """
        Check if text approximately matches target.
        Uses simple character overlap ratio.
        """
        if not text:
            return False

        # Check if target appears as substring
        if target in text:
            return True

        # Check character overlap
        text_chars = set(text)
        target_chars = set(target)
        overlap = len(text_chars & target_chars) / len(target_chars)

        return overlap >= threshold


class OCREngineManager:
    """Manages the singleton OCREngine instance."""

    _instance: OCREngine | None = None

    @classmethod
    def get(cls) -> OCREngine:
        """Get the OCR engine instance, creating it if needed."""
        if cls._instance is None:
            cls._instance = OCREngine()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (mainly for testing or reloading settings)."""
        cls._instance = None


def get_ocr_engine() -> OCREngine:
    """Get the singleton OCR engine instance."""
    return OCREngineManager.get()
