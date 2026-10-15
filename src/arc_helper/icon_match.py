"""
Perceptual-hash baseline for item icon identity.

A hovered inventory slot is cropped to the part that shows the item art,
reduced to a flat-background silhouette-sized picture, hashed, and compared by
Hamming distance with the hashes of the arctracker catalog icons. This is a
baseline, measured first. It is expected to confuse lookalikes (tiers of one
weapon, recolored parts) and so to justify an embedding model later.

Normalisation of a slot crop, in order:
1. Keep the art area: inset the border ring, drop the bottom info bar (the
   stack count, durability and tier mark). A wide weapon slot also drops its
   attachment row, and its art is turned WIDE_TILT degrees clockwise: a wide
   slot draws the weapon level, while the catalog icon (and a square slot)
   draws it on a diagonal from upper left to lower right.
2. Paint the rarity wedge (a triangle in the lower left) with the slot's own
   background colour.
3. Estimate the background from the top strip, mask every pixel that differs
   from it, and cut the picture to the mask's bounding box, so the hash does
   not depend on how large the icon is drawn.
4. Pad the box to a square on a flat background and resize.

A catalog icon (256 px RGBA) is flattened onto a dark background and goes
through the same steps 3 and 4.
"""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image

HASH_SIDE = 8
NORM_SIDE = 64
TOP_N = 5
WIDE_ASPECT = 1.3
MASK_DIFF = 30
INSET = 0.09
# A wide slot shows the weapon level, the catalog icon tilts it down to the
# right by about this many degrees (measured by eye on the catalog weapons)
WIDE_TILT = 40.0
ART_MARGIN = 0.06
MIN_MASK_PIXELS = 2
GALLERY_BACKGROUND = (22, 30, 38)
GREY_WEIGHTS = np.array([0.299, 0.587, 0.114])
METHODS = ("dhash", "phash")


class Candidate(NamedTuple):
    """One ranked gallery entry: its item id, display name and Hamming distance."""

    item_id: str
    name: str
    distance: int


@dataclass(frozen=True)
class Gallery:
    """The catalog icon hashes, one row of HASH_SIDE squared bits per item."""

    item_ids: tuple[str, ...]
    names: tuple[str, ...]
    hashes: np.ndarray
    method: str

    def __len__(self) -> int:
        return len(self.item_ids)


def art_region(crop: np.ndarray) -> np.ndarray:
    """Return the item art part of a slot crop, with the rarity wedge painted out.

    crop is an RGB array of the slot inside its glow ring. A crop wider than
    WIDE_ASPECT times its height is a weapon slot with a row of attachments
    under the art, and gets a shorter area.
    """
    height, width = crop.shape[:2]
    wide = is_wide(crop)
    left, right = round(width * INSET), round(width * (1 - INSET))
    top = round(height * INSET)
    bottom = round(height * (0.50 if wide else 0.72))
    region = crop[top:bottom, left:right, :3].copy()
    if region.size == 0:
        return region

    background = _background(region)
    if not wide:
        # The wedge in full-crop coordinates: (0, 0.45 H), (0, 0.72 H) and
        # (0.40 W, 0.72 H), shifted into the region.
        wedge = np.array(
            [
                [0 - left, round(height * 0.45) - top],
                [0 - left, round(height * 0.72) - top],
                [round(width * 0.40) - left, round(height * 0.72) - top],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(region, [wedge], tuple(int(c) for c in background))
    return region


def is_wide(crop: np.ndarray) -> bool:
    """True for a weapon slot drawn wider than WIDE_ASPECT times its height."""
    height, width = crop.shape[:2]
    return width / height > WIDE_ASPECT


def art_picture(crop: np.ndarray) -> np.ndarray:
    """Return the art region of a slot crop, turned to the catalog pose.

    This is the crop step shared by the hash and the embedding model. A square
    slot gives art_region() unchanged. A wide slot's region is turned
    WIDE_TILT degrees clockwise on a canvas large enough to keep every corner,
    with the new corners filled by the slot's backdrop colour.
    """
    region = art_region(crop)
    if region.size == 0 or not is_wide(crop):
        return region
    background = tuple(int(c) for c in _background(region))
    height, width = region.shape[:2]
    turn = cv2.getRotationMatrix2D((width / 2, height / 2), -WIDE_TILT, 1.0)
    cos, sin = abs(turn[0, 0]), abs(turn[0, 1])
    new_w = int(np.ceil(width * cos + height * sin))
    new_h = int(np.ceil(width * sin + height * cos))
    turn[0, 2] += (new_w - width) / 2
    turn[1, 2] += (new_h - height) / 2
    return cv2.warpAffine(
        region,
        turn,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background,
    )


def _background(region: np.ndarray) -> np.ndarray:
    """Median colour of the top strip of a region, the slot's dark backdrop."""
    strip = region[: max(2, region.shape[0] // 12)].reshape(-1, 3)
    return np.median(strip, axis=0)


def _art_mask(region: np.ndarray) -> tuple[np.ndarray, np.ndarray, slice, slice]:
    """Return (background colour, art mask, row slice, column slice) of a region.

    The art is the pixels that differ from the background. The slices are its
    bounding box, empty when the region is nothing but backdrop.
    """
    background = _background(region)
    diff = np.abs(region.astype(np.int16) - background.astype(np.int16)).max(axis=2)
    mask = diff > MASK_DIFF
    mask = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    ).astype(bool)
    rows = np.flatnonzero(mask.sum(axis=1) >= MIN_MASK_PIXELS)
    cols = np.flatnonzero(mask.sum(axis=0) >= MIN_MASK_PIXELS)
    if rows.size == 0 or cols.size == 0:
        return background, mask, slice(0, 0), slice(0, 0)
    return background, mask, slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def _pad_square(box: np.ndarray, fill: object, margin: float = 0.0) -> np.ndarray:
    """Centre box on a square of fill, margin times its long side on each edge."""
    long_side = max(box.shape[:2])
    side = long_side + 2 * round(long_side * margin)
    square = np.empty((side, side, *box.shape[2:]), dtype=box.dtype)
    square[:] = fill
    y0 = (side - box.shape[0]) // 2
    x0 = (side - box.shape[1]) // 2
    square[y0 : y0 + box.shape[0], x0 : x0 + box.shape[1]] = box
    return square


def normalize(region: np.ndarray) -> np.ndarray:
    """Return a NORM_SIDE square grey picture of the art in an RGB region.

    The art is found as the pixels that differ from the background, cut to
    its bounding box and padded to a square on the background grey. An empty
    mask (nothing but backdrop) gives a flat picture.
    """
    background, mask, rows, cols = _art_mask(region)
    grey = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    flat = round(float(background @ GREY_WEIGHTS))
    box = grey[rows, cols]
    if box.size == 0:
        return np.full((NORM_SIDE, NORM_SIDE), flat, dtype=np.uint8)
    box = np.where(mask[rows, cols], box, flat).astype(np.uint8)
    square = _pad_square(box, flat)
    return cv2.resize(square, (NORM_SIDE, NORM_SIDE), interpolation=cv2.INTER_AREA)


def art_square(region: np.ndarray, side: int) -> np.ndarray:
    """Return a side by side RGB picture of the art in a region, for a model.

    The art box is found as in normalize(), the backdrop around and inside it
    is painted GALLERY_BACKGROUND (so slot and catalog backdrops agree), and the
    box is centred on a square with an ART_MARGIN border.
    """
    _, mask, rows, cols = _art_mask(region)
    box = region[rows, cols]
    if box.size == 0:
        return np.full((side, side, 3), GALLERY_BACKGROUND, dtype=np.uint8)
    box = np.where(mask[rows, cols, None], box, np.array(GALLERY_BACKGROUND))
    square = _pad_square(box.astype(np.uint8), GALLERY_BACKGROUND, ART_MARGIN)
    return cv2.resize(square, (side, side), interpolation=cv2.INTER_AREA)


def dhash(grey: np.ndarray) -> np.ndarray:
    """Difference hash: HASH_SIDE squared bits of neighbouring-pixel order."""
    small = cv2.resize(
        grey, (HASH_SIDE + 1, HASH_SIDE), interpolation=cv2.INTER_AREA
    ).astype(np.int16)
    return (small[:, 1:] > small[:, :-1]).ravel()


def phash(grey: np.ndarray) -> np.ndarray:
    """Perceptual hash: the low DCT frequencies of a 4x size picture against their median."""
    side = HASH_SIDE * 4
    small = cv2.resize(grey, (side, side), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(small.astype(np.float32))[:HASH_SIDE, :HASH_SIDE]
    return (dct > np.median(dct)).ravel()


def _hash(grey: np.ndarray, method: str) -> np.ndarray:
    if method == "dhash":
        return dhash(grey)
    if method == "phash":
        return phash(grey)
    msg = f"unknown hash method {method!r}, use one of {METHODS}"
    raise ValueError(msg)


def crop_hash(crop: np.ndarray, method: str = "phash") -> np.ndarray:
    """Hash of a slot crop (RGB array) after normalisation."""
    return _hash(normalize(art_picture(crop)), method)


def icon_hash(icon: Image.Image, method: str = "phash") -> np.ndarray:
    """Hash of a catalog icon, flattened onto the gallery background."""
    return _hash(normalize(icon_region(icon)), method)


def icon_region(icon: Image.Image) -> np.ndarray:
    """A catalog icon flattened onto GALLERY_BACKGROUND, as an RGB region."""
    rgba = icon.convert("RGBA")
    backdrop = Image.new("RGBA", rgba.size, (*GALLERY_BACKGROUND, 255))
    flat = Image.alpha_composite(backdrop, rgba).convert("RGB")
    return _pad_top(np.asarray(flat))


def _pad_top(rgb: np.ndarray) -> np.ndarray:
    """Add a backdrop strip on top so the background estimate never hits the art."""
    pad = np.empty((max(2, rgb.shape[0] // 8), rgb.shape[1], 3), dtype=np.uint8)
    pad[:] = GALLERY_BACKGROUND
    return np.vstack([pad, rgb])


def build_gallery(
    icons: Mapping[str, tuple[str, Path]], method: str = "phash"
) -> Gallery:
    """Hash every cached icon. icons maps item id to (display name, PNG path)."""
    ids: list[str] = []
    names: list[str] = []
    rows: list[np.ndarray] = []
    for item_id, (name, path) in icons.items():
        with Image.open(path) as image:
            rows.append(icon_hash(image, method))
        ids.append(item_id)
        names.append(name)
    hashes = np.array(rows, dtype=bool).reshape(len(rows), HASH_SIDE * HASH_SIDE)
    return Gallery(tuple(ids), tuple(names), hashes, method)


def match(crop: np.ndarray, gallery: Gallery, top: int = TOP_N) -> list[Candidate]:
    """Return the top candidates for a slot crop, nearest first.

    crop is an RGB array of the slot inside its glow ring. Ties keep gallery
    order.
    """
    if len(gallery) == 0:
        return []
    bits = crop_hash(crop, gallery.method)
    distances = np.count_nonzero(gallery.hashes != bits, axis=1)
    order = np.argsort(distances, kind="stable")[:top]
    return [
        Candidate(gallery.item_ids[i], gallery.names[i], int(distances[i]))
        for i in order
    ]


def timed_match(
    crop: np.ndarray, gallery: Gallery, top: int = TOP_N
) -> tuple[list[Candidate], float]:
    """match() with its duration in seconds."""
    start = time.perf_counter()
    found = match(crop, gallery, top)
    return found, time.perf_counter() - start
