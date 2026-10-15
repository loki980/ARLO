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
   attachment row.
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
    wide = width / height > WIDE_ASPECT
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


def _background(region: np.ndarray) -> np.ndarray:
    """Median colour of the top strip of a region, the slot's dark backdrop."""
    strip = region[: max(2, region.shape[0] // 12)].reshape(-1, 3)
    return np.median(strip, axis=0)


def normalize(region: np.ndarray) -> np.ndarray:
    """Return a NORM_SIDE square grey picture of the art in an RGB region.

    The art is found as the pixels that differ from the background, cut to
    its bounding box and padded to a square on the background grey. An empty
    mask (nothing but backdrop) gives a flat picture.
    """
    background = _background(region)
    diff = np.abs(region.astype(np.int16) - background.astype(np.int16)).max(axis=2)
    mask = diff > MASK_DIFF
    mask = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    ).astype(bool)
    rows = np.flatnonzero(mask.sum(axis=1) >= MIN_MASK_PIXELS)
    cols = np.flatnonzero(mask.sum(axis=0) >= MIN_MASK_PIXELS)
    grey = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    flat = round(float(background @ GREY_WEIGHTS))
    if rows.size == 0 or cols.size == 0:
        return np.full((NORM_SIDE, NORM_SIDE), flat, dtype=np.uint8)
    box = grey[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    box_mask = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    box = np.where(box_mask, box, flat).astype(np.uint8)
    side = max(box.shape)
    square = np.full((side, side), flat, dtype=np.uint8)
    y0 = (side - box.shape[0]) // 2
    x0 = (side - box.shape[1]) // 2
    square[y0 : y0 + box.shape[0], x0 : x0 + box.shape[1]] = box
    return cv2.resize(square, (NORM_SIDE, NORM_SIDE), interpolation=cv2.INTER_AREA)


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
    return _hash(normalize(art_region(crop)), method)


def icon_hash(icon: Image.Image, method: str = "phash") -> np.ndarray:
    """Hash of a catalog icon, flattened onto the gallery background."""
    rgba = icon.convert("RGBA")
    backdrop = Image.new("RGBA", rgba.size, (*GALLERY_BACKGROUND, 255))
    flat = Image.alpha_composite(backdrop, rgba).convert("RGB")
    return _hash(normalize(_pad_top(np.asarray(flat))), method)


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
