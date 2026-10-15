"""Tests for the perceptual-hash icon baseline."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from arc_helper.icon_match import METHODS
from arc_helper.icon_match import TOP_N
from arc_helper.icon_match import Gallery
from arc_helper.icon_match import art_region
from arc_helper.icon_match import build_gallery
from arc_helper.icon_match import crop_hash
from arc_helper.icon_match import match

FIXTURES = Path(__file__).parent / "fixtures" / "icon_match"
# Four real slot crops from the 12 and 14 Oct captures, and the arctracker
# icon of each item
NAMES = {
    "arc_alloy": "ARC Alloy",
    "light_ammo": "Light Ammo",
    "shield_recharger": "Shield Recharger",
    "zipline": "Zipline",
}


def _slot(item_id: str) -> np.ndarray:
    with Image.open(FIXTURES / f"slot_{item_id}.png") as image:
        return np.asarray(image.convert("RGB"))


def _gallery(method: str = "phash") -> Gallery:
    icons = {k: (v, FIXTURES / f"icon_{k}.png") for k, v in NAMES.items()}
    return build_gallery(icons, method)


@pytest.mark.parametrize("method", METHODS)
def test_match_ranks_the_true_item_first(method: str) -> None:
    gallery = _gallery(method)
    for item_id in NAMES:
        found = match(_slot(item_id), gallery)
        assert found[0].item_id == item_id
        assert found[0].distance < found[1].distance


def test_match_returns_at_most_top_n_sorted_by_distance() -> None:
    gallery = _gallery()
    found = match(_slot("arc_alloy"), gallery, top=TOP_N)
    assert len(found) == len(NAMES)
    distances = [c.distance for c in found]
    assert distances == sorted(distances)
    assert len(match(_slot("arc_alloy"), gallery, top=2)) == 2


def test_match_on_an_empty_gallery_is_empty() -> None:
    empty = build_gallery({})
    assert match(_slot("arc_alloy"), empty) == []


def test_stack_count_and_wedge_do_not_change_the_hash() -> None:
    crop = _slot("arc_alloy")
    # Overwrite the info bar (the stack count) with noise. The art area is
    # above 72 percent of the slot height, so the hash must not move.
    changed = crop.copy()
    bar = int(crop.shape[0] * 0.75)
    changed[bar:] = np.random.default_rng(1).integers(
        0, 255, changed[bar:].shape, dtype=np.uint8
    )
    assert np.array_equal(crop_hash(crop), crop_hash(changed))


def test_art_region_drops_the_bar_and_a_wide_slot_keeps_less() -> None:
    square = np.zeros((200, 200, 3), dtype=np.uint8)
    wide = np.zeros((200, 300, 3), dtype=np.uint8)
    assert art_region(square).shape[0] < 200 * 0.72
    assert art_region(wide).shape[0] < art_region(square).shape[0]
