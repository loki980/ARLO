"""Tests for the DINOv2 embedding icon match.

Tests that run the model skip unless onnxruntime (the 'icons' extra) is
installed and the model file is already in the model cache. The rest use a
stand-in embedder and run everywhere.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from arc_helper import icon_embed
from arc_helper.icon_embed import Embedder
from arc_helper.icon_embed import ModelUnavailableError
from arc_helper.icon_embed import build_gallery
from arc_helper.icon_embed import crop_square
from arc_helper.icon_embed import match
from arc_helper.icon_match import TOP_N
from arc_helper.icon_match import art_picture
from arc_helper.icon_match import art_region

FIXTURES = Path(__file__).parent / "fixtures" / "icon_match"
# The same four real slot crops and arctracker icons as test_icon_match
NAMES = {
    "arc_alloy": "ARC Alloy",
    "light_ammo": "Light Ammo",
    "shield_recharger": "Shield Recharger",
    "zipline": "Zipline",
}


def _slot(item_id: str) -> np.ndarray:
    with Image.open(FIXTURES / f"slot_{item_id}.png") as image:
        return np.asarray(image.convert("RGB"))


def _icons() -> dict[str, tuple[str, Path]]:
    return {k: (v, FIXTURES / f"icon_{k}.png") for k, v in NAMES.items()}


class CountingEmbedder(Embedder):
    """Embeds a square as its mean colour, and counts the pictures it saw."""

    def __init__(self) -> None:
        super().__init__(download=False)
        self.seen = 0

    def embed(self, squares):
        self.seen += len(squares)
        rows = np.array([s.reshape(-1, 3).mean(axis=0) + 1 for s in squares])
        return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture(scope="module")
def model_embedder() -> Embedder:
    pytest.importorskip("onnxruntime")
    try:
        icon_embed.model_path(download=False)
    except ModelUnavailableError:
        pytest.skip(f"no DINOv2 model in {icon_embed.MODEL_DIR}")
    return Embedder(download=False)


def test_match_ranks_the_true_item_first(model_embedder: Embedder) -> None:
    gallery = build_gallery(_icons(), model_embedder)
    for item_id in NAMES:
        found = match(_slot(item_id), gallery, model_embedder)
        assert found[0].item_id == item_id
        assert found[0].similarity > found[1].similarity


def test_match_returns_at_most_top_n_most_similar_first() -> None:
    embedder = CountingEmbedder()
    gallery = build_gallery(_icons(), embedder)
    found = match(_slot("arc_alloy"), gallery, embedder)
    assert len(found) == min(TOP_N, len(NAMES))
    similarities = [c.similarity for c in found]
    assert similarities == sorted(similarities, reverse=True)
    assert len(match(_slot("arc_alloy"), gallery, embedder, top=2)) == 2


def test_match_on_an_empty_gallery_is_empty() -> None:
    embedder = CountingEmbedder()
    assert match(_slot("arc_alloy"), build_gallery({}, embedder), embedder) == []
    assert embedder.seen == 0


def test_gallery_is_embedded_once_and_rebuilt_when_an_icon_changes(
    tmp_path: Path,
) -> None:
    icons = {}
    for item_id, (name, path) in _icons().items():
        copy = tmp_path / path.name
        copy.write_bytes(path.read_bytes())
        icons[item_id] = (name, copy)
    cache = tmp_path / "gallery.npz"

    embedder = CountingEmbedder()
    first = build_gallery(icons, embedder, cache)
    assert embedder.seen == len(icons)
    again = build_gallery(icons, embedder, cache)
    assert embedder.seen == len(icons)
    assert np.allclose(first.vectors, again.vectors)
    assert again.item_ids == first.item_ids

    changed = icons["zipline"][1]
    stat = changed.stat()
    os.utime(changed, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    build_gallery(icons, embedder, cache)
    assert embedder.seen == 2 * len(icons)


def test_the_model_loads_lazily_and_a_missing_model_is_reported(
    tmp_path: Path,
) -> None:
    embedder = Embedder(tmp_path, download=False)
    assert not embedder.loaded
    with pytest.raises(ModelUnavailableError):
        embedder.embed([np.zeros((8, 8, 3), dtype=np.uint8)])


def test_a_wide_slot_is_turned_to_the_catalog_diagonal() -> None:
    # A wide slot with a level bright bar where the weapon art is drawn
    wide = np.full((300, 450, 3), 20, dtype=np.uint8)
    wide[90:100, 100:350] = 220
    turned = art_picture(wide)
    ys, xs = np.nonzero(turned.max(axis=2) > 120)
    slope = np.polyfit(xs, ys, 1)[0]
    # Image rows grow downwards, so a bar falling to the right has a positive slope
    assert 0.6 < slope < 1.0

    square = _slot("arc_alloy")
    assert np.array_equal(art_picture(square), art_region(square))


def test_crop_square_is_the_model_input_size() -> None:
    assert crop_square(_slot("zipline")).shape == (
        icon_embed.INPUT_SIDE,
        icon_embed.INPUT_SIDE,
        3,
    )
