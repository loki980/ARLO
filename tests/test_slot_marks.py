"""Tests for reading tier numerals and stack counts on slot crops."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from arc_helper.slot_marks import read_stack
from arc_helper.slot_marks import read_tier

FIXTURES = Path(__file__).parent / "fixtures" / "slot_marks"
# Real hovered cell crops from the 12 and 14 Oct captures
NUMERALS = {
    "torrente_i": "I",
    "torrente_ii": "II",
    "rattler_iii": "III",
    "burletta_iv": "IV",
    "wide_anvil_iv": "IV",
}
STACKS = {"medium_ammo_x80": 80, "zipline_x1": 1}
NO_MARK = "grappling_hook_no_mark"

needs_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="needs tesseract"
)


def _slot(name: str) -> np.ndarray:
    with Image.open(FIXTURES / f"{name}.png") as image:
        return np.asarray(image.convert("RGB"))


@pytest.mark.parametrize(("name", "tier"), NUMERALS.items())
def test_read_tier_reads_the_numeral(name: str, tier: str) -> None:
    read = read_tier(_slot(name))
    assert read is not None
    assert read.tier == tier
    assert 0.5 <= read.confidence <= 1.0


@pytest.mark.parametrize("name", [*STACKS, NO_MARK])
def test_read_tier_is_none_without_a_numeral(name: str) -> None:
    # The x1 stack ends in a 1, a single stroke like I, but white and serifless
    assert read_tier(_slot(name)) is None


@needs_tesseract
@pytest.mark.parametrize(("name", "count"), STACKS.items())
def test_read_stack_reads_the_count(name: str, count: int) -> None:
    assert read_stack(_slot(name)) == count


@pytest.mark.parametrize("name", [*NUMERALS, NO_MARK])
def test_read_stack_is_none_without_a_count(name: str) -> None:
    assert read_stack(_slot(name)) is None


def test_a_slot_without_a_dark_bar_reads_nothing() -> None:
    plain = np.full((200, 200, 3), 120, dtype=np.uint8)
    assert read_tier(plain) is None
    assert read_stack(plain) is None
