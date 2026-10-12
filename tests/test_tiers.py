"""Tests for Roman numeral tier parsing, resolution and stroke counting."""

import numpy as np
import pytest

from arc_helper.tiers import TIER_TOKENS
from arc_helper.tiers import count_strokes
from arc_helper.tiers import resolve_tier
from arc_helper.tiers import split_read
from arc_helper.tiers import split_tier
from arc_helper.tiers import tier_key


def test_tier_tokens_order():
    assert TIER_TOKENS == ("I", "II", "III", "IV", "V")


def test_split_tier():
    assert split_tier("Anvil IV") == ("Anvil", "IV")
    assert split_tier("Il Toro II") == ("Il Toro", "II")
    assert split_tier("Rascal") == ("Rascal", None)
    assert split_tier("Anvil Blueprint") == ("Anvil Blueprint", None)


def test_tier_key_maps_lookalikes_before_dropping_punctuation():
    assert tier_key("ARPEGGIO |V") == "arpeggioiv"
    assert tier_key("ANVIL 1l") == "anviiii"
    assert tier_key("Il Toro") == "iitoro"


def test_split_read_takes_longest_base_with_tier_suffix():
    bases = ["anvii", "iitoro", "rascai"]
    assert split_read("anviiii", bases) == ("anvii", "II")
    assert split_read("iitoroii", bases) == ("iitoro", "II")
    assert split_read("rascaiiv", bases) == ("rascai", "IV")
    # longest base wins over a shorter one that leaves a valid suffix
    assert split_read("abcii", ["abc", "abci"]) == ("abci", "I")


def test_split_read_leaves_untiered_text_alone():
    bases = ["anvii", "rascai"]
    assert split_read("rascai", bases) == ("rascai", None)  # no suffix
    assert split_read("battery", bases) == ("battery", None)
    assert split_read("anviispiitter", bases) == ("anviispiitter", None)


def test_resolve_tier():
    assert resolve_tier("II", ["I", "II", "III", "IV"]) == "II"
    # V read with no V tier: IV and I are both one edit away, V prefers IV
    assert resolve_tier("V", ["I", "II", "III", "IV"]) == "IV"
    # no V in the read: prefer the candidate without V on a tie
    assert resolve_tier("I", ["II", "IV"]) == "II"
    assert resolve_tier("I", ["II", "III", "IV"]) == "II"
    assert resolve_tier("II", []) is None


# Synthetic tier tokens: rectangles for I strokes, two diagonals for V


def _canvas(height):
    return np.zeros((height + 8, height * 4), dtype=bool)


def _stroke(img, x, height):
    img[4 : 4 + height, x : x + max(2, round(0.18 * height))] = True
    return x + max(2, round(0.18 * height))


def _vee(img, x, height):
    width, thick = round(0.8 * height), max(2, round(0.18 * height))
    for y in range(height):
        shift = y / (height - 1) * (width - thick) / 2
        for col in (x + shift, x + width - thick - shift):
            img[4 + y, round(col) : round(col) + thick] = True


def _draw(tier, height):
    img = _canvas(height)
    gap = round(0.25 * height)
    x = 4
    if tier == "V":
        _vee(img, x, height)
        return img
    for letter in tier:
        if letter == "I":
            x = _stroke(img, x, height) + gap
        else:
            _vee(img, x, height)
    return img


@pytest.mark.parametrize("height", [20, 48])
@pytest.mark.parametrize("tier", TIER_TOKENS)
def test_count_strokes_reads_synthetic_tokens(tier, height):
    vote = count_strokes(_draw(tier, height))
    assert vote.tier == tier, vote.detail
    assert vote.confidence >= 0.6


def test_count_strokes_rejects_smeared_blob():
    yy, xx = np.ogrid[:48, :80]
    blob = ((yy - 24) / 18) ** 2 + ((xx - 40) / 20) ** 2 <= 1
    vote = count_strokes(blob)
    assert vote.tier is None
    assert vote.confidence == 0


def test_count_strokes_rejects_empty_image():
    assert count_strokes(np.zeros((30, 30), dtype=bool)).tier is None
