"""Tests for the stash pass: slot identification and badges.

The embedder, the crop step and the slot mark readers are stubbed, so these
run without the model or Tesseract.
"""

import numpy as np
import pytest

from arc_helper import stash_scan
from arc_helper.database import Item
from arc_helper.grid import Slot
from arc_helper.icon_embed import EmbedGallery
from arc_helper.slot_marks import TierRead
from arc_helper.stash_scan import SlotItem
from arc_helper.stash_scan import badges_for
from arc_helper.stash_scan import distinct_gallery
from arc_helper.stash_scan import format_price
from arc_helper.stash_scan import identify_slots
from arc_helper.tooltip_layout import Rect

NAMES = ("Anvil I", "Anvil II", "Anvil III", "Metal Parts", "Rubber Parts")


def _gallery() -> EmbedGallery:
    return EmbedGallery(
        tuple(f"id{n}" for n in range(len(NAMES))),
        NAMES,
        np.eye(len(NAMES), dtype=np.float32),
    )


class QueueEmbedder:
    """Returns the given vectors in order, one per square."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.calls = 0

    def embed(self, squares):
        self.calls += 1
        assert len(squares) == len(self.vectors)
        rows = np.array(self.vectors, dtype=np.float32)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _slots(count: int, *, empty: int = 0) -> list[Slot]:
    slots = [
        Slot(Rect(100 * n, 0, 100 * n + 90, 90), "stash", filled=True)
        for n in range(count)
    ]
    slots += [
        Slot(Rect(100 * n, 100, 100 * n + 90, 190), "stash", filled=False)
        for n in range(empty)
    ]
    return slots


def _catalog(name: str) -> Item | None:
    prices = {"Anvil I": 4000, "Anvil II": 6000, "Anvil III": 7000, "Metal Parts": 810}
    if name not in prices:
        return None
    return Item(name=name, action="Sell", sell_price=prices[name])


@pytest.fixture
def stubs(monkeypatch):
    """No crop step, a settable tier read per slot and a counting stack read."""
    tiers: dict[int, str] = {}
    stack_reads: list[int] = []
    monkeypatch.setattr(stash_scan, "crop_square", lambda crop: crop)

    def fake_tier(crop):
        left = int(crop[0, 0, 0])
        return TierRead(tiers[left], 0.9, "") if left in tiers else None

    def fake_stack(_crop):
        stack_reads.append(1)
        return 3

    monkeypatch.setattr(stash_scan, "read_tier", fake_tier)
    monkeypatch.setattr(stash_scan, "read_stack", fake_stack)
    return tiers, stack_reads


def _frame(slots: list[Slot]) -> np.ndarray:
    """A frame whose red channel at each slot's corner is the slot's index."""
    rgb = np.zeros((200, 1000, 3), dtype=np.uint8)
    for n, slot in enumerate(slots):
        rgb[slot.rect.top, slot.rect.left, 0] = n
    return rgb


@pytest.mark.usefixtures("stubs")
def test_identify_skips_empty_slots_and_applies_the_confidence_rule():
    slots = _slots(3, empty=2)
    embedder = QueueEmbedder(
        [
            [0, 0, 0, 1, 0],  # Metal Parts, similarity 1, clear lead
            [0, 0, 0, 1, 0.98],  # Metal Parts against Rubber Parts, lead too small
            [0.5, 0, 0, 0.5, 0.5],  # three items tie at 0.58, below the similarity rule
        ]
    )
    found = identify_slots(
        _frame(slots), _gallery(), _catalog, embedder=embedder, slots=slots
    )
    assert embedder.calls == 1
    assert len(found) == 3
    assert found[0].item.name == "Metal Parts"
    assert found[0].confidence == pytest.approx(1.0)
    assert found[1].item is None
    assert found[1].guess == "Metal Parts"
    assert found[1].lead < stash_scan.MIN_LEAD
    assert found[2].item is None
    assert found[2].confidence < stash_scan.MIN_SIMILARITY


def test_the_tier_numeral_picks_the_tier_of_a_tiered_item(stubs):
    tiers, _ = stubs
    slots = _slots(2)
    tiers[0] = "III"
    # Both slots embed closest to Anvil I, tier siblings close behind
    embedder = QueueEmbedder([[1, 0.3, 0.2, 0, 0], [1, 0.3, 0.2, 0, 0]])
    found = identify_slots(
        _frame(slots), _gallery(), _catalog, embedder=embedder, slots=slots
    )
    assert found[0].item.name == "Anvil III"
    assert found[0].tier == "III"
    assert found[1].item.name == "Anvil I"
    assert found[1].tier == "I"


def test_stack_reads_are_optional(stubs):
    _, stack_reads = stubs
    slots = _slots(2)
    vectors = [[0, 0, 0, 1, 0], [0, 0, 0, 1, 0]]
    off = identify_slots(
        _frame(slots),
        _gallery(),
        _catalog,
        embedder=QueueEmbedder(vectors),
        slots=slots,
    )
    assert [s.stack for s in off] == [None, None]
    assert stack_reads == []
    on = identify_slots(
        _frame(slots),
        _gallery(),
        _catalog,
        embedder=QueueEmbedder(vectors),
        slots=slots,
        read_stacks=True,
    )
    assert [s.stack for s in on] == [3, 3]
    assert len(stack_reads) == 2


@pytest.mark.usefixtures("stubs")
def test_no_filled_slots_never_touches_the_embedder():
    slots = _slots(0, empty=3)
    embedder = QueueEmbedder([])
    assert (
        identify_slots(
            _frame(slots), _gallery(), _catalog, embedder=embedder, slots=slots
        )
        == []
    )
    assert embedder.calls == 0


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (810, "810"),
        (999, "999"),
        (1000, "1k"),
        (1250, "1.3k"),
        (1600, "1.6k"),
        (4000, "4k"),
        (9960, "10k"),
        (10_000, "10k"),
        (12_400, "12k"),
        (1_250_000, "1.3M"),
    ],
)
def test_format_price(value, text):
    assert format_price(value) == text


def _slot_item(item: Item | None, left: int = 0) -> SlotItem:
    return SlotItem(
        Rect(left, 10, left + 90, 100), "stash", item, None, None, 0.9, 0.2, "x"
    )


def test_badges_gold_border_for_keep_and_price_for_every_item():
    keep = _slot_item(Item(name="Anvil I", action="Keep", sell_price=4000), 0)
    some = _slot_item(Item(name="Metal Parts", action="Keep", sell_price=810), 100)
    sell = _slot_item(Item(name="Rubber Parts", action="Sell", sell_price=1250), 200)
    free = _slot_item(Item(name="Note", action="Sell", sell_price=None), 300)
    unknown = _slot_item(None, 400)
    kinds = {
        "Anvil I": "keep",
        "Metal Parts": "keep_some",
        "Rubber Parts": "sell",
        "Note": "keep",
    }
    badges = badges_for([keep, some, sell, free, unknown], lambda s: kinds[s.item.name])
    borders = [b for b in badges if b["kind"] == "border"]
    tags = [b for b in badges if b["kind"] == "tag"]
    assert [b["rect"][0] for b in borders] == [0, 100, 300]
    assert borders[0] == {
        "rect": [0, 10, 90, 100],
        "color": "#f2b134",
        "width": 3,
        "radius": 4,
        "kind": "border",
    }
    assert [(b["rect"][0], b["text"]) for b in tags] == [
        (0, "Φ 4k"),
        (100, "Φ 810"),
        (200, "Φ 1.3k"),
    ]
    assert tags[0]["corner"] == "bl"
    assert all(b["rect"][0] != 400 for b in badges)


def test_distinct_gallery_keeps_the_plain_item_of_a_shared_icon():
    vectors = np.array([[1, 0], [1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)
    gallery = EmbedGallery(
        ("bp", "gun", "metal", "other"),
        ("Anvil Blueprint", "Anvil I", "Metal Parts", "Rubber Parts"),
        vectors,
    )
    kept, twins = distinct_gallery(gallery)
    assert twins == {"Anvil I": ("Anvil Blueprint",)}
    assert kept.names == ("Anvil I", "Metal Parts", "Rubber Parts")
    assert kept.item_ids == ("gun", "metal", "other")
    assert kept.vectors.shape == (3, 2)


@pytest.mark.parametrize(
    ("twins", "numeral", "expected"),
    [
        ({"Anvil I": ("Anvil Blueprint",)}, "III", "Anvil III"),
        ({"Anvil I": ("Anvil Blueprint",)}, None, None),
        ({"Anvil I": ("Anvil Blueprint", "Amplified Anvil II")}, "III", None),
    ],
)
def test_twins_need_the_numeral_to_settle_the_item(stubs, twins, numeral, expected):
    tiers, _ = stubs
    slots = _slots(1)
    if numeral:
        tiers[0] = numeral
    embedder = QueueEmbedder([[1, 0.3, 0.2, 0, 0]])
    found = identify_slots(
        _frame(slots), _gallery(), _catalog, embedder=embedder, slots=slots, twins=twins
    )
    assert (found[0].item.name if found[0].item else None) == expected
    assert found[0].guess == "Anvil I"


def test_an_untiered_item_with_twins_is_unidentified(stubs):
    tiers, _ = stubs
    tiers[0] = "II"
    slots = _slots(1)
    gallery = EmbedGallery(
        ("hook", "metal"),
        ("Grappling Hook", "Metal Parts"),
        np.eye(2, dtype=np.float32),
    )
    twins = {"Grappling Hook": ("Grappling Hook Blueprint",)}
    found = identify_slots(
        _frame(slots),
        gallery,
        lambda name: Item(name=name, action="Sell", sell_price=1200),
        embedder=QueueEmbedder([[1, 0]]),
        slots=slots,
        twins=twins,
    )
    assert found[0].item is None
    assert found[0].guess == "Grappling Hook"
    assert found[0].confidence == pytest.approx(1.0)
