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
from arc_helper.slot_marks import BarGeometry
from arc_helper.slot_marks import TierRead
from arc_helper.stash_scan import SlotItem
from arc_helper.stash_scan import badges_for
from arc_helper.stash_scan import distinct_gallery
from arc_helper.stash_scan import format_price
from arc_helper.stash_scan import identify_slots
from arc_helper.stash_scan import owned_totals
from arc_helper.tooltip_layout import Rect
from arc_helper.verdict import Verdict

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


ICON_BOX = Rect(4, 70, 20, 86)


def _square(item, left=0, *, bar=None, stack=None) -> SlotItem:
    return SlotItem(
        Rect(left, 10, left + 90, 100), "stash", item, None, stack, 0.9, 0.2, "x", bar
    )


def _bar(icon_box=ICON_BOX) -> BarGeometry:
    return BarGeometry(
        Rect(3, 68, 87, 88), (40, 90, 120), icon_box, Rect(70, 70, 84, 86)
    )


def test_square_slot_with_icon_box_gets_a_patch_in_screen_px():
    item = Item(name="Metal Parts", action="Sell", sell_price=810)
    badges = badges_for(
        [_square(item, 100, bar=_bar())], lambda _s: Verdict("sell", "x")
    )
    assert badges == [
        {
            "rect": [103, 78, 187, 98],
            "icon_right": 120,
            "patch_right": 100 + _bar().patch_right,
            "colour": [40, 90, 120],
            "text": "Φ 810",
            "radius": 6,
            "kind": "patch",
        }
    ]


def test_no_icon_box_gives_no_patch_and_wide_slots_keep_a_dark_tag():
    item = Item(name="Metal Parts", action="Sell", sell_price=810)
    plain = badges_for(
        [_square(item, bar=_bar(icon_box=None))], lambda _s: Verdict("sell", "x")
    )
    assert plain == []
    wide = SlotItem(Rect(0, 10, 200, 100), "stash", item, None, 2, 0.9, 0.2, "x")
    (tag,) = badges_for([wide], lambda _s: Verdict("sell", "x"))
    assert (tag["kind"], tag["corner"], tag["text"]) == ("tag", "tl", "Φ 1.6k")


def test_patch_text_is_the_stacks_worth():
    item = Item(name="Metal Parts", action="Sell", sell_price=810)
    (patch,) = badges_for(
        [_square(item, bar=_bar(), stack=3)], lambda _s: Verdict("sell", "x")
    )
    assert patch["text"] == "Φ 2.4k"


def test_need_tag_is_red_with_an_outline_while_short_and_green_when_enough():
    sources = (("Expedition 5 phase 2", 6), ("Gunsmith II", 4))
    first = _square(Item(name="Metal Parts", action="Keep"), 0, stack=2)
    second = _square(Item(name="Metal Parts", action="Keep"), 100, stack=4)
    short = badges_for([first, second], lambda _s: Verdict("keep", "x", 10, sources))
    tags = [b for b in short if b["kind"] == "tag"]
    assert [(t["text"], t["fg"], t["check"]) for t in tags] == [
        ("6/10", "#ff5c4d", False)
    ] * 2
    assert tags[0]["outline"] == [255, 92, 77]
    assert tags[0]["corner"] == "tl"
    borders = [b for b in short if b["kind"] == "border"]
    assert [(b["rect"][0], b["color"]) for b in borders] == [
        (0, "#ff5c4d"),
        (100, "#ff5c4d"),
    ]
    enough = badges_for(
        [first, second], lambda _s: Verdict("keep", "x", 5, (("Quest", 5),))
    )
    assert [(b["text"], b["fg"], b["check"]) for b in enough] == [
        ("6/5", "#4fd36b", True)
    ] * 2
    assert enough[0]["outline"] == [79, 211, 107]
    assert all(b["kind"] == "tag" for b in enough)


def test_a_verdict_without_sources_gives_no_tag_and_no_outline():
    weapon = _square(Item(name="Weapon", action="Keep"), 0)
    badges = badges_for(
        [weapon, _square(None, 100)], lambda _s: Verdict("keep", "x", 1)
    )
    assert badges == []


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


def test_owned_totals_adds_stacks_and_skips_unidentified_slots():
    parts = Item(name="Metal Parts", action="Sell", sell_price=810)
    slots = [
        _slot_item(parts, 0)._replace(stack=2),
        _slot_item(parts, 100),
        _slot_item(parts, 200)._replace(stack=4),
        _slot_item(None, 300)._replace(stack=9),
    ]
    assert owned_totals(slots) == {"Metal Parts": 7}


@pytest.mark.usefixtures("stubs")
def test_identify_gives_bar_geometry_to_square_slots_only(monkeypatch):
    geometry = object()
    calls = []

    def fake_geometry(crop):
        calls.append(crop.shape)
        return geometry

    monkeypatch.setattr(stash_scan, "bar_geometry", fake_geometry)
    square = Slot(Rect(0, 0, 90, 90), "stash", filled=True)
    wide = Slot(Rect(100, 0, 280, 90), "equipment", filled=True)
    slots = [square, wide]
    found = identify_slots(
        _frame(slots),
        _gallery(),
        _catalog,
        embedder=QueueEmbedder([[0, 0, 0, 1, 0], [0, 0, 0, 1, 0]]),
        slots=slots,
    )
    assert found[0].bar is geometry
    assert found[1].bar is None
    assert calls == [(90, 90, 3)]


def test_stacks_are_read_for_identified_slots_only(stubs):
    _, stack_reads = stubs
    slots = _slots(2)
    # The second slot leads Rubber Parts by too little, so it stays unidentified
    vectors = [[0, 0, 0, 1, 0], [0, 0, 0, 1, 0.98]]
    found = identify_slots(
        _frame(slots),
        _gallery(),
        _catalog,
        embedder=QueueEmbedder(vectors),
        slots=slots,
        read_stacks=True,
    )
    assert [s.stack for s in found] == [3, None]
    assert len(stack_reads) == 1


def test_a_wide_slot_with_a_need_tag_moves_its_worth_tag_to_the_top_right():
    item = Item(name="Weapon", action="Keep", sell_price=810)
    wide = SlotItem(Rect(0, 10, 200, 100), "stash", item, None, None, 0.9, 0.2, "x")
    badges = badges_for([wide], lambda _s: Verdict("keep", "x", 1, (("Quest", 1),)))
    corners = {b["text"]: b["corner"] for b in badges if b["kind"] == "tag"}
    assert corners == {"1/1": "tl", "Φ 810": "tr"}


def _looks(slots: list[Slot], looks: list[int]) -> np.ndarray:
    """A frame whose slots show a block of the given value in the middle."""
    rgb = np.zeros((200, 1000, 3), dtype=np.uint8)
    for slot, look in zip(slots, looks, strict=True):
        top, left = slot.rect.top, slot.rect.left
        rgb[top + 40 : top + 50, left + 40 : left + 50] = look
    return rgb


@pytest.fixture
def no_bars(monkeypatch):
    monkeypatch.setattr(stash_scan, "bar_geometry", lambda _crop: None)


METAL, RUBBER, ANVIL = [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0]


@pytest.mark.usefixtures("no_bars")
def test_a_second_pass_on_the_same_or_moved_slots_never_embeds(stubs):
    _, stack_reads = stubs
    slots = _slots(3)
    cache = stash_scan.SlotCache()
    first = identify_slots(
        _looks(slots, [60, 120, 180]),
        _gallery(),
        _catalog,
        embedder=QueueEmbedder([METAL, RUBBER, ANVIL]),
        slots=slots,
        read_stacks=True,
        cache=cache,
    )
    reads = len(stack_reads)
    again = QueueEmbedder([])
    same = identify_slots(
        _looks(slots, [60, 120, 180]),
        _gallery(),
        _catalog,
        embedder=again,
        slots=slots,
        read_stacks=True,
        cache=cache,
    )
    # The looks moved one slot left, as after a scroll
    moved = identify_slots(
        _looks(slots, [120, 180, 60]),
        _gallery(),
        _catalog,
        embedder=again,
        slots=slots,
        read_stacks=True,
        cache=cache,
    )
    assert again.calls == 0
    assert len(stack_reads) == reads
    assert same == first
    assert [s.guess for s in moved] == ["Rubber Parts", "Anvil I", "Metal Parts"]
    assert [s.rect for s in moved] == [s.rect for s in slots]


@pytest.mark.usefixtures("no_bars")
def test_a_changed_stack_count_reads_only_that_slot(stubs):
    _, stack_reads = stubs
    slots = _slots(3)
    cache = stash_scan.SlotCache()
    rgb = _looks(slots, [60, 120, 180])
    identify_slots(
        rgb,
        _gallery(),
        _catalog,
        embedder=QueueEmbedder([METAL, ANVIL, RUBBER]),
        slots=slots,
        read_stacks=True,
        cache=cache,
    )
    reads = len(stack_reads)
    # A digit pixel in the second slot's bar changes
    rgb[70, slots[1].rect.left + 70] = 220
    once = QueueEmbedder([ANVIL])
    found = identify_slots(
        rgb,
        _gallery(),
        _catalog,
        embedder=once,
        slots=slots,
        read_stacks=True,
        cache=cache,
    )
    assert once.calls == 1
    assert len(stack_reads) == reads + 1
    assert [s.guess for s in found] == ["Metal Parts", "Anvil I", "Rubber Parts"]


def test_the_slot_cache_drops_the_least_recently_used_entry():
    cache = stash_scan.SlotCache(size=2)
    entry = object()
    cache.put("a", entry)
    cache.put("b", entry)
    assert cache.get("a") is entry
    cache.put("c", entry)
    assert len(cache) == 2
    assert "b" not in cache
    assert "a" in cache
    assert "c" in cache


# A 194 by 192 tile with art above a dark bar, a type icon at its left and a
# one digit count at its right (patch boundary 150 - 5 = 145)
BOX = Rect(11, 145, 181, 191)


def _tile() -> np.ndarray:
    crop = np.random.default_rng(0).integers(0, 256, (194, 192, 3), dtype=np.uint8)
    crop[145:191, 11:181] = 15
    crop[155:180, 15:40] = 220  # type icon
    crop[155:180, 150:160] = 220  # count
    return crop


def _badged(crop: np.ndarray) -> np.ndarray:
    """The tile with a red outline, a have/need pill and a price patch."""
    out = crop.copy()
    edge = stash_scan.BORDER_WIDTH
    out[:edge] = out[-edge:] = (255, 92, 77)
    out[:, :edge] = out[:, -edge:] = (255, 92, 77)
    # A have/need pill top left: 33 px text, inset 10, pill about 44 tall
    out[9:56, 9:120] = (20, 20, 20)
    # The patch in the bar's colour up to 145, its price text in white
    out[145:191, 11:145] = 15
    out[158:178, 20:90] = 230
    return out


def test_the_signature_ignores_the_tag_outline_and_patch_but_not_the_count():
    crop = _tile()
    clean = stash_scan.slot_signature(crop, BOX)
    assert stash_scan.patch_boundary(crop, BOX) == 145
    assert stash_scan.slot_signature(_badged(crop), BOX) == clean
    assert stash_scan.look_signature(_badged(crop)) == stash_scan.look_signature(crop)
    # 5 becomes 15: a leading digit left of the old count
    counted = crop.copy()
    counted[155:180, 137:147] = 220
    assert stash_scan.slot_signature(counted, BOX) != clean
    assert stash_scan.slot_signature(_badged(counted), BOX) != clean


def test_a_slot_whose_bar_is_lost_under_its_patch_still_hits(stubs, monkeypatch):
    _, stack_reads = stubs
    bar = BarGeometry(
        BOX, (15, 15, 15), Rect(15, 155, 40, 180), Rect(150, 155, 160, 180)
    )
    # As on the rendered frames: no bar is found once the patch is drawn
    monkeypatch.setattr(
        stash_scan,
        "bar_geometry",
        lambda crop: bar if crop[160, 16, 0] == 220 else None,
    )
    slots = [Slot(Rect(0, 0, 192, 194), "stash", filled=True)]
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    rgb[:194, :192] = _tile()
    cache = stash_scan.SlotCache()
    clean = identify_slots(
        rgb,
        _gallery(),
        _catalog,
        embedder=QueueEmbedder([METAL]),
        slots=slots,
        read_stacks=True,
        cache=cache,
    )
    badged = rgb.copy()
    badged[:194, :192] = _badged(_tile())
    never = QueueEmbedder([])
    found = identify_slots(
        badged,
        _gallery(),
        _catalog,
        embedder=never,
        slots=slots,
        read_stacks=True,
        cache=cache,
    )
    assert never.calls == 0
    assert len(stack_reads) == 1
    assert found == clean
    assert found[0].bar is bar
