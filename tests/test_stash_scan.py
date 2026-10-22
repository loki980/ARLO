"""Tests for the stash pass: slot identification and badges.

The embedder, the crop step and the slot mark readers are stubbed, so these
run without the model or Tesseract.
"""

import itertools
import math

import numpy as np
import pytest

from arc_helper import stash_scan
from arc_helper.database import Item
from arc_helper.grid import ARLO_GOLDS
from arc_helper.grid import GOLD_TOLERANCE
from arc_helper.grid import RED_TOLERANCE
from arc_helper.grid import SHORT_RED
from arc_helper.grid import Slot
from arc_helper.icon_embed import EmbedCandidate
from arc_helper.icon_embed import EmbedGallery
from arc_helper.icon_match import ART_MARGIN
from arc_helper.icon_match import GALLERY_BACKGROUND
from arc_helper.slot_marks import BarGeometry
from arc_helper.slot_marks import TierRead
from arc_helper.stash_scan import SlotItem
from arc_helper.stash_scan import badges_for
from arc_helper.stash_scan import distinct_gallery
from arc_helper.stash_scan import format_price
from arc_helper.stash_scan import identify_slots
from arc_helper.stash_scan import owned_totals
from arc_helper.stash_scan import worth_colours
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
    monkeypatch.setattr(stash_scan, "is_blueprint_tile", lambda _crop: None)
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
            "colour": [0xF1, 0xB8, 0x00],
            "fg": "#111111",
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


def _contrast(first: str, second: str) -> float:
    """The WCAG contrast ratio of two '#rrggbb' colours."""

    def luminance(colour: str) -> float:
        lin = []
        for i in (1, 3, 5):
            c = int(colour[i : i + 2], 16) / 255
            lin.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    dark, light = sorted([luminance(first), luminance(second)])
    return (light + 0.05) / (dark + 0.05)


DARK = "#111111"


@pytest.mark.parametrize(
    ("worth", "colours"),
    [
        (0, ("#f43b20", DARK)),
        (stash_scan.WORTH_LOW - 1, ("#f43b20", DARK)),
        (stash_scan.WORTH_LOW, ("#f43b20", DARK)),
        (500, ("#f28f00", DARK)),
        (1118, ("#e8d44a", DARK)),
        (2000, ("#b3ca3b", DARK)),
        (stash_scan.WORTH_HIGH - 1, ("#3fb950", DARK)),
        (stash_scan.WORTH_HIGH, ("#ad7500", DARK)),
        (1_000_000, ("#ad7500", DARK)),
    ],
)
def test_worth_colours_at_the_thresholds(worth, colours):
    assert worth_colours(worth) == colours


def test_worth_fill_moves_from_red_toward_green_then_gold():
    hues = [
        stash_scan.oklch(worth_colours(w)[0])[2]
        for w in range(stash_scan.WORTH_LOW, stash_scan.WORTH_HIGH)
    ]
    # Rounding to 8 bits where blue is near 0 can turn a hue back a little
    assert all(b > a - 0.5 for a, b in itertools.pairwise(hues))
    assert all(b > a for a, b in itertools.pairwise(hues[::50]))
    middle = hues[1118 - stash_scan.WORTH_LOW]
    assert (hues[0] < 40, 90 < middle < 110, hues[-1] > 140) == (True, True, True)
    below = {worth_colours(w)[0] for w in range(stash_scan.WORTH_LOW)}
    above = {worth_colours(w)[0] for w in range(stash_scan.WORTH_HIGH, 100_001)}
    assert (below, above) == ({stash_scan.WORTH_RED}, {stash_scan.WORTH_GOLD})


def _rgb_of(colour: str) -> list[int]:
    return [int(colour[i : i + 2], 16) for i in (1, 3, 5)]


def _channel_gap(colour: str, rgb: tuple[int, int, int]) -> int:
    """The largest per-channel difference of a '#rrggbb' colour from rgb."""
    return max(abs(a - b) for a, b in zip(_rgb_of(colour), rgb, strict=True))


def _oklab_gap(first: str, second: str) -> float:
    points = []
    for colour in (first, second):
        lightness, chroma, hue = stash_scan.oklch(colour)
        angle = math.radians(hue)
        points.append((lightness, chroma * math.cos(angle), chroma * math.sin(angle)))
    return math.dist(*points)


def test_gold_stays_apart_from_arlo_golds_and_from_the_yellow():
    gold = stash_scan.WORTH_GOLD
    assert all(_channel_gap(gold, g) > 3 * GOLD_TOLERANCE for g in ARLO_GOLDS)
    yellow, green = stash_scan.WORTH_YELLOW, stash_scan.WORTH_GREEN
    assert _oklab_gap(gold, yellow) >= _oklab_gap(yellow, green)


def test_red_stays_apart_from_the_short_stock_red():
    assert _channel_gap(stash_scan.WORTH_RED, SHORT_RED) > 3 * RED_TOLERANCE


def test_no_fill_on_the_scale_matches_the_grid_readers_colours():
    for code in stash_scan.FILL_CODES:
        fill = f"#{int(code):06x}"
        assert _channel_gap(fill, SHORT_RED) > RED_TOLERANCE, fill
        assert all(_channel_gap(fill, g) > GOLD_TOLERANCE for g in ARLO_GOLDS), fill


def test_worth_text_reads_on_its_fill_from_1_to_100000():
    for worth in range(1, 100_001):
        fill, text = worth_colours(worth)
        assert text in {stash_scan.WORTH_DARK_TEXT, stash_scan.WORTH_LIGHT_TEXT}
        assert _contrast(fill, text) >= 4.5, worth


def test_a_stack_of_ten_600_coin_items_is_gold_on_patch_and_pill():
    item = Item(name="Gear", action="Sell", sell_price=600)
    square = _square(item, bar=_bar(), stack=10)
    wide = SlotItem(Rect(0, 10, 200, 100), "stash", item, None, 10, 0.9, 0.2, "x")
    patch, pill = badges_for([square, wide], lambda _s: Verdict("sell", "x"))
    assert (patch["text"], patch["colour"], patch["fg"]) == (
        "Φ 6k",
        [0xAD, 0x75, 0x00],
        DARK,
    )
    assert (pill["text"], pill["bg"], pill["fg"]) == ("Φ 6k", "#ad7500ff", DARK)


def test_a_cheaper_pill_keeps_the_tag_alpha():
    item = Item(name="Gear", action="Sell", sell_price=600)
    wide = SlotItem(Rect(0, 10, 200, 100), "stash", item, None, 1, 0.9, 0.2, "x")
    (pill,) = badges_for([wide], lambda _s: Verdict("sell", "x"))
    fill, text = worth_colours(600)
    assert (pill["bg"], pill["fg"]) == (fill + "d9", text)


def _tile_with_bar() -> np.ndarray:
    """A 192 px square crop: slate body, a dark bar at rows 141 to 181, a count."""
    crop = np.full((192, 192, 3), (17, 28, 45), dtype=np.uint8)
    crop[141:181, 3:189] = (9, 12, 25)
    crop[150:170, 160:175] = (230, 230, 230)
    return crop


def test_unpatch_repaints_a_worth_patch_fill_in_the_bar_colour():
    crop = _tile_with_bar()
    crop[141:181, 3:90] = _rgb_of(worth_colours(2000)[0])
    crop[150:170, 15:40] = (0xF9, 0xEE, 0xDF)
    out = stash_scan.unpatch(crop)
    assert tuple(out[145, 60]) == (9, 12, 25)
    assert tuple(out[160, 20]) == (0xF9, 0xEE, 0xDF)
    assert tuple(out[160, 165]) == (230, 230, 230)
    assert (out[:141] == crop[:141]).all()


def test_unpatch_leaves_flat_art_in_a_scale_colour_above_the_bar_alone():
    crop = _tile_with_bar()
    crop[113:143, 2:29] = _rgb_of(stash_scan.WORTH_YELLOW)
    assert stash_scan.unpatch(crop) is crop


def test_need_tag_is_red_while_short_and_green_when_enough():
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
    # The tag says it, no tile outline is drawn
    assert [b for b in short if b["kind"] == "border"] == []
    enough = badges_for(
        [first, second], lambda _s: Verdict("keep", "x", 5, (("Quest", 5),))
    )
    assert [(b["text"], b["fg"], b["check"]) for b in enough] == [
        ("6/5", "#4fd36b", True)
    ] * 2
    assert enough[0]["outline"] == [79, 211, 107]
    assert all(b["kind"] == "tag" for b in enough)


def test_need_tag_is_a_lower_bound_while_the_map_is_not_sure():
    slot = _square(Item(name="Metal Parts", action="Keep"), 0, stack=2)
    verdict = lambda _s: Verdict("keep", "x", 10, (("Quest", 10),))  # noqa: E731
    (short,) = badges_for([slot], verdict, {"Metal Parts": 6}, sure=False)
    assert (short["text"], short["fg"], short["check"]) == ("6+/10", "#ff5c4d", False)
    (enough,) = badges_for([slot], verdict, {"Metal Parts": 12}, sure=False)
    assert (enough["text"], enough["fg"], enough["check"]) == (
        "12+/10",
        "#4fd36b",
        True,
    )
    (sure,) = badges_for([slot], verdict, {"Metal Parts": 6}, sure=True)
    assert sure["text"] == "6/10"
    (plain,) = badges_for([slot], verdict)
    assert plain["text"] == "2/10"


def test_unsure_tag_never_shows_less_than_the_slots_on_screen():
    # Metal Parts is on screen (stack 4) but missing from the partial map
    slot = _square(Item(name="Metal Parts", action="Keep"), 0, stack=4)
    verdict = lambda _s: Verdict("keep", "x", 10, (("Quest", 10),))  # noqa: E731
    (tag,) = badges_for([slot], verdict, {"Other": 9}, sure=False)
    assert tag["text"] == "4+/10"
    (tag,) = badges_for([slot], verdict, {"Other": 9}, sure=True)
    assert tag["text"] == "0/10"


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


def _candidates(*ranked: tuple[str, float]) -> list[EmbedCandidate]:
    return [EmbedCandidate(name, name, similarity) for name, similarity in ranked]


@pytest.mark.parametrize(
    ("ranked", "read", "expected"),
    [
        # Similarity exactly MIN_SIMILARITY and just below it, with a wide lead
        ((("Anvil I", 0.65), ("Metal Parts", 0.5)), None, "Anvil I"),
        ((("Anvil I", 0.6499), ("Metal Parts", 0.5)), None, None),
        # Base lead exactly MIN_LEAD and just below it
        ((("Anvil I", 0.79), ("Metal Parts", 0.75)), None, "Anvil I"),
        ((("Anvil I", 0.79), ("Metal Parts", 0.7501)), None, None),
        # Tier siblings do not count against the lead, the next base item does
        (
            (
                ("Anvil I", 0.8),
                ("Anvil II", 0.79),
                ("Anvil III", 0.78),
                ("Ferro I", 0.7),
            ),
            None,
            "Anvil I",
        ),
        (
            (
                ("Anvil I", 0.8),
                ("Anvil II", 0.79),
                ("Anvil III", 0.78),
                ("Ferro I", 0.77),
            ),
            None,
            None,
        ),
        # A read numeral passes over an untiered candidate, not without it
        (
            (("Anvil I", 0.8), ("Amplified Anvil", 0.78), ("Ferro I", 0.72)),
            "III",
            "Anvil III",
        ),
        ((("Anvil I", 0.8), ("Amplified Anvil", 0.78), ("Ferro I", 0.72)), None, None),
        # ... but never a tiered one
        ((("Anvil I", 0.8), ("Ferro I", 0.78), ("Amplified Anvil", 0.7)), "III", None),
    ],
)
def test_the_acceptance_rule_on_hand_built_candidates(ranked, read, expected):
    found = _candidates(*ranked)
    item, _, _ = stash_scan._resolve(found, read, _catalog, {})  # noqa: SLF001
    assert (item.name if item else None) == expected


def _named(name: str) -> Item:
    return Item(name=name, action="Sell", sell_price=1000)


@pytest.mark.parametrize(
    ("ranked", "read", "blueprint", "expected"),
    [
        # An untiered item and its blueprint: the tile decides
        (
            (("Grappling Hook", 0.8), ("Metal Parts", 0.5)),
            None,
            True,
            "Grappling Hook Blueprint",
        ),
        (
            (("Grappling Hook", 0.8), ("Metal Parts", 0.5)),
            None,
            False,
            "Grappling Hook",
        ),
        ((("Grappling Hook", 0.8), ("Metal Parts", 0.5)), None, None, None),
        # The blueprint ranked first, a plain tile picks the item
        (
            (("Grappling Hook Blueprint", 0.8), ("Metal Parts", 0.5)),
            None,
            False,
            "Grappling Hook",
        ),
        # A tiered item: a blueprint tile needs no numeral, a plain one does
        ((("Anvil I", 0.8), ("Metal Parts", 0.5)), None, True, "Anvil Blueprint"),
        ((("Anvil I", 0.8), ("Metal Parts", 0.5)), "III", False, "Anvil III"),
        ((("Anvil I", 0.8), ("Metal Parts", 0.5)), None, False, None),
        # Twins that are not one item and its blueprint keep the numeral rule
        ((("Kettle", 0.8), ("Metal Parts", 0.5)), None, True, None),
        ((("Kettle", 0.8), ("Metal Parts", 0.5)), None, False, None),
    ],
)
def test_the_tile_settles_an_item_and_its_blueprint(ranked, read, blueprint, expected):
    twins = {
        "Grappling Hook": ("Grappling Hook Blueprint",),
        "Grappling Hook Blueprint": ("Grappling Hook",),
        "Anvil I": ("Anvil Blueprint",),
        "Kettle": ("Amplified Kettle", "Kettle Blueprint"),
    }
    item, tier, _ = stash_scan._resolve(  # noqa: SLF001
        _candidates(*ranked), read, _named, twins, blueprint=blueprint
    )
    assert (item.name if item else None) == expected
    if expected and expected.endswith("Blueprint"):
        assert tier is None


@pytest.mark.usefixtures("stubs")
def test_identify_slots_settles_twins_by_the_tile(monkeypatch):
    slots = _slots(1)
    gallery = EmbedGallery(
        ("hook", "metal"),
        ("Grappling Hook", "Metal Parts"),
        np.eye(2, dtype=np.float32),
    )
    twins = {"Grappling Hook": ("Grappling Hook Blueprint",)}
    monkeypatch.setattr(stash_scan, "is_blueprint_tile", lambda _crop: True)
    found = identify_slots(
        _frame(slots),
        gallery,
        _named,
        embedder=QueueEmbedder([[1, 0]]),
        slots=slots,
        twins=twins,
    )
    assert found[0].item.name == "Grappling Hook Blueprint"


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


FINGERPRINT = stash_scan.slot_cache_fingerprint(NAMES, 1000, 5)
BAR = BarGeometry(Rect(5, 70, 85, 85), (9, 12, 25), Rect(6, 72, 20, 84), None)


def _user_db(tmp_path):
    from arc_helper.database import Database

    db = Database(tmp_path / "items.db", user_db_path=tmp_path / "arlo_user.db")
    # The stored entries are looked up again by name on load
    db.lookup = _catalog
    return db


def _identified(cache, embedder):
    slots = _slots(3)
    found = identify_slots(
        _looks(slots, [60, 120, 180]),
        _gallery(),
        _catalog,
        embedder=embedder,
        slots=slots,
        read_stacks=True,
        cache=cache,
    )
    return found


@pytest.mark.usefixtures("stubs")
def test_a_saved_slot_cache_gives_the_same_hits_after_reload(monkeypatch, tmp_path):
    # Every slot has a bar, so its box is kept and saved beside the entry
    monkeypatch.setattr(stash_scan, "bar_geometry", lambda _crop: BAR)
    cache = stash_scan.SlotCache()
    first = _identified(cache, QueueEmbedder([METAL, RUBBER, ANVIL]))
    db = _user_db(tmp_path)
    db.save_slot_cache(FINGERPRINT, cache.snapshot().rows())
    loaded = stash_scan.load_slot_cache(db, FINGERPRINT)
    assert len(loaded) == 3
    assert loaded.added == 0
    again = QueueEmbedder([])
    assert _identified(loaded, again) == first
    assert again.calls == 0
    assert first[0].bar == BAR


@pytest.mark.usefixtures("stubs", "no_bars")
def test_a_stored_cache_for_another_catalog_item_count_loads_empty(tmp_path):
    cache = stash_scan.SlotCache()
    _identified(cache, QueueEmbedder([METAL, RUBBER, ANVIL]))
    db = _user_db(tmp_path)
    db.save_slot_cache(FINGERPRINT, cache.snapshot().rows())
    grown = stash_scan.slot_cache_fingerprint(NAMES, 1000, 6)
    assert len(stash_scan.load_slot_cache(db, grown)) == 0
    # The rows are still stored for the fingerprint they were saved under
    assert len(stash_scan.load_slot_cache(db, FINGERPRINT)) == 3


def _key(n: int) -> tuple:
    return (True, 90, 90, bytes([n]) * 16)


def _entry(guess: str) -> stash_scan._Cached:
    return stash_scan._Cached(None, None, None, 0.5, 0.01, guess, None)  # noqa: SLF001


def test_the_size_cap_holds_on_load_and_a_full_cache_replaces_the_oldest(tmp_path):
    cache = stash_scan.SlotCache()
    for n in range(5):
        cache.put(_key(n), _entry(str(n)))
    db = _user_db(tmp_path)
    db.save_slot_cache(FINGERPRINT, cache.snapshot().rows())
    loaded = stash_scan.load_slot_cache(db, FINGERPRINT, size=3)
    # The three most recently used stay
    assert [loaded.peek(_key(n)) for n in range(5)] == [
        None,
        None,
        _entry("2"),
        _entry("3"),
        _entry("4"),
    ]
    loaded.put(_key(9), _entry("9"))
    assert len(loaded) == 3
    assert _key(2) not in loaded
    assert _key(9) in loaded


def test_a_corrupt_row_is_skipped(tmp_path):
    cache = stash_scan.SlotCache()
    cache.put(_key(1), _entry("1"))
    cache.put(_key(2), _entry("2"))
    cache.keep_bar_box((90, 90, b"\x07" * 16), Rect(5, 70, 85, 85))
    rows = cache.snapshot().rows()
    good_value = rows[0][2]
    rows[1:1] = [
        ("entry", "not json", good_value),
        ("entry", '[true, 90, 90, "zz"]', good_value),
        ("entry", rows[0][1].replace("01", "03"), "{bad"),
        ("bar", '[90, 90, "08"]', "[1, 2]"),
        ("other", "[]", "[]"),
    ]
    db = _user_db(tmp_path)
    db.save_slot_cache(FINGERPRINT, rows)
    loaded = stash_scan.load_slot_cache(db, FINGERPRINT)
    assert len(loaded) == 2
    assert loaded.peek(_key(2)) == _entry("2")
    assert loaded.peek_bar_box((90, 90, b"\x07" * 16)) == Rect(5, 70, 85, 85)


def test_saves_are_throttled_after_the_first_and_flush_saves_the_rest():
    now = [0.0]
    saves: list[list] = []
    saver = stash_scan.SlotCacheSaver(
        saves.append, interval_s=30.0, clock=lambda: now[0]
    )
    cache = stash_scan.SlotCache()
    assert not saver.after_pass(cache)
    cache.put(_key(1), _entry("1"))
    # The first pass that adds entries saves at once
    assert saver.after_pass(cache)
    assert saver.flush(timeout=5)
    assert len(saves) == 1
    now[0] = 10.0
    cache.put(_key(2), _entry("2"))
    assert not saver.after_pass(cache)
    now[0] = 29.9
    assert not saver.after_pass(cache)
    # Still unsaved after the interval, saved by the next pass
    now[0] = 30.0
    assert saver.after_pass(cache)
    assert saver.flush(timeout=5)
    now[0] = 100.0
    assert not saver.after_pass(cache)
    now[0] = 110.0
    cache.put(_key(3), _entry("3"))
    assert saver.after_pass(cache)
    assert saver.flush(timeout=5)
    now[0] = 115.0
    cache.put(_key(4), _entry("4"))
    assert not saver.after_pass(cache)
    # The stop flush saves what the throttle held back
    assert saver.flush(timeout=5)
    assert [len(rows) for rows in saves] == [1, 2, 3, 4]
    assert saver.flush(timeout=5)
    assert len(saves) == 4


def test_a_failed_save_is_logged_and_a_stuck_one_bounds_the_flush(caplog):
    import threading

    def fail(_rows):
        msg = "disk full"
        raise OSError(msg)

    saver = stash_scan.SlotCacheSaver(fail, clock=lambda: 0.0)
    cache = stash_scan.SlotCache()
    cache.put(_key(1), _entry("1"))
    with caplog.at_level("WARNING", logger="arc_helper"):
        assert saver.after_pass(cache)
        # flush queues the unsaved entries again, which fail again
        assert not saver.flush(timeout=5)
    assert "slot cache: saving failed" in caplog.text
    release = threading.Event()
    stuck = stash_scan.SlotCacheSaver(lambda _rows: release.wait(5), clock=lambda: 0.0)
    assert stuck.after_pass(cache)
    assert not stuck.flush(timeout=0.05)
    release.set()
    assert stuck.flush(timeout=5)


def _fail_once(saves: list):
    def save(rows):
        saves.append(rows)
        if len(saves) == 1:
            msg = "database is locked"
            raise OSError(msg)

    return save


def _idle(saver) -> None:
    """Wait for the saver's thread to finish what was queued, without flushing."""
    with saver._done:  # noqa: SLF001
        assert saver._done.wait_for(lambda: not saver._busy, 5)  # noqa: SLF001


def test_a_failed_save_is_retried_by_a_later_pass_after_the_interval():
    now = [0.0]
    saves: list[list] = []
    saver = stash_scan.SlotCacheSaver(_fail_once(saves), clock=lambda: now[0])
    cache = stash_scan.SlotCache()
    cache.put(_key(1), _entry("1"))
    assert saver.after_pass(cache)
    _idle(saver)
    # Later passes only hit kept entries: nothing new, but still unsaved
    now[0] = 10.0
    assert not saver.after_pass(cache)
    now[0] = 31.0
    assert saver.after_pass(cache)
    _idle(saver)
    assert [len(rows) for rows in saves] == [1, 1]
    assert saver.flush(timeout=5)
    assert len(saves) == 2


def test_stop_flush_retries_a_failed_save():
    saves: list[list] = []
    saver = stash_scan.SlotCacheSaver(_fail_once(saves), clock=lambda: 0.0)
    cache = stash_scan.SlotCache()
    cache.put(_key(1), _entry("1"))
    assert saver.after_pass(cache)
    _idle(saver)
    assert saver.flush(timeout=5)
    assert [len(rows) for rows in saves] == [1, 1]


def test_an_oversized_number_skips_only_its_row(tmp_path):
    cache = stash_scan.SlotCache()
    cache.put(_key(1), _entry("1"))
    cache.put(_key(2), _entry("2"))
    rows = cache.snapshot().rows()
    huge = rows[0][2].replace('"confidence": 0.5', '"confidence": 1' + "0" * 400)
    assert huge != rows[0][2]
    rows[1:1] = [("entry", _encode(_key(3)), huge)]
    db = _user_db(tmp_path)
    db.save_slot_cache(FINGERPRINT, rows)
    loaded = stash_scan.load_slot_cache(db, FINGERPRINT)
    assert len(loaded) == 2
    assert loaded.peek(_key(1)) == _entry("1")
    assert loaded.peek(_key(2)) == _entry("2")
    assert _key(3) not in loaded


def _encode(key: tuple) -> str:
    return stash_scan._encode_key(key)  # noqa: SLF001


def test_need_tag_have_comes_from_the_given_owned_totals():
    sources = (("Gunsmith II", 10),)
    slot = _square(Item(name="Metal Parts", action="Keep"), 0, stack=2)
    verdict = lambda _s: Verdict("keep", "x", 10, sources)  # noqa: E731
    (tag,) = badges_for([slot], verdict, {"Metal Parts": 12})
    assert (tag["text"], tag["check"]) == ("12/10", True)
    shown = [b for b in badges_for([slot], verdict) if b["kind"] == "tag"]
    assert [t["text"] for t in shown] == ["2/10"]


# Wide slots, drawn at the 4K sizes: an equipment card 288 x 192 and a
# weapon card 608 x 402, on the cards' dark backdrop
CARD_BACKDROP = (13, 18, 31)
WEDGE_BLUE = (20, 150, 230)
PILL_FILL = stash_scan.worth_colours(640)[0]


def _hex_rgb(colour: str) -> np.ndarray:
    return np.array([int(colour[i : i + 2], 16) for i in (1, 3, 5)])


def _card(width: int = 288, height: int = 192) -> np.ndarray:
    """A wide slot: backdrop, the rarity wedge bottom left, the dark bar."""
    card = np.empty((height, width, 3), dtype=np.uint8)
    card[:] = CARD_BACKDROP
    for y in range(round(0.5 * height), round(0.77 * height)):
        reach = round(0.29 * width * (y - 0.5 * height) / (0.27 * height))
        card[y, :reach] = WEDGE_BLUE
    card[round(0.77 * height) :] = (8, 10, 18)
    return card


def _draw_pill(card: np.ndarray, corner: str) -> tuple[slice, slice]:
    """ARLO's "Φ 640" pill as badge_helper draws it on a 192 px tall slot.

    Text 33 px, pill rows 10 to 55 and 110 columns from the inset, filled
    PILL_FILL at TAG_ALPHA over the slot, its dark text in the middle rows.
    Returns the pill's rows and columns.
    """
    width = card.shape[1]
    left = 10 if corner == "tl" else width - 10 - 110
    rows, cols = slice(10, 56), slice(left, left + 110)
    alpha = int(stash_scan.TAG_ALPHA, 16) / 255
    under = card[rows, cols].astype(float)
    card[rows, cols] = np.round(alpha * _hex_rgb(PILL_FILL) + (1 - alpha) * under)
    card[22:44, left + 14 : left + 96 : 3] = (17, 17, 17)
    return rows, cols


def _augment(card: np.ndarray) -> tuple[int, int, int, int]:
    """An augment-shaped icon from 0.10 to 0.74 of the height, its box."""
    height, width = card.shape[:2]
    top, bottom = round(0.10 * height), round(0.74 * height)
    for y in range(top, bottom):
        half = round(0.12 * width + 0.06 * width * (y - top) / (bottom - top))
        card[y, width // 2 - half : width // 2 + half] = (170, 170, 175)
    card[top + 30 : top + 60, width // 2 - 15 : width // 2 + 15] = (60, 90, 200)
    return (
        top,
        bottom,
        width // 2 - round(0.18 * width),
        width // 2 + round(0.18 * width),
    )


def _near(square: np.ndarray, colour: np.ndarray, tolerance: int = 40) -> int:
    return int((np.abs(square.astype(int) - colour).max(axis=2) <= tolerance).sum())


@pytest.mark.parametrize("corner", ["tl", "tr"])
def test_a_wide_slot_is_embedded_without_arlos_price_pill(corner):
    card = _card()
    _augment(card)
    _draw_pill(card, corner)
    alpha = int(stash_scan.TAG_ALPHA, 16) / 255
    drawn = np.round(
        alpha * _hex_rgb(PILL_FILL) + (1 - alpha) * np.array(CARD_BACKDROP)
    )
    assert _near(card, drawn) > 2000
    frame = np.zeros((192, 400, 3), dtype=np.uint8)
    frame[:, 50:338] = card
    slot = Slot(Rect(50, 0, 338, 192), "equipment", filled=True)
    seen = []

    class Recording(QueueEmbedder):
        def embed(self, squares):
            seen.extend(squares)
            return super().embed(squares)

    identify_slots(
        frame,
        _gallery(),
        _catalog,
        embedder=Recording([[0, 0, 0, 1, 0]]),
        slots=[slot],
    )
    (square,) = seen
    assert square.shape == (224, 224, 3)
    assert _near(square, drawn) == 0
    # The icon's blue mark is still in the view
    assert _near(square, np.array([60, 90, 200])) > 100


def test_a_wide_slot_without_a_pill_is_left_as_it_is():
    card = _card()
    _augment(card)
    assert np.array_equal(stash_scan.unpill(card), card)


def test_unpill_keeps_the_slot_signature():
    card = _card()
    _augment(card)
    clean = stash_scan.slot_signature(card)
    _draw_pill(card, "tl")
    assert not np.array_equal(stash_scan.unpill(card), card)
    assert stash_scan.slot_signature(stash_scan.unpill(card)) == clean


def test_an_augment_card_gives_a_square_with_its_whole_icon():
    card = _card()
    top, bottom, left, right = _augment(card)
    rect = Rect(0, 0, card.shape[1], card.shape[0])
    square = stash_scan.slot_square(card, rect)
    art = np.abs(square.astype(int) - np.array(GALLERY_BACKGROUND)).max(axis=2) > 30
    rows, cols = np.flatnonzero(art.any(axis=1)), np.flatnonzero(art.any(axis=0))
    tall, wide = rows[-1] + 1 - rows[0], cols[-1] + 1 - cols[0]
    # The whole icon is boxed: its height spans the square inside the margin,
    # and its width to height ratio is the icon's, not cut
    side = 224 / (1 + 2 * ART_MARGIN)
    assert tall == pytest.approx(side, abs=3)
    assert wide / tall == pytest.approx((right - left) / (bottom - top), abs=0.03)
    # No rarity wedge in the view
    assert _near(square, np.array(WEDGE_BLUE)) == 0


def _weapon_card(*, art_rows=(0.15, 0.40), art_cols=(0.2, 0.8)) -> np.ndarray:
    card = _card(608, 402)
    height, width = card.shape[:2]
    rows = slice(round(art_rows[0] * height), round(art_rows[1] * height))
    card[rows, round(art_cols[0] * width) : round(art_cols[1] * width)] = (
        150,
        150,
        150,
    )
    for n in range(3):
        left = round((0.3 + 0.14 * n) * width)
        card[round(0.55 * height) : round(0.75 * height), left : left + 70] = (
            90,
            120,
            90,
        )
    return card


@pytest.mark.parametrize(
    "card",
    [
        _weapon_card(),
        # A hand cannon, its art less than WEAPON_ASPECT times as wide as tall
        _weapon_card(art_rows=(0.10, 0.48), art_cols=(0.25, 0.65)),
    ],
    ids=["rifle", "hand cannon"],
)
def test_a_weapon_cards_square_is_unchanged(card):
    from arc_helper.icon_embed import crop_square

    rect = Rect(0, 0, card.shape[1], card.shape[0])
    assert np.array_equal(stash_scan.slot_square(card, rect), crop_square(card))


@pytest.mark.usefixtures("stubs", "no_bars")
def test_a_cache_saved_under_the_previous_version_is_not_reused(tmp_path):
    cache = stash_scan.SlotCache()
    _identified(cache, QueueEmbedder([METAL, RUBBER, ANVIL]))
    db = _user_db(tmp_path)
    old = stash_scan.slot_cache_fingerprint(
        NAMES, 1000, 5, version=stash_scan.SLOT_CACHE_VERSION - 1
    )
    db.save_slot_cache(old, cache.snapshot().rows())
    assert len(stash_scan.load_slot_cache(db, FINGERPRINT)) == 0
    assert len(stash_scan.load_slot_cache(db, old)) == 3
