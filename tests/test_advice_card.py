"""Tests for the advice card model and renderer."""

from arc_helper.advice_card import build_advice_card
from arc_helper.advice_card import render_advice_card
from arc_helper.database import Need
from arc_helper.verdict import Verdict


def _need(kind, name, tier, quantity):
    return Need("Snitch Scanner", kind, name, tier, quantity, None, None)


NEEDS = [
    _need("workshop", "Utility Station", 2, 6),
    _need("quest", "Wires Crossed", None, 1),
]
KEEP = Verdict(
    "keep",
    "7 for Utility Station II and Wires Crossed",
    7,
    (("Utility Station II", 6), ("Wires Crossed", 1)),
)


def test_keep_with_two_sources():
    card = build_advice_card(KEEP, NEEDS, owned=2)
    assert card.title == "Keep all 7: 5 more needed"
    assert card.rows == (
        ("Workshop", "Utility Station level 2", 6),
        ("Quest", "Wires Crossed", 1),
    )
    assert (card.needed_total, card.owned) == (7, 2)


def test_keep_without_owned_count():
    card = build_advice_card(KEEP, NEEDS)
    assert card.title == "Keep 7"
    assert card.owned is None
    assert len(card.rows) == 2


def test_need_missing_from_sources_gives_no_row():
    extra = [*NEEDS, _need("project", "Expedition 5", 2, 3)]
    card = build_advice_card(KEEP, extra)
    assert [row[0] for row in card.rows] == ["Workshop", "Quest"]


def test_keep_some_title_and_subtitle():
    verdict = Verdict("keep_some", "x", 3, (("Wires Crossed", 3),))
    card = build_advice_card(verdict, [_need("quest", "Wires Crossed", None, 3)], 5)
    assert card.title == "Keep 3"
    assert card.subtitle == "then sell the rest"


def test_sell_has_no_rows():
    card = build_advice_card(Verdict("sell", "Sell it"), [])
    assert card.title == "Sell it"
    assert card.subtitle == "Nothing you track needs it"
    assert card.rows == ()


def test_recycle_subtitle_is_the_reason():
    reason = "Recycles into Magnet, which you still need"
    card = build_advice_card(Verdict("recycle", reason), [])
    assert card.title == "Recycle it"
    assert card.subtitle == reason


def test_unknown_item():
    card = build_advice_card(Verdict("unknown", "Item not in database"), [], 4)
    assert card.title == "Not in the database"
    assert card.rows == ()


def test_render_width_and_height_grow_with_rows():
    one = Verdict("keep", "r", 1, (("Wires Crossed", 1),))
    small = render_advice_card(build_advice_card(one, [NEEDS[1]], 1))
    big = render_advice_card(build_advice_card(KEEP, NEEDS, 1))
    plain = render_advice_card(build_advice_card(Verdict("sell", "s"), []))
    assert small.mode == "RGBA"
    assert small.width == big.width == plain.width == 560
    assert plain.height < small.height < big.height


def test_render_scales_and_survives_long_names():
    long_need = _need("quest", "A Very Long Quest Name " * 8, None, 2)
    verdict = Verdict("keep", "r", 2, ((long_need.source_name, 2),))
    card = build_advice_card(verdict, [long_need], 1)
    image = render_advice_card(card, 1.75)
    assert image.width == 980


def test_keep_title_when_owned_equals_needed():
    card = build_advice_card(KEEP, NEEDS, owned=7)
    assert card.title == "Keep all 7: 0 more needed"
    assert build_advice_card(KEEP, NEEDS, owned=9).title == "Keep all 7: 0 more needed"


def test_owned_footer_on_cards_without_rows():
    for kind in ("sell", "recycle"):
        verdict = Verdict(kind, "r")
        without = build_advice_card(verdict, [])
        with_owned = build_advice_card(verdict, [], owned=2)
        assert with_owned.owned == 2
        assert (
            render_advice_card(with_owned).height > render_advice_card(without).height
        )
