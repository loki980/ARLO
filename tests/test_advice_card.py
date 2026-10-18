"""Tests for the advice card model and renderer."""

from arc_helper.advice_card import NO_PUBLIC_DATA
from arc_helper.advice_card import NOT_FOUND_YET
from arc_helper.advice_card import UNNAMED_SLOT
from arc_helper.advice_card import build_advice_card
from arc_helper.advice_card import build_blueprint_card
from arc_helper.advice_card import needed_parts
from arc_helper.advice_card import render_advice_card
from arc_helper.database import Blueprint
from arc_helper.database import Need
from arc_helper.database import RecycleOutput
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
    assert card.subtitle == "For Wires Crossed, then sell the rest"


def test_keep_subtitle_names_all_goals_in_row_order():
    card = build_advice_card(KEEP, NEEDS)
    assert card.subtitle == "For Utility Station level 2 and Wires Crossed"
    needs = [*NEEDS, _need("project", "Expedition 5", 2, 1)]
    verdict = KEEP._replace(sources=(*KEEP.sources, ("Expedition 5 phase 2", 1)))
    card = build_advice_card(verdict, needs)
    assert card.subtitle == (
        "For Utility Station level 2, Wires Crossed and Expedition 5 phase 2"
    )


def test_keep_without_sources_gives_the_verdict_reason():
    reason = "Weapon, keep at least one, more until you own the Anvil Blueprint"
    card = build_advice_card(Verdict("keep", reason, 1), [])
    assert card.rows == ()
    assert card.subtitle == reason
    assert build_advice_card(Verdict("keep", ""), []).subtitle == (
        "Kept by ARLO's rules"
    )


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


def _out(component, quantity=1, location="home"):
    return RecycleOutput(component, quantity, location)


RECYCLE = Verdict("recycle", "Recycle if short on materials; Sell otherwise")


def test_recycle_subtitle_one_needed_output_first():
    outs = [_out("Magnet"), _out("Advanced Mechanical Components")]
    card = build_advice_card(
        RECYCLE, [], None, outs, frozenset({"Advanced Mechanical Components"})
    )
    assert card.subtitle == (
        "Recycles into Advanced Mechanical Components, Magnet, which you still need"
    )


def test_recycle_subtitle_none_needed_and_field_ignored():
    outs = [_out("Magnet"), _out("Wires", 1, "field")]
    card = build_advice_card(RECYCLE, [], None, outs)
    assert card.subtitle == "Recycles into Magnet"


def test_recycle_subtitle_falls_back_without_outputs():
    assert build_advice_card(RECYCLE, []).subtitle == RECYCLE.reason
    only_field = [_out("Wires", 1, "field")]
    assert build_advice_card(RECYCLE, [], None, only_field).subtitle == RECYCLE.reason


def test_recycle_subtitle_quantities_above_one():
    outs = [_out("Magnet", 2), _out("Wires", 1)]
    card = build_advice_card(Verdict("use", "x"), [], None, outs)
    assert card.subtitle == "Recycles into 2 Magnet, Wires"


def test_long_recycle_subtitle_is_capped_at_two_lines():
    def card_for(count):
        outs = [_out(f"Component Number {i}", 3) for i in range(count)]
        return build_advice_card(
            RECYCLE, [], None, outs, frozenset({"Component Number 9"})
        )

    ten = render_advice_card(card_for(10))
    thirty = render_advice_card(card_for(30))
    assert thirty.width == 560
    assert ten.height == thirty.height
    assert thirty.height < 200


def test_needed_parts_uses_the_callables():
    outs = [_out("Magnet"), _out("Wires")]
    got = needed_parts(outs, lambda c: [c], lambda _c, needs: needs == ["Magnet"])
    assert got == frozenset({"Magnet"})


def test_known_blueprint_sell_card_says_you_know_it():
    verdict = Verdict("sell", "You already know this blueprint. Sell")
    assert build_advice_card(verdict, []).subtitle == "You already know this blueprint"


def test_known_blueprint_keep_card_adds_the_needs_line_after():
    verdict = Verdict(
        "keep",
        "You already know this blueprint. 1 for Wires Crossed, then sell",
        1,
        (("Wires Crossed", 1),),
    )
    card = build_advice_card(verdict, [_need("quest", "Wires Crossed", None, 1)])
    assert card.subtitle == "You already know this blueprint. For Wires Crossed"


def test_plain_sell_card_is_unchanged():
    card = build_advice_card(Verdict("sell", "Sell"), [])
    assert card.subtitle == "Nothing you track needs it"


def _blueprint(**where):
    fields = dict.fromkeys(Blueprint._fields[4:], "")
    fields.update(where)
    return Blueprint("Bobcat Blueprint", "bobcat_blueprint", "Bobcat", None, **fields)


def test_blueprint_card_with_full_where_found_data():
    info = _blueprint(
        map="Condition Only",
        map_condition="Locked Gate, Hurricane",
        scavengable="Yes",
        containers="Anywhere, First Wave Cache",
        quest_reward="Yes, Sparks Fly",
        trials_reward="Trophy Only",
    )
    card = build_blueprint_card(12, "Bobcat Blueprint", info)
    assert (card.kind, card.title, card.subtitle) == (
        "learn",
        "Bobcat Blueprint",
        NOT_FOUND_YET,
    )
    assert card.rows == (
        ("Map", "Only during a map condition"),
        ("Condition", "Locked Gate, Hurricane"),
        ("Containers", "Anywhere, First Wave Cache"),
        ("Scavengable", "Yes"),
        ("Quest reward", "Sparks Fly"),
        ("Trials reward", "Trophy Only"),
    )
    render_advice_card(card)


def test_blueprint_card_with_partial_data_shows_only_what_is_known():
    # Burletta's row: N/A and ? are unknown, a No reward is no place to look
    info = _blueprint(
        map="N/A",
        map_condition="?",
        scavengable="No",
        containers="N/A",
        quest_reward="Yes, Industrial Espionage",
        trials_reward="No",
    )
    card = build_blueprint_card(12, "Bobcat Blueprint", info)
    assert card.subtitle == NOT_FOUND_YET
    assert card.rows == (
        ("Scavengable", "No"),
        ("Quest reward", "Industrial Espionage"),
    )


def test_blueprint_card_with_no_where_found_data():
    for info in (_blueprint(), None):
        card = build_blueprint_card(12, "Bobcat Blueprint", info)
        assert (card.title, card.subtitle, card.rows) == (
            "Bobcat Blueprint",
            NO_PUBLIC_DATA,
            (),
        )
    assert render_advice_card(card).size[0] == 560


def test_blueprint_card_for_an_unnamed_slot():
    card = build_blueprint_card(26, None, None)
    assert (card.title, card.subtitle, card.rows) == (
        "Blueprint slot 26",
        UNNAMED_SLOT,
        (),
    )
    assert UNNAMED_SLOT == "Unnamed slot, not in the public data"
