"""Tests for the advice card model and renderer."""

from arc_helper.advice_card import _Canvas  # noqa: PLC2701
from arc_helper.advice_card import build_advice_card
from arc_helper.advice_card import needed_parts
from arc_helper.advice_card import owned_text
from arc_helper.advice_card import render_advice_card
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
    assert card.subtitle == "For Wires Crossed, then sell 2"


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


def test_a_lower_bound_says_so_in_the_title_and_the_line(monkeypatch):
    card = build_advice_card(KEEP, NEEDS, owned=2, owned_sure=False)
    assert card.title == "Keep all 7: up to 5 more needed"
    assert (card.owned, card.owned_sure) == (2, False)
    covered = build_advice_card(KEEP, NEEDS, owned=9, owned_sure=False)
    assert covered.title == "Keep all 7: 0 more needed"
    assert owned_text(2, sure=False) == "2+"
    assert owned_text(2, sure=True) == "2"
    shown = []
    real = _Canvas.text

    def record(self, x, y, text, *args):
        shown.append(text)
        return real(self, x, y, text, *args)

    monkeypatch.setattr(_Canvas, "text", record)
    render_advice_card(card)
    assert "2+ of 7" in shown
    assert "You have 2+" in shown
    shown.clear()
    render_advice_card(build_advice_card(KEEP, NEEDS, owned=2))
    assert "2 of 7" in shown


def _texts(monkeypatch, card):
    shown = []
    real = _Canvas.text

    def record(self, x, y, text, *args):
        shown.append(text)
        return real(self, x, y, text, *args)

    monkeypatch.setattr(_Canvas, "text", record)
    render_advice_card(card)
    return shown


SOME = Verdict("keep_some", "x", 7, (("Utility Station II", 6), ("Wires Crossed", 1)))


def test_owned_above_the_need_is_capped_in_the_bar_and_says_how_many_can_go(
    monkeypatch,
):
    card = build_advice_card(SOME, NEEDS, owned=52)
    assert card.subtitle.endswith("then sell 45")
    shown = _texts(monkeypatch, card)
    assert "7 of 7" in shown
    assert "You have 52" in shown
    # An unsure count cannot say how many can go
    unsure = build_advice_card(SOME, NEEDS, owned=52, owned_sure=False)
    assert unsure.subtitle.endswith("then sell the rest")


def test_an_item_the_partial_map_has_not_seen_shows_no_number(monkeypatch):
    card = build_advice_card(KEEP, NEEDS, owned=0, owned_sure=False)
    assert card.title == "Keep 7"
    shown = _texts(monkeypatch, card)
    assert "7 in total" in shown
    assert "Not in your saved stash yet" in shown
    assert not any("0+" in t for t in shown)
    # A known count of 1 or more keeps the N+ wording, a sure 0 is a real 0
    assert "1+ of 7" in _texts(
        monkeypatch, build_advice_card(KEEP, NEEDS, owned=1, owned_sure=False)
    )
    assert build_advice_card(KEEP, NEEDS, owned=0).title == "Keep all 7: 7 more needed"


def test_a_weapon_title_ignores_the_count_and_the_footer_shows_it(monkeypatch):
    weapon = Verdict("keep", "Weapon, keep at least one, more until you own it", 1)
    card = build_advice_card(weapon, [], owned=2)
    assert card.title == "Keep 1"
    assert "You have 2" in _texts(monkeypatch, card)
    assert build_advice_card(weapon, [], owned=2, owned_sure=False).title == "Keep 1"
