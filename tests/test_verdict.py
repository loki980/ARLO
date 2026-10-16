"""Table-driven tests for the verdict engine and the popup title choice."""

import datetime

import pytest

from arc_helper.advice_card import KIND_COLORS
from arc_helper.advice_card import build_advice_card
from arc_helper.database import Item
from arc_helper.database import Need
from arc_helper.overlay import ACTION_COLORS
from arc_helper.overlay import action_color
from arc_helper.overlay import action_word
from arc_helper.overlay import popup_title
from arc_helper.verdict import KEEP
from arc_helper.verdict import LEARN
from arc_helper.verdict import SELL
from arc_helper.verdict import Verdict
from arc_helper.verdict import compute_verdict

TODAY = datetime.date(2026, 10, 11)

# Sells for 3000 with a home output: the fallback is sell
PRESS = Item(
    name="Radial Press",
    action="Keep",
    recycle_for="1x Advanced Mechanical Components",
    sell_price=3000,
)
# Sells for 250 with a home output: the fallback is recycle
BATTERY = Item(
    name="Battery", action="Keep", recycle_for="2x Metal Parts", sell_price=250
)
# Sells for 250 without recycle outputs: the fallback is sell
CHEAP_NO_OUTPUTS = Item(name="Fabric", action="Sell", sell_price=250)
UPGRADE_ITEM = Item(
    name="Toaster",
    action="Keep until upgrade is complete; sell once done",
    sell_price=1000,
)
ANVIL = Item(name="Anvil IV", action="Recycle", sell_price=13000, category="Weapon")
KETTLE = Item(name="Kettle I", action="Recycle", sell_price=840, category="Weapon")


def _need(item, kind, name, tier, quantity, ends=None):
    return Need(item, kind, name, tier, quantity, None, ends)


GUNSMITH_IV = _need("Radial Press", "workshop", "Gunsmith", 4, 3)
QUEST = _need("Battery", "quest", "After Rain Comes", None, 2)
EXPEDITION_5 = _need("Battery", "project", "Expedition 5", 2, 4, "2026-09-22")
SHELTERED = _need("Battery", "project", "Sheltered Retreat", 1, 5, "2049-12-31")

CASES = [
    pytest.param(
        PRESS, [GUNSMITH_IV], {("workshop", "Gunsmith"): 3}, None, False,
        Verdict("keep", "3 for Gunsmith IV, then sell", 3, (("Gunsmith IV", 3),)),
        id="workshop need above the recorded level",
    ),
    pytest.param(
        PRESS, [GUNSMITH_IV], {("workshop", "Gunsmith"): 4}, None, False,
        Verdict("keep", "Keep", None),
        id="workshop need at the recorded level is filtered",
    ),
    pytest.param(
        BATTERY, [QUEST], {("quest", "After Rain Comes"): 1}, None, False,
        Verdict("keep", "Keep", None),
        id="quest done in progress",
    ),
    pytest.param(
        BATTERY, [SHELTERED], {("project", "Sheltered Retreat"): 1}, None, False,
        Verdict("keep", "Keep", None),
        id="project phase done",
    ),
    pytest.param(
        BATTERY, [EXPEDITION_5], {}, None, False,
        Verdict("keep", "Keep", None),
        id="ended project hidden",
    ),
    pytest.param(
        BATTERY, [EXPEDITION_5], {}, None, True,
        Verdict(
            "keep", "4 for Expedition 5 phase 2, then recycle", 4,
            (("Expedition 5 phase 2", 4),),
        ),
        id="ended project shown with show_ended",
    ),
    pytest.param(
        BATTERY,
        [QUEST, _need("Battery", "quest", "Trash Into Treasure", None, 1),
         SHELTERED, _need("Battery", "workshop", "Scrappy", 3, 3)],
        {}, None, False,
        Verdict(
            "keep",
            "5 for Sheltered Retreat phase 1, 3 for Scrappy III, "
            "2 for After Rain Comes, then recycle",
            11,
            (("Sheltered Retreat phase 1", 5), ("Scrappy III", 3),
             ("After Rain Comes", 2), ("Trash Into Treasure", 1)),
        ),
        id="three sources ordered by quantity",
    ),
    pytest.param(
        CHEAP_NO_OUTPUTS, [_need("Fabric", "quest", "A Quest", None, 2)],
        {}, None, False,
        Verdict("keep", "2 for A Quest, then sell", 2, (("A Quest", 2),)),
        id="fallback sell without recycle outputs",
    ),
    pytest.param(
        BATTERY, [QUEST], {}, 5, False,
        Verdict(
            "keep_some", "You have 5, need 2, recycle the rest", 2,
            (("After Rain Comes", 2),),
        ),
        id="keep_some when owned covers the need",
    ),
    pytest.param(
        UPGRADE_ITEM, [], {}, None, False,
        Verdict("keep", "Keep until upgrade is complete; sell once done", None),
        id="no needs falls back to the action string, keep before sell",
    ),
    pytest.param(
        None, [], {}, None, False,
        Verdict("unknown", "Item not in database", None),
        id="unknown item",
    ),
]  # fmt: skip


@pytest.mark.parametrize(
    ("item", "needs", "progress", "owned", "show_ended", "want"), CASES
)
def test_compute_verdict(item, needs, progress, owned, show_ended, want):
    assert compute_verdict(item, needs, progress, owned, TODAY, show_ended) == want


WEAPON_CASES = [
    pytest.param(
        ANVIL, "Anvil Blueprint",
        Verdict(
            "keep",
            "Weapon, keep at least one, more until you own the Anvil Blueprint",
            1,
        ),
        id="weapon with a blueprint entry",
    ),
    pytest.param(
        KETTLE, None, Verdict("keep", "Weapon, keep at least one", 1),
        id="weapon without a blueprint entry",
    ),
]  # fmt: skip


@pytest.mark.parametrize(("item", "blueprint", "want"), WEAPON_CASES)
def test_weapon_is_never_sold_or_recycled(item, blueprint, want):
    needs = [_need(item.name, "quest", "A Quest", None, 2)]
    got = compute_verdict(
        item, needs, {}, None, TODAY, show_ended=False, blueprint=blueprint
    )
    assert got == want


def test_popup_title_is_catalog_name_for_known_item():
    assert popup_title("ANVILIV", ANVIL) == "Anvil IV"
    assert popup_title("ANVILIV", None) == "ANVILIV"


# A blueprint that sells by its action string, and one a quest needs
GRIP_BLUEPRINT = Item(
    name="Vertical Grip III Blueprint",
    action="Sell",
    sell_price=5000,
    category="Blueprint",
)
BLUEPRINT_QUEST = _need("Vertical Grip III Blueprint", "quest", "Paper Trail", None, 1)


def _blueprint_verdict(*, learned, recorded, needs=()):
    return compute_verdict(
        GRIP_BLUEPRINT,
        list(needs),
        {},
        None,
        TODAY,
        show_ended=False,
        learned=learned,
        recorded_learned=recorded,
    )


def test_checked_blueprint_gives_the_usual_verdict_led_by_known():
    assert _blueprint_verdict(learned=True, recorded=False) == Verdict(
        SELL, "You already know this blueprint. Sell"
    )


def test_checked_blueprint_still_kept_for_a_need():
    verdict = _blueprint_verdict(learned=True, recorded=False, needs=[BLUEPRINT_QUEST])
    assert verdict.kind == KEEP
    assert (
        verdict.reason
        == "You already know this blueprint. 1 for Paper Trail, then sell"
    )


def test_unchecked_blueprint_is_learned_even_when_recorded():
    assert _blueprint_verdict(learned=False, recorded=True) == Verdict(
        LEARN, "You have not learned this blueprint yet"
    )


def test_unread_blueprint_recorded_as_owned_takes_the_known_path():
    assert _blueprint_verdict(learned=None, recorded=True) == Verdict(
        SELL, "You already know this blueprint. Sell"
    )


def test_unread_blueprint_not_recorded_is_learned_with_a_hedge():
    assert _blueprint_verdict(learned=None, recorded=False) == Verdict(
        LEARN, "Learn it unless you already know it"
    )


def test_learn_word_and_colours():
    assert action_word(LEARN) == "Learn it"
    assert action_color(LEARN) == ACTION_COLORS["LEARN"]
    assert ACTION_COLORS["LEARN"] not in {
        v for k, v in ACTION_COLORS.items() if k != "LEARN"
    }
    card = build_advice_card(
        Verdict(LEARN, "You have not learned this blueprint yet"), []
    )
    assert (card.title, card.subtitle) == (
        "Learn it",
        "You have not learned this blueprint yet",
    )
    assert KIND_COLORS[LEARN] not in {v for k, v in KIND_COLORS.items() if k != LEARN}


def test_weapon_with_an_owned_blueprint_drops_the_more_until_clause():
    got = compute_verdict(
        ANVIL,
        [],
        {},
        None,
        TODAY,
        show_ended=False,
        blueprint="Anvil Blueprint",
        owned_blueprints={"anvil blueprint"},
    )
    assert got == Verdict(
        "keep", "Weapon, keep at least one, you know its blueprint", 1
    )


def test_weapon_whose_blueprint_is_not_owned_keeps_the_more_until_clause():
    got = compute_verdict(
        ANVIL,
        [],
        {},
        None,
        TODAY,
        show_ended=False,
        blueprint="Anvil Blueprint",
        owned_blueprints={"Stiletto Blueprint"},
    )
    assert got.reason.endswith("more until you own the Anvil Blueprint")


def test_unknown_item_ignores_owned_blueprints():
    got = compute_verdict(
        None, [], {}, None, TODAY, show_ended=False, owned_blueprints={"x"}
    )
    assert got == Verdict("unknown", "Item not in database", None)
