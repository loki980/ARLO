"""
Verdict engine for ARLO.

Turns an item, the quests, projects and workshop levels that need it, and the
player's recorded progress into one piece of advice with a reason, such as
"3 for Gunsmith IV, then sell". Pure: no database, clock or settings access.
"""

import datetime
from collections.abc import Iterable
from collections.abc import Mapping
from typing import NamedTuple

from .database import Item
from .database import Need
from .database import parse_recycle_for
from .tiers import split_tier

KEEP = "keep"
KEEP_SOME = "keep_some"
SELL = "sell"
RECYCLE = "recycle"
USE = "use"
LEARN = "learn"
UNKNOWN = "unknown"
KINDS = (KEEP, KEEP_SOME, SELL, RECYCLE, USE, LEARN, UNKNOWN)

# Action string keywords, resolved by whichever appears first in the string
ACTION_KEYWORDS = (SELL, RECYCLE, KEEP, USE)

# At or above this sell price the fallback after a need is sell, not recycle
RECYCLE_PRICE_LIMIT = 2000

# Sources named in a keep reason, the rest still count toward keep_count
MAX_REASON_SOURCES = 3

ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}

WEAPON_CATEGORY = "weapon"
BLUEPRINT_CATEGORY = "blueprint"
UNKNOWN_REASON = "Item not in database"
KNOWN_BLUEPRINT_REASON = "You already know this blueprint"
UNLEARNED_REASON = "You have not learned this blueprint yet"
MAYBE_LEARNED_REASON = "Learn it unless you already know it"


class Verdict(NamedTuple):
    """Advice for one item.

    kind is one of KINDS. keep_count is how many to keep, or None when the
    advice is not a count. sources lists the active (source label, quantity)
    pairs behind a keep, largest first.
    """

    kind: str
    reason: str
    keep_count: int | None = None
    sources: tuple[tuple[str, int], ...] = ()


def roman(tier: int) -> str:
    """Roman numeral for a workshop tier, plain digits past VI."""
    return ROMAN.get(tier, str(tier))


def source_label(need: Need) -> str:
    """How a reason names a need: Gunsmith IV, Expedition 5 phase 2, a quest."""
    if need.tier is None or need.source_kind == "quest":
        return need.source_name
    if need.source_kind == "workshop":
        return f"{need.source_name} {roman(need.tier)}"
    return f"{need.source_name} phase {need.tier}"


def _ended(ends: str | None, today: datetime.date) -> bool:
    """True when an ISO end date lies before today. Unparsable means open."""
    if not ends:
        return False
    try:
        return datetime.date.fromisoformat(ends) < today
    except ValueError:
        return False


def _progress_index(
    progress: Mapping[tuple[str, str], int],
) -> dict[tuple[str, str], int]:
    """Progress keyed with case-folded source names."""
    return {(kind, name.casefold()): value for (kind, name), value in progress.items()}


def _is_active(
    need: Need,
    done: Mapping[tuple[str, str], int],
    today: datetime.date,
    *,
    show_ended: bool,
) -> bool:
    """Whether a need still counts given progress and the date.

    done is _progress_index output. A quest counts until marked done, a
    project phase until that phase is done or the project ended (unless
    show_ended), a workshop level while above the recorded level (0 when
    none). A need without a tier is never removed by progress.
    """
    value = done.get((need.source_kind, need.source_name.casefold()))
    if need.source_kind == "quest":
        return not value
    if need.source_kind == "project":
        if not show_ended and _ended(need.ends, today):
            return False
        return need.tier is None or value is None or need.tier > value
    if need.source_kind == "workshop":
        return need.tier is None or need.tier > (value or 0)
    return True


def fallback_kind(item: Item, home_outputs: Iterable | None = None) -> str:
    """Sell or recycle once the needs are covered.

    Recycle when the item has home recycle outputs and sells for under
    RECYCLE_PRICE_LIMIT, otherwise sell. home_outputs defaults to the
    item's recycle_for string.
    """
    outputs = (
        list(home_outputs)
        if home_outputs is not None
        else parse_recycle_for(item.recycle_for)
    )
    cheap = item.sell_price is not None and item.sell_price < RECYCLE_PRICE_LIMIT
    return RECYCLE if outputs and cheap else SELL


def kind_from_action(action: str) -> str:
    """The keyword that appears earliest in an action string, or unknown.

    "Keep until upgrade is complete; sell once done" resolves to keep.
    """
    lowered = action.lower()
    found = [(lowered.find(k), k) for k in ACTION_KEYWORDS if k in lowered]
    return min(found)[1] if found else UNKNOWN


def verdict_from_action(item: Item | None) -> Verdict:
    """A verdict from the item's action string alone, or unknown."""
    if item is None:
        return Verdict(UNKNOWN, UNKNOWN_REASON)
    return Verdict(kind_from_action(item.action), item.action)


def is_weapon(item: Item, blueprint: str | None) -> bool:
    """A weapon by category, or a tiered name with a Blueprint sibling."""
    if (item.category or "").strip().lower() == WEAPON_CATEGORY:
        return True
    return blueprint is not None and split_tier(item.name)[1] is not None


def is_blueprint(item: Item) -> bool:
    """A blueprint by its catalog type."""
    return (item.category or "").strip().lower() == BLUEPRINT_CATEGORY


def compute_verdict(
    item: Item | None,
    needs: Iterable[Need],
    progress: Mapping[tuple[str, str], int],
    owned: int | None,
    today: datetime.date,
    show_ended: bool,  # noqa: FBT001
    *,
    blueprint: str | None = None,
    home_outputs: Iterable | None = None,
    learned: bool | None = None,
    recorded_learned: bool = False,
    owned_blueprints: Iterable[str] = (),
) -> Verdict:
    """Advice for an item, applying the rules in order.

    A blueprint (catalog type Blueprint) comes first. It counts as learned
    when the check mark read on its hovered slot says so (learned True or
    False), or, when the mark could not be read (learned None), when
    recorded_learned says so. One not learned gives learn. A learned one
    goes through the rules below, its reason led by "You already know this
    blueprint".

    0. A weapon is kept, at least one, and more while its blueprint (the
       catalog name of the "<base> Blueprint" entry, when one exists) is not
       in owned_blueprints (names compare without regard to case).
    1. Needs are filtered to the active ones (see _is_active). Progress
       names match without regard to case.
    2. Active needs give keep with their summed quantity and up to three
       sources, then the sell or recycle fallback. When owned covers the
       count the kind is keep_some.
    3. Without active needs the item's action string decides.
    4. An unknown item gives unknown.

    owned is the stack count when known, None otherwise. home_outputs
    overrides the item's recycle_for string for the fallback choice.
    """
    if item is None:
        return Verdict(UNKNOWN, UNKNOWN_REASON)

    known_names = {n.casefold() for n in owned_blueprints}
    if is_blueprint(item):
        if learned is False:
            return Verdict(LEARN, UNLEARNED_REASON)
        if learned is None and not recorded_learned:
            return Verdict(LEARN, MAYBE_LEARNED_REASON)
        known = _item_verdict(
            item,
            needs,
            progress,
            owned,
            today,
            show_ended,
            blueprint,
            home_outputs,
            known_names,
        )
        return known._replace(reason=f"{KNOWN_BLUEPRINT_REASON}. {known.reason}")

    return _item_verdict(
        item,
        needs,
        progress,
        owned,
        today,
        show_ended,
        blueprint,
        home_outputs,
        known_names,
    )


def _item_verdict(
    item: Item,
    needs: Iterable[Need],
    progress: Mapping[tuple[str, str], int],
    owned: int | None,
    today: datetime.date,
    show_ended: bool,  # noqa: FBT001
    blueprint: str | None,
    home_outputs: Iterable | None,
    known_blueprints: set[str],
) -> Verdict:
    """compute_verdict's rules 0 to 3 for a known item."""
    if is_weapon(item, blueprint):
        reason = "Weapon, keep at least one"
        if blueprint and blueprint.casefold() in known_blueprints:
            reason += ", you know its blueprint"
        elif blueprint:
            reason += f", more until you own the {blueprint}"
        return Verdict(KEEP, reason, 1)

    done = _progress_index(progress)
    active = [n for n in needs if _is_active(n, done, today, show_ended=show_ended)]
    if not active:
        return verdict_from_action(item)

    active.sort(key=lambda n: n.quantity, reverse=True)
    sources = tuple((source_label(n), n.quantity) for n in active)
    keep_count = sum(n.quantity for n in active)
    fallback = fallback_kind(item, home_outputs)

    if owned is not None and owned >= keep_count:
        reason = f"You have {owned}, need {keep_count}, {fallback} the rest"
        return Verdict(KEEP_SOME, reason, keep_count, sources)

    named = ", ".join(
        f"{qty} for {label}" for label, qty in sources[:MAX_REASON_SOURCES]
    )
    return Verdict(KEEP, f"{named}, then {fallback}", keep_count, sources)
