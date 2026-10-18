"""Tally what the loadout panels show on one frame and check it against the header.

loadout_tally counts the identified items in the equipment, backpack, quick
use, augmented slots and safe pocket panels, sums their catalog sell prices
and compares that sum with the value the LOADOUT header shows. Stash slots
are ignored. It is a pure function: the caller finds and identifies the
slots (stash_scan.identify_slots) and reads the header
(inventory_header.read_loadout_header).

stash_map_counts counts the items of the stored stash map, owned_counts
adds the latest loadout counts to them for the have/need tags.
"""

from collections.abc import Iterable
from collections.abc import Mapping
from typing import NamedTuple

from .grid import PANELS
from .grid import STASH
from .inventory_header import LoadoutHeader
from .stash_map import StoredMap
from .stash_scan import SlotItem

# The header value includes weapon mods and durability that catalog sell
# prices miss, so a computed value within 15 percent of the header counts as
# a match. A first guess, tuned from the gaps the real-data check reports
VALUE_TOLERANCE_FRACTION = 0.15

LOADOUT_PANELS = tuple(p for p in PANELS if p != STASH)


class LoadoutTally(NamedTuple):
    """The loadout panels' items, counts and value, checked against the header.

    totals maps item name to count over all loadout panels, panel_counts
    panel to item name to count. identified and unidentified map each panel
    to its filled slot counts. A panel with no slot in the input is left out
    of every dict. agrees is None without a header, reason says why.
    """

    totals: dict[str, int]
    panel_counts: dict[str, dict[str, int]]
    identified: dict[str, int]
    unidentified: dict[str, int]
    computed_value: int
    unpriced: int
    header_value: int | None
    header_weight: float | None
    header_weight_cap: float | None
    agrees: bool | None
    reason: str


def _check(
    unidentified: int, unpriced: int, computed: int, header: LoadoutHeader | None
) -> tuple[bool | None, str]:
    if header is None:
        return None, "no header"
    if unidentified:
        return False, f"{unidentified} slots unidentified"
    if unpriced:
        return False, f"{unpriced} items without a price"
    if abs(computed - header.value) > VALUE_TOLERANCE_FRACTION * header.value:
        return False, f"value {computed:,} vs header {header.value:,}"
    return True, "matches header"


def loadout_tally(
    slot_items: Iterable[SlotItem], header: LoadoutHeader | None
) -> LoadoutTally:
    """Count the loadout slots and check their value against the header.

    An identified slot counts its stack, or 1 when the stack is None. The
    computed value sums count times sell_price over identified items with a
    sell_price, unpriced counts the identified slots without one.
    """
    loadout = [s for s in slot_items if s.panel != STASH]
    panels = [p for p in LOADOUT_PANELS if any(s.panel == p for s in loadout)]
    totals: dict[str, int] = {}
    panel_counts: dict[str, dict[str, int]] = {p: {} for p in panels}
    identified = dict.fromkeys(panels, 0)
    unidentified = dict.fromkeys(panels, 0)
    computed = unpriced = 0
    for slot in loadout:
        if slot.item is None:
            unidentified[slot.panel] += 1
            continue
        identified[slot.panel] += 1
        name = slot.item.name
        count = slot.stack if slot.stack is not None else 1
        totals[name] = totals.get(name, 0) + count
        counts = panel_counts[slot.panel]
        counts[name] = counts.get(name, 0) + count
        if slot.item.sell_price is None:
            unpriced += 1
        else:
            computed += count * slot.item.sell_price
    agrees, reason = _check(sum(unidentified.values()), unpriced, computed, header)
    return LoadoutTally(
        totals,
        panel_counts,
        identified,
        unidentified,
        computed,
        unpriced,
        header.value if header else None,
        header.weight if header else None,
        header.weight_cap if header else None,
        agrees,
        reason,
    )


def stash_map_counts(stored: StoredMap) -> dict[str, int]:
    """Item name to count over the stored map's filled, identified cells.

    A cell counts its stack, or 1 when the stack is None.
    """
    counts: dict[str, int] = {}
    for cell in stored.cells.values():
        if not cell.filled or cell.name is None:
            continue
        count = cell.stack if cell.stack is not None else 1
        counts[cell.name] = counts.get(cell.name, 0) + count
    return counts


def owned_counts(
    stored: StoredMap | None, loadout: Mapping[str, int], *, sure: bool
) -> dict[str, int] | None:
    """The stored stash map's counts plus the loadout counts, by item name.

    None when there is no stored map or it is not sure (its filled count is
    not the header's used count), so the caller falls back to the slots on
    screen (stash_scan.owned_totals).
    """
    if stored is None or not sure:
        return None
    totals = stash_map_counts(stored)
    for name, count in loadout.items():
        totals[name] = totals.get(name, 0) + count
    return totals
