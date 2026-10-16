"""Tests for the loadout tally, its header check and the stored snapshot."""

from arc_helper.database import Database
from arc_helper.database import Item
from arc_helper.grid import BACKPACK
from arc_helper.grid import EQUIPMENT
from arc_helper.grid import SAFE_POCKET
from arc_helper.grid import STASH
from arc_helper.inventory_header import LoadoutHeader
from arc_helper.inventory_ledger import loadout_tally
from arc_helper.stash_scan import SlotItem
from arc_helper.tooltip_layout import Rect

RECT = Rect(0, 0, 10, 10)
AMMO = Item(name="Light Ammo", action="Keep", sell_price=10)
RIFLE = Item(name="Stitcher I", action="Keep", sell_price=1000)
FREE = Item(name="Rusty Gear", action="Recycle")


def _slot(panel, item, stack=None):
    name = item.name if item else None
    return SlotItem(RECT, panel, item, None, stack, 0.9, 0.1, name)


def _items():
    """82 ammo plus 1 unread ammo in the backpack, a rifle, a stash slot."""
    return [
        _slot(EQUIPMENT, RIFLE),
        _slot(BACKPACK, AMMO, 82),
        _slot(BACKPACK, AMMO),
        _slot(STASH, RIFLE, 5),
        _slot(STASH, None),
    ]


def _header(value):
    return LoadoutHeader(39.5, 90.0, value)


def test_tally_counts_stacks_and_ignores_the_stash():
    tally = loadout_tally(_items(), _header(1830))
    assert tally.totals == {"Stitcher I": 1, "Light Ammo": 83}
    assert tally.panel_counts == {
        EQUIPMENT: {"Stitcher I": 1},
        BACKPACK: {"Light Ammo": 83},
    }
    assert tally.identified == {EQUIPMENT: 1, BACKPACK: 2}
    assert tally.unidentified == {EQUIPMENT: 0, BACKPACK: 0}
    assert tally.computed_value == 1830
    assert tally.unpriced == 0
    assert (tally.header_weight, tally.header_weight_cap) == (39.5, 90.0)


def test_agrees_when_identified_and_within_tolerance():
    tally = loadout_tally(_items(), _header(2000))  # 1830 is 8.5 percent off
    assert tally.agrees is True
    assert tally.reason == "matches header"


def test_disagrees_with_an_unidentified_loadout_slot():
    tally = loadout_tally([*_items(), _slot(SAFE_POCKET, None)], _header(1830))
    assert tally.unidentified[SAFE_POCKET] == 1
    assert tally.agrees is False
    assert tally.reason == "1 slots unidentified"


def test_disagrees_with_an_unpriced_item():
    tally = loadout_tally([*_items(), _slot(SAFE_POCKET, FREE)], _header(1830))
    assert tally.unpriced == 1
    assert tally.agrees is False
    assert tally.reason == "1 items without a price"


def test_disagrees_beyond_the_tolerance():
    tally = loadout_tally(_items(), _header(2200))  # 1830 is 16.8 percent off
    assert tally.agrees is False
    assert tally.reason == "value 1,830 vs header 2,200"


def test_no_header_gives_no_verdict():
    tally = loadout_tally(_items(), None)
    assert tally.agrees is None
    assert tally.reason == "no header"
    assert tally.header_value is None


def test_save_then_load_replaces_the_snapshot(tmp_path):
    db = Database(db_path=tmp_path / "items.db")
    assert db.loadout_counts() == {}
    db.save_loadout(loadout_tally(_items(), _header(1830)))
    assert db.loadout_counts() == {"Stitcher I": 1, "Light Ammo": 83}
    db.save_loadout(loadout_tally([_slot(SAFE_POCKET, AMMO, 3)], None))
    assert db.loadout_counts() == {"Light Ammo": 3}
    assert (tmp_path / "arlo_user.db").exists()
