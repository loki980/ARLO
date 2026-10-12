#!/usr/bin/env python3
"""
Hybrid Database Updater for ARLO (Arc Raiders Loot Overlay).

Combines data from three sources for best results:
  - MetaForge API: items, sell prices, stack sizes, quests
  - arctracker.io API: recycle (home) and salvage (field) outputs, quest and
    project needs for needs.csv
  - MetaForge hideout: workshop level needs for needs.csv
  - arctracker.io and the wiki Blueprints page: blueprints.csv (catalog and
    where found) and blueprint_grid.csv (Collections grid order)
  - Arc Raiders Wiki: workshop upgrades, expedition requirements, project uses,
    and recycle outputs for items arctracker does not cover

Usage:
    python update_db.py                 # Full update (overwrites items.csv)
    python update_db.py --merge         # Merge: keep existing manual overrides
    python update_db.py --dry-run       # Preview without writing files
    python update_db.py --csv-only      # Only update items.csv, skip DB rebuild
    python update_db.py --source wiki   # Force wiki-only mode
    python update_db.py --source api    # Force MetaForge-only mode (no wiki)
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import logging
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from pathlib import Path

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print(
        "Missing dependencies. Install them with:\n"
        "  pip install requests beautifulsoup4\n"
        "  -- or --\n"
        "  uv add requests beautifulsoup4"
    )
    sys.exit(1)


# =============================================================================
# Constants
# =============================================================================

METAFORGE_API_BASE = "https://metaforge.app/api/arc-raiders"
ARCTRACKER_ITEMS_URL = "https://arctracker.io/api/items"
ARCTRACKER_TIMEOUT = 60  # one call of about 4 MB
ARCTRACKER_QUESTS_URL = "https://arctracker.io/api/quests"
ARCTRACKER_PROJECTS_URL = "https://arctracker.io/api/projects"
ARCTRACKER_HIDEOUT_URL = "https://arctracker.io/api/hideout"
METAFORGE_HIDEOUT_URL = f"{METAFORGE_API_BASE}/hideout"

WIKI_URL = "https://arcraiders.wiki/wiki/Loot"
SCRIPT_DIR = Path(__file__).parent.resolve()
ITEMS_CSV = SCRIPT_DIR / "items.csv"
ITEMS_DB = SCRIPT_DIR / "items.db"
RECYCLE_OUTPUTS_CSV = SCRIPT_DIR / "recycle_outputs.csv"
RECYCLE_OUTPUTS_FIELDS = ["item", "component", "quantity", "location"]
NEEDS_CSV = SCRIPT_DIR / "needs.csv"
NEEDS_FIELDS = [
    "item",
    "source_kind",
    "source_name",
    "tier",
    "quantity",
    "starts",
    "ends",
]
WIKI_BLUEPRINTS_URL = "https://arcraiders.wiki/w/index.php?title=Blueprints&action=raw"
BLUEPRINTS_CSV = SCRIPT_DIR / "blueprints.csv"
BLUEPRINTS_FIELDS = [
    "name",
    "item_id",
    "unlocks",
    "icon_url",
    "map",
    "map_condition",
    "scavengable",
    "containers",
    "quest_reward",
    "trials_reward",
]
BLUEPRINT_GRID_CSV = SCRIPT_DIR / "blueprint_grid.csv"
BLUEPRINT_GRID_FIELDS = ["grid_index", "name"]

ITEMS_FIELDS = [
    "name",
    "action",
    "recycle_for",
    "keep_for",
    "sell_price",
    "stack_size",
    "rarity",
    "category",
]

USER_AGENT = "ARLO/1.0 (Arc Raiders Loot Overlay - https://github.com/Soygen/ARLO)"

QUICK_USE_CATEGORIES = {"Quick Use", "Shield", "Augment", "Mods", "Ammunition", "Key"}
MATERIAL_CATEGORIES = {"Basic Material", "Refined Material", "Topside Material"}

log = logging.getLogger(__name__)


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class GameItem:
    """An item normalized to a common format."""

    name: str
    rarity: str = ""
    recycles_to: str = ""
    sell_price: int = 0
    stack_size: int = 0
    category: str = ""
    uses: str = ""  # Workshop/project/expedition uses (from wiki)
    quest_uses: str = ""  # Quest requirements (from MetaForge)

    action: str = ""
    recycle_for: str = ""
    keep_for: str = ""

    # True when recycles_to came from arctracker home outputs. Merge mode then
    # replaces the old recycle_for instead of keeping it.
    recycles_from_arctracker: bool = False


@dataclass
class ArcTrackerItem:
    """One arctracker.io item. Output maps are keyed by component English name."""

    id: str
    name: str
    value: int = 0
    stack_size: int = 0
    rarity: str = ""
    item_type: str = ""
    recycles_into: dict[str, int] = field(default_factory=dict)  # home
    salvages_into: dict[str, int] = field(default_factory=dict)  # field
    icon_url: str = ""


@dataclass
class ScraperStats:
    """Track update results."""

    items_scraped: int = 0
    items_written: int = 0
    items_preserved: int = 0
    items_new: int = 0
    source: str = ""
    errors: list[str] = field(default_factory=list)


# =============================================================================
# MetaForge API
# =============================================================================


def _api_get(url: str, *, timeout: int = 30):
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    return resp.json()


def fetch_metaforge_items() -> list[dict]:
    """Fetch all items from the MetaForge API with pagination."""
    print("Fetching items from MetaForge API...")
    all_items: list[dict] = []
    page = 1

    while True:
        data = _api_get(f"{METAFORGE_API_BASE}/items?page={page}&limit=100")
        items = data.get("data", [])
        pagination = data.get("pagination", {})
        all_items.extend(items)
        total = pagination.get("total", "?")
        print(f"  Page {page}: {len(items)} items (total: {len(all_items)}/{total})")

        if not pagination.get("hasNextPage", False):
            break
        page += 1
        time.sleep(0.1)

    print(f"  Fetched {len(all_items)} items")
    return all_items


def fetch_wiki_recycle_map() -> dict[str, str]:
    """Scrape the wiki loot table and return item_name -> recycles_to string map."""
    print("  Loading wiki recycle data...")
    try:
        html = fetch_wiki_page()
    except Exception as e:  # noqa: BLE001
        print(f"  Wiki recycle fallback also failed ({e})")
        return {}

    soup = BeautifulSoup(html, "html.parser")
    loot_table = None
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if "item" in headers and "recycles to" in headers:
            loot_table = table
            break

    if not loot_table:
        print("  Could not find recycle table on wiki")
        return {}

    header_row = loot_table.find("tr")
    headers = [th.get_text(strip=True).lower() for th in header_row.find_all("th")]
    item_idx = next((i for i, h in enumerate(headers) if h == "item"), None)
    recycle_idx = next((i for i, h in enumerate(headers) if "recycles" in h), None)

    if item_idx is None or recycle_idx is None:
        return {}

    recycle_map: dict[str, str] = {}
    for row in loot_table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) <= max(item_idx, recycle_idx):
            continue
        item_cell = cells[item_idx]
        name_link = item_cell.find("a")
        name = (
            name_link.get_text(strip=True)
            if name_link
            else item_cell.get_text(strip=True)
        )
        if not name:
            continue
        recycles_to = _parse_recycles_to(cells[recycle_idx])
        if recycles_to:
            recycle_map[name] = recycles_to

    print(f"  Loaded recycle data for {len(recycle_map)} items from wiki")
    return recycle_map


# =============================================================================
# arctracker.io API
# =============================================================================


def _english_name(raw_name) -> str:
    """Return the English entry of a localized name map, or a plain string name."""
    if isinstance(raw_name, dict):
        return str(raw_name.get("en") or "").strip()
    if isinstance(raw_name, str):
        return raw_name.strip()
    return ""


def _int_or_zero(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def parse_arctracker_items(data) -> dict[str, ArcTrackerItem]:
    """Turn an /api/items payload into a lowercase English name -> record map.

    Component ids in recyclesInto and salvagesInto are resolved to English
    names, because ids differ between sources (MetaForge wires-recipe is
    arctracker wires). When two items share an English name the first wins.
    Raises ValueError when an output references a component that has no
    record or no English name, so no raw id reaches the CSV files.
    """
    raw_items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(raw_items, list):
        msg = "arctracker payload has no items list"
        raise TypeError(msg)

    id_to_name = {
        raw["id"]: _english_name(raw.get("name"))
        for raw in raw_items
        if isinstance(raw, dict) and raw.get("id") and _english_name(raw.get("name"))
    }

    def outputs(raw_map) -> dict[str, int]:
        result: dict[str, int] = {}
        if not isinstance(raw_map, dict):
            return result
        for comp_id, qty in raw_map.items():
            amount = _int_or_zero(qty)
            if amount > 0:
                comp_name = id_to_name.get(comp_id)
                if not comp_name:
                    msg = f"arctracker component {comp_id!r} has no English name"
                    raise ValueError(msg)
                result[comp_name] = result.get(comp_name, 0) + amount
        return result

    arc_items: dict[str, ArcTrackerItem] = {}
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        name = _english_name(raw.get("name"))
        if not name or name.lower() in arc_items:
            continue
        arc_items[name.lower()] = ArcTrackerItem(
            id=str(raw.get("id") or ""),
            name=name,
            value=_int_or_zero(raw.get("value")),
            stack_size=_int_or_zero(raw.get("stackSize")),
            rarity=str(raw.get("rarity") or ""),
            item_type=str(raw.get("type") or ""),
            recycles_into=outputs(raw.get("recyclesInto")),
            salvages_into=outputs(raw.get("salvagesInto")),
            icon_url=str(raw.get("imageFilename") or ""),
        )
    return arc_items


def fetch_arctracker_items() -> dict[str, ArcTrackerItem]:
    """Fetch every item from arctracker.io in one call.

    Returns an empty map after one warning when the API is unreachable,
    answers with an error status, or sends something that is not item JSON.
    """
    print(f"Fetching items from {ARCTRACKER_ITEMS_URL} ...")
    try:
        resp = requests.get(
            ARCTRACKER_ITEMS_URL,
            timeout=ARCTRACKER_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()
        arc_items = parse_arctracker_items(resp.json())
    except (requests.RequestException, ValueError, TypeError) as e:
        print(f"  WARNING: arctracker.io unavailable ({e}), using wiki recycle data")
        return {}
    print(f"  Loaded {len(arc_items)} items from arctracker.io")
    return arc_items


def format_recycle_for(outputs: dict[str, int]) -> str:
    """Format outputs in the items.csv style, for example '2x Battery, 2x Wires'."""
    return ", ".join(f"{qty}x {name}" for name, qty in outputs.items())


def build_recycle_map(arc_items: dict[str, ArcTrackerItem]) -> dict[str, str]:
    """Build the lowercase item name -> recycle_for text map.

    arctracker home outputs win. Items without them keep the wiki Recycles To
    text, and the whole map comes from the wiki when arctracker is empty.
    """
    recycle_map = {
        name.lower(): text for name, text in fetch_wiki_recycle_map().items()
    }
    for key, record in arc_items.items():
        if record.recycles_into:
            recycle_map[key] = format_recycle_for(record.recycles_into)
    return recycle_map


def build_recycle_output_rows(
    arc_items: dict[str, ArcTrackerItem],
) -> list[dict[str, str]]:
    """List home (recyclesInto) and field (salvagesInto) outputs per item."""
    rows: list[dict[str, str]] = []
    for record in arc_items.values():
        for location, outputs in (
            ("home", record.recycles_into),
            ("field", record.salvages_into),
        ):
            rows.extend(
                {
                    "item": record.name,
                    "component": component,
                    "quantity": str(qty),
                    "location": location,
                }
                for component, qty in outputs.items()
            )
    rows.sort(key=lambda r: (r["item"].lower(), r["location"], r["component"].lower()))
    return rows


def write_recycle_outputs_csv(rows: list[dict[str, str]], csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RECYCLE_OUTPUTS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _recycle_fields(
    name: str,
    arc_items: dict[str, ArcTrackerItem],
    recycle_map: dict[str, str],
) -> tuple[str, bool]:
    """Return (recycles_to, came_from_arctracker) for one item, by English name."""
    record = arc_items.get(name.lower())
    from_arctracker = bool(record and record.recycles_into)
    return recycle_map.get(name.lower(), ""), from_arctracker


def apply_arctracker_recycle(
    items: list[GameItem], arc_items: dict[str, ArcTrackerItem]
) -> None:
    """Give wiki fallback items the arctracker home outputs where they exist."""
    for item in items:
        record = arc_items.get(item.name.lower())
        if record and record.recycles_into:
            item.recycles_to = format_recycle_for(record.recycles_into)
            item.recycles_from_arctracker = True


# =============================================================================
# MetaForge quests
# =============================================================================


def fetch_metaforge_quest_items() -> dict[str, list[str]]:
    """Fetch quests and build item_name_lower -> [quest info] map."""
    print("Fetching quests from MetaForge API...")
    all_quests: list[dict] = []
    page = 1

    while True:
        try:
            data = _api_get(f"{METAFORGE_API_BASE}/quests?page={page}&limit=100")
        except Exception as e:  # noqa: BLE001
            print(f"  Quest data unavailable ({e})")
            return {}

        quests = data.get("data", [])
        pagination = data.get("pagination", {})
        all_quests.extend(quests)

        if not pagination.get("hasNextPage", False):
            break
        page += 1
        time.sleep(0.1)

    quest_map = parse_metaforge_quests(all_quests)
    print(f"  Mapped quest requirements for {len(quest_map)} items")
    return quest_map


def parse_metaforge_quests(all_quests: list[dict]) -> dict[str, list[str]]:
    """Build item_name_lower -> ['Quest (2x)', ...] from MetaForge quest records.

    Each required_items entry has id, item, item_id and quantity. The item
    name is under item.name, there is no top-level name key.
    """
    quest_map: dict[str, list[str]] = {}
    for quest in all_quests:
        quest_name = quest.get("name", "")
        for req in quest.get("required_items") or []:
            if isinstance(req, dict):
                item = req.get("item")
                iname = item.get("name", "") if isinstance(item, dict) else ""
                qty = req.get("quantity", req.get("count", 1))
                if iname:
                    quest_map.setdefault(iname.lower(), []).append(
                        f"{quest_name} ({qty}x)"
                    )
            elif isinstance(req, str) and req:
                quest_map.setdefault(req.lower(), []).append(quest_name)
    return quest_map


def _normalize_category(item_type: str) -> str:
    mapping = {
        "basic_material": "Basic Material",
        "refined_material": "Refined Material",
        "topside_material": "Topside Material",
        "recyclable": "Recyclable",
        "trinket": "Trinket",
        "nature": "Nature",
        "quick_use": "Quick Use",
        "key": "Key",
        "ammunition": "Ammunition",
        "ammo": "Ammunition",
        "shield": "Shield",
        "augment": "Augment",
        "mods": "Mods",
        "mod": "Mods",
    }
    return mapping.get(
        item_type.lower().strip(), item_type.title() if item_type else ""
    )


# =============================================================================
# Needs: quests, projects and workshop levels (needs.csv)
# =============================================================================


class NameResolver:
    """Map source item ids to English names and count the ids that do not resolve."""

    def __init__(self, id_to_name: dict[str, str]) -> None:
        self.id_to_name = id_to_name
        self.unresolved = 0

    def name(self, item_id: str, fallback_name=None) -> str:
        """Return the English name by id, then by the entry's own name, else ''."""
        name = self.id_to_name.get(item_id) or _english_name(fallback_name)
        if not name:
            self.unresolved += 1
            log.debug("needs: skipping unresolved item id %r", item_id)
        return name


def _requirement_totals(entries, id_key: str, resolver: NameResolver) -> dict[str, int]:
    """Sum requirement entries into an English name -> quantity map."""
    totals: dict[str, int] = {}
    if not isinstance(entries, list):
        return totals
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        qty = _int_or_zero(entry.get("quantity"))
        if qty <= 0:
            continue
        name = resolver.name(str(entry.get(id_key) or ""), entry.get("name"))
        if name:
            totals[name] = totals.get(name, 0) + qty
    return totals


def _need_rows(
    totals: dict[str, int],
    source_kind: str,
    source_name: str,
    tier: int | str = "",
    starts: str = "",
    ends: str = "",
) -> list[dict[str, str]]:
    return [
        {
            "item": item,
            "source_kind": source_kind,
            "source_name": source_name,
            "tier": str(tier),
            "quantity": str(qty),
            "starts": starts,
            "ends": ends,
        }
        for item, qty in totals.items()
    ]


def _iso_date(timestamp) -> str:
    """Convert a Unix timestamp in seconds to an ISO date, '' when absent."""
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        return ""
    if timestamp <= 0:
        return ""
    return datetime.fromtimestamp(timestamp, tz=UTC).date().isoformat()


def _project_name(raw_name) -> str:
    """English project name, with 'Expedition Project (Expedition 1)' as 'Expedition 1'."""
    name = _english_name(raw_name)
    match = re.search(r"\(Expedition (\d+)\)$", name)
    return f"Expedition {match.group(1)}" if match else name


def _records(data, key: str):
    """Return the records under key as a list. The live API keys them by id."""
    records = data.get(key) if isinstance(data, dict) else None
    if isinstance(records, dict):
        return list(records.values())
    return records


def parse_quest_needs(data, resolver: NameResolver) -> list[dict[str, str]]:
    """Build quest rows from an arctracker /api/quests payload."""
    quests = _records(data, "quests")
    if not isinstance(quests, list):
        msg = "arctracker quests payload has no quests list"
        raise TypeError(msg)
    rows: list[dict[str, str]] = []
    for quest in quests:
        if not isinstance(quest, dict):
            continue
        name = _english_name(quest.get("name"))
        if name:
            totals = _requirement_totals(
                quest.get("requiredItemIds"), "itemId", resolver
            )
            rows.extend(_need_rows(totals, "quest", name))
    return rows


def parse_project_needs(data, resolver: NameResolver) -> list[dict[str, str]]:
    """Build project rows from an arctracker /api/projects payload.

    The tier is the phase number. Coin-value phases (requirementCategories)
    have no requirementItemIds and so produce no rows.
    """
    projects = _records(data, "projects")
    if not isinstance(projects, list):
        msg = "arctracker projects payload has no projects list"
        raise TypeError(msg)
    rows: list[dict[str, str]] = []
    for project in projects:
        if not isinstance(project, dict):
            continue
        name = _project_name(project.get("name"))
        if not name:
            continue
        starts = _iso_date(project.get("startDate"))
        ends = _iso_date(project.get("endDate"))
        for phase in project.get("phases") or []:
            if not isinstance(phase, dict):
                continue
            totals = _requirement_totals(
                phase.get("requirementItemIds"), "itemId", resolver
            )
            tier = _int_or_zero(phase.get("phase"))
            rows.extend(_need_rows(totals, "project", name, tier, starts, ends))
    return rows


def _station_levels(levels):
    """Yield (level number, level record) from a dict keyed by level or a list."""
    if isinstance(levels, dict):
        for key, level in levels.items():
            yield _int_or_zero(key), level
    elif isinstance(levels, list):
        for level in levels:
            if isinstance(level, dict):
                yield _int_or_zero(level.get("level")), level


WorkshopLevels = dict[tuple[str, int], dict[str, int]]


def _parse_stations(
    stations, items_key: str, id_key: str, resolver: NameResolver
) -> WorkshopLevels:
    """Return (station name, level) -> {item name: quantity}, empty levels kept."""
    levels: WorkshopLevels = {}
    for station in stations:
        if not isinstance(station, dict):
            continue
        name = _english_name(station.get("name"))
        if not name:
            continue
        for level_no, level in _station_levels(station.get("levels")):
            if level_no <= 0:
                continue
            entries = level.get(items_key) if isinstance(level, dict) else None
            levels[name, level_no] = _requirement_totals(entries, id_key, resolver)
    return levels


def parse_metaforge_hideout(data, resolver: NameResolver) -> WorkshopLevels:
    """Parse the MetaForge /hideout station list (levels keyed '1', '2', ...)."""
    stations = data.get("data") if isinstance(data, dict) else data
    if not isinstance(stations, list):
        msg = "MetaForge hideout payload has no station list"
        raise TypeError(msg)
    return _parse_stations(stations, "requiredItems", "id", resolver)


def parse_arctracker_hideout(data, resolver: NameResolver) -> WorkshopLevels:
    """Parse the arctracker /api/hideout module map (levels as a list)."""
    modules = _records(data, "hideoutModules")
    if not isinstance(modules, list):
        msg = "arctracker hideout payload has no hideoutModules map"
        raise TypeError(msg)
    return _parse_stations(modules, "requirementItemIds", "itemId", resolver)


def workshop_rows(levels: WorkshopLevels) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for (station, level_no), totals in levels.items():
        rows.extend(_need_rows(totals, "workshop", station, level_no))
    return rows


def cross_check_workshop(metaforge: WorkshopLevels, arctracker: WorkshopLevels) -> int:
    """Warn once per station level both sources list with different needs.

    Stations and items match on lowercase English name. Levels only one
    source has (Outpost, Scrappy 6, Stash) are not compared. Returns the
    number of warnings.
    """

    def lowered(levels: WorkshopLevels) -> dict[tuple[str, int], dict[str, int]]:
        return {
            (station.lower(), level_no): {k.lower(): v for k, v in totals.items()}
            for (station, level_no), totals in levels.items()
        }

    mf_levels = lowered(metaforge)
    arc_levels = lowered(arctracker)
    names = {(s.lower(), n): s for s, n in metaforge}
    warnings = 0
    for key in sorted(mf_levels.keys() & arc_levels.keys()):
        mf_totals, arc_totals = mf_levels[key], arc_levels[key]
        if mf_totals == arc_totals:
            continue
        diffs = [
            f"{item} MetaForge {mf_totals.get(item, 0)} arctracker {arc_totals.get(item, 0)}"
            for item in sorted(mf_totals.keys() | arc_totals.keys())
            if mf_totals.get(item, 0) != arc_totals.get(item, 0)
        ]
        print(
            f"  WARNING: workshop {names[key]} {key[1]} differs between sources: "
            + ", ".join(diffs)
        )
        warnings += 1
    return warnings


def _fetch_needs_source(label: str, url: str, parse):
    """Fetch and parse one needs source. Returns None after one warning on failure."""
    try:
        return parse(_api_get(url))
    except (requests.RequestException, ValueError, TypeError, AttributeError) as e:
        print(f"  WARNING: {label} unavailable ({e})")
        return None


def write_needs_csv(rows: list[dict[str, str]], csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=NEEDS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _needs_sort_key(row: dict[str, str]) -> tuple:
    tier = int(row["tier"]) if row["tier"] else -1
    return (row["source_kind"], row["source_name"].lower(), tier, row["item"].lower())


def update_needs_csv(
    arc_items: dict[str, ArcTrackerItem],
    metaforge_items: list[dict],
    csv_path: Path,
) -> bool:
    """Rebuild needs.csv from arctracker quests and projects and MetaForge hideout.

    Item ids resolve to English names through the arctracker and MetaForge
    item lists. When an item list or one of the three sources is missing, the
    existing file is left untouched so a partial fetch never deletes rows.
    """
    if not arc_items or not metaforge_items:
        print("  needs.csv left unchanged: item names unavailable")
        return False

    print("Fetching quest, project and workshop needs ...")
    arc_names = NameResolver({r.id: r.name for r in arc_items.values() if r.id})
    mf_names = NameResolver(
        {
            str(raw["id"]): _english_name(raw.get("name"))
            for raw in metaforge_items
            if isinstance(raw, dict)
            and raw.get("id")
            and _english_name(raw.get("name"))
        }
    )

    quests = _fetch_needs_source(
        "arctracker quests",
        ARCTRACKER_QUESTS_URL,
        lambda d: parse_quest_needs(d, arc_names),
    )
    projects = _fetch_needs_source(
        "arctracker projects",
        ARCTRACKER_PROJECTS_URL,
        lambda d: parse_project_needs(d, arc_names),
    )
    workshop = _fetch_needs_source(
        "MetaForge hideout",
        METAFORGE_HIDEOUT_URL,
        lambda d: parse_metaforge_hideout(d, mf_names),
    )
    if quests is None or projects is None or workshop is None:
        print("  needs.csv left unchanged")
        return False

    rows = quests + projects + workshop_rows(workshop)

    arc_workshop = _fetch_needs_source(
        "arctracker hideout (cross-check)",
        ARCTRACKER_HIDEOUT_URL,
        lambda d: parse_arctracker_hideout(d, arc_names),
    )
    if arc_workshop is not None:
        cross_check_workshop(workshop, arc_workshop)

    unresolved = arc_names.unresolved + mf_names.unresolved
    if unresolved:
        print(f"  Skipped {unresolved} needs entries with an unresolved item id")

    rows.sort(key=_needs_sort_key)
    write_needs_csv(rows, csv_path)
    print(f"  Wrote {len(rows)} rows to {csv_path}")
    return True


# =============================================================================
# Blueprints: catalog (blueprints.csv) and Collections grid (blueprint_grid.csv)
# =============================================================================

BLUEPRINT_TYPE = "Blueprint"
BLUEPRINT_SUFFIX = " Blueprint"

# Wiki table header (lowercase, references removed) -> blueprints.csv column
WIKI_TABLE_COLUMNS = {
    "blueprint name": "name",
    "map": "map",
    "map condition": "map_condition",
    "scavengable": "scavengable",
    "containers": "containers",
    "quest reward": "quest_reward",
    "trials reward": "trials_reward",
}
WIKI_COLUMNS = list(WIKI_TABLE_COLUMNS.values())[1:]

_WIKI_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_WIKI_REF = re.compile(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", re.DOTALL)
_WIKI_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
_WIKI_LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
_WIKI_GRID = re.compile(r"\{\{\s*BlueprintGrid\b(.*?)\}\}", re.DOTALL)


def _wiki_text(raw: str) -> str:
    """Plain text of a wikitext cell. Links keep their label, breaks become commas."""
    text = _WIKI_COMMENT.sub("", raw)
    text = _WIKI_REF.sub("", text)
    text = _WIKI_BREAK.sub("\n", text)
    text = _WIKI_LINK.sub(r"\1", text)
    parts = (" ".join(part.split()) for part in text.split("\n"))
    return ", ".join(part for part in parts if part)


def _plain_key(name: str) -> str:
    """Case-folded name without apostrophes, so Trigger 'Nade meets Trigger Nade."""
    return " ".join(name.replace("'", "").replace("\u2019", "").casefold().split())


def _blueprint_key(name: str) -> str:
    """Matching key of a blueprint name, with the ' Blueprint' suffix added."""
    key = _plain_key(name)
    suffix = BLUEPRINT_SUFFIX.casefold()
    return key if key.endswith(suffix) else key + suffix


def parse_blueprint_grid(wikitext: str) -> list[str | None]:
    """Entries of the {{BlueprintGrid}} template in order, None for a '?' slot.

    Raises ValueError when the page has no grid template or it is empty.
    """
    match = _WIKI_GRID.search(wikitext)
    if not match:
        msg = "wiki Blueprints page has no BlueprintGrid template"
        raise ValueError(msg)
    entries: list[str | None] = []
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        name = _wiki_text(line[1:])
        entries.append(None if name in {"", "?"} else name)
    if not entries:
        msg = "wiki BlueprintGrid template is empty"
        raise ValueError(msg)
    return entries


def _wikitable_rows(wikitext: str) -> tuple[list[str], list[list[str]]]:
    """Header cells and raw data cells of the table whose header names blueprints."""
    for block in re.findall(r"^\{\|.*?^\|\}", wikitext, re.DOTALL | re.MULTILINE):
        headers: list[str] = []
        rows: list[list[str]] = []
        cells: list[str] = []
        for line in block.splitlines()[1:]:
            if line.startswith("|}"):
                break
            if line.startswith("|-"):
                if cells:
                    rows.append(cells)
                cells = []
            elif line.startswith("!"):
                headers += [_wiki_text(h).lower() for h in line[1:].split("!!")]
            elif line.startswith("|"):
                cells += line[1:].split("||")
            elif cells:
                cells[-1] += "\n" + line
        if cells:
            rows.append(cells)
        if "blueprint name" in headers:
            return headers, rows
    msg = "wiki Blueprints page has no blueprint table"
    raise ValueError(msg)


def parse_blueprint_table(wikitext: str) -> dict[str, dict[str, str]]:
    """Wiki table rows keyed by blueprint name key, valued by blueprints.csv column.

    Raises ValueError when the table or one of its columns is missing.
    """
    headers, rows = _wikitable_rows(wikitext)
    missing = [h for h in WIKI_TABLE_COLUMNS if h not in headers]
    if missing:
        msg = f"wiki blueprint table lacks columns {missing}"
        raise ValueError(msg)
    table: dict[str, dict[str, str]] = {}
    for cells in rows:
        if len(cells) != len(headers):
            log.debug("blueprints: skipping wiki table row %r", cells)
            continue
        values = {
            WIKI_TABLE_COLUMNS[h]: _wiki_text(cell)
            for h, cell in zip(headers, cells, strict=True)
            if h in WIKI_TABLE_COLUMNS
        }
        if values["name"]:
            table.setdefault(_blueprint_key(values["name"]), values)
    return table


def _unlocked_item(name: str, arc_items: dict[str, ArcTrackerItem]) -> str:
    """English name of the item a blueprint unlocks, or the stripped name."""
    base = name.removesuffix(BLUEPRINT_SUFFIX)
    exact = arc_items.get(base.lower())
    if exact:
        return exact.name
    target = _plain_key(base)
    for record in arc_items.values():
        if record.item_type != BLUEPRINT_TYPE and _plain_key(record.name) == target:
            return record.name
    return base


def build_blueprint_rows(
    catalog: list[ArcTrackerItem],
    table: dict[str, dict[str, str]],
    arc_items: dict[str, ArcTrackerItem],
) -> list[dict[str, str]]:
    """One blueprints.csv row per catalog blueprint, sorted by name."""
    rows = []
    for record in catalog:
        wiki = table.get(_blueprint_key(record.name), {})
        row = {
            "name": record.name,
            "item_id": record.id,
            "unlocks": _unlocked_item(record.name, arc_items),
            "icon_url": record.icon_url,
        }
        row.update({column: wiki.get(column, "") for column in WIKI_COLUMNS})
        rows.append(row)
    rows.sort(key=lambda r: r["name"].lower())
    return rows


def build_blueprint_grid_rows(
    grid: list[str | None], catalog: list[ArcTrackerItem]
) -> list[dict[str, str]]:
    """blueprint_grid.csv rows in game order, an empty name for unknown slots.

    A wiki name that matches no catalog name, even case-insensitively and
    with the ' Blueprint' suffix added, is logged once and left empty.
    """
    exact = {record.name: record.name for record in catalog}
    by_key = {_blueprint_key(record.name): record.name for record in catalog}
    rows = []
    for index, wiki_name in enumerate(grid, start=1):
        name = ""
        if wiki_name is not None:
            name = exact.get(wiki_name) or by_key.get(_blueprint_key(wiki_name), "")
            if not name:
                print(
                    f"  WARNING: wiki grid slot {index} {wiki_name!r} "
                    "matches no catalog blueprint"
                )
        rows.append({"grid_index": str(index), "name": name})
    return rows


def _write_rows(rows: list[dict[str, str]], fields: list[str], csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def update_blueprint_csvs(
    arc_items: dict[str, ArcTrackerItem],
    blueprints_csv: Path,
    grid_csv: Path,
) -> bool:
    """Rebuild blueprints.csv and blueprint_grid.csv from arctracker and the wiki.

    The catalog comes from arctracker items of type Blueprint, the grid order
    and the where-found columns from the wiki Blueprints page. When either
    source is missing, both files are left untouched.
    """
    catalog = [r for r in arc_items.values() if r.item_type == BLUEPRINT_TYPE]
    if not catalog:
        if arc_items:
            print("  WARNING: arctracker items list no blueprints")
        print("  blueprint CSVs left unchanged: arctracker blueprints unavailable")
        return False

    print(f"Fetching {WIKI_BLUEPRINTS_URL} ...")
    try:
        resp = requests.get(
            WIKI_BLUEPRINTS_URL, timeout=30, headers={"User-Agent": USER_AGENT}
        )
        resp.raise_for_status()
        grid = parse_blueprint_grid(resp.text)
        table = parse_blueprint_table(resp.text)
    except (requests.RequestException, ValueError) as e:
        print(f"  WARNING: wiki Blueprints page unavailable ({e})")
        print("  blueprint CSVs left unchanged")
        return False

    rows = build_blueprint_rows(catalog, table, arc_items)
    grid_rows = build_blueprint_grid_rows(grid, catalog)
    _write_rows(rows, BLUEPRINTS_FIELDS, blueprints_csv)
    _write_rows(grid_rows, BLUEPRINT_GRID_FIELDS, grid_csv)
    unknown = [r["grid_index"] for r in grid_rows if not r["name"]]
    print(f"  Wrote {len(rows)} rows to {blueprints_csv}")
    print(
        f"  Wrote {len(grid_rows)} rows to {grid_csv} "
        f"(empty slots: {', '.join(unknown) or 'none'})"
    )
    return True


# =============================================================================
# Wiki scraper
# =============================================================================


def fetch_wiki_page() -> str:
    print(f"Fetching {WIKI_URL} ...")
    resp = requests.get(WIKI_URL, timeout=30, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    print(f"  Got {len(resp.text):,} bytes")
    return resp.text


def _parse_recycles_to(cell) -> str:
    if not cell:
        return ""
    text = cell.get_text(" ", strip=True)
    if "cannot" in text.lower() or "n/a" in text.lower():
        return ""
    text = text.replace("\u00d7", "x")
    parts = re.split(r"(?<=\S)\s+(?=\d+x\s)", text)
    return ", ".join(p.strip() for p in parts if p.strip())


def _parse_uses(cell) -> str:
    if not cell:
        return ""
    text = cell.get_text(" ", strip=True)
    if not text:
        return ""
    text = text.replace("\u00d7", "x")
    return re.sub(r"\s+", " ", text).strip()


def _parse_int_cell(cell) -> int:
    if not cell:
        return 0
    text = cell.get_text(strip=True).replace(",", "").strip()
    try:
        return int(text)
    except ValueError:
        return 0


def scrape_wiki_uses() -> dict[str, str]:
    """
    Scrape ONLY the Uses column from the wiki loot table.
    Returns a dict of item_name_lower -> uses_string.
    """
    print("Scraping wiki for workshop/project/expedition data...")
    html = fetch_wiki_page()
    soup = BeautifulSoup(html, "html.parser")

    loot_table = None
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if "item" in headers and "rarity" in headers and "recycles to" in headers:
            loot_table = table
            break

    if not loot_table:
        print("  Could not find wiki loot table")
        return {}

    header_row = loot_table.find("tr")
    headers = [th.get_text(strip=True).lower() for th in header_row.find_all("th")]

    item_col = None
    uses_col = None
    for i, h in enumerate(headers):
        if h == "item":
            item_col = i
        elif h == "uses":
            uses_col = i

    if item_col is None or uses_col is None:
        print("  Could not find Item or Uses columns")
        return {}

    uses_map: dict[str, str] = {}
    for row in loot_table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) <= max(item_col, uses_col):
            continue

        item_cell = cells[item_col]
        name_link = item_cell.find("a")
        name = (
            name_link.get_text(strip=True)
            if name_link
            else item_cell.get_text(strip=True)
        )
        if not name:
            continue

        uses = _parse_uses(cells[uses_col])
        if uses:
            uses_map[name.lower()] = uses

    print(f"  Found wiki uses for {len(uses_map)} items")
    return uses_map


def fetch_items_from_wiki() -> list[GameItem]:
    """Full wiki scrape - used as complete fallback."""
    html = fetch_wiki_page()
    soup = BeautifulSoup(html, "html.parser")

    loot_table = None
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if "item" in headers and "rarity" in headers and "recycles to" in headers:
            loot_table = table
            break

    if not loot_table:
        print("ERROR: Could not find the loot table!")
        return []

    header_row = loot_table.find("tr")
    headers = [th.get_text(strip=True).lower() for th in header_row.find_all("th")]

    col_map = {}
    for i, h in enumerate(headers):
        if h == "item":
            col_map["item"] = i
        elif h == "rarity":
            col_map["rarity"] = i
        elif "recycles" in h:
            col_map["recycles_to"] = i
        elif "sell" in h:
            col_map["sell_price"] = i
        elif "stack" in h:
            col_map["stack_size"] = i
        elif h == "category":
            col_map["category"] = i
        elif h == "uses":
            col_map["uses"] = i

    items: list[GameItem] = []
    for row in loot_table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        def get_cell(key: str, cells=cells):
            idx = col_map.get(key)
            return cells[idx] if idx is not None and idx < len(cells) else None

        item_cell = get_cell("item")
        if not item_cell:
            continue
        name_link = item_cell.find("a")
        name = (
            name_link.get_text(strip=True)
            if name_link
            else item_cell.get_text(strip=True)
        )
        if not name:
            continue

        items.append(
            GameItem(
                name=name,
                rarity=get_cell("rarity").get_text(strip=True)
                if get_cell("rarity")
                else "",
                recycles_to=_parse_recycles_to(get_cell("recycles_to")),
                sell_price=_parse_int_cell(get_cell("sell_price")),
                stack_size=_parse_int_cell(get_cell("stack_size")),
                category=get_cell("category").get_text(strip=True)
                if get_cell("category")
                else "",
                uses=_parse_uses(get_cell("uses")),
            )
        )

    print(f"  Scraped {len(items)} items from wiki")
    return items


# =============================================================================
# Hybrid fetch: MetaForge + Wiki enrichment
# =============================================================================


def fetch_items_hybrid(
    arc_items: dict[str, ArcTrackerItem] | None = None,
    raw_items: list[dict] | None = None,
) -> list[GameItem]:
    """
    Best of both worlds:
      - MetaForge API for items, prices, stack sizes, quest data
      - arctracker.io for recycle data, with the wiki for items it lacks
      - Wiki for workshop/project/expedition uses

    Pass arc_items and raw_items (the MetaForge item list) when the caller
    already fetched them, to avoid a second call.
    """
    # 1. MetaForge items + quests, arctracker recycle data
    if raw_items is None:
        raw_items = fetch_metaforge_items()
    if arc_items is None:
        arc_items = fetch_arctracker_items()
    # Recycle data used to come from MetaForge's Supabase table
    # arc_item_recycle_components. Its anon key was disabled 2026-04-02 and the
    # feed has returned 401 since, so arctracker.io replaces it.
    recycle_map = build_recycle_map(arc_items)
    quest_map = fetch_metaforge_quest_items()

    # 2. Wiki: Uses column only
    try:
        wiki_uses = scrape_wiki_uses()
    except Exception as e:  # noqa: BLE001
        print(f"  Wiki scrape failed ({e}), continuing without wiki uses")
        wiki_uses = {}

    # 3. Combine into GameItems
    items: list[GameItem] = []
    for raw in raw_items:
        name = raw.get("name", "")
        if not name:
            continue

        stat_block = raw.get("stat_block") or {}
        category = _normalize_category(raw.get("item_type", ""))
        sell_price = raw.get("value", 0) or 0
        stack_size = stat_block.get("stackSize", 0) or 0
        rarity = raw.get("rarity", "")
        recycles_to, from_arctracker = _recycle_fields(name, arc_items, recycle_map)

        # Wiki uses (workshop upgrades, expeditions, projects)
        wiki_use_str = wiki_uses.get(name.lower(), "")

        # MetaForge quest uses
        quest_uses = quest_map.get(name.lower(), [])
        quest_str = "Quests: " + ", ".join(quest_uses) if quest_uses else ""

        # Combine: wiki uses first (workshop/expedition), then quests
        uses_parts = [p for p in [wiki_use_str, quest_str] if p]
        combined_uses = "; ".join(uses_parts)

        items.append(
            GameItem(
                name=name,
                rarity=rarity,
                recycles_to=recycles_to,
                sell_price=sell_price,
                stack_size=stack_size,
                category=category,
                uses=combined_uses,
                recycles_from_arctracker=from_arctracker,
            )
        )

    print(f"\n  Processed {len(items)} items (MetaForge + Wiki enrichment)")
    return items


# =============================================================================
# Action generation
# =============================================================================


def generate_action(item: GameItem) -> tuple[str, str, str]:
    has_uses = bool(item.uses.strip())
    can_recycle = bool(item.recycles_to.strip())
    category = item.category.strip()

    keep_for = item.uses if has_uses else ""
    recycle_for = item.recycles_to

    if category == "Key":
        return "Keep", "", "Unlocks locked areas"
    if category == "Quick Use":
        return (
            ("Keep", recycle_for, keep_for)
            if has_uses
            else ("Use", recycle_for, "Quick Use consumable")
        )
    if category == "Ammunition":
        return "Keep", "", "Ammo"
    if category in {"Shield", "Augment", "Mods"}:
        return "Keep", recycle_for, f"Equippable {category}"
    if category in MATERIAL_CATEGORIES:
        return (
            ("Keep", recycle_for, keep_for)
            if has_uses
            else ("Keep", recycle_for, f"{category} - crafting ingredient")
        )
    if category == "Nature":
        return (
            ("Keep", recycle_for, keep_for) if has_uses else ("Sell", recycle_for, "")
        )
    if category == "Trinket":
        return (
            ("Keep until uses complete; sell after", "", keep_for)
            if has_uses
            else ("Sell", "", "")
        )
    if category == "Recyclable":
        if has_uses:
            return "Keep until uses complete; recycle after", recycle_for, keep_for
        if can_recycle:
            return (
                ("Recycle if short on materials; Sell otherwise", recycle_for, "")
                if item.sell_price >= 2000
                else ("Recycle", recycle_for, "")
            )
        return "Sell", "", ""

    if has_uses:
        return "Keep", recycle_for, keep_for
    if can_recycle:
        return "Recycle", recycle_for, ""
    return "Sell", "", ""


# =============================================================================
# CSV / DB operations
# =============================================================================


def load_existing_csv(csv_path: Path) -> dict[str, dict[str, str]]:
    existing: dict[str, dict[str, str]] = {}
    if not csv_path.exists():
        return existing
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("name", "").strip()
            if name:
                existing[name.lower()] = {
                    "name": name,
                    "action": row.get("action", "").strip(),
                    "recycle_for": row.get("recycle_for", "").strip(),
                    "keep_for": row.get("keep_for", "").strip(),
                    "sell_price": row.get("sell_price", "").strip(),
                    "stack_size": row.get("stack_size", "").strip(),
                }
    return existing


def build_csv_rows(
    items: list[GameItem],
    existing: dict[str, dict[str, str]] | None = None,
    *,
    merge: bool = False,
) -> tuple[list[dict[str, str]], ScraperStats]:
    stats = ScraperStats(items_scraped=len(items))
    rows: list[dict[str, str]] = []

    for item in items:
        action, recycle_for, keep_for = generate_action(item)
        key = item.name.lower()
        sell_price = str(item.sell_price) if item.sell_price else ""
        stack_size = str(item.stack_size) if item.stack_size else ""

        if merge and existing and key in existing:
            old = existing[key]
            # arctracker home outputs replace the old text. Everything else
            # keeps the existing merge rule.
            if not (item.recycles_from_arctracker and recycle_for):
                recycle_for = old["recycle_for"] or recycle_for
            rows.append(
                {
                    "name": item.name,
                    "action": old["action"],
                    "recycle_for": recycle_for,
                    "keep_for": old["keep_for"] or keep_for,
                    "sell_price": sell_price,
                    "stack_size": stack_size,
                    "rarity": item.rarity,
                    "category": item.category,
                }
            )
            stats.items_preserved += 1
        else:
            rows.append(
                {
                    "name": item.name,
                    "action": action,
                    "recycle_for": recycle_for,
                    "keep_for": keep_for,
                    "sell_price": sell_price,
                    "stack_size": stack_size,
                    "rarity": item.rarity,
                    "category": item.category,
                }
            )
            if existing and key not in (existing or {}):
                stats.items_new += 1

    rows.sort(key=lambda r: r["name"].lower())
    stats.items_written = len(rows)
    return rows, stats


def write_csv(rows: list[dict[str, str]], csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ITEMS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _sync_csv_tables(csv_dir: Path, db_path: Path) -> None:
    """Load recycle_outputs.csv, needs.csv and the blueprint CSVs into the database.

    Needs the arc_helper package. Without it (a bare python with only the
    updater's dependencies) the tables are left for the app to load on launch.
    """
    src_dir = str(SCRIPT_DIR / "src")
    if src_dir not in sys.path:
        sys.path.append(src_dir)
    try:
        from arc_helper.database import Database
    except ImportError as e:
        print(f"  Skipped loading the recycle, needs and blueprint CSVs ({e})")
        return
    Database(
        db_path,
        recycle_outputs_csv=csv_dir / RECYCLE_OUTPUTS_CSV.name,
        needs_csv=csv_dir / NEEDS_CSV.name,
        blueprints_csv=csv_dir / BLUEPRINTS_CSV.name,
        blueprint_grid_csv=csv_dir / BLUEPRINT_GRID_CSV.name,
    ).sync_from_csvs()


def rebuild_database(csv_path: Path, db_path: Path) -> int:
    count = _rebuild_items_table(csv_path, db_path)
    _sync_csv_tables(csv_path.parent, db_path)
    return count


def _rebuild_items_table(csv_path: Path, db_path: Path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS items")
        conn.execute("""
            CREATE TABLE items (
                name TEXT PRIMARY KEY NOT NULL COLLATE NOCASE,
                action TEXT NOT NULL,
                recycle_for TEXT,
                keep_for TEXT,
                sell_price INTEGER,
                stack_size INTEGER,
                rarity TEXT,
                category TEXT
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_items_name ON items(name COLLATE NOCASE)"
        )

        count = 0
        with csv_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = row.get("name", "").strip()
                action = row.get("action", "").strip()
                if not name or not action:
                    continue
                recycle_for = row.get("recycle_for", "").strip() or None
                keep_for = row.get("keep_for", "").strip() or None
                sp = row.get("sell_price", "").strip()
                ss = row.get("stack_size", "").strip()

                conn.execute(
                    "INSERT INTO items (name, action, recycle_for, keep_for, sell_price, stack_size, rarity, category) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(name) DO UPDATE SET "
                    "action=excluded.action, recycle_for=excluded.recycle_for, "
                    "keep_for=excluded.keep_for, sell_price=excluded.sell_price, stack_size=excluded.stack_size, "
                    "rarity=excluded.rarity, category=excluded.category",
                    (
                        name,
                        action,
                        recycle_for,
                        keep_for,
                        int(sp) if sp else None,
                        int(ss) if ss else None,
                        (row.get("rarity") or "").strip() or None,
                        (row.get("category") or "").strip() or None,
                    ),
                )
                count += 1

        conn.commit()
        return count
    finally:
        conn.close()


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update items.csv, recycle_outputs.csv, needs.csv, the blueprint CSVs and items.db from MetaForge, arctracker.io and the Arc Raiders Wiki.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python update_db.py               # Hybrid update (MetaForge + Wiki)
  python update_db.py --merge       # Keep existing manual overrides
  python update_db.py --dry-run     # Preview changes without writing
  python update_db.py --source wiki # Wiki-only mode
  python update_db.py --source api  # MetaForge-only mode (no wiki)
        """,
    )
    parser.add_argument(
        "--merge", action="store_true", help="Preserve existing action overrides"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without writing files"
    )
    parser.add_argument(
        "--csv-only", action="store_true", help="Update CSV only, skip DB rebuild"
    )
    parser.add_argument(
        "--source",
        choices=["hybrid", "api", "wiki"],
        default="hybrid",
        help="Data source: 'hybrid' (default), 'api' MetaForge only, 'wiki' only",
    )

    args = parser.parse_args()

    arc_items: dict[str, ArcTrackerItem] = {}
    metaforge_items: list[dict] = []
    if args.source == "wiki":
        items = fetch_items_from_wiki()
        source_name = "Arc Raiders Wiki"
    elif args.source == "api":
        arc_items = fetch_arctracker_items()
        metaforge_items = fetch_metaforge_items()
        items = fetch_items_from_metaforge_only(arc_items, metaforge_items)
        source_name = "MetaForge API"
    else:
        arc_items = fetch_arctracker_items()
        try:
            metaforge_items = fetch_metaforge_items()
            items = fetch_items_hybrid(arc_items, metaforge_items)
            source_name = "MetaForge API + Wiki"
        except Exception as e:  # noqa: BLE001
            print(f"\nHybrid fetch failed ({e}), falling back to wiki only...")
            items = fetch_items_from_wiki()
            apply_arctracker_recycle(items, arc_items)
            source_name = "Arc Raiders Wiki (fallback)"

    if not items:
        print("No items found. Aborting.")
        sys.exit(1)

    existing = None
    if args.merge:
        existing = load_existing_csv(ITEMS_CSV)
        print(f"  Loaded {len(existing)} existing items for merge")

    rows, stats = build_csv_rows(items, existing, merge=args.merge)
    stats.source = source_name

    print(f"\n{'=' * 50}")
    print(f"  Source:                  {stats.source}")
    print(f"  Items fetched:           {stats.items_scraped}")
    print(f"  Items to write:          {stats.items_written}")
    if args.merge:
        print(f"  Existing preserved:      {stats.items_preserved}")
        print(f"  New items added:         {stats.items_new}")
    print(f"{'=' * 50}")

    if args.dry_run:
        print("\n[DRY RUN] Preview of first 20 items:\n")
        print(
            f"  {'Name':<35} {'Action':<40} {'Sell':>7} {'Stack':>5}  {'Recycle For':<25}"
        )
        print(f"  {'-' * 35} {'-' * 40} {'-' * 7} {'-' * 5}  {'-' * 25}")
        for row in rows[:20]:
            print(
                f"  {row['name']:<35} {row['action']:<40} {row.get('sell_price', ''):>7} {row.get('stack_size', ''):>5}  {row['recycle_for']:<25}"
            )
        print(f"\n  ... and {len(rows) - 20} more items")
        return

    write_csv(rows, ITEMS_CSV)
    print(f"\n  Wrote {len(rows)} items to {ITEMS_CSV}")

    if arc_items:
        output_rows = build_recycle_output_rows(arc_items)
        write_recycle_outputs_csv(output_rows, RECYCLE_OUTPUTS_CSV)
        print(f"  Wrote {len(output_rows)} rows to {RECYCLE_OUTPUTS_CSV}")

    update_needs_csv(arc_items, metaforge_items, NEEDS_CSV)
    update_blueprint_csvs(arc_items, BLUEPRINTS_CSV, BLUEPRINT_GRID_CSV)

    if not args.csv_only:
        db_count = rebuild_database(ITEMS_CSV, ITEMS_DB)
        print(f"  Rebuilt {ITEMS_DB} with {db_count} items")

    print("\nDone!")


def fetch_items_from_metaforge_only(
    arc_items: dict[str, ArcTrackerItem] | None = None,
    raw_items: list[dict] | None = None,
) -> list[GameItem]:
    """MetaForge-only mode (no wiki uses). Recycle data as in hybrid mode."""
    if raw_items is None:
        raw_items = fetch_metaforge_items()
    if arc_items is None:
        arc_items = fetch_arctracker_items()
    recycle_map = build_recycle_map(arc_items)
    quest_map = fetch_metaforge_quest_items()

    items: list[GameItem] = []
    for raw in raw_items:
        name = raw.get("name", "")
        if not name:
            continue
        stat_block = raw.get("stat_block") or {}
        quest_uses = quest_map.get(name.lower(), [])
        quest_str = "Quests: " + ", ".join(quest_uses) if quest_uses else ""
        recycles_to, from_arctracker = _recycle_fields(name, arc_items, recycle_map)

        items.append(
            GameItem(
                name=name,
                rarity=raw.get("rarity", ""),
                recycles_to=recycles_to,
                recycles_from_arctracker=from_arctracker,
                sell_price=raw.get("value", 0) or 0,
                stack_size=stat_block.get("stackSize", 0) or 0,
                category=_normalize_category(raw.get("item_type", "")),
                uses=quest_str,
            )
        )
    print(f"  Processed {len(items)} items from MetaForge")
    return items


# =============================================================================
# Auto-update: called by the main app on startup
# =============================================================================

LAST_UPDATE_FILE = SCRIPT_DIR / ".last_wiki_update"
AUTO_UPDATE_INTERVAL_HOURS = 24


def _should_auto_update() -> bool:
    if not LAST_UPDATE_FILE.exists():
        return True
    try:
        last_ts = float(LAST_UPDATE_FILE.read_text(encoding="utf-8").strip())
        return (time.time() - last_ts) / 3600 >= AUTO_UPDATE_INTERVAL_HOURS
    except (ValueError, OSError):
        return True


def _mark_updated() -> None:
    with contextlib.suppress(OSError):
        LAST_UPDATE_FILE.write_text(str(time.time()), encoding="utf-8")


def auto_update(*, force: bool = False) -> bool:
    """
    Run a database update if due. Called by the main app on startup.
    Uses hybrid mode (MetaForge + Wiki) with merge, throttled to 24h.
    Falls through gracefully on any error.
    """
    if not force and not _should_auto_update():
        return False

    try:
        print("Checking for item database updates...")
        arc_items = fetch_arctracker_items()
        metaforge_items: list[dict] = []
        try:
            metaforge_items = fetch_metaforge_items()
            items = fetch_items_hybrid(arc_items, metaforge_items)
        except Exception:  # noqa: BLE001
            print("  Hybrid fetch failed, trying wiki only...")
            try:
                items = fetch_items_from_wiki()
            except Exception:  # noqa: BLE001
                print("  All sources unavailable, skipping update.")
                return False
            apply_arctracker_recycle(items, arc_items)

        if not items:
            return False

        existing = load_existing_csv(ITEMS_CSV)
        rows, stats = build_csv_rows(items, existing, merge=True)

        write_csv(rows, ITEMS_CSV)
        if arc_items:
            write_recycle_outputs_csv(
                build_recycle_output_rows(arc_items), RECYCLE_OUTPUTS_CSV
            )
        update_needs_csv(arc_items, metaforge_items, NEEDS_CSV)
        update_blueprint_csvs(arc_items, BLUEPRINTS_CSV, BLUEPRINT_GRID_CSV)
        rebuild_database(ITEMS_CSV, ITEMS_DB)
        _mark_updated()

        new_count = stats.items_written - stats.items_preserved
        print(
            f"  Database updated: {stats.items_written} items ({new_count} new, {stats.items_preserved} preserved)"
        )

    except Exception as e:  # noqa: BLE001
        print(f"  Database update skipped (error: {e})")
        return False
    else:
        return True


if __name__ == "__main__":
    main()
