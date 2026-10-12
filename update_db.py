#!/usr/bin/env python3
"""
Hybrid Database Updater for ARLO (Arc Raiders Loot Overlay).

Combines data from three sources for best results:
  - MetaForge API: items, sell prices, stack sizes, quests
  - arctracker.io API: recycle (home) and salvage (field) outputs
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
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from dataclasses import field
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

WIKI_URL = "https://arcraiders.wiki/wiki/Loot"
SCRIPT_DIR = Path(__file__).parent.resolve()
ITEMS_CSV = SCRIPT_DIR / "items.csv"
ITEMS_DB = SCRIPT_DIR / "items.db"
RECYCLE_OUTPUTS_CSV = SCRIPT_DIR / "recycle_outputs.csv"
RECYCLE_OUTPUTS_FIELDS = ["item", "component", "quantity", "location"]

USER_AGENT = "ARLO/1.0 (Arc Raiders Loot Overlay - https://github.com/Soygen/ARLO)"

QUICK_USE_CATEGORIES = {"Quick Use", "Shield", "Augment", "Mods", "Ammunition", "Key"}
MATERIAL_CATEGORIES = {"Basic Material", "Refined Material", "Topside Material"}


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
) -> list[GameItem]:
    """
    Best of both worlds:
      - MetaForge API for items, prices, stack sizes, quest data
      - arctracker.io for recycle data, with the wiki for items it lacks
      - Wiki for workshop/project/expedition uses

    Pass arc_items when the caller already fetched them, to avoid a second call.
    """
    # 1. MetaForge items + quests, arctracker recycle data
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
                }
            )
            if existing and key not in (existing or {}):
                stats.items_new += 1

    rows.sort(key=lambda r: r["name"].lower())
    stats.items_written = len(rows)
    return rows, stats


def write_csv(rows: list[dict[str, str]], csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "action",
                "recycle_for",
                "keep_for",
                "sell_price",
                "stack_size",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def rebuild_database(csv_path: Path, db_path: Path) -> int:
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
                stack_size INTEGER
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
                    "INSERT INTO items (name, action, recycle_for, keep_for, sell_price, stack_size) "
                    "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(name) DO UPDATE SET "
                    "action=excluded.action, recycle_for=excluded.recycle_for, "
                    "keep_for=excluded.keep_for, sell_price=excluded.sell_price, stack_size=excluded.stack_size",
                    (
                        name,
                        action,
                        recycle_for,
                        keep_for,
                        int(sp) if sp else None,
                        int(ss) if ss else None,
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
        description="Update items.csv, recycle_outputs.csv and items.db from MetaForge, arctracker.io and the Arc Raiders Wiki.",
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
    if args.source == "wiki":
        items = fetch_items_from_wiki()
        source_name = "Arc Raiders Wiki"
    elif args.source == "api":
        arc_items = fetch_arctracker_items()
        items = fetch_items_from_metaforge_only(arc_items)
        source_name = "MetaForge API"
    else:
        arc_items = fetch_arctracker_items()
        try:
            items = fetch_items_hybrid(arc_items)
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

    if not args.csv_only:
        db_count = rebuild_database(ITEMS_CSV, ITEMS_DB)
        print(f"  Rebuilt {ITEMS_DB} with {db_count} items")

    print("\nDone!")


def fetch_items_from_metaforge_only(
    arc_items: dict[str, ArcTrackerItem] | None = None,
) -> list[GameItem]:
    """MetaForge-only mode (no wiki uses). Recycle data as in hybrid mode."""
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
        try:
            items = fetch_items_hybrid(arc_items)
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
