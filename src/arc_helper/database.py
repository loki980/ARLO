"""
Database module for ARLO.
SQLite operations for item recommendations with Pydantic models.
"""

import csv
import difflib
import re
import sqlite3
import string
from collections.abc import Iterable
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import NamedTuple

from pydantic import BaseModel
from pydantic import Field

from .config import get_settings
from .config import logger
from .tiers import TierVote
from .tiers import resolve_tier
from .tiers import split_read
from .tiers import split_tier
from .tiers import tier_key

if TYPE_CHECKING:
    from .inventory_ledger import LoadoutTally

HOME = "home"
FIELD = "field"
LOCATIONS = (HOME, FIELD)
NEED_KINDS = ("quest", "project", "workshop")
RECYCLE_OUTPUTS_CSV_NAME = "recycle_outputs.csv"
NEEDS_CSV_NAME = "needs.csv"
PROGRESS_CSV_NAME = "progress.csv"
BLUEPRINTS_CSV_NAME = "blueprints.csv"
BLUEPRINT_GRID_CSV_NAME = "blueprint_grid.csv"
USER_DB_NAME = "arlo_user.db"

# "2x Metal Parts", "2 X Metal Parts", "2xMetal Parts"
_COUNTED_ENTRY = re.compile(r"(\d+)\s*[xX]\s*(.*)")

# SQLite's NOCASE collation folds ASCII letters only
_NOCASE = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)

# A stroke-count tier vote at least this confident overrides the read tier
TIER_VOTE_MIN_CONFIDENCE = 0.7

# The tolerant lookup ignores reads shorter than this many key characters
TOLERANT_MIN_LENGTH = 4


def _within_one_edit(a: str, b: str) -> bool:
    """True when a and b differ by at most one insertion, deletion or substitution."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    if len(a) == len(b):
        return a[i + 1 :] == b[i + 1 :]
    return a[i:] == b[i + 1 :]


def _match_key(name: str) -> str:
    """Lowercase, alphanumeric-only key for OCR-tolerant name matching.

    OCR frequently drops spaces ("IL TORO II" -> "ILTOROII") or appends stray
    punctuation, so collapsing to alphanumerics lets those still match the
    stored name. Roman-numeral tiers (I/V) survive, so variants stay distinct.
    """
    return "".join(c for c in name.lower() if c.isalnum())


PREFIX_MIN_KEY = 6


def _boundary_key(read: str) -> tuple[str, list[bool]]:
    """The read's match key and, per key character, whether a word ends there.

    A word ends after a key character when the next read character is not
    alphanumeric or the read ends.
    """
    key: list[str] = []
    ends: list[bool] = []
    for i, c in enumerate(read.lower()):
        if not c.isalnum():
            continue
        key.append(c)
        ends.append(i + 1 >= len(read) or not read[i + 1].isalnum())
    return "".join(key), ends


class Item(BaseModel):
    """Item model matching database schema."""

    name: str = Field(..., min_length=1, max_length=200)
    action: str = Field(..., min_length=1, max_length=100)
    recycle_for: str | None = Field(default=None, max_length=500)
    keep_for: str | None = Field(default=None, max_length=500)
    sell_price: int | None = Field(default=None)
    stack_size: int | None = Field(default=None)
    rarity: str | None = Field(default=None)
    category: str | None = Field(default=None)


class RecycleOutput(NamedTuple):
    """One component an item recycles into, at home or in the field."""

    component: str
    quantity: int
    location: str


class Need(NamedTuple):
    """One quest, project or workshop level that needs an item."""

    item: str
    source_kind: str
    source_name: str
    tier: int | None
    quantity: int
    starts: str | None
    ends: str | None


class Blueprint(NamedTuple):
    """One catalog blueprint, with where the wiki says it is found."""

    name: str
    item_id: str | None
    unlocks: str | None
    icon_url: str | None
    map: str | None
    map_condition: str | None
    scavengable: str | None
    containers: str | None
    quest_reward: str | None
    trials_reward: str | None


BLUEPRINT_COLUMNS = ", ".join(Blueprint._fields)


def _optional_int(text: str | None) -> int | None:
    """Parse a CSV cell into an int, or None when empty."""
    text = (text or "").strip()
    return int(text) if text else None


def _optional_text(text: str | None) -> str | None:
    """Strip a CSV cell, or None when empty."""
    return (text or "").strip() or None


def read_recycle_outputs_csv(path: Path) -> list[tuple[str, str, int, str]]:
    """Parse recycle_outputs.csv into (item, component, quantity, location) rows.

    Malformed rows and repeats of a key already seen (matched with the same
    ASCII case folding as the table's NOCASE key) are skipped with a debug log.
    """
    rows: list[tuple[str, str, int, str]] = []
    seen: set[tuple[str, str, str]] = set()
    with Path(path).open(encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            item = (raw.get("item") or "").strip()
            component = (raw.get("component") or "").strip()
            location = (raw.get("location") or "").strip()
            try:
                quantity = int((raw.get("quantity") or "").strip())
            except ValueError:
                quantity = 0
            key = (item.translate(_NOCASE), component.translate(_NOCASE), location)
            if (
                not item
                or not component
                or quantity <= 0
                or location not in LOCATIONS
                or key in seen
            ):
                logger.debug(f"Skipping recycle_outputs.csv row: {raw!r}")
                continue
            seen.add(key)
            rows.append((item, component, quantity, location))
    return rows


def read_needs_csv(path: Path) -> list[Need]:
    """Parse needs.csv into Need rows.

    Malformed rows and repeats of a key already seen are skipped with a
    debug log.
    """
    rows: list[Need] = []
    seen: set[tuple] = set()
    with Path(path).open(encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            try:
                need = Need(
                    item=(raw.get("item") or "").strip(),
                    source_kind=(raw.get("source_kind") or "").strip(),
                    source_name=(raw.get("source_name") or "").strip(),
                    tier=_optional_int(raw.get("tier")),
                    quantity=int((raw.get("quantity") or "").strip()),
                    starts=_optional_text(raw.get("starts")),
                    ends=_optional_text(raw.get("ends")),
                )
            except ValueError:
                logger.debug(f"Skipping needs.csv row: {raw!r}")
                continue
            key = (
                need.item.translate(_NOCASE),
                need.source_kind,
                need.source_name,
                need.tier,
            )
            if (
                not need.item
                or not need.source_name
                or need.source_kind not in NEED_KINDS
                or key in seen
            ):
                logger.debug(f"Skipping needs.csv row: {raw!r}")
                continue
            seen.add(key)
            rows.append(need)
    return rows


def read_progress_csv(path: Path) -> list[tuple[str, str, int]]:
    """Parse progress.csv into (source_kind, source_name, value) rows.

    value is the workshop level reached, 1 for a finished quest, or the
    highest project phase done. Malformed rows, negative values and repeats
    of a key already seen are skipped with a debug log.
    """
    rows: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str]] = set()
    with Path(path).open(encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            kind = (raw.get("source_kind") or "").strip()
            name = (raw.get("source_name") or "").strip()
            try:
                value = int((raw.get("value") or "").strip())
            except ValueError:
                value = -1
            key = (kind, name.translate(_NOCASE))
            if kind not in NEED_KINDS or not name or value < 0 or key in seen:
                logger.debug(f"Skipping progress.csv row: {raw!r}")
                continue
            seen.add(key)
            rows.append((kind, name, value))
    return rows


def read_blueprints_csv(path: Path) -> list[Blueprint]:
    """Parse blueprints.csv into Blueprint rows.

    Rows without a name and repeats of a name already seen are skipped with
    a debug log. Empty cells become None.
    """
    rows: list[Blueprint] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            name = (raw.get("name") or "").strip()
            key = name.translate(_NOCASE)
            if not name or key in seen:
                logger.debug(f"Skipping blueprints.csv row: {raw!r}")
                continue
            seen.add(key)
            rows.append(
                Blueprint(
                    name,
                    *(_optional_text(raw.get(f)) for f in Blueprint._fields[1:]),
                )
            )
    return rows


def read_blueprint_grid_csv(path: Path) -> list[tuple[int, str | None]]:
    """Parse blueprint_grid.csv into (grid_index, name or None) rows.

    Rows whose grid_index is not a positive integer and repeats of an index
    already seen are skipped with a debug log.
    """
    rows: list[tuple[int, str | None]] = []
    seen: set[int] = set()
    with Path(path).open(encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            try:
                index = int((raw.get("grid_index") or "").strip())
            except ValueError:
                index = 0
            if index < 1 or index in seen:
                logger.debug(f"Skipping blueprint_grid.csv row: {raw!r}")
                continue
            seen.add(index)
            rows.append((index, _optional_text(raw.get("name"))))
    return rows


def parse_recycle_for(text: str | None) -> list[tuple[str, int]]:
    """Parse a recycle_for string into (component, quantity) pairs.

    Entries are comma-separated "Nx Name". The x may be upper or lower case
    with optional whitespace around it, and an entry without a count means
    one. Malformed entries are skipped with a debug log, never raised.
    """
    if not text:
        return []

    outputs: list[tuple[str, int]] = []
    for raw in text.split(","):
        entry = raw.strip()
        if not entry:
            continue
        match = _COUNTED_ENTRY.fullmatch(entry)
        if match:
            quantity, component = int(match.group(1)), match.group(2).strip()
        elif entry[0].isalpha():
            component, quantity = entry, 1
        else:
            component, quantity = "", 0  # e.g. a count without the x
        if quantity <= 0 or not any(c.isalpha() for c in component):
            logger.debug(f"Skipping malformed recycle_for entry: {entry!r}")
            continue
        outputs.append((component, quantity))
    return outputs


class Database:
    """SQLite database handler for items."""

    def __init__(
        self,
        db_path: Path | None = None,
        *,
        recycle_outputs_csv: Path | None = None,
        needs_csv: Path | None = None,
        progress_csv: Path | None = None,
        user_db_path: Path | None = None,
        blueprints_csv: Path | None = None,
        blueprint_grid_csv: Path | None = None,
    ):
        """Initialize database connection.

        The CSV paths and the user database (arlo_user.db, the player's own
        state) default to the configured ones when db_path is not given, and
        to the files next to db_path otherwise. The user database is opened
        only on first use, and items.db is never written for user data.
        """
        if db_path is None:
            settings = get_settings()
            db_path = settings.database_path
            default_outputs = settings.recycle_outputs_csv_path
            default_needs = settings.needs_csv_path
            default_progress = settings.progress_csv_path
            default_user_db = settings.user_database_path
            default_blueprints = settings.blueprints_csv_path
            default_grid = settings.blueprint_grid_csv_path
        else:
            default_outputs = Path(db_path).parent / RECYCLE_OUTPUTS_CSV_NAME
            default_needs = Path(db_path).parent / NEEDS_CSV_NAME
            default_progress = Path(db_path).parent / PROGRESS_CSV_NAME
            default_user_db = Path(db_path).parent / USER_DB_NAME
            default_blueprints = Path(db_path).parent / BLUEPRINTS_CSV_NAME
            default_grid = Path(db_path).parent / BLUEPRINT_GRID_CSV_NAME
        self.db_path = db_path
        self.recycle_outputs_csv = Path(recycle_outputs_csv or default_outputs)
        self.needs_csv = Path(needs_csv or default_needs)
        self.progress_csv = Path(progress_csv or default_progress)
        self.user_db_path = Path(user_db_path or default_user_db)
        self.blueprints_csv = Path(blueprints_csv or default_blueprints)
        self.blueprint_grid_csv = Path(blueprint_grid_csv or default_grid)
        self._user_db_ready = False
        self._tier_index: dict[str, dict[str, str]] = {}
        self._name_keys: list[tuple[str, str]] = []
        self._init_db()
        self._build_tier_index()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS items (
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
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_items_name
                ON items(name COLLATE NOCASE)
            """)

            # Migrate: add new columns if upgrading from older schema
            existing_cols = {
                row[1] for row in conn.execute("PRAGMA table_info(items)").fetchall()
            }
            if "sell_price" not in existing_cols:
                conn.execute("ALTER TABLE items ADD COLUMN sell_price INTEGER")
            if "stack_size" not in existing_cols:
                conn.execute("ALTER TABLE items ADD COLUMN stack_size INTEGER")
            if "rarity" not in existing_cols:
                conn.execute("ALTER TABLE items ADD COLUMN rarity TEXT")
            if "category" not in existing_cols:
                conn.execute("ALTER TABLE items ADD COLUMN category TEXT")

            # Structured recycle outputs. When recycle_outputs.csv exists,
            # sync_from_csvs loads every row from it. Otherwise home rows are
            # derived from items.recycle_for on every open, because update_db
            # writes items directly and never calls load_csv, and field rows
            # come from elsewhere and are kept.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recycle_outputs (
                    item TEXT COLLATE NOCASE,
                    component TEXT COLLATE NOCASE,
                    quantity INTEGER,
                    location TEXT CHECK (location IN ('home', 'field')),
                    PRIMARY KEY (item, component, location)
                )
            """)
            self._refresh_home_outputs(conn)

            # What each quest, project and workshop level needs, from needs.csv
            conn.execute("""
                CREATE TABLE IF NOT EXISTS needs (
                    item TEXT COLLATE NOCASE,
                    source_kind TEXT CHECK (
                        source_kind IN ('quest', 'project', 'workshop')
                    ),
                    source_name TEXT,
                    tier INTEGER NULL,
                    quantity INTEGER,
                    starts TEXT NULL,
                    ends TEXT NULL,
                    PRIMARY KEY (item, source_kind, source_name, tier)
                )
            """)

            # Blueprint catalog and Collections grid order, from the CSVs
            conn.execute("""
                CREATE TABLE IF NOT EXISTS blueprints (
                    name TEXT PRIMARY KEY NOT NULL COLLATE NOCASE,
                    item_id TEXT,
                    unlocks TEXT,
                    icon_url TEXT,
                    map TEXT,
                    map_condition TEXT,
                    scavengable TEXT,
                    containers TEXT,
                    quest_reward TEXT,
                    trials_reward TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS blueprint_grid (
                    grid_index INTEGER PRIMARY KEY,
                    name TEXT NULL COLLATE NOCASE
                )
            """)

            conn.commit()

    def _refresh_home_outputs(self, conn: sqlite3.Connection) -> None:
        """Derive the home rows from items.recycle_for unless the CSV exists.

        recycle_outputs.csv is authoritative when present and sync_from_csvs
        loads it, so the derivation does not run then.
        """
        if not self.recycle_outputs_csv.exists():
            self._rebuild_home_outputs(conn)

    @staticmethod
    def _derived_home_outputs(conn: sqlite3.Connection) -> list[tuple[str, str, int]]:
        """Home rows as a rebuild would write them, in insertion order.

        Repeated components of one item add up, matched with the same ASCII
        case folding as the table's NOCASE key, and keep their first spelling.
        """
        rows: dict[tuple[str, str], list] = {}
        for row in conn.execute("SELECT name, recycle_for FROM items").fetchall():
            for component, quantity in parse_recycle_for(row["recycle_for"]):
                key = (row["name"].translate(_NOCASE), component.translate(_NOCASE))
                if key in rows:
                    rows[key][2] += quantity
                else:
                    rows[key] = [row["name"], component, quantity]
        return [
            (item, component, quantity) for item, component, quantity in rows.values()
        ]

    @classmethod
    def _rebuild_home_outputs(cls, conn: sqlite3.Connection) -> None:
        """Replace all home recycle rows with ones parsed from items.recycle_for.

        Writes nothing when the stored home rows already match, so opening
        an unchanged database leaves the file untouched.
        """
        stored = [
            (row["item"], row["component"], row["quantity"])
            for row in conn.execute(
                "SELECT item, component, quantity FROM recycle_outputs "
                "WHERE location = ? ORDER BY rowid",
                (HOME,),
            )
        ]
        derived = cls._derived_home_outputs(conn)
        if stored == derived:
            return
        conn.execute("DELETE FROM recycle_outputs WHERE location = ?", (HOME,))
        conn.executemany(
            "INSERT INTO recycle_outputs (item, component, quantity, location) "
            "VALUES (?, ?, ?, ?)",
            [
                (item, component, quantity, HOME)
                for item, component, quantity in derived
            ],
        )

    def load_recycle_outputs_csv(self, path: Path | str) -> bool:
        """Replace all recycle_outputs rows, home and field, with the CSV rows.

        Writes nothing when the stored rows already match. Returns True when
        the table changed.
        """
        rows = read_recycle_outputs_csv(Path(path))
        with self._get_connection() as conn:
            stored = [
                tuple(row)
                for row in conn.execute(
                    "SELECT item, component, quantity, location "
                    "FROM recycle_outputs ORDER BY rowid"
                )
            ]
            if stored == rows:
                return False
            conn.execute("DELETE FROM recycle_outputs")
            conn.executemany(
                "INSERT INTO recycle_outputs (item, component, quantity, location) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        return True

    def load_needs_csv(self, path: Path | str) -> bool:
        """Replace all needs rows with the CSV rows.

        Writes nothing when the stored rows already match. Returns True when
        the table changed.
        """
        rows = read_needs_csv(Path(path))
        with self._get_connection() as conn:
            stored = [
                Need._make(row)
                for row in conn.execute(
                    "SELECT item, source_kind, source_name, tier, quantity, "
                    "starts, ends FROM needs ORDER BY rowid"
                )
            ]
            if stored == rows:
                return False
            conn.execute("DELETE FROM needs")
            conn.executemany(
                "INSERT INTO needs (item, source_kind, source_name, tier, "
                "quantity, starts, ends) VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        return True

    def _get_user_connection(self) -> sqlite3.Connection:
        """Connect to the user database, creating it and its tables once."""
        conn = sqlite3.connect(self.user_db_path)
        conn.row_factory = sqlite3.Row
        if not self._user_db_ready:
            # What the player has finished: workshop level reached, 1 for a
            # done quest, highest project phase done
            conn.execute("""
                CREATE TABLE IF NOT EXISTS progress (
                    source_kind TEXT CHECK (
                        source_kind IN ('quest', 'project', 'workshop')
                    ),
                    source_name TEXT COLLATE NOCASE,
                    value INTEGER,
                    PRIMARY KEY (source_kind, source_name)
                )
            """)
            # Blueprints the player has found, with when ARLO recorded it
            conn.execute("""
                CREATE TABLE IF NOT EXISTS blueprint_owned (
                    name TEXT PRIMARY KEY NOT NULL COLLATE NOCASE,
                    found_at TEXT NOT NULL
                )
            """)
            # The last loadout snapshot: each item's count per panel, and one
            # row of what the LOADOUT header showed and whether it agreed
            conn.execute("""
                CREATE TABLE IF NOT EXISTS loadout_items (
                    name TEXT NOT NULL COLLATE NOCASE,
                    panel TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    seen_at TEXT NOT NULL,
                    PRIMARY KEY (name, panel)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS loadout_snapshot (
                    seen_at TEXT NOT NULL,
                    header_value INTEGER,
                    header_weight REAL,
                    agrees INTEGER
                )
            """)
            # The stash pass's slot cache: rows from SlotCache snapshots in
            # least recently used order, valid only for their fingerprint
            conn.execute("""
                CREATE TABLE IF NOT EXISTS slot_cache (
                    kind TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    fingerprint TEXT NOT NULL,
                    PRIMARY KEY (kind, key)
                )
            """)
            # The stored stash map (stash_map.StashMapStore): one row of the
            # view it was built on, its geometry in shares of the frame
            # height, the STASH header last read and the map's version
            # (stash_map.MAP_VERSION, NULL on maps saved before it), and one
            # row per slot with its content, look fingerprint and when that
            # was seen
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stash_map_view (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    tab INTEGER,
                    side INTEGER NOT NULL,
                    grid_left INTEGER NOT NULL,
                    pitch REAL NOT NULL,
                    column_left REAL NOT NULL,
                    header_used INTEGER,
                    header_capacity INTEGER,
                    header_seen_at TEXT,
                    version INTEGER
                )
            """)
            view_columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(stash_map_view)").fetchall()
            }
            if "version" not in view_columns:
                conn.execute("ALTER TABLE stash_map_view ADD COLUMN version INTEGER")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stash_map_cells (
                    row INTEGER NOT NULL,
                    col INTEGER NOT NULL,
                    filled INTEGER NOT NULL,
                    name TEXT,
                    tier TEXT,
                    stack INTEGER,
                    confidence REAL,
                    look BLOB,
                    last_seen TEXT NOT NULL,
                    PRIMARY KEY (row, col)
                )
            """)
            conn.commit()
            self._user_db_ready = True
        return conn

    def load_progress_csv(self, path: Path | str) -> bool:
        """Replace all progress rows in the user database with the CSV rows.

        Writes nothing when the stored rows already match. Returns True when
        the table changed.
        """
        rows = read_progress_csv(Path(path))
        with self._get_user_connection() as conn:
            stored = [
                tuple(row)
                for row in conn.execute(
                    "SELECT source_kind, source_name, value "
                    "FROM progress ORDER BY rowid"
                )
            ]
            if stored == rows:
                return False
            conn.execute("DELETE FROM progress")
            conn.executemany(
                "INSERT INTO progress (source_kind, source_name, value) "
                "VALUES (?, ?, ?)",
                rows,
            )
            conn.commit()
        return True

    def progress_map(self) -> dict[tuple[str, str], int]:
        """Recorded progress as {(source_kind, source_name): value}.

        Empty, without creating the user database, when it does not exist.
        """
        if not self.user_db_path.exists():
            return {}
        with self._get_user_connection() as conn:
            rows = conn.execute(
                "SELECT source_kind, source_name, value FROM progress"
            ).fetchall()
        return {(row[0], row[1]): row[2] for row in rows}

    def load_blueprints_csv(self, path: Path | str) -> bool:
        """Replace all blueprints rows with the CSV rows.

        Writes nothing when the stored rows already match. Returns True when
        the table changed.
        """
        rows = read_blueprints_csv(Path(path))
        placeholders = ", ".join("?" * len(Blueprint._fields))
        with self._get_connection() as conn:
            stored = [
                Blueprint._make(row)
                for row in conn.execute(
                    f"SELECT {BLUEPRINT_COLUMNS} FROM blueprints ORDER BY rowid"
                )
            ]
            if stored == rows:
                return False
            conn.execute("DELETE FROM blueprints")
            conn.executemany(
                f"INSERT INTO blueprints ({BLUEPRINT_COLUMNS}) VALUES ({placeholders})",
                rows,
            )
            conn.commit()
        return True

    def load_blueprint_grid_csv(self, path: Path | str) -> bool:
        """Replace all blueprint_grid rows with the CSV rows.

        Writes nothing when the stored rows already match. Returns True when
        the table changed.
        """
        rows = read_blueprint_grid_csv(Path(path))
        with self._get_connection() as conn:
            stored = [
                tuple(row)
                for row in conn.execute(
                    "SELECT grid_index, name FROM blueprint_grid ORDER BY rowid"
                )
            ]
            if stored == rows:
                return False
            conn.execute("DELETE FROM blueprint_grid")
            conn.executemany(
                "INSERT INTO blueprint_grid (grid_index, name) VALUES (?, ?)", rows
            )
            conn.commit()
        return True

    def blueprint_catalog(self) -> list[Blueprint]:
        """Every catalog blueprint, in CSV order (sorted by name)."""
        with self._get_connection() as conn:
            rows = conn.execute(
                f"SELECT {BLUEPRINT_COLUMNS} FROM blueprints ORDER BY rowid"
            ).fetchall()
        return [Blueprint._make(row) for row in rows]

    def blueprint_at(self, grid_index: int) -> str | None:
        """Catalog name in a Collections grid slot, None for unknown or absent."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT name FROM blueprint_grid WHERE grid_index = ?", (grid_index,)
            ).fetchone()
        return row["name"] if row else None

    def grid_index_of(self, name: str) -> int | None:
        """Collections grid slot of a blueprint (ASCII case ignored), or None."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT grid_index FROM blueprint_grid "
                "WHERE name = ? COLLATE NOCASE ORDER BY grid_index LIMIT 1",
                (name.strip(),),
            ).fetchone()
        return row["grid_index"] if row else None

    def set_blueprint_owned(self, name: str, *, owned: bool) -> None:
        """Record in the user database that a blueprint is found, or forget it.

        Marking an already owned blueprint keeps its first found_at.
        """
        name = name.strip()
        with self._get_user_connection() as conn:
            if owned:
                conn.execute(
                    "INSERT OR IGNORE INTO blueprint_owned (name, found_at) "
                    "VALUES (?, ?)",
                    (name, datetime.now(UTC).isoformat(timespec="seconds")),
                )
            else:
                conn.execute(
                    "DELETE FROM blueprint_owned WHERE name = ? COLLATE NOCASE",
                    (name,),
                )
            conn.commit()

    def owned_blueprints(self) -> set[str]:
        """Names of the blueprints marked found.

        Empty, without creating the user database, when it does not exist.
        """
        if not self.user_db_path.exists():
            return set()
        with self._get_user_connection() as conn:
            rows = conn.execute("SELECT name FROM blueprint_owned").fetchall()
        return {row[0] for row in rows}

    def save_loadout(self, tally: "LoadoutTally") -> None:
        """Replace the stored loadout snapshot with this tally, in one transaction."""
        seen_at = datetime.now(UTC).isoformat(timespec="seconds")
        agrees = None if tally.agrees is None else int(tally.agrees)
        rows = [
            (name, panel, count, seen_at)
            for panel, counts in tally.panel_counts.items()
            for name, count in counts.items()
        ]
        with self._get_user_connection() as conn:
            conn.execute("DELETE FROM loadout_items")
            conn.execute("DELETE FROM loadout_snapshot")
            conn.executemany(
                "INSERT INTO loadout_items (name, panel, count, seen_at) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.execute(
                "INSERT INTO loadout_snapshot "
                "(seen_at, header_value, header_weight, agrees) VALUES (?, ?, ?, ?)",
                (seen_at, tally.header_value, tally.header_weight, agrees),
            )

    def loadout_counts(self) -> dict[str, int]:
        """Item name to count over every panel of the stored loadout snapshot.

        Empty, without creating the user database, when it does not exist.
        """
        if not self.user_db_path.exists():
            return {}
        with self._get_user_connection() as conn:
            rows = conn.execute(
                "SELECT name, SUM(count) FROM loadout_items GROUP BY name"
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def save_slot_cache(
        self, fingerprint: str, rows: Iterable[tuple[str, str, str]]
    ) -> None:
        """Replace the stored slot cache with these (kind, key, value) rows.

        The rows keep their order, in one transaction, under fingerprint.
        """
        with self._get_user_connection() as conn:
            conn.execute("DELETE FROM slot_cache")
            conn.executemany(
                "INSERT INTO slot_cache (kind, key, value, position, fingerprint) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    (kind, key, value, n, fingerprint)
                    for n, (kind, key, value) in enumerate(rows)
                ),
            )

    def slot_cache_rows(self, fingerprint: str) -> list[tuple[str, str, str]]:
        """The stored slot cache's (kind, key, value) rows in their saved order.

        Empty when the stored cache has another fingerprint, or, without
        creating the user database, when it does not exist.
        """
        if not self.user_db_path.exists():
            return []
        with self._get_user_connection() as conn:
            rows = conn.execute(
                "SELECT kind, key, value FROM slot_cache WHERE fingerprint = ? "
                "ORDER BY position",
                (fingerprint,),
            ).fetchall()
        return [(row[0], row[1], row[2]) for row in rows]

    def load_stash_map(self) -> tuple[sqlite3.Row | None, list[sqlite3.Row]]:
        """The stored stash map's view row and its cell rows.

        (None, []) without creating the user database when it does not exist.
        """
        if not self.user_db_path.exists():
            return None, []
        with self._get_user_connection() as conn:
            view = conn.execute("SELECT * FROM stash_map_view WHERE id = 1").fetchone()
            cells = conn.execute(
                "SELECT * FROM stash_map_cells ORDER BY row, col"
            ).fetchall()
        return view, cells

    def save_stash_map(
        self, view: dict, cells: list[dict], *, replace: bool = False
    ) -> None:
        """Write the stash map's view row and upsert these cells, in one transaction.

        view has the stash_map_view columns but id (version None when not
        given), each cell the stash_map_cells columns. Cells not given are
        left as stored, or deleted first when replace is True (a new map in
        place of an old one).
        """
        with self._get_user_connection() as conn:
            if replace:
                conn.execute("DELETE FROM stash_map_cells")
            conn.execute(
                "INSERT OR REPLACE INTO stash_map_view (id, tab, side, grid_left, "
                "pitch, column_left, header_used, header_capacity, header_seen_at, "
                "version) VALUES (1, :tab, :side, :grid_left, :pitch, :column_left, "
                ":header_used, :header_capacity, :header_seen_at, :version)",
                {"version": None, **view},
            )
            conn.executemany(
                "INSERT OR REPLACE INTO stash_map_cells (row, col, filled, name, "
                "tier, stack, confidence, look, last_seen) VALUES (:row, :col, "
                ":filled, :name, :tier, :stack, :confidence, :look, :last_seen)",
                cells,
            )

    def sync_from_csvs(self) -> None:
        """Load the recycle, needs, blueprint and progress CSVs that exist."""
        if self.recycle_outputs_csv.exists():
            self.load_recycle_outputs_csv(self.recycle_outputs_csv)
        if self.needs_csv.exists():
            self.load_needs_csv(self.needs_csv)
        if self.blueprints_csv.exists():
            self.load_blueprints_csv(self.blueprints_csv)
        if self.blueprint_grid_csv.exists():
            self.load_blueprint_grid_csv(self.blueprint_grid_csv)
        if self.progress_csv.exists():
            self.load_progress_csv(self.progress_csv)

    def needs_for(self, name: str) -> list[Need]:
        """Every quest, project and workshop row that needs an item."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT item, source_kind, source_name, tier, quantity, starts, ends
                FROM needs
                WHERE item = ? COLLATE NOCASE
                ORDER BY source_kind, source_name, tier
                """,
                (name.strip(),),
            ).fetchall()
        return [Need._make(row) for row in rows]

    def blueprint_for(self, name: str) -> str | None:
        """Catalog name of the "<base> Blueprint" entry for an item, or None.

        The base is the name without its tier numeral, so Anvil IV maps to
        Anvil Blueprint. The match is exact apart from ASCII case.
        """
        base, _ = split_tier(name.strip())
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT name FROM items WHERE name = ? COLLATE NOCASE",
                (f"{base} Blueprint",),
            ).fetchone()
        return row["name"] if row else None

    def _build_tier_index(self) -> None:
        """Map each tiered base, in tier_key form, to {tier: catalog name}.

        Also keeps every catalog name with its tier_key for the tolerant
        lookup.
        """
        index: dict[str, dict[str, str]] = {}
        name_keys: list[tuple[str, str]] = []
        with self._get_connection() as conn:
            for row in conn.execute("SELECT name FROM items"):
                base, tier = split_tier(row["name"])
                if tier is not None:
                    index.setdefault(tier_key(base), {})[tier] = row["name"]
                name_keys.append((tier_key(row["name"]), row["name"]))
        self._tier_index = index
        self._name_keys = name_keys

    def _resolve_tiered(self, name: str, tier_vote: TierVote | None) -> str | None:
        """Resolve a garbled tiered read to a catalog name, or None.

        The read must be a tiered base, optionally followed by a tier token
        in tolerant form. The read tier resolves to the nearest tier that
        exists for that base. A confident stroke-count vote for a tier that
        exists wins over the read tier.
        """
        key = tier_key(name)
        if not key or not self._tier_index:
            return None
        if key in self._tier_index:
            base, read_tier = key, None
        else:
            base, read_tier = split_read(key, self._tier_index)
            if read_tier is None:
                return None
        tiers = self._tier_index[base]
        chosen = resolve_tier(read_tier, tiers) if read_tier else None

        if tier_vote is not None and tier_vote.tier is not None:
            if (
                tier_vote.confidence >= TIER_VOTE_MIN_CONFIDENCE
                and tier_vote.tier in tiers
            ):
                chosen = tier_vote.tier
            logger.debug(
                f"Tier for {name!r}: read {read_tier}, vote {tier_vote.tier} "
                f"({tier_vote.confidence:.2f}), chose {chosen}"
            )
        return tiers[chosen] if chosen else None

    def _resolve_tolerant(self, name: str, tier_vote: TierVote | None) -> str | None:
        """Resolve a partial or one-edit read to the one item it can be, or None.

        Candidates are the catalog names whose tier_key the read's tier_key
        is a prefix of, or is within one edit of. One untiered candidate
        resolves to itself. Candidates that are all tiers of one base
        resolve to the tier a confident stroke vote names, or, with a single
        candidate, to that tier when the read ends in it (the edit was in
        the base, as in "BURLETT IV"). Anything else stays unknown rather
        than guessing a tier or an item.
        """
        key = tier_key(name)
        if len(key) < TOLERANT_MIN_LENGTH:
            return None
        candidates = [
            catalog
            for catalog_key, catalog in self._name_keys
            if catalog_key.startswith(key) or _within_one_edit(key, catalog_key)
        ]
        if not candidates:
            return None

        split = [split_tier(c) for c in candidates]
        bases = {tier_key(base) for base, _ in split}
        chosen: str | None = None
        if len(candidates) == 1 and split[0][1] is None:
            chosen = candidates[0]
        elif len(bases) == 1 and all(tier for _, tier in split):
            tiers = self._tier_index.get(bases.pop(), {})
            if (
                tier_vote is not None
                and tier_vote.tier in tiers
                and tier_vote.confidence >= TIER_VOTE_MIN_CONFIDENCE
            ):
                chosen = tiers[tier_vote.tier]
            elif len(candidates) == 1 and key.endswith(tier_key(split[0][1] or "")):
                chosen = candidates[0]
        logger.debug(
            f"Tolerant lookup for {name!r}: candidates {candidates}, chose {chosen}"
        )
        return chosen

    def recycle_outputs(self, name: str) -> list[RecycleOutput]:
        """All recycle outputs for an item, home rows first, in source order."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT component, quantity, location FROM recycle_outputs
                WHERE item = ? COLLATE NOCASE
                ORDER BY location = 'field', rowid
                """,
                (name.strip(),),
            ).fetchall()
        return [
            RecycleOutput(row["component"], row["quantity"], row["location"])
            for row in rows
        ]

    def home_only_outputs(self, name: str) -> list[RecycleOutput]:
        """Home outputs the field does not give, or gives fewer of.

        A component the field yields at least as many of is dropped. One it
        yields fewer of keeps the difference.
        """
        outputs = self.recycle_outputs(name)
        field = {
            o.component.lower(): o.quantity for o in outputs if o.location == FIELD
        }
        extras: list[RecycleOutput] = []
        for output in outputs:
            if output.location != HOME:
                continue
            remaining = output.quantity - field.get(output.component.lower(), 0)
            if remaining > 0:
                extras.append(output._replace(quantity=remaining))
        return extras

    @staticmethod
    def _match_prefix(conn, read: str):
        """The row of the longest catalog name opening the read, or None.

        The name's match key must be a prefix of the read's key, end at a
        word boundary in the read and be at least PREFIX_MIN_KEY long.
        """
        key, ends = _boundary_key(read)
        best = None
        best_len = PREFIX_MIN_KEY - 1
        for candidate in conn.execute("SELECT * FROM items"):
            cand_key = _match_key(candidate["name"])
            n = len(cand_key)
            if (
                n > best_len
                and n <= len(key)
                and key.startswith(cand_key)
                and ends[n - 1]
            ):
                best, best_len = candidate, n
        return best

    def lookup(self, name: str, tier_vote: TierVote | None = None) -> Item | None:
        """Look up an item by name (case-insensitive, with fuzzy matching).

        tier_vote is the stroke-count vote on the name's tier token, when the
        OCR engine produced one. A confident vote picks the tier of a tiered
        base even when the read text matched another tier exactly.
        """
        clean_name = name.strip()

        with self._get_connection() as conn:
            # First try exact match
            cursor = conn.execute(
                "SELECT * FROM items WHERE name = ? COLLATE NOCASE",
                (clean_name,),
            )
            row = cursor.fetchone()

            # Whitespace/punctuation-insensitive exact match. OCR often drops
            # spaces ("ILTOROII") or adds stray characters; this still resolves
            # to the stored "Il Toro II" without the looseness of LIKE/fuzzy.
            if not row:
                target = _match_key(clean_name)
                if target:
                    for candidate in conn.execute("SELECT * FROM items"):
                        if _match_key(candidate["name"]) == target:
                            row = candidate
                            break

            # Tier resolver: a tiered base plus a garbled numeral ("ANVIL 1l",
            # "ARPEGGIO |V"). Runs on a miss, or on a hit when a confident
            # vote may change the tier, and skips the looser LIKE and fuzzy
            # steps.
            confident_vote = (
                tier_vote is not None
                and tier_vote.tier is not None
                and tier_vote.confidence >= TIER_VOTE_MIN_CONFIDENCE
            )
            if not row or confident_vote:
                resolved = self._resolve_tiered(clean_name, tier_vote)
                if resolved:
                    row = conn.execute(
                        "SELECT * FROM items WHERE name = ?", (resolved,)
                    ).fetchone()

            # A catalog name followed by junk ("CRACKED BIOSCANNER ;MER:...")
            # resolves to the longest name that opens the read at a word end
            if not row:
                row = self._match_prefix(conn, clean_name)

            # If no exact match, try LIKE match
            if not row:
                cursor = conn.execute(
                    """
                    SELECT * FROM items
                    WHERE name LIKE ? COLLATE NOCASE
                    ORDER BY LENGTH(name) ASC
                    LIMIT 1
                    """,
                    (f"%{clean_name}%",),
                )
                row = cursor.fetchone()

            # If still no match, try fuzzy matching (handles OCR typos)
            if not row:
                cursor = conn.execute("SELECT name FROM items")
                all_names = [r[0] for r in cursor.fetchall()]

                # Find closest match with at least 80% similarity
                matches = difflib.get_close_matches(
                    clean_name, all_names, n=1, cutoff=0.8
                )

                if matches:
                    match_name = matches[0]
                    cursor = conn.execute(
                        "SELECT * FROM items WHERE name = ?", (match_name,)
                    )
                    row = cursor.fetchone()

            # Last resort: a partial read ("BURLETT IV") that only one
            # catalog item can be
            if not row:
                resolved = self._resolve_tolerant(clean_name, tier_vote)
                if resolved:
                    row = conn.execute(
                        "SELECT * FROM items WHERE name = ?", (resolved,)
                    ).fetchone()

            if row:
                return self._row_to_item(row)
            return None

    def count(self) -> int:
        """Get total item count."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM items")
            return cursor.fetchone()[0]

    def load_csv(self, csv_path: Path | str, *, clear_existing: bool = True) -> int:
        """
        Load items from a CSV file into the database.

        CSV must have columns: name, action, recycle_for, keep_for
        Optional columns: sell_price, stack_size, rarity, category

        Args:
            csv_path: Path to the CSV file
            clear_existing: If True, clear existing items before loading

        Returns:
            Number of items loaded
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            msg = f"CSV file not found: {csv_path}"
            raise FileNotFoundError(msg)

        if clear_existing:
            self.clear()

        count = 0
        with self._get_connection() as conn:
            with Path(csv_path).open(encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    name = row.get("name", "").strip()
                    action = row.get("action", "").strip()
                    recycle_for = row.get("recycle_for", "").strip() or None
                    keep_for = row.get("keep_for", "").strip() or None

                    # New columns (optional in CSV for backwards compat)
                    sell_price_str = row.get("sell_price", "").strip()
                    stack_size_str = row.get("stack_size", "").strip()
                    sell_price = int(sell_price_str) if sell_price_str else None
                    stack_size = int(stack_size_str) if stack_size_str else None
                    rarity = _optional_text(row.get("rarity"))
                    category = _optional_text(row.get("category"))

                    if not name or not action:
                        continue

                    conn.execute(
                        """
                        INSERT INTO items (name, action, recycle_for, keep_for, sell_price, stack_size, rarity, category)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(name) DO UPDATE SET
                            action = excluded.action,
                            recycle_for = excluded.recycle_for,
                            keep_for = excluded.keep_for,
                            sell_price = excluded.sell_price,
                            stack_size = excluded.stack_size,
                            rarity = excluded.rarity,
                            category = excluded.category
                        """,
                        (
                            name,
                            action,
                            recycle_for,
                            keep_for,
                            sell_price,
                            stack_size,
                            rarity,
                            category,
                        ),
                    )
                    count += 1

            self._refresh_home_outputs(conn)
            conn.commit()

        self._build_tier_index()
        return count

    def log_missing_item(self, name: str) -> None:
        """Log an unknown item to missing_items.csv for easy addition later."""
        if not name:
            return

        missing_file = self.db_path.parent / "missing_items.csv"

        try:
            # Append to file
            with missing_file.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([name])
        except Exception:  # noqa: BLE001
            pass  # Fail silently to not interrupt the user

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> Item:
        """Convert database row to Item model."""
        return Item(
            name=row["name"],
            action=row["action"],
            recycle_for=row["recycle_for"],
            keep_for=row["keep_for"],
            sell_price=row["sell_price"] if "sell_price" in row.keys() else None,
            stack_size=row["stack_size"] if "stack_size" in row.keys() else None,
            rarity=row["rarity"],
            category=row["category"],
        )

    def get_all_items(self) -> list[Item]:
        """Get all items from the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT name, action, recycle_for, keep_for, sell_price, stack_size, "
                "rarity, category FROM items ORDER BY name"
            )
            return [
                Item(
                    name=row["name"],
                    action=row["action"],
                    recycle_for=row["recycle_for"],
                    keep_for=row["keep_for"],
                    sell_price=row["sell_price"],
                    stack_size=row["stack_size"],
                    rarity=row["rarity"],
                    category=row["category"],
                )
                for row in cursor.fetchall()
            ]

    def clear(self) -> None:
        """Delete all items and the home recycle rows derived from them."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM items")
            conn.execute("DELETE FROM recycle_outputs WHERE location = ?", (HOME,))
            conn.commit()
        self._tier_index = {}


def get_database() -> Database:
    """Get a database instance."""
    return Database()


def load_csv_to_database(csv_path: Path | str, *, clear_existing: bool = True) -> int:
    """
    Convenience function to load a CSV file into the database.

    Args:
        csv_path: Path to CSV file (columns: name, action, recycle_for, keep_for, sell_price, stack_size)
        clear_existing: If True, clears existing data before loading

    Returns:
        Number of items loaded
    """
    db = get_database()
    return db.load_csv(Path(csv_path), clear_existing=clear_existing)
