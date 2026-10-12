"""
Database module for ARLO.
SQLite operations for item recommendations with Pydantic models.
"""

import csv
import difflib
import re
import sqlite3
import string
from pathlib import Path
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

HOME = "home"
FIELD = "field"
LOCATIONS = (HOME, FIELD)
NEED_KINDS = ("quest", "project", "workshop")
RECYCLE_OUTPUTS_CSV_NAME = "recycle_outputs.csv"
NEEDS_CSV_NAME = "needs.csv"

# "2x Metal Parts", "2 X Metal Parts", "2xMetal Parts"
_COUNTED_ENTRY = re.compile(r"(\d+)\s*[xX]\s*(.*)")

# SQLite's NOCASE collation folds ASCII letters only
_NOCASE = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)

# A stroke-count tier vote at least this confident overrides the read tier
TIER_VOTE_MIN_CONFIDENCE = 0.7


def _match_key(name: str) -> str:
    """Lowercase, alphanumeric-only key for OCR-tolerant name matching.

    OCR frequently drops spaces ("IL TORO II" -> "ILTOROII") or appends stray
    punctuation, so collapsing to alphanumerics lets those still match the
    stored name. Roman-numeral tiers (I/V) survive, so variants stay distinct.
    """
    return "".join(c for c in name.lower() if c.isalnum())


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
    ):
        """Initialize database connection.

        The two CSV paths default to the configured ones when db_path is not
        given, and to the files next to db_path otherwise.
        """
        if db_path is None:
            settings = get_settings()
            db_path = settings.database_path
            default_outputs = settings.recycle_outputs_csv_path
            default_needs = settings.needs_csv_path
        else:
            default_outputs = Path(db_path).parent / RECYCLE_OUTPUTS_CSV_NAME
            default_needs = Path(db_path).parent / NEEDS_CSV_NAME
        self.db_path = db_path
        self.recycle_outputs_csv = Path(recycle_outputs_csv or default_outputs)
        self.needs_csv = Path(needs_csv or default_needs)
        self._tier_index: dict[str, dict[str, str]] = {}
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

    def sync_from_csvs(self) -> None:
        """Load recycle_outputs.csv and needs.csv, whichever of them exist."""
        if self.recycle_outputs_csv.exists():
            self.load_recycle_outputs_csv(self.recycle_outputs_csv)
        if self.needs_csv.exists():
            self.load_needs_csv(self.needs_csv)

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

    def _build_tier_index(self) -> None:
        """Map each tiered base, in tier_key form, to {tier: catalog name}."""
        index: dict[str, dict[str, str]] = {}
        with self._get_connection() as conn:
            for row in conn.execute("SELECT name FROM items"):
                base, tier = split_tier(row["name"])
                if tier is not None:
                    index.setdefault(tier_key(base), {})[tier] = row["name"]
        self._tier_index = index

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
