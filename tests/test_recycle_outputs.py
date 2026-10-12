"""Tests for the recycle_outputs table derived from items.recycle_for."""

from arc_helper.database import Database
from arc_helper.database import RecycleOutput
from arc_helper.database import parse_recycle_for

CSV_HEADER = "name,action,recycle_for,keep_for,sell_price,stack_size\n"


def _db_with_items(tmp_path, items):
    db = Database(db_path=tmp_path / "items.db")
    with db._get_connection() as conn:  # noqa: SLF001 - test seam
        conn.executemany(
            "INSERT INTO items (name, action, recycle_for) VALUES (?, 'Sell', ?)",
            items,
        )
        conn.commit()
    return db


def _add_field(db, rows):
    with db._get_connection() as conn:  # noqa: SLF001 - test seam
        conn.executemany("INSERT INTO recycle_outputs VALUES (?, ?, ?, 'field')", rows)
        conn.commit()


def test_parse_counted_entries():
    assert parse_recycle_for("2x Metal Parts, 4x Wires") == [
        ("Metal Parts", 2),
        ("Wires", 4),
    ]


def test_parse_tolerates_case_spacing_and_missing_count():
    assert parse_recycle_for("2X Metal Parts,3 x Wires,  5xRope ,Battery") == [
        ("Metal Parts", 2),
        ("Wires", 3),
        ("Rope", 5),
        ("Battery", 1),
    ]


def test_parse_empty_and_none():
    assert parse_recycle_for("") == []
    assert parse_recycle_for(None) == []


def test_parse_skips_malformed_entries():
    assert parse_recycle_for("12, 2x, 0x Wires, ,3x Rope, -1x Gears") == [("Rope", 3)]


def test_init_rebuilds_home_rows_and_keeps_field_rows(tmp_path):
    _db_with_items(tmp_path, [("Battery", "2x Metal Parts")])
    db = Database(db_path=tmp_path / "items.db")  # reopened after the insert
    assert db.recycle_outputs("battery") == [RecycleOutput("Metal Parts", 2, "home")]

    # The updater rewrites items directly, without load_csv
    with db._get_connection() as conn:  # noqa: SLF001 - test seam
        conn.execute("UPDATE items SET recycle_for = '5x Wires'")
        conn.commit()
    _add_field(db, [("Battery", "Metal Parts", 1)])

    db = Database(db_path=tmp_path / "items.db")
    assert db.recycle_outputs("Battery") == [
        RecycleOutput("Wires", 5, "home"),
        RecycleOutput("Metal Parts", 1, "field"),
    ]


def test_clear_drops_home_rows_and_keeps_field_rows(tmp_path):
    _db_with_items(tmp_path, [("Battery", "2x Metal Parts")])
    db = Database(db_path=tmp_path / "items.db")
    _add_field(db, [("Battery", "Metal Parts", 1)])

    db.clear()

    assert db.recycle_outputs("Battery") == [RecycleOutput("Metal Parts", 1, "field")]


def test_reload_rebuilds_home_rows_and_keeps_field_rows(tmp_path):
    csv_path = tmp_path / "items.csv"
    csv_path.write_text(
        CSV_HEADER + 'Battery,Sell,"2x Metal Parts, 1x Wires",,250,15\n'
    )
    db = Database(db_path=tmp_path / "items.db")
    db.load_csv(csv_path, clear_existing=True)
    _add_field(db, [("Battery", "Metal Parts", 1)])

    csv_path.write_text(CSV_HEADER + "Battery,Sell,3x Metal Parts,,250,15\n")
    db.load_csv(csv_path, clear_existing=True)

    assert db.recycle_outputs("Battery") == [
        RecycleOutput("Metal Parts", 3, "home"),
        RecycleOutput("Metal Parts", 1, "field"),
    ]


def test_home_only_field_absent_returns_all_home_rows(tmp_path):
    _db_with_items(tmp_path, [("Battery", "2x Metal Parts, 4x Wires")])
    db = Database(db_path=tmp_path / "items.db")
    assert db.home_only_outputs("Battery") == [
        RecycleOutput("Metal Parts", 2, "home"),
        RecycleOutput("Wires", 4, "home"),
    ]


def test_home_only_field_equal_returns_empty(tmp_path):
    _db_with_items(tmp_path, [("Battery", "2x Metal Parts, 4x Wires")])
    db = Database(db_path=tmp_path / "items.db")
    _add_field(db, [("Battery", "Metal Parts", 2), ("Battery", "Wires", 4)])
    assert db.home_only_outputs("Battery") == []


def test_home_only_field_subset_returns_missing(tmp_path):
    _db_with_items(tmp_path, [("Battery", "2x Metal Parts, 4x Wires")])
    db = Database(db_path=tmp_path / "items.db")
    _add_field(db, [("Battery", "Metal Parts", 2)])
    assert db.home_only_outputs("Battery") == [RecycleOutput("Wires", 4, "home")]


def test_home_only_field_lower_quantity_returns_difference(tmp_path):
    _db_with_items(tmp_path, [("Battery", "2x Metal Parts, 4x Wires")])
    db = Database(db_path=tmp_path / "items.db")
    _add_field(db, [("Battery", "Metal Parts", 2), ("Battery", "Wires", 1)])
    assert db.home_only_outputs("Battery") == [RecycleOutput("Wires", 3, "home")]


def test_reopening_unchanged_database_leaves_file_untouched(tmp_path):
    _db_with_items(tmp_path, [("Battery", "2x Metal Parts, 1x wires, 3x Wires")])
    path = tmp_path / "items.db"
    db = Database(db_path=path)  # derives the home rows from the insert
    _add_field(db, [("Battery", "Metal Parts", 1)])
    Database(db_path=path)
    before = path.read_bytes()

    Database(db_path=path)

    assert path.read_bytes() == before
