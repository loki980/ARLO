"""Tests for the needs table loaded from needs.csv."""

from arc_helper.database import Database
from arc_helper.database import Need

NEEDS_HEADER = "item,source_kind,source_name,tier,quantity,starts,ends\n"
NEEDS_ROWS = (
    "Radial Press,workshop,Gunsmith,4,3,,\n"
    "Radial Press,quest,Wires Crossed,,1,,\n"
    "Radial Press,project,Expedition 5,2,4,2026-09-08,2026-10-07\n"
    "Wires,workshop,Gunsmith,2,5,,\n"
)


def _write_needs(tmp_path, rows=NEEDS_ROWS):
    path = tmp_path / "needs.csv"
    path.write_text(NEEDS_HEADER + rows, encoding="utf-8")
    return path


def test_needs_for_matches_nocase_in_order(tmp_path):
    db = Database(db_path=tmp_path / "items.db")
    db.load_needs_csv(_write_needs(tmp_path))

    assert db.needs_for("radial press") == [
        Need("Radial Press", "project", "Expedition 5", 2, 4, "2026-09-08", "2026-10-07"),
        Need("Radial Press", "quest", "Wires Crossed", None, 1, None, None),
        Need("Radial Press", "workshop", "Gunsmith", 4, 3, None, None),
    ]  # fmt: skip
    assert db.needs_for("Battery") == []


def test_load_replaces_all_rows(tmp_path):
    db = Database(db_path=tmp_path / "items.db")
    db.load_needs_csv(_write_needs(tmp_path))

    db.load_needs_csv(_write_needs(tmp_path, "Wires,workshop,Gunsmith,2,6,,\n"))

    assert db.needs_for("Radial Press") == []
    assert db.needs_for("Wires") == [
        Need("Wires", "workshop", "Gunsmith", 2, 6, None, None)
    ]


def test_second_load_of_same_csv_leaves_file_untouched(tmp_path):
    db = Database(db_path=tmp_path / "items.db")
    path = _write_needs(tmp_path)
    assert db.load_needs_csv(path)
    before = (tmp_path / "items.db").read_bytes()

    assert not db.load_needs_csv(path)

    assert (tmp_path / "items.db").read_bytes() == before


def test_sync_loads_needs_csv_next_to_database(tmp_path):
    _write_needs(tmp_path)
    db = Database(db_path=tmp_path / "items.db")

    db.sync_from_csvs()

    assert len(db.needs_for("Radial Press")) == 3
