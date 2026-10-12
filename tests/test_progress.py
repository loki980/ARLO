"""Tests for the progress table loaded from progress.csv."""

from arc_helper.database import Database

PROGRESS_HEADER = "source_kind,source_name,value\n"
PROGRESS_ROWS = (
    "workshop,Gunsmith,3\n"
    "quest,After Rain Comes,1\n"
    "project,Expedition 5,2\n"
    "station,Gunsmith,1\n"
    "workshop,Scrappy,many\n"
)


def _write_progress(tmp_path, rows=PROGRESS_ROWS):
    path = tmp_path / "progress.csv"
    path.write_text(PROGRESS_HEADER + rows, encoding="utf-8")
    return path


def test_progress_map_skips_malformed_rows(tmp_path):
    db = Database(db_path=tmp_path / "items.db")

    assert db.load_progress_csv(_write_progress(tmp_path))

    assert db.progress_map() == {
        ("workshop", "Gunsmith"): 3,
        ("quest", "After Rain Comes"): 1,
        ("project", "Expedition 5"): 2,
    }


def test_second_load_of_same_csv_leaves_file_untouched(tmp_path):
    db = Database(db_path=tmp_path / "items.db")
    path = _write_progress(tmp_path)
    assert db.load_progress_csv(path)
    before = (tmp_path / "items.db").read_bytes()

    assert not db.load_progress_csv(path)

    assert (tmp_path / "items.db").read_bytes() == before


def test_missing_progress_file_is_fine(tmp_path):
    db = Database(db_path=tmp_path / "items.db")
    before = (tmp_path / "items.db").read_bytes()

    db.sync_from_csvs()

    assert db.progress_map() == {}
    assert (tmp_path / "items.db").read_bytes() == before


def test_sync_loads_progress_csv_next_to_database(tmp_path):
    _write_progress(tmp_path, "workshop,Gunsmith,2\n")
    db = Database(db_path=tmp_path / "items.db")

    db.sync_from_csvs()

    assert db.progress_map() == {("workshop", "Gunsmith"): 2}
