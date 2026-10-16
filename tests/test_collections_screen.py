"""Tests for the Collections screen reader and the ownership it records."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from arc_helper.collections_screen import CollectionsCell
from arc_helper.collections_screen import CollectionsRead
from arc_helper.collections_screen import read_collections
from arc_helper.collections_screen import record_collections
from arc_helper.database import Database
from arc_helper.tooltip_layout import Rect

FIXTURES = Path(__file__).parent / "fixtures"
# Jeffrey's Collections screen of 15 Oct, 3840x2160 scaled to 2000x1125:
# FOUND: 25/91 on the BLUEPRINTS tab
COLLECTIONS = FIXTURES / "collections_blueprints_capture.png"
# Grid indices of the 25 found cells, read by eye
FOUND = [4, 5, 7, 8, 18, 31, 32, 33, 35, 37, 42, 46, 47, 49]
FOUND += [61, 67, 73, 74, 75, 76, 79, 80, 81, 89, 91]


def _frame(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


@pytest.mark.parametrize("size", [None, (3840, 2160)])
def test_fixture_read(size):
    image = Image.open(COLLECTIONS).convert("RGB")
    if size is not None:
        image = image.resize(size, Image.LANCZOS)

    read = read_collections(np.asarray(image))

    assert read is not None
    assert (read.found_count, read.total) == (25, 91)
    assert len(read.cells) == 91
    assert [cell.grid_index for cell in read.cells] == list(range(1, 92))
    assert read.found_indices == FOUND
    assert read.consistent
    # First cell, a found tile with art over half of it, the last cell
    assert not read.cells[0].found
    assert read.cells[17].found
    assert read.cells[90].found


def test_other_screen_reads_none():
    assert read_collections(_frame(FIXTURES / "grid_stash_card_capture.png")) is None


def test_blacked_out_cell_is_inconsistent_and_records_nothing(tmp_path):
    frame = _frame(COLLECTIONS).copy()
    cell = read_collections(frame).cells[3]
    inner = cell.rect
    frame[inner.top + 5 : inner.bottom - 5, inner.left + 5 : inner.right - 5] = 3

    read = read_collections(frame)

    assert read is not None
    assert len(read.found_indices) == 24
    assert not read.consistent
    db = _db(tmp_path)
    assert record_collections(db, read) == []
    assert not (tmp_path / "arlo_user.db").exists()


GRID_CSV = (
    "grid_index,name\n1,Anvil Blueprint\n2,\n3,Banjo Blueprint\n4,\n5,Cello Blueprint\n"
)


def _db(tmp_path):
    (tmp_path / "blueprint_grid.csv").write_text(GRID_CSV, encoding="utf-8")
    db = Database(db_path=tmp_path / "items.db")
    db.sync_from_csvs()
    return db


def _read(found: list[bool], *, consistent: bool = True) -> CollectionsRead:
    cells = [
        CollectionsCell(i + 1, Rect(0, 0, 1, 1), flag) for i, flag in enumerate(found)
    ]
    return CollectionsRead(sum(found), len(found), cells, consistent)


def test_record_marks_known_cells_and_returns_unknown_found(tmp_path):
    db = _db(tmp_path)
    db.set_blueprint_owned("Banjo Blueprint", owned=True)

    unknown = record_collections(db, _read([True, True, False, False, True]))

    assert unknown == [2]
    assert db.owned_blueprints() == {"Anvil Blueprint", "Cello Blueprint"}


def test_record_skips_inconsistent_read(tmp_path):
    db = _db(tmp_path)
    db.set_blueprint_owned("Banjo Blueprint", owned=True)

    unknown = record_collections(db, _read([True, True, False], consistent=False))

    assert unknown == []
    assert db.owned_blueprints() == {"Banjo Blueprint"}
