"""Tests for the blueprint tables, the owned blueprints and the icon cache."""

import sqlite3
from datetime import datetime

import pytest
import requests

from arc_helper import config
from arc_helper import icons
from arc_helper.database import Blueprint
from arc_helper.database import Database

BLUEPRINTS_CSV = (
    "name,item_id,unlocks,icon_url,map,map_condition,scavengable,containers,"
    "quest_reward,trials_reward\n"
    "Anvil Blueprint,anvil_blueprint,Anvil,https://cdn/anvil.png,All,Any,Yes,"
    "Raider Containers,No,Yes\n"
    "Banjo Blueprint,banjo_blueprint,Banjo,https://cdn/banjo.png,,,,,,\n"
    ",nameless_blueprint,Nameless,,,,,,,\n"
    "anvil blueprint,anvil_again,Anvil,,,,,,,\n"
)
GRID_CSV = (
    "grid_index,name\n"
    "1,Silencer II Blueprint\n"
    "2,\n"
    "3,Anvil Blueprint\n"
    "x,Broken Blueprint\n"
    "3,Banjo Blueprint\n"
)


def _db(tmp_path):
    (tmp_path / "blueprints.csv").write_text(BLUEPRINTS_CSV, encoding="utf-8")
    (tmp_path / "blueprint_grid.csv").write_text(GRID_CSV, encoding="utf-8")
    db = Database(db_path=tmp_path / "items.db")
    db.sync_from_csvs()
    return db


def test_sync_loads_catalog_and_grid(tmp_path):
    db = _db(tmp_path)

    catalog = db.blueprint_catalog()
    assert [b.name for b in catalog] == ["Anvil Blueprint", "Banjo Blueprint"]
    assert catalog[0] == Blueprint(
        "Anvil Blueprint", "anvil_blueprint", "Anvil", "https://cdn/anvil.png",
        "All", "Any", "Yes", "Raider Containers", "No", "Yes",
    )  # fmt: skip
    assert catalog[1].map is None
    assert catalog[1].trials_reward is None

    assert db.blueprint_at(1) == "Silencer II Blueprint"
    assert db.blueprint_at(2) is None
    assert db.blueprint_at(3) == "Anvil Blueprint"
    assert db.blueprint_at(4) is None
    assert db.grid_index_of("anvil blueprint") == 3
    assert db.grid_index_of("Banjo Blueprint") is None


def test_second_sync_leaves_items_db_untouched(tmp_path):
    db = _db(tmp_path)
    before = (tmp_path / "items.db").read_bytes()

    db.sync_from_csvs()
    Database(db_path=tmp_path / "items.db").sync_from_csvs()

    assert (tmp_path / "items.db").read_bytes() == before


def test_owned_blueprints_live_in_user_db_only(tmp_path):
    db = _db(tmp_path)
    before = (tmp_path / "items.db").read_bytes()
    assert db.owned_blueprints() == set()
    assert not (tmp_path / "arlo_user.db").exists()

    db.set_blueprint_owned("Anvil Blueprint", owned=True)
    db.set_blueprint_owned("Banjo Blueprint", owned=True)
    db.set_blueprint_owned("banjo blueprint", owned=False)

    assert db.owned_blueprints() == {"Anvil Blueprint"}
    assert (tmp_path / "arlo_user.db").exists()
    assert (tmp_path / "items.db").read_bytes() == before


def test_marking_owned_twice_keeps_first_found_at(tmp_path):
    db = _db(tmp_path)
    db.set_blueprint_owned("Anvil Blueprint", owned=True)
    with sqlite3.connect(tmp_path / "arlo_user.db") as conn:
        conn.execute(
            "UPDATE blueprint_owned SET found_at = '2026-01-02T03:04:05+00:00'"
        )

    db.set_blueprint_owned("ANVIL BLUEPRINT", owned=True)

    with sqlite3.connect(tmp_path / "arlo_user.db") as conn:
        rows = conn.execute("SELECT name, found_at FROM blueprint_owned").fetchall()
    assert rows == [("Anvil Blueprint", "2026-01-02T03:04:05+00:00")]
    assert datetime.fromisoformat(rows[0][1]).tzinfo is not None


def test_blueprint_for_still_reads_items(tmp_path):
    items_csv = tmp_path / "items.csv"
    items_csv.write_text(
        "name,action,recycle_for,keep_for\nAnvil Blueprint,Keep,,Learn it\n",
        encoding="utf-8",
    )
    db = _db(tmp_path)
    db.load_csv(items_csv)

    assert db.blueprint_for("Anvil IV") == "Anvil Blueprint"
    assert db.blueprint_for("Banjo") is None


PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 16


class _IconResponse:
    def __init__(self, content=PNG, status=200):
        self.content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            msg = f"{self.status_code} Not Found"
            raise requests.HTTPError(msg)


def _fake_icon_get(monkeypatch, response):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(icons.requests, "get", fake_get)
    return calls


def test_icon_downloads_once_and_is_cached(monkeypatch, tmp_path):
    calls = _fake_icon_get(monkeypatch, _IconResponse())
    cache = tmp_path / "cache" / "icons" / "blueprints"

    first = icons.blueprint_icon_path("anvil_blueprint", cache)
    second = icons.blueprint_icon_path("anvil_blueprint", cache)

    assert first == second == cache / "anvil_blueprint.png"
    assert first.read_bytes() == PNG
    assert len(calls) == 1
    url, kwargs = calls[0]
    assert url == "https://cdn.arctracker.io/items/v2/anvil_blueprint.png"
    assert kwargs["headers"]["User-Agent"].startswith("ARLO/")
    assert kwargs["timeout"] > 0


def _no_network(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("the cache-only lookup reached the network")

    monkeypatch.setattr(icons.requests, "get", fail)


@pytest.mark.parametrize("item_id", ["anvil_blueprint", None, "../anvil"])
def test_cached_icon_is_none_on_a_miss_without_the_network(
    monkeypatch, tmp_path, item_id
):
    _no_network(monkeypatch)
    assert icons.cached_blueprint_icon(item_id, tmp_path) is None


def test_cached_icon_finds_a_cached_file(monkeypatch, tmp_path):
    _no_network(monkeypatch)
    (tmp_path / "anvil_blueprint.png").write_bytes(PNG)
    assert (
        icons.cached_blueprint_icon("anvil_blueprint", tmp_path)
        == tmp_path / "anvil_blueprint.png"
    )


def test_icon_cache_dir_defaults_to_settings(monkeypatch, tmp_path):
    _fake_icon_get(monkeypatch, _IconResponse())
    settings = icons.get_settings()
    monkeypatch.setattr(settings, "icon_cache_dir", tmp_path / "icons")

    assert icons.blueprint_icon_path("anvil_blueprint").parent == tmp_path / "icons"


@pytest.mark.parametrize(
    ("item_id", "response"),
    [
        ("anvil_blueprint", requests.ConnectionError("offline")),
        ("anvil_blueprint", _IconResponse(status=404)),
        ("anvil_blueprint", _IconResponse(content=b"<html>")),
        ("../anvil", _IconResponse()),
    ],
    ids=["offline", "not-found", "not-png", "bad-id"],
)
def test_icon_failure_raises_clear_error(monkeypatch, tmp_path, item_id, response):
    _fake_icon_get(monkeypatch, response)

    with pytest.raises(icons.IconDownloadError, match="anvil"):
        icons.blueprint_icon_path(item_id, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_save_to_env_keeps_icon_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "load_dotenv", lambda *_a, **_k: None)
    settings = config.Settings(icon_cache_dir=tmp_path / "my_icons")

    settings.save_to_env(tmp_path / ".env")

    lines = (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
    assert f"ICON_CACHE_DIR={tmp_path / 'my_icons'}" in lines
