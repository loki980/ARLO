"""Tests for the item database updater (update_db.py). Network is always faked."""

import csv
import json
import sys
from pathlib import Path

import pytest
import requests

import update_db
from arc_helper.database import Database
from arc_helper.database import RecycleOutput

FIXTURES = Path(__file__).parent / "fixtures"

WIKI_HTML = """
<table>
  <tr><th>Image</th><th>Item</th><th>Category</th><th>Rarity</th>
      <th>Recycles To</th><th>Sell Price</th><th>Stack Size</th><th>Uses</th></tr>
  <tr><td></td><td><a>Power Bank</a></td><td>Recyclable</td><td>Rare</td>
      <td>2x Battery 1x Wires</td><td>1000</td><td>3</td><td></td></tr>
  <tr><td></td><td><a>Old Lamp</a></td><td>Recyclable</td><td>Common</td>
      <td>1x Metal Parts</td><td>100</td><td>5</td><td></td></tr>
</table>
"""

METAFORGE_ITEMS = {
    "pagination": {"hasNextPage": False, "total": 2},
    "data": [
        {"id": "power-bank", "name": "Power Bank", "item_type": "Recyclable",
         "rarity": "Rare", "value": 1000, "stat_block": {"stackSize": 3}},
        {"id": "old-lamp", "name": "Old Lamp", "item_type": "Recyclable",
         "rarity": "Common", "value": 100, "stat_block": {"stackSize": 5}},
    ],
}  # fmt: skip


def _fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeResponse:
    def __init__(self, payload=None, *, status=200, text=""):
        self._payload = payload
        self.status_code = status
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            msg = f"{self.status_code} Server Error"
            raise requests.HTTPError(msg)

    def json(self):
        if self._payload is None:
            return json.loads(self.text)  # raises on invalid JSON
        return self._payload


def _needs_routes(needs):
    """Needs source URL -> fixture payload."""
    return {
        update_db.ARCTRACKER_QUESTS_URL: needs["arctracker_quests"],
        update_db.ARCTRACKER_PROJECTS_URL: needs["arctracker_projects"],
        update_db.METAFORGE_HIDEOUT_URL: needs["metaforge_hideout"],
        update_db.ARCTRACKER_HIDEOUT_URL: needs["arctracker_hideout"],
    }


def _fake_get(arctracker_response, *, metaforge_fails=False, needs=None, failures=None):
    """Route requests.get by URL. arctracker_response is a FakeResponse or an exception.

    needs is the needs_sources.json fixture. failures maps a URL to the
    FakeResponse or exception that replaces its normal answer.
    """
    needs = needs or _fixture("needs_sources.json")
    routes = _needs_routes(needs)
    failures = failures or {}

    def fake_get(url, **_kwargs):
        if url in failures:
            if isinstance(failures[url], Exception):
                raise failures[url]
            return failures[url]
        if metaforge_fails and url.startswith(update_db.METAFORGE_API_BASE):
            msg = "metaforge down"
            raise requests.ConnectionError(msg)
        if url in routes:
            return FakeResponse(routes[url])
        if url == update_db.ARCTRACKER_ITEMS_URL:
            if isinstance(arctracker_response, Exception):
                raise arctracker_response
            return arctracker_response
        if url == update_db.WIKI_URL:
            return FakeResponse(text=WIKI_HTML)
        if "/items?" in url:
            data = METAFORGE_ITEMS["data"] + needs["metaforge_items"]
            return FakeResponse({**METAFORGE_ITEMS, "data": data})
        if "/quests?" in url:
            return FakeResponse(_fixture("metaforge_quests_after_rain_comes.json"))
        msg = f"unexpected URL {url}"
        raise AssertionError(msg)

    return fake_get


def _run_main(
    monkeypatch, tmp_path, arctracker_response, *, metaforge_fails=False, entry="main"
):
    items_csv = tmp_path / "items.csv"
    outputs_csv = tmp_path / "recycle_outputs.csv"
    with items_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["name", "action", "recycle_for", "keep_for", "sell_price", "stack_size"]
        )
        writer.writerow(["Power Bank", "Keep", "9x Old Text", "Manual", "1000", "3"])
    monkeypatch.setattr(update_db, "ITEMS_CSV", items_csv)
    monkeypatch.setattr(update_db, "ITEMS_DB", tmp_path / "items.db")
    monkeypatch.setattr(update_db, "LAST_UPDATE_FILE", tmp_path / ".last_update")
    monkeypatch.setattr(update_db, "RECYCLE_OUTPUTS_CSV", outputs_csv)
    monkeypatch.setattr(update_db, "NEEDS_CSV", tmp_path / "needs.csv")
    fake_get = _fake_get(arctracker_response, metaforge_fails=metaforge_fails)
    monkeypatch.setattr(update_db.requests, "get", fake_get)
    monkeypatch.setattr(sys, "argv", ["update_db.py", "--merge", "--csv-only"])
    if entry == "main":
        update_db.main()
    else:
        assert update_db.auto_update(force=True)
    with items_csv.open(encoding="utf-8") as f:
        rows = {r["name"]: r for r in csv.DictReader(f)}
    return rows, outputs_csv


def test_parse_arctracker_record_has_both_maps():
    arc = update_db.parse_arctracker_items(_fixture("arctracker_items.json"))
    power_bank = arc["power bank"]
    assert power_bank.id == "power_bank"
    assert power_bank.name == "Power Bank"
    assert (power_bank.value, power_bank.stack_size) == (1000, 3)
    assert (power_bank.rarity, power_bank.item_type) == ("Rare", "Recyclable")
    assert power_bank.recycles_into == {"Battery": 2, "Wires": 2}
    assert power_bank.salvages_into == {"Battery": 2}
    assert power_bank.icon_url == "https://cdn.arctracker.io/items/v2/power_bank.png"


def test_name_matching_is_case_insensitive():
    arc = update_db.parse_arctracker_items(_fixture("arctracker_items.json"))
    recycle_map = {"power bank": "2x Battery, 2x Wires"}
    assert update_db._recycle_fields("POWER BANK", arc, recycle_map) == (  # noqa: SLF001
        "2x Battery, 2x Wires",
        True,
    )


def test_recycle_output_rows_for_power_bank(tmp_path):
    arc = update_db.parse_arctracker_items(_fixture("arctracker_items.json"))
    rows = update_db.build_recycle_output_rows(arc)
    power_bank = [r for r in rows if r["item"] == "Power Bank"]
    assert power_bank == [
        {"item": "Power Bank", "component": "Battery", "quantity": "2", "location": "field"},
        {"item": "Power Bank", "component": "Battery", "quantity": "2", "location": "home"},
        {"item": "Power Bank", "component": "Wires", "quantity": "2", "location": "home"},
    ]  # fmt: skip
    out = tmp_path / "recycle_outputs.csv"
    update_db.write_recycle_outputs_csv(rows, out)
    assert out.read_text(encoding="utf-8").splitlines()[0] == (
        "item,component,quantity,location"
    )


def test_recycle_for_format():
    assert (
        update_db.format_recycle_for({"Battery": 2, "Wires": 2})
        == "2x Battery, 2x Wires"
    )


def test_quest_parser_reads_item_name(monkeypatch):
    monkeypatch.setattr(update_db.requests, "get", _fake_get(FakeResponse({})))
    quest_map = update_db.fetch_metaforge_quest_items()
    assert quest_map["wires"] == ["After Rain Comes (5x)"]
    assert quest_map["battery"] == ["After Rain Comes (2x)"]


def test_arctracker_success_writes_outputs_and_recycle_for(monkeypatch, tmp_path):
    payload = FakeResponse(_fixture("arctracker_items.json"))
    rows, outputs_csv = _run_main(monkeypatch, tmp_path, payload)
    # arctracker home outputs replace the old text, the merged action stays
    assert rows["Power Bank"]["recycle_for"] == "2x Battery, 2x Wires"
    assert rows["Power Bank"]["action"] == "Keep"
    # no arctracker record, so the wiki text stays
    assert rows["Old Lamp"]["recycle_for"] == "1x Metal Parts"
    with outputs_csv.open(encoding="utf-8") as f:
        power_bank = [r for r in csv.DictReader(f) if r["item"] == "Power Bank"]
    assert len(power_bank) == 3


@pytest.mark.parametrize(
    "failure",
    [
        requests.ConnectionError("connection refused"),
        FakeResponse(status=503),
        FakeResponse(text="<html>not json</html>"),
    ],
    ids=["connection-error", "non-200", "invalid-json"],
)
def test_arctracker_failure_falls_back_to_wiki(monkeypatch, tmp_path, capsys, failure):
    rows, outputs_csv = _run_main(monkeypatch, tmp_path, failure)
    out = capsys.readouterr().out
    assert out.count("WARNING") == 1
    assert "Done!" in out
    # merge keeps the old text, new items take the wiki text
    assert rows["Power Bank"]["recycle_for"] == "9x Old Text"
    assert rows["Old Lamp"]["recycle_for"] == "1x Metal Parts"
    assert not outputs_csv.exists()


@pytest.mark.parametrize("entry", ["main", "auto_update"])
def test_metaforge_failure_still_uses_arctracker(monkeypatch, tmp_path, entry):
    payload = FakeResponse(_fixture("arctracker_items.json"))
    rows, outputs_csv = _run_main(
        monkeypatch, tmp_path, payload, metaforge_fails=True, entry=entry
    )
    # wiki fallback items get arctracker home outputs, manual fields stay
    assert rows["Power Bank"]["recycle_for"] == "2x Battery, 2x Wires"
    assert rows["Power Bank"]["action"] == "Keep"
    assert rows["Power Bank"]["keep_for"] == "Manual"
    assert rows["Old Lamp"]["recycle_for"] == "1x Metal Parts"
    with outputs_csv.open(encoding="utf-8") as f:
        power_bank = [r for r in csv.DictReader(f) if r["item"] == "Power Bank"]
    assert len(power_bank) == 3


def _drop_rubber_parts(payload):
    payload["items"] = [i for i in payload["items"] if i["id"] != "rubber_parts"]
    return payload


def _strip_rubber_parts_english(payload):
    for item in payload["items"]:
        if item["id"] == "rubber_parts":
            del item["name"]["en"]
    return payload


@pytest.mark.parametrize(
    "corrupt",
    [_drop_rubber_parts, _strip_rubber_parts_english],
    ids=["missing-record", "no-english-name"],
)
def test_unresolved_component_rejects_payload(monkeypatch, tmp_path, capsys, corrupt):
    payload = corrupt(_fixture("arctracker_items.json"))
    with pytest.raises(ValueError, match="rubber_parts"):
        update_db.parse_arctracker_items(payload)
    rows, outputs_csv = _run_main(monkeypatch, tmp_path, FakeResponse(payload))
    assert capsys.readouterr().out.count("WARNING") == 1
    assert not outputs_csv.exists()
    assert "rubber_parts" not in (tmp_path / "items.csv").read_text(encoding="utf-8")
    assert rows["Power Bank"]["recycle_for"] == "9x Old Text"


# needs.csv


def _resolvers(needs):
    arc = update_db.parse_arctracker_items(needs["arctracker_items"])
    arc_names = update_db.NameResolver({r.id: r.name for r in arc.values()})
    mf_names = update_db.NameResolver(
        {i["id"]: i["name"] for i in needs["metaforge_items"]}
    )
    return arc_names, mf_names


def _short(rows):
    return [(r["item"], r["source_name"], r["tier"], r["quantity"]) for r in rows]


def test_wires_crossed_quest_rows():
    needs = _fixture("needs_sources.json")
    arc_names, _ = _resolvers(needs)
    rows = update_db.parse_quest_needs(needs["arctracker_quests"], arc_names)
    assert _short(rows) == [
        ("Snitch Scanner", "Wires Crossed", "", "1"),
        ("Surveyor Vault", "Wires Crossed", "", "1"),
        ("Sentinel Firing Core", "Wires Crossed", "", "1"),
    ]
    assert {r["source_kind"] for r in rows} == {"quest"}


def test_expedition_phase_rows_have_tier_and_dates():
    needs = _fixture("needs_sources.json")
    arc_names, _ = _resolvers(needs)
    rows = update_db.parse_project_needs(needs["arctracker_projects"], arc_names)
    # phase 5 only has requirementCategories (coin value), so no rows for it
    assert _short(rows) == [
        ("Metal Parts", "Expedition 5", "1", "150"),
        ("Plastic Parts", "Expedition 5", "1", "100"),
        ("ARC Alloy", "Expedition 5", "1", "80"),
        ("Magnet", "Expedition 5", "1", "15"),
    ]
    assert {(r["starts"], r["ends"]) for r in rows} == {("2026-07-07", "2026-09-22")}


def test_gunsmith_level_4_rows_resolve_metaforge_ids():
    needs = _fixture("needs_sources.json")
    _, mf_names = _resolvers(needs)
    levels = update_db.parse_metaforge_hideout(needs["metaforge_hideout"], mf_names)
    rows = update_db.workshop_rows(levels)
    assert [r for r in _short(rows) if r[2] == "4"] == [
        ("Radial Press", "Gunsmith", "4", "3"),
        ("Magnetic Accelerator", "Gunsmith", "4", "3"),
        ("Emperor Modulator", "Gunsmith", "4", "1"),
    ]
    # MetaForge id sentinel-part resolves to its English name
    assert ("Sentinel Firing Core", "Gunsmith", "3", "4") in _short(rows)


def test_cross_check_warns_once_per_disagreeing_level(capsys):
    needs = _fixture("needs_sources.json")
    arc_names, mf_names = _resolvers(needs)
    mf = update_db.parse_metaforge_hideout(needs["metaforge_hideout"], mf_names)
    modules = needs["arctracker_hideout"]["hideoutModules"]
    agreeing = update_db.parse_arctracker_hideout(
        needs["arctracker_hideout"], arc_names
    )
    assert update_db.cross_check_workshop(mf, agreeing) == 0
    modules["weapon_bench"]["levels"][1]["requirementItemIds"][0]["quantity"] = 2
    arc = update_db.parse_arctracker_hideout(needs["arctracker_hideout"], arc_names)
    assert update_db.cross_check_workshop(mf, arc) == 1
    out = capsys.readouterr().out
    assert out.count("WARNING") == 1
    assert "Gunsmith 4" in out
    assert "radial press MetaForge 3 arctracker 2" in out


def test_unknown_item_id_is_skipped_and_counted():
    needs = _fixture("needs_sources.json")
    arc_names, _ = _resolvers(needs)
    quest = needs["arctracker_quests"]["quests"]["wires_crossed"]
    quest["requiredItemIds"].append({"itemId": "mystery_part", "quantity": 2})
    rows = update_db.parse_quest_needs(needs["arctracker_quests"], arc_names)
    assert len(rows) == 3
    assert all("mystery" not in value for r in rows for value in r.values())
    assert arc_names.unresolved == 1


def _needs_inputs():
    needs = _fixture("needs_sources.json")
    arc = update_db.parse_arctracker_items(needs["arctracker_items"])
    return arc, needs["metaforge_items"]


@pytest.mark.parametrize(
    "url",
    [
        update_db.ARCTRACKER_QUESTS_URL,
        update_db.ARCTRACKER_PROJECTS_URL,
        update_db.METAFORGE_HIDEOUT_URL,
    ],
    ids=["quests", "projects", "hideout"],
)
@pytest.mark.parametrize(
    "failure",
    [
        requests.ConnectionError("connection refused"),
        FakeResponse(status=503),
        FakeResponse(text="<html>not json</html>"),
    ],
    ids=["connection-error", "non-200", "invalid-json"],
)
def test_needs_source_failure_keeps_existing_csv(
    monkeypatch, tmp_path, capsys, url, failure
):
    needs_csv = tmp_path / "needs.csv"
    needs_csv.write_text("old,rows\n", encoding="utf-8")
    fake_get = _fake_get(FakeResponse({}), failures={url: failure})
    monkeypatch.setattr(update_db.requests, "get", fake_get)
    arc, mf_items = _needs_inputs()
    assert not update_db.update_needs_csv(arc, mf_items, needs_csv)
    out = capsys.readouterr().out
    assert out.count("WARNING") == 1
    assert needs_csv.read_text(encoding="utf-8") == "old,rows\n"


@pytest.mark.parametrize("entry", ["main", "auto_update"])
def test_update_writes_sorted_needs_csv(monkeypatch, tmp_path, entry):
    payload = _fixture("arctracker_items.json")
    payload["items"] += _fixture("needs_sources.json")["arctracker_items"]["items"]
    _run_main(monkeypatch, tmp_path, FakeResponse(payload), entry=entry)
    with (tmp_path / "needs.csv").open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            "item", "source_kind", "source_name", "tier", "quantity", "starts", "ends",
        ]  # fmt: skip
        rows = [
            (r["source_kind"], r["source_name"], r["tier"], r["item"]) for r in reader
        ]
    assert rows[0] == ("project", "Expedition 5", "1", "ARC Alloy")
    assert ("quest", "Wires Crossed", "", "Snitch Scanner") in rows
    assert ("workshop", "Gunsmith", "4", "Radial Press") in rows
    assert len(rows) == 4 + 3 + 6
    assert rows == sorted(
        rows, key=lambda r: (r[0], r[1].lower(), int(r[2] or -1), r[3].lower())
    )


OLD_HEADER = "name,action,recycle_for,keep_for,sell_price,stack_size\n"
NEW_HEADER = OLD_HEADER.rstrip("\n") + ",rarity,category\n"


def test_merge_refreshes_rarity_and_category(monkeypatch, tmp_path):
    payload = FakeResponse(_fixture("arctracker_items.json"))
    # the existing items.csv is the old format, without the two columns
    rows, _ = _run_main(monkeypatch, tmp_path, payload)
    assert rows["Power Bank"]["rarity"] == "Rare"
    assert rows["Power Bank"]["category"] == "Recyclable"
    assert rows["Power Bank"]["action"] == "Keep"  # merged fields stay


def test_load_csv_reads_old_and_new_format(tmp_path):
    db = Database(db_path=tmp_path / "items.db")
    old_csv = tmp_path / "old.csv"
    old_csv.write_text(OLD_HEADER + "Battery,Sell,,,250,15\n", encoding="utf-8")
    db.load_csv(old_csv)
    battery = db.lookup("Battery")
    assert (battery.rarity, battery.category) == (None, None)

    new_csv = tmp_path / "new.csv"
    new_csv.write_text(
        NEW_HEADER + "Battery,Sell,,,250,15,Common,Recyclable\n", encoding="utf-8"
    )
    db.load_csv(new_csv)
    battery = db.lookup("Battery")
    assert (battery.rarity, battery.category) == ("Common", "Recyclable")


def test_rebuild_database_fills_columns_and_loads_csvs(tmp_path):
    items_csv = tmp_path / "items.csv"
    items_csv.write_text(
        NEW_HEADER + "Power Bank,Recycle,2x Battery,,1000,3,Rare,Recyclable\n",
        encoding="utf-8",
    )
    (tmp_path / "recycle_outputs.csv").write_text(
        "item,component,quantity,location\n"
        "Power Bank,Battery,2,home\nPower Bank,Wires,2,home\n"
        "Power Bank,Battery,2,field\n",
        encoding="utf-8",
    )
    (tmp_path / "needs.csv").write_text(
        "item,source_kind,source_name,tier,quantity,starts,ends\n"
        "Power Bank,workshop,Gunsmith,4,3,,\n",
        encoding="utf-8",
    )

    assert update_db.rebuild_database(items_csv, tmp_path / "items.db") == 1

    db = Database(db_path=tmp_path / "items.db")
    assert db.lookup("Power Bank").rarity == "Rare"
    assert db.home_only_outputs("Power Bank") == [RecycleOutput("Wires", 2, "home")]
    assert [n.quantity for n in db.needs_for("Power Bank")] == [3]
