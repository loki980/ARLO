"""Tests for the item database updater (update_db.py). Network is always faked."""

import csv
import json
import sys
from pathlib import Path

import pytest
import requests

import update_db

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


def _fake_get(arctracker_response, *, metaforge_fails=False):
    """Route requests.get by URL. arctracker_response is a FakeResponse or an exception."""

    def fake_get(url, **_kwargs):
        if metaforge_fails and url.startswith(update_db.METAFORGE_API_BASE):
            msg = "metaforge down"
            raise requests.ConnectionError(msg)
        if url == update_db.ARCTRACKER_ITEMS_URL:
            if isinstance(arctracker_response, Exception):
                raise arctracker_response
            return arctracker_response
        if url == update_db.WIKI_URL:
            return FakeResponse(text=WIKI_HTML)
        if "/items?" in url:
            return FakeResponse(METAFORGE_ITEMS)
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
