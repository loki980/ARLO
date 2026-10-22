"""Tests for the cursor eval_captures.py uses to locate a card."""

import importlib.util
import json
from pathlib import Path

from PIL import Image

SCRIPT = Path(__file__).parent.parent / "scripts" / "eval_captures.py"


def _load():
    spec = importlib.util.spec_from_file_location("eval_captures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capture(folder: Path, sidecar: dict | None) -> tuple[Path, Image.Image]:
    image = Image.new("RGB", (200, 100))
    path = folder / "a_card.png"
    image.save(path)
    if sidecar is not None:
        path.with_suffix(".json").write_text(json.dumps(sidecar))
    return path, image


def test_sidecar_cursor_is_used(tmp_path):
    path, image = _capture(tmp_path, {"cursor": [150, 30], "origin": [0, 0]})
    cursor, source = _load()._cursor_for(path, image)  # noqa: SLF001
    assert (cursor.x, cursor.y, source) == (150, 30, "sidecar")


def test_sidecar_origin_is_taken_off_the_cursor(tmp_path):
    path, image = _capture(tmp_path, {"cursor": [1210, 810], "origin": [1000, 700]})
    cursor, source = _load()._cursor_for(path, image)  # noqa: SLF001
    assert (cursor.x, cursor.y, source) == (210, 110, "sidecar")


def test_centre_without_a_sidecar_or_with_a_null_cursor(tmp_path):
    module = _load()
    path, image = _capture(tmp_path, None)
    cursor, source = module._cursor_for(path, image)  # noqa: SLF001
    assert (cursor.x, cursor.y, source) == (100, 50, "centre")
    path, image = _capture(tmp_path, {"cursor": None, "origin": [0, 0]})
    cursor, source = module._cursor_for(path, image)  # noqa: SLF001
    assert (cursor.x, cursor.y, source) == (100, 50, "centre")
