"""Tests for the capture library: on only with --capture, limited, saved off-thread."""

import argparse
import logging
import threading
import time

import pytest
from PIL import Image

from arc_helper import config as config_module
from arc_helper import main as main_module
from arc_helper.config import SettingsManager
from arc_helper.config import capture_library_in_env
from arc_helper.config import get_settings
from arc_helper.ocr import CAPTURE_QUEUE_SIZE
from arc_helper.ocr import OCREngine


@pytest.fixture(autouse=True)
def fresh_settings(monkeypatch, tmp_path):
    """Each test starts with no settings singleton and the library off."""
    monkeypatch.setattr(SettingsManager, "_instance", None)
    monkeypatch.setattr(SettingsManager, "_capture_library", False)
    monkeypatch.setattr(SettingsManager, "_capture_library_limit", 300)
    monkeypatch.delenv("CAPTURE_LIBRARY", raising=False)
    # No real .env is consulted for the warning
    monkeypatch.setattr(config_module, "APP_DIR", tmp_path)


def test_capture_library_env_is_ignored_by_settings_and_engine(monkeypatch):
    monkeypatch.setenv("CAPTURE_LIBRARY", "true")
    monkeypatch.setenv("CAPTURE_LIBRARY_LIMIT", "5")
    assert get_settings().capture_library is False
    assert get_settings().capture_library_limit == 300
    assert OCREngine().capture_library is False


def test_capture_library_in_env_or_env_file_logs_one_warning(
    monkeypatch, tmp_path, caplog
):
    no_env = argparse.Namespace(capture=None)
    with caplog.at_level(logging.INFO, logger="arc_helper"):
        main_module.apply_capture_args(no_env)
    assert not caplog.records

    (tmp_path / ".env").write_text("CAPTURE_LIBRARY=true\n")
    assert capture_library_in_env()
    with caplog.at_level(logging.INFO, logger="arc_helper"):
        main_module.apply_capture_args(no_env)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [r.getMessage() for r in warnings] == [
        "CAPTURE_LIBRARY in .env is ignored, start ARLO with --capture to collect captures"
    ]
    assert get_settings().capture_library is False

    (tmp_path / ".env").unlink()
    monkeypatch.setenv("CAPTURE_LIBRARY", "false")
    assert capture_library_in_env()


def test_save_to_env_drops_capture_library_keeps_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(config_module, "load_dotenv", lambda *_a, **_k: None)
    env_path = tmp_path / "saved.env"
    get_settings().save_to_env(env_path)
    lines = env_path.read_text().splitlines()
    assert not [line for line in lines if line.startswith("CAPTURE_LIBRARY=")]
    assert [line for line in lines if line.startswith("CAPTURE_LIBRARY_DIR=")]


def test_capture_flag_turns_library_on_for_the_run(caplog):
    assert main_module.parse_args([]).capture is None
    assert main_module.parse_args(["--capture"]).capture == 300
    args = main_module.parse_args(["--capture", "25"])
    with caplog.at_level(logging.INFO, logger="arc_helper"):
        main_module.apply_capture_args(args)
    settings = get_settings()
    assert settings.capture_library is True
    assert settings.capture_library_limit == 25
    info = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    expected = (
        "Capture library on for this run: up to 25 images in "
        f"{settings.capture_library_dir}"
    )
    assert info == [expected]
    # A settings reload, as after applying a profile, keeps the switch
    assert SettingsManager.reload().capture_library is True
    engine = OCREngine()
    assert engine.capture_library is True
    assert engine.capture_library_limit == 25


@pytest.mark.parametrize("bad", [["--capture", "0"], ["--capture", "-1"], ["--bogus"]])
def test_bad_arguments_exit_before_anything_starts(monkeypatch, bad):
    calls = []

    def must_not_run(*_args, **_kwargs):
        calls.append("ran")

    monkeypatch.setattr(main_module, "check_first_run", must_not_run)
    monkeypatch.setattr(main_module, "get_ocr_engine", must_not_run)
    monkeypatch.setattr(main_module, "Application", must_not_run)
    with pytest.raises(SystemExit) as exit_info:
        main_module.main(bad)
    assert exit_info.value.code == 2
    assert calls == []


def _image() -> Image.Image:
    return Image.new("RGB", (8, 8))


def test_full_queue_drops_capture_without_blocking(monkeypatch, tmp_path):
    engine = OCREngine()
    engine.capture_library_dir = tmp_path
    started = threading.Event()
    release = threading.Event()

    def blocked_write(_image, _stamp, _label):
        started.set()
        release.wait(5)

    monkeypatch.setattr(engine, "_write_capture", blocked_write)
    engine._save_to_library(_image(), "FIRST")  # noqa: SLF001
    assert started.wait(5)  # the writer is stuck on the first image
    for n in range(CAPTURE_QUEUE_SIZE):
        engine._save_to_library(_image(), f"QUEUED {n}")  # noqa: SLF001
    assert engine._capture_queue.full()  # noqa: SLF001

    begin = time.monotonic()
    engine._save_to_library(_image(), "DROPPED")  # noqa: SLF001
    assert time.monotonic() - begin < 0.5
    assert engine._capture_count == 1 + CAPTURE_QUEUE_SIZE  # noqa: SLF001

    release.set()
    engine._capture_queue.join()  # noqa: SLF001


def test_capture_count_stops_at_the_limit(tmp_path, caplog):
    engine = OCREngine()
    engine.capture_library_dir = tmp_path
    engine.capture_library_limit = 3
    with caplog.at_level(logging.INFO, logger="arc_helper"):
        for n in range(6):
            engine._save_to_library(_image(), f"ITEM {n}")  # noqa: SLF001
            engine._capture_queue.join()  # noqa: SLF001
    assert engine._capture_count == 3  # noqa: SLF001
    assert sorted(p.name.split("_", 1)[1] for p in tmp_path.iterdir()) == [
        "ITEM-0.png",
        "ITEM-1.png",
        "ITEM-2.png",
    ]
    reached = [r for r in caplog.records if "limit of 3 reached" in r.getMessage()]
    assert len(reached) == 1
