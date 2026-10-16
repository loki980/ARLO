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


GRID_SIZE = (800, 480)


def _grid_frame(base: int = 40, paint=()) -> Image.Image:
    """A flat frame of one grey with each (rect, value) of paint filled in."""
    frame = Image.new("RGB", GRID_SIZE, (base, base, base))
    for rect, value in paint:
        frame.paste((value, value, value), tuple(rect))
    return frame


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def grid_run(monkeypatch, tmp_path):
    """A capture-run engine, a scanner without stash badges and a fake clock."""
    from types import SimpleNamespace

    from arc_helper import ocr as ocr_module
    from arc_helper.main import Scanner

    clock = _Clock()
    monkeypatch.setattr(
        ocr_module,
        "time",
        SimpleNamespace(
            monotonic=clock.monotonic, perf_counter=time.perf_counter, sleep=time.sleep
        ),
    )
    engine = OCREngine()
    engine.capture_library = True
    engine.capture_library_dir = tmp_path
    engine._last_origin = ocr_module.LayoutPoint(0, 0)  # noqa: SLF001
    engine.search_width = engine.search_height = 0
    scanner = Scanner(root=None, overlay=None, status=None, db=None, hotkey=None)
    scanner.stash = None  # stash badges off

    def feed(frames, step_s, card=None, overlays=()):
        """Offer each frame, step_s apart, as one scan cycle each."""
        for n, frame in enumerate(frames):
            clock.now = n * step_s
            engine._last_capture = frame  # noqa: SLF001
            engine._last_card = card(n) if callable(card) else card  # noqa: SLF001
            scanner.overlay_rects.publish(
                overlays(n) if callable(overlays) else overlays
            )
            scanner._feed_stash(engine)  # noqa: SLF001
            scanner._scan_periods.append(step_s)  # noqa: SLF001
        engine._capture_queue.join()  # noqa: SLF001
        return sorted(p.name.split("_", 1)[1] for p in tmp_path.iterdir())

    return SimpleNamespace(engine=engine, scanner=scanner, feed=feed, dir=tmp_path)


def test_still_grid_saves_one_frame(grid_run):
    assert grid_run.feed([_grid_frame()] * 5, 0.5) == ["grid.png"]


def test_distinct_grid_frames_half_a_second_apart_all_save(grid_run):
    frames = [_grid_frame(40 + 30 * n) for n in range(5)]
    assert len(grid_run.feed(frames, 0.5)) == 5
    assert grid_run.engine._capture_count == 5  # noqa: SLF001


def test_grid_frames_closer_than_the_gap_save_first_and_fifth(grid_run):
    saved = []
    grid_run.engine._queue_capture = lambda image, _label: saved.append(image) or True  # noqa: SLF001
    frames = [_grid_frame(40 + 30 * n) for n in range(5)]
    grid_run.feed(frames, 0.1)
    assert saved == [frames[0], frames[4]]


def test_moving_card_and_arlo_window_save_nothing_new(grid_run):
    from arc_helper.tooltip_layout import Rect

    cards = [Rect(100, 100, 200, 200), Rect(500, 250, 600, 350)]
    windows = [Rect(650, 20, 750, 80), Rect(20, 380, 120, 460)]
    frames = [_grid_frame(paint=[(cards[n], 250), (windows[n], 180)]) for n in range(2)]
    saved = grid_run.feed(
        frames, 0.5, card=lambda n: cards[n], overlays=lambda n: [windows[n]]
    )
    assert saved == ["grid.png"]


def test_card_only_change_saves_nothing_new(grid_run):
    from arc_helper.tooltip_layout import Rect

    card = Rect(300, 150, 450, 330)
    frames = [_grid_frame(paint=[(card, value)]) for value in (250, 120, 200)]
    assert grid_run.feed(frames, 0.5, card=card) == ["grid.png"]


def test_limit_stops_grid_saves(grid_run):
    grid_run.engine.capture_library_limit = 3
    frames = [_grid_frame(40 + 30 * n) for n in range(6)]
    assert len(grid_run.feed(frames, 0.5)) == 3
    assert grid_run.engine._capture_count == 3  # noqa: SLF001


def test_no_grid_frames_without_a_capture_run(grid_run):
    offered = []
    grid_run.engine.capture_library = False
    grid_run.engine._queue_capture = lambda *a: offered.append(a) or True  # noqa: SLF001
    frames = [_grid_frame(40 + 30 * n) for n in range(3)]
    assert grid_run.feed(frames, 0.5) == []
    assert offered == []
