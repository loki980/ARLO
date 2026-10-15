"""Tests for the top-centre band search that finds the INVENTORY trigger in raid."""

from pathlib import Path

import pytest
from PIL import Image
from PIL import ImageDraw

from arc_helper import ocr as ocr_module
from arc_helper.config import get_settings
from arc_helper.ocr import OCREngine

FIXTURES = Path(__file__).parent / "fixtures"
SCREEN = (3840, 2160)
BAND_LEFT = int(SCREEN[0] * OCREngine.TRIGGER_BAND_LEFT)


class Box:
    """A configured trigger box in physical pixels."""

    def __init__(self, x, y, width, height):
        self.x, self.y, self.width, self.height = x, y, width, height
        self.bbox = (x, y, x + width, y + height)


# The calibrated stash view boxes of the machine the fixtures come from
BOXES = [Box(1523, 46, 236, 35), Box(1302, 66, 244, 40)]


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setenv("OCR_ENGINE", "tesseract")
    monkeypatch.setattr(get_settings(), "ocr_engine", "tesseract")
    return OCREngine()


def use_screen(monkeypatch, screen: Image.Image) -> list[tuple]:
    """Serve captures from a fake screen. Returns the list of grabbed bboxes."""
    grabs: list[tuple] = []

    def grab(bbox):
        grabs.append(bbox)
        return screen.crop(bbox)

    monkeypatch.setattr(ocr_module, "grab_screen", grab)
    monkeypatch.setattr(ocr_module, "get_screen_resolution", lambda: SCREEN)
    return grabs


def raid_screen() -> Image.Image:
    """A 4K screen with the in-raid tab bar (INVENTORY, MAP, LOGBOOK, SYSTEM)."""
    screen = Image.new("RGB", SCREEN, (24, 32, 56))
    screen.paste(Image.open(FIXTURES / "raid_top_band.png"), (BAND_LEFT, 0))
    return screen


def test_stash_view_triggers_through_the_configured_box(engine, monkeypatch):
    # The strip is the top of a stash view capture, where the tab sits in box 1
    screen = Image.new("RGB", SCREEN)
    screen.paste(Image.open(FIXTURES / "stash_top_strip.png"), (0, 0))
    grabs = use_screen(monkeypatch, screen)

    assert engine.check_trigger_any(BOXES)

    # Only the configured box was read, the band was never captured
    assert grabs == [BOXES[0].bbox]


def test_raid_inventory_misses_the_boxes_and_triggers_through_the_band(
    engine, monkeypatch
):
    use_screen(monkeypatch, raid_screen())

    # The extra MAP tab shifts the word so the boxes cut it off
    assert not any(engine.check_trigger(box) for box in BOXES)
    assert engine.check_trigger_any(BOXES)


def test_band_without_the_word_does_not_trigger(engine, monkeypatch):
    screen = raid_screen()
    # Paint over the INVENTORY pill (left 40 percent of the fixture) and leave the
    # MAP, LOGBOOK and SYSTEM tabs
    ImageDraw.Draw(screen).rectangle(
        (BAND_LEFT, 0, BAND_LEFT + 640, 156), fill=(24, 32, 56)
    )
    use_screen(monkeypatch, screen)

    assert not engine.check_trigger_any(BOXES)
    # A miss keeps what each place read, for the trigger miss log line
    places = [place for place, _ in engine.last_trigger_reads]
    assert places == ["region1", "region2", "band"]
    assert "LOGBOOK" in engine.last_trigger_reads[-1][1]


def test_empty_screen_does_not_trigger(engine, monkeypatch):
    use_screen(monkeypatch, Image.new("RGB", SCREEN, (10, 10, 10)))

    assert not engine.check_trigger_any(BOXES)
