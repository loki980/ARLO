"""Shared test setup: no test ever reaches the real screen.

Code under test asks arc_helper.screen.get_backend for the screen (through
get_screen_resolution, for one). On Wayland the real backend opens the
ScreenCast portal and waits for the user to approve it, which hangs the
test and can put a dialog on the desktop. Every test therefore starts with a
fake 3840x2160 backend in place. A test that needs another backend still
replaces get_backend in its own module, as before.
"""

import pytest
from PIL import Image

from arc_helper import screen
from arc_helper.screen.base import Point


class FakeScreenBackend:
    """A 3840x2160 screen with nothing on it and the cursor at the origin."""

    tk_scale = 1.0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    @staticmethod
    def grab(bbox: tuple[int, int, int, int]) -> Image.Image:
        left, top, right, bottom = bbox
        return Image.new("RGB", (max(1, right - left), max(1, bottom - top)))

    @staticmethod
    def cursor_position() -> Point:
        return Point(x=0, y=0)

    @staticmethod
    def resolution() -> tuple[int, int]:
        return (3840, 2160)


@pytest.fixture(autouse=True)
def fake_screen_backend(monkeypatch):
    """Put the fake backend in place of the real one for every test."""
    backend = FakeScreenBackend()
    monkeypatch.setattr(screen, "_backend", backend)
    return backend
