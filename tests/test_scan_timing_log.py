"""The flicker debug log has millisecond timestamps and names timer hides."""

import io
import logging
import re
from types import SimpleNamespace

from arc_helper.logging_config import console_formatter
from arc_helper.overlay import OverlayWindow


class _FakeWindow:
    """Records after() timers so the test can fire them by hand."""

    def __init__(self):
        self.timers: dict[str, object] = {}

    def after(self, ms, callback):  # noqa: ARG002
        timer_id = f"after#{len(self.timers)}"
        self.timers[timer_id] = callback
        return timer_id

    def after_cancel(self, timer_id):
        self.timers.pop(timer_id, None)

    def withdraw(self):
        pass


def test_popup_timer_hide_logged_with_milliseconds():
    popup = OverlayWindow.__new__(OverlayWindow)
    popup.window = _FakeWindow()
    popup.settings = SimpleNamespace(overlay=SimpleNamespace(display_time=2.5))
    popup._hide_after_id = None  # noqa: SLF001 - test seam
    popup._current_item = "Anvil IV"  # noqa: SLF001

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(console_formatter())
    logger = logging.getLogger("arc_helper")
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        popup._reschedule_hide()  # noqa: SLF001 - the module's own scheduling
        (fire,) = popup.window.timers.values()
        fire()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)

    lines = stream.getvalue().splitlines()
    stamp = r"^\d{2}:\d{2}:\d{2}\.\d{3} \[DEBUG\] "
    assert re.match(stamp + r"popup hide timer armed delay=2500 ms$", lines[0])
    assert re.match(stamp + r"popup hide reason=timer item=Anvil IV$", lines[1])
