# ruff: noqa: INP001
"""Probe the badge hand-off against a child that stops reading its stdin.

BadgeSurface is started against a fake child (a Python one-liner that sleeps
5 s before it reads stdin), never the real badge helper and never a window.
The script times 100 draw calls while the child is not reading, waits past
HANDOFF_STALL_S and writes what it saw to the file named on the command line.
The timings are observations, not pass or fail thresholds.

    uv run python scripts/probe_badge_handoff.py OUT.txt
"""

import logging
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

from arc_helper import badge_surface
from arc_helper.badge_surface import BadgeSurface
from arc_helper.badge_surface import Border
from arc_helper.badge_surface import Tag
from arc_helper.config import logger

CHILD = "import sys, time; time.sleep(5); sys.stdin.read()"
DRAWS = 100
# Enough badges that one message is far bigger than a pipe buffer (64 KiB),
# so the writer really blocks on the child that is not reading
BADGES = 1500


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def main(out: Path) -> None:
    real_popen = subprocess.Popen
    children = []

    def fake_popen(_cmd, **kwargs):
        child = real_popen([sys.executable, "-c", CHILD], **kwargs)
        children.append(child)
        return child

    badge_surface.subprocess.Popen = fake_popen
    badge_surface.find_system_python = lambda: sys.executable
    # The real lookup can open a screen capture portal, which a probe must not
    badge_surface.get_screen_resolution = lambda: (3840, 2160)
    records = _Records()
    logger.addHandler(records)

    surface = BadgeSurface()
    started = surface.start()
    child = children[-1]
    lines = [
        f"child: {sys.executable} -c {CHILD!r}",
        f"started: {started}",
        f"HANDOFF_STALL_S: {badge_surface.HANDOFF_STALL_S}",
    ]
    badges = [
        Tag((i, i, i + 50, i + 50), f"{i} coins", corner="br") for i in range(BADGES)
    ]
    message_bytes = len(badge_surface.encode_message("draw", badges).encode())
    lines.append(f"draw message size: {message_bytes} bytes")

    times = []
    t0 = time.monotonic()
    for i in range(DRAWS):
        call = [Border((i, 0, i + 10, 10)), *badges]
        begin = time.perf_counter()
        surface.draw(call)
        times.append(time.perf_counter() - begin)
    lines.extend(
        [
            f"draw calls: {DRAWS}",
            f"median draw call: {statistics.median(times) * 1e6:.1f} us",
            f"max draw call: {max(times) * 1e6:.1f} us",
            f"all draws returned after: {time.monotonic() - t0:.3f} s",
        ]
    )
    # draw encodes its message in the caller, so most of a call is this
    encode = []
    for _ in range(20):
        begin = time.perf_counter()
        badge_surface.encode_message("draw", badges)
        encode.append(time.perf_counter() - begin)
    lines.extend(
        [
            f"encode_message alone, median of 20: {statistics.median(encode) * 1e6:.1f} us",
            f"child still running after the draws: {child.poll() is None}",
        ]
    )

    wait_s = badge_surface.HANDOFF_STALL_S + 1.0
    time.sleep(wait_s)
    elapsed = time.monotonic() - t0
    writers = [t for t in threading.enumerate() if t.name == "badge-writer"]
    lines.extend(
        [
            f"waited until: {elapsed:.3f} s after the first draw",
            f"child return code: {child.poll()}",
            f"surface available: {surface.available}",
            f"writer thread alive: {any(t.is_alive() for t in writers)}",
        ]
    )

    begin = time.perf_counter()
    surface.draw([Border((0, 0, 10, 10))])
    surface.clear()
    lines.append(
        f"draw and clear after the stall took: "
        f"{(time.perf_counter() - begin) * 1e6:.1f} us"
    )
    begin = time.monotonic()
    surface.stop()
    lines.extend(
        [
            f"stop took: {time.monotonic() - begin:.3f} s",
            f"warnings ({len(records.messages)}):",
        ]
    )
    lines.extend(f"  {m}" for m in records.messages)
    logger.removeHandler(records)

    text = "\n".join(lines) + "\n"
    out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
