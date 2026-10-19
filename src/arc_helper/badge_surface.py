"""BadgeSurface: ARLO's handle on the badge helper process.

The helper (badge_helper.py) draws gold slot borders, price tags and greyed
icons on one full-screen, transparent, click-through, topmost window. It
needs gi with GTK 3, cairo and python-xlib, which only the system Python
has, so this module finds that Python, starts the helper under it and sends
it one JSON line per draw, clear or translate.

draw, clear and translate never block their caller: they leave the message
for a writer thread, started with the helper, which sends only the newest
draw or clear. A translate is sent after the draw or translates still
waiting, as it moves what they draw. A helper that stops reading for
HANDOFF_STALL_S is killed and badges are turned off.

Every failure is soft: without a suitable Python, or after the helper dies,
BadgeSurface logs one warning and does nothing, so ARLO never crashes over
badges. Rects are physical screen pixels as grid.find_slots returns them.
"""

import contextlib
import json
import shutil
import subprocess
import threading
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

from arc_helper.cell_border import HOVER_GOLD
from arc_helper.config import get_screen_resolution
from arc_helper.config import logger

HELPER = Path(__file__).resolve().with_name("badge_helper.py")
SYSTEM_PYTHON = "/usr/bin/python3"
IMPORT_CHECK = (
    "import gi\n"
    "gi.require_version('Gtk', '3.0')\n"
    "from gi.repository import Gtk\n"
    "import cairo\n"
    "import Xlib\n"
)
CHECK_TIMEOUT_S = 10
STOP_TIMEOUT_S = 2
HANDOFF_STALL_S = 2.0


@dataclass(frozen=True)
class Border:
    """A rounded outline whose outer edge lies on rect."""

    rect: tuple[int, int, int, int]
    color: str = HOVER_GOLD
    width: int = 3
    radius: int = 4
    kind: str = "border"


@dataclass(frozen=True)
class Tag:
    """A short text on a dark pill in one corner of rect (tl, tr, bl or br).

    check puts a small green check mark before the text, and outline, an RGB
    colour, draws a 2 px rounded outline around the pill.
    """

    rect: tuple[int, int, int, int]
    text: str
    corner: str = "bl"
    fg: str = "#ffffff"
    bg: str = "#141414d9"
    check: bool = False
    outline: tuple[int, int, int] | None = None
    kind: str = "tag"


@dataclass(frozen=True)
class Patch:
    """An opaque fill on a tile's bottom bar, with the worth on it.

    rect is the bar in absolute physical px. The helper sizes the fill to the
    text, at least to icon_right + 4 and never past patch_right, and rounds
    its bottom-left corner by radius. The text is drawn in fg ('#rrggbb'),
    in the game's cream count ink when fg is None.
    """

    rect: tuple[int, int, int, int]
    icon_right: int
    patch_right: int
    colour: tuple[int, int, int]
    text: str
    radius: int = 6
    fg: str | None = None
    kind: str = "patch"


@dataclass(frozen=True)
class Icon:
    """An image file drawn greyed and dimmed inside rect, under every other kind.

    path is a PNG the helper loads and caches. It is drawn in luminance only
    at alpha, fitted and centred inside rect less inset (a share of the
    rect's width and height) on each side.
    """

    rect: tuple[int, int, int, int]
    path: str
    alpha: float = 0.35
    inset: float = 0.12
    kind: str = "icon"


Badge = Border | Tag | Patch | Icon


def encode_message(
    op: str,
    badges: list[Badge] = (),
    region: tuple[int, int, int, int] | None = None,
    dy: int = 0,
) -> str:
    """One JSON line for the helper: op is 'draw', 'clear' or 'translate'.

    A translate moves the drawn badges inside region by dy px.
    """
    message: dict = {"op": op}
    if op == "draw":
        message["badges"] = [
            {**asdict(b), "rect": [int(v) for v in b.rect]} for b in badges
        ]
    elif op == "translate":
        message["region"] = [int(v) for v in region]
        message["dy"] = int(dy)
    return json.dumps(message, ensure_ascii=False) + "\n"


def _has_imports(python: str) -> bool:
    try:
        result = subprocess.run(
            [python, "-c", IMPORT_CHECK],
            capture_output=True,
            timeout=CHECK_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def find_system_python() -> str | None:
    """The first of /usr/bin/python3 and python3 on PATH that imports gi, cairo, Xlib."""
    candidates = [SYSTEM_PYTHON, shutil.which("python3")]
    seen = set()
    for python in candidates:
        if not python or python in seen or not Path(python).exists():
            continue
        seen.add(python)
        if _has_imports(python):
            return python
    return None


def physical_size_args() -> list[str]:
    """The helper's --physical W H arguments, or none when the size is unknown."""
    try:
        width, height = get_screen_resolution()
    except Exception as e:  # noqa: BLE001 - the helper falls back to GTK's factor
        logger.warning("Badge surface: screen size unknown (%s)", e)
        return []
    if width <= 0 or height <= 0:
        return []
    return ["--physical", str(int(width)), str(int(height))]


class BadgeSurface:
    """Start, feed and stop the badge helper. Safe to call from any thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Wakes the writer for a new message or a stop, and the watchdog when
        # a write starts or ends
        self._wake = threading.Condition(self._lock)
        self._proc: subprocess.Popen | None = None
        self._python: str | None = None
        self._disabled = False
        self._pending: str | None = None
        self._write_started: float | None = None
        self._stopping = False
        self._writer: threading.Thread | None = None
        self._watchdog: threading.Thread | None = None

    @property
    def available(self) -> bool:
        """True while the helper is running and accepting messages."""
        proc = self._proc
        return proc is not None and proc.poll() is None

    def _disable(self, reason: str) -> None:
        if not self._disabled:
            logger.warning("Badge surface off: %s", reason)
        self._disabled = True

    def _python_or_disable(self) -> str | None:
        if self._python is None and not self._disabled:
            self._python = find_system_python()
            if self._python is None:
                self._disable(
                    "no system Python with gi (GTK 3), cairo and python-xlib was found",
                )
        return self._python

    def start(self) -> bool:
        """Start the helper. False, and a no-op surface, when it cannot run."""
        with self._lock:
            if self.available:
                return True
            if self._disabled:
                return False
            python = self._python_or_disable()
            if python is None:
                return False
            try:
                proc = subprocess.Popen(
                    [python, str(HELPER), "--live", *physical_size_args()],
                    stdin=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                )
            except OSError as e:
                self._disable(f"the helper did not start ({e})")
                return False
            self._proc = proc
            self._pending = None
            self._write_started = None
            self._stopping = False
            self._writer = threading.Thread(
                target=self._write_loop,
                args=(proc,),
                name="badge-writer",
                daemon=True,
            )
            self._watchdog = threading.Thread(
                target=self._watch_loop,
                args=(proc,),
                name="badge-watchdog",
                daemon=True,
            )
            self._writer.start()
            self._watchdog.start()
            return True

    def _lost(self, proc: subprocess.Popen, reason: str) -> None:
        """Turn badges off after proc failed. The caller holds the lock."""
        self._disable(reason)
        if self._proc is proc:
            self._proc = None
            self._pending = None
        self._wake.notify_all()

    def _write_loop(self, proc: subprocess.Popen) -> None:
        """Send the newest message to proc until stop or a failure."""
        while True:
            with self._wake:
                while self._pending is None and not self._stopping:
                    if self._proc is not proc:
                        return
                    self._wake.wait()
                if self._stopping or self._proc is not proc:
                    return
                line, self._pending = self._pending, None
                code = proc.poll()
                if code is not None:
                    self._lost(proc, f"the helper exited with code {code}")
                    return
                self._write_started = time.monotonic()
                self._wake.notify_all()
            try:
                proc.stdin.write(line)
                proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as e:
                with self._wake:
                    self._write_started = None
                    # stop() kills a stuck helper on purpose, that is no failure
                    if not self._stopping:
                        self._lost(proc, f"the helper stopped ({e})")
                return
            with self._wake:
                self._write_started = None
                self._wake.notify_all()

    def _watch_loop(self, proc: subprocess.Popen) -> None:
        """Kill proc when one write to it takes longer than HANDOFF_STALL_S."""
        with self._wake:
            while not self._stopping and self._proc is proc:
                started = self._write_started
                if started is None:
                    self._wake.wait()
                    continue
                left = started + HANDOFF_STALL_S - time.monotonic()
                if left > 0:
                    self._wake.wait(left)
                    continue
                self._lost(
                    proc,
                    f"the helper stalled, a message was not read within "
                    f"{HANDOFF_STALL_S} s",
                )
                break
            else:
                return
        with contextlib.suppress(OSError):
            proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=STOP_TIMEOUT_S)

    def _send(self, line: str, *, after: bool = False) -> None:
        with self._wake:
            if self._proc is None or self._disabled or self._stopping:
                return
            if after and self._pending is not None:
                # A translate moves what the waiting messages draw, so it follows them
                self._pending += line
            else:
                # Latest wins: a message the writer has not taken yet is replaced
                self._pending = line
            self._wake.notify_all()

    def draw(self, badges: list[Badge]) -> None:
        """Replace the drawn badges with these. Returns without waiting."""
        self._send(encode_message("draw", badges))

    def clear(self) -> None:
        """Remove every badge. Returns without waiting."""
        self._send(encode_message("clear"))

    def translate(self, region: tuple[int, int, int, int], dy: int) -> None:
        """Move the drawn badges inside region by dy px, clipped to it. Returns at once.

        One small message, sent after a draw still waiting. A later draw or
        clear replaces it.
        """
        self._send(encode_message("translate", region=region, dy=dy), after=True)

    def stop(self) -> None:
        """End the writer and the helper, which exits when its stdin closes.

        A writer stuck on a stalled helper is given STOP_TIMEOUT_S, then the
        helper is killed to free it.
        """
        with self._wake:
            proc, self._proc = self._proc, None
            writer, self._writer = self._writer, None
            self._watchdog = None
            self._pending = None
            self._stopping = True
            self._wake.notify_all()
        if writer is not None:
            writer.join(STOP_TIMEOUT_S)
            if writer.is_alive() and proc is not None:
                with contextlib.suppress(OSError):
                    proc.kill()
                writer.join(STOP_TIMEOUT_S)
        if proc is None:
            return
        # Only close stdin once no write to it is underway
        if writer is None or not writer.is_alive():
            with contextlib.suppress(OSError, ValueError):
                proc.stdin.close()
        try:
            proc.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def render_png(
    badges: list[Badge],
    size: tuple[int, int],
    out: Path,
    python: str | None = None,
) -> bool:
    """Draw badges on a transparent PNG with the helper's own cairo code.

    Needs no display. False when no suitable system Python exists or the
    helper fails (its error is logged).
    """
    python = python or find_system_python()
    if python is None:
        logger.warning("Badge PNG not drawn: no system Python with gi and cairo")
        return False
    width, height = size
    cmd = [python, str(HELPER), "--png", str(out), "--size", str(width), str(height)]
    result = subprocess.run(
        cmd,
        input=encode_message("draw", badges),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode != 0:
        logger.warning("Badge PNG not drawn: %s", result.stderr.strip())
        return False
    return True
