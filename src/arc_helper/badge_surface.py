"""BadgeSurface: ARLO's handle on the badge helper process.

The helper (badge_helper.py) draws gold slot borders and price tags on one
full-screen, transparent, click-through, topmost window. It needs gi with
GTK 3, cairo and python-xlib, which only the system Python has, so this
module finds that Python, starts the helper under it and sends it one JSON
line per draw or clear.

Every failure is soft: without a suitable Python, or after the helper dies,
BadgeSurface logs one warning and does nothing, so ARLO never crashes over
badges. Rects are physical screen pixels as grid.find_slots returns them.
"""

import contextlib
import json
import shutil
import subprocess
import threading
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
    """An opaque bar-coloured fill on a tile's bottom bar, with the worth on it.

    rect is the bar in absolute physical px. The helper sizes the fill to the
    text, at least to icon_right + 4 and never past patch_right, and rounds
    its bottom-left corner by radius. The text is drawn white.
    """

    rect: tuple[int, int, int, int]
    icon_right: int
    patch_right: int
    colour: tuple[int, int, int]
    text: str
    radius: int = 6
    kind: str = "patch"


Badge = Border | Tag | Patch


def encode_message(op: str, badges: list[Badge] = ()) -> str:
    """One JSON line for the helper: op is 'draw' or 'clear'."""
    message: dict = {"op": op}
    if op == "draw":
        message["badges"] = [
            {**asdict(b), "rect": [int(v) for v in b.rect]} for b in badges
        ]
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
        self._proc: subprocess.Popen | None = None
        self._python: str | None = None
        self._disabled = False

    @property
    def available(self) -> bool:
        """True while the helper is running and accepting messages."""
        return self._proc is not None and self._proc.poll() is None

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
                self._proc = subprocess.Popen(
                    [python, str(HELPER), "--live", *physical_size_args()],
                    stdin=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                )
            except OSError as e:
                self._disable(f"the helper did not start ({e})")
                return False
            return True

    def _send(self, line: str) -> None:
        with self._lock:
            if self._proc is None:
                return
            code = self._proc.poll()
            if code is not None:
                self._disable(f"the helper exited with code {code}")
                self._proc = None
                return
            try:
                self._proc.stdin.write(line)
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as e:
                self._disable(f"the helper stopped ({e})")
                self._proc = None

    def draw(self, badges: list[Badge]) -> None:
        """Replace the drawn badges with these."""
        self._send(encode_message("draw", badges))

    def clear(self) -> None:
        """Remove every badge."""
        self._send(encode_message("clear"))

    def stop(self) -> None:
        """End the helper: it exits when its stdin closes."""
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is None:
            return
        with contextlib.suppress(OSError):
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
