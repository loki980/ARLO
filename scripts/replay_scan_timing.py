# ruff: noqa: INP001, SLF001
"""Replay saved captures through the scan loop's card tracking.

Usage: uv run python scripts/replay_scan_timing.py <captures folder> [list.txt]

Feeds the captures in order (the files named in list.txt, else every PNG in
the folder, by name) through the real Scanner scan loop on a fake clock set
from each file's time stamp. Each cycle locates the card with the real card
search (crop_card, the cursor assumed at the image centre) and builds the
real card signature. Reading is stubbed: the read returns the name from the
file name (a capture saved as none is a failed read), so the replay shows
which cycles would read and which would not.

Prints one row per capture: file, card state (new, same, gone, hide, none)
and the work done (read or no read), then the read count before (one per
capture, and how many of those had a card) and after.

Needs no display, no Tesseract and no items.db.
"""

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

import arc_helper.main as main_module
from arc_helper.database import Item
from arc_helper.main import Scanner
from arc_helper.ocr import card_signature
from arc_helper.ocr import crop_card
from arc_helper.tooltip_layout import Point


def _stamp(path: Path) -> float:
    """Seconds of the capture's time stamp, from its 20261014-210133-924 name."""
    date, clock, millis = path.name.split("_", 1)[0].split("-")
    when = datetime.strptime(f"{date}{clock}", "%Y%m%d%H%M%S")  # noqa: DTZ007
    return when.timestamp() + int(millis) / 1000


def _label(path: Path) -> str:
    return path.stem.split("_", 1)[1]


class _Clock:
    now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _ReplayEngine:
    """Locates the card in each capture in turn, reads by file name."""

    last_tier_vote = None

    def __init__(self, files: list[Path], clock: _Clock):
        self.files = files
        self.clock = clock
        self.index = -1
        self.read_this_step = False
        self.card_found = False
        self._last_card = None
        self._last_origin = None
        self.rows: list[tuple[str, bool, bool]] = []

    def locate_card_at_cursor(self):
        self._finish_step()
        self.index += 1
        path = self.files[self.index]
        self.clock.now = _stamp(path)
        image = Image.open(path).convert("RGB")
        self._last_origin = Point(0, 0)
        found = crop_card(image, Point(image.width // 2, image.height // 2))
        self._last_card = found[1] if found else None
        self.card_found = found is not None
        self.read_this_step = False
        if found is None:
            return None
        return card_signature(image, found[1], self._last_origin)

    def read_located_card(self) -> str | None:
        """The name from the file name, None for a capture saved as none."""
        self.read_this_step = True
        label = _label(self.files[self.index])
        return None if label == "none" else label

    def _finish_step(self) -> None:
        if self.index >= 0:
            self.rows.append(
                (self.files[self.index].name, self.card_found, self.read_this_step)
            )

    @staticmethod
    def layout_from_last_capture():
        return None


class _Hotkey:
    """Held until every capture has been fed, then released."""

    def __init__(self, engine: _ReplayEngine, scanner_ref: list):
        self.engine = engine
        self.scanner_ref = scanner_ref

    def is_held(self) -> bool:
        if self.engine.index + 1 >= len(self.engine.files):
            self.scanner_ref[0]._running = False
            return False
        return True

    def stop(self) -> None:
        pass


class _Db:
    @staticmethod
    def lookup(name, tier_vote=None):  # noqa: ARG004
        return Item(name=name, action="Keep")

    @staticmethod
    def recycle_outputs(name):  # noqa: ARG004
        return []


def _settings():
    return SimpleNamespace(
        scan=SimpleNamespace(tooltip_scan_interval=0.3, trigger_scan_interval=0.5),
        overlay=SimpleNamespace(display_time=4.0, cooldown=2.0, tooltip_chips=False),
        show_ended_projects=False,
        trigger_region=None,
        trigger_region2=None,
    )


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    folder = Path(sys.argv[1])
    if len(sys.argv) > 2:
        names = Path(sys.argv[2]).read_text(encoding="utf-8").split()
        files = [folder / name for name in names]
    else:
        files = sorted(folder.glob("*.png"))

    clock = _Clock()
    engine = _ReplayEngine(files, clock)
    ref: list = []
    scanner = Scanner(root=None, overlay=None, status=None, db=_Db(), hotkey=None)
    scanner.hotkey = _Hotkey(engine, ref)
    ref.append(scanner)
    scanner._running = True
    scanner._verdict_for = lambda *_: None

    states: list[str] = []
    cycle = scanner._scan_cycle

    def traced_cycle(ocr, settings):
        cycle(ocr, settings)
        states.append(scanner._card_state)

    scanner._scan_cycle = traced_cycle
    main_module.get_settings = _settings
    main_module.get_ocr_engine = lambda: engine
    main_module.time.monotonic = clock.monotonic
    main_module.time.sleep = clock.sleep
    scanner._scan_loop()
    engine._finish_step()

    print(f"{'capture':<64} {'card':<5} {'state':<5} work")
    for (name, found, read), state in zip(engine.rows, states, strict=True):
        work = "read" if read else "no read"
        print(f"{name:<64} {'yes' if found else 'no':<5} {state:<5} {work}")
    with_card = sum(found for _, found, _ in engine.rows)
    reads = sum(read for _, _, read in engine.rows)
    print()
    print(f"captures: {len(engine.rows)}, with a card: {with_card}")
    print(f"reads before (one per capture): {len(engine.rows)}, of a card: {with_card}")
    print(f"reads after: {reads}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
