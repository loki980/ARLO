# ruff: noqa: INP001, SLF001
"""Show the Collections hover assist on screenshots, with a fake cursor.

Usage: uv run python scripts/collections_assist_demo.py --out DIR SHOT.png...

For each screenshot of the Collections screen (BLUEPRINTS tab) the real
Scanner reads the screen and its hover check runs with a fake cursor over a
set of cells: one empty cell per map value among the empty cells, every
unnamed empty slot and one found cell. Writes to DIR/demo/<shot>/ the card
the assist queues for each (slot_NN.png) and the screenshot with the
hovered cell outlined and the card drawn in the home box where ARLO puts it
(slot_NN_on_screen.png). Prints the empty cells with their blueprint, the
card texts and the timings: the per-round hover check, the screen read, and
the time from the cursor arriving on a cell to the card being queued while
the scan loop runs (trigger OCR stubbed out).

Only the item database is read. The user database is a temporary one, so
the ownership the screen read records never reaches the real one.
"""

import argparse
import math
import statistics
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageDraw

import arc_helper.main as main_module
from arc_helper.advice_window import render_in_box
from arc_helper.collections_screen import read_collections
from arc_helper.database import Database
from arc_helper.layout_home import fixed_home
from arc_helper.main import Scanner
from arc_helper.main import ScannerState
from arc_helper.tooltip_layout import Point

REPO = Path(__file__).resolve().parent.parent
ROUNDS = 2000
ARRIVALS = 12


class _Cursor:
    """The fake backend: a cursor the demo moves."""

    at = Point(0, 0)

    def cursor_position(self):
        return self.at


class _Hotkey:
    def is_held(self):  # noqa: PLR6301
        return False

    def start(self):
        pass

    def stop(self):
        pass


class _NoTrigger:
    """The trigger OCR, stubbed: never INVENTORY, at no cost."""

    def check_trigger_any(self, regions):  # noqa: ARG002, PLR6301
        return False


def _middle(rect) -> Point:
    return Point((rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2)


def _drain(scanner) -> list[tuple]:
    out = []
    while not scanner.ui_queue.empty():
        out.append(scanner.ui_queue.get_nowait())
    return out


def _card_text(card) -> str:
    lines = [f"    title: {card.title}", f"    subtitle: {card.subtitle}"]
    lines += [f"    {label}: {value}" for label, value in card.rows]
    return "\n".join(lines)


def _pick(read, db) -> list:
    """One empty cell per map value, every unnamed empty slot, one found cell."""
    catalog = {row.name: row for row in db.blueprint_catalog()}
    picked, maps = [], set()
    for cell in read.cells:
        if cell.found:
            continue
        name = db.blueprint_at(cell.grid_index)
        if name is None:
            picked.append(cell)
            continue
        info = catalog.get(name)
        key = info.map if info is not None else None
        if key not in maps:
            maps.add(key)
            picked.append(cell)
    picked.append(next(cell for cell in read.cells if cell.found))
    return picked


def _timed(fn, rounds: int) -> float:
    """Mean seconds per call over rounds calls."""
    began = time.perf_counter()
    for _ in range(rounds):
        fn()
    return (time.perf_counter() - began) / rounds


def _arrivals(scanner, cursor, cells) -> list[float]:
    """Seconds from the cursor landing on a new empty cell to the card queued.

    The scan loop runs in its own thread with its real sleeps and the
    Collections look every 2 s, the cursor moves at uneven times.
    """
    scanner._running = True
    scanner.state = ScannerState.IDLE
    thread = threading.Thread(target=scanner._scan_loop, daemon=True)
    thread.start()
    off = Point(-10, -10)
    times = []
    try:
        for i in range(ARRIVALS):
            cell = cells[i % len(cells)]
            cursor.at = off
            time.sleep(0.6 + 0.137 * i)
            _drain(scanner)
            cursor.at = _middle(cell.rect)
            landed = time.perf_counter()
            while True:
                if any(m[0] == "advice" for m in _drain(scanner)):
                    times.append(time.perf_counter() - landed)
                    break
                if time.perf_counter() - landed > 5:
                    times.append(float("nan"))
                    break
                time.sleep(0.001)
    finally:
        scanner._running = False
        thread.join(timeout=3)
    return times


def _on_screen(shot: Image.Image, cell, card) -> Image.Image:
    """The screenshot with the cell outlined and the card in the home box."""
    image = shot.convert("RGBA")
    ImageDraw.Draw(image).rectangle(
        (
            cell.rect.left - 3,
            cell.rect.top - 3,
            cell.rect.right + 2,
            cell.rect.bottom + 2,
        ),
        outline=(255, 210, 0, 255),
        width=3,
    )
    if card is not None:
        home = fixed_home(image.size)
        image.alpha_composite(render_in_box(card, home), (home.left, home.top))
    return image.convert("RGB")


def demo(shot_path: Path, out: Path, db: Database, monkeypatch_backend) -> None:
    shot = Image.open(shot_path).convert("RGB")
    frame = np.asarray(shot)
    began = time.perf_counter()
    read = read_collections(frame)
    read_s = time.perf_counter() - began
    print(f"== {shot_path} ({shot.size[0]}x{shot.size[1]})")
    if read is None:
        print("  not the Collections screen")
        return
    print(
        f"  read found={read.found_count}/{read.total} cells={len(read.cells)}"
        f" consistent={read.consistent} in {read_s * 1000:.0f} ms"
    )
    print("  empty cells and their blueprint:")
    catalog = {row.name: row for row in db.blueprint_catalog()}
    for cell in read.cells:
        if not cell.found:
            name = db.blueprint_at(cell.grid_index)
            info = catalog.get(name) if name else None
            where = f"map={info.map!r}" if info is not None else ""
            print(f"    slot {cell.grid_index:2d} {name or '(unnamed)'} {where}")

    cursor = _Cursor()
    monkeypatch_backend(cursor)
    scanner = Scanner(
        root=None, overlay=None, status=None, db=db, hotkey=_Hotkey(), advice_card=True
    )
    scanner._grab_full_screen = lambda: frame
    scanner._check_collections(0.0)

    folder = out / "demo" / shot_path.stem
    folder.mkdir(parents=True, exist_ok=True)
    print("  hover cards:")
    for cell in _pick(read, db):
        cursor.at = _middle(cell.rect)
        scanner._check_collections_hover()
        messages = _drain(scanner)
        cards = [m[2] for m in messages if m[0] == "advice"]
        card = cards[0] if cards else None
        state = "found" if cell.found else "empty"
        kinds = [m[0] for m in messages]
        print(f"  slot {cell.grid_index} ({state}) at {tuple(cursor.at)}: {kinds}")
        if card is not None:
            print(_card_text(card))
            render_in_box(card, fixed_home(shot.size)).save(
                folder / f"slot_{cell.grid_index:02d}.png"
            )
        _on_screen(shot, cell, card).save(
            folder / f"slot_{cell.grid_index:02d}_on_screen.png"
        )
        # Off the grid between cells, so the next one shows anew
        cursor.at = Point(-10, -10)
        scanner._check_collections_hover()
        _drain(scanner)

    empty = [cell for cell in read.cells if not cell.found]
    found = next(cell for cell in read.cells if cell.found)
    print("  timings:")
    for label, at in (
        ("touch (same empty cell)", _middle(empty[0].rect)),
        ("found cell", _middle(found.rect)),
        ("off the grid", Point(-10, -10)),
    ):
        cursor.at = _middle(empty[0].rect)
        scanner._check_collections_hover()
        cursor.at = at
        per = _timed(scanner._check_collections_hover, ROUNDS)
        _drain(scanner)
        print(f"    hover check, {label}: {per * 1e6:.1f} us a round")
    toggles = [_middle(empty[0].rect), _middle(empty[1].rect)]
    i = [0]

    def new_cell():
        i[0] += 1
        cursor.at = toggles[i[0] % 2]
        scanner._check_collections_hover()

    per = _timed(new_cell, 200)
    _drain(scanner)
    print(f"    hover check, new empty cell (card built): {per * 1e6:.1f} us")
    print(f"    screen read (every 2 s): {read_s * 1000:.0f} ms")

    scanner._collections_hover = None
    times = _arrivals(scanner, cursor, empty[:6])
    good = [t for t in times if not math.isnan(t)]
    print(
        f"    cursor on a cell to card queued, {len(times)} arrivals:"
        f" min {min(good) * 1000:.0f} ms, median {statistics.median(good) * 1000:.0f}"
        f" ms, max {max(good) * 1000:.0f} ms, missed {len(times) - len(good)}"
    )
    print(f"    each: {', '.join(f'{t * 1000:.0f}' for t in times)} ms")
    print(f"  pictures in {folder}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("shots", type=Path, nargs="+")
    args = parser.parse_args()
    main_module.get_ocr_engine = _NoTrigger
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(db_path=REPO / "items.db", user_db_path=Path(tmp) / "user.db")

        def backend(cursor):
            main_module.get_backend = lambda: cursor

        for shot in args.shots:
            demo(shot, args.out, db, backend)


if __name__ == "__main__":
    main()
