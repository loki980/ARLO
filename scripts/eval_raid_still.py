# ruff: noqa: INP001
"""Check the stash stillness compare on a real frame, and the stash pass cost.

    uv run python scripts/eval_raid_still.py <frame.png> <report.txt>
    uv run --extra icons python scripts/eval_raid_still.py <frame.png> \\
        <report.txt> --contention [--ort-threads N]

Builds pairs of frames from one 3840x2160 capture of the inventory and
prints whether the old stash_still (the whole frame) and the new one (only
inside the slots find_slots finds) call each pair still:

  A, the world moves: everything outside the found slots shifted right by
    12 px and brightened by 20, the slots kept.
  B, items move: filled backpack slot interiors swapped pairwise, as many
    swaps as it takes for the new compare to reach STILL_MEAN.
  C, identical: a copy of the frame.

With --contention it also times, median of 10 each, the trigger band read
(OCREngine.band_has_trigger on the top centre band) and the full-screen
item card search (ocr.crop_card at a filled slot centre), first alone and
then while a stash pass with stacks (main.stash_identifier) runs over and
over in another thread, and the stash pass itself alone and under the
measurement. --ort-threads N sets intra_op_num_threads on the stash pass's
onnxruntime session (the DINOv2 embedder runs on onnxruntime, not torch).

The report goes to the given path and to stdout.
"""

import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from arc_helper.grid import BACKPACK
from arc_helper.grid import STILL_MEAN
from arc_helper.grid import STILL_SCALE
from arc_helper.grid import find_slots
from arc_helper.grid import frames_still
from arc_helper.main import free_rects
from arc_helper.main import stash_still
from arc_helper.tooltip_layout import Rect

SHIFT_PX = 12
BRIGHTEN = 20
INSET = 8
RUNS = 10


def old_stash_still(a: np.ndarray, b: np.ndarray, ignore: list[Rect]) -> bool:
    """stash_still as it was: the whole frame outside the ignore rects."""
    if not ignore:
        return frames_still(a, b)
    if a.shape != b.shape:
        return False
    rects = free_rects(a.shape, ignore)
    return frames_still(a, b, rects) if rects else True


def slot_mask(shape: tuple[int, ...], rects: list[Rect]) -> np.ndarray:
    mask = np.zeros(shape[:2], bool)
    for r in rects:
        mask[r.top : r.bottom, r.left : r.right] = True
    return mask


def world_moves(rgb: np.ndarray, rects: list[Rect]) -> np.ndarray:
    moved = np.roll(rgb, SHIFT_PX, axis=1).astype(np.int16) + BRIGHTEN
    moved = np.clip(moved, 0, 255).astype(np.uint8)
    keep = slot_mask(rgb.shape, rects)
    moved[keep] = rgb[keep]
    return moved


def interior(r: Rect) -> tuple[slice, slice]:
    return slice(r.top + INSET, r.bottom - INSET), slice(
        r.left + INSET, r.right - INSET
    )


def items_move(rgb: np.ndarray, slots: list) -> tuple[np.ndarray, int, float]:
    """Swap filled backpack interiors pairwise until the new compare moves."""
    filled = [s.rect for s in slots if s.panel == BACKPACK and s.filled]
    rects = [s.rect for s in slots]
    pairs = [
        (a, b)
        for a, b in zip(filled[::2], filled[1::2], strict=False)
        if (a.width, a.height) == (b.width, b.height)
    ]
    moved = rgb.copy()
    swaps = 0
    for a, b in pairs:
        ia, ib = interior(a), interior(b)
        moved[ia], moved[ib] = rgb[ib], rgb[ia]
        swaps += 1
        if not stash_still(rgb, moved, []):
            break
    return moved, swaps, slot_mean(rgb, moved, rects)


def slot_mean(a: np.ndarray, b: np.ndarray, rects: list[Rect]) -> float:
    """The mean frames_still compares, over the cells wholly inside the slots."""
    import cv2

    def small(rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        size = (w // STILL_SCALE, h // STILL_SCALE)
        return cv2.cvtColor(
            cv2.resize(rgb, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY
        )

    diff = cv2.absdiff(small(a), small(b))
    area = np.zeros(diff.shape, bool)
    s = STILL_SCALE
    for r in rects:
        area[-(-r.top // s) : r.bottom // s, -(-r.left // s) : r.right // s] = True
    return float(diff[area].mean())


def stillness(rgb: np.ndarray, out: list[str]) -> None:
    start = time.perf_counter()
    slots = find_slots(rgb)
    took = (time.perf_counter() - start) * 1000
    rects = [s.rect for s in slots]
    filled = sum(s.filled for s in slots)
    out.extend(
        (
            f"find_slots: {len(slots)} slots ({filled} filled) in {took:.0f} ms",
            f"STILL_MEAN {STILL_MEAN}",
        )
    )
    world = world_moves(rgb, rects)
    out.append(
        f"pair A (world shifted {SHIFT_PX} px, +{BRIGHTEN} outside the slots): "
        f"old still {old_stash_still(rgb, world, [])} "
        f"new still {stash_still(rgb, world, [])} "
        f"slot mean {slot_mean(rgb, world, rects):.2f}"
    )
    items, swaps, mean = items_move(rgb, slots)
    out.append(
        f"pair B (backpack interiors swapped, {swaps} swaps): "
        f"old still {old_stash_still(rgb, items, [])} "
        f"new still {stash_still(rgb, items, [])} slot mean {mean:.2f}"
    )
    same = rgb.copy()
    out.append(
        f"pair C (identical): old still {old_stash_still(rgb, same, [])} "
        f"new still {stash_still(rgb, same, [])}"
    )


def median_ms(fn, runs: int = RUNS) -> float:
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def contention(rgb: np.ndarray, threads: int | None, out: list[str]) -> None:
    import onnxruntime as ort

    from arc_helper.config import get_settings
    from arc_helper.database import Database
    from arc_helper.main import stash_identifier
    from arc_helper.ocr import OCREngine
    from arc_helper.ocr import crop_card
    from arc_helper.tooltip_layout import Point

    if threads is not None:
        made = ort.SessionOptions

        def capped() -> ort.SessionOptions:
            options = made()
            options.intra_op_num_threads = threads
            return options

        ort.SessionOptions = capped
    out.append(f"onnxruntime intra-op threads {threads or 'default'}")
    height, width = rgb.shape[:2]
    image = Image.fromarray(rgb)
    band = image.crop(
        (
            int(width * OCREngine.TRIGGER_BAND_LEFT),
            0,
            int(width * OCREngine.TRIGGER_BAND_RIGHT),
            int(height * OCREngine.TRIGGER_BAND_BOTTOM),
        )
    )
    filled = [s.rect for s in find_slots(rgb) if s.filled]
    cursor = Point(width // 2, height // 2)
    if filled:
        r = filled[0]
        cursor = Point((r.left + r.right) // 2, (r.top + r.bottom) // 2)

    def read_band() -> None:
        OCREngine.band_has_trigger(band)

    def search_card() -> None:
        crop_card(image, cursor)

    identify = stash_identifier(Database(), get_settings().icon_cache_dir)
    identify(rgb)  # load the gallery and warm the model

    pass_alone = median_ms(lambda: identify(rgb), 3)
    band_alone = median_ms(read_band)
    card_alone = median_ms(search_card)

    stop = threading.Event()
    pass_times: list[float] = []

    def passes() -> None:
        while not stop.is_set():
            start = time.perf_counter()
            identify(rgb)
            pass_times.append((time.perf_counter() - start) * 1000)

    worker = threading.Thread(target=passes, daemon=True)
    worker.start()
    time.sleep(0.5)
    band_busy = median_ms(read_band)
    card_busy = median_ms(search_card)
    stop.set()
    worker.join()
    pass_busy = statistics.median(pass_times) if pass_times else float("nan")

    band_line = (
        f"band read: alone {band_alone:.1f} ms, beside a pass {band_busy:.1f} ms"
    )
    card = f"card search: alone {card_alone:.1f} ms, beside a pass {card_busy:.1f} ms"
    runs = len(pass_times)
    stash = f"stash pass: alone {pass_alone:.0f} ms (median of 3), "
    stash += f"under the measurement {pass_busy:.0f} ms (median of {runs})"
    slow = f"slowdown: band x{band_busy / band_alone:.2f}, "
    slow += f"card x{card_busy / card_alone:.2f}"
    out.extend((band_line, card, stash, slow))


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        return 2
    frame, report = Path(args[0]), Path(args[1])
    threads = None
    if "--ort-threads" in args:
        threads = int(args[args.index("--ort-threads") + 1])
    with Image.open(frame) as image:
        rgb = np.asarray(image.convert("RGB"))
    out = [f"frame {frame.name} {rgb.shape[1]}x{rgb.shape[0]}"]
    stillness(rgb, out)
    if "--contention" in args:
        contention(rgb, threads, out)
    text = "\n".join(out) + "\n"
    report.parent.mkdir(parents=True, exist_ok=True)
    if "--contention" in args and report.exists():
        text = report.read_text(encoding="utf-8") + "\n" + text
    report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
