# ruff: noqa: INP001, SLF001
"""Compare the stack count reader with the old Tesseract one on labelled slots.

Usage:
    uv run --extra icons python scripts/eval_stack_reads.py --labels <csv>
        --captures <folder> --upscaled-dir <folder> --out <folder>
    uv run python scripts/eval_stack_reads.py --labels <csv>
        --captures <folder> --upscaled-dir <folder> --templates

The labels CSV has one row per slot: file, left, top, right, bottom (frame
pixels), count (a number, or none for a slot without a count) and note. A
file is looked up in --captures first, then in --upscaled-dir, and slots of
files found in --upscaled-dir are reported apart as upscaled. Rows noted
occluded are skipped. Rows noted dimmed (a count drawn grey on a slot the
game fades out) are scored with the real captures.

The default run reads every slot with the old reader (one Tesseract call
per count, kept here as it was) and the new one, and writes real-data.txt
to --out: right, wrong and None counts for each, single digits apart, the
stack read time of one pass (the sum over a capture's labelled slots,
median over the real captures) and every wrong or missed count.

--templates builds the digit templates from the real captures' labelled
counts and prints the _DIGIT_HEX source for slot_marks.py, with how far
each training glyph sits from its own and the next nearest template.
"""

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pytesseract
from PIL import Image

from arc_helper import slot_marks
from arc_helper.slot_marks import read_stack

REPEATS = 3  # timing passes per capture, the fastest is kept

# The reader before this change, kept to measure against
OLD_WHITE_MIN = 180.0
OLD_OCR_HEIGHT = 48
OLD_OCR_PAD = 12
OLD_OCR_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789"


def _old_stack_digits(slot_rgb: np.ndarray) -> np.ndarray | None:
    bar = slot_marks._find_bar(slot_rgb)
    if bar is None:
        return None
    word = slot_marks._last_word(bar)
    if word is None or word[1] < slot_marks._RIGHT_HALF * bar.ink.shape[1]:
        return None
    ink = bar.ink[:, word[0] : word[1]]
    if float(np.median(bar.lum[:, word[0] : word[1]][ink])) < OLD_WHITE_MIN:
        return None
    glyphs = slot_marks._runs(ink.any(axis=0), 1)
    if len(glyphs) < 2:
        return None
    rows = np.flatnonzero(ink.any(axis=1))
    digit_rows = np.flatnonzero(ink[:, glyphs[1][0] :].any(axis=1))
    times_rows = np.flatnonzero(ink[:, glyphs[0][0] : glyphs[0][1]].any(axis=1))
    digit_height = digit_rows[-1] - digit_rows[0] + 1
    times_height = times_rows[-1] - times_rows[0] + 1
    if (
        times_height > slot_marks._TIMES_MAX_HEIGHT * digit_height
        or times_rows[0] <= rows[0]
    ):
        return None
    left = word[0] + glyphs[1][0]
    return bar.lum[digit_rows[0] : digit_rows[-1] + 1, left : word[1]]


def old_read_stack(slot_rgb: np.ndarray) -> int | None:
    digits = _old_stack_digits(slot_rgb)
    if digits is None:
        return None
    scale = OLD_OCR_HEIGHT / digits.shape[0]
    image = Image.fromarray(np.clip(255 - digits, 0, 255).astype(np.uint8))
    image = image.resize(
        (max(1, round(digits.shape[1] * scale)), OLD_OCR_HEIGHT),
        Image.Resampling.LANCZOS,
    )
    padded = Image.new(
        "L", (image.width + 2 * OLD_OCR_PAD, image.height + 2 * OLD_OCR_PAD), 255
    )
    padded.paste(image, (OLD_OCR_PAD, OLD_OCR_PAD))
    try:
        text = pytesseract.image_to_string(padded, config=OLD_OCR_CONFIG)
    except (pytesseract.TesseractError, OSError):
        return None
    text = text.strip()
    return int(text) if text.isdigit() else None


def _load(labels: Path, captures: Path, upscaled: Path) -> list[dict]:
    frames: dict[Path, np.ndarray] = {}
    slots = []
    with labels.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["note"] == "occluded":
                continue
            path = captures / row["file"]
            group = "real"
            if not path.exists():
                path = upscaled / row["file"]
                group = "upscaled"
            if path not in frames:
                with Image.open(path) as image:
                    frames[path] = np.asarray(image.convert("RGB"))
            left, top, right, bottom = (
                int(row[k]) for k in ("left", "top", "right", "bottom")
            )
            count = None if row["count"] == "none" else int(row["count"])
            slots.append(
                {
                    "file": row["file"],
                    "rect": (left, top, right, bottom),
                    "count": count,
                    "note": row["note"],
                    "group": group,
                    "crop": frames[path][top:bottom, left:right],
                }
            )
    return slots


def _templates(slots: list[dict]) -> None:
    cells: dict[int, list[np.ndarray]] = {d: [] for d in range(10)}
    for slot in slots:
        if slot["group"] != "real" or slot["count"] is None or slot["note"]:
            continue
        found = slot_marks._stack_digits(slot["crop"])
        glyphs = slot_marks._glyph_cells(*found) if found else None
        text = str(slot["count"])
        if not glyphs or len(glyphs) != len(text):
            print(f"skipped {slot['file']} {slot['rect']} {text}", file=sys.stderr)
            continue
        for digit, cell in zip(text, glyphs, strict=True):
            cells[int(digit)].append(cell)
    print("_DIGIT_HEX = (")
    means = []
    for digit in range(10):
        if not cells[digit]:
            sys.exit(f"no labelled glyph of {digit}")
        mean = np.mean(cells[digit], axis=0)
        means.append(np.round(mean * 15) / 15)
        text = "".join(f"{int(v):x}" for v in np.round(mean * 15).ravel())
        print(f"    # {digit}, {len(cells[digit])} glyphs")
        print(f'    (\n        "{text[:80]}"\n        "{text[80:]}"\n    ),')
    print(")")
    # The same distance as slot_marks._match_digit, 1 - correlation
    vectors = np.array([slot_marks._match_vector(m) for m in means])
    worst_best, worst_gap = 0.0, 1.0
    for digit, group in cells.items():
        for cell in group:
            distances = 1.0 - vectors @ slot_marks._match_vector(cell)
            other = np.delete(distances, digit).min()
            worst_best = max(worst_best, float(distances[digit]))
            worst_gap = min(worst_gap, float(other - distances[digit]))
    print(
        f"# largest distance to its own template {worst_best:.3f}, "
        f"smallest gap to the next {worst_gap:.3f}",
        file=sys.stderr,
    )


def _score(slots: list[dict], key: str) -> tuple[int, int, int]:
    right = sum(1 for s in slots if s[key] == s["count"])
    wrong = sum(1 for s in slots if s[key] is not None and s[key] != s["count"])
    missed = sum(1 for s in slots if s[key] is None)
    return right, wrong, missed


def _report(slots: list[dict], out: Path) -> None:
    lines = []

    def table(title: str, group: list[dict]) -> None:
        counted = [s for s in group if s["count"] is not None]
        lines.append(f"{title}: {len(counted)} labelled counts")
        for key in ("old", "new"):
            right, wrong, missed = _score(counted, key)
            lines.append(f"  {key}: right {right}, wrong {wrong}, None {missed}")
        empty = [s for s in group if s["count"] is None]
        for key in ("old", "new") if empty else ():
            false = sum(1 for s in empty if s[key] is not None)
            lines.append(
                f"  {key}: a count on {false} of {len(empty)} slots labelled none"
            )

    real = [s for s in slots if s["group"] == "real"]
    upscaled = [s for s in slots if s["group"] == "upscaled"]
    single = [s for s in real if s["count"] is not None and s["count"] < 10]
    table("Real 4K captures, all", real)
    table("Real 4K captures, single digits", single)
    plain = [s for s in real if s["note"] != "dimmed"]
    table("Real 4K captures without the dimmed slots", plain)
    for name in sorted({s["file"] for s in upscaled}):
        table(f"Upscaled {name}", [s for s in upscaled if s["file"] == name])

    lines.extend(("", "Stack read time of one pass (labelled slots of a capture), ms"))
    passes = {}
    for name in sorted({s["file"] for s in real}):
        group = [s for s in real if s["file"] == name]
        passes[name] = (
            min(sum(s["old_ms"][n] for s in group) for n in range(REPEATS)),
            min(sum(s["new_ms"][n] for s in group) for n in range(REPEATS)),
            len(group),
        )
        old_ms, new_ms, count = passes[name]
        lines.append(f"  {name}: {count} slots, old {old_ms:.1f}, new {new_ms:.2f}")
    old_median = statistics.median(p[0] for p in passes.values())
    new_median = statistics.median(p[1] for p in passes.values())
    lines.extend(
        (
            (
                f"  median over {len(passes)} captures: old {old_median:.1f}, "
                f"new {new_median:.2f} ({new_median / old_median:.4f} of old)"
            ),
            "",
            "Upscaled raid-inventory-4k.png, every labelled count",
        )
    )
    lines.extend(
        f"  {s['rect']} label {s['count']} old {s['old']} new {s['new']}  {s['note']}"
        for s in upscaled
        if s["file"] == "raid-inventory-4k.png" and s["count"] is not None
    )
    lines.extend(("", "Every wrong or missed count (label, old, new)"))
    lines.extend(
        f"  {s['group']} {s['file']} {s['rect']} label {s['count']} "
        f"old {s['old']} new {s['new']} {s['note']}"
        for s in slots
        if s["old"] != s["count"] or s["new"] != s["count"]
    )
    text = "\n".join(lines) + "\n"
    (out / "real-data.txt").write_text(text, encoding="utf-8")
    print(text, end="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--upscaled-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--templates", action="store_true")
    args = parser.parse_args()
    slots = _load(args.labels, args.captures, args.upscaled_dir)
    if args.templates:
        _templates(slots)
        return
    if args.out is None:
        parser.error("--out is needed without --templates")
    old_read_stack(slots[0]["crop"])  # start-up cost outside the timing
    for slot in slots:
        slot["old_ms"], slot["new_ms"] = [], []
        for _ in range(REPEATS):
            start = time.perf_counter()
            slot["old"] = old_read_stack(slot["crop"])
            middle = time.perf_counter()
            slot["new"] = read_stack(slot["crop"])
            end = time.perf_counter()
            slot["old_ms"].append((middle - start) * 1000)
            slot["new_ms"].append((end - middle) * 1000)
    args.out.mkdir(parents=True, exist_ok=True)
    _report(slots, args.out)


if __name__ == "__main__":
    main()
