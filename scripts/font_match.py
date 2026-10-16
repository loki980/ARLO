# ruff: noqa: INP001, RUF001, SLF001
"""Match the patch text to the game's own stack count, and show it.

    uv run --extra icons python scripts/font_match.py --captures <folder> \\
        --names <stash_scan.csv> --out <folder>

Reads the frame, slot rect and stack of each row of the names CSV (the
stash scan's output), finds each square slot's bar and right mark with
slot_marks.bar_geometry and measures the game's digits there: cap height,
stroke width, ink area per digit and the baseline's distance from the bar
bottom, all in px. It renders the same digits in the bundled Barlow Semi
Condensed weights for comparison, then draws the patch on the real tiles in
the old style (Bold at 0.55 of the bar height, centred) and the new style
(through badge_surface.render_png), measures both the same way, and writes
font_match.png (before and after at 3x zoom) and real-data.txt to --out.

The old style is drawn by this script under the system Python (the one with
gi and cairo), which is why only the standard library is imported at the top.
"""

import argparse
import csv
import importlib.util
import string
import subprocess
import sys
from pathlib import Path

ZOOM = 3
TILES = 8
MIN_COUNTS = 10
OLD_SHARE = 0.55
OLD_MIN_SHARE = 0.4
OLD_WEIGHT = "Bold"
# A rendered digit's ink is measured at half of its peak above the bar
HALF = 0.5
# Rows above the bar kept in the sheet's crops
CONTEXT_ROWS = 24
CAP_SHARE = 0.7  # Barlow's digit height as a share of its font size
PHI = "Φ"


def _load_helper():
    helper_path = (
        Path(__file__).resolve().parents[1] / "src" / "arc_helper" / "badge_helper.py"
    )
    spec = importlib.util.spec_from_file_location("badge_helper_old", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_old(out: Path, width: int, height: int) -> int:
    """Draw badges from stdin in the style before the font match, to a PNG.

    Run under the system Python. The patch text is Bold at OLD_SHARE of the
    bar height, centred on its ink, as badge_helper drew it before.
    """
    helper = _load_helper()
    helper.PATCH_TEXT_SHARE = OLD_SHARE
    helper.PATCH_TEXT_MIN_SHARE = OLD_MIN_SHARE
    helper.PATCH_FONT_WEIGHT = OLD_WEIGHT

    def draw_patch(cr, badge, family):
        fitted = helper.patch_text(cr, badge, family)
        if fitted is None:
            return
        text, edge = fitted
        left, top, _right, bottom = badge["rect"]
        helper._bottom_left_rounded_rect(
            cr, left, top, edge - left, bottom - top, float(badge.get("radius", 6))
        )
        cr.set_source_rgba(*helper.parse_rgb(badge["colour"]))
        cr.fill()
        cr.set_source_rgba(1, 1, 1, 1)
        text.draw(left + helper.PATCH_PAD, top + (bottom - top - text.height) / 2)

    helper._draw_patch = draw_patch
    _op, badges = helper.parse_message(sys.stdin.readline())
    helper.render_png(badges, width, height, out)
    return 0


def _runs(flags) -> list[tuple[int, int]]:
    spans, start = [], None
    for i, on in enumerate([*flags, False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            spans.append((start, i))
            start = None
    return spans


def measure(lum, background: float, digits: int) -> dict | None:
    """Cap height, stroke, ink area per digit and baseline row of a count.

    lum is the luminance of a mark box (a glyph then `digits` digits). None
    when the glyph columns do not match.
    """
    import numpy as np

    peak = float(lum.max())
    if peak - background < 60:
        return None
    mask = lum > background + HALF * (peak - background)
    coverage = np.clip((lum - background) / (peak - background), 0, 1)
    glyphs = _runs(mask.any(axis=0))
    if len(glyphs) != digits + 1:
        return None
    caps, strokes, bottoms, areas = [], [], [], []
    for a, b in glyphs[1:]:
        sub = mask[:, a:b]
        rows = np.flatnonzero(sub.any(axis=1))
        caps.append(rows[-1] - rows[0] + 1)
        bottoms.append(rows[-1] + 1)
        areas.append(float(coverage[:, a:b].sum()))
        lo = rows[0] + int(0.25 * len(rows))
        hi = rows[0] + int(0.75 * len(rows))
        for r in range(lo, hi + 1):
            for s, e in _runs(sub[r]):
                if e - s < 0.6 * (b - a):
                    strokes.append(e - s)
    return {
        "cap": float(np.median(caps)),
        "stroke": float(np.median(strokes)) if strokes else 0.0,
        "bottom": float(np.median(bottoms)),
        "areas": areas,
    }


def _tiles(captures: Path, names: Path):
    """(frame name, crop, slot rect, row, bar geometry) of each readable tile."""
    import numpy as np
    from PIL import Image

    from arc_helper.slot_marks import bar_geometry

    rows = [
        r
        for r in csv.DictReader(names.open(encoding="utf-8"))
        if r["frame"] != "frame" and r["stack"]
    ]
    current, frame = None, None
    for r in rows:
        if r["frame"] != current:
            current = r["frame"]
            frame = np.asarray(Image.open(captures / current).convert("RGB"))
        left, top, right, bottom = (
            int(r[k]) for k in ("left", "top", "right", "bottom")
        )
        if abs((right - left) - (bottom - top)) > 6:
            continue
        crop = frame[top:bottom, left:right]
        bar = bar_geometry(crop)
        if bar is None or bar.right_mark_box is None or bar.icon_box is None:
            continue
        yield current, crop, (left, top, right, bottom), r, bar


def _lum(crop, box):
    return crop[box.top : box.bottom, box.left : box.right].astype(float).mean(axis=2)


def measure_game(captures: Path, names: Path) -> list[dict]:
    out = []
    for frame, crop, _rect, row, bar in _tiles(captures, names):
        got = measure(
            _lum(crop, bar.right_mark_box), sum(bar.colour) / 3, len(row["stack"])
        )
        if got is None:
            continue
        mark = bar.right_mark_box
        got.update(
            frame=frame,
            stack=row["stack"],
            bar_h=bar.bar_rect.bottom - bar.bar_rect.top,
            gap=bar.bar_rect.bottom - (mark.top + got["bottom"]),
            chars=row["stack"],
        )
        out.append(got)
    return out


def _span(label: str, value: float, span: tuple[float, float]) -> str:
    return f"  {label} {value:.1f} px (min {span[0]:.0f}, max {span[1]:.0f})"


def candidate_rows(helper_dir: Path, cap: float) -> list[str]:
    """Each bundled weight's digits at the game's cap height, same measure."""
    import numpy as np
    from PIL import Image
    from PIL import ImageDraw
    from PIL import ImageFont

    lines = []
    for weight in ("Bold", "SemiBold"):
        path = helper_dir / f"BarlowSemiCondensed-{weight}.ttf"
        size = cap / CAP_SHARE
        font = ImageFont.truetype(str(path), size)
        caps, strokes, areas = [], [], {}
        for ch in string.digits:
            img = Image.new("L", (100, 70), 0)
            ImageDraw.Draw(img).text((6, 6), "×" + ch, font=font, fill=255)
            got = measure(np.asarray(img).astype(float), 0, 1)
            if got is None:
                continue
            caps.append(got["cap"])
            strokes.append(got["stroke"])
            areas[ch] = got["areas"][0]
        lines.append(
            f"  {weight:9s} font size {size:.1f} px  cap {np.median(caps):.1f}  "
            f"stroke {np.median(strokes):.1f}  ink area per digit: "
            + " ".join(f"{c}:{a:.0f}" for c, a in sorted(areas.items()))
        )
    return lines


def _patch(bar, left_top, text):
    from arc_helper.badge_surface import Patch

    left, top = left_top
    rect = bar.bar_rect
    return Patch(
        (left + rect.left, top + rect.top, left + rect.right, top + rect.bottom),
        icon_right=left + bar.icon_box.right,
        patch_right=left + bar.patch_right,
        colour=bar.colour,
        text=text,
    )


def _render_old(badges, size, out: Path, python: str) -> bool:
    from arc_helper.badge_surface import encode_message

    cmd = [python, str(Path(__file__).resolve()), "--old-png", str(out)]
    cmd += ["--size", str(size[0]), str(size[1])]
    result = subprocess.run(
        cmd,
        input=encode_message("draw", badges),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0


def _composite(crop, png: Path):
    import numpy as np
    from PIL import Image

    base = Image.fromarray(crop).convert("RGBA")
    base.alpha_composite(Image.open(png).convert("RGBA"))
    return np.asarray(base.convert("RGB"))


def _price_text(row: dict) -> str:
    from arc_helper.stash_scan import format_price

    return f"{PHI} {format_price(int(row['sell_price'] or 1) * int(row['stack']))}"


def build(captures: Path, names: Path, out: Path) -> int:
    import numpy as np
    from PIL import Image
    from PIL import ImageDraw

    from arc_helper import badge_helper
    from arc_helper.badge_surface import find_system_python
    from arc_helper.badge_surface import render_png

    python = find_system_python()
    if python is None:
        print("no system Python with gi and cairo", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    game = measure_game(captures, names)
    if len(game) < MIN_COUNTS:
        print(f"only {len(game)} counts measured", file=sys.stderr)
        return 1

    def med(key):
        return float(np.median([g[key] for g in game]))

    cap, stroke, gap, bar_h = med("cap"), med("stroke"), med("gap"), med("bar_h")
    digit_areas: dict[str, list[float]] = {}
    for g in game:
        for ch, area in zip(g["chars"], g["areas"], strict=True):
            digit_areas.setdefault(ch, []).append(area)
    n_frames = len({g["frame"] for g in game})
    cap_range = (min(g["cap"] for g in game), max(g["cap"] for g in game))
    stroke_range = (min(g["stroke"] for g in game), max(g["stroke"] for g in game))
    gap_range = (min(g["gap"] for g in game), max(g["gap"] for g in game))
    report = [
        f"Game stack counts measured: {len(game)} in {n_frames} real 4K captures",
        f"  bar height {bar_h:.0f} px",
        _span("digit cap height", cap, cap_range),
        _span("stroke", stroke, stroke_range),
        _span("digit bottom to bar bottom", gap, gap_range),
        "  ink area per digit: "
        + " ".join(f"{c}:{np.median(v):.0f}" for c, v in sorted(digit_areas.items())),
        "",
        "Bundled weights at the game's cap height:",
        *candidate_rows(badge_helper.FONT_DIR, cap),
        "",
    ]

    # Tiles for the sheet and the before and after measures
    chosen, seen = [], set()
    for frame, crop, rect, row, bar in _tiles(captures, names):
        key = (frame, row["stack"])
        if key in seen or not _price_text(row).split()[1].isdigit():
            continue
        seen.add(key)
        chosen.append((frame, crop, rect, row, bar))
    step = max(1, len(chosen) // TILES)
    chosen = chosen[::step][:TILES]
    sheet_rows, old_m, new_m = [], [], []
    for _frame, crop, _rect, row, bar in chosen:
        text = _price_text(row)
        badge = _patch(bar, (0, 0), text)
        size = (crop.shape[1], crop.shape[0])
        old_png, new_png = out / "_old.png", out / "_new.png"
        if not _render_old([badge], size, old_png, python):
            return 1
        if not render_png([badge], size, new_png, python=python):
            return 1
        pair = []
        for png, store in ((old_png, old_m), (new_png, new_m)):
            tile = _composite(crop, png)
            box = bar.bar_rect
            hi = box.bottom
            region = (
                tile[box.top : hi, box.left : bar.patch_right]
                .astype(float)
                .mean(axis=2)
            )
            got = measure(region, sum(bar.colour) / 3, len(text.split()[1]))
            if got:
                store.append(
                    {
                        "cap": got["cap"],
                        "stroke": got["stroke"],
                        "gap": hi - box.top - got["bottom"],
                    }
                )
            pair.append(tile[max(0, box.top - CONTEXT_ROWS) : hi])
        sheet_rows.append((text, row["stack"], pair))
    for png in (out / "_old.png", out / "_new.png"):
        png.unlink(missing_ok=True)

    for label, store in (("before (Bold, 0.55)", old_m), ("after", new_m)):
        if store:
            report.append(
                f"Patch digits {label}, {len(store)} tiles: cap "
                f"{np.median([m['cap'] for m in store]):.1f} px, stroke "
                f"{np.median([m['stroke'] for m in store]):.1f} px, digit bottom to bar "
                f"bottom {np.median([m['gap'] for m in store]):.1f} px"
            )
    report += [
        "",
        (
            f"Chosen: {badge_helper.PATCH_FONT_FAMILY} {badge_helper.PATCH_FONT_WEIGHT} "
            f"(bundled), size {badge_helper.PATCH_TEXT_SHARE} of the bar height "
            f"({badge_helper.PATCH_TEXT_SHARE * bar_h:.1f} px on the {bar_h:.0f} px bar), "
            f"baseline {badge_helper.PATCH_BASELINE_SHARE} of the bar height above the "
            f"bar bottom ({badge_helper.PATCH_BASELINE_SHARE * bar_h:.1f} px)."
        ),
    ]
    (out / "real-data.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    # The comparison sheet: before left, after right, 3x
    width = sheet_rows[0][2][0].shape[1] * ZOOM
    label_h = 22
    rows_img = []
    for text, stack, pair in sheet_rows:
        height = pair[0].shape[0] * ZOOM
        row_img = Image.new("RGB", (2 * width + 10, height + label_h), (30, 30, 30))
        draw = ImageDraw.Draw(row_img)
        draw.text(
            (4, 4), f"before: patch '{text[2:]}', game x{stack}", fill=(255, 255, 255)
        )
        draw.text(
            (width + 14, 4),
            f"after: patch '{text[2:]}', game x{stack}",
            fill=(255, 255, 255),
        )
        for n, tile in enumerate(pair):
            zoomed = Image.fromarray(tile).resize(
                (width, height), Image.Resampling.LANCZOS
            )
            row_img.paste(zoomed, (n * (width + 10), label_h))
        rows_img.append(row_img)
    sheet = Image.new(
        "RGB", (rows_img[0].width, sum(r.height + 6 for r in rows_img)), (30, 30, 30)
    )
    y = 0
    for r in rows_img:
        sheet.paste(r, (0, y))
        y += r.height + 6
    sheet.save(out / "font_match.png")
    print("\n".join(report))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--captures", type=Path)
    parser.add_argument("--names", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--old-png", type=Path, help="internal: the old style, system Python"
    )
    parser.add_argument("--size", type=int, nargs=2)
    args = parser.parse_args()
    if args.old_png:
        return render_old(args.old_png, *args.size)
    if not (args.captures and args.names and args.out):
        parser.error("--captures, --names and --out are required")
    return build(args.captures, args.names, args.out)


if __name__ == "__main__":
    sys.exit(main())
