# ruff: noqa: INP001
"""Worth at a glance: demo images and the composite check for the worth colours.

    uv run --extra icons python scripts/worth_colour_demo.py --captures DIR
        --list STASH_SCAN.CSV [--extra PNG ...] --out DIR

The captures are the frames named in the list's first column (an
eval_stash_scan.py stash_scan.csv), read from --captures.

Composite check: each capture gets the stash pass (identify_slots, uncached,
with stacks, on the stash gallery), its badges are drawn over it with the
badge helper's PNG mode and the pass runs again on that composite. This is
done twice, once with the badges as they were before the worth colours
(patches in the bar's colour with cream text, pills in TAG_BG with white
text) and once with the worth colours. Each composite is compared with the
original, slot by slot.

Demo images: the new badges over four of the captures (the most slots worth
WORTH_HIGH or more, the most under WORTH_LOW, then the most identified) and
every --extra frame, full size and half size, in OUT/demo, with
OUT/demo/scale.png showing a patch at each of SCALE_WORTHS.

The report goes to stdout and to OUT/real-data.txt.
"""

import argparse
import csv
import sys
from operator import itemgetter
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from arc_helper.badge_surface import Patch
from arc_helper.badge_surface import render_png
from arc_helper.config import get_settings
from arc_helper.database import Database
from arc_helper.grid import find_slots
from arc_helper.main import load_stash_gallery
from arc_helper.main import surface_badges
from arc_helper.stash_scan import PRICE_PREFIX
from arc_helper.stash_scan import TAG_BG
from arc_helper.stash_scan import WORTH_HIGH
from arc_helper.stash_scan import WORTH_LOW
from arc_helper.stash_scan import badges_for
from arc_helper.stash_scan import format_price
from arc_helper.stash_scan import identify_slots
from arc_helper.stash_scan import is_wide
from arc_helper.stash_scan import unpatch
from arc_helper.stash_scan import verdicts_for
from arc_helper.stash_scan import worth_colours

DEMO_CAPTURES = 4
# Slots in the original and the composite are the same slot when their
# corners are this many px apart at most
MATCH_PX = 6
SCALE_WORTHS = (50, 250, 500, 1000, 2000, 3000, 4999, 5000, 20000)
# A common bar's colour from the captures, behind the scale's patches
SCALE_BAR = (30, 34, 40)
# The pills' text before the worth colours
OLD_TAG_FG = "#ffffff"


def worth_of(slot) -> int:
    """The stack's worth as badges_for counts it, 0 without a sell price."""
    if slot.item is None or not slot.item.sell_price:
        return 0
    return slot.item.sell_price * (slot.stack if slot.stack is not None else 1)


def old_style(badges: list[dict], slots) -> list[dict]:
    """The badges as drawn before the worth colours: bar-coloured patches."""
    bar_colour = {}
    for slot in slots:
        if slot.bar is not None:
            key = (
                slot.rect.left + slot.bar.bar_rect.left,
                slot.rect.top + slot.bar.bar_rect.top,
            )
            bar_colour[key] = list(slot.bar.colour)
    old = []
    for badge in badges:
        b = dict(badge)
        if b["kind"] == "patch":
            b["colour"] = bar_colour[b["rect"][0], b["rect"][1]]
            b.pop("fg")
        elif b["kind"] == "tag" and b["text"].startswith(PRICE_PREFIX):
            b["fg"], b["bg"] = OLD_TAG_FG, TAG_BG
        old.append(b)
    return old


def composite(rgb: np.ndarray, badges: list[dict], scratch: Path) -> np.ndarray:
    """The frame with the badges drawn over it by the badge helper."""
    frame = Image.fromarray(rgb)
    if not render_png(surface_badges(badges), frame.size, scratch):
        msg = "the badge helper could not draw the PNG"
        raise RuntimeError(msg)
    with Image.open(scratch) as layer:
        shown = Image.alpha_composite(frame.convert("RGBA"), layer.convert("RGBA"))
    scratch.unlink()
    return np.asarray(shown.convert("RGB"))


def compare(before, after) -> dict:
    """Counts of what the composite pass read differently from the original."""
    left = list(after)
    out = {
        "slots": (len(before), len(after)),
        "items": (
            sum(s.item is not None for s in before),
            sum(s.item is not None for s in after),
        ),
        "unmatched": 0,
        "item": 0,
        "colour": 0,
        "icon": 0,
        "stack": 0,
        "details": [],
    }
    for slot in before:
        match = next(
            (
                s
                for s in left
                if abs(s.rect.left - slot.rect.left) <= MATCH_PX
                and abs(s.rect.top - slot.rect.top) <= MATCH_PX
            ),
            None,
        )
        if match is None:
            out["unmatched"] += 1
            out["details"].append(f"  slot at {tuple(slot.rect)} not found again")
            continue
        left.remove(match)
        name = slot.item.name if slot.item else slot.guess
        if (slot.item and slot.item.name) != (match.item and match.item.name):
            out["item"] += 1
            out["details"].append(
                f"  {tuple(slot.rect)} item {name} -> "
                f"{match.item.name if match.item else None}"
            )
        old_bar, new_bar = slot.bar, match.bar
        if (old_bar and old_bar.colour) != (new_bar and new_bar.colour):
            out["colour"] += 1
            out["details"].append(
                f"  {tuple(slot.rect)} {name} bar colour "
                f"{old_bar and old_bar.colour} -> {new_bar and new_bar.colour}"
            )
        if (old_bar and old_bar.icon_box) != (new_bar and new_bar.icon_box):
            out["icon"] += 1
            out["details"].append(
                f"  {tuple(slot.rect)} {name} icon box "
                f"{old_bar and tuple(old_bar.icon_box or ())} -> "
                f"{new_bar and tuple(new_bar.icon_box or ())}"
            )
        if slot.stack != match.stack:
            out["stack"] += 1
            out["details"].append(
                f"  {tuple(slot.rect)} {name} stack {slot.stack} -> {match.stack}"
            )
    return out


def row(name: str, style: str, c: dict) -> str:
    return (
        f"{name[:44]:44} {style:4} {c['slots'][0]:>3}/{c['slots'][1]:<3} "
        f"{c['items'][0]:>3}/{c['items'][1]:<3} {c['unmatched']:>5} {c['item']:>5} "
        f"{c['colour']:>6} {c['icon']:>5} {c['stack']:>5}"
    )


HEADER = (
    f"{'capture':44} {'kind':4} {'slots':7} {'items':7} {'lost':>5} {'item':>5} "
    f"{'colour':>6} {'icon':>5} {'stack':>5}"
)


def save_demo(rgb: np.ndarray, name: str, demo: Path) -> None:
    full = Image.fromarray(rgb)
    full.save(demo / f"{name}.png")
    full.resize((full.width // 2, full.height // 2), Image.LANCZOS).save(
        demo / f"{name}.half.png"
    )


def slot_lines(slots) -> list[str]:
    lines = []
    for slot in sorted(slots, key=lambda s: (s.rect.top, s.rect.left)):
        if slot.item is None:
            continue
        worth = worth_of(slot)
        fill, text = worth_colours(worth) if worth else ("-", "-")
        lines.append(
            f"  {slot.panel:9} {tuple(slot.rect)} {slot.item.name[:30]:30} "
            f"stack={slot.stack} worth={worth} fill={fill} text={text}"
        )
    return lines


def scale_strip(out: Path) -> None:
    """A patch at each of SCALE_WORTHS on a common bar, labelled below."""
    tile, gap = 220, 12
    width = len(SCALE_WORTHS) * (tile + gap) + gap
    patches = []
    for n, worth in enumerate(SCALE_WORTHS):
        x = gap + n * (tile + gap)
        fill, text = worth_colours(worth)
        patches.append(
            Patch(
                (x, 20, x + tile, 64),
                icon_right=x + 30,
                patch_right=x + tile - 30,
                colour=tuple(int(fill[i : i + 2], 16) for i in (1, 3, 5)),
                text=PRICE_PREFIX + format_price(worth),
                fg=text,
            )
        )
    base = Image.new("RGB", (width, 110), (12, 12, 14))
    draw = ImageDraw.Draw(base)
    for n in range(len(SCALE_WORTHS)):
        x = gap + n * (tile + gap)
        draw.rectangle((x, 20, x + tile - 1, 63), fill=SCALE_BAR)
    layer = out.with_suffix(".layer.png")
    if not render_png(patches, base.size, layer):
        msg = "the badge helper could not draw the PNG"
        raise RuntimeError(msg)
    with Image.open(layer) as drawn:
        shown = Image.alpha_composite(base.convert("RGBA"), drawn.convert("RGBA"))
    layer.unlink()
    draw = ImageDraw.Draw(shown)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for n, worth in enumerate(SCALE_WORTHS):
        x = gap + n * (tile + gap)
        draw.text(
            (x, 76),
            f"{worth:,}  {worth_colours(worth)[0]}",
            fill=(220, 220, 220),
            font=font,
        )
    shown.convert("RGB").save(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--extra", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    demo = args.out / "demo"
    demo.mkdir(parents=True, exist_ok=True)
    scratch = args.out / "layer.tmp.png"

    with args.list.open(encoding="utf-8") as fh:
        names = list(dict.fromkeys(r["frame"] for r in csv.DictReader(fh)))
    db = Database()
    settings = get_settings()
    gallery, twins = load_stash_gallery(settings.icon_cache_dir)
    verdict_of = verdicts_for(db, show_ended=settings.show_ended_projects)

    def stash_pass(rgb: np.ndarray):
        return identify_slots(
            rgb, gallery, db.lookup, twins=twins, read_stacks=True, cache=None
        )

    report = [
        "Composite check: the stash pass on each capture, then again on the capture",
        "with its badges drawn over it. old = bar-coloured patches (before),",
        "new = worth colours. slots and items are original/composite. lost = slots",
        "not found again, item = identified item changed, colour = bar colour read",
        "changed, icon = type icon box moved, stack = stack read changed.",
        "",
        HEADER,
    ]
    details = []
    totals = {"old": {}, "new": {}}
    kept = []
    untouched = [0, 0]
    for name in names:
        rgb = np.asarray(Image.open(args.captures / name).convert("RGB"))
        for slot in find_slots(rgb):
            r = slot.rect
            if slot.filled and not is_wide(r):
                crop = rgb[r.top : r.bottom, r.left : r.right]
                untouched[0] += 1
                untouched[1] += unpatch(crop) is crop
        slots = stash_pass(rgb)
        badges = badges_for(slots, verdict_of)
        results = {}
        for style, drawn in (("old", old_style(badges, slots)), ("new", badges)):
            shown = composite(rgb, drawn, scratch)
            results[style] = compare(slots, stash_pass(shown))
            report.append(row(name, style, results[style]))
            details += (
                [f"{name} {style}:", *results[style]["details"]]
                if results[style]["details"]
                else []
            )
            for key, value in results[style].items():
                if key == "details":
                    continue
                if isinstance(value, tuple):
                    prev = totals[style].get(key, (0, 0))
                    totals[style][key] = (prev[0] + value[0], prev[1] + value[1])
                else:
                    totals[style][key] = totals[style].get(key, 0) + value
        worths = [worth_of(s) for s in slots if worth_of(s)]
        kept.append(
            (
                name,
                sum(w >= WORTH_HIGH for w in worths),
                sum(w < WORTH_LOW for w in worths),
                sum(s.item is not None for s in slots),
            )
        )
        print(report[-2], report[-1], sep="\n", flush=True)
    report += [
        "",
        (
            f"unpatch on the captures without badges: {untouched[1]} of "
            f"{untouched[0]} filled square slots left as they are"
        ),
    ]
    report += [
        row("TOTAL", style, {**totals[style], "details": []})
        for style in ("old", "new")
    ]
    worse = [
        key
        for key in ("unmatched", "item", "colour", "icon", "stack")
        if totals["new"][key] > totals["old"][key]
    ]
    report.append(f"new worse than old on: {', '.join(worse) or 'nothing'} (totals)")
    report += ["", "Differences, slot by slot:", *(details or ["  none"])]

    picks: list[str] = []
    for key in (itemgetter(1), itemgetter(2), itemgetter(3)):
        for entry in sorted(kept, key=key, reverse=True):
            if entry[0] not in picks:
                picks.append(entry[0])
                break
    for entry in sorted(kept, key=itemgetter(3), reverse=True):
        if len(picks) >= DEMO_CAPTURES:
            break
        if entry[0] not in picks:
            picks.append(entry[0])
    report += ["", "Demo images (in demo/, each with a .half.png copy):"]
    frames = [(args.captures / n, Path(n).stem) for n in picks]
    frames += [(p, p.stem) for p in args.extra]
    for path, stem in frames:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        slots = stash_pass(rgb)
        save_demo(composite(rgb, badges_for(slots, verdict_of), scratch), stem, demo)
        worths = [worth_of(s) for s in slots if worth_of(s)]
        report.append(
            f"{stem}.png: {sum(w >= WORTH_HIGH for w in worths)} slots worth "
            f"{WORTH_HIGH:,}+, {sum(w < WORTH_LOW for w in worths)} under {WORTH_LOW}"
        )
        report += slot_lines(slots)
    scale_strip(demo / "scale.png")
    report += ["", "Scale strip: demo/scale.png"]
    report += [
        f"  {w:>6,}: fill {worth_colours(w)[0]} text {worth_colours(w)[1]}"
        for w in SCALE_WORTHS
    ]
    text = "\n".join(report) + "\n"
    (args.out / "real-data.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
