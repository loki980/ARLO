# ruff: noqa: INP001
"""Draw the stash badges on a capture, the way the live surface would.

Usage: uv run --extra icons python scripts/badge_demo.py CAPTURE.png OUT.png

Runs the live stash pass on the capture (stash_identifier with stacks, then
stash_badge_maker), so the tiles get the same patches, have/need tags and
outlines as on screen. The badge helper's PNG mode draws them and they are
composited over the capture to OUT. OUT's name with .zoom.png in place of
.png gets a 3x crop of the first four identified square slots in reading
order, the original on the left and the badged tile on the right. Prints
what each of the four shows.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from arc_helper.badge_surface import render_png
from arc_helper.config import get_settings
from arc_helper.database import Database
from arc_helper.main import stash_badge_maker
from arc_helper.main import stash_identifier
from arc_helper.main import surface_badges
from arc_helper.stash_scan import is_wide

ZOOM = 3
TILES = 4
PAD = 8


def describe(tile: list[dict]) -> str:
    """The badges drawn on one tile, with their texts."""
    parts = []
    for b in tile:
        text = f"{b['kind']} {b.get('text', '')}".strip()
        if b["kind"] == "tag" and b.get("outline"):
            text += " (check)" if b.get("check") else " (outlined)"
        parts.append(text)
    return ", ".join(parts) or "no badge"


def tile_badges(badges: list[dict], rect: tuple) -> list[dict]:
    """The badges that lie within one slot's rect."""
    left, top, right, bottom = rect
    return [
        b
        for b in badges
        if left <= b["rect"][0] < right and top <= b["rect"][1] < bottom
    ]


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    capture_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    capture = Image.open(capture_path).convert("RGB")
    db = Database()
    settings = get_settings()
    identify = stash_identifier(db, settings.icon_cache_dir)
    make = stash_badge_maker(db, lambda: settings.show_ended_projects)
    slot_items = identify(np.asarray(capture))
    badges = make(slot_items)
    layer_path = out.with_suffix(".layer.png")
    if not render_png(surface_badges(badges), capture.size, layer_path):
        print("The badge helper could not draw the PNG, see the log above.")
        return 1
    with Image.open(layer_path) as layer:
        shown = Image.alpha_composite(capture.convert("RGBA"), layer.convert("RGBA"))
    layer_path.unlink()
    shown = shown.convert("RGB")
    shown.save(out)
    print(f"{len(slot_items)} slots, {len(badges)} badges drawn to {out}")

    squares = sorted(
        (s for s in slot_items if s.item is not None and not is_wide(s.rect)),
        key=lambda s: (s.rect.top, s.rect.left),
    )[:TILES]
    crops = []
    for slot in squares:
        box = tuple(slot.rect)
        pair = [
            img.crop(box).resize(
                ((box[2] - box[0]) * ZOOM, (box[3] - box[1]) * ZOOM), Image.NEAREST
            )
            for img in (capture, shown)
        ]
        crops.append(pair)
        print(
            f"{slot.item.name} stack={slot.stack}: {describe(tile_badges(badges, box))}"
        )
    if crops:
        width = max(p[0].width for p in crops)
        height = max(p[0].height for p in crops)
        sheet = Image.new(
            "RGB",
            (2 * width + 3 * PAD, len(crops) * (height + PAD) + PAD),
            (20, 20, 20),
        )
        for row, (before, after) in enumerate(crops):
            y = PAD + row * (height + PAD)
            sheet.paste(before, (PAD, y))
            sheet.paste(after, (2 * PAD + width, y))
        sheet.save(out.with_suffix(".zoom.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
