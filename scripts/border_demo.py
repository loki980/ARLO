# ruff: noqa: INP001
"""Visual check of the gold cell border on this display.

Usage: uv run python scripts/border_demo.py <out.png> [capture.png]

Loads a full-screen capture of a keep item (Stiletto IV in the loadout by
default), finds the card with the cursor at the image centre and the
hovered cell through the same find-and-convert path the scanner uses, with
the capture origin at (0, 0). It then shows the capture around the cell in
a borderless window at its real screen position, draws the real CellBorder
around the cell for 3 seconds, takes a screenshot with spectacle and saves
the region around the cell to out.png.

Needs a 3840x2160 display and spectacle (KDE). The Tk scale is the physical
width over Tk's screen width, so the screen backend (and its portal
session) is never started.
"""

import subprocess
import sys
import tempfile
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from PIL import ImageTk

from arc_helper import cell_border
from arc_helper.database import get_database
from arc_helper.main import BORDER_WIDTH_TK
from arc_helper.main import Scanner
from arc_helper.main import hovered_cell_on_screen
from arc_helper.tooltip_layout import Point
from arc_helper.tooltip_layout import Rect
from arc_helper.tooltip_layout import find_tooltip_card
from arc_helper.verdict import KEEP

CAPTURES = Path("/home/jefflamb/workspace/ARLO/debug/captures")
DEFAULT_CAPTURE = CAPTURES / "20261012-001706-128_STILETTOIV.png"
ITEM = {"STILETTOIV": "Stiletto IV"}
SHOW_MS = 3000
SHOT_MS = 1500
MARGIN = 200


def _verdict_kind(capture: Path) -> str:
    """Verdict for the item named in the capture file, from the real database."""
    db = get_database()
    tag = capture.stem.rsplit("_", 1)[-1]
    item = db.lookup(ITEM.get(tag, tag))
    if item is None:
        return "unknown"
    return Scanner._verdict_for(SimpleNamespace(db=db), item, False).kind  # noqa: FBT003, SLF001


def main() -> int:
    if len(sys.argv) not in {2, 3}:
        print("usage: border_demo.py <out.png> [capture.png]")
        return 2
    out = Path(sys.argv[1])
    capture_path = Path(sys.argv[2]) if len(sys.argv) == 3 else DEFAULT_CAPTURE

    kind = _verdict_kind(capture_path)
    print(f"capture {capture_path.name}, verdict {kind}")
    if kind != KEEP:
        print("not a keep item, the scanner would draw no border")
        return 1

    capture = Image.open(capture_path).convert("RGB")
    rgb = np.asarray(capture)
    card = find_tooltip_card(rgb, Point(rgb.shape[1] // 2, rgb.shape[0] // 2))
    rect = hovered_cell_on_screen(capture, Point(0, 0), card, [])
    print(f"card {card}, cell on screen {rect}")
    if rect is None:
        return 1

    root = tk.Tk()
    root.withdraw()
    tk_scale = capture.width / root.winfo_screenwidth()
    cell_border.phys_to_tk = lambda value: round(value / tk_scale)
    width_px = max(1, round(BORDER_WIDTH_TK * tk_scale))
    print(f"tk scale {tk_scale:.4f}, border width {width_px} px")

    # The game picture around the cell, at its own screen position
    region = Rect(
        max(0, rect.left - MARGIN),
        max(0, rect.top - MARGIN),
        min(capture.width, rect.right + MARGIN),
        min(capture.height, rect.bottom + MARGIN),
    )
    left, top = round(region.left / tk_scale), round(region.top / tk_scale)
    size = (
        round(region.right / tk_scale) - left,
        round(region.bottom / tk_scale) - top,
    )
    backdrop = tk.Toplevel(root)
    backdrop.overrideredirect(boolean=True)
    backdrop.attributes("-topmost", True)  # noqa: FBT003
    photo = ImageTk.PhotoImage(capture.crop(region).resize(size))
    tk.Label(backdrop, image=photo, borderwidth=0).pack()
    backdrop.geometry(f"{size[0]}x{size[1]}+{left}+{top}")

    border = cell_border.CellBorder(root)
    border.show(rect, width_px)
    border.show(rect, width_px)  # a repeat must not re-map

    def shoot() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shot = Path(tmp) / "screen.png"
            subprocess.run(
                ["/usr/bin/spectacle", "-b", "-n", "-f", "-o", str(shot)],
                check=True,
                timeout=15,
            )
            with Image.open(shot) as screen:
                screen.crop(region).save(out)
        print(f"wrote {out}")

    root.after(SHOT_MS, shoot)
    root.after(SHOW_MS, root.quit)
    root.mainloop()
    border.hide(reason="demo_done")
    root.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
