# ruff: noqa: INP001
"""Show the Snitch Scanner advice card in the LOADOUT box for 3 s.

Usage: uv run python scripts/advice_window_demo.py OUTPUT_PNG

Saves a screenshot of the box area. If the live capture fails or comes back
blank (locked screen), the card is composed onto the newest 3840x2160 capture
in debug/captures at the computed place instead, and that is printed.
"""

import datetime
import sys
import tkinter as tk
from pathlib import Path

from PIL import Image

from arc_helper.advice_card import build_advice_card
from arc_helper.advice_card import render_advice_card
from arc_helper.advice_window import AdviceWindow
from arc_helper.advice_window import fit_scale
from arc_helper.advice_window import flatten_on_panel
from arc_helper.advice_window import place_card
from arc_helper.database import HOME
from arc_helper.database import Database
from arc_helper.main import POPUP_HOME_LEFT
from arc_helper.main import POPUP_HOME_TOP
from arc_helper.screen import get_backend
from arc_helper.tooltip_layout import Rect
from arc_helper.verdict import compute_verdict

BOTTOM = 0.892
CAPTURES = Path(__file__).resolve().parents[1] / "debug" / "captures"
FALLBACK_CAPTURE = Path("/home/jefflamb/workspace/ARLO/debug/captures")


def sample_card():
    db = Database()
    name = "Snitch Scanner"
    item = db.lookup(name)
    needs = db.needs_for(name)
    home = [o for o in db.recycle_outputs(name) if o.location == HOME]
    verdict = compute_verdict(
        item,
        needs,
        db.progress_map(),
        None,
        datetime.date.today(),  # noqa: DTZ011
        False,  # noqa: FBT003
        blueprint=db.blueprint_for(name),
        home_outputs=home,
    )
    return build_advice_card(verdict, needs, None)


def fallback(card, out):
    screen = (3840, 2160)
    box = Rect(
        round(screen[0] * POPUP_HOME_LEFT),
        round(screen[1] * POPUP_HOME_TOP),
        round(screen[0] * 0.533),
        round(screen[1] * BOTTOM),
    )
    pool = FALLBACK_CAPTURE if FALLBACK_CAPTURE.exists() else CAPTURES
    for path in sorted(pool.glob("*.png"), reverse=True):
        with Image.open(path) as img:
            if img.size == screen:
                full = img.convert("RGB")
                break
    else:
        sys.exit("no full-screen capture found")
    image = render_advice_card(card, fit_scale(box.right - box.left))
    x, y = place_card(box, image.size, screen[1])
    full.paste(flatten_on_panel(image), (x, y))
    full.crop((x - 60, y - 60, x + image.width + 60, y + image.height + 60)).save(out)
    print(f"composed onto {path} at {x},{y} (live capture failed)")


def main() -> None:
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    card = sample_card()
    try:
        backend = get_backend()
    except OSError as exc:
        print(f"screen backend unavailable: {exc}")
        fallback(card, out)
        return
    screen_w, screen_h = backend.resolution()
    box = Rect(
        round(screen_w * POPUP_HOME_LEFT),
        round(screen_h * POPUP_HOME_TOP),
        round(screen_w * 0.533),
        round(screen_h * BOTTOM),
    )
    root = tk.Tk()
    root.withdraw()
    window = AdviceWindow(root)
    window.show(card, box)
    state = {}

    def grab():
        rect = window.rect_phys()
        state["rect"] = rect
        try:
            shot = backend.grab(
                (rect.left - 60, rect.top - 60, rect.right + 60, rect.bottom + 60)
            )
            state["shot"] = shot.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            print(f"capture failed: {exc}")

    root.after(1500, grab)
    root.after(3000, root.quit)
    root.mainloop()
    window.hide()
    shot = state.get("shot")
    if shot is not None and shot.getextrema() != ((0, 0), (0, 0), (0, 0)):
        shot.save(out)
        print(f"live screenshot of {state['rect']} saved to {out}")
    else:
        fallback(card, out)
    root.destroy()


if __name__ == "__main__":
    main()
