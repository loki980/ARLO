# ruff: noqa: INP001
"""Show the real popup for a sample KEEP verdict and save a screenshot.

Usage: uv run python scripts/popup_text_demo.py popup_text_demo.png

Grabs the popup region from the screen. When the grab fails or comes back
blank, composes the popup's rendered label images into the PNG instead.
"""

import sys
import tkinter as tk

from PIL import Image
from PIL import ImageGrab

from arc_helper.database import HOME
from arc_helper.database import Item
from arc_helper.database import RecycleOutput
from arc_helper.overlay import BG_PANEL
from arc_helper.overlay import OverlayWindow
from arc_helper.screen import get_backend
from arc_helper.textrender import has_scalable_fonts
from arc_helper.verdict import Verdict


def _compose(popup: OverlayWindow) -> Image.Image:
    """Stack the label images top to bottom on the panel colour."""
    labels = [
        popup.name_label,
        popup.action_label,
        popup.reason_label,
        popup.info_label,
        popup.notes_label,
    ]
    photos = [
        getattr(label, "_arlo_photo", None)
        for label in labels
        if getattr(label, "_arlo_photo", None)
    ]
    images = [ImageTkToPil(p) for p in photos]
    width = max(i.width for i in images) + 40
    height = sum(i.height + 6 for i in images) + 30
    canvas = Image.new("RGB", (width, height), BG_PANEL)
    y = 15
    for image in images:
        canvas.paste(image, (20, y))
        y += image.height + 6
    return canvas


def ImageTkToPil(photo) -> Image.Image:  # noqa: N802
    """Read a PhotoImage back into a PIL image through its PPM data."""
    data = photo.tk.call(photo, "data", "-format", "ppm")
    from io import BytesIO

    return Image.open(BytesIO(data)).convert("RGB")


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "popup_text_demo.png"
    root = tk.Tk()
    root.withdraw()
    print("scalable fonts:", has_scalable_fonts(root))
    popup = OverlayWindow(root)
    popup.overlay_x, popup.overlay_y = 200, 200
    item = Item(
        name="Fireball Burner",
        action="Keep",
        sell_price=640,
        stack_size=3,
        category="weapon",
    )
    verdict = Verdict("keep", "8 for Refiner II, 1 for Test Case, then recycle")
    outputs = [
        RecycleOutput("ARC Alloy", 1, HOME),
        RecycleOutput("Crude Explosives", 1, HOME),
    ]
    popup.show(item.name, item, verdict, outputs)
    for _ in range(20):
        root.update()
    root.after(200)
    root.update()
    window = popup.window
    k = get_backend().tk_scale
    box = (
        round(window.winfo_rootx() * k),
        round(window.winfo_rooty() * k),
        round((window.winfo_rootx() + window.winfo_width()) * k),
        round((window.winfo_rooty() + window.winfo_height()) * k),
    )
    print("popup box:", box)
    image = None
    try:
        image = ImageGrab.grab(bbox=box)
        if len(image.resize((16, 16)).getcolors() or []) < 3:
            print("grab came back blank")
            image = None
    except Exception as exc:  # noqa: BLE001
        print("grab failed:", exc)
    if image is None:
        image = _compose(popup)
        print("composed from label images")
    image.save(out)
    print("saved", out, image.size)
    popup.hide()
    root.destroy()


if __name__ == "__main__":
    main()
