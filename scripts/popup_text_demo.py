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
from arc_helper.main import chip_payload
from arc_helper.overlay import BG_PANEL
from arc_helper.overlay import OverlayWindow
from arc_helper.overlay import TooltipChips
from arc_helper.screen import get_backend
from arc_helper.screen import phys_to_tk
from arc_helper.textrender import has_scalable_fonts
from arc_helper.tooltip_layout import FOUND_IN
from arc_helper.tooltip_layout import RECYCLES_INTO
from arc_helper.tooltip_layout import Rect
from arc_helper.tooltip_layout import TooltipLayout
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


def _grab(root: tk.Tk, window: tk.Misc, extra: int = 0) -> Image.Image | None:
    """Screenshot of a Tk window (grown by extra tk px), None when it fails."""
    for _ in range(20):
        root.update()
    root.after(200)
    root.update()
    k = get_backend().tk_scale
    box = (
        round((window.winfo_rootx() - extra) * k),
        round((window.winfo_rooty() - extra) * k),
        round((window.winfo_rootx() + window.winfo_width() + extra) * k),
        round((window.winfo_rooty() + window.winfo_height() + extra) * k),
    )
    print("grab box:", box)
    try:
        image = ImageGrab.grab(bbox=box)
    except Exception as exc:  # noqa: BLE001
        print("grab failed:", exc)
        return None
    if len(image.resize((16, 16)).getcolors() or []) < 3:
        print("grab came back blank")
        return None
    return image


def _chips_shot(root, item, verdict, outputs):
    """Three chips placed on a cream stand-in for the game's tooltip card."""
    card = Rect(1400, 300, 2300, 900)
    layout = TooltipLayout(
        card,
        {
            FOUND_IN: Rect(1440, 520, 1700, 560),
            RECYCLES_INTO: Rect(1440, 700, 1660, 740),
        },
    )
    window = tk.Toplevel(root)
    window.overrideredirect(boolean=True)
    window.configure(bg="#f9eedf")
    window.geometry(
        f"{phys_to_tk(card.width)}x{phys_to_tk(card.height)}"
        f"+{phys_to_tk(card.left)}+{phys_to_tk(card.top)}"
    )
    window.update()
    chips = TooltipChips(root)
    home_only = [RecycleOutput("Crude Explosives", 1, HOME)]
    payload = chip_payload(
        item, [*outputs, RecycleOutput("Field Part", 1, "field")], home_only, verdict
    )
    chips.show(item.name, layout, **payload)
    image = _grab(root, window)
    chips.hide()
    window.destroy()
    return image


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
    image = _grab(root, popup.window)
    if image is None:
        image = _compose(popup)
        print("composed from label images")
    popup.hide()
    chips_image = _chips_shot(root, item, verdict, outputs)
    if chips_image is not None:
        canvas = Image.new(
            "RGB",
            (max(image.width, chips_image.width), image.height + chips_image.height),
            "#000000",
        )
        canvas.paste(image, (0, 0))
        canvas.paste(chips_image, (0, image.height))
        image = canvas
    image.save(out)
    print("saved", out, image.size)
    root.destroy()


if __name__ == "__main__":
    main()
