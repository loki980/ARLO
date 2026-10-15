"""Render overlay text with Pillow when Tk has no scalable fonts.

Some Tk builds (python-build-standalone on Linux) only know the bitmap font
"fixed", so every requested font size is ignored. In that case the overlay
draws its text into an image with a real TrueType font and shows the image
in the label instead.
"""

from __future__ import annotations

import math
import subprocess
import tkinter as tk
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tkinter import font as tkfont

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageTk

_FALLBACKS: dict[bool, tuple[str, ...]] = {
    False: (
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ),
    True: (
        "/usr/share/fonts/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ),
}


@dataclass(frozen=True)
class TextStyle:
    """What a label needs to redraw its text as an image."""

    size: int  # points, as the Tk font would take
    bold: bool
    fg: str
    bg: str


def families_scalable(families: tuple[str, ...] | list[str]) -> bool:
    """True when Tk knows a family other than the bitmap font "fixed"."""
    return any(name.lower() != "fixed" for name in families)


def has_scalable_fonts(root: tk.Misc) -> bool:
    """Whether Tk on this root can scale fonts, computed once per root."""
    cached = getattr(root, "_arlo_scalable_fonts", None)
    if isinstance(cached, bool):
        return cached
    result = False
    try:
        families = tkfont.families(root)
        if families_scalable(families):
            probe = tkfont.Font(root, family="Segoe UI", size=20)
            result = probe.actual("family").lower() != "fixed"
    except tk.TclError:
        result = False
    root._arlo_scalable_fonts = result  # noqa: SLF001
    return result


@lru_cache(maxsize=2)
def find_font_file(bold: bool) -> str | None:  # noqa: FBT001
    """Path of a TrueType sans font, bold or regular, None when none is found."""
    pattern = "sans:bold" if bold else "sans"
    try:
        out = subprocess.run(
            ["fc-match", "-f", "%{file}", pattern],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        out = ""
    if out.lower().endswith(".ttf") and Path(out).is_file():
        return out
    for path in _FALLBACKS[bold]:
        if Path(path).is_file():
            return path
    return None


def tk_scaling(root: tk.Misc) -> float:
    """Pixels per point for this root."""
    try:
        return float(root.tk.call("tk", "scaling"))
    except (tk.TclError, ValueError):
        return 96 / 72


@lru_cache(maxsize=32)
def _load_font(path: str, pixels: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, pixels)


def wrap_text(text: str, font: ImageFont.FreeTypeFont, wraplength: int) -> list[str]:
    """Split text into lines, honouring newlines and wrapping at word breaks.

    wraplength is in pixels, 0 means no wrapping. A single word wider than
    wraplength stays on its own line.
    """
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if wraplength <= 0:
            lines.append(paragraph)
            continue
        current = ""
        for word in paragraph.split(" "):
            trial = f"{current} {word}" if current else word
            if current and font.getlength(trial) > wraplength:
                lines.append(current)
                current = word
            else:
                current = trial
        lines.append(current)
    return lines


def render_text(
    text: str,
    style: TextStyle,
    font_path: str,
    scaling: float,
    wraplength: int = 0,
) -> Image.Image:
    """Draw text as an RGB image, fg on bg, at round(points x scaling) pixels."""
    pixels = max(1, round(style.size * scaling))
    font = _load_font(font_path, pixels)
    lines = wrap_text(text, font, wraplength)
    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    width = max(1, *(math.ceil(font.getlength(line)) for line in lines))
    # A little room so glyph overhang is not clipped
    width += max(1, pixels // 10)
    image = Image.new("RGB", (width, line_h * len(lines)), style.bg)
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((0, i * line_h), line, font=font, fill=style.fg)
    return image


def render_photo(
    root: tk.Misc,
    text: str,
    style: TextStyle,
    font_path: str,
    wraplength: int = 0,
) -> ImageTk.PhotoImage:
    """render_text as a Tk image owned by root."""
    image = render_text(text, style, font_path, tk_scaling(root), wraplength)
    return ImageTk.PhotoImage(image, master=root)
