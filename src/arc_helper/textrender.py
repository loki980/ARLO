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


FONT_DIR = Path(__file__).parent / "assets" / "fonts"
BUNDLED_CONDENSED = {
    False: "BarlowSemiCondensed-SemiBold.ttf",
    True: "BarlowSemiCondensed-Bold.ttf",
}
CONDENSED_FAMILY = "Barlow Semi Condensed"


def _fc_match(pattern: str) -> str | None:
    """TrueType file fc-match picks for a pattern, None when unusable."""
    try:
        out = subprocess.run(
            ["fc-match", "-f", "%{file}", pattern],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if out.lower().endswith(".ttf") and Path(out).is_file():
        return out
    return None


@lru_cache(maxsize=8)
def find_font_file(bold: bool, condensed: bool = False) -> str | None:  # noqa: FBT001, FBT002
    """Path of a TrueType sans font, bold or regular, None when none is found.

    condensed asks for the bundled Barlow Semi Condensed (SemiBold, or Bold
    when bold), then fc-match for Fira Sans Condensed, then the plain sans.
    """
    if condensed:
        bundled = FONT_DIR / BUNDLED_CONDENSED[bold]
        if bundled.is_file():
            return str(bundled)
        found = _fc_match("Fira Sans Condensed:bold" if bold else "Fira Sans Condensed")
        if found and "condensed" in found.lower():
            return found
    found = _fc_match("sans:bold" if bold else "sans")
    if found:
        return found
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


def _missing_glyph(font: ImageFont.FreeTypeFont, char: str) -> bool:
    """True when the font has no glyph for char and would draw its notdef box."""
    if ord(char) < 128:
        return False
    return bytes(font.getmask(char)) == bytes(font.getmask("\ufdd0"))


def _runs(
    line: str, font: ImageFont.FreeTypeFont, fallback: ImageFont.FreeTypeFont | None
) -> list[tuple[str, ImageFont.FreeTypeFont]]:
    """Split a line into runs, each drawn with the font that has its glyphs."""
    if fallback is None:
        return [(line, font)]
    runs: list[tuple[str, ImageFont.FreeTypeFont]] = []
    for char in line:
        chosen = fallback if _missing_glyph(font, char) else font
        if runs and runs[-1][1] is chosen:
            runs[-1] = (runs[-1][0] + char, chosen)
        else:
            runs.append((char, chosen))
    return runs


def render_text(
    text: str,
    style: TextStyle,
    font_path: str,
    scaling: float,
    wraplength: int = 0,
    fallback_path: str | None = None,
) -> Image.Image:
    """Draw text as an RGB image, fg on bg, at round(points x scaling) pixels.

    Characters the font lacks (the credit sign in Barlow) are drawn with
    fallback_path's font when given.
    """
    pixels = max(1, round(style.size * scaling))
    font = _load_font(font_path, pixels)
    fallback = _load_font(fallback_path, pixels) if fallback_path else None
    lines = wrap_text(text, font, wraplength)
    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    all_runs = [_runs(line, font, fallback) for line in lines]
    width = max(
        1,
        *(math.ceil(sum(f.getlength(run) for run, f in runs)) for runs in all_runs),
    )
    # A little room so glyph overhang is not clipped
    width += max(1, pixels // 10)
    image = Image.new("RGB", (width, line_h * len(lines)), style.bg)
    draw = ImageDraw.Draw(image)
    for i, runs in enumerate(all_runs):
        x = 0.0
        for run, run_font in runs:
            draw.text((x, i * line_h), run, font=run_font, fill=style.fg)
            x += run_font.getlength(run)
    return image


def render_photo(
    root: tk.Misc,
    text: str,
    style: TextStyle,
    font_path: str,
    wraplength: int = 0,
    fallback_path: str | None = None,
) -> ImageTk.PhotoImage:
    """render_text as a Tk image owned by root."""
    image = render_text(
        text, style, font_path, tk_scaling(root), wraplength, fallback_path
    )
    return ImageTk.PhotoImage(image, master=root)
