"""Badge surface helper: draws gold borders and price tags over the game.

Runs under the system Python (it needs gi with GTK 3, cairo and python-xlib,
which ARLO's own environment does not have), so it imports nothing from
arc_helper. BadgeSurface in badge_surface.py starts it and talks to it.

Live mode (--live) opens one full-screen GTK 3 POPUP window on X11 or
XWayland with the RGBA visual, never takes focus, stays above other windows
and has an empty input shape, so every click falls through to the game. It
reads one JSON message per line from stdin:

    {"op": "draw", "badges": [...]}   replace the drawn set with these badges
    {"op": "clear"}                   remove every badge

and exits when stdin closes. Badge rects are [left, top, right, bottom] in
physical screen pixels, right and bottom exclusive.

PNG mode (--png OUT --size W H) reads one draw message from stdin and draws
it with the same cairo code onto a transparent W x H PNG, with no display.
"""

import argparse
import ctypes
import ctypes.util
import json
import math
import os
import sys
from pathlib import Path

FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
FONT_FILE = FONT_DIR / "BarlowSemiCondensed-Bold.ttf"
FONT_FAMILY = "Barlow Semi Condensed"
FALLBACK_FAMILY = "Sans"
CORNERS = ("tl", "tr", "bl", "br")
# The tag's text height as a share of the slot height, and its limits in px
TAG_TEXT_SHARE = 0.17
TAG_TEXT_MIN = 11
TAG_TEXT_MAX = 40
RAISE_MS = 1000


def parse_color(text: str) -> tuple[float, float, float, float]:
    """Turn '#rrggbb' or '#rrggbbaa' into cairo's 0..1 RGBA."""
    digits = text.lstrip("#")
    if len(digits) not in {6, 8}:
        msg = f"colour {text!r} is not #rrggbb or #rrggbbaa"
        raise ValueError(msg)
    values = [int(digits[i : i + 2], 16) / 255 for i in range(0, len(digits), 2)]
    if len(values) == 3:
        values.append(1.0)
    return tuple(values)


def parse_message(line: str) -> tuple[str, list[dict]]:
    """Return the op and badge list of one JSON line from BadgeSurface."""
    message = json.loads(line)
    op = message.get("op")
    if op not in {"draw", "clear"}:
        msg = f"unknown op {op!r}"
        raise ValueError(msg)
    badges = list(message.get("badges", [])) if op == "draw" else []
    for badge in badges:
        if badge.get("kind") not in {"border", "tag"}:
            msg = f"unknown badge kind {badge.get('kind')!r}"
            raise ValueError(msg)
        if len(badge.get("rect", ())) != 4:
            msg = "a badge rect needs left, top, right and bottom"
            raise ValueError(msg)
        if badge["kind"] == "tag" and badge.get("corner", "bl") not in CORNERS:
            msg = f"unknown tag corner {badge['corner']!r}"
            raise ValueError(msg)
    return op, badges


def register_font(path: Path = FONT_FILE) -> bool:
    """Add the bundled font to fontconfig so cairo and Pango find it by name.

    False when fontconfig or the file is missing, and the fallback font is
    used instead.
    """
    lib = ctypes.util.find_library("fontconfig")
    if lib is None or not path.is_file():
        return False
    try:
        fc = ctypes.CDLL(lib)
        fc.FcConfigAppFontAddFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        return bool(fc.FcConfigAppFontAddFile(None, os.fsencode(path)))
    except (OSError, AttributeError):
        return False


def _rounded_rect(cr, x: float, y: float, w: float, h: float, r: float) -> None:
    r = max(0.0, min(r, w / 2, h / 2))
    cr.new_sub_path()
    cr.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    cr.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    cr.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    cr.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    cr.close_path()


def _draw_border(cr, badge: dict) -> None:
    left, top, right, bottom = badge["rect"]
    width = float(badge.get("width", 3))
    # The stroke sits wholly inside the rect, its outer edge on the rect edge
    half = width / 2
    _rounded_rect(
        cr,
        left + half,
        top + half,
        right - left - width,
        bottom - top - width,
        float(badge.get("radius", 4)) - half,
    )
    cr.set_source_rgba(*parse_color(badge.get("color", "#f2b134")))
    cr.set_line_width(width)
    cr.stroke()


class _Text:
    """Measures and draws one line of text with Pango, or cairo's toy API."""

    def __init__(self, cr, text: str, pixels: int, family: str) -> None:
        self.cr = cr
        self.text = text
        self.layout = None
        try:
            import gi

            gi.require_version("Pango", "1.0")
            gi.require_version("PangoCairo", "1.0")
            from gi.repository import Pango
            from gi.repository import PangoCairo
        except (ImportError, ValueError):
            import cairo

            cr.select_font_face(family, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            cr.set_font_size(pixels)
            ext = cr.text_extents(text)
            ascent = cr.font_extents()[0]
            self.width, self.height, self.ascent = ext.x_advance, pixels, ascent
            return
        self.layout = PangoCairo.create_layout(cr)
        desc = Pango.FontDescription.from_string(f"{family}, {FALLBACK_FAMILY} Bold")
        desc.set_absolute_size(pixels * Pango.SCALE)
        self.layout.set_font_description(desc)
        self.layout.set_text(text, -1)
        ink, logical = self.layout.get_pixel_extents()
        # Centre on the ink so the pill looks even above and below the digits
        self.width = logical.width
        self.height = ink.height
        self.ascent = -ink.y
        self.pango = PangoCairo

    def draw(self, x: float, y: float) -> None:
        """Draw with the text's ink top at y."""
        if self.layout is None:
            self.cr.move_to(x, y + self.ascent)
            self.cr.show_text(self.text)
            return
        self.cr.move_to(x, y + self.ascent)
        self.pango.show_layout(self.cr, self.layout)


def _draw_tag(cr, badge: dict, family: str) -> None:
    left, top, right, bottom = badge["rect"]
    pixels = round((bottom - top) * TAG_TEXT_SHARE)
    pixels = max(TAG_TEXT_MIN, min(TAG_TEXT_MAX, pixels))
    text = _Text(cr, str(badge.get("text", "")), pixels, family)
    pad_x = pixels * 0.35
    pad_y = pixels * 0.3
    inset = max(3.0, pixels * 0.3)
    pill_w = text.width + 2 * pad_x
    pill_h = text.height + 2 * pad_y
    corner = badge.get("corner", "bl")
    x = left + inset if corner[1] == "l" else right - inset - pill_w
    y = top + inset if corner[0] == "t" else bottom - inset - pill_h
    _rounded_rect(cr, x, y, pill_w, pill_h, pill_h / 2)
    cr.set_source_rgba(*parse_color(badge.get("bg", "#141414d9")))
    cr.fill()
    cr.set_source_rgba(*parse_color(badge.get("fg", "#ffffff")))
    text.draw(x + pad_x, y + pad_y)


def draw_badges(cr, badges: list[dict], family: str = FONT_FAMILY) -> None:
    """Clear the surface to transparent and draw every badge, in order."""
    import cairo

    cr.save()
    cr.set_operator(cairo.OPERATOR_CLEAR)
    cr.paint()
    cr.restore()
    cr.set_operator(cairo.OPERATOR_OVER)
    for badge in badges:
        if badge["kind"] == "border":
            _draw_border(cr, badge)
        else:
            _draw_tag(cr, badge, family)


def render_png(badges: list[dict], width: int, height: int, out: Path) -> None:
    """Draw badges on a transparent width x height PNG."""
    import cairo

    family = FONT_FAMILY if register_font() else FALLBACK_FAMILY
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    cr = cairo.Context(surface)
    draw_badges(cr, badges, family)
    surface.flush()
    surface.write_to_png(str(out))


def _empty_input_shape(xid: int) -> bool:
    """Empty the SHAPE input region on the window and every parent below root."""
    from Xlib import X
    from Xlib import display as xlib_display
    from Xlib.ext import shape

    xdisplay = xlib_display.Display()
    try:
        if not xdisplay.has_extension("SHAPE"):
            return False
        root_id = xdisplay.screen().root.id
        xwindow = xdisplay.create_resource_object("window", xid)
        while xwindow.id != root_id:
            xwindow.shape_rectangles(shape.SO.Set, shape.SK.Input, X.Unsorted, 0, 0, [])
            parent = xwindow.query_tree().parent
            if parent is None or parent.id == root_id:
                break
            xwindow = parent
        xdisplay.sync()
        return True
    finally:
        xdisplay.close()


def run_live() -> int:
    """Show the surface and redraw it on every stdin message until stdin closes."""
    if not os.environ.get("DISPLAY"):
        print(
            "badge helper: DISPLAY is unset, no X display to draw on", file=sys.stderr
        )
        return 2
    os.environ["GDK_BACKEND"] = "x11"
    import gi

    gi.require_version("Gtk", "3.0")
    gi.require_version("Gdk", "3.0")
    gi.require_version("GdkX11", "3.0")
    from gi.repository import Gdk
    from gi.repository import GdkX11  # noqa: F401 - adds get_xid() to Gdk.Window
    from gi.repository import GLib
    from gi.repository import Gtk

    if not Gtk.init_check(sys.argv)[0] or Gdk.Display.get_default() is None:
        print("badge helper: GTK 3 could not open the X display", file=sys.stderr)
        return 2
    screen = Gdk.Screen.get_default()
    visual = screen.get_rgba_visual()
    if visual is None or not screen.is_composited():
        print(
            "badge helper: no composited RGBA visual on this display", file=sys.stderr
        )
        return 2
    family = FONT_FAMILY if register_font() else FALLBACK_FAMILY
    root = screen.get_root_window()
    win = Gtk.Window(type=Gtk.WindowType.POPUP)  # POPUP means override-redirect
    win.set_visual(visual)
    win.set_app_paintable(True)
    win.set_accept_focus(False)
    win.set_keep_above(True)
    win.move(0, 0)
    win.resize(root.get_width(), root.get_height())
    state = {"badges": []}
    area = Gtk.DrawingArea()

    def on_draw(_area, cr) -> bool:
        # Badge rects are physical pixels, GTK draws in logical ones
        scale = win.get_scale_factor()
        cr.scale(1 / scale, 1 / scale)
        draw_badges(cr, state["badges"], family)
        return True

    area.connect("draw", on_draw)
    win.add(area)
    win.show_all()
    gdk_window = win.get_window()
    try:
        _empty_input_shape(gdk_window.get_xid())
    except Exception as e:  # noqa: BLE001 - the badges still show, they eat clicks
        print(
            f"badge helper: could not make the surface click-through: {e}",
            file=sys.stderr,
        )

    stdin = GLib.IOChannel.unix_new(sys.stdin.fileno())

    def on_input(channel, condition) -> bool:
        if condition & GLib.IOCondition.IN:
            status, line, _length, _end = channel.read_line()
            if status == GLib.IOStatus.EOF or not line:
                Gtk.main_quit()
                return False
            try:
                op, badges = parse_message(line)
            except ValueError as e:
                print(f"badge helper: ignored a bad message: {e}", file=sys.stderr)
                return True
            state["badges"] = badges if op == "draw" else []
            area.queue_draw()
            return True
        Gtk.main_quit()
        return False

    GLib.io_add_watch(
        stdin,
        GLib.PRIORITY_DEFAULT,
        GLib.IOCondition.IN | GLib.IOCondition.HUP | GLib.IOCondition.ERR,
        on_input,
    )
    GLib.timeout_add(RAISE_MS, lambda: gdk_window.raise_() or True)  # stay on top
    Gtk.main()
    return 0


def run_png(out: Path, width: int, height: int) -> int:
    """Read one draw message from stdin and write it as a transparent PNG."""
    line = sys.stdin.readline()
    _op, badges = parse_message(line) if line.strip() else ("clear", [])
    render_png(badges, width, height, out)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="show the surface")
    mode.add_argument("--png", type=Path, help="write one draw message to this PNG")
    parser.add_argument("--size", type=int, nargs=2, metavar=("W", "H"))
    args = parser.parse_args()
    if args.live:
        return run_live()
    if args.size is None:
        parser.error("--png needs --size W H")
    return run_png(args.png, *args.size)


if __name__ == "__main__":
    sys.exit(main())
