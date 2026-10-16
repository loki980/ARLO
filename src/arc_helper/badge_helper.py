"""Badge surface helper: draws gold borders, tags and bar patches over the game.

Runs under the system Python (it needs gi with GTK 3, cairo and python-xlib,
which ARLO's own environment does not have), so it imports nothing from
arc_helper. BadgeSurface in badge_surface.py starts it and talks to it.

Live mode (--live) opens one full-screen GTK 3 POPUP window on X11 or
XWayland with the RGBA visual, never takes focus, stays above other windows
and has an empty input shape, so every click falls through to the game. It
reads one JSON message per line from stdin:

    {"op": "draw", "badges": [...]}   replace the drawn set with these badges
    {"op": "clear"}                   remove every badge
    {"op": "translate", "region": [...], "dy": 12}
                                      move the drawn badges inside region

and exits when stdin closes. Badge rects and the region are [left, top,
right, bottom] in physical screen pixels, right and bottom exclusive. A
translate moves every drawn badge whose rect's centre lies inside region
down by dy px (up when negative), drops a moved badge that no longer meets
the region and draws the rest clipped to it (translate_badges).

PNG mode (--png OUT --size W H) reads the messages on stdin, applies them in
order and draws the result with the same cairo code onto a transparent
W x H PNG, with no display.
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
# Tags draw in Bold. Patches draw in SemiBold of the same family, whose stroke
# matches the game's own stack count (scripts/font_match.py measures it)
TAG_FONT_WEIGHT = "Bold"
PATCH_FONT_FILE = FONT_DIR / "BarlowSemiCondensed-SemiBold.ttf"
PATCH_FONT_FAMILY = FONT_FAMILY
PATCH_FONT_WEIGHT = "Semi-Bold"
FALLBACK_FAMILY = "Sans"
CORNERS = ("tl", "tr", "bl", "br")
# The tag's text height as a share of the slot height, and its limits in px
TAG_TEXT_SHARE = 0.17
TAG_TEXT_MIN = 11
TAG_TEXT_MAX = 40
# The check mark's colour and its width as a share of the tag's text height
CHECK_GREEN = "#5fd35f"
CHECK_SHARE = 0.8
OUTLINE_WIDTH = 2
# A patch's font size as a share of the bar height (digits 24 px tall on the
# 44 px bar of a 4K tile, like the game's count), the smallest it may shrink
# to, the step it shrinks by, and its padding and icon gap in px
PATCH_TEXT_SHARE = 0.8
PATCH_TEXT_MIN_SHARE = 0.57
PATCH_TEXT_STEP = 0.9
# The game's own count ink, measured on 44 counts in real 4K captures
PATCH_TEXT_RGB = (0xF9 / 255, 0xEE / 255, 0xDF / 255)
PATCH_PAD = 8
# The digits' baseline sits this share of the bar height above the bar bottom,
# where the game's count sits (9.5 px on the 44 px bar)
PATCH_BASELINE_SHARE = 0.216
PATCH_ICON_GAP = 4
# Draw order: patches first, then tags, then borders on top
DRAW_ORDER = {"patch": 0, "tag": 1, "border": 2}
KINDS = set(DRAW_ORDER)
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


def parse_rgb(value) -> tuple[float, float, float, float]:
    """Turn an [r, g, b] list of 0..255 ints, or a '#rrggbb' string, into RGBA."""
    if isinstance(value, str):
        return parse_color(value)
    if len(value) != 3:
        msg = f"colour {value!r} is not [r, g, b]"
        raise ValueError(msg)
    return (*(int(c) / 255 for c in value), 1.0)


def parse_message(line: str) -> tuple[str, list[dict]]:
    """Return the op and badge list of one draw or clear line from BadgeSurface."""
    return _parsed(json.loads(line))


def _parsed(message: dict) -> tuple[str, list[dict]]:
    op = message.get("op")
    if op not in {"draw", "clear"}:
        msg = f"unknown op {op!r}"
        raise ValueError(msg)
    badges = list(message.get("badges", [])) if op == "draw" else []
    for badge in badges:
        if badge.get("kind") not in KINDS:
            msg = f"unknown badge kind {badge.get('kind')!r}"
            raise ValueError(msg)
        if len(badge.get("rect", ())) != 4:
            msg = "a badge rect needs left, top, right and bottom"
            raise ValueError(msg)
        if badge["kind"] == "tag" and badge.get("corner", "bl") not in CORNERS:
            msg = f"unknown tag corner {badge['corner']!r}"
            raise ValueError(msg)
        if (
            badge["kind"] == "patch"
            and not {
                "icon_right",
                "patch_right",
                "colour",
            }
            <= badge.keys()
        ):
            msg = "a patch needs icon_right, patch_right and colour"
            raise ValueError(msg)
    return op, badges


def _inside(x: float, y: float, region) -> bool:
    left, top, right, bottom = region
    return left <= x < right and top <= y < bottom


def translate_badges(badges: list[dict], region, dy: int) -> list[dict]:
    """The drawn badges after their region scrolled by dy px.

    A badge whose rect's centre lies inside region, or that an earlier
    translate of this region moved (clipped to it), moves down by dy (up
    when negative) and is clipped to region from then on. A moved badge
    whose rect no longer meets the region is dropped. Other badges stay.
    """
    left, top, right, bottom = (int(v) for v in region)
    kept = []
    clip = [left, top, right, bottom]
    for badge in badges:
        x0, y0, x1, y1 = badge["rect"]
        followed = badge.get("clip") == clip
        if not followed and not _inside((x0 + x1) / 2, (y0 + y1) / 2, clip):
            kept.append(badge)
            continue
        y0, y1 = y0 + dy, y1 + dy
        if y1 <= top or y0 >= bottom or x1 <= left or x0 >= right:
            continue
        kept.append({**badge, "rect": [x0, y0, x1, y1], "clip": clip})
    return kept


def apply_message(badges: list[dict], line: str) -> list[dict]:
    """The badges drawn after one JSON line from BadgeSurface, given those drawn now."""
    message = json.loads(line)
    if message.get("op") == "translate":
        region, dy = message.get("region", ()), message.get("dy")
        if len(region) != 4 or not isinstance(dy, int):
            msg = "a translate needs a region of four edges and an int dy"
            raise ValueError(msg)
        return translate_badges(badges, region, dy)
    op, drawn = _parsed(message)
    return drawn if op == "draw" else []


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


def register_fonts() -> bool:
    """register_font for the tag file and the patch file, True when the tag's took."""
    registered = register_font()
    register_font(PATCH_FONT_FILE)
    return registered


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
    # BadgeSurface always sends the colour (cell_border.HOVER_GOLD by default)
    cr.set_source_rgba(*parse_color(badge["color"]))
    cr.set_line_width(width)
    cr.stroke()


class _Text:
    """Measures and draws one line of text with Pango, or cairo's toy API."""

    def __init__(
        self, cr, text: str, pixels: float, family: str, weight: str = TAG_FONT_WEIGHT
    ) -> None:
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
            self.baseline = ascent
            return
        self.layout = PangoCairo.create_layout(cr)
        desc = Pango.FontDescription.from_string(
            f"{family}, {FALLBACK_FAMILY} {weight}"
        )
        desc.set_absolute_size(pixels * Pango.SCALE)
        self.layout.set_font_description(desc)
        self.layout.set_text(text, -1)
        ink, logical = self.layout.get_pixel_extents()
        # Centre on the ink so the pill looks even above and below the digits
        self.width = logical.width
        self.height = ink.height
        self.ascent = -ink.y
        self.baseline = self.layout.get_baseline() / Pango.SCALE
        self.pango = PangoCairo

    def draw_on_baseline(self, x: float, baseline: float) -> None:
        """Draw with the text's baseline at the y given."""
        if self.layout is None:
            self.cr.move_to(x, baseline)
            self.cr.show_text(self.text)
            return
        self.cr.move_to(x, baseline - self.baseline)
        self.pango.show_layout(self.cr, self.layout)

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
    check_w = text.height * CHECK_SHARE if badge.get("check") else 0.0
    check_gap = pad_x if check_w else 0.0
    pill_w = check_w + check_gap + text.width + 2 * pad_x
    pill_h = text.height + 2 * pad_y
    corner = badge.get("corner", "bl")
    x = left + inset if corner[1] == "l" else right - inset - pill_w
    y = top + inset if corner[0] == "t" else bottom - inset - pill_h
    _rounded_rect(cr, x, y, pill_w, pill_h, pill_h / 2)
    cr.set_source_rgba(*parse_color(badge.get("bg", "#141414d9")))
    cr.fill()
    if badge.get("outline") is not None:
        # The outline's inner edge lies on the pill's edge
        half = OUTLINE_WIDTH / 2
        _rounded_rect(
            cr,
            x - half,
            y - half,
            pill_w + OUTLINE_WIDTH,
            pill_h + OUTLINE_WIDTH,
            pill_h / 2 + half,
        )
        cr.set_source_rgba(*parse_rgb(badge["outline"]))
        cr.set_line_width(OUTLINE_WIDTH)
        cr.stroke()
    if check_w:
        _draw_check(cr, x + pad_x, y + pad_y, check_w, text.height)
    cr.set_source_rgba(*parse_color(badge.get("fg", "#ffffff")))
    text.draw(x + pad_x + check_w + check_gap, y + pad_y)


def _draw_check(cr, x: float, y: float, w: float, h: float) -> None:
    """A green check mark in the w x h box at x, y."""
    import cairo

    cr.save()
    cr.set_line_width(max(1.5, h * 0.18))
    cr.set_line_cap(cairo.LINE_CAP_ROUND)
    cr.set_line_join(cairo.LINE_JOIN_ROUND)
    cr.move_to(x + w * 0.1, y + h * 0.55)
    cr.line_to(x + w * 0.4, y + h * 0.85)
    cr.line_to(x + w * 0.95, y + h * 0.15)
    cr.set_source_rgba(*parse_color(CHECK_GREEN))
    cr.stroke()
    cr.restore()


def _bottom_left_rounded_rect(
    cr, x: float, y: float, w: float, h: float, r: float
) -> None:
    """A rectangle with only its bottom-left corner rounded by r."""
    r = max(0.0, min(r, w, h))
    cr.new_sub_path()
    cr.move_to(x, y)
    cr.line_to(x + w, y)
    cr.line_to(x + w, y + h)
    cr.line_to(x + r, y + h)
    cr.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    cr.close_path()


def patch_text(cr, badge: dict, family: str):
    """The patch's text and fill right edge, or None when the text cannot fit.

    The font size starts at PATCH_TEXT_SHARE of the bar height and shrinks by
    PATCH_TEXT_STEP down to PATCH_TEXT_MIN_SHARE until it fits, padded by
    PATCH_PAD on both sides, before patch_right.
    """
    left, top, _right, bottom = badge["rect"]
    bar_h = bottom - top
    patch_right = badge["patch_right"]
    smallest = bar_h * PATCH_TEXT_MIN_SHARE
    pixels = bar_h * PATCH_TEXT_SHARE
    while True:
        text = _Text(
            cr,
            str(badge.get("text", "")),
            pixels,
            PATCH_FONT_FAMILY if family == FONT_FAMILY else family,
            PATCH_FONT_WEIGHT,
        )
        text_edge = left + PATCH_PAD + text.width + PATCH_PAD
        if text_edge <= patch_right:
            edge = max(badge["icon_right"] + PATCH_ICON_GAP, text_edge)
            # Whole pixels keep the fill's right edge crisp
            return text, min(math.ceil(edge), patch_right)
        if pixels <= smallest:
            return None
        pixels = max(smallest, pixels * PATCH_TEXT_STEP)


def _draw_patch(cr, badge: dict, family: str) -> None:
    fitted = patch_text(cr, badge, family)
    if fitted is None:
        return
    text, edge = fitted
    left, top, _right, bottom = badge["rect"]
    _bottom_left_rounded_rect(
        cr, left, top, edge - left, bottom - top, float(badge.get("radius", 6))
    )
    cr.set_source_rgba(*parse_rgb(badge["colour"]))
    cr.fill()
    cr.set_source_rgba(*PATCH_TEXT_RGB, 1)
    baseline = bottom - PATCH_BASELINE_SHARE * (bottom - top)
    text.draw_on_baseline(left + PATCH_PAD, baseline)


def draw_badges(cr, badges: list[dict], family: str = FONT_FAMILY) -> None:
    """Clear the surface to transparent and draw every badge.

    Patches go first, then tags, then borders, so an outline is never hidden
    under a fill. Badges of one kind keep their message order.
    """
    import cairo

    cr.save()
    cr.set_operator(cairo.OPERATOR_CLEAR)
    cr.paint()
    cr.restore()
    cr.set_operator(cairo.OPERATOR_OVER)
    for badge in sorted(badges, key=lambda b: DRAW_ORDER[b["kind"]]):
        clip = badge.get("clip")
        if clip is not None:
            # A badge a translate moved stays inside its scrolled region
            cr.save()
            cr.rectangle(clip[0], clip[1], clip[2] - clip[0], clip[3] - clip[1])
            cr.clip()
        if badge["kind"] == "border":
            _draw_border(cr, badge)
        elif badge["kind"] == "patch":
            _draw_patch(cr, badge, family)
        else:
            _draw_tag(cr, badge, family)
        if clip is not None:
            cr.restore()


def render_png(badges: list[dict], width: int, height: int, out: Path) -> None:
    """Draw badges on a transparent width x height PNG."""
    import cairo

    family = FONT_FAMILY if register_fonts() else FALLBACK_FAMILY
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


def draw_scale(
    window: tuple[int, int],
    physical: tuple[int, int] | None,
    gtk_scale: int,
) -> tuple[float, float]:
    """Factors from physical px to the window's coordinates, per axis.

    With the physical screen size the factor is window / physical, so a rect in
    physical px lands on the same physical pixels. Without it, fall back to
    1 / gtk_scale on both axes.
    """
    if physical is None or physical[0] <= 0 or physical[1] <= 0:
        return 1 / gtk_scale, 1 / gtk_scale
    return window[0] / physical[0], window[1] / physical[1]


def checked_physical(physical: tuple[int, int] | None) -> tuple[int, int] | None:
    """The physical size when usable. Otherwise None, with one warning on stderr."""
    if physical is not None and physical[0] > 0 and physical[1] > 0:
        return physical
    print(
        "badge helper: no physical screen size given, scaling by GTK's factor",
        file=sys.stderr,
    )
    return None


def run_live(physical: tuple[int, int] | None = None) -> int:
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
    physical = checked_physical(physical)
    family = FONT_FAMILY if register_fonts() else FALLBACK_FAMILY
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
        # Badge rects are physical pixels, the window is in X root coordinates
        sx, sy = draw_scale(
            (root.get_width(), root.get_height()), physical, win.get_scale_factor()
        )
        cr.scale(sx, sy)
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
                state["badges"] = apply_message(state["badges"], line)
            except ValueError as e:
                print(f"badge helper: ignored a bad message: {e}", file=sys.stderr)
                return True
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
    """Apply the messages on stdin in order and write the result as a transparent PNG."""
    badges: list[dict] = []
    for line in sys.stdin:
        if line.strip():
            badges = apply_message(badges, line)
    render_png(badges, width, height, out)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="show the surface")
    mode.add_argument("--png", type=Path, help="write one draw message to this PNG")
    parser.add_argument("--size", type=int, nargs=2, metavar=("W", "H"))
    parser.add_argument(
        "--physical",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        help="physical screen size, for --live",
    )
    args = parser.parse_args()
    if args.live:
        return run_live(tuple(args.physical) if args.physical else None)
    if args.size is None:
        parser.error("--png needs --size W H")
    return run_png(args.png, *args.size)


if __name__ == "__main__":
    sys.exit(main())
