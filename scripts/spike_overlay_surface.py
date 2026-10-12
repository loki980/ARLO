# ruff: noqa: INP001, FBT003
"""Spike: one full-screen, transparent, click-through, topmost surface.

Usage: uv run python scripts/spike_overlay_surface.py --mode x11|layer|auto --seconds 15

Draws a 4 px amber rectangle outline at the screen centre, a '3/8' badge in
the top-left quarter and a crosshair at the pointer (static at the centre
where the pointer position cannot be read). Exits 0 with one report line, or
2 with one plain sentence on any missing dependency or unsupported session.
Findings and the test procedure: docs/spikes/2026-10-11-overlay-surface.md
"""

import argparse
import ctypes.util
import os
import subprocess
import sys

AMBER = (1.0, 0.72, 0.1)
LAYER_LIB = "gtk4-layer-shell"


class SpikeError(Exception):
    """A failure that ends the spike with one sentence and exit code 2."""


def _gi(versions: dict[str, str]) -> None:
    try:
        import gi

        for namespace, version in versions.items():
            gi.require_version(namespace, version)
    except (ImportError, ValueError) as e:
        msg = f"PyGObject or a GI namespace is missing ({e}), run uv sync --all-extras."
        raise SpikeError(msg) from None


def draw(cr, width: float, height: float, pointer, scale: float) -> None:
    """Paint the three test marks on a cleared, fully transparent surface."""
    import cairo

    cr.set_operator(cairo.OPERATOR_CLEAR)
    cr.paint()
    cr.set_operator(cairo.OPERATOR_OVER)
    # 4 device px amber outline, a quarter of the screen, centred.
    cr.set_source_rgb(*AMBER)
    cr.set_line_width(4 / scale)
    cr.rectangle(width * 3 / 8, height * 3 / 8, width / 4, height / 4)
    cr.stroke()
    # Filled '3/8' badge in the top-left quarter.
    bx, by, bw, bh = width / 8, height / 8, 72, 36
    cr.set_source_rgba(0.08, 0.08, 0.08, 0.85)
    cr.rectangle(bx, by, bw, bh)
    cr.fill()
    cr.set_source_rgb(*AMBER)
    cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    cr.set_font_size(22)
    ext = cr.text_extents("3/8")
    cr.move_to(bx + (bw - ext.width) / 2 - ext.x_bearing, by + (bh + ext.height) / 2)
    cr.show_text("3/8")
    # Thin crosshair, 24 logical px arms.
    cx, cy = pointer or (width / 2, height / 2)
    cr.set_source_rgb(1.0, 1.0, 1.0)
    cr.set_line_width(2 / scale)
    cr.move_to(cx - 24, cy)
    cr.line_to(cx + 24, cy)
    cr.move_to(cx, cy - 24)
    cr.line_to(cx, cy + 24)
    cr.stroke()


def x11_empty_input(xid: int) -> bool:
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


def run_x11(seconds: int) -> str:
    if not os.environ.get("DISPLAY"):
        msg = "No X display is reachable (DISPLAY is unset), so x11 mode cannot run."
        raise SpikeError(msg)
    os.environ["GDK_BACKEND"] = "x11"
    _gi({"Gtk": "3.0", "Gdk": "3.0", "GdkX11": "3.0"})
    from gi.repository import Gdk
    from gi.repository import GdkX11  # noqa: F401 - adds get_xid() to Gdk.Window
    from gi.repository import GLib
    from gi.repository import Gtk

    if not Gtk.init_check(sys.argv)[0] or Gdk.Display.get_default() is None:
        msg = f"GTK3 could not open X display {os.environ['DISPLAY']}, so x11 mode cannot run."
        raise SpikeError(msg)
    screen = Gdk.Screen.get_default()
    visual = screen.get_rgba_visual()
    if visual is None or not screen.is_composited():
        msg = "The X server has no composited RGBA visual, so per-pixel alpha is impossible in x11 mode."
        raise SpikeError(msg)
    root = screen.get_root_window()
    width, height = root.get_width(), root.get_height()

    win = Gtk.Window(type=Gtk.WindowType.POPUP)  # POPUP means override-redirect
    win.set_visual(visual)
    win.set_app_paintable(True)
    win.set_accept_focus(False)
    win.set_keep_above(True)
    win.move(0, 0)
    win.resize(width, height)
    scale = win.get_scale_factor()
    pointer: list = [None]
    area = Gtk.DrawingArea()
    area.connect("draw", lambda _a, cr: draw(cr, width, height, pointer[0], scale))
    win.add(area)
    win.show_all()
    gdk_window = win.get_window()
    try:
        passthrough = x11_empty_input(gdk_window.get_xid())
    except Exception:  # noqa: BLE001 - reported as passthrough=no, as in clickthrough.py
        passthrough = False
    device = Gdk.Display.get_default().get_default_seat().get_pointer()

    def tick() -> bool:
        _, x, y = device.get_position()
        if (x, y) != pointer[0]:
            pointer[0] = (x, y)
            area.queue_draw()
        return True

    GLib.timeout_add(16, tick)
    GLib.timeout_add(1000, lambda: gdk_window.raise_() or True)  # stay on top
    GLib.timeout_add(seconds * 1000, Gtk.main_quit)
    Gtk.main()
    xft = Gtk.Settings.get_default().get_property("gtk-xft-dpi") / 1024 / 96
    return (
        f"mode=x11 surface={width}x{height} logical, {width * scale}x{height * scale} px "
        f"scale={scale} (xft {xft:.2f}) passthrough={'yes' if passthrough else 'no'} "
        f"pointer=live"
    )


def run_layer(seconds: int) -> str:
    if not os.environ.get("WAYLAND_DISPLAY"):
        msg = "No Wayland display is reachable (WAYLAND_DISPLAY is unset), so layer mode cannot run."
        raise SpikeError(msg)
    lib = ctypes.util.find_library(LAYER_LIB)
    if lib is None:
        msg = f"{LAYER_LIB} is not installed (pacman -S {LAYER_LIB}), so layer mode cannot run."
        raise SpikeError(msg)
    preload = os.environ.get("LD_PRELOAD", "")
    if lib not in preload:
        # The library must load before libwayland-client, so restart with it preloaded.
        env = {**os.environ, "LD_PRELOAD": f"{lib} {preload}".strip()}
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    os.environ["GDK_BACKEND"] = "wayland"
    _gi({"Gtk": "4.0", "Gdk": "4.0", "Gtk4LayerShell": "1.0"})
    import cairo
    from gi.repository import Gdk
    from gi.repository import GLib
    from gi.repository import Gtk
    from gi.repository import Gtk4LayerShell

    if not Gtk.init_check():
        msg = "GTK4 could not open the Wayland display, so layer mode cannot run."
        raise SpikeError(msg)
    if not Gtk4LayerShell.is_supported():
        msg = "The compositor does not offer wlr-layer-shell, so layer mode cannot run."
        raise SpikeError(msg)

    win = Gtk.Window()
    Gtk4LayerShell.init_for_window(win)
    Gtk4LayerShell.set_namespace(win, "arlo-spike")
    Gtk4LayerShell.set_layer(win, Gtk4LayerShell.Layer.OVERLAY)
    for edge in ("LEFT", "RIGHT", "TOP", "BOTTOM"):
        Gtk4LayerShell.set_anchor(win, getattr(Gtk4LayerShell.Edge, edge), True)
    Gtk4LayerShell.set_exclusive_zone(win, -1)  # cover panels too, reserve nothing
    Gtk4LayerShell.set_keyboard_mode(win, Gtk4LayerShell.KeyboardMode.NONE)
    css = Gtk.CssProvider()
    css.load_from_string("window { background: transparent; }")
    Gtk.StyleContext.add_provider_for_display(
        Gdk.Display.get_default(), css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
    )
    seen = {"size": (0, 0), "scale": 1.0}

    def on_draw(_area, cr, w: int, h: int) -> None:
        surface = win.get_surface()
        seen["size"], seen["scale"] = (w, h), surface.get_scale()
        surface.set_input_region(cairo.Region())  # empty: every click falls through
        draw(cr, w, h, None, seen["scale"])

    area = Gtk.DrawingArea()
    area.set_draw_func(on_draw)
    win.set_child(area)
    win.present()
    loop = GLib.MainLoop()
    GLib.timeout_add(seconds * 1000, loop.quit)
    loop.run()
    (w, h), scale = seen["size"], seen["scale"]
    if w == 0:
        msg = "The layer surface was never drawn, so the compositor did not map it."
        raise SpikeError(msg)
    return (
        f"mode=layer surface={w}x{h} logical, {round(w * scale)}x{round(h * scale)} px "
        f"scale={scale:g} passthrough=yes pointer=static"
    )


def run_auto(seconds: int) -> str:
    cmd = [sys.executable, __file__, "--mode", "layer", "--seconds", str(seconds)]
    child = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if child.returncode == 0:
        return f"auto picked {child.stdout.strip()}"
    lines = [ln for ln in child.stderr.splitlines() if ln.strip()]
    reason = lines[-1] if lines else f"exit code {child.returncode}"
    return f"auto picked {run_x11(seconds)} (layer failed: {reason})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("x11", "layer", "auto"), default="auto")
    parser.add_argument("--seconds", type=int, default=15)
    args = parser.parse_args()
    runner = {"x11": run_x11, "layer": run_layer, "auto": run_auto}[args.mode]
    try:
        print(runner(args.seconds))
    except SpikeError as e:
        print(e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
