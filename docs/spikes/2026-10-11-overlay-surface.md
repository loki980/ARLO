# Spike: one transparent, click-through surface over the game

Roadmap item 7. Date: 2026-10-11.

Question: can ARLO draw one full-screen, per-pixel transparent, click-through,
topmost surface over Arc Raiders on KDE Plasma Wayland, with the game running
under XWayland (Proton)? Phase 4 (badges on every stash cell) depends on the
answer. Today's popup is a Tk window, which only has whole-window alpha.

Script: `scripts/spike_overlay_surface.py`. It draws three marks on an
otherwise fully transparent surface:

- a 4 px amber rectangle outline, a quarter of the screen in size, exactly centred
- a dark badge with the amber text `3/8`, its top-left corner at one eighth of
  the screen width and height
- a small white crosshair at the pointer (x11 mode) or at the screen centre
  (layer mode)

After the given number of seconds it prints one line: the mode used, the
surface size in logical units and in pixels, the scale factor seen, whether the
input pass-through was applied and whether the crosshair follows the pointer.
Any missing dependency or unsupported session prints one plain sentence and
exits 2.

## What was tried

**Mode x11.** A GTK3 `POPUP` window (override-redirect) on the XWayland
server, with the screen's RGBA visual, app-paintable, no focus, keep-above,
sized to the X root window. After it is mapped, the SHAPE input region is
emptied with python-xlib on the window and each parent below root, the same
way `src/arc_helper/clickthrough.py` does it for the Tk popup. It re-raises
itself once a second and polls the pointer every 16 ms.

**Mode layer.** A GTK4 window turned into a wlr-layer-shell surface through
gtk4-layer-shell: overlay layer, anchored to all four edges, exclusive zone -1
(covers panels, reserves no space), keyboard interactivity none, transparent
CSS background, and an empty input region set on the `GdkSurface` before
every draw. gtk4-layer-shell has to load before libwayland-client, so the
script restarts itself once with `LD_PRELOAD` set to the library. A Wayland
client cannot read the global pointer position, and with an empty input region
the surface gets no motion events, so the crosshair is static in this mode.

**Mode auto.** Runs layer mode in a child process. If that exits non-zero it
runs x11 mode and appends the layer failure sentence to the report line.

## What this machine has

`pacman -Q gtk4 gtk4-layer-shell python-gobject qt6-base pyside6`:

```
error: package 'gtk4-layer-shell' was not found
gtk4 1:4.22.5-1.1
python-gobject 3.56.3-1
qt6-base 6.11.2-3.1
pyside6 6.11.2-2.1
```

`pacman -Q gtk3 python-cairo python-xlib`:

```
gtk3 1:3.24.52-1.1
python-cairo 1.29.1-1.1
python-xlib 0.33-7
```

`uv pip list` in the project venv (after `uv sync --all-extras`), filtered:

```
pycairo                   1.29.0
pygobject                 3.56.3
python-xlib               0.33
```

Session: KWin and plasmashell 6.7.5, `XDG_SESSION_TYPE=wayland`,
`DISPLAY=:0`, `WAYLAND_DISPLAY=wayland-0`, `xrdb -query` reports
`Xft.dpi: 96`.

## What ran here

The worker shell had the live Plasma Wayland session, not a headless or nested
one, so both modes ran against the real desktop (without the game open).

| Mode  | Result |
|-------|--------|
| x11   | Ran, exit 0: `mode=x11 surface=2194x1234 logical, 2194x1234 px scale=1 (xft 1.00) passthrough=yes pointer=live` |
| layer | Did not run, exit 2: `gtk4-layer-shell is not installed (pacman -S gtk4-layer-shell), so layer mode cannot run.` |
| auto  | Fell back to x11, exit 0, with the layer sentence appended |

While x11 mode was running, a python-xlib check of the root window's children
found the spike window as a depth 32 (ARGB) window, override-redirect, 2194 by
1234, with zero input shape rectangles, and last in the stacking order (on
top). Nobody looked at the screen or clicked during these runs, so whether the
marks render correctly and whether clicks reach the window underneath is
still unverified.

The failure paths were checked too: `DISPLAY` unset, `DISPLAY=:99` (no
server) and `WAYLAND_DISPLAY` unset each print one sentence and exit 2.

A plain GTK4 window on the same Wayland session (no layer shell) reported a
surface scale of 1.75, so a layer surface should see 2194x1234 logical and
3840x2160 pixels, with true per-pixel rendering at physical resolution.

### Scale finding

`Xft.dpi: 96` and an X root of 2194x1234 mean KWin runs XWayland apps in
"scaled by the system" mode. Every X client, the game included, sees a
2194x1234 screen and KWin upscales its buffer by 1.75. For the x11 overlay
this means:

- coordinates are X pixels on a 2194x1234 grid. A point at physical pixel
  (x, y) is at (x / 1.75, y / 1.75) in the overlay
- the 4 px outline shows as roughly 7 physical pixels and may look slightly
  soft, because KWin scales the whole buffer

The layer surface would draw at 3840x2160 directly, so it would be crisp, but
the integration would then map from game coordinates to physical pixels.

## What Jeffrey runs with the game open

Run the game in borderless or fullscreen as usual. From a terminal in the
ARLO checkout (fish), each command waits 5 seconds so there is time to switch
to the game:

```
uv sync --all-extras
sleep 5; uv run python scripts/spike_overlay_surface.py --mode x11 --seconds 15
```

For layer mode, gtk4-layer-shell has to be installed first (your call, the
spike does not install anything):

```
sudo pacman -S gtk4-layer-shell
sleep 5; uv run python scripts/spike_overlay_surface.py --mode layer --seconds 15
sleep 5; uv run python scripts/spike_overlay_surface.py --mode auto --seconds 15
```

During each 15 second run, move the mouse, click inside the game (open and
close the inventory for example) and press a key. Paste the printed line back.

## What to report for each mode

1. Drawn over the game: yes or no. All three marks visible on top of the game
   with the game visible everywhere else (no black or grey background).
2. Clicks pass through: yes or no. Clicks and key presses reach the game while
   the marks are on screen, and the game keeps focus.
3. Scale correct: yes or no. The amber rectangle sits exactly in the middle of
   the screen with equal margins, the badge sits one eighth in from the
   top-left corner, and in x11 mode the crosshair centre stays on the pointer
   tip. Also say whether the lines look crisp or soft.
4. Any flicker or lag: the marks blinking, the game stuttering, or the
   crosshair trailing behind the pointer.

## Decision

Pending Jeffrey's run.
