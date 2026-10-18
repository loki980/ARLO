# ARLO Overlay Extension: Design

**Date:** 2026-10-11
**Status:** Draft, phase 1 in progress
**Scope:** Extend ARLO from a cursor tooltip popup into a stash-aware overlay that shows, on every item, what it sells for, what it recycles into at home, and whether a workshop upgrade, expedition project, or quest still needs it. Six phases. Phase 1 (tooltip augmentation) is specified in full here. Later phases are outlined and get their own detail sections when they start.

## Problem

ARLO today reads the hovered item's name by OCR and shows one popup in a fixed screen corner with an action string and a free-text "keep for" line. Three things limit it:

1. **Needs are unstructured.** The `items` table has six columns (name, action, recycle_for, keep_for, sell_price, stack_size). Quest, project, and workshop requirements live in `keep_for` as wiki prose such as "Projects Expedition 1 (30x)". Nothing can compute "how many do I still need" from that.
2. **Recycling is one string.** `recycle_for` is one free-text list. The game distinguishes recycling at the Speranza recycler from salvaging in the field, and the outputs differ for most items (the ARC Tracker data set shows 461 items where the two maps differ).
3. **The popup is far from the item.** The information sits in a corner while the player reads the game's own tooltip. The player wants the sell value beside the game's "Can be found in" block and the home-only recycle outputs beside the game's "Recycles into" icons.

A fourth goal, inventory awareness, needs per-cell recognition of the inventory grid, which ARLO does not attempt today.

## Findings that shape the design

From the codebase survey (2026-10-11):

- The scanner thread already captures a cursor-relative region (`OCREngine.capture_around_cursor`, `src/arc_helper/ocr.py`) and `preprocess_tooltip` already isolates the cream tooltip background by color mask. The tooltip's screen rectangle is therefore one step away: the mask's bounding box plus the capture origin.
- `OverlayWindow.set_position` (`src/arc_helper/overlay.py`) exists and is unused. Overlay windows are click-through Tk toplevels positioned through `phys_to_tk`, which divides physical pixels by the backend's scale factor (1.75 on the reference 4K at 175 percent setup).
- UI updates cross from the scanner thread to Tk through `ui_queue`, drained by `Application._pump_ui_queue` (`src/arc_helper/main.py`) every 50 ms. New overlay elements must use the same path.
- On Wayland the backend keeps the latest full frame in memory, so a full-screen grab is a numpy crop. On Windows and X11, mss captures per grab. Both are fine for a 4 Hz grid scan.
- The action color is chosen by the first keyword found in the action string, with SELL checked before KEEP, so "Keep until upgrade is complete; sell once done" renders gold. The verdict engine in phase 2 replaces this.
- No tests cover the overlay, the scan loop, or the updater. Phases add pure-function seams so the logic is testable without Tk or Tesseract.

From the ARC Tracker 2.3.2 teardown (ideas only, the app is all rights reserved):

- It finds slots with a YOLO detector and identifies icons with an embedding model against a reference gallery. Slot sizes are a fixed lattice that scales with resolution, so a lattice fit is enough for one known setup.
- It learns references from the hover tooltip: when the tooltip title is read, the slot crop under the cursor is stored under that item. ARLO already reads the title, so this is free.
- Per-cell badges show owned over needed, colored by state, with the sell value in the gap between icon and count. The hover card sits beside the game's tooltip, trying below, right, left, above.
- Inventory is a ledger built from one guided stash scroll, anchored on the stash header total, then kept current by deltas. Progress is read from the workshop, project, and quest screens.
- Scanning is gated: only on stash, inventory, container, or raid-end screens, paused while the grid moves, at most every 2 s when still.

## Decisions

- **Additive first.** Phase 1 adds two small overlay chips beside the game tooltip and leaves the existing corner popup as it is. The popup overhaul is phase 2, when the verdict engine exists to feed it.
- **Structured tables beside the existing one.** `items` keeps its six columns so the CSV workflow, manual overrides, and the weekly updater keep working. New tables `recycle_outputs` (phase 1) and `needs` (phase 2) are derived or fetched, never hand-edited.
- **Home versus field is a column, not a guess.** `recycle_outputs.location` is `home` or `field`. Phase 1 populates `home` rows from the existing `recycle_for` strings, which come from the MetaForge recycler table. `field` rows arrive when the data-source research names a source. Until then the recycle chip shows the home outputs under a "home recycler" label, and switches to "home only" extras automatically once field rows exist.
- **Pure logic in plain modules, Tk and Tesseract at the edges.** Geometry, label matching, placement, and the recycle diff are pure functions with unit tests. The OCR engine and the overlay classes are thin adapters.
- **Local only.** No account, no uploads, no analytics. The inventory ledger (phase 5) stays in the local SQLite file.
- **One known setup first.** The reference configuration is Linux, KDE Wayland, 3840x2160 at 175 percent, the game under XWayland. Windows must keep working, but new calibration is tuned on the reference setup and generalized afterwards.

## Architecture

### Data model

```
items             (unchanged) name, action, recycle_for, keep_for, sell_price, stack_size
recycle_outputs   item TEXT COLLATE NOCASE, component TEXT COLLATE NOCASE, quantity INTEGER,
                  location TEXT CHECK (location IN ('home', 'field')),
                  PRIMARY KEY (item, component, location)
needs             (phase 2) item, source_kind (quest | project | workshop | outpost), source_name,
                  tier INTEGER NULL, quantity INTEGER
progress          (phase 2, manual; phase 6, read from screen) source_kind, source_name, tier_done
inventory         (phase 5) item, count, seen_at, source (stash | loadout)
```

`recycle_outputs` is rebuilt from `recycle_for` whenever `items` is reloaded from CSV, for the `home` location only. `field` rows are left in place by that rebuild.

### Tooltip geometry (phase 1)

A new module `src/arc_helper/tooltip_layout.py` with pure functions:

- `find_cream_rect(rgb: np.ndarray) -> Rect | None`. Same color thresholds as `preprocess_tooltip` (RGB 240 to 255, 225 to 250, 210 to 240), rows over 50 percent cream, columns over 30 percent cream, tight columns over 50 percent. Returns the rectangle in capture coordinates or None.
- `find_label_rows(words: list[Word], scale: float) -> dict[str, Rect]`. Takes Tesseract word boxes (text, left, top, width, height, line number) from `pytesseract.image_to_data` run on the cream crop upscaled by `scale`, groups words by line, joins the line's text, and matches the two labels with a tolerant comparison (uppercase, alphanumerics only, Levenshtein-style ratio of at least 0.8 via `difflib`), so "RECYCLES INT0" still matches. Returns `{"found_in": Rect, "recycles_into": Rect}` with whatever was found, divided back by `scale`.
- `to_screen(rect: Rect, origin: Point) -> Rect`. Capture coordinates plus the capture origin give physical screen pixels.
- `place_chips(tooltip: Rect, labels: dict[str, Rect], chip_sizes: dict[str, Size], screen: Size, pad: int) -> dict[str, Point]`. The price chip is right-aligned inside the tooltip on the found-in label's row: `x = tooltip.right - pad - width`, `y = label.top`. The recycle chip sits on the icon row under the recycles-into label: `x = tooltip.left + round(0.45 * tooltip.width)`, `y = label.bottom + pad`. Both are clamped so the chip stays on screen. A label that was not found yields no entry for its chip.

`OCREngine` gains `layout_from_last_capture() -> TooltipLayout | None`, which uses the image and origin saved by the previous `capture_around_cursor` call, runs `find_cream_rect`, runs `image_to_data` with `--psm 6` on the cream crop at 2x, and returns the tooltip rectangle and label rows in physical screen pixels. It is called only after the scanner's cooldown check passed, so the extra OCR pass runs once per shown item, not every 0.3 s.

### Overlay chips (phase 1)

`src/arc_helper/overlay.py` gains `TooltipChips`, two click-through Tk toplevels in the existing palette (dark panel, alpha 0.92, small info font, accent bar on the left edge):

- **Price chip**: `SELLS FOR 250 ₡`, and when the stack size is known, a second dim line `×15 = 3,750 ₡`.
- **Recycle chip**: when home-only extras exist, `HOME ONLY  +2 Wires  +4 Metal Parts` with the "HOME ONLY" label in the accent color. When field rows are absent, `HOME RECYCLER  2 Metal Parts` in the neutral info color. When home rows are absent, the chip stays hidden.

`show(layout, price_text, stack_text, recycle_label, recycle_text)` measures each chip with `update_idletasks`, converts sizes to physical pixels through the backend's `tk_scale`, calls `place_chips`, and positions with `phys_to_tk`. `hide()` withdraws both. The chips auto-hide on the same timer as the popup and whenever the scanner leaves the ACTIVE state. A new setting `TOOLTIP_CHIPS` (default true) disables them.

### Scanner wiring (phase 1)

In `Scanner._handle_detected_item`, after the cooldown check and the database lookup, when chips are enabled and the item is known: call `ocr.layout_from_last_capture()`, compute the recycle lines from `db.home_only_outputs(name)` and `db.recycle_outputs(name)`, and put one message `("chips", layout, payload)` on `ui_queue`. `_pump_ui_queue` routes it to `TooltipChips.show`. When the scanner returns to IDLE it queues `("chips_hide",)`.

### Verdict engine (phase 2, outline)

`verdict(item, needs, progress, inventory) -> Verdict(kind, reason, keep_count)` with kinds keep, keep_some, sell, recycle, use. Reasons name the source: "Keep 3 for Gunsmith IV, then sell". The popup draws from the verdict instead of the action string, and the color comes from the kind. Manual overrides in `items.csv` still win when present.

### Grid recognition (phase 3, outline)

Full-frame grab while the INVENTORY trigger is active and the frame is still (mean absolute difference against the previous frame under a threshold). Lattice fit from horizontal and vertical edge projections, validated against a calibrated cell size per resolution profile. Each occupied cell is identified by perceptual hash against a gallery built from item icons, with hover-to-learn adding crops under the OCR-confirmed name. Stack counts by OCR of the count corner with a digit whitelist.

### Badge surface (phase 4, outline)

One full-screen click-through surface with per-pixel transparency. Tk cannot do this. A spike compares a GTK4 layer-shell window (Wayland native, needs the compositor's layer-shell support) against an XWayland window with an ARGB visual. The spike decides before any badge code is written.

### Inventory ledger (phase 5) and progress readers (phase 6)

As described in findings: guided stash scroll anchored on the header total, delta updates, unsure flags, then one screen reader at a time for workshop, projects, and quests, with manual progress entry available from phase 2 onward.

## Phases

| Phase | Deliverable | Depends on |
|---|---|---|
| 1 | Price chip and recycle chip beside the game tooltip, `recycle_outputs` table, tests for geometry, labels, placement, recycle diff | nothing |
| 2 | `needs` and `progress` tables, updater fills them, verdict engine, popup overhaul | data-source answer |
| 3 | Grid lattice fit, cell identification, stack counts, hover-to-learn | 1 |
| 4 | Badge surface spike, then per-cell badges | 2, 3 |
| 5 | Inventory ledger | 3 |
| 6 | Progress readers | 2, 5, screenshots of the project and workshop pages |

## Blueprint data (findings of 11 Oct, folded in 18 Oct)

Three public sources were checked: arctracker.io, MetaForge and the Arc Raiders Wiki.

- **Catalog.** arctracker `/api/items` has exactly 91 records with `type == "Blueprint"`, matching the game's FOUND: x/91. Each has a 256 px PNG icon (91 of 91 fetch with status 200). Match on the type, never on the name, since a few trinkets contain "blueprint". MetaForge also lists 91, but its ids have typos, so join on the English name.
- **Grid order.** The wiki's Blueprints page lists the grid in game order. The Collections grid has 17 columns (the first plan said 18, which does not fit 91 slots in a 6-cell last row). Read as 17 columns, the wiki order matched all 20 found cells of Jeffrey's screenshot by icon. Slots 26, 40 and 56 are unnamed on the wiki. They are Banjo, Stiletto and Tether Launcher in an unknown arrangement, to be settled on screen.
- **Where found.** arctracker and MetaForge item fields are empty for blueprints. The wiki table gives map, condition, container category, quest and trials for 84 of the 91. MetaForge has an undocumented player-vote endpoint (container, map and event counts) that covers 87. Its terms ask projects to attribute and to ask first, which is an open decision for Jeffrey.
- **Drop chances.** No source has them.
- **Not used.** arctracker's per-user blueprint endpoint needs an app key and a user key, and is player status, not catalog data. ARLO reads ownership from the Collections screen instead.

## Testing

- Unit tests for every pure function in `tooltip_layout.py` with synthetic arrays and fake word boxes, including: cream rect on a dark frame, no cream at all, a label with an OCR typo, a tooltip at the right screen edge (clamping), and a missing label.
- Unit tests for `recycle_outputs` population from `recycle_for` strings ("2x Metal Parts, 4x Wires", empty, malformed) and for the home-only diff (field subset, field equal, field absent, field quantity lower than home).
- No Tk tests. Overlay classes stay thin.
- An optional integration test, skipped when Tesseract is missing, runs the locator over a real screenshot fixture and asserts the two labels are found.
- Real-data check before calling phase 1 done: run ARLO on the reference setup, hover an item with a recycle section, and confirm the chips land on the found-in row and the icon row.

## Out of scope for phase 1

- Right-side tooltip flip (`capture_around_cursor` TODO). The cream rect search already tolerates any tooltip position inside the capture.
- The popup overhaul, the verdict engine, and any `needs` data.
- Field recycle rows. The table and the diff are ready, the source is pending.

## Open questions

- Which source provides field (salvage) outputs and the full workshop and project needs: MetaForge, the wiki, or manual curation. The data-source research answers this.
- Whether the game's own "Recycles into" row shows home or field outputs. The chip wording assumes the game shows the field result and the extras come from home recycling, matching the player's observation. If the game shows home outputs, the chip label flips to "field only" with the same data.
- Compositor support for layer-shell on the reference setup (KDE Plasma supports it) for phase 4.
