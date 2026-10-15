
## Configuration

All configuration is stored in the `.env` file in the application directory. You can edit this file with any text editor.

### Configuration Options

```env
# =============================================================================
# TRIGGER REGIONS - Where "INVENTORY" text appears
# =============================================================================

# Region 1 - Menu mode
TRIGGER_REGION_X=1450
TRIGGER_REGION_Y=23
TRIGGER_REGION_WIDTH=173
TRIGGER_REGION_HEIGHT=44

# Region 2 - In-raid mode
TRIGGER_REGION2_X=1295
TRIGGER_REGION2_Y=38
TRIGGER_REGION2_WIDTH=173
TRIGGER_REGION2_HEIGHT=44

# =============================================================================
# TOOLTIP SEARCH - Window around the cursor searched for the item card
# =============================================================================

TOOLTIP_SEARCH_WIDTH=0           # Width of the window, centred on the cursor (physical px, 0 = full screen)
TOOLTIP_SEARCH_HEIGHT=0          # Height of the window, centred on the cursor (physical px, 0 = full screen)

# =============================================================================
# TOOLTIP CAPTURE - Fallback box captured relative to the cursor
# =============================================================================

TOOLTIP_CAPTURE_WIDTH=550
TOOLTIP_CAPTURE_HEIGHT=550
TOOLTIP_CAPTURE_OFFSET_X=50      # Horizontal offset from cursor
TOOLTIP_CAPTURE_OFFSET_Y=-500    # Vertical offset from cursor (negative = above)

# =============================================================================
# OVERLAY SETTINGS - How the recommendation overlay appears
# =============================================================================

OVERLAY_X=100                    # Overlay X position on screen
OVERLAY_Y=100                    # Overlay Y position on screen
OVERLAY_DISPLAY_TIME=4.0         # How long the overlay stays visible (seconds)
OVERLAY_COOLDOWN=2.0             # Minimum time between showing same item (seconds)
TOOLTIP_CHIPS=true               # Show sell price and home recycle chips beside the game tooltip
SHOW_ENDED_PROJECTS=false        # Count needs from projects whose end date has passed

# =============================================================================
# SCANNING SETTINGS - How often the app scans the screen
# =============================================================================

TRIGGER_SCAN_INTERVAL=0.5        # Seconds between trigger scans
TOOLTIP_SCAN_INTERVAL=0.3        # Seconds between tooltip scans

# =============================================================================
# CARD TITLE READER
# =============================================================================

OCR_ENGINE=paddle                # paddle, tesseract_title or tesseract

# =============================================================================
# DEBUG SETTINGS
# =============================================================================

DEBUG_MODE=false                 # Enable to save debug images and logs
SHOW_CAPTURE_AREA=false          # Show red rectangle around capture area
CAPTURE_LIBRARY_DIR=debug/captures  # Folder for the capture library
```

`CAPTURE_LIBRARY` is no longer read. The capture library saves raw tooltip captures for offline OCR evaluation only for a run started with `uv run arc-helper --capture` (at most 300 images, or N with `--capture N`). A leftover `CAPTURE_LIBRARY` line in `.env` is ignored with a warning in the log. `CAPTURE_LIBRARY_DIR` still sets the folder.

### Progress File (progress.csv)

The popup advice counts what quests, projects and workshop levels still need an item. To stop counting what you have already finished, copy `progress.example.csv` to `progress.csv` next to `items.csv` and edit it. `progress.csv` is not tracked, and the app loads it on every launch into `arlo_user.db`, a second SQLite file next to `items.db` that contains your own state (also not tracked). `items.db` is never written for your progress, so the weekly item database refresh never conflicts with it. `arlo_user.db` is created on the first load. Delete it to forget the loaded progress, and the next launch rebuilds it from `progress.csv`.

```csv
source_kind,source_name,value
workshop,Gunsmith,3
quest,After Rain Comes,1
project,Expedition 5,2
```

| Column | Description |
|--------|-------------|
| `source_kind` | `workshop`, `quest` or `project` |
| `source_name` | The station, quest or project name as it appears in `needs.csv` (case does not matter) |
| `value` | For `workshop`, the level the station has reached: needs for that level and below stop counting. For `quest`, `1` when the quest is done. For `project`, the highest phase done: that phase and earlier ones stop counting |

A station without a row counts as level 0, so every level still counts. Rows with an unknown `source_kind`, an empty name or a value that is not a whole number are skipped. The file is CSV only, so it cannot contain comments.

Projects that have an end date before today (Expedition 1 to 5, for example) are left out of the advice by default. Set `SHOW_ENDED_PROJECTS=true` in `.env` to count them again.

### Finding the item card

On every scan ARLO captures a window of `TOOLTIP_SEARCH_WIDTH` by `TOOLTIP_SEARCH_HEIGHT` physical pixels centred on the cursor (clamped to the screen) and looks in it for the game's item card, the solid cream rectangle. When several cream areas qualify, such as the card and the stash's info panel, the one nearest the cursor wins. Only the card is read, so a card drawn left of the cursor, below it, or further away is still found, and the capture offsets no longer need tuning for each screen.

Make the window larger if a card is drawn so far from the cursor that it falls outside, and smaller to cut capture cost. On Wayland the window is a crop of a frame that is already captured, so its size costs little. On Windows and X11 the window is grabbed from the screen on every scan, so keep it bounded.

When no card is found in the window, ARLO falls back to the `TOOLTIP_CAPTURE_*` box, captured at the given offset from the cursor exactly as before.

### Card title reader (OCR_ENGINE)

`OCR_ENGINE` picks how the item name is read from the located card:

| Value | What it does |
|-------|--------------|
| `paddle` (default) | PP-OCRv4 through the optional `rapidocr-onnxruntime` package. One pass reads every line of the card, so the price and recycle chips find their rows without a second OCR pass |
| `tesseract_title` | Finds the title line by the card's layout and reads only that line with Tesseract. No extra package |
| `tesseract` | The earlier reader: two Tesseract passes over the whole card |

Install the paddle reader with `uv sync --extra paddle` (`uv sync --all-extras` includes it). When the package is missing, ARLO logs one warning with that command and uses `tesseract_title`. An unknown value falls back the same way. The `INVENTORY` trigger check always uses Tesseract. With `paddle` or `tesseract_title` a scan that finds no card reads nothing, with `tesseract` it still reads the `TOOLTIP_CAPTURE_*` box. Debug mode logs the reader name and the read time of every scan.

### Debug Mode

Enable debug mode to troubleshoot OCR issues:

1. Set `DEBUG_MODE=true` in `.env`
2. Run the application
3. Check the `debug/` folder for:
   - `trigger_raw.png` / `trigger_processed.png` - Trigger region captures
   - `tooltip_capture_raw.png` / `tooltip_capture_processed.png` - The located card crop (or the fallback box) and its processed form
   - `tooltip_mask.png`, `tooltip_cropped.png`, etc. - OCR preprocessing steps
   - `arc_helper.log` - Detailed log file with all operations
