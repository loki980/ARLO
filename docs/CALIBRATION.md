## Calibration Tool

The Calibration tool (`Calibrate.exe` or `uv run arc-calibrate`) provides a GUI to configure the screen regions for your specific setup.


TODO: add screenshots

### When to Calibrate

- First time using the app with an unsupported resolution
- After changing your screen resolution
- After changing Arc Raiders display settings
- If item detection isn't working correctly

### How to Calibrate

1. **Launch Arc Raiders** and open your inventory (in the menu)
2. **Run the Calibration tool**:
   - Pre-built: Run `Calibrate.exe`
   - From source: Run `uv run arc-calibrate`

3. **Configure Trigger Region 1 (Menu Mode)**:
   - Click "Show Region" to see the yellow rectangle on screen
   - Adjust the sliders until it covers the "INVENTORY" text
   - Click "Test OCR" - should show "✓ INVENTORY detected!"
   - Click "Hide"

4. **Configure Trigger Region 2 (In-Raid Mode)**:
   - Open your inventory while in a raid
   - Repeat the same process with the blue rectangle
   - The "INVENTORY" text may be in a different position in-raid

5. **Configure Tooltip Region**:
   - Hover over an item in your inventory
   - Adjust the green rectangle to cover the item name area
   - It is important to make sure that the rectangle covers the whole tooltip area regardless of where the cursor is located and regardless of how long the tooltip is. TODO: ADD SCREENSHOTS
   - Click "Test OCR" to verify item names are detected

6. **Click "Save Configuration"** to save to `.env` file

### Tooltip capture is now a fallback

ARLO now finds the item card automatically: each scan searches a window centred on the cursor (`TOOLTIP_SEARCH_WIDTH` by `TOOLTIP_SEARCH_HEIGHT`, see [Configuration](CONFIGURATION.md)) for the cream card nearest the cursor and reads only that card, wherever the game drew it. The tooltip capture box set in step 5 (`TOOLTIP_CAPTURE_*`) is used only when no card is found in that window, so a rough setting is enough.

### Database Management

The Calibration tool also includes database management features:

- **Load CSV...** - Import items from a CSV file (replaces existing items)
- **View Items** - See all items currently in the database
- **Clear Database** - Remove all items from the database

This makes it easy to update your item database without running any commands.
