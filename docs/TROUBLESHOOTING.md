## Troubleshooting

Here are some common errors you may experience and how to deal with them.

### "INVENTORY not detected"

- Make sure Arc Raiders is running in **borderless windowed** or **windowed** mode (not exclusive fullscreen)
- Run the Calibration tool to verify the trigger region covers the "INVENTORY" text exactly
- Enable debug mode and check `debug/trigger_processed.png` to see what the OCR is seeing

### "Item not found in database"

- Check that the item name in your CSV matches exactly how it appears in-game (case-sensitive, typically ALL CAPS)
- Enable debug mode and check the logs to see what item name was extracted
- The OCR might be misreading characters - check `debug/tooltip_capture_processed.png`

### "OCR is reading incorrect text"

- The tooltip must have a cream/beige background for proper detection
- Colored tags (rarity, item type) should be automatically filtered out
- Run the Calibration tool and use "Test OCR" on the tooltip region
- Enable debug mode and examine the preprocessing images in `debug/`

### Collecting captures for tier problems

When a weapon shows the wrong tier (I, II, III, IV, V), collect captures so the tier reading can be measured and tuned:

1. Start ARLO with `uv run arc-helper --capture`. Captures are saved for that run only, at most 300 images, or N with `--capture N`. The folder is `CAPTURE_LIBRARY_DIR` in `.env`, default `debug/captures`
2. Hover the tiered items that come out wrong. Each scan saves the raw capture as `<timestamp>_<read name>.png`, at most one per name every 2 seconds
3. Optionally add a `labels.csv` to the folder with the columns `file,expected` (the catalog name, for example `Anvil II`)
4. Run `uv run python scripts/eval_captures.py debug/captures` to see the read text, the stroke-count vote and the resolved item for every capture, plus accuracy when labels are present. It reads each file the way the scanner does: it locates the item card with the cursor assumed at the image centre, prints its rectangle in the `card` column (a dash when there is none) and reads the name from the card crop with the reader `OCR_ENGINE` names (with `tesseract` it reads the whole image when no card is found). Pass a labels file after the folder (`scripts/eval_captures.py debug/captures labels.csv`) when it is kept elsewhere. The last lines name the reader actually used and the median read and layout times per card. Prefix `OCR_ENGINE=tesseract_title` or another value to compare readers
5. Restart ARLO without `--capture` when done. A normal start never saves captures

### "OCR_ENGINE=paddle needs the rapidocr-onnxruntime package"

The default title reader needs an optional package. Install it from the ARLO folder with `uv sync --extra paddle` (or `uv sync --all-extras`), then restart ARLO. Until then ARLO reads titles with `tesseract_title` and logs this warning once per start. To stop the warning without installing, set `OCR_ENGINE=tesseract_title` in `.env`.

### "Application won't start"

- Check that the `.env` file exists (copy from `.env.example` if missing)
- Make sure the `.env` file has valid values (no syntax errors)
- Check the console window for error messages
- If debug mode was enabled, check `arc_helper.log` for details

### "High CPU usage"

Increase the scan intervals in `.env`:
```env
TRIGGER_SCAN_INTERVAL=1.0
TOOLTIP_SCAN_INTERVAL=0.5
```

### "Tesseract not found" (when running from source)

1. Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
2. Either add to system PATH, or set in `.env`:
   ```env
   TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
   ```
3. Verify installation: `tesseract --version`
