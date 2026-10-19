## Item Database

Items are stored in `items.db` (SQLite database) with the following columns: `name`, `action`, `recycle_for`, `keep_for`, `sell_price`, `stack_size`, `rarity` and `category`. The database also has tables loaded from the CSVs next to `items.csv`:

- `recycle_outputs` (`item`, `component`, `quantity`, `location`) is loaded from `recycle_outputs.csv`, home and field rows alike. When that file is missing, the home rows are derived from `items.recycle_for` instead.
- `needs` (`item`, `source_kind`, `source_name`, `tier`, `quantity`, `starts`, `ends`) is loaded from `needs.csv`.
- `blueprints` and `blueprint_grid` are loaded from `blueprints.csv` and `blueprint_grid.csv` (see Blueprints below).

The app loads these files on every launch, after the auto-update, and the updater loads them when it rebuilds `items.db`. A load writes nothing when the table already matches the file. The database updates automatically from the [MetaForge API](https://metaforge.app/arc-raiders) on every app launch, with the [Arc Raiders Wiki](https://arcraiders.wiki/wiki/Loot) as a fallback.

### Automatic Updates on Launch

Every time `ARLO.exe` (or `uv run arc-helper`) starts, it checks for item database updates. This is throttled to once per 24 hours so it doesn't slow down repeated launches. The update runs in merge mode, meaning your manual action overrides are always preserved.

The updater pulls data from these sources:

- **MetaForge items API** (`/api/arc-raiders/items`) - Item names, categories, sell prices, stack sizes, rarity
- **MetaForge quests API** (`/api/arc-raiders/quests`) - Which items are needed for which quests
- **arctracker.io items API** (`https://arctracker.io/api/items`, public, no key) - What each item gives when recycled at home (Speranza) and when salvaged in the field. Items are matched to MetaForge by English name, case-insensitive.
- **Arc Raiders Wiki** (Loot page) - Workshop, expedition and project uses, plus recycle outputs for items arctracker does not cover

The home outputs fill `recycle_for` in `items.csv` (for example `2x Battery, 2x Wires`). Both home and field outputs are written to `recycle_outputs.csv` next to `items.csv`, one row per output with the columns `item,component,quantity,location`, where `location` is `home` or `field`.

The old MetaForge Supabase recycle feed was dropped because its key was disabled on 2026-04-02. If arctracker.io is unreachable, the updater logs a warning, takes recycle data from the wiki, and leaves `recycle_outputs.csv` as it was. If MetaForge is unreachable, the updater falls back to scraping the wiki. If everything is down, the app starts normally with whatever database it already has.

The throttle timestamp is stored in `.last_wiki_update` in the app directory. Delete this file to force an update on the next launch.

### Manual Update from Command Line

You can also run the updater directly for more control:

```bash
# Full update from MetaForge API (default)
uv run python update_db.py

# Merge mode - keeps your manual overrides
uv run python update_db.py --merge

# Dry run - preview changes without writing any files
uv run python update_db.py --dry-run

# Force wiki scraper instead of API
uv run python update_db.py --source wiki

# CSV only - update the CSV without rebuilding the database
uv run python update_db.py --csv-only
```

Or use the makefile shortcuts:
```bash
make update-db          # Full update
make update-db-merge    # Merge mode
make update-db-dry      # Dry run preview
```

### Item Needs (needs.csv)

`needs.csv` sits next to `items.csv` and lists which quests, projects and workshop levels need which items, and how many. The updater regenerates it on every run, from `main()` and from the on-launch auto-update, and it is committed like `items.csv`. Each row is one item for one source:

| Column | Description |
|--------|-------------|
| `item` | English item name |
| `source_kind` | `quest`, `project` or `workshop` |
| `source_name` | English quest, project or station name. Expeditions are named `Expedition 1` to `Expedition 5` |
| `tier` | Workshop level or project phase number, empty for quests |
| `quantity` | How many of the item that source needs |
| `starts`, `ends` | ISO start and end dates for projects that have them, empty otherwise |

Rows are sorted by `source_kind`, `source_name`, `tier` and `item`. Where each kind comes from:

- **quest**: arctracker.io quests API (`https://arctracker.io/api/quests`), the `requiredItemIds` of each quest
- **project**: arctracker.io projects API (`https://arctracker.io/api/projects`), the `requirementItemIds` of each phase. Coin-value phases have no item rows.
- **workshop**: MetaForge hideout API (`/api/arc-raiders/hideout`), the only source with Outpost levels. Item ids are resolved to names through the MetaForge items list. The updater compares each station level with the arctracker.io hideout API (`https://arctracker.io/api/hideout`) and logs a warning where they disagree, without changing the rows.

Item ids that do not resolve to an English name are skipped and counted. If any of the three sources fails, the updater logs a warning and leaves the existing `needs.csv` untouched. The wiki is not used for `needs.csv`.

### Blueprints (blueprints.csv and blueprint_grid.csv)

Two more files sit next to `items.csv` and are regenerated by the updater on every run, from `main()` and from the on-launch auto-update. Both are committed like `items.csv`, and the app loads them into the `blueprints` and `blueprint_grid` tables of `items.db`.

`blueprints.csv` has one row per blueprint in the game's catalog (91 as of 2026-10-12), sorted by name. The catalog is every arctracker.io item of type `Blueprint`. The where-found columns come from the table on the wiki [Blueprints](https://arcraiders.wiki/wiki/Blueprints) page and are empty for blueprints the table does not list (Advanced Camera, Banjo, Bantam, Grappling Hook, Stiletto, Tether Launcher and Yank Grenade as of 2026-10-12).

| Column | Description |
|--------|-------------|
| `name` | arctracker.io English name, for example `Anvil Blueprint` |
| `item_id` | arctracker.io id, for example `anvil_blueprint` |
| `unlocks` | The item the blueprint unlocks: the name without ` Blueprint`, spelled as the arctracker.io item when one exists (`Trigger 'Nade`). Weapons keep the base name (`Anvil`), since only the tiers exist as items |
| `icon_url` | arctracker.io icon URL (256 px PNG) |
| `map`, `map_condition`, `scavengable`, `containers`, `quest_reward`, `trials_reward` | The wiki table cells as plain text, links reduced to their label and line breaks to commas. `?` means the wiki does not know yet |

`blueprint_grid.csv` has the columns `grid_index,name`, one row per slot of the in-game Collections grid, in game order. `grid_index` runs from 1, row by row, 17 slots per row, so slot 42 is row 3 column 8. `name` is the catalog name, or empty where the slot is unknown. The order comes from the `{{BlueprintGrid}}` template on the same wiki page, which matched all 20 found cells of a real Collections screenshot. Wiki names are matched to the catalog ignoring case and apostrophes and with ` Blueprint` added. A name that still does not match is logged as a warning and its slot left empty, never guessed.

As of 2026-10-12 three slots are unknown: 26, 40 and 56. Three catalog blueprints are not placed in the grid: Banjo Blueprint, Stiletto Blueprint and Tether Launcher Blueprint. Each unknown slot is one of those three, and the wiki does not say which. Editors are still placing them, so the positions can change on any update.

If arctracker.io or the wiki page is unreachable, or the page has no grid or table, the updater logs a warning and leaves both files untouched.

Which blueprints the player has found is stored in `arlo_user.db` (table `blueprint_owned`, the name and an ISO timestamp of when it was recorded), never in `items.db`.

Blueprint icons are downloaded from arctracker.io on first use and cached under `cache/icons/blueprints/` (the `ICON_CACHE_DIR` setting). The cache is ignored by git and icons are never committed.

### How Verdicts Are Formed

The popup shows a verdict (KEEP, KEEP SOME, SELL, RECYCLE, USE or UNKNOWN) with a reason under it, and the verdict picks the color. The rules apply in order:

1. **Weapons** (category `Weapon`, or a tiered name such as Anvil IV whose base has a `<base> Blueprint` entry) are always kept: "Weapon, keep at least one", plus "more until you own the Anvil Blueprint" when the catalog has a `<base> Blueprint` entry.
2. **Active needs** from `needs.csv` decide next. A quest counts until `progress.csv` marks it done, a project phase until that phase is done or the project has ended (see `SHOW_ENDED_PROJECTS` in [CONFIGURATION.md](CONFIGURATION.md)), and a workshop level while it is above the level recorded for that station. The verdict is KEEP with the summed quantity, naming up to three sources by quantity, then what to do with the rest: "3 for Gunsmith IV, then sell". The rest is recycled when the item has home recycle outputs and sells for under 2,000, and sold otherwise.
3. **No active needs**: the item's `action` string decides, by the first of sell, recycle, keep or use that appears in it. "Keep until upgrade is complete; sell once done" is KEEP.
4. **Unknown items** show UNKNOWN.

`progress.csv` is loaded into a `progress` table (`source_kind`, `source_name`, `value`) in `arlo_user.db` next to `items.db`, never into `items.db` itself. KEEP SOME ("You have 5, need 2, recycle the rest") comes from the owned count: the saved stash map plus the loadout, with stash badges on. While the map is partial the count is a lower bound and shows as "N+".

### How Action Generation Works

The updater auto-generates action recommendations based on item category and uses:

- **Basic/Refined/Topside Materials** - `Keep` (crafting ingredients)
- **Trinkets** with no uses - `Sell`
- **Recyclables** with workshop/quest uses - `Keep until uses complete; recycle after`
- **Recyclables** with no uses - `Recycle`
- **Keys** - `Keep`
- **Quick Use / Mods / Augments / Shields** - `Keep` or `Use`
- Items needed for quests get a "Quests:" note in the keep_for field

### Merge Mode

The on-launch auto-update always uses merge mode. When you run the updater manually, you can choose between full update or merge mode.

In merge mode the updater will:
- **Preserve** your existing `action`, `recycle_for`, and `keep_for` for items already in your CSV
- **Add** any new items with auto-generated recommendations
- **Fill in** empty `recycle_for` / `keep_for` fields from API data
- **Always update** `sell_price`, `stack_size`, `rarity` and `category` to the latest values

### GitHub Actions (Optional)

If you fork this repo, the included GitHub Actions workflow at `.github/workflows/update-db.yml` can sync the database on a schedule:

- Runs weekly on Mondays at 6:00 UTC in merge mode
- Auto-commits updated `items.csv`, `items.db`, `recycle_outputs.csv` and `needs.csv` if changes are detected
- Can be triggered manually from the **Actions** tab on GitHub

No secrets or configuration needed - just push the workflow file and it works.

---

### Manual CSV Management

You can also edit `items.csv` directly. The CSV has the following columns:

```csv
name,action,recycle_for,keep_for,sell_price,stack_size,rarity,category
Advanced Arc Powercell,Keep,2x ARC Powercell,Medical Lab: Surge Shield Recharger,640,5,Rare,Topside Material
ARC Alloy,Keep,2x Metal Parts,Workshop Explosives Station 1 (6x),200,15,Uncommon,Topside Material
Air Freshener,Sell,,,2000,5,Uncommon,Trinket
Alarm Clock,Recycle,"1x Processor, 6x Plastic Parts",,1000,3,Rare,Recyclable
```

| Column | Description |
|--------|-------------|
| `name` | Item name as it appears in-game (case-insensitive matching) |
| `action` | Recommended action: `KEEP`, `RECYCLE`, `SELL`, `USE`, `TRASH`, or any custom text |
| `recycle_for` | What you get when recycling (shown when action is RECYCLE) |
| `keep_for` | Why to keep it (shown when action is KEEP or USE) |
| `sell_price` | Sell value in credits (shown on overlay) |
| `stack_size` | Maximum stack size (shown on overlay) |
| `rarity` | Item rarity from MetaForge, for example `Common` or `Rare` |
| `category` | Item category from MetaForge, for example `Recyclable` or `Trinket` |

The `sell_price`, `stack_size`, `rarity` and `category` columns are optional for backwards compatibility. If missing, the overlay just won't show that info for those items.

### Managing the Database via Calibration Tool

1. Edit `items.csv` in Excel, Google Sheets, or any text editor
2. Run `Calibrate.exe` (or `uv run arc-calibrate`)
3. In the "Item Database" section at the bottom:
   - Click **"Load CSV..."** to import your updated CSV file
   - Click **"View Items"** to see all items currently in the database
   - Click **"Clear Database"** to remove all items and start fresh

The database is automatically created on first run. Loading a CSV file will replace all existing items.

### Tips for the CSV File

- You can use Excel or Google Sheets to edit the CSV - just make sure to save as CSV format
- Item names are matched case-insensitively (e.g., "Arc Circuitry" will match "ARC CIRCUITRY")
- Leave `recycle_for` or `keep_for` empty if not applicable
- The `action` field can be any text - it will be displayed as-is on the overlay
- If you customize an action and want to keep it, use `--merge` mode (or just let the on-launch auto-update handle it - it always uses merge mode)
