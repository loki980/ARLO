# ARLO roadmap

Source of truth for the overlay extension. The published page at
https://claude.ai/artifact/6mrLLrRXHBxS4zRzMv2vwd shows the same roadmap as a
page to read. Its source is `docs/roadmap.html`. Update this file first, then
republish the page to the same URL.

Design detail is in `docs/superpowers/specs/2026-10-11-overlay-extension-design.md`.
Markers such as "done 11 Oct" or "added 11 Oct" mark changes in the last 14
days and are dropped after that. Completed steps keep their date.

## Goal, in Jeffrey's words (11 Oct 2026)

"I'd like an overlay that highlights items that I need to upgrade my bench, for
projects, etc. Some attached, some quests." "First and foremost, I'd like item
info (how much it sells for, what it recycles to back home) superimposed on
every item in the game." "Next to 'can be found in' on the right hand side,
show 'sells for'. Next to 'recycles into' icons, add the additional items the
item recycles into. It should have a highlight or something correlating to the
tooltip indicating these are additional if recycled back in Speranza, not in
the field." "The existing popup is great, but it really needs an overhaul."
"This needs to be aware of my own inventory somehow."

## Phase 0. Look before building. Done 11 Oct

Decided 11 Oct. Code survey, ARC Tracker teardown (ideas only, the app is all
rights reserved), data-source study with evidence, design written.

1. Code survey. Done 11 Oct.
2. ARC Tracker 2.3.2 teardown. Done 11 Oct.
3. Data sources checked. Found the MetaForge recycle feed dead since 2 Apr
   2026 and the quest reader reading a field that does not exist. Chose the
   arctracker.io public API for items, quests and projects, MetaForge for
   workshop levels (the only source with Outpost). Done 11 Oct.
4. Design written. Done 11 Oct.

## Phase 1. Chips on the tooltip. Underway from 11 Oct, about 76 percent

Jeffrey's words: "Next to 'can be found in' on the right hand side, show
'sells for'. Next to 'recycles into' icons, add the additional items the item
recycles into." Decided 11 Oct.

Done:

1. The two chips and the geometry that places them. Price anchored to the
   label when the card is cut off by the capture. Done 11 Oct.
2. Home and field recycling kept apart in the database. Done 11 Oct.
3. Weekly data feed repaired: dead feed removed, arctracker.io as the recycle
   source, quest reader fixed, the weekly job installs the project. Done
   11 Oct.
4. Roman numerals resolved from the catalog and from the picture. Jeffrey's
   words: "I have a suspicion that the roman numerals identification is
   better, but not fixed. This should be tested and maybe improved." A
   capture library, a catalog-aware resolver, a stroke-count vote. Rarity
   checked as a signal and rejected (Anvil I to IV are all Rare). Done
   11 Oct.
5. New data loaded into the database, catalog refreshed to 1,109 items, the
   database stable on open. Done 11 Oct.
6. First capture session measured: 125 captures, 7 with a card and all 7
   read right (Anvil IV, Stiletto IV, vote IV at 0.88 on the real font). Done
   11 Oct.

7. Find the card anywhere on screen. 99 of the 125 captures held no card
   because the fixed cursor-relative box missed where the game drew it.
   Replaced the box with a search window and OpenCV connected components on
   the cream mask, nearest solid rectangle to the cursor, old box as the
   fallback. Retires the right-side flip TODO. OpenCV is now a dependency.
   Added 11 Oct, done 11 Oct (review by the fallback reviewer pending).

Ahead, in order:

8. Jeffrey's live check of the chips: where they land, and whether the game's
   own "Recycles into" row shows Battery only or Battery and Wires for a
   Power Bank.
9. A capture run for tiers I to III so the stroke thresholds are tuned on
   every numeral.
10. Decide the recycle chip label, "home only" or "field only", from step 8.
11. Replace the card title reader on evidence. Brought forward 12 Oct after
    Jeffrey's second live test: "Burletta IV, Anvil I ... both don't get
    detected. Are you sure an overhaul on the image recognition isn't
    warranted here?" The Rattler III capture settled it: the card was found,
    the plain read dropped the numeral and the uppercase read fused III into
    H, and 49 of the last 60 captures read as nothing. Steps: label the
    capture library by eye (started 12 Oct), then a bake-off of three
    readers (today's Tesseract, Tesseract with the title line isolated by
    geometry, PaddleOCR under onnxruntime) on the labeled set, the winner
    behind an OCR_ENGINE setting, and the catalog matcher tolerant of
    partial reads. Changed 12 Oct.
12. Flicker, second cause, 12 Oct: "it's still flickering, but the frequency
    has increased." Raw reads of one item vary between scans and each
    variant counted as a new item, re-showing both windows at scan speed.
    Fix: key everything on the resolved catalog name, never re-map a window
    for the same item. Added 12 Oct, done 12 Oct, reviewed. Jeffrey, 12 Oct
    after testing: "the flickering and the fading in and out of the pop-ups
    are still there. For tomorrow." Still open. Leading suspect: a scan now
    takes about a second (full-screen card search plus two OCR passes plus
    the 0.3 s sleep), while the popup hides after OVERLAY_DISPLAY_TIME,
    which is 1.0 s in the example settings. The keep-alive touch only
    arrives once per scan, so the timer can expire between scans. Other
    suspects to rule out with timestamps: the INVENTORY trigger recheck
    every third scan dropping the scanner to IDLE (which hides the chips),
    scans that read nothing letting the timer lapse, and KWin's fade
    animation on map and unmap. First step tomorrow: log each show, touch,
    hide and timer expiry with a timestamp during one hover, then fix the
    cause the log shows.
13. Search the whole screen for the card. The labeled capture library (320
    captures, 113 with a card) showed 56 cards missing their title row
    because the cursor-centred window cut tall weapon cards at the top. The
    search window now defaults to the full screen. Added 12 Oct, done 12 Oct,
    reviewed. Confirmed by Jeffrey in game 12 Oct: "It seems like all the
    weapon identification is working now."

## Phase 2. Advice with a reason. Underway from 11 Oct, about 89 percent

Jeffrey's words: "Keep until upgrade is complete; sell once done" is the kind
of advice he wants, with the source named. And, after the popup told him to
recycle a fully upgraded Anvil IV: "Fully upgraded anything is great. 1 of any
kind of weapon shouldn't be recycled, especially if you don't have the
blueprint." Decided 11 Oct.

Done:

1. The needs list: `needs.csv`, 289 rows (80 quest, 128 project, 81
   workshop) from arctracker quests and projects plus MetaForge workshop
   levels, all nine Frozen Trail quests and the Outpost levels. Loaded into
   the database. Done 11 Oct.

2. The verdict engine. Rule 0: weapons are keep, at least one, more until the
   blueprint is owned. Then active needs minus progress, a reason naming up
   to three sources, the fallback (sell or recycle) from the item's data.
   Radial Press reads "keep, 3 for Gunsmith IV, then sell". Added 11 Oct
   (weapon rule), done 11 Oct, accepted by the fallback reviewer.
3. Popup overhaul: the verdict word colored by the verdict (fixes the
   keyword-order bug that colored "Keep until upgrade is complete; sell once
   done" gold), the catalog name instead of the raw read (the popup showed
   ANVILIV), home recycle outputs on the notes line. Changed 11 Oct, done
   11 Oct.
4. Manual progress file (`progress.csv`, not committed) for bench levels,
   finished quests and project phases, loaded into `arlo_user.db`, a second
   untracked database so the weekly refresh never conflicts with it. Done
   11 Oct.

Ahead:

5. Jeffrey's decisions: ended projects (Expedition 1 to 4) hidden by default
   until he says otherwise, coin-funding phases produce no needs.

## Phase 3. Read the grid. Next, after phase 1

Jeffrey's words: "OpenCV sounds like the frontier option for this work."
Answer recorded 11 Oct: OpenCV for geometry and gating, an embedding model
for icon identity, OCR for text. Changed 11 Oct.

Will contain:

1. A perceptual-hash baseline, measured first, expected to fail on lookalikes.
2. Slot lattice from edges and contours, frame stillness check, only while
   the INVENTORY trigger is active.
3. An embedding model (DINOv2-small under onnxruntime, Apache-2.0) over a
   gallery built from the stable arctracker icon PNGs plus crops learned from
   hovering (ARLO already reads the tooltip).
4. Tier numerals on icons by glyph matching (the loadout shows a IV glyph on
   the weapon card), stack counts by OCR.

## Phase 4. Badges on every cell. Next, after phases 2 and 3

Decided 11 Oct. Carries the Wayland risk.

1. Overlay layer experiment: `scripts/spike_overlay_surface.py` with
   findings in `docs/spikes/2026-10-11-overlay-surface.md`. The X11 mode
   ran on the live Plasma session (depth 32, on top, empty input shape). The
   layer-shell mode needs `gtk4-layer-shell`, not installed. Done 11 Oct.
2. Jeffrey's run with the game open decides the mode.

Will contain: badge drawing from verdicts and the grid, hidden while the
grid moves.

## Phase 4b. Blueprint tracker. Next, after phases 3 and 4. Added 11 Oct

Jeffrey's words, 11 Oct, after the weapon rule said "more until you own the
blueprint": "We should build in some kind of blueprint tracker assist, too.
The blank boxes should be greyed out images of the actual blueprint. On hover
it shows the name and best places to find it based on public data." His
Collections screen shows FOUND: 20/91 on an 18 by 6 grid. Decided 11 Oct.

1. Data: the 91 blueprints, the order of the Collections grid, and where each
   drops, from public sources. Done 11 Oct. arctracker lists exactly 91
   items of type Blueprint with 256 px icons. The wiki's Blueprints page
   has the grid order (17 columns, 88 names and 3 unnamed slots: Banjo,
   Stiletto, Tether Launcher in an unknown order), validated by icon against
   all 20 of Jeffrey's found cells. Where found: the wiki table (map,
   container category, quest, trials) for 84, and a MetaForge vote endpoint
   (player votes by container, map and event) for 87. No source has drop
   chances. The 3 unnamed slots get resolved from the screen by elimination.
   Findings in the handoff folder, to be folded into the design spec when
   step 2 starts.
   Data pipeline done 12 Oct: the updater writes blueprints.csv and
   blueprint_grid.csv (91 each, slots 26, 40, 56 unknown), the database
   loads them, ownership lives in arlo_user.db, icons are cached on demand.
   Reviewed and merged. Known fragility: the grid parser counts every line of
   the wiki template as a slot, so a named parameter added by an editor would
   shift later slots.
2. Read the Collections screen: the BLUEPRINTS tab and FOUND count, the
   lattice, found versus empty cells, cells mapped to blueprints by the grid
   order (or learned from the found icons when no public order exists), and
   ownership stored in the user database. Feeds the weapon rule on its own.
3. Hover assist: with the cursor over an empty cell, the popup shows the
   blueprint's name and the best places to find it.
4. Greyed icons drawn in the empty cells on the overlay surface (phase 4).

## Phase 5. The stash ledger. Next, after phase 3

Jeffrey's words: "Maybe when my hotkey opens the inventory, you start
watching and updating a local database of my inventory?" Decided 11 Oct.
Local only.

Will contain: a guided stash scroll anchored on the header count, deltas from
later scans, an unsure flag when the numbers disagree.

## Phase 6. Progress from the screen. Next, after phases 2 and 5

Jeffrey's words: "You might need to know how the project pages are laid out
and infer what progress is made per project." Decided 11 Oct.

Will contain: screenshots of the workshop and project pages at 3840x2160
from Jeffrey, then one reader at a time for the workshop overview, station
pages, project pages and the quest list.

## Open decisions for Jeffrey

- Which recycle row the game's own tooltip shows (field or home). One hover
  on a Power Bank settles the chip label.
- Whether ended projects (Expedition 1 to 4) are shown, hidden, or hidden by
  default. Hidden by default until he says.
- Whether coin-funding project phases appear as a line.
- In-game check of recycle rows where the APIs and the wiki disagree (Anvil I
  to IV and Arpeggio I to IV look swapped, plus Wasp Driver and Synthesized
  Fuel).
- Whether to install `gtk4-layer-shell` (`sudo pacman -S gtk4-layer-shell`)
  so the Wayland-native overlay mode can be tried alongside the X11 mode.
- Four catalog names appear twice ignoring case (Colorful Shoes and three
  others). The database keeps the last row. Leave it, or dedupe in the
  updater.
- Blueprint find locations: MetaForge's terms ask for attribution and ask
  paid projects to contact them first, and the vote endpoint is
  undocumented. ARLO is free. Use it with attribution in the README, or stay
  on the wiki table alone.

## Known follow-ups

- Add `TOOLTIP_CHIPS=true`, `CAPTURE_LIBRARY=false` and
  `CAPTURE_LIBRARY_DIR=debug/captures` to `.env.example` (blocked for agents
  by permission settings).
- The ruff baseline on master has 9 lint errors and 7 unformatted files
  outside the new code.
- 158 items have no rarity because the API gives none.
- The capture library should record the cursor position with each image
  (sidecar or file name) so the evaluation script picks the same card the
  live scanner did now that captures are full screen. Added 12 Oct.
