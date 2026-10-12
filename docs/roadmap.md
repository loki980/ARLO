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

## Phase 1. Chips on the tooltip. Underway from 11 Oct, about 65 percent

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

Ahead, in order:

7. Find the card anywhere on screen. 99 of the 125 captures held no card
   because the fixed cursor-relative box missed where the game drew it.
   Replace the box with a search window and OpenCV connected components on
   the cream mask, nearest solid rectangle to the cursor, old box as the
   fallback. Retires the right-side flip TODO. Added 11 Oct, started 11 Oct.
8. Jeffrey's live check of the chips: where they land, and whether the game's
   own "Recycles into" row shows Battery only or Battery and Wires for a
   Power Bank.
9. A capture run for tiers I to III so the stroke thresholds are tuned on
   every numeral.
10. Decide the recycle chip label, "home only" or "field only", from step 8.
11. OCR engine trial (PaddleOCR PP-OCRv5 under onnxruntime), only if accuracy
    still falls short after step 9. Decided 11 Oct as an option.

## Phase 2. Advice with a reason. Underway from 11 Oct, about 22 percent

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

Ahead, in order:

2. The verdict engine. Rule 0: weapons are keep, at least one, more until the
   blueprint is owned. Then active needs minus progress, a reason naming up
   to three sources, the fallback (sell or recycle) from the item's data.
   Added 11 Oct (weapon rule), started 11 Oct.
3. Popup overhaul: the verdict word colored by the verdict (fixes the
   keyword-order bug that colored "Keep until upgrade is complete; sell once
   done" gold), the catalog name instead of the raw read (the popup showed
   ANVILIV), home recycle outputs on the notes line. Changed 11 Oct.
4. Manual progress file (`progress.csv`, not committed) for bench levels,
   finished quests and project phases, until phase 6 reads them.
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

## Known follow-ups

- Add `TOOLTIP_CHIPS=true`, `CAPTURE_LIBRARY=false` and
  `CAPTURE_LIBRARY_DIR=debug/captures` to `.env.example` (blocked for agents
  by permission settings).
- The ruff baseline on master has 9 lint errors and 7 unformatted files
  outside the new code.
- 158 items have no rarity because the API gives none.
