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
    partial reads. Changed 12 Oct. Bake-off resumed 14 Oct from the stopped
    run's work, measuring only. Wiring the winner in follows from its
    numbers. Changed 14 Oct. Bake-off done 14 Oct: on 56 labeled cards
    paddle resolved 56 at 54 ms per card, Tesseract with the title line
    isolated 56 at 116 ms, today's reader 55 at 504 ms. Jeffrey, 14 Oct:
    "the delay between overlay shown on hovering new items is pretty slow.
    Improve if possible." So part 2 switches the app to paddle and reads the
    chips' label rows from the same pass. Done 14 Oct, reviewed: on the full
    pipeline (title plus the chips' row search) paddle took 400 ms a card,
    the isolated-title Tesseract reader 389 ms (popup at 135 ms), today's
    reader 777 ms, all equally accurate. The isolated-title reader is the
    default (no extra dependency, popup sooner), paddle stays selectable
    with OCR_ENGINE=paddle. A scan with no card now reads nothing instead
    of guessing from a box around the cursor. Waiting on Jeffrey's feel for
    the speed in game. Jeffrey's capture session, 14 Oct: Garnet (a weapon
    mod card) read as "GARNET REQUIRED RESOURCES G NCIL PARTS" and showed
    UNKNOWN. Fixed 14 Oct, reviewed: light grey description text now ends
    the title, Garnet resolves to Garnet Stencil, 25 of 25 visible titles
    that night, 56 of 56 on 12 Oct.
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
    cause the log shows. The timestamped log is built (14 Oct, reviewed):
    with DEBUG_MODE=true, arc_helper.log now has millisecond times, each
    scan's start, period and duration, each popup and chip show, touch and
    hide with its reason, every hide timer armed or cancelled, and each
    scanner state change. Changed 14 Oct. Jeffrey's 10 s hover log (14 Oct)
    confirmed the leading suspect: scans every 1.1 to 2.5 s against a 1 s
    hide timer, so both windows hid and re-mapped on every scan. Fix (14 Oct,
    reviewed): the hide delay follows the measured scan speed, twice the
    slowest recent scan period, capped at 6 s, never below
    OVERLAY_DISPLAY_TIME. Jeffrey, 14 Oct, after testing: "nice. fixed."
    Done 14 Oct.
13. Search the whole screen for the card. The labeled capture library (320
    captures, 113 with a card) showed 56 cards missing their title row
    because the cursor-centred window cut tall weapon cards at the top. The
    search window now defaults to the full screen. Added 12 Oct, done 12 Oct,
    reviewed. Confirmed by Jeffrey in game 12 Oct: "It seems like all the
    weapon identification is working now."
14. Capture library only on request. Jeffrey, 14 Oct: "only allow this to
    run if I'm using claude." The library was still on in his .env after a
    test, saving a 2 MB full-screen image about every 2 seconds on the scan
    thread (471 files, 285 MB). It now turns on only with
    `uv run arc-helper --capture`, ignores .env, stops after a per-run limit,
    and saves on a background thread. Added 14 Oct, done 14 Oct, accepted
    by Codex with no items.
15. Gold border on the hovered item. Jeffrey, 14 Oct: "would be nice if these items were higlighted (onscreen gold border around the item?)  At least a big \"KEEP\" on the popup card.  Would also be nice if the cost for each item would show up on the small item icons."
    First an investigation on the 34 full-screen captures: can the hovered
    cell be found beside the tooltip card (the game outlines it in white),
    how reliably and how fast. Then a gold frame drawn around it when the
    verdict is KEEP. Added 14 Oct. Investigation done 14 Oct: the game draws
    an animated glow ring (not a white outline) around the hovered cell,
    found right in 34 of 34 full-screen captures in about 4 ms. The detector
    and a click-through gold frame are merged. Wired into the scanner 14 Oct,
    reviewed and merged: the frame is drawn just outside the game's glow
    ring while the verdict is keep and hides with the chips. Waiting on
    Jeffrey's look in game (the live screenshot could not be taken, the
    screen was locked). Jeffrey's capture session the same night showed two
    failures: no frame when the card opens left of the cell (Stiletto I in
    quick use), and a frame across the card's ACTIONS tab when ARLO's
    corner popup covered the card (Anvil I). Fixed 14 Oct, reviewed: 34 of
    35 of that night's captures right (21 before), all 8 left-opening cards
    right. The left-opening items that night were USE or unknown, and the
    frame only shows for keep.
16. Never read a card ARLO itself covers. Found in Jeffrey's capture
    session, 14 Oct: an Anvil I card opened under the corner popup, the
    next scan read body text and showed UNKNOWN, the wider UNKNOWN popup
    uncovered the title, and the popup flipped between KEEP and UNKNOWN
    every scan. A read whose card overlaps ARLO's popup is dropped. Jeffrey,
    14 Oct, with a screenshot: "You should move the upper left popup to the
    red box -- it'll look nicer and be closer to the cursor." The popup's
    home is now the empty LOADOUT panel area below the weapon slots, and it
    steps to the top-left corner only while a card covers that spot. Done
    14 Oct, reviewed (one edge case carried into step 17 and fixed there).
17. Fast show and hide. Jeffrey, 14 Oct: "really wish the overlay was
    faaster about showing/hiding the popups." The scan loop now watches the
    card every cycle: a new card is read at once, the same card is never
    read again, and a card gone for two cycles hides everything at once (a
    3 s hold is only a safety net). No pause after a cycle with a card.
    Replaying the session's 44 captures, reads dropped from 44 to 26. Added
    14 Oct, done 14 Oct, reviewed.
18. ARLO wakes up in raid too. Jeffrey, 14 Oct: "Items in found containers
    and arc do not get the overlay." In raid the MAP tab shifts the
    INVENTORY tab out of the calibrated trigger boxes, so ARLO never woke.
    A wide band across the top centre is searched when the boxes miss
    (about 60 to 75 ms, only when the boxes fail). Done 14 Oct, reviewed:
    his in-raid screenshot triggers, 142 of 144 stash captures still
    trigger through the boxes, no false triggers. Waiting on a raid test.

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
6. Readable popup text and a big verdict word. Jeffrey, 14 Oct: "At least
   a big \"KEEP\" on the popup card." Found 14 Oct: uv's Python bundles a Tk
   with only the bitmap font "fixed", so on Linux every popup font size is
   ignored. Text is drawn with Pillow from a system font when Tk has no
   scalable fonts. Jeffrey, 14 Oct, clarifying: "the KEEP shouldn't be on the
   popup in the upper left, it should be near the \"HOME ONLY\" badge -- I
   need to see the keep on the item itself!" So the big verdict word is a
   third chip on the game's card, on the RECYCLES INTO row beside HOME ONLY,
   and the corner popup keeps its old size. Jeffrey, the same evening,
   sharing ARC Tracker's card: "It looks like a part of the game itself. You
   should strive for this." So the chips use a bundled game-style font
   (Barlow Semi Condensed) on a dark slate panel with a coloured left
   stripe. Built and reviewed 14 Oct (Codex, two rounds), merged. Added
   14 Oct, changed 14 Oct. Waiting on Jeffrey's look in game.
7. A card beside the game's tooltip in the style of ARC Tracker's overlay,
   from Jeffrey's screenshot, 14 Oct: the verdict as a header bar ("Keep
   all 4: 6 more needed"), a needed-for list with a progress bar (quest,
   workshop level, count), and "You have" with the stash count once phase
   5 knows it. Replaces the corner popup. Added 14 Oct. Jeffrey, 14 Oct,
   sharing four more ARC Tracker screenshots (keep, sell, recycle cards):
   "This is super nice. You should just copy it." Part 1, drawing the card
   from the verdict data, done 14 Oct, reviewed. The recycle wording names
   the parts you still need, done 14 Oct. The window that shows it in the
   LOADOUT panel box, done 14 Oct. Hooked into the scanner in place of the
   corner popup (ADVICE_CARD=false brings the popup back), done 14 Oct,
   reviewed. Waiting on Jeffrey's look in game. Changed 14 Oct.

## Phase 3. Read the grid. Next, after phase 1

Jeffrey's words: "OpenCV sounds like the frontier option for this work."
Answer recorded 11 Oct: OpenCV for geometry and gating, an embedding model
for icon identity, OCR for text. Changed 11 Oct.

Will contain:

1. A perceptual-hash baseline, measured first, expected to fail on lookalikes.
   Done 14 Oct, reviewed: on 110 slot crops labeled from the hovered cell
   and the title read, 46 percent top-1, weapons 9 of 45, wide weapon slots
   all missed, tiers sometimes differ only in the numeral. The evidence for
   step 3.
2. Slot lattice from edges and contours, frame stillness check, only while
   the INVENTORY trigger is active. Done 14 Oct, reviewed: every slot and
   its panel in about 28 ms a frame, 3 uncovered slots missed in 144 frames.
3. An embedding model (DINOv2-small under onnxruntime, Apache-2.0) over a
   gallery built from the stable arctracker icon PNGs plus crops learned from
   hovering (ARLO already reads the tooltip). Done 14 Oct, reviewed: on the
   same 110 crops, top-1 70 percent (hash 50), top-5 99 percent, base item
   95.5 percent, weapons 89 percent on the base item but 25 percent on the
   exact tier. Wide weapon slots needed a 40 degree turn to match the
   catalog art. About 29 ms a slot on the CPU, model 88.5 MB downloaded on
   first use. Recommendation: the model names the base item, the tier comes
   from the slot numeral (step 4).
4. Tier numerals on icons by glyph matching (the loadout shows a IV glyph on
   the weapon card), stack counts by OCR. Done 14 Oct, reviewed: tiers 44 of
   44 weapon slots (square and wide), no false reads, 0.4 ms a slot. Stacks
   52 of 53. With the icon model, weapons identified exactly 39 of 44 (11
   before). Measured on the crops the reader was tuned on.

## Phase 4. Badges on every cell. Underway from 14 Oct

Decided 11 Oct. Carries the Wayland risk.

1. Overlay layer experiment: `scripts/spike_overlay_surface.py` with
   findings in `docs/spikes/2026-10-11-overlay-surface.md`. The X11 mode
   ran on the live Plasma session (depth 32, on top, empty input shape). The
   layer-shell mode needs `gtk4-layer-shell`, not installed. Done 11 Oct.
2. Jeffrey's run with the game open decides the mode.

Will contain: badge drawing from verdicts and the grid, hidden while the
grid moves. Jeffrey, 14 Oct, asked for a gold border on items to keep and
the sell price on the small item icons: both become badge styles here,
since they need every cell read (phase 3). Changed 14 Oct.

3. The badge surface: one full-screen transparent click-through layer drawn
   by a helper under the system Python with GTK3 (the spike's x11 mode),
   fed badges by ARLO. Done 14 Oct, reviewed. Added 14 Oct.
4. The stash pass: identify every visible slot from one frame (slot
   lattice, icon model, tier numeral, stack count) and turn it into badges,
   gold for keep and a price on every icon. Done 14 Oct, reviewed: on 12
   sample frames, 264 of 360 filled slots right, 7 wrong (empty or covered
   slots, no price shown), the rest unidentified and left without a badge.
   Blueprints that share their item's icon get no badge unless the tier
   settles it. About 0.6 s a pass. Added 14 Oct.
5. Wire the stash pass and the badge surface into the scanner, only while
   the grid is still. Done 15 Oct, reviewed (STASH_BADGES, on by default):
   a background pass runs when two frames are still, badges clear when the
   grid moves or the inventory closes. The first pass after start takes
   about 12 s while the icon model loads. Waiting on Jeffrey's live look.
   Follow-up from review: make the badge hand-off non-blocking so a hung
   helper can never stall the hover path. Added 14 Oct.
6. Tiles like ARC Tracker. Jeffrey, 15 Oct, after the scale fix: "it's
   bleeding through and doesn't look seamless", and of ARC Tracker's legend:
   "This is great!!! Ensure this approach is implemented." The type icon is
   covered by an opaque patch in the tile's own bar colour with the stack's
   worth on it, a red "have/need" tag and outline on items still short, a
   green check tag when you have enough, gold only on the hovered slot.
   Done 15 Oct, reviewed: bar geometry and counts, the drawing protocol and
   the wiring into the live badges, merged in three parts. Added 15 Oct.
7. A bolder gold frame. Jeffrey, 15 Oct: "The gold border needs to be more
   obvious." The hovered frame is now twice as thick and a brighter gold,
   and the red outline on short items is thicker too. Done 15 Oct. Added
   15 Oct.

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

## Phase 5. The stash ledger. Underway from 15 Oct, moved up 15 Oct

Jeffrey's words: "Maybe when my hotkey opens the inventory, you start
watching and updating a local database of my inventory?" Decided 11 Oct.
Local only.

Jeffrey, 15 Oct: "we need to work on inventory tracking and live updating of
the inventory based on smart parsing of the visible page. This includes
tallying inventory and stash as one. Tough but possible. It will not really
be useful without this, you know?" Moved up 15 Oct.

1. Read the anchors: the stash header (used and capacity slots, total
   value) and the loadout header (weight and value). Done 15 Oct, reviewed.
2. Stitch the stash across scrolling: the scrollbar position and the
   overlap with the last frame give each visible row its absolute place.
   Needs a capture session scrolling the whole stash, waiting on Jeffrey's
   screen recording (the capture library only saves frames when a card is
   read). Must also handle the category filter tabs, since old captures
   only show filtered stash headers. Added 15 Oct.
3. The ledger: stash plus backpack, quick use, safe pocket and equipment
   in one local store, updated from every still frame. Added 15 Oct.
4. Use it: total owned feeds the have and need tags, the advice card's "You
   have N" and "keep N more". Added 15 Oct.
5. Unsure flag: when the ledger disagrees with the headers, say so instead
   of showing wrong numbers. Added 15 Oct.

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

- Add `TOOLTIP_CHIPS=true` and `CAPTURE_LIBRARY_DIR=debug/captures` to
  `.env.example` (blocked for agents by permission settings). Changed 14 Oct:
  `CAPTURE_LIBRARY` is no longer a setting, so it does not belong there, and
  Jeffrey can delete it from his `.env`.
- The ruff baseline on master has 9 lint errors and 7 unformatted files
  outside the new code.
- 158 items have no rarity because the API gives none.
- The capture library should record the cursor position with each image
  (sidecar or file name) so the evaluation script picks the same card the
  live scanner did now that captures are full screen. Added 12 Oct.
