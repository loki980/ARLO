# ARLO roadmap

Source of truth for the overlay extension. The published page at
https://claude.ai/artifact/6mrLLrRXHBxS4zRzMv2vwd shows the same roadmap as a
page to read. Its source is `docs/roadmap.html`. Update this file first, then
republish the page to the same URL.

Design detail is in `docs/superpowers/specs/2026-10-11-overlay-extension-design.md`.
Markers such as "done 11 Oct" or "added 11 Oct" mark changes in the last 14
days and are dropped after that. Completed steps keep their date.

## Overall progress

About 83 percent of the whole roadmap, weighted by effort: 67 of 71 steps
done, 6 of 8 phases done (18 Oct, checked against the code). Weights: phase 0
counts 1, phases 2 and 4b count 2, every other phase 3. A phase's percent is
its steps done over its steps. Phase 6 counts its 3 planned items as steps.
Dropped steps do not count. Jeffrey, 15 Oct: "the roadmap should show overall
progress for the entire roadmap". Added 15 Oct.

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
2. ARC Tracker teardown. Version 2.3.2, ideas only, the app is all rights
   reserved. Done 11 Oct.
3. Data sources checked. Found the MetaForge recycle feed dead since 2 Apr
   2026 and the quest reader reading a field that does not exist. Chose the
   arctracker.io public API for items, quests and projects, MetaForge for
   workshop levels (the only source with Outpost). Done 11 Oct.
4. Design written. Done 11 Oct.

## Phase 1. Hover an item and read the whole story. Done 17 Oct

One small chip on the game's tooltip: HOME ONLY beside the Recycles into
icons, naming the extra parts you only get from the Speranza recycler. The
game already shows the price and the advice card gives the advice, so the
price and KEEP chips went on 17 Oct. ARLO also finds the card anywhere, reads
its title fast and frames the hovered slot in gold. Changed 17 Oct: Jeffrey
dropped the sells-for chip, the card already shows the price. 17 of 17 steps
done, steps 8 to 10 dropped and not counted.

Jeffrey's words: "Next to 'can be found in' on the right hand side, show
'sells for'. Next to 'recycles into' icons, add the additional items the item
recycles into." Decided 11 Oct.

Done (steps 8 to 10 were dropped 18 Oct):

1. The two chips and the geometry that places them. Price anchored to the
   label when the card is cut off by the capture. Done 11 Oct.
2. Home and field recycling kept apart. Stored apart in the database. Done
   11 Oct.
3. Weekly data feed repaired. Dead feed removed, arctracker.io as the recycle
   source, quest reader fixed, the weekly job installs the project. Done
   11 Oct.
4. Roman numerals resolved from the catalog and the picture. Jeffrey's
   words: "I have a suspicion that the roman numerals identification is
   better, but not fixed. This should be tested and maybe improved." A
   capture library, a catalog-aware resolver, a stroke-count vote. Rarity
   checked as a signal and rejected (Anvil I to IV are all Rare). Done
   11 Oct.
5. New data loaded, catalog grown to 1,109 items. Refreshed from the new
   feed, the database stable on open. Done 11 Oct.
6. First capture session measured. 125 captures, 7 with a card and all 7
   read right (Anvil IV, Stiletto IV, vote IV at 0.88 on the real font). Done
   11 Oct.
7. Find the card anywhere on screen. 99 of the 125 captures contained no card
   because the fixed cursor-relative box missed where the game drew it.
   Replaced the box with a search window and OpenCV connected components on
   the cream mask, nearest solid rectangle to the cursor, old box as the
   fallback. Retires the right-side flip TODO. OpenCV is now a dependency.
   Added 11 Oct, done 11 Oct, reviewed.

8. Your live check of the chips. Where they land, and whether the game's own
   "Recycles into" row shows Battery only or Battery and Wires for a Power
   Bank. Dropped 18 Oct: the chips it checked are gone, and Jeffrey's 17 Oct
   screenshot showed the game's recycle row lists the home outputs, so HOME
   ONLY stays.
9. Capture run for tiers I to III. So the stroke thresholds are tuned on
   every numeral. Dropped 18 Oct: the labeled capture library already has
   tiers I to IV and the catalog resolver settles the tier.
10. Decide the recycle chip label. "Home only" or "field only", from step 8.
    Dropped 18 Oct: settled by the same screenshot.
11. A faster title reader. Replace the card title reader on evidence. Brought forward 12 Oct after
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
    of guessing from a box around the cursor. Jeffrey's capture session, 14 Oct: Garnet (a weapon
    mod card) read as "GARNET REQUIRED RESOURCES G NCIL PARTS" and showed
    UNKNOWN. Fixed 14 Oct, reviewed: light grey description text now ends
    the title, Garnet resolves to Garnet Stencil, 25 of 25 visible titles
    that night, 56 of 56 on 12 Oct.
12. Popups stay put while you hover. Flicker, second cause, 12 Oct: "it's still flickering, but the frequency
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
13. Weapons identified in game. Search the whole screen for the card. The
    labeled capture library (320
    captures, 113 with a card) showed 56 cards missing their title row
    because the cursor-centred window cut tall weapon cards at the top. The
    search window now defaults to the full screen. Added 12 Oct, done 12 Oct,
    reviewed. Confirmed by Jeffrey in game 12 Oct: "It seems like all the
    weapon identification is working now."
14. Captures only when you ask. Capture library only on request. Jeffrey, 14 Oct: "only allow this to
    run if I'm using claude." The library was still on in his .env after a
    test, saving a 2 MB full-screen image about every 2 seconds on the scan
    thread (471 files, 285 MB). It now turns on only with
    `uv run arc-helper --capture`, ignores .env, stops after a per-run limit,
    and saves on a background thread. Added 14 Oct, done 14 Oct, accepted
    by Codex with no items.
15. Gold frame on the hovered item. Jeffrey, 14 Oct: "would be nice if these items were higlighted (onscreen gold border around the item?)  At least a big \"KEEP\" on the popup card.  Would also be nice if the cost for each item would show up on the small item icons."
    First an investigation on the 34 full-screen captures: can the hovered
    cell be found beside the tooltip card (the game outlines it in white),
    how reliably and how fast. Then a gold frame drawn around it when the
    verdict is KEEP. Added 14 Oct. Investigation done 14 Oct: the game draws
    an animated glow ring (not a white outline) around the hovered cell,
    found right in 34 of 34 full-screen captures in about 4 ms. The detector
    and a click-through gold frame are merged. Wired into the scanner 14 Oct,
    reviewed and merged: the frame is drawn just outside the game's glow
    ring while the verdict is keep and hides with the chips. Jeffrey's capture session the same night showed two
    failures: no frame when the card opens left of the cell (Stiletto I in
    quick use), and a frame across the card's ACTIONS tab when ARLO's
    corner popup covered the card (Anvil I). Fixed 14 Oct, reviewed: 34 of
    35 of that night's captures right (21 before), all 8 left-opening cards
    right. The left-opening items that night were USE or unknown, and the
    frame only shows for keep. Changed 15 Oct: the frame now marks every
    identified hovered item (Phase 4, step 6).
16. Never read a card ARLO covers. Found in Jeffrey's capture
    session, 14 Oct: an Anvil I card opened under the corner popup, the
    next scan read body text and showed UNKNOWN, the wider UNKNOWN popup
    uncovered the title, and the popup flipped between KEEP and UNKNOWN
    every scan. A read whose card overlaps ARLO's popup is dropped. Jeffrey,
    14 Oct, with a screenshot: "You should move the upper left popup to the
    red box -- it'll look nicer and be closer to the cursor." The popup's
    home is now the empty LOADOUT panel area below the weapon slots, and it
    steps to the top-left corner only while a card covers that spot. Done
    14 Oct, reviewed (one edge case moved into step 17 and fixed there).
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
    (about 60 to 75 ms, only when the boxes fail). Done 14 Oct, reviewed,
    reviewer items overruled: his in-raid screenshot triggers, 142 of 144 stash captures still
    trigger through the boxes, no false triggers.
19. No advice for a garbled read. Found in the 15 Oct log: the advice card
    showed reads such as "CRACKED BIOSCANNER ;MER:;{JC:..." and "IUST'". A
    catalog name followed by junk now resolves to that name, and a read
    that matches nothing shows no advice. Done 15 Oct, reviewed. Added 15 Oct.
20. A tidier card. Jeffrey, 17 Oct: "I think the \"sells for\" overlay can
    be removed. It's already on the card. It'd be nice if the \"Home only\"
    badge lined up." Then "I think the \"keep/use/recycle/sell\" should go
    away too", and "get rid of the \"scanning\" and \"inventory\" -- these
    are almost like debug msgs". Done 17 Oct, reviewed, reviewer items overruled: the price and verdict
    chips are gone, the status window is off unless SHOW_STATUS_WINDOW is
    set, and HOME ONLY sits on the recycle icon row (4 of 4 test cards).
    Added 17 Oct.

## Phase 2. Advice with a reason. Done 18 Oct

Jeffrey's words: "Keep until upgrade is complete; sell once done" is the kind
of advice he wants, with the source named. And, after the popup told him to
recycle a fully upgraded Anvil IV: "Fully upgraded anything is great. 1 of any
kind of weapon shouldn't be recycled, especially if you don't have the
blueprint." Decided 11 Oct.

Done:

1. The needs list. `needs.csv`, 289 rows (80 quest, 128 project, 81
   workshop) from arctracker quests and projects plus MetaForge workshop
   levels, all nine Frozen Trail quests and the Outpost levels. Loaded into
   the database. Done 11 Oct.

2. The verdict engine, with the weapon rule. Rule 0: weapons are keep, at least one, more until the
   blueprint is owned. Then active needs minus progress, a reason naming up
   to three sources, the fallback (sell or recycle) from the item's data.
   Radial Press reads "keep, 3 for Gunsmith IV, then sell". Added 11 Oct
   (weapon rule), done 11 Oct, accepted by the fallback reviewer.
3. Popup overhaul. The verdict word colored by the verdict (fixes the
   keyword-order bug that colored "Keep until upgrade is complete; sell once
   done" gold), the catalog name instead of the raw read (the popup showed
   ANVILIV), home recycle outputs on the notes line. Changed 11 Oct, done
   11 Oct.
4. Manual progress file. A file (`progress.csv`, not committed) for bench levels,
   finished quests and project phases, loaded into `arlo_user.db`, a second
   untracked database so the weekly refresh never conflicts with it. Done
   11 Oct.

6. Readable text in a game-style font. Readable popup text and a big
   verdict word. Jeffrey, 14 Oct: "At least
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
   14 Oct, changed 14 Oct, done 14 Oct. Changed 17 Oct: the KEEP chip it
   added was retired by Phase 1 step 20, the advice card gives the advice.
7. A card that looks like part of the game. A card beside the game's tooltip in the style of ARC Tracker's overlay,
   from Jeffrey's screenshot, 14 Oct: the verdict as a header bar ("Keep
   all 4: 6 more needed"), a needed-for list with a progress bar (quest,
   workshop level, count). Replaces the corner popup. Added 14 Oct. Jeffrey, 14 Oct,
   sharing four more ARC Tracker screenshots (keep, sell, recycle cards):
   "This is super nice. You should just copy it." Part 1, drawing the card
   from the verdict data, done 14 Oct, reviewed. The recycle wording names
   the parts you still need, done 14 Oct. The window that shows it in the
   LOADOUT panel box, done 14 Oct. Hooked into the scanner in place of the
   corner popup (ADVICE_CARD=false brings the popup back), done 14 Oct,
   reviewed. Done 14 Oct, in daily use, and Jeffrey's 15 Oct report led to
   step 8. Changed 14 Oct. Changed 18 Oct: the "You have N" line with the
   stash count moved into Phase 5, step 4.
8. The advice card stays under your second gun. Jeffrey, 15 Oct: "The popup
   has moved back to the upper left?? It should be beneath my 2nd gun".
   Home is found from the weapon slots on screen (base and raid differ),
   the card steps just aside when a game card opens over it and comes back
   as soon as it is free. Done 15 Oct, reviewed. Added 15 Oct.
9. Learn or sell a blueprint. Jeffrey, 15 Oct, of the Bettina Blueprint:
   "Unlearned blueprint ... def not sell, should learn this." Then "if
   already know it, should sell, though", and "vertical grip 3, sell,
   bettina, consume/learn." The game draws a check mark on a learned
   blueprint's tile, so ARLO reads it: no mark gives "Learn it", a mark
   gives the usual advice. Each read is remembered for the weapon rule.
   Done 15 Oct, reviewed. Added 15 Oct.
10. Known blueprints in the advice. The weapon rule stops saying "more
    until you own the blueprint" once ARLO has seen it learned, and the
    advice card says "You already know this blueprint" when that is why it
    says sell. Done 15 Oct, reviewed. Added 15 Oct.

Ahead:

5. Your decisions on ended projects and coin phases. Jeffrey's decisions: ended projects (Expedition 1 to 4) hidden by default
   until he says otherwise, coin-funding phases produce no needs. Done 18
   Oct: Jeffrey chose to keep both, ended projects hidden by default
   (SHOW_ENDED_PROJECTS brings them back) and coin phases skipped.

## Phase 3. Read the grid. Done 14 Oct

Jeffrey's words: "OpenCV sounds like the frontier option for this work."
Answer recorded 11 Oct: OpenCV for geometry and gating, an embedding model
for icon identity, OCR for text. Changed 11 Oct.

Done:

1. A perceptual-hash baseline. Measured first, expected to fail on
   lookalikes.
   Done 14 Oct, reviewed: on 110 slot crops labeled from the hovered cell
   and the title read, 46 percent top-1, weapons 9 of 45, wide weapon slots
   all missed, tiers sometimes differ only in the numeral. The evidence for
   step 3.
2. Slot lattice. From edges and contours, frame stillness check, only while
   the INVENTORY trigger is active. Done 14 Oct, reviewed: every slot and
   its panel in about 28 ms a frame, 3 uncovered slots missed in 144 frames.
3. An embedding model for icon identity. DINOv2-small under onnxruntime
   (Apache-2.0) over a
   gallery built from the stable arctracker icon PNGs plus crops learned from
   hovering (ARLO already reads the tooltip). Done 14 Oct, reviewed: on the
   same 110 crops, top-1 70 percent (hash 50), top-5 99 percent, base item
   95.5 percent, weapons 89 percent on the base item but 25 percent on the
   exact tier. Wide weapon slots needed a 40 degree turn to match the
   catalog art. About 29 ms a slot on the CPU, model 88.5 MB downloaded on
   first use. Recommendation: the model names the base item, the tier comes
   from the slot numeral (step 4).
4. Tier numerals and stack counts. Tier numerals on icons by glyph matching
   (the loadout shows a IV glyph on the weapon card), stack counts by OCR. Done 14 Oct, reviewed: tiers 44 of
   44 weapon slots (square and wide), no false reads, 0.4 ms a slot. Stacks
   52 of 53. With the icon model, weapons identified exactly 39 of 44 (11
   before). Measured on the crops the reader was tuned on.

## Phase 4. Badges on every cell. Done 18 Oct

Decided 11 Oct. The Wayland risk sat here and is settled: the X11 layer works
over the game. Delivered: prices on every icon (steps 4 to 6), the gold frame
on the hovered slot (steps 6, 7, 10). 21 of 21 steps done.

Done:

1. Overlay layer experiment. A script, `scripts/spike_overlay_surface.py`,
   with findings in `docs/spikes/2026-10-11-overlay-surface.md`. The X11 mode
   ran on the live Plasma session (depth 32, on top, empty input shape). The
   layer-shell mode needs `gtk4-layer-shell`, not installed. Done 11 Oct.
2. Your run decides the drawing mode. Jeffrey's run with the game open
   decided it: X11 mode, live with the game. The layer-shell mode is only
   needed if X11 ever fails. Done 14 Oct.
3. The badge layer. One full-screen transparent click-through layer drawn
   by a helper under the system Python with GTK3 (the spike's x11 mode),
   fed badges by ARLO. Done 14 Oct, reviewed. Added 14 Oct.
4. The stash pass. Identify every visible slot from one frame (slot
   lattice, icon model, tier numeral, stack count) and turn it into badges,
   gold for keep and a price on every icon. Done 14 Oct, reviewed: on 12
   sample frames, 264 of 360 filled slots right, 7 wrong (empty or covered
   slots, no price shown), the rest unidentified and left without a badge.
   Blueprints that share their item's icon get no badge unless the tier
   settles it. About 0.6 s a pass. Added 14 Oct.
5. Gold borders and prices, live. Wire the stash pass and the badge surface into the scanner, only while
   the grid is still. Done 15 Oct, reviewed (STASH_BADGES, on by default):
   a background pass runs when two frames are still, badges clear when the
   grid moves or the inventory closes. The first pass after start takes
   about 12 s while the icon model loads.
   Follow-up from review: make the badge hand-off non-blocking so a hung
   helper can never stall the hover path. Added 14 Oct, done as step 12.
6. Tiles like ARC Tracker. Jeffrey, 15 Oct, after the scale fix: "it's
   bleeding through and doesn't look seamless", and of ARC Tracker's legend:
   "This is great!!! Ensure this approach is implemented." The type icon is
   covered by an opaque patch in the tile's own bar colour with the stack's
   worth on it, a red "have/need" tag and outline on items still short, a
   green check tag when you have enough, gold only on the hovered slot.
   Done 15 Oct, reviewed: bar geometry and counts, the drawing protocol and
   the wiring into the live badges, merged in three parts. Added 15 Oct. Changed 18 Oct: the red outline on
   short tiles is gone at Jeffrey's ask, the have/need tag says it.
7. A bolder gold frame. Jeffrey, 15 Oct: "The gold border needs to be more
   obvious." The hovered frame is now twice as thick and a brighter gold,
   and the red outline on short items is thicker too (outline dropped 18 Oct). Done 15 Oct. Added
   15 Oct.
8. A steady inventory trigger. Jeffrey, 15 Oct: "the \"fade in fade out\"
   problem is back. Also, the grid doesn't work back in base either". The
   log showed ARLO deciding the inventory closed 54 times in an hour after
   one missed check each, hiding the popups and clearing the badges. It
   now waits for misses over 2 seconds and logs why each check missed.
   Done 15 Oct, reviewed, reviewer items overruled. Added 15 Oct.
9. Badges over a live background. Jeffrey, 15 Oct: "grid doesn't work in
   game inventory". In raid the game world moves behind the inventory, and
   the stillness check compared the whole screen, so every pass was thrown
   away. It now compares only the slots. Done 15 Oct, reviewed. Added 15 Oct.
10. The gold frame fits wide slots. Jeffrey, 15 Oct: "yellow misdraws
    sometimes." On a weapon slot the frame covered only its right part. It
    now snaps to the whole slot found under the cursor. Built with Phase 2,
    step 9. Done 15 Oct, reviewed. Added 15 Oct.
11. Badges in about a second. Jeffrey, 15 Oct: "grid view takes a long
    time to highlight everything". A pass takes 3 to 6 s live and is thrown
    away whole when anything changes mid-pass. Items already seen are
    remembered by their look, badges draw in batches as slots are
    identified, and unchanged slots keep theirs. Started 15 Oct, paused
    halfway for the night at Jeffrey's ask, underway again 15 Oct evening.
    Part 1 (keep the badges of unchanged slots, a fading card hides
    nothing) done 15 Oct, reviewed. Part 2 (remember items by their look, a
    repeat view in about 55 ms, the model loaded at start) done 15 Oct,
    reviewed. Done 15 Oct. Added 15 Oct.
12. Never stall on the badge helper. From the 14 Oct review: badges are
    handed to the helper in the scan's own thread, so a helper that stops
    reading could freeze the popups. A writer thread now sends them, the
    newest set wins, and a stuck helper is stopped. Done 15 Oct, reviewed.
    Added 15 Oct.
13. Badges right after a scroll. Jeffrey, 15 Oct: "it's pretty good!
    taking about 2-3 seconds to render with an inventory scroll. Not
    amazing, not terrible. Any other ideas you can come up with to make
    this even more responsive?" A latency log line, a shorter gap between
    passes, and badges for slots already known drawn before the pass.
    Done 15 Oct, reviewed: a one-row scroll replay draws in about 200 ms
    after the grid stops, was about 950 ms. Added 15 Oct.
14. Badges that move with the scroll. Measure how far the grid moved and
    shift the badges at once, so only the new row needs reading. Done
    15 Oct, reviewed: on the replay all 22 badges stay on their items
    through the scroll. Added 15 Oct.
15. Price text in the game's own font. Jeffrey, 15 Oct, on his backpack
    screenshot: "Can you make the font and font size match?" The worth on
    the patch matches the game's count in font, weight, size and baseline,
    and its cream colour. Done 15 Oct, reviewed. Added 15 Oct.
16. Price patches that hold still. The bar finder loses some bars once
    ARLO's own patch is drawn, so a patch could flicker. Done 15 Oct,
    reviewed: lost bars 19 to 1 of 1011 passes. Added 15 Oct.
17. Slot boxes that hold still. The slot finder moves 1 or 2 boxes by
    about 4 px under ARLO's red outline. Done 15 Oct, reviewed: no box
    moves, every slot hits the cache. Added 15 Oct.
18. Badges that never blink. Jeffrey, 15 Oct: "I'm still getting a little
    bit of flicker, to be honest. It's not always, but it's happened a
    couple times now." The log showed a card fading in or out read as the
    grid moving (46 clears, median 1.57 s to come back), and hover hiding
    the hovered slot's badges (127 times). The slot under the cursor is
    ignored, a big change waits one frame, hidden slots come back from the
    cache. Done 15 Oct, reviewed: on the replay a fading card, a hover and a
    pass with fewer badges clear and hide nothing. Added 15 Oct.
19. Tell a blueprint tile from its item. The blue grid background settles
    19 right guesses that a blueprint twin blocked. Done 15 Oct, reviewed:
    283 right instead of 269 on the captures, still 7 wrong. Added 15 Oct.
20. The overlay keeps up with the inventory. Jeffrey, 17 Oct: "there's
    still a noticible delay both showing and hiding the overlay. Is it
    possible to make this faster?" His log: badges hid 2.2 s after the
    inventory closed and came 2.1 s after it opened. The overlay now hides
    the moment the grid leaves the screen and shows known items on the
    first frame. Done 17 Oct, reviewed. The log showed the badges already
    cleared about 230 ms after closing, so the slow part left is the hover
    card (about 360 ms to show). Added 17 Oct.
21. Worth at a glance. Jeffrey, 17 Oct: "When looting, I'm often looking to
    sell the cheapest thing in my inventory. Can you make some sort of
    at-a-glance color gradient for the costs? Anything that's 5k or more
    needs to stand out. Stacks of expensive things should stand out too.
    The idea is I need to, at a glance, determine what to drop to make room
    for something better." Price patches get darker the cheaper the stack
    (near black under 250), brighter toward 5,000, and white with black
    text from 5,000 up. Grey on purpose, so no value colour is mistaken for
    a rarity. Added 17 Oct. Done 18 Oct, reviewed, reviewer items overruled. ARLO paints
    over its own patches before reading a bar, so the new colours read as
    well as before (0 item changes on 12 captures, against 6 with the old
    patches).
22. Worth in colour. Jeffrey, 18 Oct, after seeing step 21's greys: "the
    gradient is black->grey->white. Can we instead be red to yellow to green
    to gold?" Red for the cheapest stacks, through yellow and green, gold
    from 5,000 up. Green and gold are also rarity colours, Jeffrey's call.
    The gold stays apart from ARLO's hover frame gold so the grid reader
    never confuses them. Added 18 Oct. Done 18 Oct, reviewed: ARLO reads
    its own coloured patches as well as the greys (0 item changes on 12
    captures). The gold is a deep gold (#ad7500), kept apart from ARLO's
    hover frame gold, closest in look to the orange at about 500.

## Phase 4b. Blueprint tracker. Done 18 Oct. Added 11 Oct

Jeffrey's words, 11 Oct, after the weapon rule said "more until you own the
blueprint": "We should build in some kind of blueprint tracker assist, too.
The blank boxes should be greyed out images of the actual blueprint. On hover
it shows the name and best places to find it based on public data." His
Collections screen shows FOUND: 20/91 on a grid of 17 columns (first noted as 18 by 6, corrected 18 Oct). Decided 11 Oct.

Done:

1. Blueprint data. The 91 blueprints, the order of the Collections grid, and where each
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
   shift later slots. Done 12 Oct.
2. Read the Collections screen. Jeffrey's screenshot of 15 Oct, FOUND
   25/91. Part 1, the reader and the ownership record, done 15 Oct,
   reviewed (25 found cells of 91, checked by eye).
   Part 2, read while idle whenever the screen is open, once per visit,
   done 15 Oct, reviewed. The BLUEPRINTS tab and FOUND count, the
   lattice, found versus empty cells, cells mapped to blueprints by the grid
   order (or learned from the found icons when no public order exists), and
   ownership stored in the user database. Feeds the weapon rule on its own. Done 15 Oct.

Ahead, in order:

3. Hover assist. With the cursor over an empty cell, the advice card shows
   the blueprint's name and the best places to find it. Done 18 Oct,
   reviewed: the card names the blueprint, its map and condition, the
   container types, whether it is scavengable and a quest or trials reward,
   and says plainly when a slot is unnamed in the public data.
4. Greyed icons in the empty cells. Drawn on the overlay surface (phase 4).
   Done 18 Oct, reviewed: 63 of the 66 empty cells on Jeffrey's screenshot
   show their blueprint in grey, the other 3 are the unnamed slots.

## Phase 5. ARLO knows what you own. Started 15 Oct, 85 percent, 6 of 7 steps. Moved up 15 Oct

Jeffrey's words: "Maybe when my hotkey opens the inventory, you start
watching and updating a local database of my inventory?" Decided 11 Oct.
Local only.

Jeffrey, 15 Oct: "we need to work on inventory tracking and live updating of
the inventory based on smart parsing of the visible page. This includes
tallying inventory and stash as one. Tough but possible. It will not really
be useful without this, you know?" Moved up 15 Oct.

Done:

1. Read the headers. The anchors are the stash header (used and capacity
   slots, total value) and the loadout header (weight and value). Done 15 Oct, reviewed.
2. Stitch the stash across scrolling. Part 0, capture runs save the whole
   grid, not only hovered cards. Jeffrey, 15 Oct: "You need to be able to
   take a screenshot of the entire screen, not just hover cards." Done
   15 Oct, reviewed. His scroll run saved 67 frames (header 275 of 304).
   Part 1, stitching those frames offline into one stash map, done 15 Oct,
   reviewed: rows 0 to 71, 275 filled slots, the header says 275. Part 2
   stitches live from the scanner. Jeffrey, 17 Oct: "is it possible to do
   this once and update as we go and stored between sesssions? Sometimes
   it'll be hard to see that I, for example, copy things to my stash.
   Howeveer, upon seeing it, you can update your saved values." Part 2a,
   known items remembered across restarts, and part 2b, the stash map
   saved and updated whenever rows are seen and checked against the
   header, both done 17 Oct, reviewed: a restart's first pass takes 78 ms
   instead of 845 ms, and the saved map rebuilds the same 275 cells. The
   stash header now reads while scrolled too. Done 17 Oct. Added 15 Oct.
3. The ledger. Stash plus backpack, quick use, safe pocket and equipment
   in one local store, updated from every still frame. Part 1, the loadout
   panels counted from one frame, checked against the LOADOUT header's
   value and saved locally, done 15 Oct, reviewed. It found the header
   reader failing in raid and next to a card, and small stacks unread.
   Part 1b, the header in raid and next to a card, reads both screenshots,
   done 15 Oct, reviewed, reviewer items overruled: its second review
   rejected one edge case (a value under 1,000 next to a card reads
   nothing), Jeffrey: "accept with notes". Part 1c, faster and surer stack
   counts (digit templates, 153 of 158 right and none wrong, 946 ms to 16 ms
   a pass), done 15 Oct, reviewed. Part 2, the map fed from every pass and
   the combined totals in the have/need tags, done 18 Oct, reviewed:
   Durable Cloth's tag went from 3/6 (visible slots) to 54/6 (whole stash).
   Added 15 Oct. Done 18 Oct.
Ahead, in order:

4. Use it. Use it on the advice card: You have N and keep N more, from the
   same totals the tags use. Added 15 Oct. Changed 18 Oct: the have and need
   tags already use the totals (step 3), the advice card is what is left.
   Done 18 Oct, reviewed: the card shows "You have 51" and "6 of 6, then
   sell 45" when the map is sure, "N+" while it is partial, "Not in your
   saved stash yet" for an item it has not seen, and weapons keep their
   "Keep 1".
5. An unsure flag. When the ledger disagrees with the headers, say so
   instead of showing wrong numbers. The stored map already falls back to
   on-screen counts when it disagrees with the header, what is left is
   saying so on screen. Added 15 Oct. Changed 18 Oct. Brought forward 18 Oct
   after Jeffrey's raid screenshot: "in-game, my inventoryu doesn't take
   into account the items in my stash". His map had 51 cells against 281
   used, so the tags fell back to the loadout alone. Now a partial map still
   counts, and the tag shows "52+/73" until a full scroll makes it sure.
   Done 18 Oct, reviewed: a tag never shows less than the slots on screen.
6. The map keeps building after a restart. Jeffrey, 18 Oct, after a full
   scroll of his stash from the top: "I still see 2+/3 on rusted bolts, for
   example". None of his 42 frames placed: after a restart ARLO has no
   last frame, and its saved looks no longer match because its own tags and
   patches changed on the tiles. Fix: anchor on the top of the stash and
   leave ARLO's own drawings out of the looks. Added 18 Oct. Done 18 Oct,
   reviewed, reviewer items overruled: the cause was a map started partway
   down the stash, so a restart froze it. A frame at the top of the stash
   now places as row 0 on its own. On Jeffrey's 18 Oct capture 74 of 91
   frames place and the map fills to 284, the header's count (none placed
   before). The old 51-cell map is dropped, one scroll from the top rebuilds
   it.
7. The map fills on one scroll. Jeffrey, 21 Oct, after one scroll from the
   top on the step 6 build: "ok, done with the scroll". The map started
   from the top as designed but stopped at 92 of 272 cells and then fell to
   80. A stash pass took about 1.5 s, so ARLO saw a view only every few
   seconds and the scroll outran it, and one view placed in the wrong spot.
   Step 1, the investigation on his capture of that scroll, underway 21 Oct.
   Added 21 Oct.

## Phase 6. It reads your progress too. Planned, after phases 2 and 5

Jeffrey's words: "You might need to know how the project pages are laid out
and infer what progress is made per project." Decided 11 Oct.

Will contain:

1. Screenshots of your workshop and project pages at your resolution, from
   you. Screenshots of the workshop and project pages at 3840x2160 from
   Jeffrey.
2. A reader for the workshop overview and station pages.
3. A reader for project pages and the quest list.

## Open decisions for Jeffrey

- Whether to install `gtk4-layer-shell` (`sudo pacman -S gtk4-layer-shell`)
  so the Wayland-native overlay mode can be tried alongside the X11 mode.
  Only needed if X11 ever fails.
- Blueprint find locations: MetaForge's terms ask for attribution and ask
  paid projects to contact them first, and the vote endpoint is
  undocumented. ARLO is free. Use it with attribution in the README, or stay
  on the wiki table alone.
- Banjo's price: it sells for 2,000 in ARLO and 1,000 in ARC Tracker. Check
  in game.
- 7 recycle rows differ between ARLO and ARC Tracker, mostly Sheet Metal from
  furniture. Check in game.
- In-game check of recycle rows where the APIs and the wiki disagree (Anvil I
  to IV and Arpeggio I to IV look swapped, plus Wasp Driver and Synthesized
  Fuel).
- Four catalog names appear twice ignoring case (Colorful Shoes and three
  others). The database keeps the last row. Leave it, or dedupe in the
  updater.

## Known follow-ups

- Done 18 Oct: the greyed Collections icons have their own setting
  (COLLECTIONS_ICONS), hide while the cursor is on a found cell where the
  game shows its tooltip, and can no longer stay up after pause.

- Done 18 Oct: docs/ITEMS.md says where KEEP SOME's owned count comes from.

- Add `TOOLTIP_CHIPS=true`, `CAPTURE_LIBRARY_DIR=debug/captures`,
  `OCR_ENGINE`, `ADVICE_CARD`, `STASH_BADGES` and `SHOW_STATUS_WINDOW` to
  `.env.example` (blocked for agents by permission settings). Changed 14 Oct:
  `CAPTURE_LIBRARY` is no longer a setting, so it does not belong there, and
  Jeffrey can delete it from his `.env`.
- The ruff baseline on master has 9 lint errors and 7 unformatted files
  outside the new code.
- 158 items have no rarity because the API gives none.
- The capture library should record the cursor position with each image
  (sidecar or file name) so the evaluation script picks the same card the
  live scanner did now that captures are full screen. Added 12 Oct.
- Done 18 Oct: the README, `docs/CONFIGURATION.md` and the `TOOLTIP_CHIPS`
  description describe the current overlay (no price chip, status window
  off by default, how to quit, how to fill the stash map), and the design
  spec has a Blueprint data section.
