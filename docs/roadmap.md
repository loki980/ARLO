# ARLO roadmap

Source of truth for the overlay extension. The published page at
https://claude.ai/artifact/6mrLLrRXHBxS4zRzMv2vwd shows the same roadmap as a
readable page. Its HTML source is `docs/roadmap.html`. Update this file first,
then republish the page to the same URL.

Design detail lives in `docs/superpowers/specs/2026-10-11-overlay-extension-design.md`.

## Goal, in Jeffrey's words (2026-10-11)

"I'd like an overlay that highlights items that I need to upgrade my bench, for
projects, etc. Some attached, some quests." "First and foremost, I'd like item
info (how much it sells for, what it recycles to back home) superimposed on
every item in the game." "Next to 'can be found in' on the right hand side,
show 'sells for'. Next to 'recycles into' icons, add the additional items the
item recycles into. It should have a highlight or something correlating to the
tooltip indicating these are additional if recycled back in Speranza, not in
the field." "The existing popup is great, but it really needs an overhaul."
"This needs to be aware of my own inventory somehow."

## Order

Items are numbered in build order. Each has the decision date and its state.

1. **Investigate and design.** Codebase survey, ARC Tracker 2.3.2 teardown
   (ideas only, the app is all rights reserved), data-source study. Decided
   2026-10-11. Done 2026-10-11.
2. **Repair the data feed.** Remove the MetaForge Supabase recycle call (key
   disabled 2026-04-02, every sync since fell back to the wiki). Use the
   arctracker.io public API for items with recycle and salvage split, fix the
   MetaForge quest parser (it read a field that does not exist), write
   `recycle_outputs.csv`. Decided 2026-10-11. Built, reviewed (two rounds,
   accepted) and merged to the `overlay-extension` branch 2026-10-11.
3. **Phase 1, chips on the tooltip.** Price chip on the "Can be found in" row,
   home-only recycle extras beside the "Recycles into" icons, `recycle_outputs`
   table with a home and field column. Decided 2026-10-11. Built, reviewed
   (two rounds, accepted) and on the `overlay-extension` branch 2026-10-11.
   Awaiting the live check on the 4K Wayland setup before merging to master.
4. **Load `recycle_outputs.csv` into the table** so the field column fills from
   the feed and the home rows stop being derived from `recycle_for`. Add the CSV
   to the weekly workflow. Decided 2026-10-11. Not started.
5. **Phase 2, verdicts with reasons.** `needs` and `progress` tables filled
   from arctracker quests and projects plus MetaForge workshop levels (the only
   source with Outpost). A verdict engine replaces the free-text actions. Popup
   overhaul: "Keep 3 for Gunsmith IV, then sell", color by verdict. Manual
   progress setting. Decided 2026-10-11. Not started.
6. **Phase 3, read the grid.** Lattice fit from a still full frame while the
   INVENTORY trigger is active, icon identity by perceptual hash with
   hover-to-learn, stack counts by OCR. Decided 2026-10-11. Not started.
7. **Phase 4, badges on every cell.** Spike a full-screen click-through
   surface with per-pixel transparency first (Tk cannot do it), then owned over
   needed on each cell, colored by verdict, hidden while scrolling. Decided
   2026-10-11. Not started. Carries the Wayland risk.
8. **Phase 5, the stash ledger.** Guided stash scroll anchored on the header
   total, deltas afterwards, unsure flags. Local only. Decided 2026-10-11. Not
   started.
9. **Phase 6, progress from the screen.** Readers for the workshop and project
   pages. Needs screenshots of those pages at 3840x2160. Decided 2026-10-11.
   Not started.

## Open decisions for Jeffrey

- Which recycle row the game's own tooltip shows (field or home). One hover in
  game settles the chip label.
- Whether ended projects (Expedition 1 to 4) are shown, hidden, or hidden by
  default.
- Whether coin-funding project phases appear as a line.
- In-game check of recycle rows where the APIs and the wiki disagree (Anvil I
  to IV and Arpeggio I to IV look swapped, plus Wasp Driver and Synthesized
  Fuel).

## Known follow-ups

- Add `TOOLTIP_CHIPS=true` to `.env.example` (blocked for agents by permission
  settings).
- The ruff baseline on master has 9 lint errors and 7 unformatted files
  outside the new code.
- MetaForge now lists 1,109 items against 606 in the shipped database. The
  weekly workflow has been taking the wiki-only path.
