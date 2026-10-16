# ruff: noqa: INP001
"""Read the stash and loadout headers on saved captures and fixtures.

Usage: uv run python scripts/eval_inventory_header.py --out <folder>
           --expected <csv> --captures <folder> [fixture.png ...]
           [--old <csv>]

Runs read_stash_header and read_loadout_header on every file named in the
expected CSV (columns file, stash, loadout, written as used/capacity/value,
weight/cap/value or none, by viewing each capture) found in the captures
folder, then on each fixture path. The old reads come from the --old CSV
(default: the ledger-s1 header_reads.csv, made with the fixed-height reader)
and are matched by file name, fixtures have none. Writes real-data.txt in the
output folder with one line per file (old, new and expected for each reader)
and the match counts. The two fixtures are checked against FIXTURE_EXPECTED.
The captures folder is only read.
"""

import argparse
import csv
import re
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

from arc_helper.grid import EQUIPMENT
from arc_helper.grid import STASH
from arc_helper.grid import find_slots
from arc_helper.inventory_header import StashHeader
from arc_helper.inventory_header import read_loadout_header
from arc_helper.inventory_header import read_stash_header

OLD_READS = Path(
    "/home/jefflamb/workspace/ARLO/.claude/worktrees/overlay-roadmap/.claude"
    "/handoff/ledger-s1/attempt-1/header_reads.csv"
)
# Header values read by eye on the two fixtures (neither shows a stash capacity)
FIXTURE_EXPECTED = {
    "raid-inventory-4k.png": ("none", "39.5/90.0/116685"),
    "base-stiletto-4k.png": ("none", "32.7/90.0/61737"),
}


def _text(read) -> str:
    """A read as used/capacity/value or weight/cap/value, none for None."""
    if read is None:
        return "none"
    if isinstance(read, StashHeader):
        return f"{read.used}/{read.capacity}/{read.value}"
    return f"{read.weight:.1f}/{read.weight_cap:.1f}/{read.value}"


def _old_text(old: dict[str, tuple[str, str]], name: str) -> str:
    """The old reads of a name as 'stash loadout', n/a when it has none."""
    if name not in old:
        return "n/a n/a"
    parts = []
    for cell in old[name]:
        numbers = re.findall(r"=(\d[\d.]*)", cell)
        parts.append("/".join(numbers) if numbers else "none")
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--old", type=Path, default=OLD_READS)
    parser.add_argument("fixtures", nargs="*", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with args.expected.open() as handle:
        cases = [
            (args.captures / r["file"], r["stash"], r["loadout"])
            for r in csv.DictReader(handle)
        ]
    with args.old.open() as handle:
        old = {r["file"]: (r["stash"], r["loadout"]) for r in csv.DictReader(handle)}
    cases += [(p, *FIXTURE_EXPECTED[p.name]) for p in args.fixtures]

    lines, times = [], []
    kept = fixed = lost = 0
    for path, want_stash, want_loadout in cases:
        rgb = np.array(Image.open(path).convert("RGB"))
        slots = find_slots(rgb)
        stash = [s for s in slots if s.panel == STASH]
        equipment = [s for s in slots if s.panel == EQUIPMENT]
        start = time.perf_counter()
        new_stash = _text(read_stash_header(rgb, stash))
        new_loadout = _text(read_loadout_header(rgb, equipment))
        times.append((time.perf_counter() - start) * 1000)
        old_stash, old_loadout = _old_text(old, path.name).split()
        ok = new_stash == want_stash and new_loadout == want_loadout
        if path.name in old:
            was = old_stash == want_stash and old_loadout == want_loadout
            kept += was and ok
            fixed += ok and not was
            lost += was and not ok
        lines.append(
            f"{path.name}\n"
            f"  stash   old {old_stash}  new {new_stash}  expected {want_stash}\n"
            f"  loadout old {old_loadout}  new {new_loadout}  expected {want_loadout}"
            f"  {'MATCH' if ok else 'DIFFERS'}"
        )
    lines.extend(
        [
            (
                f"captures matching before and after {kept}, "
                f"newly matching {fixed}, regressed {lost}"
            ),
            f"frames {len(times)}, median ms {statistics.median(times):.0f}",
        ]
    )
    report = "\n".join(lines)
    (args.out / "real-data.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
