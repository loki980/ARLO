# ruff: noqa: INP001, PLC2701
"""Measure the DINOv2 embedding match beside the hash on the labeled slot crops.

Usage:
    uv run --extra icons python scripts/icon_embed_eval.py <captures> <out folder>

The labeled set, the gallery (every arctracker item of the labeled items'
types) and the known bad labels are those of scripts/icon_hash_baseline.py,
imported from it. Every table is on the cleaned set (bad labels removed).

Writes to the out folder: results.txt (the side by side table, per family,
base item, wide slots, similarity thresholds and timings), labeled_embed.csv
(one row per crop with its top 5) and contact_embed.png (crops beside their
top-1 embedding match).
"""

import csv
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from icon_hash_baseline import BAD_LABELS
from icon_hash_baseline import _arctracker_items
from icon_hash_baseline import _contact_sheet
from icon_hash_baseline import _gallery_icons
from icon_hash_baseline import _label_crops
from icon_hash_baseline import _pct

from arc_helper import icon_embed
from arc_helper import icon_match
from arc_helper.config import get_settings

WEAPON_TYPES = {
    "Assault Rifle",
    "Battle Rifle",
    "Hand Cannon",
    "LMG",
    "Pistol",
    "SMG",
    "Shotgun",
    "Sniper Rifle",
}
MATERIAL_TYPES = {
    "Basic Material",
    "Nature",
    "Recyclable",
    "Refined Material",
    "Topside Material",
}
FAMILIES = ("weapons", "materials", "others")
TIER = re.compile(r"\s+(I|II|III|IV|V)$")
THRESHOLDS = (0.60, 0.65, 0.70, 0.75, 0.80)
MARGINS = (0.00, 0.02, 0.04, 0.06, 0.08)


def _family(item_type: str) -> str:
    if item_type in WEAPON_TYPES:
        return "weapons"
    if item_type in MATERIAL_TYPES:
        return "materials"
    return "others"


def _base(name: str) -> str:
    """The item name without its tier numeral or a ' Stencil' suffix, lowercase."""
    return TIER.sub("", name).lower().removesuffix(" stencil")


def _score(rows: list[dict], label_ids: dict, find) -> tuple[list[dict], list[float]]:
    """Run find(crop) on every row with a known id. Return scored rows and times."""
    scored, times = [], []
    for row in rows:
        item_id = label_ids.get(row["name"])
        if item_id is None:
            continue
        start = time.perf_counter()
        found = find(row["crop"])
        times.append(time.perf_counter() - start)
        ids = [c.item_id for c in found]
        scored.append(
            {
                **row,
                "id": item_id,
                "found": found,
                "top1": ids[:1] == [item_id],
                "top5": item_id in ids,
                "base1": _base(found[0].name) == _base(row["name"]),
                "wide": icon_match.is_wide(row["crop"]),
            }
        )
    return scored, times


def _line(label: str, rows: list[dict]) -> str:
    n = len(rows)
    return (
        f"  {label:<28} top-1 {_pct(sum(r['top1'] for r in rows), n):<16}"
        f" top-5 {_pct(sum(r['top5'] for r in rows), n):<16}"
        f" base top-1 {_pct(sum(r['base1'] for r in rows), n)}"
    )


def _table(title: str, scored: dict[str, list[dict]], family: dict) -> list[str]:
    lines = [f"== {title} =="]
    for method, rows in scored.items():
        lines.extend((f"{method}:", _line("all", rows)))
        lines.extend(
            _line(fam, [r for r in rows if family[r["id"]] == fam]) for fam in FAMILIES
        )
        lines.append(_line("wide slots", [r for r in rows if r["wide"]]))
    return lines


def _base_margin(found: list) -> float:
    """Top-1 similarity minus the best candidate of another base item in the top 5.

    With every top 5 entry a tier of the top-1 item, the 5th similarity stands
    in for the best other item (a lower bound on the margin).
    """
    top = _base(found[0].name)
    others = [c.similarity for c in found[1:] if _base(c.name) != top]
    return found[0].similarity - (others[0] if others else found[-1].similarity)


def _thresholds(rows: list[dict]) -> list[str]:
    """Coverage and precision of accepting the top-1 base item.

    Accept when the top-1 similarity is at least t and its margin over the best
    other base item (not another tier of itself) is at least m. The tier is
    left to the slot's tier mark, so a hit here is a base item hit.
    """
    lines = [
        (
            "== embedding: accept the top-1 base item when similarity >= t and "
            "margin over the best other base item >= m =="
        )
    ]
    for label, key in (("exact item", "top1"), ("base item", "base1")):
        for verdict, want in (("right", True), ("wrong", False)):
            sims = [r["found"][0].similarity for r in rows if r[key] == want]
            gaps = [_base_margin(r["found"]) for r in rows if r[key] == want]
            if sims:
                lines.append(
                    f"  {label} {verdict} ({len(sims)}): similarity min "
                    f"{min(sims):.3f} median {statistics.median(sims):.3f}, "
                    f"base margin min {min(gaps):.3f} median "
                    f"{statistics.median(gaps):.3f}"
                )
    lines.append(
        "  t, m: accepted, base right of accepted (precision), base right refused"
    )
    total = sum(r["base1"] for r in rows)
    for t in THRESHOLDS:
        for m in MARGINS:
            ok = [
                r
                for r in rows
                if r["found"][0].similarity >= t and _base_margin(r["found"]) >= m
            ]
            hits = sum(r["base1"] for r in ok)
            lines.append(
                f"  {t:.2f}, {m:.2f}: {len(ok)}/{len(rows)}, "
                f"{_pct(hits, len(ok))}, {total - hits}"
            )
    return lines


def _per_item(rows: list[dict]) -> list[str]:
    lines = ["== embedding per item: crops, top-1 hits, top-5 hits, top-1 answers =="]
    per_item: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_item[r["name"]].append(r)
    for name, rs in sorted(per_item.items()):
        answers: dict[str, int] = defaultdict(int)
        for r in rs:
            answers[f"{r['found'][0].name} {r['found'][0].similarity:.2f}"] += 1
        said = ", ".join(f"{n} x{c}" for n, c in sorted(answers.items()))
        lines.append(
            f"  {name}: {len(rs)}, {sum(r['top1'] for r in rs)}, "
            f"{sum(r['top5'] for r in rs)}, {said}"
        )
    return lines


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: icon_embed_eval.py <captures folder> <out folder>")
        return 2
    folder, out = Path(sys.argv[1]), Path(sys.argv[2])
    if not folder.is_dir():
        print(f"not a folder: {folder}")
        return 2
    out.mkdir(parents=True, exist_ok=True)

    labeled, _ = _label_crops(folder)
    clean = [r for r in labeled if not r["file"].startswith(tuple(BAD_LABELS))]
    cache_dir = get_settings().icon_cache_dir
    items = _arctracker_items(cache_dir)
    icons, label_ids, _ = _gallery_icons(labeled, items, cache_dir)
    family = {i["id"]: _family(i["type"]) for i in items}

    model = icon_embed.model_path()
    report = [
        (
            f"cleaned labeled crops: {len(clean)} of {len(labeled)}, "
            f"gallery {len(icons)} icons"
        ),
        (
            f"model: {icon_embed.MODEL_REPO}@{icon_embed.MODEL_REVISION[:12]} "
            f"{model.name}, {model.stat().st_size / 1e6:.1f} MB, CPU onnxruntime"
        ),
        f"wide slots turned {icon_match.WIDE_TILT:.0f} degrees clockwise",
        "",
    ]

    scored: dict[str, list[dict]] = {}
    timings: list[str] = []
    for method in icon_match.METHODS:
        gallery = icon_match.build_gallery(icons, method)
        scored[method], times = _score(
            clean, label_ids, lambda c, g=gallery: icon_match.match(c, g)
        )
        timings.append(f"  {method}: median {1000 * statistics.median(times):.2f} ms")

    embed_rows: list[dict] = []
    for pooling in icon_embed.POOLINGS:
        embedder = icon_embed.Embedder(pooling=pooling)
        cache_file = out / f"gallery_{pooling.replace('+', '_')}.npz"
        cache_file.unlink(missing_ok=True)
        start = time.perf_counter()
        gallery = icon_embed.build_gallery(icons, embedder, cache_file)
        built = time.perf_counter() - start
        start = time.perf_counter()
        icon_embed.build_gallery(icons, embedder, cache_file)
        loaded = time.perf_counter() - start
        cache_file.unlink()
        label = f"embed {pooling}"
        scored[label], times = _score(
            clean, label_ids, lambda c, g=gallery, e=embedder: icon_embed.match(c, g, e)
        )
        timings.append(
            f"  {label}: median {1000 * statistics.median(times):.1f} ms, max "
            f"{1000 * max(times):.1f} ms per crop, gallery build {built:.1f} s, "
            f"cached gallery load {1000 * loaded:.1f} ms"
        )
        if pooling == icon_embed.POOLING:
            embed_rows = scored[label]
            default_gallery = gallery

    report.extend(_table("cleaned set, side by side", scored, family))
    report.extend(("", "== runtime per crop (crop step, model, rank) =="))
    report.extend(timings)
    report.append("")

    # The wide crops without the tilt, to show what the tilt does
    tilt = icon_match.WIDE_TILT
    icon_match.WIDE_TILT = 0.0
    wide = [r for r in clean if icon_match.is_wide(r["crop"])]
    untilted, _ = _score(
        wide,
        label_ids,
        lambda c: icon_embed.match(c, default_gallery, icon_embed.default_embedder()),
    )
    dhash_gallery = icon_match.build_gallery(icons, "dhash")
    untilted_hash, _ = _score(
        wide, label_ids, lambda c: icon_match.match(c, dhash_gallery)
    )
    icon_match.WIDE_TILT = tilt
    report.extend(
        (
            "== wide slots with no tilt (the step 1 crop) ==",
            _line(f"embed {icon_embed.POOLING}", untilted),
            _line("dhash", untilted_hash),
            "",
        )
    )
    report.extend(_thresholds(embed_rows))
    report.append("")
    report.extend(_per_item(embed_rows))

    for r in embed_rows:
        r["found"] = [
            icon_match.Candidate(c.item_id, c.name, round(100 * c.similarity))
            for c in r["found"]
        ]
    _contact_sheet(embed_rows, icons, out / "contact_embed.png")
    with (out / "labeled_embed.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "item", "top1", "top5", "answers (similarity x100)"])
        for r in embed_rows:
            answers = "; ".join(f"{c.name} {c.distance}" for c in r["found"])
            writer.writerow([r["file"], r["name"], r["top1"], r["top5"], answers])
    text = "\n".join(report)
    (out / "results.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
