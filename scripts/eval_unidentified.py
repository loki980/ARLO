"""
Which filled stash slots stay unidentified, and why, on real captures.

Runs stash_scan.identify_slots (no slot cache, read_stacks on) with the
production gallery, twins and Database.lookup (main.load_stash_gallery) on
every capture frame and on video frames every --every seconds. The
embedding vectors identify_slots computes are recorded, so each slot's
ranked candidates, tier read and blueprint tile are known and its outcome
can be explained by replaying stash_scan._resolve's branches.

Unidentified slots are deduplicated by stash_scan.look_signature, with the
number of observations of each look. Each look gets a crop (and a crop with
some context around it) in <out>/unidentified/, its top three guesses and
the reason the rule gave none.

Hand labels are read from <out>/labels.json when present:
{"<look id>": {"name": "<catalog name or ?>", "cause": "<cause group>"}}.
They drive the cause counts and the correct, wrong and uncertain counts of
each threshold proposal. A proposal is replayed on every observation, and
the observations it newly accepts or newly rejects are counted.

usage: eval_unidentified.py --frames GLOB [--video FILE] --out DIR [--every S]
"""

import argparse
import csv
import glob
import hashlib
import json
import logging
import subprocess
import sys
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image

from arc_helper import stash_scan
from arc_helper.badge_helper import CHECK_GREEN
from arc_helper.config import get_settings
from arc_helper.database import Database
from arc_helper.grid import find_slots
from arc_helper.icon_embed import EmbedCandidate
from arc_helper.icon_embed import crop_square
from arc_helper.icon_embed import default_embedder
from arc_helper.main import STASH_EXCLUDED_TYPES
from arc_helper.main import STASH_ITEMS_FILE
from arc_helper.main import load_stash_gallery
from arc_helper.slot_marks import bar_geometry
from arc_helper.slot_marks import is_blueprint_tile
from arc_helper.slot_marks import read_stack
from arc_helper.slot_marks import read_tier
from arc_helper.stash_scan import ENOUGH_GREEN
from arc_helper.stash_scan import FILL_CODES
from arc_helper.stash_scan import SHORT_RED
from arc_helper.stash_scan import identify_slots
from arc_helper.stash_scan import is_wide
from arc_helper.stash_scan import look_signature

CONTEXT = 0.35  # context crop margin, as a share of the slot side
# ARLO's own colours: every price fill, the tag outlines and the check
ARLO_CODES = np.union1d(
    FILL_CODES,
    np.array([int(c[1:], 16) for c in (SHORT_RED, ENOUGH_GREEN, CHECK_GREEN)]),
)
# The icon zone: from this share of the slot height (below the tag band) to
# stash_scan.SIG_LOOK_BOTTOM (above the bar), inside a border band
COVER_TOP = 0.25
COVER_MIN = 30  # ARLO-coloured pixels in the icon zone that count as covered
SWEEPS = [
    ("lead >= 0.03", 0.65, 0.03),
    ("lead >= 0.02", 0.65, 0.02),
    ("lead >= 0.01", 0.65, 0.01),
    ("lead >= 0.00", 0.65, 0.0),
    ("similarity >= 0.60", 0.60, 0.04),
    ("similarity >= 0.55", 0.55, 0.04),
    ("similarity >= 0.60, lead >= 0.02", 0.60, 0.02),
]


class Recording:
    """An embedder that keeps the last batch of vectors it returned."""

    def __init__(self, inner):
        self.inner = inner
        self.last: np.ndarray | None = None

    def embed(self, squares):
        self.last = self.inner.embed(squares)
        return self.last


class Obs(NamedTuple):
    source: str
    frame: str
    rect: tuple
    look: str
    found: list[EmbedCandidate]
    tier_read: str | None
    blueprint: bool | None
    item: str | None
    tier: str | None
    stack: int | None
    confidence: float
    lead: float
    reason: str
    bar_rgb: tuple | None
    panel: str
    vector: np.ndarray
    cover: int | None
    band: int | None


def arlo_pixels(raw: np.ndarray) -> tuple[int, int]:
    """ARLO-coloured pixels in a lossless crop: over the icon, and in the tag band.

    Exact colour matches only, so it is used on PNG captures and not on video.
    A price fill or tag outline in the icon zone is a badge drawn for another
    slot position (stale during a scroll), since ARLO draws its own badges
    only in the top band and on the bar.
    """
    h, w = raw.shape[:2]
    b = stash_scan._band(raw)
    codes = (
        (raw[..., 0].astype(np.int32) << 16)
        | (raw[..., 1].astype(np.int32) << 8)
        | raw[..., 2]
    )
    hit = np.isin(codes, ARLO_CODES)
    top = round(COVER_TOP * h)
    bottom = round(stash_scan.SIG_LOOK_BOTTOM * h)
    cover = int(hit[top:bottom, b : w - b].sum())
    band = int(hit[:top, b : w - b].sum())
    return cover, band


def look_id(crop: np.ndarray) -> str:
    return hashlib.sha1(repr(look_signature(crop)).encode()).hexdigest()[:10]


def reason_for(found, read, lookup, twins, blueprint) -> str:
    """Which branch of stash_scan._resolve gave the slot its outcome."""
    top = found[0]
    catalog_tier = stash_scan.name_tier(top.name)
    settled = catalog_tier is not None and read is not None
    lead = stash_scan.base_lead(found, tiered_only=settled)
    if top.similarity < stash_scan.MIN_SIMILARITY:
        return "low_similarity"
    if lead < stash_scan.MIN_LEAD:
        return "low_lead"
    alike = twins.get(top.name, ())
    picked = (
        stash_scan._tile_twin(top.name, alike, blueprint=blueprint) if alike else None
    )
    if picked is not None and not stash_scan._is_blueprint_name(picked):
        if stash_scan.name_tier(picked) is not None and read is None:
            return "tier_unread"
    if picked is None and alike:
        if not (settled and all(stash_scan.name_tier(t) is None for t in alike)):
            return "twins_unsettled"
    item, _, _ = stash_scan._resolve(found, read, lookup, twins, blueprint=blueprint)
    return "identified" if item is not None else "lookup_none"


def resolve_with(obs: Obs, lookup, twins, min_sim: float, min_lead: float):
    old = stash_scan.MIN_SIMILARITY, stash_scan.MIN_LEAD
    stash_scan.MIN_SIMILARITY, stash_scan.MIN_LEAD = min_sim, min_lead
    try:
        item, _, _ = stash_scan._resolve(
            obs.found, obs.tier_read, lookup, twins, blueprint=obs.blueprint
        )
    finally:
        stash_scan.MIN_SIMILARITY, stash_scan.MIN_LEAD = old
    return item.name if item is not None else None


def video_frames(path: Path, every: float):
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path),
        ],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    width, height = (int(v) for v in probe.stdout.strip().split(","))
    size = width * height * 3
    proc = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps=1/{every}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:",
        ],
        stdout=subprocess.PIPE,
    )  # fmt: skip
    n = 0
    while True:
        data = proc.stdout.read(size)
        if len(data) < size:
            break
        yield (
            f"t={n * every:06.1f}s",
            np.frombuffer(data, np.uint8).reshape(height, width, 3),
        )
        n += 1
    proc.wait()


def save_crops(rgb: np.ndarray, rect: tuple, folder: Path, name: str) -> None:
    left, top, right, bottom = rect
    Image.fromarray(rgb[top:bottom, left:right]).save(folder / f"{name}.png")
    m = round(CONTEXT * (right - left))
    h, w = rgb.shape[:2]
    box = (max(0, left - m), max(0, top - m), min(w, right + m), min(h, bottom + m))
    Image.fromarray(rgb[box[1] : box[3], box[0] : box[2]]).save(
        folder / f"{name}_context.png"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--video")
    parser.add_argument("--out", required=True)
    parser.add_argument("--every", type=float, default=2.0)
    args = parser.parse_args()
    out = Path(args.out)
    crops_dir = out / "unidentified"
    crops_dir.mkdir(parents=True, exist_ok=True)
    logging.getLogger("arc_helper").setLevel(logging.ERROR)
    labels_path = out / "labels.json"
    labels = (
        json.loads(labels_path.read_text(encoding="utf-8"))
        if labels_path.is_file()
        else {}
    )

    cache_dir = get_settings().icon_cache_dir
    gallery, twins = load_stash_gallery(cache_dir)
    db = Database()
    lookup = db.lookup
    embedder = Recording(default_embedder())
    lines: list[str] = []

    listing = json.loads((cache_dir / STASH_ITEMS_FILE).read_text(encoding="utf-8"))
    excluded = [i for i in listing["items"] if i.get("type") in STASH_EXCLUDED_TYPES]
    no_icon = [
        i
        for i in listing["items"]
        if i.get("type") not in STASH_EXCLUDED_TYPES
        and not (cache_dir / f"{i['id']}.png").is_file()
    ]
    lines.append(
        f"gallery: {len(gallery)} distinct icons, {sum(map(len, twins.values()))} "
        f"twins absorbed, catalog listing {len(listing['items'])} items, "
        f"{len(excluded)} of excluded types, {len(no_icon)} without a cached icon"
    )
    for item in no_icon:
        lines.append(f"  no icon: {item['name']['en']} ({item.get('type')})")
    lines.append(
        f"rule: MIN_SIMILARITY={stash_scan.MIN_SIMILARITY} "
        f"MIN_LEAD={stash_scan.MIN_LEAD} TOP_N={stash_scan.TOP_N}"
    )

    observations: list[Obs] = []
    first_frame: set[str] = set()
    per_source = defaultdict(Counter)

    def run(source: str, frame: str, rgb: np.ndarray) -> None:
        slots = find_slots(rgb)
        filled = [s for s in slots if s.filled]
        per_source[source]["frames"] += 1
        if not filled:
            per_source[source]["frames without filled slots"] += 1
            return
        embedder.last = None
        items = identify_slots(
            rgb,
            gallery,
            lookup,
            embedder=embedder,
            read_stacks=True,
            slots=slots,
            twins=twins,
            cache=None,
        )
        ranked = stash_scan._rank(embedder.last, gallery)
        for slot, found, result, vector in zip(
            filled, ranked, items, embedder.last, strict=True
        ):
            crop = stash_scan._slot_crop(rgb, slot.rect)
            r = slot.rect
            raw = rgb[r.top : r.bottom, r.left : r.right]
            tier_read = read_tier(crop)
            read = tier_read.tier if tier_read else None
            blueprint = is_blueprint_tile(crop)
            reason = reason_for(found, read, lookup, twins, blueprint)
            if (reason == "identified") != (result.item is not None):
                reason += "_MISMATCH"
            look = look_id(crop)
            bar = None if is_wide(slot.rect) else bar_geometry(crop)
            stack = result.stack
            if result.item is None:
                stack = read_stack(crop)
            obs = Obs(
                source,
                frame,
                tuple(slot.rect),
                look,
                found,
                read,
                blueprint,
                result.item.name if result.item else None,
                result.tier,
                stack,
                result.confidence,
                result.lead,
                reason,
                tuple(bar.colour) if bar else None,
                slot.panel,
                vector,
                *(arlo_pixels(raw) if source == "capture" else (None, None)),
            )
            observations.append(obs)
            per_source[source]["filled"] += 1
            per_source[source]["identified" if obs.item else "unidentified"] += 1
            if obs.item is None and look not in first_frame:
                first_frame.add(look)
                save_crops(rgb, tuple(slot.rect), crops_dir, look)
                Image.fromarray(crop_square(crop)).save(crops_dir / f"{look}_model.png")

    paths = sorted(glob.glob(args.frames))
    for path in paths:
        try:
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"))
        except Exception as exc:  # noqa: BLE001 - any unreadable file is counted
            per_source["capture"]["unreadable"] += 1
            lines.append(f"unreadable: {path} ({type(exc).__name__})")
            continue
        run("capture", Path(path).name, rgb)
        print(f"capture {Path(path).name}", file=sys.stderr)
    if args.video:
        for name, rgb in video_frames(Path(args.video), args.every):
            run("video", name, rgb)
            print(f"video {name}", file=sys.stderr)

    lines.append("")
    lines.append(f"capture files matched: {len(paths)}")
    for source, counts in per_source.items():
        parts = ", ".join(f"{k} {v}" for k, v in counts.items())
        lines.append(f"{source}: {parts}")
    unident = [o for o in observations if o.item is None]
    frames_with = Counter((o.source, o.frame) for o in unident)
    lines.append(
        f"unidentified observations {len(unident)} of {len(observations)} filled, "
        f"in {len(frames_with)} frames, "
        f"{len({o.look for o in unident})} distinct looks"
    )
    lines.append(
        "mismatches between replayed reason and identify_slots: "
        f"{sum(o.reason.endswith('_MISMATCH') for o in observations)}"
    )

    lines.append("")
    lines.append(
        f"ARLO COVER (captures only, >= {COVER_MIN} ARLO-coloured px in the icon zone)"
    )
    for panel in sorted({o.panel for o in observations}):
        for state in ("identified", "unidentified"):
            group = [
                o
                for o in observations
                if o.source == "capture"
                and o.panel == panel
                and (o.item is not None) == (state == "identified")
            ]
            covered = sum(o.cover >= COVER_MIN for o in group)
            lines.append(f"  {panel} {state}: covered {covered} of {len(group)}")
    capture_un = [o for o in unident if o.source == "capture"]
    lines.append(
        f"  all panels unidentified: covered "
        f"{sum(o.cover >= COVER_MIN for o in capture_un)} of {len(capture_un)}"
    )

    # Per look table
    by_look: dict[str, list[Obs]] = defaultdict(list)
    for o in unident:
        by_look[o.look].append(o)
    order = sorted(by_look, key=lambda k: (-len(by_look[k]), k))
    lines.append("")
    lines.append("UNIDENTIFIED LOOKS (most observed first)")
    for k in order:
        group = by_look[k]
        o = group[0]
        sources = Counter(g.source for g in group)
        reasons = Counter(g.reason for g in group)
        top3 = "  |  ".join(f"{c.name} {c.similarity:.4f}" for c in o.found[:3])
        sims = [g.confidence for g in group]
        leads = [g.lead for g in group]
        stacks = Counter(g.stack for g in group)
        label = labels.get(k, {})
        lines.append(
            f"{k}  obs {len(group)} ({', '.join(f'{s} {n}' for s, n in sources.items())})"
            f"  reason {dict(reasons)}"
        )
        lines.append(
            f"    first {o.source} {o.frame} panel {o.panel} rect {o.rect}"
            f"  size {o.rect[2] - o.rect[0]}x{o.rect[3] - o.rect[1]}"
        )
        lines.append(f"    top3: {top3}")
        lines.append(
            f"    similarity {min(sims):.4f}..{max(sims):.4f}"
            f"  lead {min(leads):.4f}..{max(leads):.4f}"
            f"  tier read {Counter(g.tier_read for g in group)}"
            f"  blueprint tile {Counter(g.blueprint for g in group)}"
        )
        lines.append(f"    stack {dict(stacks)}  bar colour {o.bar_rgb}")
        covers = [g.cover for g in group if g.cover is not None]
        if covers:
            lines.append(
                f"    ARLO pixels over the icon (captures): {min(covers)}..{max(covers)}"
                f", covered {sum(c >= COVER_MIN for c in covers)} of {len(covers)}"
            )
        twins_of = stash_scan_twins(o.found[0].name, twins)
        if twins_of:
            lines.append(f"    twins of top: {twins_of}")
        lines.append(
            f"    label: {label.get('name', '(none)')}  cause: {label.get('cause', '(none)')}"
        )
        true_name = label.get("name")
        if true_name and true_name != "?":
            lines.append(
                f"    true name in gallery: {true_rank(o, true_name, gallery, twins)}"
            )

    # Cause counts
    lines.append("")
    lines.append("REPLAYED RULE REASON (observations, looks)")
    reason_obs = Counter(o.reason for o in unident)
    reason_looks = Counter(by_look[k][0].reason for k in order)
    for r, n in reason_obs.most_common():
        lines.append(f"  {r}: {n} observations, {reason_looks[r]} looks (first obs)")
    lines.append("")
    lines.append("HAND-LABELLED CAUSE (observations, looks)")
    cause_obs, cause_looks = Counter(), Counter()
    for k in order:
        cause = labels.get(k, {}).get("cause", "unlabelled")
        cause_obs[cause] += len(by_look[k])
        cause_looks[cause] += 1
    for c, n in cause_obs.most_common():
        lines.append(f"  {c}: {n} observations, {cause_looks[c]} looks")
    lines.append("")
    lines.append("HAND-LABELLED ITEM (observations, looks)")
    name_obs, name_looks = Counter(), Counter()
    for k in order:
        name = labels.get(k, {}).get("name", "unlabelled")
        name_obs[name] += len(by_look[k])
        name_looks[name] += 1
    for c, n in name_obs.most_common():
        lines.append(f"  {c}: {n} observations, {name_looks[c]} looks")

    # Per frame, and whether the top guess already names the labelled item
    lines.append("")
    lines.append("UNIDENTIFIED PER FRAME (median and max, by panel)")
    for source in ("capture", "video"):
        frames = {o.frame for o in observations if o.source == source}
        for panel in ("stash", "equipment"):
            counts = [
                sum(
                    1
                    for o in unident
                    if o.source == source and o.frame == f and o.panel == panel
                )
                for f in frames
            ]
            if counts:
                lines.append(
                    f"  {source} {panel}: frames {len(counts)}, median "
                    f"{float(np.median(counts)):g}, max {max(counts)}, "
                    f"frames with none {counts.count(0)}"
                )
    lines.append("")
    lines.append("TOP GUESS VERSUS HAND LABEL, by cause (observations)")
    agree = defaultdict(Counter)
    for o in unident:
        label = labels.get(o.look, {})
        name, cause = label.get("name"), label.get("cause", "unlabelled")
        top = o.found[0].name
        if o.tier_read and stash_scan.name_tier(top):
            top = f"{stash_scan.base_name(top)} {o.tier_read}"
        if not name or name == "?":
            agree[cause]["label uncertain"] += 1
        elif name.casefold() == top.casefold():
            agree[cause]["top guess is the item"] += 1
        else:
            agree[cause]["top guess is another item"] += 1
    for cause, counts in sorted(agree.items(), key=lambda kv: -sum(kv[1].values())):
        parts = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        lines.append(f"  {cause}: {parts}")

    # Threshold proposals
    lines.append("")
    lines.append("THRESHOLD PROPOSALS (replayed on every observation)")
    for title, min_sim, min_lead in SWEEPS:
        newly: list[tuple[Obs, str]] = []
        rejected = changed = 0
        for o in observations:
            now = resolve_with(o, lookup, twins, min_sim, min_lead)
            if o.item is None and now is not None:
                newly.append((o, now))
            elif o.item is not None and now is None:
                rejected += 1
            elif o.item is not None and now != o.item:
                changed += 1
        verdicts = Counter()
        look_verdicts: dict[str, str] = {}
        for o, now in newly:
            name = labels.get(o.look, {}).get("name")
            if not name or name == "?":
                v = "uncertain"
            elif name.casefold() == now.casefold():
                v = "correct"
            else:
                v = "wrong"
            verdicts[v] += 1
            look_verdicts[o.look] = f"{now} -> {v}"
        lines.append(
            f"  {title}: newly accepted {len(newly)} observations "
            f"({len(look_verdicts)} looks): correct {verdicts['correct']}, "
            f"wrong {verdicts['wrong']}, uncertain {verdicts['uncertain']}"
            f"  | accepted now rejected {rejected}, accepted changed item {changed}"
        )
        for look, text in sorted(look_verdicts.items()):
            lines.append(f"      {look}: {text}")

    lines.extend(tier_rule_lines(observations, lookup, twins, labels))
    write_observations(out, observations)
    text = "\n".join(lines) + "\n"
    (out / "real-data.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


def tier_rule(o: Obs, lookup, twins) -> str | None:
    """The proposed tier rule: a tiered item whose only twin is its own tier's
    blueprint ("Compensator II" and "Compensator II Blueprint") takes the
    catalog tier when no numeral is read, since that pair names one tier.
    """
    if o.reason != "tier_unread":
        return None
    top = o.found[0].name
    alike = twins.get(top, ())
    tier = stash_scan.name_tier(top)
    own = f"{top} blueprint".casefold()
    if tier is None or len(alike) != 1 or alike[0].casefold() != own:
        return None
    item = lookup(top)
    return item.name if item is not None else None


def tier_rule_lines(observations, lookup, twins, labels) -> list[str]:
    newly = [(o, tier_rule(o, lookup, twins)) for o in observations]
    newly = [(o, name) for o, name in newly if name is not None]
    verdicts = Counter()
    looks = {}
    for o, name in newly:
        label = labels.get(o.look, {}).get("name")
        if not label or label == "?":
            v = "uncertain"
        elif label.casefold() == name.casefold():
            v = "correct"
        else:
            v = "wrong"
        verdicts[v] += 1
        looks[o.look] = f"{name} -> {v}"
    tier_unread = sum(o.reason == "tier_unread" for o in observations)
    out = [
        "",
        "TIER RULE PROPOSAL (catalog tier when the only twin is that tier's blueprint)",
        f"  tier_unread observations {tier_unread}, newly accepted {len(newly)} "
        f"({len(looks)} looks): correct {verdicts['correct']}, wrong "
        f"{verdicts['wrong']}, uncertain {verdicts['uncertain']}"
        "  | accepted now rejected 0 (it only acts on tier_unread slots)",
    ]
    out.extend(f"      {look}: {text}" for look, text in sorted(looks.items()))
    return out


def true_rank(o: Obs, name: str, gallery, twins) -> str:
    """The rank and similarity of the gallery entry that has name (or absorbed it)."""
    key = name.casefold()
    owner = next(
        (
            kept
            for kept in gallery.names
            if kept.casefold() == key
            or key in (t.casefold() for t in twins.get(kept, ()))
        ),
        None,
    )
    if owner is None:
        return "not in the gallery"
    row = gallery.vectors @ o.vector
    i = gallery.names.index(owner)
    rank = int((row > row[i]).sum()) + 1
    via = "" if owner.casefold() == key else f" (as twin of {owner})"
    return f"rank {rank} similarity {row[i]:.4f}{via}"


def write_observations(out: Path, observations: list[Obs]) -> None:
    """Every observation as a CSV row, for step 2 and for labelling."""
    with (out / "observations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source", "frame", "panel", "left", "top", "right", "bottom",
                "look", "item", "tier", "stack", "reason", "confidence", "lead",
                "tier_read", "blueprint_tile", "bar_rgb", "top1", "sim1",
                "top2", "sim2", "top3", "sim3", "arlo_cover_px", "arlo_band_px",
            ]
        )  # fmt: skip
        for o in observations:
            top = [v for c in o.found[:3] for v in (c.name, f"{c.similarity:.4f}")]
            writer.writerow(
                [
                    o.source, o.frame, o.panel, *o.rect, o.look, o.item or "",
                    o.tier or "", "" if o.stack is None else o.stack, o.reason,
                    o.confidence, o.lead, o.tier_read or "",
                    "" if o.blueprint is None else o.blueprint,
                    "" if o.bar_rgb is None else ";".join(map(str, o.bar_rgb)),
                    *top, "" if o.cover is None else o.cover,
                    "" if o.band is None else o.band,
                ]
            )  # fmt: skip
    looks = [o for o in observations if o.item is None]
    np.savez(
        out / "unidentified_vectors.npz",
        looks=np.array([o.look for o in looks]),
        vectors=np.array([o.vector for o in looks]),
    )


def stash_scan_twins(name: str, twins) -> tuple:
    return tuple(twins.get(name, ()))


if __name__ == "__main__":
    sys.exit(main())
