"""
Identify every visible inventory slot on one frame and turn it into badges.

identify_slots runs the pieces of phase 3 over one full-screen frame:

1. grid.find_slots finds every slot and whether it is filled. Empty slots
   are skipped.
2. Each filled slot crop goes through icon_embed's crop step and the
   DINOv2 embedder in one batch, then is ranked against the gallery by
   cosine similarity.
3. The top candidate is accepted when its similarity is at least
   MIN_SIMILARITY and it leads the best candidate of another base item in
   the top TOP_N by at least MIN_LEAD (tier siblings sit within a few
   hundredths of each other, so they do not count against the lead).
4. slot_marks.read_tier reads the tier numeral. For a tiered item the base
   name plus the read tier is looked up, so the numeral decides the tier and
   the embedding only the base item. Without a numeral the candidate's own
   name is looked up.
5. With read_stacks, slot_marks.read_stack reads the xN count of each
   identified slot (about 68 ms a slot with a count, so it is off by
   default).
6. slot_marks.bar_geometry finds the bottom bar of each square slot: its
   box, colour, type icon box and count or numeral box. Wide slots get none.

owned_totals adds up the stacks of each identified item over the slots.

distinct_gallery drops the gallery entries whose icon equals another
item's, which would otherwise never clear the lead rule, and returns the
dropped names as twins. A slot whose top name has twins is identified only
when the tier numeral settles which one it is.

badges_for turns the identified slots into badge dicts in the badge
surface's message format (badge_surface.encode_message): for every
identified item with a sell price, a patch over the type icon of a square
slot's bottom bar (a dark pill top left on wide slots) showing the stack's
worth. For an item an unfinished goal still needs, a have/need tag top left,
red with a red tile outline while the stash is short, green with a check mark
when it is enough.

Pure functions, no Tk or screen access. The embedder is only touched when a
frame has filled slots.
"""

import datetime
import re
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from .database import HOME
from .database import Database
from .database import Item
from .grid import Slot
from .grid import find_slots
from .icon_embed import EmbedCandidate
from .icon_embed import EmbedGallery
from .icon_embed import crop_square
from .icon_embed import default_embedder
from .slot_marks import BarGeometry
from .slot_marks import bar_geometry
from .slot_marks import read_stack
from .slot_marks import read_tier
from .tooltip_layout import Rect
from .verdict import Verdict
from .verdict import compute_verdict

DUPLICATE_SIMILARITY = 0.995
MIN_SIMILARITY = 0.65
MIN_LEAD = 0.04
TOP_N = 5
TIER = re.compile(r"\s+(I|II|III|IV|V)$")

# Badge look from Jeffrey's ARC Tracker screenshots. Gold is only the hovered
# cell's frame (cell_border.HOVER_GOLD), never a stash outline
SHORT_RED = "#ff5c4d"
ENOUGH_GREEN = "#4fd36b"
BORDER_WIDTH = 5
BORDER_RADIUS = 4
PRICE_PREFIX = "Φ "
NEED_CORNER = "tl"
PATCH_RADIUS = 6
TAG_CORNER = "bl"
# On wide slots (weapons) the game shows the ammo count bottom left, so the
# price goes top left there. A slot is wide from this width to height ratio
TAG_CORNER_WIDE = "tl"
WIDE_ASPECT = 1.25
TAG_FG = "#ffffff"
TAG_BG = "#141414d9"


class SlotItem(NamedTuple):
    """One filled slot and the item in it.

    item is the catalog item, or None when the match is below the confidence
    rule or the name does not resolve. guess is the top gallery name either
    way, confidence its similarity and lead its margin over the best other
    base item. tier is the read numeral, else the tier in the catalog name.
    stack is the read count, None when not read or not shown. bar is the
    bottom bar's geometry from slot_marks.bar_geometry, None on wide slots
    or when no bar is found.
    """

    rect: Rect
    panel: str
    item: Item | None
    tier: str | None
    stack: int | None
    confidence: float
    lead: float
    guess: str | None
    bar: BarGeometry | None = None


def base_name(name: str) -> str:
    """The name without a trailing tier numeral."""
    return TIER.sub("", name)


def name_tier(name: str) -> str | None:
    """The trailing tier numeral of a name, or None."""
    found = TIER.search(name)
    return found.group(1) if found else None


def base_lead(found: Sequence[EmbedCandidate]) -> float:
    """Top similarity minus the best candidate of another base item.

    When every candidate is a tier of the top item, the last one stands in
    for the best other item (a lower bound on the lead).
    """
    if not found:
        return 0.0
    top = base_name(found[0].name).casefold()
    others = [c.similarity for c in found[1:] if base_name(c.name).casefold() != top]
    rest = others[0] if others else found[-1].similarity
    return found[0].similarity - rest if len(found) > 1 else found[0].similarity


def distinct_gallery(
    gallery: EmbedGallery, threshold: float = DUPLICATE_SIMILARITY
) -> tuple[EmbedGallery, dict[str, tuple[str, ...]]]:
    """The gallery with one entry per distinct icon, and each kept name's twins.

    Many catalog items share one icon (a weapon and its blueprint, the
    variants of an amplified weapon, the security codes). Their embeddings
    are equal, so no slot can lead over its twins. Of each group whose
    similarity reaches threshold the shortest name is kept (the plain item
    before its blueprint or variant). The second value maps a kept name to
    the names it absorbed, for identify_slots' twins.
    """
    order = sorted(
        range(len(gallery)), key=lambda i: (len(gallery.names[i]), gallery.names[i])
    )
    kept: list[int] = []
    absorbed: dict[str, list[str]] = {}
    for i in order:
        if kept:
            similarity = gallery.vectors[kept] @ gallery.vectors[i]
            best = int(np.argmax(similarity))
            if float(similarity[best]) >= threshold:
                owner = gallery.names[kept[best]]
                absorbed.setdefault(owner, []).append(gallery.names[i])
                continue
        kept.append(i)
    kept.sort()
    distinct = EmbedGallery(
        tuple(gallery.item_ids[i] for i in kept),
        tuple(gallery.names[i] for i in kept),
        gallery.vectors[kept],
    )
    return distinct, {name: tuple(names) for name, names in absorbed.items()}


def _rank(vectors: np.ndarray, gallery: EmbedGallery) -> list[list[EmbedCandidate]]:
    similarity = vectors @ gallery.vectors.T
    ranked = []
    for row in similarity:
        order = np.argsort(-row, kind="stable")[:TOP_N]
        ranked.append(
            [
                EmbedCandidate(gallery.item_ids[i], gallery.names[i], float(row[i]))
                for i in order
            ]
        )
    return ranked


def _resolve(
    found: Sequence[EmbedCandidate],
    read: str | None,
    lookup: Callable[[str], Item | None],
    twins: Mapping[str, Sequence[str]],
) -> tuple[Item | None, str | None, float]:
    """The catalog item, its tier and the lead for one slot's candidates.

    When the top name has twins (items with the same icon), the item is kept
    only when the read numeral settles it: the top name is tiered, a numeral
    was read and no twin is tiered (an untiered blueprint or variant cannot
    show a numeral).
    """
    lead = base_lead(found)
    top = found[0]
    catalog_tier = name_tier(top.name)
    tier = read or catalog_tier
    if top.similarity < MIN_SIMILARITY or lead < MIN_LEAD:
        return None, tier, lead
    settled = catalog_tier is not None and read is not None
    alike = twins.get(top.name, ())
    if alike and not (settled and all(name_tier(t) is None for t in alike)):
        return None, tier, lead
    name = f"{base_name(top.name)} {read}" if settled else top.name
    return lookup(name), tier, lead


def identify_slots(
    rgb: np.ndarray,
    gallery: EmbedGallery,
    lookup: Callable[[str], Item | None],
    *,
    embedder=None,
    read_stacks: bool = False,
    slots: Iterable[Slot] | None = None,
    twins: Mapping[str, Sequence[str]] | None = None,
) -> list[SlotItem]:
    """Every filled slot on an RGB frame with its item, tier, stack and confidence.

    lookup resolves a catalog name (base plus tier) to an item, usually
    Database.lookup. embedder defaults to icon_embed.default_embedder().
    slots defaults to grid.find_slots(rgb). twins maps a gallery name to the
    names sharing its icon (from distinct_gallery), None means no groups.
    Slots come back in find_slots order.
    """
    if slots is None:
        slots = find_slots(rgb)
    filled = [s for s in slots if s.filled]
    if not filled or len(gallery) == 0:
        return []
    crops = [
        rgb[s.rect.top : s.rect.bottom, s.rect.left : s.rect.right] for s in filled
    ]
    vectors = (embedder or default_embedder()).embed([crop_square(c) for c in crops])
    result = []
    for slot, crop, found in zip(filled, crops, _rank(vectors, gallery), strict=True):
        tier_read = read_tier(crop)
        item, tier, lead = _resolve(
            found, tier_read.tier if tier_read else None, lookup, twins or {}
        )
        stack = read_stack(crop) if read_stacks and item is not None else None
        bar = None if is_wide(slot.rect) else bar_geometry(crop)
        result.append(
            SlotItem(
                slot.rect,
                slot.panel,
                item,
                tier,
                stack,
                round(found[0].similarity, 4),
                round(lead, 4),
                found[0].name,
                bar,
            )
        )
    return result


def owned_totals(slot_items: Iterable[SlotItem]) -> dict[str, int]:
    """How many of each identified item the slots show, keyed by item name.

    A slot counts its read stack, or 1 when no stack was read. Unidentified
    slots are skipped.
    """
    totals: dict[str, int] = {}
    for slot in slot_items:
        if slot.item is None:
            continue
        count = slot.stack if slot.stack is not None else 1
        totals[slot.item.name] = totals.get(slot.item.name, 0) + count
    return totals


def format_price(value: int) -> str:
    """A compact price: 810, 1.3k, 10k, 1.2M."""
    if value < 1000:
        return str(value)
    if value < 10_000:
        tenths = (value + 50) // 100
        if tenths < 100:
            whole, part = divmod(tenths, 10)
            return f"{whole}.{part}k" if part else f"{whole}k"
        return "10k"
    if value < 1_000_000:
        return f"{(value + 500) // 1000}k"
    tenths = (value + 50_000) // 100_000
    whole, part = divmod(tenths, 10)
    return f"{whole}.{part}M" if part else f"{whole}M"


def verdicts_for(
    db: Database,
    today: datetime.date | None = None,
    *,
    show_ended: bool = False,
) -> Callable[[SlotItem], Verdict]:
    """The verdict of a slot's item, as main.py's _verdict_for decides it.

    The owned count stays unknown (None), as in main.py, since one slot's
    stack is not the whole stash.
    """
    day = today or datetime.datetime.now().astimezone().date()
    progress = db.progress_map()

    def verdict(slot: SlotItem) -> Verdict:
        item = slot.item
        if item is None:
            return compute_verdict(None, [], {}, None, day, show_ended)
        home = [o for o in db.recycle_outputs(item.name) if o.location == HOME]
        return compute_verdict(
            item,
            db.needs_for(item.name),
            progress,
            None,
            day,
            show_ended,
            blueprint=db.blueprint_for(item.name),
            home_outputs=home,
        )

    return verdict


def verdict_kinds(
    db: Database,
    today: datetime.date | None = None,
    *,
    show_ended: bool = False,
) -> Callable[[SlotItem], str]:
    """The verdict kind of a slot's item, see verdicts_for."""
    verdict_of = verdicts_for(db, today, show_ended=show_ended)
    return lambda slot: verdict_of(slot).kind


def is_wide(rect: Rect) -> bool:
    """Whether a slot is wide (a weapon slot), from WIDE_ASPECT."""
    width, height = rect.right - rect.left, rect.bottom - rect.top
    return width >= WIDE_ASPECT * height


def tag_corner(rect: Rect) -> str:
    """The price pill's corner: top left on wide slots, else bottom left."""
    return TAG_CORNER_WIDE if is_wide(rect) else TAG_CORNER


def _rgb(hex_colour: str) -> list[int]:
    """'#rrggbb' as an [r, g, b] list, the wire format of outlines."""
    return [int(hex_colour[i : i + 2], 16) for i in (1, 3, 5)]


def patch_badge(slot: SlotItem, worth: int) -> dict | None:
    """The patch over a square slot's type icon, None when there is no icon box."""
    bar = slot.bar
    if bar is None or bar.icon_box is None:
        return None
    left, top = slot.rect.left, slot.rect.top
    return {
        "rect": [
            left + bar.bar_rect.left,
            top + bar.bar_rect.top,
            left + bar.bar_rect.right,
            top + bar.bar_rect.bottom,
        ],
        "icon_right": left + bar.icon_box.right,
        "patch_right": left + bar.patch_right,
        "colour": list(bar.colour),
        "text": PRICE_PREFIX + format_price(worth),
        "radius": PATCH_RADIUS,
        "kind": "patch",
    }


def badges_for(
    slot_items: Iterable[SlotItem],
    verdict_of: Callable[[SlotItem], Verdict],
) -> list[dict]:
    """Badge dicts for the identified slots, in the badge surface's format.

    Every identified item with a sell price gets its stack's worth (the unit
    price times the stack, 1 when unread): a patch on a square slot with a
    type icon, a dark pill top left on a wide slot (top right when the slot
    also has a need tag), and nothing on a square slot without a type icon.
    An item with active needs (a verdict with sources) gets a have/need tag
    top left, have from owned_totals and need the sum of the sources. Short
    stock is red with a red tile outline, enough is green with a check mark
    and no outline. Unidentified slots get no badge. verdict_of gives a
    slot's verdict, for example verdicts_for(db).
    """
    slots = list(slot_items)
    owned = owned_totals(slots)
    badges: list[dict] = []
    for slot in slots:
        item = slot.item
        if item is None:
            continue
        rect = [int(v) for v in slot.rect]
        need = sum(qty for _, qty in verdict_of(slot).sources)
        if need:
            have = owned.get(item.name, 0)
            short = have < need
            colour = SHORT_RED if short else ENOUGH_GREEN
            badges.append(
                {
                    "rect": rect,
                    "text": f"{have}/{need}",
                    "corner": NEED_CORNER,
                    "fg": colour,
                    "bg": TAG_BG,
                    "check": not short,
                    "outline": _rgb(colour),
                    "kind": "tag",
                }
            )
            if short:
                badges.append(
                    {
                        "rect": rect,
                        "color": colour,
                        "width": BORDER_WIDTH,
                        "radius": BORDER_RADIUS,
                        "kind": "border",
                    }
                )
        if not item.sell_price:
            continue
        worth = item.sell_price * (slot.stack if slot.stack is not None else 1)
        if is_wide(slot.rect):
            badges.append(
                {
                    "rect": rect,
                    "text": PRICE_PREFIX + format_price(worth),
                    "corner": "tr" if need else TAG_CORNER_WIDE,
                    "fg": TAG_FG,
                    "bg": TAG_BG,
                    "kind": "tag",
                }
            )
        elif (patch := patch_badge(slot, worth)) is not None:
            badges.append(patch)
    return badges
