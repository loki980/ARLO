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
   hundredths of each other, so they do not count against the lead). When
   the top name is tiered and a tier numeral is read, untiered candidates
   (an amplified variant, a blueprint) do not count against it either: they
   cannot show a numeral.
4. slot_marks.read_tier reads the tier numeral. For a tiered item the base
   name plus the read tier is looked up, so the numeral decides the tier and
   the embedding only the base item. Without a numeral the candidate's own
   name is looked up.
5. With read_stacks, slot_marks.read_stack reads the xN count of each
   identified slot (about 68 ms a slot with a count, so it is off by
   default).
6. slot_marks.bar_geometry finds the bottom bar of each square slot: its
   box, colour, type icon box and count or numeral box. Wide slots get none.

With a SlotCache, identify_slots first looks each slot up by slot_signature,
a hash of the crop without ARLO's own drawings (see slot_signature), which
needs the slot's bar from step 6. A hit reuses the slot's item, tier, stack,
confidence and bar without embedding or reading the tier or stack, so slots
already seen, also after a scroll, cost little more than find_slots and the
bar. Only the misses are embedded, in one call.

owned_totals adds up the stacks of each identified item over the slots.

distinct_gallery drops the gallery entries whose icon equals another
item's, which would otherwise never clear the lead rule, and returns the
dropped names as twins. A slot whose top name has twins is identified only
when the tier numeral settles which one it is.

badges_for turns the identified slots into badge dicts in the badge
surface's message format (badge_surface.encode_message): for every
identified item with a sell price, a patch over the type icon of a square
slot's bottom bar (a pill top left on wide slots) showing the stack's
worth, both filled by worth_colours so the cheap slots and those worth
WORTH_HIGH or more show at a glance. For an item an unfinished goal still
needs, a have/need tag top left, red with a red tile outline while the stash
is short, green with a check mark when it is enough.

Pure functions, no Tk or screen access. The embedder is only touched when a
frame has filled slots.
"""

import datetime
import hashlib
import json
import math
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from .badge_helper import PATCH_TEXT_RGB
from .badge_helper import TAG_TEXT_MAX
from .badge_helper import TAG_TEXT_MIN
from .badge_helper import TAG_TEXT_SHARE
from .config import logger
from .database import HOME
from .database import Database
from .database import Item
from .grid import Slot
from .grid import find_slots
from .icon_embed import EmbedCandidate
from .icon_embed import EmbedGallery
from .icon_embed import crop_square
from .icon_embed import default_embedder
from .slot_marks import _ICON_START_MAX
from .slot_marks import _INK_ABOVE
from .slot_marks import _RIGHT_HALF
from .slot_marks import BarGeometry
from .slot_marks import _Bar
from .slot_marks import _luminance
from .slot_marks import _runs
from .slot_marks import _word_box
from .slot_marks import _words
from .slot_marks import bar_geometry
from .slot_marks import is_blueprint_tile
from .slot_marks import read_stack
from .slot_marks import read_tier
from .tooltip_layout import Rect
from .verdict import Verdict
from .verdict import compute_verdict

DUPLICATE_SIMILARITY = 0.995
MIN_SIMILARITY = 0.65
MIN_LEAD = 0.04
# Deep enough that the lead is measured against the best other base item,
# not a stand-in, also when untiered candidates are passed over
TOP_N = 10
TIER = re.compile(r"\s+(I|II|III|IV|V)$")
BLUEPRINT_SUFFIX = " blueprint"  # casefolded, a blueprint's catalog name ends so

# The slot cache: entries kept, least recently used dropped first
SLOT_CACHE_SIZE = 4096
# Part of the saved slot cache's fingerprint. Bump it whenever slot_signature
# or look_signature change, so entries keyed the old way are dropped on load
SLOT_CACHE_VERSION = 1
# After the first save, the slot cache is saved at most once per this many
# seconds, and stop waits at most SLOT_CACHE_STOP_WAIT_S for the last save
SLOT_CACHE_SAVE_S = 30.0
SLOT_CACHE_STOP_WAIT_S = 2.0
# What slot_signature leaves out, as shares of the slot unless named. The
# border band (the red outline and the hovered frame's edge), at least
# SIG_BORDER_MIN px
SIG_BORDER = 0.05
SIG_BORDER_MIN = 6
# The top band where tags go, in tag text heights (a tag pill with its inset
# and outline is about 1.9 of them tall)
SIG_TAG_ROWS = 2.2
# Rows above a square slot's bar also left out with the patch part of it
SIG_BAR_MARGIN = 2
# look_signature leaves out a square slot's rows from this share of its
# height down, above every bar measured (bar tops at 0.734 to 0.758)
SIG_LOOK_BOTTOM = 0.7
# Every SIG_STEP-th pixel, its low SIG_SHIFT bits dropped
SIG_STEP = 2
SIG_SHIFT = 2

# Badge look from Jeffrey's ARC Tracker screenshots. Gold is only the hovered
# cell's frame (cell_border.HOVER_GOLD), never a stash outline
SHORT_RED = "#ff5c4d"
ENOUGH_GREEN = "#4fd36b"
# The short-stock tile outline, no longer drawn (17 Oct) but on older
# captures, which grid.py still reads past
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
TAG_BG = "#141414d9"
# The wide-slot price pill's alpha below WORTH_HIGH, TAG_BG's own
TAG_ALPHA = "d9"
# The pill is fully opaque from WORTH_HIGH, so the gold stands out
TAG_ALPHA_HIGH = "ff"

# Worth at a glance: the price patch and pill are filled by the stack's worth,
# red for the cheapest stacks through yellow and green, gold from WORTH_HIGH.
# The patch covers only the icon end of the bar, the rest keeps the rarity.
# Below WORTH_LOW a slot is cheap enough to drop first
WORTH_LOW = 250
# From WORTH_HIGH a slot must stand out
WORTH_HIGH = 5000
# The fill below WORTH_LOW and where the ramp starts. Every channel is more
# than 3 times grid.RED_TOLERANCE from SHORT_RED in at least one channel
# (45 in blue), so a cheap patch is never read as a short-stock outline
WORTH_RED = "#f43b20"
# The ramp passes this lemon yellow at WORTH_MID, the middle of the log range
WORTH_YELLOW = "#e8d44a"
WORTH_MID = math.sqrt(WORTH_LOW * WORTH_HIGH)
# Where the ramp ends, just below WORTH_HIGH
WORTH_GREEN = "#3fb950"
# The fill from WORTH_HIGH up, a deep gold: more than 3 times
# grid.GOLD_TOLERANCE from both ARLO_GOLDS in red (83 and 69), so a patch is
# never taken for ARLO's hover frame, and further from WORTH_YELLOW in OKLab
# (0.266) than the yellow is from WORTH_GREEN (0.212)
WORTH_GOLD = "#ad7500"
# Halvings of the chroma when a ramp colour falls outside sRGB, and how far
# outside a channel may be (it is clipped after)
GAMUT_STEPS = 24
GAMUT_SLACK = 1e-4
# The two text colours. Each fill takes the one with the higher contrast
WORTH_DARK_TEXT = "#111111"
# The game's own count ink as badge_helper draws it
WORTH_LIGHT_TEXT = "#" + "".join(f"{round(c * 255):02x}" for c in PATCH_TEXT_RGB)

# The passes run on captures that show ARLO's own patches, and a patch fill
# hides the bar from slot_marks (the bar is darker, about 15).
# unpatch repaints a patch's fill in the bar's colour first. A patch is the
# largest run of columns, starting in the left PATCH_START_MAX of a square
# crop's rows from PATCH_TOP share of its height down, where one exact fill
# of the worth scale appears in PATCH_MIN_ROWS rows or more, with
# PATCH_MIN_SHARE of the crop area or more of that fill. Gaps under
# PATCH_BRIDGE columns (the text) are bridged. The fill must start below
# the first of those rows and reach PATCH_BOTTOM_MIN of the height, as the
# bar does (bar tops at 0.734 to 0.758, bottoms about 0.94), which leaves
# out flat item art above the bar
PATCH_TOP = 0.7
PATCH_BOTTOM_MIN = 0.85
PATCH_MIN_SHARE = 0.01
PATCH_MIN_ROWS = 3
PATCH_BRIDGE = 4
PATCH_START_MAX = 0.3


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


def base_lead(found: Sequence[EmbedCandidate], *, tiered_only: bool = False) -> float:
    """Top similarity minus the best candidate of another base item.

    With tiered_only, untiered candidates are passed over too (for a slot
    whose tier numeral was read). When no candidate is left, the last one
    stands in for the best other item (a lower bound on the lead).
    """
    if not found:
        return 0.0
    top = base_name(found[0].name).casefold()
    others = [
        c.similarity
        for c in found[1:]
        if base_name(c.name).casefold() != top
        and not (tiered_only and name_tier(c.name) is None)
    ]
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


class _Cached(NamedTuple):
    """The part of a SlotItem that depends only on the slot's look."""

    item: Item | None
    tier: str | None
    stack: int | None
    confidence: float
    lead: float
    guess: str | None
    bar: BarGeometry | None


class SlotCache:
    """Identified slots by signature, the least recently used dropped at size.

    Beside the entries it keeps, by look_signature, the bar box of the last
    square slot with that look, so a slot whose bar is not found under a
    drawn patch can still be keyed (see identify_slots). Keep one per
    gallery: its entries are only valid for the gallery and lookup that
    made them.

    added counts the entries and bar boxes kept since the cache was made or
    loaded, so a saver can tell when it has something new. snapshot and
    from_rows turn the cache into rows for Database.save_slot_cache and
    back (see load_slot_cache).
    """

    def __init__(self, size: int = SLOT_CACHE_SIZE):
        self.size = size
        self.added = 0
        self._entries: OrderedDict[tuple, _Cached] = OrderedDict()
        self._bars: OrderedDict[tuple, Rect] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: tuple) -> bool:
        return key in self._entries

    def get(self, key: tuple) -> _Cached | None:
        found = self._entries.get(key)
        if found is not None:
            self._entries.move_to_end(key)
        return found

    def peek(self, key: tuple) -> _Cached | None:
        """The entry for a key without marking it used, None when none is kept."""
        return self._entries.get(key)

    def peek_bar_box(self, look: tuple) -> Rect | None:
        """bar_box without marking the look used."""
        return self._bars.get(look)

    def put(self, key: tuple, value: _Cached) -> None:
        self.added += 1
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self.size:
            self._entries.popitem(last=False)

    def bar_box(self, look: tuple) -> Rect | None:
        """The bar box last kept for a look, None when none was."""
        found = self._bars.get(look)
        if found is not None:
            self._bars.move_to_end(look)
        return found

    def keep_bar_box(self, look: tuple, box: Rect) -> None:
        if self._bars.get(look) != box:
            self.added += 1
        self._bars[look] = box
        self._bars.move_to_end(look)
        while len(self._bars) > self.size:
            self._bars.popitem(last=False)

    def snapshot(self) -> "SlotCacheSnapshot":
        """The entries and bar boxes as they are now, least recently used first.

        Only copies the two dicts, in their order, so it is quick enough for
        the pass thread. A dict copy makes no object per entry, which a
        list of pairs would, enough to start a full garbage collection of
        about 20 ms on the pass thread. rows turns it into rows for the
        database.
        """
        return SlotCacheSnapshot(dict(self._entries), dict(self._bars), self.added)

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[tuple[str, str, str]],
        lookup: Callable[[str], Item | None],
        size: int = SLOT_CACHE_SIZE,
    ) -> "SlotCache":
        """A cache from (kind, key, value) rows, least recently used first.

        The rows are those SlotCacheSnapshot.rows makes. Each entry's item
        is looked up again by name, so the catalog's prices of now are
        used, and an entry whose item is no longer in the catalog is
        dropped. A row that cannot be read is skipped. At most size entries
        and bar boxes are kept, the most recently used ones. added starts
        at 0.
        """
        cache = cls(size)
        items: dict[str, Item | None] = {}
        skipped = 0
        for kind, key, value in rows:
            try:
                if kind == _ENTRY_ROW:
                    entry = _decode_entry(json.loads(value), lookup, items)
                    if entry is not None:
                        cache.put(_decode_key(key, entry=True), entry)
                elif kind == _BAR_ROW:
                    cache.keep_bar_box(_decode_key(key, entry=False), _rect(value))
                else:
                    skipped += 1
            except (ValueError, TypeError, KeyError, IndexError, OverflowError):
                skipped += 1
        if skipped:
            logger.debug("slot cache: skipped %d unreadable rows", skipped)
        cache.added = 0
        return cache


_ENTRY_ROW = "entry"
_BAR_ROW = "bar"


class SlotCacheSnapshot(NamedTuple):
    """A SlotCache's entries and bar boxes at one moment, and its added count."""

    entries: dict[tuple, _Cached]
    bars: dict[tuple, Rect]
    added: int

    def rows(self) -> list[tuple[str, str, str]]:
        """(kind, key, value) text rows, entries then bar boxes, oldest first."""
        rows = [
            (_ENTRY_ROW, _encode_key(key), json.dumps(_encode_entry(entry)))
            for key, entry in self.entries.items()
        ]
        rows += [
            (_BAR_ROW, _encode_key(look), json.dumps(list(box)))
            for look, box in self.bars.items()
        ]
        return rows


def _encode_key(key: tuple) -> str:
    """A signature key as JSON text, its digest bytes as hex."""
    return json.dumps([v.hex() if isinstance(v, bytes) else v for v in key])


def _decode_key(text: str, *, entry: bool) -> tuple:
    """_encode_key back. An entry key is (read_stacks, height, width, digest),
    a look key (height, width, digest). Raises ValueError on any other shape.
    """
    values = json.loads(text)
    if not isinstance(values, list):
        raise TypeError(text)
    if entry:
        if not values or type(values[0]) is not bool:
            raise ValueError(text)
        flag, rest = values[:1], values[1:]
    else:
        flag, rest = [], values
    if (
        len(rest) != 3
        or not all(type(v) is int for v in rest[:2])
        or not isinstance(rest[2], str)
    ):
        raise ValueError(text)
    return (*flag, rest[0], rest[1], bytes.fromhex(rest[2]))


def _rect(value) -> Rect:
    values = json.loads(value) if isinstance(value, str) else value
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError(value)
    if not all(type(v) is int for v in values):
        raise ValueError(value)
    return Rect(*values)


def _optional_rect(value) -> Rect | None:
    return None if value is None else _rect(value)


def _encode_entry(entry: _Cached) -> dict:
    bar = entry.bar
    return {
        "item": entry.item.name if entry.item is not None else None,
        "tier": entry.tier,
        "stack": entry.stack,
        "confidence": entry.confidence,
        "lead": entry.lead,
        "guess": entry.guess,
        "bar": None
        if bar is None
        else [
            list(bar.bar_rect),
            list(bar.colour),
            None if bar.icon_box is None else list(bar.icon_box),
            None if bar.right_mark_box is None else list(bar.right_mark_box),
        ],
    }


def _decode_entry(
    data: dict,
    lookup: Callable[[str], Item | None],
    items: dict[str, Item | None],
) -> _Cached | None:
    """_encode_entry back, None when its item is no longer in the catalog.

    items memoises lookup by name. Raises on a malformed entry.
    """
    name = data["item"]
    item = None
    if name is not None:
        if not isinstance(name, str):
            raise TypeError(name)
        if name not in items:
            found = lookup(name)
            same = found is not None and found.name.casefold() == name.casefold()
            items[name] = found if same else None
        item = items[name]
        if item is None:
            return None
    bar = data["bar"]
    if bar is not None:
        box, colour, icon, mark = bar
        if len(colour) != 3 or not all(type(v) is int for v in colour):
            raise ValueError(colour)
        bar = BarGeometry(
            _rect(box), tuple(colour), _optional_rect(icon), _optional_rect(mark)
        )
    stack = data["stack"]
    if stack is not None and type(stack) is not int:
        raise TypeError(stack)
    return _Cached(
        item,
        data["tier"],
        stack,
        float(data["confidence"]),
        float(data["lead"]),
        data["guess"],
        bar,
    )


def slot_cache_fingerprint(
    names: Sequence[str],
    gallery_bytes: int,
    item_count: int,
    version: int = SLOT_CACHE_VERSION,
) -> str:
    """What a saved slot cache is valid for, as a sha256 hex digest.

    Taken over the gallery's names in order, the embedding file's byte
    size, the catalog's item count and version, joined by newlines. A
    stored cache with another fingerprint is not loaded.
    """
    text = "\n".join([*names, str(gallery_bytes), str(item_count), str(version)])
    return hashlib.sha256(text.encode()).hexdigest()


def load_slot_cache(
    db: Database, fingerprint: str, size: int = SLOT_CACHE_SIZE
) -> SlotCache:
    """The slot cache stored in db's user database for this fingerprint.

    Empty when none is stored, when the stored one has another fingerprint
    or when reading fails (logged). Items are looked up with db.lookup.
    """
    try:
        rows = db.slot_cache_rows(fingerprint)
        return SlotCache.from_rows(rows, db.lookup, size)
    except Exception:  # noqa: BLE001 - a lost cache only costs one slow pass
        logger.warning("slot cache: loading failed", exc_info=True)
        return SlotCache(size)


class SlotCacheSaver:
    """Saves a SlotCache in the background when it has new entries.

    after_pass, called after each pass, queues a save of the cache when
    entries or bar boxes were kept since the last queued save: at once the
    first time, then at most once per interval_s on clock while new ones
    remain unsaved. Queuing only takes a snapshot and wakes the saver's
    daemon thread, started with the saver (starting a thread from the pass
    thread costs it a GIL switch interval), so the pass thread never waits
    for the write. A newer snapshot queued before the thread took the
    older one replaces it. flush queues what is still unsaved, throttle or not, and waits a
    bounded time for the queued saves. save gets SlotCacheSnapshot.rows.
    A failed save is logged, never raised, and its entries count as unsaved
    again: the next pass after interval_s, or flush, queues them anew. The
    writer never retries on its own.
    """

    def __init__(
        self,
        save: Callable[[list[tuple[str, str, str]]], None],
        *,
        interval_s: float = SLOT_CACHE_SAVE_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._save = save
        self._interval_s = interval_s
        self._clock = clock
        self._done = threading.Condition()
        self._pending: SlotCacheSnapshot | None = None
        self._busy = False
        self._cache: SlotCache | None = None
        # added of the last snapshot queued and of the last one written
        self._queued_added = 0
        self._saved_added = 0
        self._failed = False
        self._queued_at: float | None = None
        threading.Thread(
            target=self._write, name="slot-cache-save", daemon=True
        ).start()

    def after_pass(self, cache: SlotCache) -> bool:
        """Queue a save when the cache has unsaved entries and the throttle allows.

        True when one was queued.
        """
        self._cache = cache
        with self._done:
            if cache.added == self._queued_added:
                return False
        now = self._clock()
        if self._queued_at is not None and now - self._queued_at < self._interval_s:
            return False
        self._queued_at = now
        return self._queue(cache)

    def flush(self, timeout: float = SLOT_CACHE_STOP_WAIT_S) -> bool:
        """Queue what is unsaved and wait up to timeout s for every queued save.

        True when the queued saves were written when it returns, False when
        one is still running or the last one failed.
        """
        cache = self._cache
        with self._done:
            unsaved = cache is not None and cache.added != self._queued_added
        if unsaved:
            self._queue(cache)
        deadline = time.monotonic() + timeout
        with self._done:
            while self._busy:
                left = deadline - time.monotonic()
                if left <= 0:
                    logger.warning("slot cache: last save still running at stop")
                    return False
                self._done.wait(left)
            return not self._failed

    def _queue(self, cache: SlotCache) -> bool:
        try:
            snapshot = cache.snapshot()
        except RuntimeError:
            # Changed by a pass while copied (only from flush at stop)
            logger.warning("slot cache: changed while copied, not saved")
            return False
        with self._done:
            self._queued_added = snapshot.added
            self._pending = snapshot
            self._busy = True
            self._done.notify_all()
        return True

    def _write(self) -> None:
        while True:
            with self._done:
                while self._pending is None:
                    self._busy = False
                    self._done.notify_all()
                    self._done.wait()
                snapshot, self._pending = self._pending, None
            start = time.perf_counter()
            try:
                rows = snapshot.rows()
                self._save(rows)
            except Exception:  # noqa: BLE001 - a lost save only costs one slow pass
                logger.warning("slot cache: saving failed", exc_info=True)
                with self._done:
                    self._failed = True
                    if self._pending is None:
                        # Unsaved again, for the next pass or flush to queue
                        self._queued_added = self._saved_added
                continue
            with self._done:
                self._failed = False
                self._saved_added = snapshot.added
            logger.debug(
                "slot cache: saved %d rows in %.0f ms",
                len(rows),
                (time.perf_counter() - start) * 1000,
            )


def _band(crop: np.ndarray) -> int:
    """Width of the border band slot_signature leaves out."""
    return max(SIG_BORDER_MIN, round(SIG_BORDER * min(crop.shape[:2])))


def _unmarked(crop: np.ndarray) -> np.ndarray:
    """A copy of the crop with its border band and top tag band zeroed."""
    height, width = crop.shape[:2]
    kept = np.array(crop, dtype=np.uint8)
    band = _band(crop)
    text = max(TAG_TEXT_MIN, min(TAG_TEXT_MAX, round(height * TAG_TEXT_SHARE)))
    kept[: max(band, round(SIG_TAG_ROWS * text))] = 0
    kept[height - band :] = 0
    kept[:, :band] = 0
    kept[:, width - band :] = 0
    return kept


def _digest(kept: np.ndarray) -> tuple:
    height, width = kept.shape[:2]
    sampled = np.ascontiguousarray(kept[::SIG_STEP, ::SIG_STEP] >> SIG_SHIFT)
    return (height, width, hashlib.blake2b(sampled.tobytes(), digest_size=16).digest())


def patch_boundary(crop: np.ndarray, box: Rect) -> int:
    """Where the price patch can reach in the bar box of a square slot crop.

    The right mark is found as slot_marks.bar_geometry finds it, on the
    bar box given, but only in the rows and columns inside the border band,
    where the red outline cannot add ink. The boundary is then
    BarGeometry.patch_right: the mark stays right of it, the patch left.
    """
    band = _band(crop)
    height, width = crop.shape[:2]
    inner = Rect(
        max(box.left, band),
        max(box.top, band),
        min(box.right, width - band),
        min(box.bottom, height - band),
    )
    if inner.right <= inner.left or inner.bottom <= inner.top:
        return BarGeometry(box, (0, 0, 0), None, None).patch_right
    lum = _luminance(crop)[inner.top : inner.bottom, inner.left : inner.right]
    background = float(np.median(lum))
    bar = _Bar(lum, lum > background + _INK_ABOVE, background, inner)
    words = _words(bar)
    # Shares of the whole bar box, as bar_geometry measures them
    start = inner.left - box.left
    icon = words and start + words[0][0] < _ICON_START_MAX * box.width
    rest = words[1:] if icon else words
    right_half = _RIGHT_HALF * box.width
    mark = rest[-1] if rest and start + rest[-1][1] >= right_half else None
    mark_box = _word_box(bar, mark) if mark else None
    return BarGeometry(box, (0, 0, 0), None, mark_box).patch_right


def slot_signature(crop: np.ndarray, bar_box: Rect | None = None) -> tuple:
    """A key for a slot crop's look that ignores what ARLO draws on it.

    Left out: a border band on every side (the red outline drawn inside the
    rect, the hovered frame's edge), the top band where the have/need tag
    and a wide slot's price pill go (sized like badge_helper's tag), and,
    given the square slot's bar box (from slot_marks.bar_geometry), the bar
    left of patch_boundary, where the price patch goes. The stack count or
    tier numeral is the bar's right mark, which the boundary stays left of,
    so a changed count is a new key. The rest is sampled every SIG_STEP
    pixels with its low bits dropped and hashed with the crop's size.
    ARLO's drawings are left out rather than keyed on a capture before
    badges, because the captures the pass gets already show the badges.
    """
    kept = _unmarked(crop)
    if bar_box is not None:
        top = max(0, bar_box.top - SIG_BAR_MARGIN)
        kept[top:, : patch_boundary(crop, bar_box)] = 0
    return _digest(kept)


def look_signature(crop: np.ndarray) -> tuple:
    """slot_signature's key without the bar rows, the same with any count."""
    kept = _unmarked(crop)
    kept[round(SIG_LOOK_BOTTOM * kept.shape[0]) :] = 0
    return _digest(kept)


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


def _is_blueprint_name(name: str) -> bool:
    return name.casefold().endswith(BLUEPRINT_SUFFIX)


def _tile_twin(top: str, alike: Sequence[str], *, blueprint: bool | None) -> str | None:
    """The twin a slot's tile settles, or None when the tile cannot.

    Only a group of one item and its blueprint is settled: a blueprint tile
    picks the blueprint, a plain tile the item.
    """
    if blueprint is None:
        return None
    group = (top, *alike)
    blueprints = [n for n in group if _is_blueprint_name(n)]
    plain = [n for n in group if not _is_blueprint_name(n)]
    if len(blueprints) != 1 or len(plain) != 1:
        return None
    return blueprints[0] if blueprint else plain[0]


def _own_blueprint_only(picked: str, alike: Sequence[str]) -> bool:
    """Whether the item's only twin is the blueprint named after it.

    "Compensator II" and "Compensator II Blueprint" name one tier, so the
    catalog tier is known without a numeral. "Anvil I" with "Anvil Blueprint"
    does not, that blueprint has no tier.
    """
    own = f"{picked}{BLUEPRINT_SUFFIX}"
    return len(alike) == 1 and alike[0].casefold() == own.casefold()


def _resolve(
    found: Sequence[EmbedCandidate],
    read: str | None,
    lookup: Callable[[str], Item | None],
    twins: Mapping[str, Sequence[str]],
    *,
    blueprint: bool | None = None,
) -> tuple[Item | None, str | None, float]:
    """The catalog item, its tier and the lead for one slot's candidates.

    When the top name has twins (items with the same icon) and they are an
    item and its blueprint, the tile decides (blueprint is is_blueprint_tile's
    answer): a blueprint tile picks the blueprint, a plain tile the item,
    whose tier, when it has one, is the read numeral (none read leaves it
    unidentified, unless the blueprint is named after that very tiered item,
    as "Compensator II Blueprint" is, and the catalog tier is taken).
    Otherwise the item is kept only when the read numeral
    settles it: the top name is tiered, a numeral was read and no twin is
    tiered (an untiered blueprint or variant cannot show a numeral). For the
    same reason a settled slot's lead passes over the untiered candidates.
    """
    top = found[0]
    catalog_tier = name_tier(top.name)
    tier = read or catalog_tier
    settled = catalog_tier is not None and read is not None
    lead = base_lead(found, tiered_only=settled)
    if top.similarity < MIN_SIMILARITY or lead < MIN_LEAD:
        return None, tier, lead
    alike = twins.get(top.name, ())
    picked = _tile_twin(top.name, alike, blueprint=blueprint) if alike else None
    if picked is not None and _is_blueprint_name(picked):
        return lookup(picked), None, lead
    if picked is not None:
        # The item, a tiered one still needs the numeral for its tier
        picked_tier = name_tier(picked)
        tier = read or picked_tier
        if picked_tier is not None and read is None:
            if not _own_blueprint_only(picked, alike):
                return None, tier, lead
            return lookup(picked), picked_tier, lead
        name = f"{base_name(picked)} {read}" if picked_tier else picked
        return lookup(name), tier, lead
    if alike and not (settled and all(name_tier(t) is None for t in alike)):
        return None, tier, lead
    name = f"{base_name(top.name)} {read}" if settled else top.name
    return lookup(name), tier, lead


def _slot_crop(rgb: np.ndarray, rect: Rect) -> np.ndarray:
    """A slot's crop of the frame, unpatched when it is square."""
    crop = rgb[rect.top : rect.bottom, rect.left : rect.right]
    return crop if is_wide(rect) else unpatch(crop)


def identify_slots(
    rgb: np.ndarray,
    gallery: EmbedGallery,
    lookup: Callable[[str], Item | None],
    *,
    embedder=None,
    read_stacks: bool = False,
    slots: Iterable[Slot] | None = None,
    twins: Mapping[str, Sequence[str]] | None = None,
    cache: SlotCache | None = None,
) -> list[SlotItem]:
    """Every filled slot on an RGB frame with its item, tier, stack and confidence.

    lookup resolves a catalog name (base plus tier) to an item, usually
    Database.lookup. embedder defaults to icon_embed.default_embedder().
    slots defaults to grid.find_slots(rgb). twins maps a gallery name to the
    names sharing its icon (from distinct_gallery), None means no groups.
    With cache, a slot whose slot_signature is cached takes the cached item,
    tier, stack, confidence and bar, and every other slot is identified and
    cached. The misses are embedded in one call. Slots come back in
    find_slots order.
    """
    if slots is None:
        slots = find_slots(rgb)
    filled = [s for s in slots if s.filled]
    if not filled or len(gallery) == 0:
        return []
    crops = [_slot_crop(rgb, s.rect) for s in filled]
    bars = [
        None if is_wide(slot.rect) else bar_geometry(crop)
        for slot, crop in zip(filled, crops, strict=True)
    ]
    keys: list[tuple | None] = [None] * len(crops)
    known: list[_Cached | None] = [None] * len(crops)
    if cache is not None:
        for n, crop in enumerate(crops):
            box = bars[n].bar_rect if bars[n] is not None else None
            keys[n] = (read_stacks, *slot_signature(crop, box))
            known[n] = cache.get(keys[n])
            if known[n] is None and not is_wide(filled[n].rect):
                # A patch or outline drawn over the bar can hide it from
                # bar_geometry or move its box. Key again on the box this
                # look had when it was identified
                kept_box = cache.bar_box(look_signature(crop))
                if kept_box is not None and kept_box != box:
                    again = (read_stacks, *slot_signature(crop, kept_box))
                    known[n] = cache.get(again)
    missed = [n for n, entry in enumerate(known) if entry is None]
    if missed:
        squares = [crop_square(crops[n]) for n in missed]
        vectors = (embedder or default_embedder()).embed(squares)
        for n, found in zip(missed, _rank(vectors, gallery), strict=True):
            crop = crops[n]
            tier_read = read_tier(crop)
            item, tier, lead = _resolve(
                found,
                tier_read.tier if tier_read else None,
                lookup,
                twins or {},
                blueprint=is_blueprint_tile(crop),
            )
            stack = read_stack(crop) if read_stacks and item is not None else None
            bar = bars[n]
            known[n] = _Cached(
                item,
                tier,
                stack,
                round(found[0].similarity, 4),
                round(lead, 4),
                found[0].name,
                bar,
            )
            if cache is not None:
                cache.put(keys[n], known[n])
                if bar is not None:
                    cache.keep_bar_box(look_signature(crop), bar.bar_rect)
    return [
        SlotItem(slot.rect, slot.panel, *entry)
        for slot, entry in zip(filled, known, strict=True)
    ]


def peek_slots(
    rgb: np.ndarray,
    cache: SlotCache,
    *,
    read_stacks: bool = False,
    slots: Iterable[Slot] | None = None,
) -> list[SlotItem]:
    """The filled slots of a frame whose look is in the cache, with their entries.

    Each slot is keyed as identify_slots keys it (with the bar box kept for
    its look when its own misses), but the cache is only read: no slot is
    embedded, no tier or stack read, nothing added and no entry marked used.
    Slots not in the cache are left out. read_stacks picks the entries as
    identify_slots with that read_stacks keeps them.
    """
    if slots is None:
        slots = find_slots(rgb)
    found = []
    for slot in slots:
        if not slot.filled:
            continue
        r = slot.rect
        crop = _slot_crop(rgb, r)
        bar = None if is_wide(r) else bar_geometry(crop)
        box = bar.bar_rect if bar is not None else None
        entry = cache.peek((read_stacks, *slot_signature(crop, box)))
        if entry is None and not is_wide(r):
            kept_box = cache.peek_bar_box(look_signature(crop))
            if kept_box is not None and kept_box != box:
                entry = cache.peek((read_stacks, *slot_signature(crop, kept_box)))
        if entry is not None:
            found.append(SlotItem(r, slot.panel, *entry))
    return found


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


def _linear(channel: int) -> float:
    """An sRGB channel 0..255 as linear light 0..1."""
    c = channel / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _srgb(linear: float) -> int:
    """Linear light as the nearest sRGB channel 0..255, clipped."""
    c = 12.92 * linear if linear <= 0.0031308 else 1.055 * linear ** (1 / 2.4) - 0.055
    return max(0, min(255, round(c * 255)))


def oklch(hex_colour: str) -> tuple[float, float, float]:
    """A '#rrggbb' colour as OKLCH: lightness, chroma and hue in degrees."""
    r, g, b = (_linear(v) for v in _rgb(hex_colour))
    lms = (
        0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b,
        0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b,
        0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b,
    )
    l_, m_, s_ = (math.copysign(abs(v) ** (1 / 3), v) for v in lms)
    lightness = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b_ = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return lightness, math.hypot(a, b_), math.degrees(math.atan2(b_, a)) % 360


def _linear_of_oklch(
    lightness: float, chroma: float, hue: float
) -> tuple[float, float, float]:
    a = chroma * math.cos(math.radians(hue))
    b = chroma * math.sin(math.radians(hue))
    l_ = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return (
        4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
        -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
        -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
    )


def _hex_of_oklch(lightness: float, chroma: float, hue: float) -> str:
    """An OKLCH colour as '#rrggbb'.

    Outside sRGB the chroma is lowered until the colour fits, keeping its
    lightness and hue, so the scale's hue never jumps where it clips.
    """
    low, high = 0.0, chroma
    for _ in range(GAMUT_STEPS):
        channels = _linear_of_oklch(lightness, high, hue)
        if all(-GAMUT_SLACK <= c <= 1 + GAMUT_SLACK for c in channels):
            break
        middle = (low + high) / 2
        inside = _linear_of_oklch(lightness, middle, hue)
        if all(-GAMUT_SLACK <= c <= 1 + GAMUT_SLACK for c in inside):
            low = middle
        else:
            high = middle
    channels = _linear_of_oklch(lightness, high, hue)
    return "#" + "".join(f"{_srgb(c):02x}" for c in channels)


def _mix(first: str, second: str, share: float) -> str:
    """The colour share of the way from first to second in OKLCH.

    The hue takes the short way round, so red to green passes yellow.
    """
    l1, c1, h1 = oklch(first)
    l2, c2, h2 = oklch(second)
    turn = (h2 - h1 + 180) % 360 - 180
    return _hex_of_oklch(
        l1 + share * (l2 - l1), c1 + share * (c2 - c1), h1 + share * turn
    )


def contrast(first: str, second: str) -> float:
    """The WCAG contrast ratio of two '#rrggbb' colours."""

    def luminance(colour: str) -> float:
        r, g, b = (_linear(v) for v in _rgb(colour))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    dark, light = sorted([luminance(first), luminance(second)])
    return (light + 0.05) / (dark + 0.05)


def _text_on(fill: str) -> str:
    """WORTH_DARK_TEXT or WORTH_LIGHT_TEXT, whichever reads better on fill."""
    dark = contrast(fill, WORTH_DARK_TEXT)
    return (
        WORTH_DARK_TEXT
        if dark >= contrast(fill, WORTH_LIGHT_TEXT)
        else WORTH_LIGHT_TEXT
    )


def _worth_fill(worth: int) -> str:
    if worth >= WORTH_HIGH:
        return WORTH_GOLD
    if worth < WORTH_LOW:
        return WORTH_RED
    if worth < WORTH_MID:
        share = math.log(worth / WORTH_LOW) / math.log(WORTH_MID / WORTH_LOW)
        return _mix(WORTH_RED, WORTH_YELLOW, share)
    share = math.log(worth / WORTH_MID) / math.log((WORTH_HIGH - 1) / WORTH_MID)
    return _mix(WORTH_YELLOW, WORTH_GREEN, share)


def worth_colours(worth: int) -> tuple[str, str]:
    """The (fill, text) of a price patch or pill for a stack's worth, as '#rrggbb'.

    Below WORTH_LOW WORTH_RED. From WORTH_LOW to WORTH_HIGH a ramp in OKLCH,
    even steps in log worth being even steps of colour, from WORTH_RED
    through orange to WORTH_YELLOW at WORTH_MID, then through yellow-green
    to WORTH_GREEN just below WORTH_HIGH. From WORTH_HIGH WORTH_GOLD. The
    text is the one of WORTH_DARK_TEXT and WORTH_LIGHT_TEXT that reads
    better on the fill.
    """
    fill = _worth_fill(worth)
    return fill, _text_on(fill)


def _fill_codes() -> np.ndarray:
    """Every fill worth_colours can return, as sorted 0xrrggbb ints."""
    fills = {_worth_fill(worth) for worth in range(WORTH_LOW, WORTH_HIGH)}
    fills |= {WORTH_RED, WORTH_GOLD}
    return np.array(sorted(int(f[1:], 16) for f in fills), dtype=np.int32)


# The fills unpatch looks for, computed once from the scale
FILL_CODES = _fill_codes()


def unpatch(crop: np.ndarray) -> np.ndarray:
    """A square slot crop with ARLO's price patch fill in the bar's colour.

    The fill is found as the comment above PATCH_MIN_SHARE says, among the
    colours of FILL_CODES, and its pixels take the per-channel median of
    the other pixels of its rows that are not ink, mostly the bar right of
    it. The text stays, as it did on patches drawn in the bar's colour. A
    crop without a patch comes back as it is, the same array.
    """
    height, width = crop.shape[:2]
    top = round(PATCH_TOP * height)
    lower = np.asarray(crop[top:, :, :3])
    codes = (
        (lower[..., 0].astype(np.int32) << 16)
        | (lower[..., 1].astype(np.int32) << 8)
        | lower[..., 2]
    )
    known = codes[np.isin(codes, FILL_CODES)]
    least = PATCH_MIN_SHARE * height * width
    if known.size < least:
        return crop
    values, counts = np.unique(known, return_counts=True)
    if counts.max() < least:
        return crop
    fill = codes == values[np.argmax(counts)]
    runs = [
        run
        for run in _runs(fill.sum(axis=0) >= PATCH_MIN_ROWS, PATCH_BRIDGE)
        if run[0] < PATCH_START_MAX * width
    ]
    if not runs:
        return crop
    begin, end = max(runs, key=lambda run: fill[:, run[0] : run[1]].sum())
    rows = np.flatnonzero(fill[:, begin:end].any(axis=1))
    first, last = int(rows[0]), int(rows[-1]) + 1
    if first == 0 or top + last < PATCH_BOTTOM_MIN * height:
        return crop
    region = np.zeros_like(fill)
    region[first:last, begin:end] = fill[first:last, begin:end]
    rest = np.zeros_like(fill)
    rest[first:last] = True
    rest[first:last, begin:end] = False
    if not rest.any():
        return crop
    # Ink (the count or numeral, the red outline) stays out of the median
    lum = _luminance(lower)
    plain = rest & (lum <= float(np.median(lum[rest])) + _INK_ABOVE)
    out = np.array(crop, copy=True)
    colour = np.median(lower[plain], axis=0).round().astype(np.uint8)
    out[top:, :, :3][region] = colour
    return out


def patch_badge(slot: SlotItem, worth: int) -> dict | None:
    """The patch over a square slot's type icon, None when there is no icon box.

    It is filled and its text coloured by worth_colours.
    """
    bar = slot.bar
    if bar is None or bar.icon_box is None:
        return None
    left, top = slot.rect.left, slot.rect.top
    fill, text = worth_colours(worth)
    return {
        "rect": [
            left + bar.bar_rect.left,
            top + bar.bar_rect.top,
            left + bar.bar_rect.right,
            top + bar.bar_rect.bottom,
        ],
        "icon_right": left + bar.icon_box.right,
        "patch_right": left + bar.patch_right,
        "colour": _rgb(fill),
        "fg": text,
        "text": PRICE_PREFIX + format_price(worth),
        "radius": PATCH_RADIUS,
        "kind": "patch",
    }


def badges_for(
    slot_items: Iterable[SlotItem],
    verdict_of: Callable[[SlotItem], Verdict],
    owned: Mapping[str, int] | None = None,
    *,
    sure: bool = True,
) -> list[dict]:
    """Badge dicts for the identified slots, in the badge surface's format.

    Every identified item with a sell price gets its stack's worth (the unit
    price times the stack, 1 when unread): a patch on a square slot with a
    type icon, a pill top left on a wide slot (top right when the slot
    also has a need tag), both coloured by worth_colours, and nothing on a square slot without a type icon.
    An item with active needs (a verdict with sources) gets a have/need tag
    top left, have from owned (item name to count, for example the stored
    stash map plus the loadout) when given, else from owned_totals of these
    slots, and need the sum of the sources. While sure is False (the map
    behind owned is partial) have is the larger of owned and the slots on
    screen, and the tag reads N+/M, a lower bound. Short
    stock is red, enough is green with a check mark. No tile outline: Jeffrey
    found the tag enough (17 Oct). Unidentified slots get no badge. verdict_of gives a
    slot's verdict, for example verdicts_for(db).
    """
    slots = list(slot_items)
    visible = owned_totals(slots)
    if owned is None:
        owned, sure = visible, True
    badges: list[dict] = []
    for slot in slots:
        item = slot.item
        if item is None:
            continue
        rect = [int(v) for v in slot.rect]
        need = sum(qty for _, qty in verdict_of(slot).sources)
        if need:
            have = owned.get(item.name, 0)
            if not sure:
                have = max(have, visible.get(item.name, 0))
            short = have < need
            colour = SHORT_RED if short else ENOUGH_GREEN
            badges.append(
                {
                    "rect": rect,
                    "text": f"{have}{'' if sure else '+'}/{need}",
                    "corner": NEED_CORNER,
                    "fg": colour,
                    "bg": TAG_BG,
                    "check": not short,
                    "outline": _rgb(colour),
                    "kind": "tag",
                }
            )
        if not item.sell_price:
            continue
        worth = item.sell_price * (slot.stack if slot.stack is not None else 1)
        if is_wide(slot.rect):
            fill, text = worth_colours(worth)
            alpha = TAG_ALPHA_HIGH if worth >= WORTH_HIGH else TAG_ALPHA
            badges.append(
                {
                    "rect": rect,
                    "text": PRICE_PREFIX + format_price(worth),
                    "corner": "tr" if need else TAG_CORNER_WIDE,
                    "fg": text,
                    "bg": fill + alpha,
                    "kind": "tag",
                }
            )
        elif (patch := patch_badge(slot, worth)) is not None:
            badges.append(patch)
    return badges
