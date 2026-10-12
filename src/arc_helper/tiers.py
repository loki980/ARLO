"""Roman numeral tier handling for item names.

Weapon names end in a tier token (I to V) that OCR often garbles: "II" reads
as "Il" or "1l", "IV" as "|V", and the whitelisted pass glues the token to the
word before it. These pure functions split catalog names, build an
OCR-tolerant key, resolve a garbled read against the tiers that exist, and
count the strokes of a tier token in a binary word image.
"""

import re
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np

TIER_TOKENS = ("I", "II", "III", "IV", "V")

# A tier token in tolerant (tier_key) form, for example "ii" or "iv"
TIER_LIKE = re.compile(r"^(i{1,3}|iv|v)$")

_TOLERANT = str.maketrans({"l": "i", "1": "i", "|": "i"})


class TierVote(NamedTuple):
    """A tier read from the shape of the tier token itself."""

    tier: str | None
    confidence: float
    detail: str


def split_tier(name: str) -> tuple[str, str | None]:
    """Split a catalog name into (base, tier) by its last space-separated token.

    Returns (name, None) when the name does not end in a tier token.
    """
    parts = name.strip().rsplit(maxsplit=1)
    if len(parts) == 2 and parts[1] in TIER_TOKENS:
        return parts[0], parts[1]
    return name.strip(), None


def tier_key(text: str) -> str:
    """Tolerant match key: lowercase alphanumerics with l, 1 and | read as i.

    The mapping runs before non-alphanumerics are dropped so "|V" becomes
    "iv". Apply it to catalog bases and OCR reads alike so both sides agree.
    """
    mapped = text.lower().translate(_TOLERANT)
    return "".join(c for c in mapped if c.isalnum())


def split_read(text: str, bases: Iterable[str]) -> tuple[str, str | None]:
    """Split a tier_key read into (base, tier) using the known tiered bases.

    Picks the longest base such that text is that base plus a valid tier
    token in tolerant form, and returns the tier as its canonical token
    ("II"). Returns (text, None) when no base fits.
    """
    best: str | None = None
    for base in bases:
        if (
            base
            and text.startswith(base)
            and TIER_LIKE.match(text[len(base) :])
            and (best is None or len(base) > len(best))
        ):
            best = base
    if best is None:
        return text, None
    return best, text[len(best) :].upper()


def _levenshtein(a: str, b: str) -> int:
    """Edit distance between two short strings."""
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def resolve_tier(read_tier: str, available_tiers: Iterable[str]) -> str | None:
    """Pick the tier a read most likely means among the tiers that exist.

    The exact tier wins when present. Otherwise the nearest by edit
    distance, ties broken toward candidates containing V when the read
    contains V and toward candidates without V otherwise, then by tier
    order. None when nothing is available.
    """
    read = read_tier.upper()
    tiers = list(available_tiers)
    if not tiers:
        return None
    if read in tiers:
        return read
    wants_v = "V" in read

    def rank(tier: str) -> tuple[int, bool, int]:
        order = TIER_TOKENS.index(tier) if tier in TIER_TOKENS else len(TIER_TOKENS)
        return _levenshtein(read, tier), ("V" in tier) != wants_v, order

    return min(tiers, key=rank)


# Stroke counting thresholds, as fractions of the word height
_GAP = 0.10  # an ink gap at least this wide separates two runs
_MIN_STROKE_EXTENT = 0.5  # a run shorter than this is noise, not a stroke
_NARROW_MAX = 0.55  # an I stroke is at most this wide
_V_MIN = 1.3 * 0.5  # a V is at least this wide
_V_MAX = 1.3  # and at most this wide
_IV_RATIO = 1.6  # in IV the V run is at least this many times the I run
_V_TAPER = 1.6  # a V's ink spans this many times wider at the top than the bottom
_V_FILL_MAX = 0.7  # a V leaves its notch empty, a smeared blob does not


def _no_vote(detail: str) -> TierVote:
    return TierVote(None, 0.0, detail)


def _margin(value: float, threshold: float, full: float) -> float:
    """0 at the threshold, 1 at full, clipped. full may be above or below."""
    return float(min(1.0, max(0.0, (value - threshold) / (full - threshold))))


def _row_extent(row: np.ndarray) -> int:
    """Distance from the first to the last ink pixel in a row, inclusive."""
    cols = np.flatnonzero(row)
    return int(cols[-1] - cols[0] + 1) if cols.size else 0


def _v_shape(run: np.ndarray) -> tuple[float, str]:
    """Score a wide run as a V: ink tapering toward the bottom, notch open.

    Returns a margin score, or -1 when the run is not V shaped.
    """
    rows = np.flatnonzero(run.any(axis=1))
    band = max(1, (rows[-1] - rows[0] + 1) // 4)
    top = np.mean([_row_extent(r) for r in run[rows[0] : rows[0] + band]])
    bottom = np.mean([_row_extent(r) for r in run[rows[-1] - band + 1 : rows[-1] + 1]])
    taper = float(top / max(bottom, 1.0))
    fill = float(run.sum()) / run.size
    detail = f"taper={taper:.2f} fill={fill:.2f}"
    if taper < _V_TAPER or fill > _V_FILL_MAX:
        return -1.0, detail
    score = min(
        _margin(taper, _V_TAPER, _V_TAPER * 1.5), _margin(fill, _V_FILL_MAX, 0.55)
    )
    return score, detail


def _column_runs(ink: np.ndarray, min_gap: int) -> tuple[list[np.ndarray], list[int]]:
    """Split a word into ink column runs, bridging gaps narrower than min_gap.

    Returns the runs as column slices of ink and the widths of the gaps
    that separate them.
    """
    inked = ink.any(axis=0)
    spans: list[list[int]] = []
    gaps: list[int] = []
    x = 0
    width = inked.size
    while x < width:
        end = x
        while end < width and inked[end] == inked[x]:
            end += 1
        if inked[x]:
            if spans and x - spans[-1][1] < min_gap:
                spans[-1][1] = end
            else:
                if spans:
                    gaps.append(x - spans[-1][1])
                spans.append([x, end])
        x = end
    return [ink[:, a:b] for a, b in spans], gaps


def count_strokes(binary: np.ndarray) -> TierVote:
    """Read a tier token from a binary word image by column projection.

    binary is a 2D array of one word with ink True. Ink columns form runs
    separated by gaps of at least 10 percent of the word height. One, two
    or three narrow runs read as I, II or III, a narrow run followed by a
    run at least 1.6 times wider as IV, and a single wide tapering run as
    V. Anything else is no vote with confidence 0.
    """
    ink = np.asarray(binary, dtype=bool)
    if ink.ndim != 2 or not ink.any():
        return _no_vote("no ink")
    rows = np.flatnonzero(ink.any(axis=1))
    cols = np.flatnonzero(ink.any(axis=0))
    ink = ink[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    height = ink.shape[0]
    if height < 6:
        return _no_vote(f"too small h={height}")

    runs, gaps = _column_runs(ink, max(1, round(_GAP * height)))
    # Drop specks: a stroke spans most of the word height
    strokes = [
        run
        for run in runs
        if _row_extent(run.any(axis=1)) >= _MIN_STROKE_EXTENT * height
    ]
    widths = [s.shape[1] / height for s in strokes]
    detail = f"h={height} widths={[round(w, 2) for w in widths]} gaps={gaps}"
    if not strokes:
        return _no_vote(detail)

    gap_score = _margin(min(gaps) / height, _GAP, _GAP * 2) if gaps else 1.0
    narrow = [w <= _NARROW_MAX for w in widths]

    if all(narrow) and len(strokes) <= 3:
        score = min(_margin(w, _NARROW_MAX, _NARROW_MAX / 2) for w in widths)
        if len(strokes) > 1:
            score = min(score, gap_score)
        return TierVote(TIER_TOKENS[len(strokes) - 1], 0.5 + 0.5 * score, detail)

    if len(strokes) == 1 and _V_MIN <= widths[0] <= _V_MAX:
        score, v_detail = _v_shape(strokes[0])
        if score < 0:
            return _no_vote(f"{detail} {v_detail}")
        return TierVote("V", 0.5 + 0.5 * score, f"{detail} {v_detail}")

    if len(strokes) == 2 and narrow[0] and _IV_RATIO * widths[0] <= widths[1] <= _V_MAX:
        score, v_detail = _v_shape(strokes[1])
        if score < 0:
            return _no_vote(f"{detail} {v_detail}")
        ratio_score = _margin(widths[1] / widths[0], _IV_RATIO, _IV_RATIO * 2)
        score = min(score, gap_score, ratio_score)
        return TierVote("IV", 0.5 + 0.5 * score, f"{detail} {v_detail}")

    return _no_vote(detail)
