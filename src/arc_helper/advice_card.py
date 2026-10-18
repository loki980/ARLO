"""The advice card: a dark game-style panel drawn as an image.

build_advice_card turns a verdict and the needs behind it into a small data
model, render_advice_card draws that model with Pillow. The look is the ARC
Tracker card idea (header block with a colour stripe, a Needed for section
with a progress track, one row per source) drawn from scratch with the
bundled Barlow Semi Condensed. Nothing here touches Tk or the database.
"""

from collections.abc import Callable
from collections.abc import Iterable
from typing import NamedTuple

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from .database import HOME
from .database import Need
from .database import RecycleOutput
from .textrender import _load_font
from .textrender import _runs
from .textrender import find_font_file
from .verdict import KEEP
from .verdict import KEEP_SOME
from .verdict import KNOWN_BLUEPRINT_REASON
from .verdict import LEARN
from .verdict import RECYCLE
from .verdict import SELL
from .verdict import UNKNOWN
from .verdict import USE
from .verdict import Verdict
from .verdict import source_label

AMBER = "#f2b134"
GREEN = "#3fbf6f"
BLUE = "#2e9bff"
GREY = "#7d8289"
# Violet, the hue of the game's blueprint grid. No other kind, chip or stash
# badge uses it (keep gold, short red, enough green, sell green, recycle blue)
VIOLET = "#a678ff"
KIND_COLORS = {
    KEEP: AMBER,
    KEEP_SOME: AMBER,
    SELL: GREEN,
    RECYCLE: BLUE,
    USE: BLUE,
    LEARN: VIOLET,
    UNKNOWN: GREY,
}

PANEL = "#1b1f26"
PANEL_BORDER = "#2c313a"
HEADER_BG = "#252a33"
TRACK = "#3a3f48"
DIVIDER = "#2c313a"
WHITE = "#ffffff"
MUTED = "#b8bcc4"

SOURCE_LABELS = {"quest": "Quest", "workshop": "Workshop", "project": "Project"}

# Reference geometry in pixels at scale 1.0
WIDTH = 560
MARGIN = 14  # panel edge to header block
PAD = 20  # padding inside blocks
RADIUS = 12
HEADER_RADIUS = 8
STRIPE = 4
TITLE_SIZE = 26
SUBTITLE_SIZE = 17
SUBTITLE_MAX_LINES = 2
BODY_SIZE = 18
KIND_COLUMN = 96
ROW_GAP = 10
SUPERSAMPLE = 4
ELLIPSIS = "…"
BOLD = "bold"
REGULAR = "regular"


class AdviceCard(NamedTuple):
    """What the card shows.

    kind is a verdict kind. owned is the stack count, None when unknown.
    owned_sure is False when owned is a lower bound (the stash map is
    partial), and the card then shows it as N+. rows are (kind label, name, quantity) per source. needed_total is the
    count to keep, None when the verdict gave no count.
    """

    kind: str
    title: str
    subtitle: str | None
    needed_total: int | None
    owned: int | None
    rows: tuple[tuple[str, str, int], ...]
    owned_sure: bool = True


WEAPON_REASON = "Weapon, keep at least one"
NOT_SAVED = "Not in your saved stash yet"


def owned_unseen(card: "AdviceCard") -> bool:
    """True when the partial map has not seen the item, so no number is shown."""
    return card.owned == 0 and not card.owned_sure


def owned_text(owned: int, sure: bool) -> str:  # noqa: FBT001
    """The owned count as shown, '52' when sure and '52+' for a lower bound."""
    return str(owned) if sure else f"{owned}+"


def _row_name(need: Need) -> str:
    """Name column text: quest name, Utility Station level 2, Project phase 2."""
    if need.tier is None or need.source_kind == "quest":
        return need.source_name
    if need.source_kind == "workshop":
        return f"{need.source_name} level {need.tier}"
    return f"{need.source_name} phase {need.tier}"


def _rows(verdict: Verdict, needs: Iterable[Need]) -> tuple[tuple[str, str, int], ...]:
    """One row per verdict source, matched to a need by label and quantity.

    Each need is used once, so equal sources keep their multiplicity, and a
    need that is not among verdict.sources gives no row.
    """
    pool = list(needs)
    rows = []
    for label, quantity in verdict.sources:
        for i, need in enumerate(pool):
            if source_label(need) == label and need.quantity == quantity:
                pool.pop(i)
                kind = SOURCE_LABELS.get(need.source_kind, need.source_kind.title())
                rows.append((kind, _row_name(need), quantity))
                break
    return tuple(rows)


def _keep_title(count: int | None, owned: int | None, sure: bool = True) -> str:  # noqa: FBT001, FBT002
    if count is None:
        return "Keep"
    if owned is not None:
        more = max(0, count - owned)
        # A lower bound that already covers the count leaves nothing to find
        if not sure and more:
            return f"Keep all {count}: up to {more} more needed"
        return f"Keep all {count}: {more} more needed"
    return f"Keep {count}"


def _join_goals(names: list[str]) -> str:
    """'A', 'A and B', 'A, B and C'."""
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _keep_reason(verdict: Verdict, rows: tuple[tuple[str, str, int], ...]) -> str:
    """Why to keep: the goals behind the rows, else the verdict's own reason."""
    if rows:
        return "For " + _join_goals([name for _, name, _ in rows])
    return verdict.reason or "Kept by ARLO's rules"


def needed_parts(
    outputs: Iterable[RecycleOutput],
    needs_for: Callable[[str], list[Need]],
    is_needed: Callable[[str, list[Need]], bool],
) -> frozenset[str]:
    """Names of the recycle outputs that are still needed.

    needs_for(component) gives the needs of a component, is_needed(component,
    needs) says whether those needs make it worth keeping.
    """
    return frozenset(
        o.component for o in outputs if is_needed(o.component, needs_for(o.component))
    )


def _recycle_subtitle(
    outputs: Iterable[RecycleOutput], needed: frozenset[str]
) -> str | None:
    """'Recycles into A, 2 B, which you still need', None without home outputs."""
    home = [o for o in outputs if o.location == HOME]
    if not home:
        return None
    # sorted is stable, so needed parts come first and keep their order
    home.sort(key=lambda o: o.component not in needed)
    parts = [
        f"{o.quantity} {o.component}" if o.quantity > 1 else o.component for o in home
    ]
    text = "Recycles into " + ", ".join(parts)
    if any(o.component in needed for o in home):
        text += ", which you still need"
    return text


def _known_subtitle(kind: str, subtitle: str | None) -> str:
    """A known blueprint's subtitle: the known line, then the usual one."""
    if kind == SELL or not subtitle:
        return KNOWN_BLUEPRINT_REASON
    if subtitle.startswith(KNOWN_BLUEPRINT_REASON):
        return subtitle
    return f"{KNOWN_BLUEPRINT_REASON}. {subtitle}"


def build_advice_card(
    verdict: Verdict,
    needs: Iterable[Need],
    owned: int | None = None,
    recycle_outputs: Iterable[RecycleOutput] = (),
    needed_parts: frozenset[str] = frozenset(),
    owned_sure: bool = True,  # noqa: FBT001, FBT002
) -> AdviceCard:
    """The card model for a verdict. Rows exist for keep and keep_some only.

    Keep and keep_some always carry a reason in the subtitle: the goals behind
    the rows, else verdict.reason.

    For recycle and use the subtitle names the home recycle outputs and which
    are still needed, falling back to verdict.reason without outputs.
    """
    kind = verdict.kind
    subtitle: str | None = None
    rows: tuple[tuple[str, str, int], ...] = ()
    total = verdict.keep_count
    unseen = owned == 0 and not owned_sure
    if kind == KEEP:
        # A weapon keeps one and more until its blueprint is owned, so its
        # title ignores the count and only the footer shows it
        weapon = verdict.reason.startswith(WEAPON_REASON)
        title = _keep_title(total, None if unseen or weapon else owned, owned_sure)
        rows = _rows(verdict, needs)
        subtitle = _keep_reason(verdict, rows)
    elif kind == KEEP_SOME:
        title = "Keep" if total is None else f"Keep {total}"
        rows = _rows(verdict, needs)
        spare = owned - total if owned is not None and total is not None else 0
        rest = f"sell {spare}" if owned_sure and spare > 0 else "sell the rest"
        subtitle = f"{_keep_reason(verdict, rows)}, then {rest}"
    elif kind == SELL:
        title, subtitle, total = "Sell it", "Nothing you track needs it", None
    elif kind == RECYCLE:
        title, total = "Recycle it", None
        subtitle = _recycle_subtitle(recycle_outputs, needed_parts) or verdict.reason
    elif kind == USE:
        title, total = "Use it", None
        subtitle = _recycle_subtitle(recycle_outputs, needed_parts) or verdict.reason
    elif kind == LEARN:
        title, subtitle, total = "Learn it", verdict.reason, None
    else:
        title, total = "Not in the database", None
    if kind != LEARN and verdict.reason.startswith(KNOWN_BLUEPRINT_REASON):
        subtitle = _known_subtitle(kind, subtitle)
    return AdviceCard(kind, title, subtitle, total, owned, rows, owned_sure)


class _Fonts:
    """The card's faces at one scale, with the sans fallback for odd glyphs."""

    def __init__(self, scale: float):
        self.scale = scale
        bold = find_font_file(bold=True, condensed=True)
        semi = find_font_file(bold=False, condensed=True)
        if bold is None or semi is None:
            msg = "no TrueType font available for the advice card"
            raise RuntimeError(msg)
        self.fallback = {
            BOLD: find_font_file(bold=True),
            REGULAR: find_font_file(bold=False),
        }
        self.faces = {
            (size, face): _load_font(bold if face == BOLD else semi, self.px(size))
            for size in (TITLE_SIZE, SUBTITLE_SIZE, BODY_SIZE)
            for face in (BOLD, REGULAR)
        }

    def px(self, value: float) -> int:
        return max(1, round(value * self.scale))

    def get(self, size: int, bold: str) -> ImageFont.FreeTypeFont:
        return self.faces[size, bold]

    def runs(self, text: str, size: int, bold: str):
        font = self.get(size, bold)
        path = self.fallback[bold]
        fb = _load_font(path, self.px(size)) if path else None
        return _runs(text, font, fb)

    def width(self, text: str, size: int, bold: str) -> float:
        return sum(f.getlength(run) for run, f in self.runs(text, size, bold))

    def fit(self, text: str, size: int, bold: str, limit: float) -> str:
        """Text cut with an ellipsis so it is at most limit pixels wide."""
        if self.width(text, size, bold) <= limit:
            return text
        while text and self.width(text + ELLIPSIS, size, bold) > limit:
            text = text[:-1]
        return text.rstrip() + ELLIPSIS


def _wrap(fonts: _Fonts, text: str, size: int, bold: str, limit: float) -> list[str]:
    """Wrap at spaces to limit pixels, an overlong single word is ellipsized."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if current and fonts.width(trial, size, bold) > limit:
            lines.append(current)
            current = word
        else:
            current = trial
    lines.append(current)
    return [fonts.fit(line, size, bold, limit) for line in lines]


class _Canvas:
    """Collects shapes (drawn supersampled) and text (drawn at final size)."""

    def __init__(self, fonts: _Fonts):
        self.fonts = fonts
        self.shapes: list[tuple] = []
        self.texts: list[tuple] = []

    def rect(self, box, fill, radius=0, outline=None):
        self.shapes.append((box, fill, radius, outline))

    def text(self, x, y, text, size, bold, color, anchor="l"):
        """Place text with its line box top at y. anchor l or r sets x's side."""
        self.texts.append((x, y, text, size, bold, color, anchor))

    def render(self, width: int, height: int) -> Image.Image:
        k = SUPERSAMPLE
        layer = Image.new("RGBA", (width * k, height * k), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for box, fill, radius, outline in self.shapes:
            x0, y0, x1, y1 = (round(v * k) for v in box)
            r = round(radius * k)
            if r:
                draw.rounded_rectangle(
                    (x0, y0, x1 - 1, y1 - 1),
                    r,
                    fill=fill,
                    outline=outline,
                    width=k if outline else 0,
                )
            else:
                draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=fill)
        image = layer.resize((width, height), Image.LANCZOS)
        draw = ImageDraw.Draw(image)
        for x, y, text, size, bold, color, anchor in self.texts:
            runs = self.fonts.runs(text, size, bold)
            total = sum(f.getlength(run) for run, f in runs)
            pos = x - total if anchor == "r" else x
            for run, font in runs:
                draw.text((pos, y), run, font=font, fill=color)
                pos += font.getlength(run)
        return image


def _line_height(fonts: _Fonts, size: int, bold: str) -> int:
    ascent, descent = fonts.get(size, bold).getmetrics()
    return ascent + descent


def render_advice_card(card: AdviceCard, scale: float = 1.0) -> Image.Image:
    """Draw the card as an RGBA image, 560 px wide at scale 1.0."""
    fonts = _Fonts(scale)
    s = fonts.px
    canvas = _Canvas(fonts)
    width = s(WIDTH)
    accent = KIND_COLORS.get(card.kind, GREY)

    inner_left = s(MARGIN)
    inner_right = width - s(MARGIN)
    text_left = inner_left + s(STRIPE + PAD)
    text_right = inner_right - s(PAD)
    body_left = s(MARGIN + PAD)
    body_right = width - s(MARGIN + PAD)

    # Header block
    y = s(MARGIN)
    header_top = y
    y += s(16)
    title_lines = _wrap(fonts, card.title, TITLE_SIZE, BOLD, text_right - text_left)
    title_h = _line_height(fonts, TITLE_SIZE, BOLD)
    for line in title_lines:
        canvas.text(text_left, y, line, TITLE_SIZE, BOLD, WHITE)
        y += title_h
    if card.subtitle:
        y += s(2)
        sub_h = _line_height(fonts, SUBTITLE_SIZE, REGULAR)
        sub_limit = text_right - text_left
        sub_lines = _wrap(fonts, card.subtitle, SUBTITLE_SIZE, REGULAR, sub_limit)
        if len(sub_lines) > SUBTITLE_MAX_LINES:
            keep = SUBTITLE_MAX_LINES - 1
            tail = fonts.fit(
                " ".join(sub_lines[keep:]), SUBTITLE_SIZE, REGULAR, sub_limit
            )
            sub_lines = [*sub_lines[:keep], tail]
        for line in sub_lines:
            canvas.text(text_left, y, line, SUBTITLE_SIZE, REGULAR, MUTED)
            y += sub_h
    y += s(16)
    header_bottom = y
    canvas.rect(
        (inner_left, header_top, inner_right, header_bottom),
        HEADER_BG,
        s(HEADER_RADIUS),
    )
    # Stripe with the header's rounded left corners
    canvas.rect(
        (inner_left, header_top, inner_left + s(HEADER_RADIUS * 2), header_bottom),
        accent,
        s(HEADER_RADIUS),
    )
    canvas.rect(
        (
            inner_left + s(STRIPE),
            header_top,
            inner_left + s(HEADER_RADIUS * 2) + 1,
            header_bottom,
        ),
        HEADER_BG,
    )

    body_h = _line_height(fonts, BODY_SIZE, REGULAR)
    bold_h = _line_height(fonts, BODY_SIZE, BOLD)
    y = header_bottom + s(PAD)
    if card.rows:
        total = card.needed_total
        if total is None:
            total = sum(q for _, _, q in card.rows)
        canvas.text(body_left, y, "Needed for", BODY_SIZE, REGULAR, MUTED)
        if card.owned is None or owned_unseen(card):
            amount = f"{total} in total"
        elif card.owned >= total:
            amount = f"{total} of {total}"
        else:
            amount = f"{owned_text(card.owned, card.owned_sure)} of {total}"
        canvas.text(body_right, y, amount, BODY_SIZE, BOLD, AMBER, "r")
        y += body_h + s(8)
        track_h = max(2, s(5))
        canvas.rect((body_left, y, body_right, y + track_h), TRACK, track_h // 2)
        if card.owned and total > 0:
            frac = min(1.0, card.owned / total)
            fill_right = body_left + max(
                track_h, round((body_right - body_left) * frac)
            )
            canvas.rect((body_left, y, fill_right, y + track_h), AMBER, track_h // 2)
        y += track_h + s(ROW_GAP + 4)
        qty_widths = [
            fonts.width(f"\u00d7{q}", BODY_SIZE, BOLD) for _, _, q in card.rows
        ]
        qty_room = max(qty_widths) + s(12)
        name_left = body_left + s(KIND_COLUMN)
        for (kind_label, name, quantity), _ in zip(card.rows, qty_widths, strict=True):
            lines = _wrap(
                fonts, name, BODY_SIZE, REGULAR, body_right - qty_room - name_left
            )
            canvas.text(body_left, y, kind_label, BODY_SIZE, REGULAR, MUTED)
            canvas.text(body_right, y, f"\u00d7{quantity}", BODY_SIZE, BOLD, WHITE, "r")
            for line in lines:
                canvas.text(name_left, y, line, BODY_SIZE, REGULAR, WHITE)
                y += body_h
            y += s(ROW_GAP)
        y -= s(ROW_GAP)
        y += s(PAD - 4)
    if not card.rows:
        y = header_bottom + s(PAD - 4)
    if card.owned is not None:
        canvas.rect((body_left, y, body_right, y + max(1, s(1))), DIVIDER)
        y += max(1, s(1)) + s(14)
        if owned_unseen(card):
            canvas.text(body_left, y, NOT_SAVED, BODY_SIZE, REGULAR, MUTED)
        else:
            shown = owned_text(card.owned, card.owned_sure)
            canvas.text(body_left, y, f"You have {shown}", BODY_SIZE, BOLD, WHITE)
            canvas.text(
                body_right, y, f"{shown} in stash", BODY_SIZE, REGULAR, MUTED, "r"
            )
        y += bold_h + s(PAD - 4)
    elif not card.rows:
        y = header_bottom
    height = y + s(MARGIN)

    body = canvas.render(width, height)
    base = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    panel = Image.new("RGBA", (width * SUPERSAMPLE, height * SUPERSAMPLE), (0, 0, 0, 0))
    ImageDraw.Draw(panel).rounded_rectangle(
        (0, 0, width * SUPERSAMPLE - 1, height * SUPERSAMPLE - 1),
        s(RADIUS) * SUPERSAMPLE,
        fill=PANEL,
        outline=PANEL_BORDER,
        width=max(1, s(1)) * SUPERSAMPLE,
    )
    base.alpha_composite(panel.resize((width, height), Image.LANCZOS))
    base.alpha_composite(body)
    return base
