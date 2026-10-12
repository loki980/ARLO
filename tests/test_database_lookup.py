"""Tests for OCR-tolerant item name lookup."""

from arc_helper.database import Database
from arc_helper.tiers import TierVote


def _db(tmp_path):
    db = Database(db_path=tmp_path / "items.db")
    db.clear()  # pytest may reuse tmp_path across runs; guarantee an empty table
    with db._get_connection() as conn:  # noqa: SLF001 - test seam
        conn.executemany(
            "INSERT INTO items (name, action) VALUES (?, ?)",
            [
                ("Il Toro II", "Sell"),
                ("Anvil IV", "Recycle"),
                ("Rascal I", "Sell"),
                ("Rascal II", "Keep"),
            ],
        )
        conn.commit()
    return db


def test_exact_match(tmp_path):
    assert _db(tmp_path).lookup("Il Toro II").name == "Il Toro II"


def test_case_insensitive(tmp_path):
    assert _db(tmp_path).lookup("ANVIL IV").name == "Anvil IV"


def test_dropped_spaces(tmp_path):
    # OCR ran the words together
    assert _db(tmp_path).lookup("ILTOROII").name == "Il Toro II"


def test_stray_trailing_punctuation(tmp_path):
    assert _db(tmp_path).lookup("ANVIL IV)").name == "Anvil IV"


def test_distinct_tiers_stay_distinct(tmp_path):
    # The numeral must not be normalized away - I and II are different items
    assert _db(tmp_path).lookup("RASCAL I").name == "Rascal I"
    assert _db(tmp_path).lookup("RASCAL II").name == "Rascal II"


def test_unknown_returns_none(tmp_path):
    assert _db(tmp_path).lookup("NONEXISTENT WIDGET") is None


# Tier resolver. The database is reopened after the rows go in because the
# tier index is built when the database opens.

_TIERED = [
    *(f"Anvil {t}" for t in ("I", "II", "III", "IV")),
    *(f"Arpeggio {t}" for t in ("I", "II", "III", "IV")),
    *(f"Il Toro {t}" for t in ("I", "II", "III", "IV")),
    *(f"Bettina {t}" for t in ("I", "II", "III", "IV")),
    "Rascal",
    *(f"Rascal {t}" for t in ("II", "III", "IV")),
    "Anvil Blueprint",
    "Battery",
]


def _tier_db(tmp_path):
    path = tmp_path / "tiers.db"
    db = Database(db_path=path)
    db.clear()
    with db._get_connection() as conn:  # noqa: SLF001 - test seam
        conn.executemany(
            "INSERT INTO items (name, action) VALUES (?, ?)",
            [(name, "Keep") for name in _TIERED],
        )
        conn.commit()
    return Database(db_path=path)


def _vote(tier, confidence):
    return TierVote(tier, confidence, "test")


def test_tier_garbled_ii(tmp_path):
    assert _tier_db(tmp_path).lookup("ANVIL Il").name == "Anvil II"


def test_tier_digit_and_l(tmp_path):
    assert _tier_db(tmp_path).lookup("ANVIL 1l").name == "Anvil II"


def test_tier_pipe_for_i(tmp_path):
    assert _tier_db(tmp_path).lookup("ARPEGGIO |V").name == "Arpeggio IV"


def test_tier_missing_v_resolves_to_iv(tmp_path):
    assert _tier_db(tmp_path).lookup("ARPEGGIO V").name == "Arpeggio IV"


def test_tier_glued(tmp_path):
    assert _tier_db(tmp_path).lookup("ANVILIV").name == "Anvil IV"


def test_tier_il_inside_base_glued(tmp_path):
    assert _tier_db(tmp_path).lookup("ILTOROII").name == "Il Toro II"


def test_tier_il_inside_base_spaced(tmp_path):
    assert _tier_db(tmp_path).lookup("IL TOROII").name == "Il Toro II"


def test_tier_untiered_base(tmp_path):
    assert _tier_db(tmp_path).lookup("RASCAL").name == "Rascal"


def test_tier_glued_beside_untiered_base(tmp_path):
    assert _tier_db(tmp_path).lookup("RASCALII").name == "Rascal II"


def test_tier_exact_tier_one_stays(tmp_path):
    assert _tier_db(tmp_path).lookup("BETTINA I").name == "Bettina I"


def test_tier_resolver_leaves_untiered_names(tmp_path):
    db = _tier_db(tmp_path)
    assert db.lookup("BATTERY").name == "Battery"
    assert db.lookup("ANVIL BLUEPRINT").name == "Anvil Blueprint"
    assert db.lookup("NONEXISTENT WIDGET") is None


def test_tier_vote_overrides_read_tier(tmp_path):
    vote = _vote("III", 0.9)
    assert _tier_db(tmp_path).lookup("ANVIL Il", vote).name == "Anvil III"


def test_tier_vote_overrides_exact_hit(tmp_path):
    # The plain read dropped a stroke and matched Bettina I exactly
    assert _tier_db(tmp_path).lookup("BETTINA I", _vote("II", 0.8)).name == "Bettina II"


def test_tier_vote_picks_tier_for_bare_base(tmp_path):
    # The whitelisted pass dropped the numeral entirely
    assert (
        _tier_db(tmp_path).lookup("ARPEGGIO", _vote("III", 0.8)).name == "Arpeggio III"
    )


def test_tier_weak_vote_is_ignored(tmp_path):
    assert _tier_db(tmp_path).lookup("ANVIL Il", _vote("III", 0.6)).name == "Anvil II"


def test_tier_vote_for_missing_tier_is_ignored(tmp_path):
    assert (
        _tier_db(tmp_path).lookup("ARPEGGIO V", _vote("V", 0.9)).name == "Arpeggio IV"
    )
