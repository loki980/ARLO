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


# Tolerant lookup, the last resort for partial and one-edit reads

_TOLERANT = [
    *(f"Rattler {t}" for t in ("I", "II", "III", "IV")),
    *(f"Burletta {t}" for t in ("I", "II", "III", "IV")),
    "Ferro",
    "Fervo",
]


def _tolerant_db(tmp_path):
    path = tmp_path / "tolerant.db"
    db = Database(db_path=path)
    db.clear()
    with db._get_connection() as conn:  # noqa: SLF001 - test seam
        conn.executemany(
            "INSERT INTO items (name, action) VALUES (?, ?)",
            [(name, "Keep") for name in _TOLERANT],
        )
        conn.commit()
    return Database(db_path=path)


def test_tolerant_dropped_numeral_takes_vote_tier(tmp_path):
    db = _tolerant_db(tmp_path)
    assert db.lookup("RATTLER", _vote("III", 0.9)).name == "Rattler III"


def test_tolerant_one_edit_in_base_keeps_read_tier(tmp_path):
    assert _tolerant_db(tmp_path).lookup("BURLETT IV").name == "Burletta IV"


def test_tolerant_fused_numeral_needs_a_vote(tmp_path):
    db = _tolerant_db(tmp_path)
    assert db.lookup("RATTLERH") is None
    assert db.lookup("RATTLERH", _vote("III", 0.9)).name == "Rattler III"


def test_tolerant_read_matching_two_names_stays_unknown(tmp_path):
    # One substitution from both Ferro and Fervo
    assert _tolerant_db(tmp_path).lookup("FERDO") is None


# Prefix step: a catalog name followed by junk


def _prefix_db(tmp_path):
    db = _db(tmp_path)
    with db._get_connection() as conn:  # noqa: SLF001 - test seam
        conn.executemany(
            "INSERT INTO items (name, action) VALUES (?, ?)",
            [
                ("Cracked Bioscanner", "Keep"),
                ("Wires", "Sell"),
                ("Wires Crossed", "Keep"),
            ],
        )
        conn.commit()
    return db


def test_prefix_catalog_name_plus_junk(tmp_path):
    read = "CRACKED BIOSCANNER ;MER:;{JC:'EDDI:TO CRAFTING MATERIALS. @ MEDICAL"
    assert _prefix_db(tmp_path).lookup(read).name == "Cracked Bioscanner"


def test_prefix_too_short_is_ignored(tmp_path):
    # "Wires" is 5 key characters, under the 6 minimum
    assert _prefix_db(tmp_path).lookup("WIRES ;;JUNK TEXT HERE") is None


def test_prefix_needs_a_word_boundary(tmp_path):
    assert _prefix_db(tmp_path).lookup("CRACKED BIOSCANNERZZ ;;") is None


def test_prefix_prefers_the_longest_name(tmp_path):
    assert _prefix_db(tmp_path).lookup("WIRES CROSSED ;;JUNK").name == "Wires Crossed"


def test_read_matched_before_is_unchanged_by_the_prefix_step(tmp_path):
    db = _prefix_db(tmp_path)
    assert db.lookup("ILTOROII").name == "Il Toro II"
    assert db.lookup("Cracked Bioscanner").name == "Cracked Bioscanner"
