"""
Unit tests for the Scotland crime pure modules (mapping + payload).
These run WITHOUT app.py / Flask / DB — they exercise the two pure functions that
carry the governance-critical behaviour:
  * native categories mapped to display + comparison_quality (never over-claimed)
  * payload assembly: real counts pass through; missing data -> None (never zero)
Run: python3 -m pytest tests/test_scotland_crime.py -q   (from repo root, with scotland/ importable)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scotland.scotland_crime_mapping import map_scotland_category
from scotland.scotland_crime_payload import assemble_scotland_crime


# ---------- mapping ----------

def test_mapping_very_good():
    assert map_scotland_category("Crimes of dishonesty", "Shoplifting") == ("Shoplifting", "very_good")
    assert map_scotland_category("Crimes of dishonesty", "Robbery") == ("Robbery", "very_good")

def test_mapping_housebreaking_good():
    d, q = map_scotland_category("Crimes of dishonesty", "Housebreaking")
    assert d == "Burglary / Housebreaking" and q == "good"

def test_mapping_pedal_cycle_before_motor():
    # "Theft of a pedal cycle" must NOT be swallowed by the motor-vehicle rule
    d, q = map_scotland_category("Crimes of dishonesty", "Theft of a pedal cycle")
    assert d == "Bicycle theft" and q == "approximate"

def test_mapping_motor_vehicle():
    d, q = map_scotland_category("Crimes of dishonesty", "Theft of a motor vehicle")
    assert d == "Vehicle crime" and q == "good"

def test_mapping_vandalism_approximate():
    d, q = map_scotland_category("Damage and reckless behaviour", "Vandalism")
    assert d == "Criminal damage / arson" and q == "approximate"

def test_mapping_public_order_not_comparable():
    d, q = map_scotland_category("Antisocial offences", "Threatening or abusive behaviour")
    assert q == "not_comparable"

def test_mapping_unknown_defaults_not_comparable():
    # An unrecognised native category is shown in Scottish terms only, never claimed comparable
    d, q = map_scotland_category("Some new group", "A brand new offence type")
    assert q == "not_comparable"
    assert d == "A brand new offence type"   # native preserved as the display fallback


# ---------- payload ----------

def _rows_ward():
    return [
        {"native_group": "Crimes of dishonesty", "native_category": "Housebreaking",
         "display_category": "Burglary / Housebreaking", "comparison_quality": "good",
         "recorded_count": 34, "detected_count": 5},
        {"native_group": "Damage and reckless behaviour", "native_category": "Vandalism",
         "display_category": "Criminal damage / arson", "comparison_quality": "approximate",
         "recorded_count": 20, "detected_count": 8},
        {"native_group": "Antisocial offences", "native_category": "Threatening or abusive behaviour",
         "display_category": "Public order / ASB", "comparison_quality": "not_comparable",
         "recorded_count": 0, "detected_count": 0},   # a REAL zero from source is allowed
    ]

def test_payload_real_counts():
    geo = {"type": "multi_member_ward", "code": "S13002930", "name": "Rutherglen South",
           "council": "South Lanarkshire", "period": "2025-26", "source": "Police Scotland"}
    out = assemble_scotland_crime(geo, _rows_ward())
    assert out is not None
    assert out["jurisdiction"] == "scotland"
    m = out["metrics"]
    assert m["total"] == 54                       # 34 + 20 + 0
    assert m["categories"]["Housebreaking"] == 34
    assert m["geography_type"] == "multi_member_ward"
    assert m["period"] == "2025-26"
    assert m["not_comparable_present"] is True
    assert "provisional" in m["notice"].lower()   # ward source flagged as management info
    assert len(m["comparison"]) == 3
    assert m["jurisdiction"] == "scotland"

def test_payload_missing_data_returns_none():
    # No rows -> unavailable (caller emits honest "unavailable", never zero)
    assert assemble_scotland_crime({"type": "multi_member_ward", "period": "2025-26"}, []) is None

def test_payload_all_unknown_counts_returns_none():
    # Rows exist but every recorded_count is None -> no real signal -> None (never zero)
    rows = [{"native_category": "Housebreaking", "recorded_count": None, "detected_count": None}]
    assert assemble_scotland_crime({"type": "multi_member_ward", "period": "2025-26"}, rows) is None

def test_payload_la_source_notice_accredited():
    geo = {"type": "local_authority", "code": "S12000029", "name": "South Lanarkshire",
           "period": "2025-26", "source": "Scottish Government (statistics.gov.scot)"}
    rows = [{"native_group": "Crimes of dishonesty", "native_category": "Crimes of dishonesty",
             "display_category": "Other theft", "comparison_quality": "approximate",
             "recorded_count": 1200, "detected_count": None, "rate_per_10000": 380.5}]
    out = assemble_scotland_crime(geo, rows)
    assert out is not None
    assert "accredited official statistics" in out["metrics"]["notice"].lower()
    assert out["metrics"]["total"] == 1200


# ---------- regression guard: crime/offence split (the 758-vs-483 defect) ----------

def _rows_rutherglen_full():
    """Real group totals for Rutherglen South, 2025-26 (Police Scotland MMW),
    exactly as returned by scotland.crime_by_area. Crimes (1-5) = 483;
    offences (6-8) = 275. The headline MUST be 483, never 758."""
    return [
        {"native_group": "Non-sexual crimes of violence", "native_category": "Non-sexual crimes of violence", "recorded_count": 121, "comparison_quality": "good"},
        {"native_group": "Sexual Crimes",                 "native_category": "Sexual Crimes",                 "recorded_count": 35,  "comparison_quality": "good"},
        {"native_group": "Crimes of Dishonesty",          "native_category": "Crimes of Dishonesty",          "recorded_count": 121, "comparison_quality": "good"},
        {"native_group": "Damage and reckless behaviour", "native_category": "Damage and reckless behaviour", "recorded_count": 86,  "comparison_quality": "approximate"},
        {"native_group": "Crimes against society",        "native_category": "Crimes against society",        "recorded_count": 120, "comparison_quality": "approximate"},
        {"native_group": "Antisocial offences",           "native_category": "Antisocial offences",           "recorded_count": 121, "comparison_quality": "not_comparable"},
        {"native_group": "Miscellaneous Offences",        "native_category": "Miscellaneous Offences",        "recorded_count": 35,  "comparison_quality": "not_comparable"},
        {"native_group": "Road traffic offences",         "native_category": "Road traffic offences",         "recorded_count": 119, "comparison_quality": "not_comparable"},
    ]

def test_crime_total_excludes_offences():
    geo = {"type": "multi_member_ward", "code": "S13003105", "name": "Rutherglen South",
           "council": "South Lanarkshire", "period": "2025-26", "source": "Police Scotland"}
    m = assemble_scotland_crime(geo, _rows_rutherglen_full())["metrics"]
    assert m["total"] == 483, f"crime total must exclude offences (got {m['total']})"
    assert m["total_offences"] == 275, f"offences must be counted separately (got {m['total_offences']})"
    assert m["total"] + m["total_offences"] == 758
    assert len(m["categories"]) == 5           # 5 crime groups only
    assert "Road traffic offences" not in m["categories"]
    assert "Road traffic offences" in m["offence_by_group"]

def test_group_classification():
    from scotland.scotland_crime_payload import classify_group
    assert classify_group("Non-sexual crimes of violence") == (1, "crime")
    assert classify_group("Sexual Crimes") == (2, "crime")     # not swallowed by non-sexual
    assert classify_group("Road traffic offences") == (8, "offence")
    assert classify_group("Antisocial offences") == (6, "offence")
    assert classify_group("Some future group") == (None, "unknown")
    assert classify_group("3") == (3, "crime")                 # numeric group accepted
    assert classify_group("7") == (7, "offence")
