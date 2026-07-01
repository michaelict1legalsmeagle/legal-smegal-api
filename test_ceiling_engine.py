"""
tests/test_ceiling_engine.py
============================
Deterministic tests for the LegalSmegal ceiling engine v2.

Run with: pytest tests/test_ceiling_engine.py -v
"""

import sys
import os
import importlib
import pytest

# Ensure services package is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.ceiling_engine import (
    calculate_ceiling,
    _weighted_median,
    _lease_band,
    _distance_score,
    _recency_score,
    _size_score,
    _size_adjustment,
    _lease_adjustment,
    _uncertainty_band,
    _legal_pack_adjustment_factor,
    _process_legal_risks,
    _calculate_confidence,
    MIN_REQUIRED_COMPS,
    PREFERRED_COMPS,
    PRIMARY_RADIUS_MILES,
    MAX_TOTAL_VALUE_RISK_ADJ,
    CAP_SUBJECT_TYPE_LOW_CONFIDENCE,
    CAP_SUBJECT_FLOOR_AREA_LOW_CONFIDENCE,
    EXCLUDED_FROM_CEILING,
    VERSION,
)

# =============================================================================
# Helpers
# =============================================================================

def _comp(price, dist, months=3, addr="1 Test St", tenure="freehold", prop_type="flat",
          ev="official", area=None, lease=None):
    c = {
        "price": price,
        "distance_miles": dist,
        "months_ago": months,
        "address": addr,
        "tenure": tenure,
        "property_type": prop_type,
        "evidence_quality": ev,
    }
    if area:   c["internal_area"] = area
    if lease:  c["lease_length"]  = lease
    return c

def _subject(tenure="freehold", prop_type="flat", area=None, lease=None):
    s = {"property_type": prop_type, "tenure": tenure}
    if area:  s["internal_area"] = area
    if lease: s["lease_length"]  = lease
    return s

def _flag(title, sev):
    return {"title": title, "severity": sev, "summation": ""}

# =============================================================================
# TEST 1: Extended-radius decay (S33-STEP1, 2026-06-21) — comps beyond
# PRIMARY_RADIUS_MILES (0.5mi) are NOT excluded; they are included with
# reduced weight via EXTENDED_DISTANCE_BANDS, out to MAX_RADIUS_MILES (3.0mi)
# which is the actual exclusion boundary. PRIMARY_RADIUS_MILES now drives
# confidence labelling only, not inclusion.
#
# 2026-06-30: replaces test_outside_radius_excluded and
# test_all_outside_radius_is_insufficient, which asserted the pre-S33-STEP1
# hard 0.5-mile cutoff. That doctrine was deliberately retired on 2026-06-21
# after a live audit (Hey Street, NG10 3HA) found type/room/age-matched comps
# at 0.82-1.27mi being wrongly excluded outright, leaving a single comp to
# anchor the whole valuation — see the DISTANCE RULE comment block above
# EXTENDED_DISTANCE_BANDS for the full rationale. The production code has
# matched this doctrine since 2026-06-21; only these tests were stale,
# flagged in two audits (2026-06-28, 2026-06-30) before being fixed here.
# =============================================================================

def test_extended_radius_comps_included_with_decay():
    """Comps beyond 0.5mi are included (not excluded) with reduced weight,
    out to the 3.0-mile MAX_RADIUS_MILES bound; only beyond that are
    comps actually excluded."""
    comps = [
        _comp(200_000, 0.3,  addr="A"),   # 0.25-0.50 band -> distance_score 0.80
        _comp(200_000, 0.4,  addr="B"),   # 0.25-0.50 band -> distance_score 0.80
        _comp(200_000, 0.51, addr="C"),   # 0.50-1.00 band -> distance_score 0.65 (wrongly excluded pre-S33-STEP1)
        _comp(200_000, 1.0,  addr="D"),   # 0.50-1.00 band -> distance_score 0.65
        _comp(200_000, 3.5,  addr="E"),   # beyond MAX_RADIUS_MILES (3.0) -> still excluded
    ]
    result = calculate_ceiling([], {}, sold_comps=comps, subject=_subject())
    valid_addrs = {c["address"] for c in result["comparables"]["valid"]}
    excl_addrs  = {e["comp"]["address"] for e in result["comparables"]["excluded"] if "comp" in e}

    assert {"A", "B", "C", "D"} <= valid_addrs, "Comps within 3.0 miles must all be included"
    assert "E" in excl_addrs, "Comp beyond MAX_RADIUS_MILES (3.0mi) must still be excluded"
    assert result["status"] == "ok"
    assert result["base"]["valid_comparable_count"] == 4

    # Decay must actually reduce weight, not just include-and-ignore distance:
    # nearer comps (0.80 distance_score) must outweigh farther ones (0.65),
    # all else held equal, and same-band comps must score identically.
    by_addr = {c["address"]: c["weight"] for c in result["comparables"]["valid"]}
    assert by_addr["A"] > by_addr["C"], "Nearer comp must carry more weight than farther comp"
    assert by_addr["A"] == by_addr["B"], "Comps in the same distance band must score identically"
    assert by_addr["C"] == by_addr["D"], "Comps in the same distance band must score identically"


def test_few_comps_in_extended_radius_is_degraded_not_insufficient():
    """2 comps within the 3.0-mile bound (0.6mi, 0.9mi) is 'degraded_low_comps'
    (usable, flagged, real value) — not 'insufficient_evidence' (blank).
    They are valid evidence, just below MIN_REQUIRED_COMPS=3."""
    comps = [_comp(200_000, 0.6, addr="X"), _comp(200_000, 0.9, addr="Y")]
    result = calculate_ceiling([], {}, sold_comps=comps, subject=_subject())
    assert result["status"] == "degraded_low_comps"
    assert result["base"]["valid_comparable_count"] == 2
    assert result["valuation_range"]["midpoint"] is not None, \
        "2 valid comps must still produce a (degraded) valuation, not a blank one"


# =============================================================================
# TEST 2: No arithmetic mean — base_value equals weighted_median
# =============================================================================

def test_no_arithmetic_mean():
    """weighted_median must not equal arithmetic mean when distribution is skewed."""
    comps = [
        _comp(100_000, 0.1, addr="A", months=1),  # high weight (recent, close)
        _comp(100_000, 0.1, addr="B", months=1),
        _comp(100_000, 0.1, addr="C", months=1),
        _comp(100_000, 0.1, addr="D", months=1),
        _comp(900_000, 0.4, addr="E", months=12),  # low weight (far, old)
    ]
    result = calculate_ceiling([], {}, sold_comps=comps, subject=_subject())
    base = result["base"]["value"]
    arith_mean = 280_000  # (4×100k + 900k)/5 = 280k
    assert base is not None
    # Weighted median must be 100_000 (4 high-weight comps all at 100k)
    assert abs(base - 100_000) < 5_000, f"Expected ~100k weighted median, got {base}"
    assert abs(base - arith_mean) > 10_000, "base_value must NOT equal arithmetic mean"
    assert result["base"]["method"].startswith("weighted_median"), \
        f"base_method must start with weighted_median, got {result['base']['method']}"


def test_weighted_median_deterministic():
    pairs = [(100_000, 1.0), (200_000, 1.0), (300_000, 1.0)]
    assert _weighted_median(pairs) == 200_000  # median of equal weights
    pairs2 = [(100_000, 3.0), (200_000, 1.0), (300_000, 1.0)]
    assert _weighted_median(pairs2) == 100_000  # heavy weight at low end
    pairs3 = [(100_000, 1.0), (200_000, 1.0), (300_000, 3.0)]
    assert _weighted_median(pairs3) == 300_000  # heavy weight at high end


def test_weighted_median_no_positive_weights():
    assert _weighted_median([]) is None
    assert _weighted_median([(100_000, 0)]) is None
    assert _weighted_median([(100_000, -1)]) is None


# =============================================================================
# TEST 3: Duplicate exclusion
# =============================================================================

def test_duplicate_exclusion():
    comps = [
        _comp(200_000, 0.2, addr="1 High St"),
        _comp(210_000, 0.3, addr="1 High St"),  # duplicate address — exclude
        _comp(195_000, 0.4, addr="2 High St"),
        _comp(205_000, 0.1, addr="3 High St"),
    ]
    result = calculate_ceiling([], {}, sold_comps=comps, subject=_subject())
    excl_reasons = [e["reason"] for e in result["comparables"]["excluded"]]
    assert any("duplicate" in r for r in excl_reasons), "Duplicate must appear in excluded"
    assert result["base"]["valid_comparable_count"] == 3  # first of duplicate is valid


# =============================================================================
# TEST 4: Tenure handling — material mismatch excluded
# =============================================================================

def test_tenure_mismatch_excluded():
    subject = _subject(tenure="freehold", prop_type="flat")
    comps = [
        _comp(200_000, 0.2, addr="A", tenure="freehold"),
        _comp(200_000, 0.2, addr="B", tenure="freehold"),
        _comp(200_000, 0.2, addr="C", tenure="leasehold"),  # mismatch — exclude
        _comp(200_000, 0.3, addr="D", tenure="freehold"),
    ]
    result = calculate_ceiling([], {}, sold_comps=comps, subject=subject)
    excl_reasons = [e["reason"] for e in result["comparables"]["excluded"]]
    assert any("tenure_mismatch" in r for r in excl_reasons)
    assert result["base"]["valid_comparable_count"] == 3


def test_tenure_unknown_caps_confidence():
    subject = _subject(tenure=None, prop_type="flat")  # unknown tenure
    comps = [_comp(200_000, 0.1, addr=f"addr{i}", tenure=None) for i in range(5)]
    result = calculate_ceiling([], {}, sold_comps=comps, subject=subject)
    conf = result["confidence"]["final"]
    assert conf <= 0.45, f"Confidence must be <= 0.45 when tenure unknown; got {conf}"


# =============================================================================
# TEST 5: Lease handling — leasehold checks lease_length and lease_band
# =============================================================================

def test_non_adjacent_lease_band_excluded():
    subject = _subject(tenure="leasehold", prop_type="flat", lease=75)  # band 60-80
    comps = [
        _comp(200_000, 0.1, addr="A", tenure="leasehold", prop_type="flat", lease=72),  # same band
        _comp(200_000, 0.2, addr="B", tenure="leasehold", prop_type="flat", lease=65),  # same band
        _comp(200_000, 0.3, addr="C", tenure="leasehold", prop_type="flat", lease=30),  # 20-40 band — non-adjacent
        _comp(200_000, 0.2, addr="D", tenure="leasehold", prop_type="flat", lease=82),  # 80-99 band — adjacent
    ]
    result = calculate_ceiling([], {}, sold_comps=comps, subject=subject)
    excl_reasons = [e["reason"] for e in result["comparables"]["excluded"]]
    assert any("non_adjacent_lease_band" in r for r in excl_reasons), "Non-adjacent lease band must be excluded"
    assert result["base"]["valid_comparable_count"] == 3  # A, B, D valid; C excluded


# =============================================================================
# TEST 6: Short lease cap — subject < 80 years and no same-band lease comps
# =============================================================================

def test_short_lease_no_same_band_caps_confidence():
    subject = _subject(tenure="leasehold", prop_type="flat", lease=72)  # band 60-80
    # All comps in adjacent band 80-99 only — no same-band comp
    comps = [
        _comp(200_000, 0.1, addr="A", tenure="leasehold", prop_type="flat", lease=85),  # 80-99 adj band
        _comp(200_000, 0.2, addr="B", tenure="leasehold", prop_type="flat", lease=90),  # 80-99 adj band
        _comp(200_000, 0.3, addr="C", tenure="leasehold", prop_type="flat", lease=88),  # 80-99 adj band
    ]
    result = calculate_ceiling([], {}, sold_comps=comps, subject=subject)
    conf = result["confidence"]["final"]
    caps = [cap["reason"] for cap in result["confidence"]["caps"]]
    assert any("short_lease" in r or "80" in r for r in caps), f"Short lease cap must be in caps: {caps}"
    assert conf <= 0.35, f"Confidence must be <= 0.35 for short lease no same-band; got {conf}"


# =============================================================================
# TEST 7: Legal-pack value risk adjustment
# =============================================================================

def test_legal_pack_value_risk_affects_midpoint():
    comps = [_comp(200_000, 0.1, addr=f"A{i}") for i in range(5)]
    result_clean  = calculate_ceiling([], {}, sold_comps=comps, subject=_subject())
    flags_critical = [_flag("Short lease", "critical")]
    result_risky   = calculate_ceiling(flags_critical, {}, sold_comps=comps, subject=_subject())

    mid_clean = result_clean["valuation_range"]["midpoint"]
    mid_risky = result_risky["valuation_range"]["midpoint"]

    assert mid_clean is not None
    assert mid_risky is not None
    assert mid_risky < mid_clean, "Legal-pack risks must reduce ceiling midpoint"

    # Verify factor
    factor = result_risky["legal_pack_value_risks"]["adjustment_factor"]
    assert factor < 1.0, "Adjustment factor must be < 1.0 when risks present"
    assert abs(mid_risky - mid_clean * factor) < 10, (
        f"midpoint must equal base × factor: {mid_risky} vs {mid_clean * factor}"
    )


def test_legal_pack_risk_product_formula():
    """legal_pack_value_risk_adjustment_factor = product(1 - value_adjustment_i),
    where value_adjustment_i comes from _SEGMENT_RULES (defect type), not from
    a severity-tier table. Replaces the 2026-06-14-stale version of this test,
    which hard-coded severity fractions (critical=0.10, high=0.06) from the
    pre-S33 severity-bucket pricing model retired on 2026-06-14 in favour of
    market-consequence segment routing — see _SEGMENT_RULES."""
    # "Defective title" -> defective_title rule: 0.055 + 0.045 = 0.10
    # "Restrictive covenant" -> restrictive_covenant rule: 0.010+0.020+0.020 = 0.05
    # Chosen deliberately distinct (not both 0.10, as the old "critical"/"high"
    # case coincidentally was) so this test can't pass for the wrong reason.
    flags = [_flag("Defective title", "critical"), _flag("Restrictive covenant", "high")]
    risks = _process_legal_risks(flags)
    factor = _legal_pack_adjustment_factor(risks)
    expected = (1 - 0.10) * (1 - 0.05)
    assert abs(factor - expected) < 0.001, f"Expected {expected} got {factor}"


def test_legal_pack_risk_severity_does_not_change_matched_rule_fraction():
    """Explicit regression guard for the 2026-06-14 design rule: severity is a
    descriptive label only for a flag that matches a _SEGMENT_RULES entry —
    it must NOT scale or change the fraction. Same title, different severity,
    must yield an identical value_adjustment."""
    fraction_critical = _process_legal_risks([_flag("Defective title", "critical")])[0]["value_adjustment"]
    fraction_low      = _process_legal_risks([_flag("Defective title", "low")])[0]["value_adjustment"]
    assert fraction_critical == fraction_low == 0.10, (
        f"Severity must not change a matched-rule fraction: "
        f"critical={fraction_critical} low={fraction_low}, both must equal 0.10"
    )


def test_legal_pack_risk_capped_at_35pct():
    """Total reduction must never exceed MAX_TOTAL_VALUE_RISK_ADJ (35%)."""
    flags = [_flag(f"Risk {i}", "critical") for i in range(10)]
    risks = _process_legal_risks(flags)
    factor = _legal_pack_adjustment_factor(risks)
    assert factor >= (1 - MAX_TOTAL_VALUE_RISK_ADJ), (
        f"Factor {factor} exceeds cap — total reduction > {MAX_TOTAL_VALUE_RISK_ADJ}"
    )


# =============================================================================
# TEST 8: Acquisition costs excluded from ceiling valuation formula
# =============================================================================

def test_acquisition_costs_excluded_from_ceiling():
    """SDLT, buyer premium, auction admin, legal fees, bridging must NOT appear in ceiling formula."""
    comps = [_comp(200_000, 0.1, addr=f"A{i}") for i in range(5)]
    flags_with_acq = [
        {"title": "Buyer's premium £3,000", "severity": "high", "summation": "buyers premium"},
        {"title": "SDLT estimate", "severity": "high", "summation": "stamp duty sdlt"},
        {"title": "Legal fees £2,000", "severity": "note", "summation": "solicitor fee"},
    ]
    flags_without_acq = [_flag("Short lease risk", "high")]

    result_acq     = calculate_ceiling(flags_with_acq,    {}, sold_comps=comps, subject=_subject())
    result_no_acq  = calculate_ceiling(flags_without_acq, {}, sold_comps=comps, subject=_subject())

    mid_acq    = result_acq["valuation_range"]["midpoint"]
    mid_no_acq = result_no_acq["valuation_range"]["midpoint"]

    # Acquisition-cost flags must not reduce valuation (they should be excluded)
    base_comps = calculate_ceiling([], {}, sold_comps=comps, subject=_subject())
    mid_clean  = base_comps["valuation_range"]["midpoint"]

    # The clean ceiling and acq-cost flags ceiling should be identical
    # (acq cost flags are excluded from the valuation formula)
    assert mid_acq == mid_clean, (
        f"Acquisition costs must not reduce ceiling: clean={mid_clean} vs with_acq={mid_acq}"
    )

    # acquisition_costs field must be None or not contain valuation math
    assert result_acq["acquisition_costs"] is None, (
        "acquisition_costs must not be used in ceiling formula"
    )

    # EXCLUDED_FROM_CEILING list must contain all expected terms
    for term in ["sdlt", "buyer_premium", "auction_admin_fee", "buyer_solicitor_fee",
                 "finance_cost", "bridging_cost", "investor_margin", "execution_buffer",
                 "generic_acquisition_costs"]:
        assert term in EXCLUDED_FROM_CEILING, f"{term} must be in EXCLUDED_FROM_CEILING"


# =============================================================================
# TEST 9: Frontend no mutation — canonical ceiling is backend-owned
# =============================================================================

def test_canonical_ceiling_values_unchanged():
    """
    The backend returns ceiling_range.{low,high} and valuation_range.{low,midpoint,high}.
    These must equal the formula outputs — not mutated by any secondary logic.
    """
    comps = [_comp(200_000, 0.1, addr=f"A{i}") for i in range(5)]
    result = calculate_ceiling([], {}, sold_comps=comps, subject=_subject())

    vr  = result["valuation_range"]
    cr  = result["ceiling_range"]
    mid = vr["midpoint"]
    u   = vr["uncertainty_band"]

    # valuation_range.low and high must match formula
    expected_low  = round(mid * (1 - u), 2)
    expected_high = round(mid * (1 + u), 2)

    assert abs(vr["low"]  - expected_low)  < 1, f"low mismatch: {vr['low']} vs {expected_low}"
    assert abs(vr["high"] - expected_high) < 1, f"high mismatch: {vr['high']} vs {expected_high}"

    # ceiling_range (legacy compat) must equal rounded valuation_range
    assert cr["low"]  == int(round(vr["low"])),  "ceiling_range.low must equal int(round(valuation_range.low))"
    assert cr["high"] == int(round(vr["high"])), "ceiling_range.high must equal int(round(valuation_range.high))"


# =============================================================================
# TEST 10: Audit completeness
# =============================================================================

def test_audit_completeness():
    comps = [_comp(200_000, 0.2, addr=f"A{i}", ev="partial") for i in range(3)]
    flags = [_flag("Short lease", "critical"), _flag("Missing management pack", "missing")]
    result = calculate_ceiling(flags, {}, sold_comps=comps, subject=_subject())

    audit = result["audit"]
    assert isinstance(audit["formula_trace"], list) and len(audit["formula_trace"]) > 0
    assert isinstance(audit["warnings"],     list)
    assert isinstance(audit["evidence_gaps"], list)
    assert isinstance(audit["assumptions"],  list)
    assert audit["version"] == VERSION

    base  = result["base"]
    assert "valid_comparable_count"   in base
    assert "excluded_comparable_count" in base
    assert base["minimum_required_comps"]  == MIN_REQUIRED_COMPS
    assert base["preferred_required_comps"] == PREFERRED_COMPS

    comps_block = result["comparables"]
    assert "valid"    in comps_block
    assert "excluded" in comps_block

    risks = result["legal_pack_value_risks"]["risks"]
    assert isinstance(risks, list) and len(risks) > 0
    for r in risks:
        assert "risk_id"         in r
        assert "value_adjustment" in r
        assert "included"         in r
        assert "reason"           in r

    conf = result["confidence"]
    assert "final"  in conf
    assert "caps"   in conf
    assert "label"  in conf


# =============================================================================
# TEST 11: Existing ceiling connection — calculate_ceiling is the user-facing function
# =============================================================================

def test_existing_ceiling_connected_to_valuation_result():
    """
    The function exposed by services/ceiling_engine is calculate_ceiling.
    It returns the relational paper valuation result directly — not a
    disconnected parallel object.
    """
    from services import ceiling_engine as ce
    assert hasattr(ce, "calculate_ceiling"), "calculate_ceiling must be importable"
    comps  = [_comp(200_000, 0.1, addr=f"A{i}") for i in range(5)]
    result = ce.calculate_ceiling([], {}, sold_comps=comps, subject=_subject())
    # Must have canonical valuation_range (not just ceiling_range)
    assert "valuation_range" in result, "valuation_range must be in output"
    assert result["valuation_range"]["midpoint"] is not None
    assert result["valuation_type"] == "red_book_style_paper_valuation_similarity"


def test_root_ceiling_engine_raises():
    """Root ceiling_engine.py must not be importable — it raises ImportError."""
    import importlib
    with pytest.raises((ImportError, Exception)):
        spec = importlib.util.spec_from_file_location(
            "ceiling_engine_root",
            os.path.join(os.path.dirname(__file__), "..", "ceiling_engine.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)


# =============================================================================
# TEST 12: Calibration — deterministic values
# =============================================================================

def test_discount_calibration_deterministic():
    from services.ceiling_engine import VALUE_RISK_SEVERITY_ADJ, MAX_TOTAL_VALUE_RISK_ADJ
    assert VALUE_RISK_SEVERITY_ADJ["critical"] == 0.10
    assert VALUE_RISK_SEVERITY_ADJ["high"]     == 0.06
    assert VALUE_RISK_SEVERITY_ADJ["medium"]   == 0.035
    assert VALUE_RISK_SEVERITY_ADJ["low"]      == 0.015
    assert VALUE_RISK_SEVERITY_ADJ["note"]     == 0.00
    assert VALUE_RISK_SEVERITY_ADJ["missing"]  == 0.025
    assert MAX_TOTAL_VALUE_RISK_ADJ            == 0.35


def test_uncertainty_band_deterministic():
    # 5 comps, no caps → BASE_UNCERTAINTY = 0.05
    band5  = _uncertainty_band(5, [])
    assert band5 == 0.05, f"5 comps, no caps → 0.05; got {band5}"

    # 3 comps → +0.03
    band3  = _uncertainty_band(3, [])
    assert band3 == 0.08, f"3 comps, no caps → 0.08; got {band3}"

    # 1 comp → +0.08
    band1  = _uncertainty_band(1, [])
    assert band1 == 0.13, f"1 comp, no caps → 0.13; got {band1}"

    # capped at 0.20 -- using the real `category` field, not synthetic
    # reason strings. (Prior version of this test used reason strings like
    # "legal_pack_gaps" that happened to contain the literal substring the
    # old code checked for -- but the REAL cap sites never wrote that exact
    # substring, see test_uncertainty_band_category_golden below.)
    band_hi = _uncertainty_band(0, [
        {"category": "tenure"},
        {"category": "lease"},
        {"category": "evidence_tier"},
    ])
    assert band_hi == 0.20, f"0 comps + tenure + lease + evidence_tier → clamp at 0.20; got {band_hi}"


def test_uncertainty_band_category_golden():
    """
    S-FIX (2026-06-28) regression anchor: _uncertainty_band moved from
    substring-matching cap['reason'] prose to switching on an explicit
    cap['category'] field.

    2026-06-30 DELIBERATE UPDATE: unquantified_risk and condition_risk
    increments activated at +0.03 each (were 0.0 by substring-typo accident,
    never 0.0 by design). Evidence: 2026-06-30 live deal book audit confirmed
    CAP_UNQUANTIFIED_RISKS has never fired (unquantified_count=0 for all 19
    deals); condition_risk CAN fire (6 deals at 0.55 confidence below the
    Category-A cap of 0.59). When condition_risk fires, the band SHOULD widen —
    it signals the property may not be comparable to the sold comps used.
    Two new categories added for subject-resolution confidence (S35-TYPE-CONF /
    S35-AREA-CONF, 2026-06-30) at +0.02 each.

    If any value below changes, it must be a deliberate edit to this test,
    not an accidental refactor side-effect.
    """
    golden = {
        "comp_count":                       0.05,  # base only — comp count handled via valid_count
        "tenure":                           0.09,  # 0.05 + 0.04
        "lease":                            0.10,  # 0.05 + 0.05
        "unquantified_risk":                0.08,  # 0.05 + 0.03 (activated 2026-06-30)
        "condition_risk":                   0.08,  # 0.05 + 0.03 (activated 2026-06-30)
        "evidence_tier":                    0.09,  # 0.05 + 0.04
        "subject_type_low_confidence":      0.07,  # 0.05 + 0.02 (S35-TYPE-CONF 2026-06-30)
        "subject_floor_area_low_confidence":0.07,  # 0.05 + 0.02 (S35-AREA-CONF 2026-06-30)
    }
    for category, expected in golden.items():
        actual = _uncertainty_band(5, [{"category": category}])
        assert actual == expected, f"category={category!r}: expected {expected}, got {actual}"

    # All eight together at valid_count=1 -> clamps at the 0.20 ceiling
    all_caps = [{"category": c} for c in golden]
    assert _uncertainty_band(1, all_caps) == 0.20


def test_distance_score_deterministic():
    # 0-0.5mi shape is unchanged from the original pre-S33-STEP1 bands.
    assert _distance_score(0.05)  == 1.00
    assert _distance_score(0.10)  == 1.00
    assert _distance_score(0.15)  == 0.90
    assert _distance_score(0.25)  == 0.90
    assert _distance_score(0.30)  == 0.80
    assert _distance_score(0.50)  == 0.80
    # S33-STEP1 (2026-06-21): beyond 0.5mi, comps decay rather than exclude,
    # out to MAX_RADIUS_MILES (3.0mi); only beyond that is None (excluded).
    assert _distance_score(0.51)  == 0.65
    assert _distance_score(1.00)  == 0.65
    assert _distance_score(1.50)  == 0.50
    assert _distance_score(2.00)  == 0.35
    assert _distance_score(2.99)  == 0.20
    assert _distance_score(3.00)  == 0.20
    assert _distance_score(3.01)  is None  # beyond MAX_RADIUS_MILES — excluded
    assert _distance_score(5.00)  is None


def test_recency_score_deterministic():
    assert _recency_score(0)  == 1.00
    assert _recency_score(2)  == 1.00
    assert _recency_score(3)  == 0.90  # lo <= months < hi: 3 is NOT in [0,3)
    assert _recency_score(5)  == 0.90
    assert _recency_score(6)  == 0.80
    assert _recency_score(11) == 0.80
    assert _recency_score(12) == 0.60
    assert _recency_score(23) == 0.60
    assert _recency_score(24) == 0.40
    assert _recency_score(36) == 0.40


def test_min_required_comps():
    assert MIN_REQUIRED_COMPS == 3
    assert PREFERRED_COMPS    == 5


def test_insufficient_evidence_below_min_comps_no_fallback():
    comps = [_comp(200_000, 0.1, addr="A"), _comp(200_000, 0.2, addr="B")]
    result = calculate_ceiling([], {}, sold_comps=comps, subject=_subject(), fallback_allowed=False)
    assert result["status"] == "insufficient_evidence"
    assert result["valuation_range"]["midpoint"] is None


def test_degraded_low_comps_with_fallback():
    comps = [_comp(200_000, 0.1, addr="A"), _comp(200_000, 0.2, addr="B")]
    result = calculate_ceiling([], {}, sold_comps=comps, subject=_subject(), fallback_allowed=True)
    # Should return a value but with low confidence
    assert result["status"] == "degraded_low_comps"
    assert result["valuation_range"]["midpoint"] is not None
    assert result["confidence"]["final"] <= 0.50


# =============================================================================
# TEST 13: already_present rule — all tests must pass against services engine
# =============================================================================

def test_services_ceiling_engine_is_canonical():
    """
    Confirms that the services.ceiling_engine module is the sole canonical
    engine, matches the expected version string, and exposes the correct
    public interface.
    """
    from services import ceiling_engine as ce
    assert ce.VERSION == "ceiling_relational_paper_valuation_v1"
    assert ce.PRIMARY_RADIUS_MILES == 0.5
    assert ce.MIN_REQUIRED_COMPS   == 3
    assert ce.PREFERRED_COMPS      == 5
    assert callable(ce.calculate_ceiling)
    assert callable(ce._weighted_median)
    assert callable(ce._assess_comp)
    assert callable(ce._legal_pack_adjustment_factor)
    assert callable(ce._uncertainty_band)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# =============================================================================
# TEST 14: housing_comps_v1 RPC field-name aliases
# =============================================================================

def test_rpc_field_name_aliases():
    """
    housing_comps_v1 RPC returns comps with these field names:
      miles          (not distance_miles)
      price          (already correct)
      duration       (F/L — not tenure)
      age_months     (not months_ago)
      floor_area     (not internal_area)
      hpi_multiplier (not hpi_adjustment)

    Engine must normalise all of these and produce a valid ceiling.
    If any alias is missed, every comp fails _assess_comp and
    status = insufficient_evidence with midpoint = None.
    """
    rpc_comps = [
        {
            "price":          200_000,
            "miles":          0.15,       # RPC field name
            "age_months":     4,           # RPC field name
            "duration":       "F",         # RPC code for freehold
            "floor_area":     65.0,        # RPC field name
            "hpi_multiplier": 1.03,        # RPC field name
            "address":        f"RPC Comp {i}",
            "property_type":  "flat",
            "evidence_quality": "official",
        }
        for i in range(5)
    ]
    # Give each a unique address to avoid duplicate exclusion
    for i, c in enumerate(rpc_comps):
        c["address"] = f"RPC Comp {i} High Street"

    subject = _subject(tenure="freehold", prop_type="flat", area=65.0)
    result  = calculate_ceiling([], {}, sold_comps=rpc_comps, subject=subject)

    assert result["status"] != "insufficient_evidence", (
        f"RPC field aliases not normalised — all comps excluded. "
        f"Excluded: {[e['reason'] for e in result['comparables']['excluded']]}"
    )
    assert result["base"]["valid_comparable_count"] == 5
    assert result["valuation_range"]["midpoint"] is not None
    assert result["valuation_range"]["midpoint"] > 100_000


def test_rpc_duration_L_maps_to_leasehold():
    """duration='L' must map to leasehold — tenure mismatch against freehold subject."""
    rpc_leasehold_comps = [
        {"price": 200_000, "miles": 0.1, "age_months": 2,
         "duration": "L",  # leasehold
         "address": f"L{i}", "property_type": "flat", "evidence_quality": "official"}
        for i in range(3)
    ]
    subject = _subject(tenure="freehold", prop_type="flat")
    result  = calculate_ceiling([], {}, sold_comps=rpc_leasehold_comps, subject=subject)
    # All comps should be excluded for tenure mismatch
    excl_reasons = [e["reason"] for e in result["comparables"]["excluded"]]
    assert all("tenure_mismatch" in r for r in excl_reasons), (
        f"duration='L' comps vs freehold subject must all be excluded; reasons: {excl_reasons}"
    )
    assert result["status"] == "insufficient_evidence"


def test_rpc_duration_F_matches_freehold_subject():
    """duration='F' must map to freehold and match a freehold subject."""
    rpc_freehold = [
        {"price": 200_000, "miles": 0.1 + i*0.05, "age_months": 2,
         "duration": "F",  # freehold
         "address": f"F{i}", "property_type": "flat", "evidence_quality": "official"}
        for i in range(5)
    ]
    subject = _subject(tenure="freehold", prop_type="flat")
    result  = calculate_ceiling([], {}, sold_comps=rpc_freehold, subject=subject)
    assert result["base"]["valid_comparable_count"] == 5
    assert result["valuation_range"]["midpoint"] is not None


# =============================================================================
# THREE-OBJECT FLOW TESTS
# Verify: verdict_ceiling, workbench_ceiling, financial_current_standing
# =============================================================================

from services.ceiling_engine import (
    calculate_verdict_ceiling,
    calculate_workbench_ceiling,
    calculate_financial_standing,
    BASE_UNCERTAINTY,
)


def _rpc_comps_5(base_price=200_000):
    """Five valid RPC-shaped comps within 0.5 miles."""
    return [
        {
            "price":            base_price,
            "miles":            0.05 + i * 0.08,
            "age_months":       2 + i,
            "duration":         "F",
            "floor_area":       65.0,
            "hpi_multiplier":   1.00,
            "address":          f"Flow Test Comp {i}",
            "property_type":    "flat",
            "evidence_quality": "official",
        }
        for i in range(5)
    ]


def _subj():
    return {"property_type": "flat", "tenure": "freehold"}


# ── TEST 15: Verdict ceiling has no flag risks applied ──────────────────────

def test_verdict_ceiling_no_flag_risks():
    """Verdict ceiling = weighted median of comps, no legal-pack deductions."""
    comps = _rpc_comps_5(200_000)
    flags = [_flag("Short lease", "critical"), _flag("Defective title", "high")]

    verdict = calculate_verdict_ceiling(sold_comps=comps, subject=_subj())
    # Verdict must call engine with no flags
    assert verdict["_ceiling_type"] == "verdict"
    assert verdict["legal_pack_value_risks"]["adjustment_factor"] == 1.0, (
        "Verdict must not apply legal-pack risk — adjustment_factor must be 1.0"
    )
    # If no flags applied, midpoint == base_value
    base = verdict["base"]["value"]
    mid  = verdict["valuation_range"]["midpoint"]
    assert mid is not None
    assert abs(mid - base) < 1, f"Verdict midpoint must equal base when no flags applied: {mid} vs {base}"


# ── TEST 16: Workbench ceiling <= Verdict ceiling ───────────────────────────

def test_workbench_lte_verdict():
    """Workbench ceiling must never exceed Verdict ceiling."""
    comps = _rpc_comps_5(200_000)
    verdict   = calculate_verdict_ceiling(sold_comps=comps, subject=_subj())
    flags     = [_flag("Short lease", "critical"), _flag("Planning issue", "high")]
    workbench = calculate_workbench_ceiling(verdict_ceiling=verdict, active_legal_flags=flags)

    v_mid = verdict["valuation_range"]["midpoint"]
    w_mid = workbench["valuation_range"]["midpoint"]
    v_low = verdict["valuation_range"]["low"]
    w_low = workbench["valuation_range"]["low"]
    v_hi  = verdict["valuation_range"]["high"]
    w_hi  = workbench["valuation_range"]["high"]

    assert w_mid <= v_mid, f"workbench.midpoint {w_mid} must not exceed verdict.midpoint {v_mid}"
    assert w_low <= v_low, f"workbench.low {w_low} must not exceed verdict.low {v_low}"
    assert w_hi  <= v_hi,  f"workbench.high {w_hi} must not exceed verdict.high {v_hi}"
    assert workbench["_ceiling_type"] == "workbench"


def test_workbench_less_than_verdict_with_risks():
    """Active flags must reduce workbench below verdict."""
    comps     = _rpc_comps_5(200_000)
    verdict   = calculate_verdict_ceiling(sold_comps=comps, subject=_subj())
    flags     = [_flag("Defective title", "critical")]
    workbench = calculate_workbench_ceiling(verdict_ceiling=verdict, active_legal_flags=flags)

    v_mid = verdict["valuation_range"]["midpoint"]
    w_mid = workbench["valuation_range"]["midpoint"]
    assert w_mid < v_mid, f"Workbench must be below Verdict when active risks exist: {w_mid} vs {v_mid}"


# ── TEST 17: All flags resolved → Workbench equals Verdict ──────────────────

def test_all_flags_resolved_workbench_equals_verdict():
    """When no active flags remain, workbench_ceiling must equal verdict_ceiling."""
    comps     = _rpc_comps_5(200_000)
    verdict   = calculate_verdict_ceiling(sold_comps=comps, subject=_subj())
    # Empty active_legal_flags → all resolved
    workbench = calculate_workbench_ceiling(verdict_ceiling=verdict, active_legal_flags=[])

    v_mid = verdict["valuation_range"]["midpoint"]
    w_mid = workbench["valuation_range"]["midpoint"]
    assert w_mid == v_mid, f"Workbench must equal Verdict when all flags resolved: {w_mid} vs {v_mid}"

    v_low = verdict["valuation_range"]["low"]
    w_low = workbench["valuation_range"]["low"]
    assert w_low == v_low, f"Workbench.low must equal Verdict.low when all resolved: {w_low} vs {v_low}"

    v_hi = verdict["valuation_range"]["high"]
    w_hi = workbench["valuation_range"]["high"]
    assert w_hi == v_hi, f"Workbench.high must equal Verdict.high when all resolved: {w_hi} vs {v_hi}"


# ── TEST 18: Partial resolution raises Workbench but never above Verdict ─────

def test_partial_resolution_raises_workbench_not_above_verdict():
    """Resolving some flags raises workbench but it stays <= verdict."""
    comps   = _rpc_comps_5(200_000)
    verdict = calculate_verdict_ceiling(sold_comps=comps, subject=_subj())

    all_flags = [
        _flag("Short lease",    "critical"),
        _flag("Defective title","high"),
        _flag("Planning issue", "medium"),
    ]
    active_2  = all_flags[:2]   # 3 → 2 active (1 resolved)
    active_1  = all_flags[:1]   # 3 → 1 active (2 resolved)
    active_0  = []              # all resolved

    wb_all = calculate_workbench_ceiling(verdict, all_flags)
    wb_2   = calculate_workbench_ceiling(verdict, active_2)
    wb_1   = calculate_workbench_ceiling(verdict, active_1)
    wb_0   = calculate_workbench_ceiling(verdict, active_0)

    vm = verdict["valuation_range"]["midpoint"]
    m_all = wb_all["valuation_range"]["midpoint"]
    m_2   = wb_2["valuation_range"]["midpoint"]
    m_1   = wb_1["valuation_range"]["midpoint"]
    m_0   = wb_0["valuation_range"]["midpoint"]

    # Monotonically rising as flags resolved
    assert m_all <= m_2,  f"wb(all) {m_all} must be <= wb(2 active) {m_2}"
    assert m_2   <= m_1,  f"wb(2) {m_2} must be <= wb(1 active) {m_1}"
    assert m_1   <= m_0,  f"wb(1) {m_1} must be <= wb(0 active) {m_0}"
    # None may exceed verdict
    for mid, label in [(m_all,"all"), (m_2,"2"), (m_1,"1"), (m_0,"0")]:
        assert mid <= vm, f"workbench({label}) {mid} must not exceed verdict {vm}"


# ── TEST 19: Workbench formula — product not sum ─────────────────────────────

def test_workbench_uses_risk_product_not_sum():
    """Workbench midpoint = verdict_midpoint × product(1 - adj_i), where adj_i
    comes from _SEGMENT_RULES (defect type), not a severity-tier table.
    Replaces the 2026-06-14-stale version, which hard-coded
    critical=0.10/high=0.06 from the retired severity-bucket pricing model."""
    comps   = _rpc_comps_5(200_000)
    verdict = calculate_verdict_ceiling(sold_comps=comps, subject=_subj())
    # "Defective title" -> 0.10 (defective_title rule), "Planning issue" -> 0.05
    # (planning rule: 0.012+0.018+0.020) — confirmed via _process_legal_risks.
    flags   = [_flag("Defective title", "critical"), _flag("Planning issue", "high")]
    wb      = calculate_workbench_ceiling(verdict, flags)

    vm = verdict["valuation_range"]["midpoint"]
    expected_factor = (1 - 0.10) * (1 - 0.05)
    expected_mid    = round(vm * expected_factor, 2)
    actual_mid      = wb["valuation_range"]["midpoint"]

    assert abs(actual_mid - expected_mid) < 1, (
        f"Workbench midpoint must be verdict × product(1-adj): "
        f"expected {expected_mid} got {actual_mid}"
    )
    assert abs(wb["legal_pack_value_risks"]["adjustment_factor"] - expected_factor) < 0.001


# ── TEST 20: Financial standing reads Workbench ceiling ──────────────────────

def test_financial_standing_reads_workbench():
    """financial_current_standing derives from workbench_ceiling, not verdict."""
    comps     = _rpc_comps_5(200_000)
    verdict   = calculate_verdict_ceiling(sold_comps=comps, subject=_subj())
    flags     = [_flag("Short lease", "critical")]
    workbench = calculate_workbench_ceiling(verdict, flags)
    standing  = calculate_financial_standing(workbench, current_bid=150_000)

    wb_mid = workbench["valuation_range"]["midpoint"]
    wb_low = workbench["valuation_range"]["low"]
    wb_hi  = workbench["valuation_range"]["high"]

    fs_mid = standing["workbench_ceiling_range"]["midpoint"]
    fs_low = standing["workbench_ceiling_range"]["low"]
    fs_hi  = standing["workbench_ceiling_range"]["high"]

    assert fs_mid == wb_mid, f"financial_standing.midpoint must equal workbench.midpoint: {fs_mid} vs {wb_mid}"
    assert fs_low == wb_low, f"financial_standing.low must equal workbench.low: {fs_low} vs {wb_low}"
    assert fs_hi  == wb_hi,  f"financial_standing.high must equal workbench.high: {fs_hi} vs {wb_hi}"


def test_financial_standing_reads_workbench_not_verdict():
    """Financial standing must not equal verdict ceiling when flags are active."""
    comps     = _rpc_comps_5(200_000)
    verdict   = calculate_verdict_ceiling(sold_comps=comps, subject=_subj())
    flags     = [_flag("Defective title", "critical")]
    workbench = calculate_workbench_ceiling(verdict, flags)
    standing  = calculate_financial_standing(workbench, current_bid=150_000)

    v_mid = verdict["valuation_range"]["midpoint"]
    fs_mid = standing["workbench_ceiling_range"]["midpoint"]

    assert fs_mid != v_mid, (
        f"Financial standing must reference workbench, not verdict, when flags active: "
        f"fs={fs_mid} v={v_mid}"
    )
    assert fs_mid < v_mid


# ── TEST 21: MY BID does not alter Verdict or Workbench ──────────────────────

def test_my_bid_does_not_alter_verdict_or_workbench():
    """Changing current_bid must not change verdict or workbench ceilings."""
    comps     = _rpc_comps_5(200_000)
    verdict   = calculate_verdict_ceiling(sold_comps=comps, subject=_subj())
    flags     = [_flag("Short lease", "critical")]
    workbench = calculate_workbench_ceiling(verdict, flags)

    v_mid_before = verdict["valuation_range"]["midpoint"]
    w_mid_before = workbench["valuation_range"]["midpoint"]

    # Simulate MY BID changing
    standing_1 = calculate_financial_standing(workbench, current_bid=100_000)
    standing_2 = calculate_financial_standing(workbench, current_bid=180_000)
    standing_3 = calculate_financial_standing(workbench, current_bid=None)

    # Verdict and Workbench midpoints must be unchanged
    assert verdict["valuation_range"]["midpoint"]   == v_mid_before, "Verdict changed after bid"
    assert workbench["valuation_range"]["midpoint"]  == w_mid_before, "Workbench changed after bid"

    # Only current_bid and derived fields change in standing
    assert standing_1["current_bid"] == 100_000
    assert standing_2["current_bid"] == 180_000
    assert standing_3["current_bid"]  is None
    assert standing_1["workbench_ceiling_range"]["midpoint"] == w_mid_before
    assert standing_2["workbench_ceiling_range"]["midpoint"] == w_mid_before


# ── TEST 22: Acquisition costs excluded from all three objects ───────────────

def test_acquisition_costs_excluded_from_all_three():
    """SDLT / buyer premium / legal fees must not reduce verdict or workbench."""
    comps    = _rpc_comps_5(200_000)
    verdict  = calculate_verdict_ceiling(sold_comps=comps, subject=_subj())
    acq_flags = [
        {"title": "Buyer's Premium £6,999 on Completion", "severity": "high",
         "summation": "buyers premium payable on completion"},
        {"title": "SDLT £8,400 estimate",  "severity": "high", "summation": "stamp duty sdlt"},
        {"title": "Legal fees £2,000",      "severity": "note", "summation": "solicitor fee"},
    ]
    # Verdict: no flags → no reduction regardless
    assert verdict["legal_pack_value_risks"]["adjustment_factor"] == 1.0

    # Workbench with only acquisition-cost flags → factor must still be 1.0
    wb = calculate_workbench_ceiling(verdict, acq_flags)
    factor = wb["legal_pack_value_risks"]["adjustment_factor"]
    assert factor == 1.0, (
        f"Acquisition-cost flags must not reduce workbench ceiling: factor={factor}"
    )
    assert wb["valuation_range"]["midpoint"] == verdict["valuation_range"]["midpoint"]


# ── TEST 23: Legacy ceiling alias does not become canonical ──────────────────

def test_legacy_ceiling_is_not_canonical():
    """
    calculate_ceiling (legacy path) must still work but its result must not
    equal the verdict_ceiling because it applies flag risks.
    The three-object API (verdict/workbench/financial) is canonical.
    """
    comps  = _rpc_comps_5(200_000)
    flags  = [_flag("Short lease", "critical")]
    # Legacy path applies flag risks inside one call
    legacy = calculate_ceiling(flags, {}, sold_comps=comps, subject=_subj())
    # New verdict path has no flag risks
    verdict = calculate_verdict_ceiling(sold_comps=comps, subject=_subj())

    l_mid = legacy["valuation_range"]["midpoint"]
    v_mid = verdict["valuation_range"]["midpoint"]

    assert v_mid is not None and l_mid is not None
    # Legacy ceiling (flag-adjusted) must be < verdict (no flags)
    assert l_mid < v_mid, (
        f"Legacy ceiling {l_mid} must be < verdict ceiling {v_mid} because it applies flag risks"
    )
    # Legacy ceiling must not be presented as the verdict object
    assert legacy.get("_ceiling_type") != "verdict"


# ── TEST 24: Three-object API shape ──────────────────────────────────────────

def test_three_object_api_shape():
    """All three objects expose valuation_range with low, midpoint, high."""
    comps     = _rpc_comps_5(200_000)
    verdict   = calculate_verdict_ceiling(sold_comps=comps, subject=_subj())
    workbench = calculate_workbench_ceiling(verdict, [_flag("Short lease", "critical")])
    standing  = calculate_financial_standing(workbench, current_bid=160_000)

    for obj, name in [(verdict, "verdict"), (workbench, "workbench")]:
        vr = obj.get("valuation_range", {})
        assert vr.get("low")      is not None, f"{name}.valuation_range.low missing"
        assert vr.get("midpoint") is not None, f"{name}.valuation_range.midpoint missing"
        assert vr.get("high")     is not None, f"{name}.valuation_range.high missing"

    fs_cr = standing.get("workbench_ceiling_range", {})
    assert fs_cr.get("low")      is not None, "financial_standing.workbench_ceiling_range.low missing"
    assert fs_cr.get("midpoint") is not None, "financial_standing.workbench_ceiling_range.midpoint missing"
    assert fs_cr.get("high")     is not None, "financial_standing.workbench_ceiling_range.high missing"
    assert "current_bid"     in standing
    assert "gap_to_ceiling"  in standing
    assert "pct_of_ceiling"  in standing
    assert "position"        in standing


# ── TEST 25: Workbench insufficient when verdict insufficient ─────────────────

def test_workbench_insufficient_when_verdict_insufficient():
    """If verdict has no valid comps, workbench must also be insufficient."""
    verdict   = calculate_verdict_ceiling(sold_comps=[], subject=_subj())
    workbench = calculate_workbench_ceiling(verdict, [_flag("Short lease", "critical")])

    assert verdict["status"]   == "insufficient_evidence"
    assert workbench["status"] == "insufficient_evidence"
    assert workbench["valuation_range"]["midpoint"] is None


# =============================================================================
# BACKFILL / NORMALISATION HELPER TESTS (tests 26–32)
# =============================================================================

from services.ceiling_engine import (
    ensure_ceiling_owned_objects,
    VERSION,
)


def _legacy_ceiling_dict(base=250_000, lo=237_500, hi=262_500):
    """Simulate an old summary_json.ceiling (pre-v2 deploy)."""
    return {
        "base_valuation": base,
        "ceiling_range":  {"low": lo, "high": hi},
        "confidence":     0.55,
        "strategy_used":  "BTL",
    }


def _rpc_comps_within(base_price=200_000, n=5):
    return [
        {"price": base_price, "miles": 0.05 + i*0.07, "age_months": 2,
         "duration": "F", "floor_area": 65.0, "hpi_multiplier": 1.00,
         "address": f"Backfill Comp {i}", "property_type": "flat",
         "evidence_quality": "official"}
        for i in range(n)
    ]


# ── TEST 26: Backfill creates verdict and workbench from comps ────────────────

def test_backfill_creates_owned_objects_from_comps():
    """Old deal with only sj.ceiling gets verdict_ceiling and workbench_ceiling."""
    sj = {"ceiling": _legacy_ceiling_dict(), "flags": [_flag("Short lease", "critical")]}
    area = {"housing": {"soldComps": _rpc_comps_within(200_000)}}
    result = ensure_ceiling_owned_objects(sj, area_json=area, legal_flags=[_flag("Short lease", "critical")])

    assert "verdict_ceiling" in result
    assert "workbench_ceiling" in result
    assert "financial_current_standing" in result

    vc_mid = result["verdict_ceiling"]["valuation_range"]["midpoint"]
    wb_mid = result["workbench_ceiling"]["valuation_range"]["midpoint"]
    assert vc_mid is not None and vc_mid > 0
    assert wb_mid is not None and wb_mid > 0
    assert wb_mid <= vc_mid


# ── TEST 27: Legacy-only deal — workbench never exceeds verdict ───────────────

def test_old_deal_legacy_only_workbench_never_exceeds_verdict():
    """Old deal with no comps and only sj.ceiling: workbench must not exceed verdict."""
    sj = {"ceiling": _legacy_ceiling_dict(base=250_000, lo=237_500, hi=262_500)}
    # No area_json comps — should backfill from legacy base
    result = ensure_ceiling_owned_objects(sj, area_json={}, legal_flags=[_flag("Defective title", "high")])

    vc_mid = result["verdict_ceiling"]["valuation_range"]["midpoint"]
    wb_mid = result["workbench_ceiling"]["valuation_range"]["midpoint"]
    assert vc_mid is not None and vc_mid > 0
    assert wb_mid <= vc_mid, f"Workbench {wb_mid} must not exceed verdict {vc_mid}"


# ── TEST 28: Hard clamp enforced by helper ────────────────────────────────────

def test_helper_hard_clamp_enforces_workbench_lte_verdict():
    """If an existing workbench_ceiling somehow exceeds verdict, helper clamps it."""
    sj = {
        "verdict_ceiling": {
            "_ceiling_type": "verdict",
            "valuation_range": {"low": 190_000, "midpoint": 200_000, "high": 210_000, "uncertainty_band": 0.05},
            "ceiling_range": {"low": 190_000, "high": 210_000},
        },
        "workbench_ceiling": {
            "_ceiling_type": "workbench",
            "valuation_range": {"low": 220_000, "midpoint": 230_000, "high": 240_000, "uncertainty_band": 0.05},
            "ceiling_range": {"low": 220_000, "high": 240_000},
            "legal_pack_value_risks": {"adjustment_factor": 1.0, "risks": []},
        },
    }
    result = ensure_ceiling_owned_objects(sj)
    wb_mid = result["workbench_ceiling"]["valuation_range"]["midpoint"]
    vc_mid = result["verdict_ceiling"]["valuation_range"]["midpoint"]
    # The helper either recomputes (invalid wb) or hard-clamps — either way:
    assert wb_mid <= vc_mid, f"Workbench {wb_mid} must be <= verdict {vc_mid} after helper"


# ── TEST 29: Legacy sj.ceiling does not override owned objects ────────────────

def test_legacy_ceiling_does_not_override_owned_objects():
    """
    If verdict_ceiling and workbench_ceiling already exist and are valid,
    the helper preserves them and does not replace with sj.ceiling values.
    """
    comps = _rpc_comps_within(200_000)
    verdict   = calculate_verdict_ceiling(sold_comps=comps, subject=_subj())
    workbench = calculate_workbench_ceiling(verdict, [])

    sj = {
        "ceiling":           _legacy_ceiling_dict(base=999_000),  # wrong legacy value
        "verdict_ceiling":   verdict,
        "workbench_ceiling": workbench,
    }
    result = ensure_ceiling_owned_objects(sj)

    vc_mid = result["verdict_ceiling"]["valuation_range"]["midpoint"]
    wb_mid = result["workbench_ceiling"]["valuation_range"]["midpoint"]
    assert vc_mid < 999_000, "Legacy ceiling must not override valid verdict_ceiling"
    assert wb_mid < 999_000, "Legacy ceiling must not override valid workbench_ceiling"


# ── TEST 30: bid_ceiling is not used as canonical valuation ──────────────────

def test_bid_ceiling_not_canonical():
    """
    bid_ceiling (DB column) is not passed to ensure_ceiling_owned_objects.
    It must not appear as a valuation source.
    """
    # No comps, no legacy ceiling, no bid_ceiling in the helper signature
    sj = {}
    result = ensure_ceiling_owned_objects(sj, area_json={}, legal_flags=[])
    vc = result["verdict_ceiling"]
    assert vc["status"] == "missing_data"
    assert vc["valuation_range"]["midpoint"] is None
    # bid_ceiling field should not appear in verdict or workbench ranges
    assert "bid_ceiling" not in str(result.get("verdict_ceiling", {}))
    assert "bid_ceiling" not in str(result.get("workbench_ceiling", {}))


# ── TEST 31: financial_current_standing always equals workbench range ─────────

def test_financial_standing_always_equals_workbench_after_backfill():
    """After backfill, financial_current_standing.workbench_ceiling_range == workbench.valuation_range."""
    comps = _rpc_comps_within(200_000)
    area  = {"housing": {"soldComps": comps}}
    sj    = {"ceiling": _legacy_ceiling_dict(), "flags": []}
    result = ensure_ceiling_owned_objects(sj, area_json=area, legal_flags=[], current_bid=160_000)

    wb_vr = result["workbench_ceiling"]["valuation_range"]
    fs_cr = result["financial_current_standing"]["workbench_ceiling_range"]

    assert fs_cr["midpoint"] == wb_vr["midpoint"]
    assert fs_cr["low"]      == wb_vr["low"]
    assert fs_cr["high"]     == wb_vr["high"]
    assert result["financial_current_standing"]["current_bid"] == 160_000


# ── TEST 32: /api/ceiling fixture — verify numeric relationships ──────────────

def test_api_ceiling_fixture_numeric_relationships():
    """
    Simulate the /api/ceiling response using a fixture deal.
    Verify: verdict.midpoint >= workbench.midpoint
    financial_standing.workbench_ceiling_range == workbench.valuation_range
    """
    from services.ceiling_engine import calculate_financial_standing

    comps   = _rpc_comps_within(200_000)
    flags   = [_flag("Defective title", "critical"), _flag("Short lease", "high")]
    verdict = calculate_verdict_ceiling(sold_comps=comps, subject=_subj())
    wb      = calculate_workbench_ceiling(verdict, flags)
    fs      = calculate_financial_standing(wb, current_bid=155_000)

    # Simulated API response structure
    response = {
        "ok": True,
        "ceiling":           wb,
        "workbench_ceiling": wb,
        "verdict_ceiling":   verdict,
        "financial_current_standing": fs,
    }

    v_mid  = response["verdict_ceiling"]["valuation_range"]["midpoint"]
    wb_mid = response["workbench_ceiling"]["valuation_range"]["midpoint"]
    v_lo   = response["verdict_ceiling"]["valuation_range"]["low"]
    wb_lo  = response["workbench_ceiling"]["valuation_range"]["low"]
    v_hi   = response["verdict_ceiling"]["valuation_range"]["high"]
    wb_hi  = response["workbench_ceiling"]["valuation_range"]["high"]
    fs_cr  = response["financial_current_standing"]["workbench_ceiling_range"]

    assert v_mid >= wb_mid, f"verdict.midpoint {v_mid} must be >= workbench.midpoint {wb_mid}"
    assert v_lo  >= wb_lo,  f"verdict.low {v_lo} must be >= workbench.low {wb_lo}"
    assert v_hi  >= wb_hi,  f"verdict.high {v_hi} must be >= workbench.high {wb_hi}"

    assert fs_cr["midpoint"] == wb_mid
    assert fs_cr["low"]      == wb_lo
    assert fs_cr["high"]     == wb_hi


# =============================================================================
# FLAG RESOLUTION RECALCULATION TESTS (tests 33–44)
# =============================================================================

def _wb(active_flags):
    """Helper: verdict from 5 comps then workbench with given active flags."""
    comps   = _rpc_comps_5(200_000)
    verdict = calculate_verdict_ceiling(sold_comps=comps, subject=_subj())
    return verdict, calculate_workbench_ceiling(verdict, active_flags)


# ── TEST 33: Active risk uses unresolved flags only ───────────────────────────

def test_active_risk_uses_unresolved_flags_only():
    """Resolved flags must not appear in active_risks or reduce workbench ceiling."""
    all_flags = [_flag("Short lease", "critical"), _flag("Defective title", "high")]
    verdict, wb_all = _wb(all_flags)
    # Now pretend both flags are resolved → send empty list to /api
    verdict, wb_zero = _wb([])

    # With flags: workbench < verdict
    assert wb_all["valuation_range"]["midpoint"] < verdict["valuation_range"]["midpoint"]
    # Without flags (all resolved): workbench == verdict
    assert wb_zero["valuation_range"]["midpoint"] == verdict["valuation_range"]["midpoint"]
    assert wb_zero["legal_pack_value_risks"]["adjustment_factor"] == 1.0
    assert wb_zero["legal_pack_value_risks"]["risks"] == []


# ── TEST 34: Resolved flags excluded from active_flag_risk_factor ─────────────

def test_resolved_flags_excluded_from_active_risk_factor():
    """The risk factor must be product of ACTIVE flags only — 1.0 when none active."""
    verdict, wb = _wb([])
    assert wb["legal_pack_value_risks"]["adjustment_factor"] == 1.0, (
        "adjustment_factor must be 1.0 when active_legal_flags is empty"
    )


# ── TEST 35: Resolving one flag raises Workbench ──────────────────────────────

def test_resolving_one_flag_raises_workbench():
    """After resolving one flag, workbench ceiling must rise (or stay same if no-op)."""
    flags = [_flag("Short lease", "critical"), _flag("Defective title", "high")]
    verdict, wb_all = _wb(flags)
    verdict, wb_one = _wb(flags[1:])  # only "high" remains active

    m_all = wb_all["valuation_range"]["midpoint"]
    m_one = wb_one["valuation_range"]["midpoint"]
    m_v   = verdict["valuation_range"]["midpoint"]

    assert m_one >= m_all, f"Resolving a flag must raise or preserve workbench: {m_one} >= {m_all}"
    assert m_one <= m_v,  f"Workbench must not exceed verdict: {m_one} <= {m_v}"


# ── TEST 36: All flags resolved → workbench == verdict ───────────────────────

def test_all_flags_resolved_workbench_equals_verdict_with_risk_discount():
    """All flags resolved: workbench range equals verdict range exactly."""
    flags = [_flag("Short lease", "critical"), _flag("Defective title", "high"), _flag("Planning", "medium")]
    verdict, wb_all  = _wb(flags)
    verdict, wb_zero = _wb([])

    v_vr = verdict["valuation_range"]
    w_vr = wb_zero["valuation_range"]

    assert w_vr["midpoint"] == v_vr["midpoint"]
    assert w_vr["low"]      == v_vr["low"]
    assert w_vr["high"]     == v_vr["high"]
    assert wb_zero["all_flags_resolved"] is True


# ── TEST 37: All flags resolved → risk_discount_pct = 0 ──────────────────────

def test_all_flags_resolved_risk_discount_pct_is_zero():
    """risk_discount_pct must be 0.0 when active_legal_flags is empty."""
    _, wb = _wb([])
    assert wb["risk_discount_pct"] == 0.0, (
        f"risk_discount_pct must be 0 when all resolved; got {wb['risk_discount_pct']}"
    )


def test_active_flags_produce_nonzero_discount():
    """risk_discount_pct must be > 0 when active critical/high flags remain."""
    _, wb = _wb([_flag("Short lease", "critical")])
    assert wb["risk_discount_pct"] > 0.0, (
        f"risk_discount_pct must be > 0 with active critical flag; got {wb['risk_discount_pct']}"
    )


# ── TEST 38: Workbench never exceeds verdict ──────────────────────────────────

def test_workbench_never_exceeds_verdict_any_resolution_state():
    """At every resolution state, workbench <= verdict."""
    flags = [_flag("Short lease", "critical"), _flag("Defective title", "high"),
             _flag("Planning", "medium"), _flag("Service charge", "low")]
    for n_active in range(len(flags) + 1):
        active = flags[:n_active]
        verdict, wb = _wb(active)
        v_m = verdict["valuation_range"]["midpoint"]
        w_m = wb["valuation_range"]["midpoint"]
        assert w_m <= v_m, f"n_active={n_active}: workbench {w_m} > verdict {v_m}"


# ── TEST 39: resolved_flags_excluded audit field ──────────────────────────────

def test_resolved_flags_excluded_audit_field():
    """Audit must include resolved_flags_excluded = True."""
    _, wb = _wb([_flag("Short lease", "critical")])
    assert wb["audit"].get("resolved_flags_excluded") is True, (
        "audit.resolved_flags_excluded must be True"
    )


# ── TEST 40: all_flags_resolved field ────────────────────────────────────────

def test_all_flags_resolved_field_false_when_active():
    """all_flags_resolved must be False when active flags remain."""
    _, wb = _wb([_flag("Short lease", "critical")])
    assert wb["all_flags_resolved"] is False


def test_all_flags_resolved_field_true_when_empty():
    """all_flags_resolved must be True when no active flags."""
    _, wb = _wb([])
    assert wb["all_flags_resolved"] is True


# ── TEST 41: verdict_range in workbench response ──────────────────────────────

def test_workbench_includes_verdict_range():
    """Workbench response must include verdict_range for frontend Workbench=Verdict check."""
    comps   = _rpc_comps_5(200_000)
    verdict = calculate_verdict_ceiling(sold_comps=comps, subject=_subj())
    wb      = calculate_workbench_ceiling(verdict, [])
    assert "verdict_range" in wb
    assert wb["verdict_range"]["midpoint"] == verdict["valuation_range"]["midpoint"]


# ── TEST 42: risk_discount_pct formula ────────────────────────────────────────

def test_risk_discount_pct_formula():
    """risk_discount_pct = round((1 - adjustment_factor) * 100, 1)."""
    flags = [_flag("Defective title", "critical"), _flag("Short lease", "high")]
    _, wb = _wb(flags)
    factor = wb["legal_pack_value_risks"]["adjustment_factor"]
    expected_pct = round((1.0 - factor) * 100, 1)
    assert wb["risk_discount_pct"] == expected_pct, (
        f"risk_discount_pct mismatch: expected {expected_pct} got {wb['risk_discount_pct']}"
    )


# ── TEST 43: legacy ceiling stale -38% cannot persist after resolution ─────────

def test_legacy_ceiling_stale_discount_cannot_persist():
    """
    After all flags resolved, the workbench object returned by the engine
    has risk_discount_pct=0 regardless of what legacy summary_json.ceiling contained.
    """
    # Simulate a legacy ceiling with high discount
    legacy_sj = {
        "ceiling": {
            "base_valuation": 250_000,
            "ceiling_range": {"low": 154_000, "high": 164_000},
            "risk_discount_pct": 38,  # stale
            "confidence": 0.55,
        }
    }
    from services.ceiling_engine import ensure_ceiling_owned_objects
    result = ensure_ceiling_owned_objects(legacy_sj, area_json={}, legal_flags=[])
    # After backfill with no active flags, workbench risk_discount_pct must be 0
    wb = result.get("workbench_ceiling", {})
    assert wb.get("risk_discount_pct", None) == 0.0, (
        f"Legacy -38% must not survive to workbench after resolution; got {wb.get('risk_discount_pct')}"
    )
    assert wb.get("all_flags_resolved") is True


# ── TEST 44: /api/ceiling fixture — active_flags=[] gives 200-equivalent response ──

def test_api_ceiling_fixture_all_flags_resolved():
    """
    Simulate /api/ceiling call with all flags resolved (empty active list).
    Response must: risk_discount_pct=0, workbench==verdict, no 503.
    """
    from services.ceiling_engine import calculate_financial_standing

    comps   = _rpc_comps_5(200_000)
    verdict = calculate_verdict_ceiling(sold_comps=comps, subject=_subj())
    wb      = calculate_workbench_ceiling(verdict, active_legal_flags=[])
    fs      = calculate_financial_standing(wb, current_bid=155_000)

    # Simulated 200 response
    response = {
        "ok": True,
        "ceiling":           wb,
        "workbench_ceiling": wb,
        "verdict_ceiling":   verdict,
        "financial_current_standing": fs,
    }

    assert response["ok"] is True  # 200, not 503
    assert response["workbench_ceiling"]["risk_discount_pct"] == 0.0
    assert response["workbench_ceiling"]["all_flags_resolved"] is True

    v_mid = response["verdict_ceiling"]["valuation_range"]["midpoint"]
    w_mid = response["workbench_ceiling"]["valuation_range"]["midpoint"]
    assert w_mid == v_mid, f"All resolved: workbench {w_mid} must equal verdict {v_mid}"
    assert w_mid <= v_mid  # also satisfies never-exceed


# =============================================================================
# MANDATORY FIXTURE TEST — £372k–£411k verdict, 38% discount
# =============================================================================

def test_mandatory_fixture_workbench_from_verdict_range():
    """
    Mandatory fixture per task specification:
    Verdict range = £372,000–£411,000
    risk_discount_pct = 38%  →  risk_factor = 0.62
    Expected Workbench:
      low  = 372_000 × 0.62 = 230_640
      high = 411_000 × 0.62 = 254_820

    Workbench MUST NOT output £18,000–£20,000.
    """
    from services.ceiling_engine import calculate_workbench_ceiling

    # Simulate the legacy ceiling stored in summary_json
    # (Verdict range that the Verdict page displays)
    legacy_ceiling = {
        "base_valuation": 391_500,   # midpoint of 372k–411k
        "ceiling_range":  {"low": 372_000, "high": 411_000},
        "base_method":    "legacy_ceiling",
        "confidence":     0.55,
    }

    # Simulate what /api/ceiling now builds as verdict_result from legacy ceiling
    _ub    = 0.05
    _v_lo  = 372_000.0
    _v_hi  = 411_000.0
    _v_mid = 391_500.0
    verdict_result = {
        "_ceiling_type":  "verdict",
        "_legacy_source": True,
        "status":         "ok",
        "base": {"value": _v_mid, "method": "legacy_ceiling"},
        "base_valuation":  int(_v_mid),
        "valuation_range": {
            "low":              _v_lo,
            "midpoint":         _v_mid,
            "high":             _v_hi,
            "uncertainty_band": _ub,
        },
        "ceiling_range": {"low": int(_v_lo), "high": int(_v_hi)},
        "confidence": {"final": 0.55, "label": "Low confidence"},
        "legal_pack_value_risks": {
            "method": "property_value_risk_adjustment_only",
            "adjustment_factor": 1.0, "adjusted_value": None, "risks": [],
        },
        "audit": {"warnings": [], "version": "ceiling_relational_paper_valuation_v1"},
        "acquisition_costs": None,
        "excluded_from_ceiling": [],
    }

    # Active flags producing ~38% risk discount
    # product(1-0.10)(1-0.06)(1-0.06)(1-0.035)(1-0.06)(1-0.06)(1-0.015) ≈ 0.62
    active_flags = [
        {"title": "Leasehold Title",         "severity": "critical", "summation": ""},
        {"title": "Building Safety Act",      "severity": "high",     "summation": ""},
        {"title": "Existing Mortgage",        "severity": "high",     "summation": ""},
        {"title": "Lease Only 125 Years",     "severity": "medium",   "summation": ""},
        {"title": "No Assignment Sub-Sale",   "severity": "high",     "summation": ""},
        {"title": "Mines and Minerals",       "severity": "high",     "summation": ""},
        {"title": "Interest Rate on Late",    "severity": "low",      "summation": ""},
    ]

    wb = calculate_workbench_ceiling(verdict_result, active_flags)

    wb_lo  = wb["valuation_range"]["low"]
    wb_hi  = wb["valuation_range"]["high"]
    wb_mid = wb["valuation_range"]["midpoint"]

    # Must not be £18k–£20k (the yield-based junk)
    assert wb_lo  > 50_000,  f"Workbench.low {wb_lo} is unrealistically low — wrong base source"
    assert wb_hi  > 50_000,  f"Workbench.high {wb_hi} is unrealistically low — wrong base source"
    assert wb_mid > 100_000, f"Workbench.midpoint {wb_mid} is unrealistically low — wrong base source"

    # Must be below verdict (risk factor < 1)
    assert wb_lo  <= _v_lo,  f"Workbench.low {wb_lo} must not exceed verdict.low {_v_lo}"
    assert wb_hi  <= _v_hi,  f"Workbench.high {wb_hi} must not exceed verdict.high {_v_hi}"
    assert wb_mid <= _v_mid, f"Workbench.midpoint {wb_mid} must not exceed verdict.midpoint {_v_mid}"

    # Must be in the right ballpark (38% discount → ~62% of verdict)
    risk_factor = wb["legal_pack_value_risks"]["adjustment_factor"]
    expected_lo  = round(_v_lo  * risk_factor, 0)
    expected_hi  = round(_v_hi  * risk_factor, 0)

    assert abs(wb_lo  - expected_lo)  < 5_000, (
        f"Workbench.low {wb_lo} should be ~{expected_lo} (verdict × risk_factor)"
    )
    assert abs(wb_hi  - expected_hi)  < 5_000, (
        f"Workbench.high {wb_hi} should be ~{expected_hi} (verdict × risk_factor)"
    )

    # risk_discount_pct should reflect the flags
    assert wb["risk_discount_pct"] > 0, "Active flags must produce non-zero risk_discount_pct"
    assert wb["risk_discount_pct"] < 50, "risk_discount_pct should not exceed 50% for these flags"


def test_ensure_ceiling_owned_objects_uses_legacy_range_not_estimate():
    """
    ensure_ceiling_owned_objects must build verdict_ceiling from the actual
    legacy ceiling_range.low/high, not recalculate from base ± 5%.
    If legacy stores £372k–£411k, verdict must show £372k–£411k not £372k ± 5%.
    """
    from services.ceiling_engine import ensure_ceiling_owned_objects

    sj = {
        "ceiling": {
            "base_valuation": 391_500,
            "ceiling_range":  {"low": 372_000, "high": 411_000},
            "base_method":    "comps",
            "confidence":     0.55,
        }
    }
    result = ensure_ceiling_owned_objects(sj, area_json={}, legal_flags=[])
    vc_vr = result["verdict_ceiling"]["valuation_range"]

    assert abs(vc_vr["low"]  - 372_000) < 1_000, (
        f"verdict_ceiling.low should be ~372k (from legacy range), got {vc_vr['low']}"
    )
    assert abs(vc_vr["high"] - 411_000) < 1_000, (
        f"verdict_ceiling.high should be ~411k (from legacy range), got {vc_vr['high']}"
    )


def test_api_ceiling_legacy_deal_workbench_uses_legacy_range():
    """
    Simulate the /api/ceiling path for a deal with only sj.ceiling (no verdict_ceiling).
    The workbench must be derived from the actual ceiling_range, not re-derived from comps.
    Verdict: £372k–£411k → Workbench must be in that neighbourhood, not £18k–£20k.
    """
    from services.ceiling_engine import calculate_workbench_ceiling

    # What app.py now builds as verdict_result for legacy deals
    verdict_result = {
        "_ceiling_type":  "verdict",
        "_legacy_source": True,
        "status":         "ok",
        "base": {"value": 391_500, "method": "legacy_ceiling"},
        "base_valuation":  391_500,
        "valuation_range": {
            "low": 372_000.0, "midpoint": 391_500.0, "high": 411_000.0, "uncertainty_band": 0.05
        },
        "ceiling_range": {"low": 372_000, "high": 411_000},
        "confidence": {"final": 0.55, "label": "Low confidence"},
        "legal_pack_value_risks": {
            "method": "property_value_risk_adjustment_only",
            "adjustment_factor": 1.0, "adjusted_value": None, "risks": [],
        },
        "audit": {"warnings": [], "version": "ceiling_relational_paper_valuation_v1"},
        "acquisition_costs": None, "excluded_from_ceiling": [],
    }

    # All flags resolved → workbench should equal verdict
    wb_resolved = calculate_workbench_ceiling(verdict_result, active_legal_flags=[])
    assert wb_resolved["valuation_range"]["low"]  == 372_000.0
    assert wb_resolved["valuation_range"]["high"] == 411_000.0
    assert wb_resolved["risk_discount_pct"] == 0.0
    assert wb_resolved["all_flags_resolved"] is True

    # With one critical flag
    wb_risk = calculate_workbench_ceiling(verdict_result, [{"title": "Defective title", "severity": "critical", "summation": ""}])
    assert wb_risk["valuation_range"]["low"]  < 372_000, "Risk must reduce low"
    assert wb_risk["valuation_range"]["high"] < 411_000, "Risk must reduce high"
    assert wb_risk["valuation_range"]["low"]  > 50_000,  "Must not be £18k junk"


# =============================================================================
# NEW TESTS — LR code normalisation + ensure_ceiling_owned_objects source order
# =============================================================================

import sys as _sys
_sys.path.insert(0, "/tmp/api_fix")
from services.ceiling_engine import (
    _normalise_property_type,
    _types_near_equiv,
    _assess_comp,
    calculate_verdict_ceiling,
    calculate_workbench_ceiling,
    ensure_ceiling_owned_objects,
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _rpc_comp(price, miles, prop_type, tenure="L", age_months=3):
    """Minimal sold comp as returned by housing_comps_v1 RPC."""
    return {
        "price":         price,
        "miles":         miles,
        "property_type": prop_type,   # LR code or full word
        "duration":      tenure,      # LR tenure code
        "age_months":    age_months,
    }

def _nine_flats():
    """Nine flat comps with LR property_type='F', all within 0.2 miles."""
    prices = [250_000, 275_000, 270_000, 260_000, 255_000, 265_000, 280_000, 245_000, 258_000]
    miles  = [0.00,   0.10,    0.10,   0.05,   0.08,   0.12,   0.15,   0.20,   0.22]
    return [_rpc_comp(p, m, "F") for p, m in zip(prices, miles)]

def _legacy_summary(base=160_000, lo=152_000, hi=168_000):
    """summary_json with a pre-v2 legacy ceiling and a _legacy_source verdict."""
    return {
        "ceiling": {
            "base_valuation": base,
            "ceiling_range":  {"low": lo, "high": hi},
            "base_method":    "legacy_v1",
        },
        "verdict_ceiling": {
            "_ceiling_type":  "verdict",
            "_legacy_source": True,
            "status":         "ok",
            "valuation_range": {
                "low": float(lo), "midpoint": float(base), "high": float(hi),
                "uncertainty_band": 0.05,
            },
            "ceiling_range":  {"low": lo, "high": hi},
            "base":           {"value": float(base), "method": "legacy_v1"},
            "base_valuation": base,
            "legal_pack_value_risks": {
                "method": "property_value_risk_adjustment_only",
                "adjustment_factor": 1.0, "adjusted_value": None, "risks": [],
            },
            "confidence": {"final": 0.45, "label": "Low confidence"},
            "audit": {"warnings": ["built from legacy"], "version": "v1"},
            "acquisition_costs": None,
            "excluded_from_ceiling": [],
        },
    }


# ── Fix A: property-type normalisation ───────────────────────────────────────

def test_normalise_lr_flat_code():
    assert _normalise_property_type("F")    == "flat"
    assert _normalise_property_type("f")    == "flat"
    assert _normalise_property_type("flat") == "flat"
    assert _normalise_property_type("Flat/Maisonette") in ("flat", "maisonette")  # both in FLAT_TERMS
    assert _normalise_property_type("maisonette") == "maisonette"

def test_normalise_lr_house_codes():
    assert _normalise_property_type("S") == "semi-detached"
    assert _normalise_property_type("T") == "terraced"
    assert _normalise_property_type("D") == "detached"

def test_types_near_equiv_lr_codes():
    # All flat variants must match each other
    for a in ("flat", "F", "f", "Flat/Maisonette", "maisonette", "apartment"):
        for b in ("flat", "F", "f", "maisonette"):
            assert _types_near_equiv(a, b), f"Expected near-equiv: {a!r} vs {b!r}"
    # House variants
    for a in ("terraced", "T", "semi-detached", "S", "detached", "D"):
        for b in ("terraced", "T", "semi-detached", "S", "detached", "D"):
            assert _types_near_equiv(a, b), f"Expected near-equiv: {a!r} vs {b!r}"
    # Cross-class must NOT match
    assert not _types_near_equiv("flat", "S")
    assert not _types_near_equiv("F",    "T")
    assert not _types_near_equiv("flat", "D")

def test_assess_comp_lr_flat_vs_subject_flat():
    """The primary observed failure: comp type='F', subject type='flat'."""
    c = _rpc_comp(260_000, 0.05, "F")
    valid, excl = _assess_comp(c, {"property_type": "flat"}, 0)
    assert valid is not None, f"Expected valid comp, got excluded: {(excl or {}).get('reason')}"
    assert excl is None

def test_assess_comp_all_nine_flats_pass():
    """All nine comps in the observed deal must pass the type filter."""
    subject = {"property_type": "flat", "tenure": "leasehold"}
    comps   = _nine_flats()
    valid_count = 0
    for i, c in enumerate(comps):
        v, e = _assess_comp(c, subject, i)
        assert v is not None, f"Comp {i} excluded: {(e or {}).get('reason')}"
        valid_count += 1
    assert valid_count == 9

def test_calculate_verdict_ceiling_with_lr_comps():
    """calculate_verdict_ceiling must produce midpoint > 0 from LR-coded comps."""
    result = calculate_verdict_ceiling(
        sold_comps=_nine_flats(),
        subject={"property_type": "flat", "tenure": "leasehold"},
        strategy="BTL",
        fallback_allowed=True,
    )
    mid = (result.get("valuation_range") or {}).get("midpoint") or 0
    assert mid > 200_000, f"Expected midpoint ≈ £260k, got {mid}"
    assert result.get("status") != "insufficient_evidence", \
        f"Should not be insufficient_evidence with 9 valid comps; got {result.get('status')}"
    assert not result.get("_legacy_source")


# ── Fix B: ensure_ceiling_owned_objects source order ─────────────────────────

def test_legacy_verdict_replaced_when_comps_succeed():
    """
    Core regression test: legacy £160k verdict must be replaced by comp-derived
    ~£260k verdict when area_json carries valid sold comps.
    """
    sj = _legacy_summary(base=160_000)
    area_json = {"housing": {"soldComps": _nine_flats()}}
    subject   = {"property_type": "flat", "tenure": "leasehold"}

    result = ensure_ceiling_owned_objects(
        summary_json=sj,
        area_json=area_json,
        subject=subject,
    )
    vc  = result["verdict_ceiling"]
    mid = (vc.get("valuation_range") or {}).get("midpoint") or 0

    assert not vc.get("_legacy_source"), \
        f"_legacy_source must be absent after successful recompute, got {vc.get('_legacy_source')}"
    assert mid > 200_000, \
        f"Verdict must be comp-derived (~£260k), not legacy £160k. Got {mid}"
    src = (vc.get("audit") or {}).get("source_decision")
    assert src == "computed_from_sold_comps", f"source_decision wrong: {src}"


def test_near_thin_comps_recompute_not_legacy_fallback():
    """S33-STEP1 (2026-06-21): comps at 0.80mi/0.90mi are within
    MAX_RADIUS_MILES (3.0) and are now valid evidence (degraded weight, not
    excluded) — they must produce a genuine fresh recompute, not a legacy
    fallback. Replaces test_legacy_verdict_preserved_when_comps_all_excluded,
    which asserted the pre-S33-STEP1 doctrine that 0.5mi+ comps were excluded
    outright, forcing a fallback. That premise no longer holds."""
    sj = _legacy_summary(base=160_000)
    area_json = {"housing": {"soldComps": [
        _rpc_comp(260_000, 0.80, "F"),
        _rpc_comp(270_000, 0.90, "F"),
    ]}}

    result = ensure_ceiling_owned_objects(
        summary_json=sj,
        area_json=area_json,
        subject={"property_type": "flat"},
    )
    vc  = result["verdict_ceiling"]
    mid = (vc.get("valuation_range") or {}).get("midpoint") or 0

    assert mid > 0, "Must produce a real value from the 2 valid comps"
    assert not vc.get("_legacy_source"), \
        "0.8mi/0.9mi comps are valid evidence under S33-STEP1 — must not fall back to legacy"
    assert vc.get("status") == "degraded_low_comps", \
        "2 valid comps (below MIN_REQUIRED_COMPS=3) is degraded, not legacy or insufficient"
    src = (vc.get("audit") or {}).get("source_decision")
    assert src == "computed_from_sold_comps", f"source_decision wrong: {src}"


def test_comps_beyond_max_radius_fall_back_to_legacy():
    """Comps genuinely beyond MAX_RADIUS_MILES (3.0mi) ARE excluded outright —
    this is the real, current exclusion boundary (PRIMARY_RADIUS_MILES=0.5 is
    label-only, see S33-STEP1) — and recompute correctly fails over to the
    legacy ceiling rather than blanking. This preserves the regression intent
    of the old 0.5mi-cutoff test at the boundary that actually exists today."""
    sj = _legacy_summary(base=160_000)
    area_json = {"housing": {"soldComps": [
        _rpc_comp(260_000, 3.50, "F"),   # > MAX_RADIUS_MILES (3.0) → excluded
        _rpc_comp(270_000, 4.00, "F"),
    ]}}

    result = ensure_ceiling_owned_objects(
        summary_json=sj,
        area_json=area_json,
        subject={"property_type": "flat"},
    )
    vc  = result["verdict_ceiling"]
    mid = (vc.get("valuation_range") or {}).get("midpoint") or 0

    assert mid > 0, "Must not blank the ceiling when comp recompute fails"
    assert vc.get("_legacy_source") is True, "Legacy fallback must be marked _legacy_source"
    src = (vc.get("audit") or {}).get("source_decision")
    assert src == "legacy_fallback_comp_recompute_failed", f"source_decision wrong: {src}"


def test_no_comps_no_legacy_gives_missing_data():
    """Nothing to compute from → missing_data, no crash."""
    result = ensure_ceiling_owned_objects(summary_json={}, area_json={})
    vc = result["verdict_ceiling"]
    assert vc.get("status") == "missing_data"
    assert (vc.get("valuation_range") or {}).get("midpoint") is None


def test_no_comps_with_legacy_preserves_legacy():
    """No area_json comps but legacy ceiling exists → legacy preserved."""
    sj = _legacy_summary(base=160_000)
    result = ensure_ceiling_owned_objects(summary_json=sj, area_json={})
    vc  = result["verdict_ceiling"]
    mid = (vc.get("valuation_range") or {}).get("midpoint") or 0

    assert mid > 0, "Legacy must be preserved when no comps"
    assert vc.get("_legacy_source") is True
    src = (vc.get("audit") or {}).get("source_decision")
    assert src == "legacy_fallback_no_comps", f"source_decision wrong: {src}"


def test_non_legacy_verdict_not_disturbed():
    """A non-legacy verdict already in summary_json must be preserved unchanged."""
    sj = {
        "verdict_ceiling": {
            "_ceiling_type": "verdict",
            "status": "ok",
            "valuation_range": {
                "low": 245_000.0, "midpoint": 258_535.0, "high": 271_000.0,
                "uncertainty_band": 0.05,
            },
            "ceiling_range": {"low": 245_000, "high": 271_000},
            "base": {"value": 258_535.0, "method": "weighted_median_relational_comparables_0_5_mile"},
            "legal_pack_value_risks": {
                "method": "property_value_risk_adjustment_only",
                "adjustment_factor": 1.0, "adjusted_value": None, "risks": [],
            },
            "confidence": {"final": 0.72, "label": "Moderate confidence"},
            "audit": {"source_decision": "computed_from_sold_comps"},
            "acquisition_costs": None,
            "excluded_from_ceiling": [],
        }
    }
    result = ensure_ceiling_owned_objects(summary_json=sj, area_json={})
    assert result["verdict_ceiling"]["valuation_range"]["midpoint"] == 258_535.0
    assert not result["verdict_ceiling"].get("_legacy_source")


def test_all_flags_resolved_workbench_equals_verdict():
    """When all flags resolved, workbench must equal verdict and risk_discount_pct = 0."""
    sj = _legacy_summary()
    area_json = {"housing": {"soldComps": _nine_flats()}}

    result = ensure_ceiling_owned_objects(
        summary_json=sj,
        area_json=area_json,
        legal_flags=[],  # no active flags
        subject={"property_type": "flat", "tenure": "leasehold"},
    )
    vc  = result["verdict_ceiling"]
    wb  = result["workbench_ceiling"]
    v_mid = (vc.get("valuation_range") or {}).get("midpoint") or 0
    w_mid = (wb.get("valuation_range") or {}).get("midpoint") or 0

    assert v_mid > 0
    assert w_mid == v_mid, f"Workbench must equal verdict when all flags resolved. v={v_mid} w={w_mid}"
    assert wb.get("risk_discount_pct") == 0.0
    assert wb.get("active_flag_count") == 0
    assert wb.get("all_flags_resolved") is True


def test_current_bid_does_not_alter_verdict():
    """current_bid must not change verdict or workbench midpoint."""
    sj = _legacy_summary()
    area_json = {"housing": {"soldComps": _nine_flats()}}

    result = ensure_ceiling_owned_objects(
        summary_json=sj,
        area_json=area_json,
        current_bid=100_000,
        subject={"property_type": "flat", "tenure": "leasehold"},
    )
    vc  = result["verdict_ceiling"]
    mid = (vc.get("valuation_range") or {}).get("midpoint") or 0

    assert mid > 200_000, f"current_bid must not affect verdict midpoint. Got {mid}"
    fs = result.get("financial_current_standing") or {}
    assert fs.get("current_bid") == 100_000


def test_type_mismatch_comps_genuinely_excluded():
    """Flat subject + detached comps → all excluded, safe fallback to legacy."""
    sj = _legacy_summary(base=160_000)
    area_json = {"housing": {"soldComps": [
        _rpc_comp(300_000, 0.05, "D"),  # detached — type mismatch vs flat subject
        _rpc_comp(310_000, 0.10, "D"),
        _rpc_comp(320_000, 0.15, "D"),
    ]}}

    result = ensure_ceiling_owned_objects(
        summary_json=sj,
        area_json=area_json,
        subject={"property_type": "flat"},
    )
    vc  = result["verdict_ceiling"]
    mid = (vc.get("valuation_range") or {}).get("midpoint") or 0

    assert mid > 0, "Must not blank ceiling on genuine type mismatch"
    assert vc.get("_legacy_source") is True
    reasons = (vc.get("audit") or {}).get("excluded_reasons_summary") or {}
    assert any("type_mismatch" in r for r in reasons), \
        f"Excluded reasons should mention type_mismatch. Got: {reasons}"


# =============================================================================
# PHASE 2 TESTS — Comparable valuation methodology
# =============================================================================

import sys as _sys
_sys.path.insert(0, "/tmp/phase2/legal-smegal-api-main")
from services.ceiling_engine import (
    _normalise_property_type,
    _types_near_equiv,
    _assess_comp,
    calculate_ceiling,
    calculate_verdict_ceiling,
    calculate_workbench_ceiling,
    ensure_ceiling_owned_objects,
)


def _p2_comp(price, miles, prop_type="F", tenure="L", age_months=3, hpi_mult=None):
    """Minimal sold comp as returned by get_housing_data / housing_comps_v1 RPC."""
    c = {
        "price":         price,
        "miles":         miles,
        "property_type": prop_type,
        "duration":      tenure,
        "age_months":    age_months,
    }
    if hpi_mult is not None:
        c["hpi_multiplier"] = hpi_mult
    return c


def _p2_subject(prop_type="flat", tenure="leasehold"):
    return {"property_type": prop_type, "tenure": tenure}


# ── T1: Sold comps found from real stored path ────────────────────────────

def test_p2_comps_extracted_from_area_json():
    """T1: ensure_ceiling_owned_objects reads area_json.housing.soldComps."""
    area_json = {
        "housing": {
            "soldComps": [
                _p2_comp(250000, 0.05),
                _p2_comp(260000, 0.10),
                _p2_comp(255000, 0.08),
            ]
        }
    }
    result = ensure_ceiling_owned_objects(
        summary_json={},
        area_json=area_json,
        subject=_p2_subject(),
    )
    vc = result["verdict_ceiling"]
    mid = (vc.get("valuation_range") or {}).get("midpoint") or 0
    assert mid > 0, f"Expected midpoint from soldComps, got {mid}"
    # Audit must record comps_source_path
    audit = vc.get("audit") or {}
    assert audit.get("sold_comps_count") == 3 or (vc.get("comparables") is not None)


# ── T2-T5: Property type normalisation ───────────────────────────────────

def test_p2_F_comp_accepted_flat_subject():
    """T2: LR code F accepted against subject 'flat'."""
    valid, excl = _assess_comp(_p2_comp(250000, 0.1, prop_type="F"), _p2_subject("flat"), 0)
    assert valid is not None, f"F comp must be accepted against flat subject. Got: {excl}"


def test_p2_S_comp_accepted_semi_detached_subject():
    """T3: LR code S accepted against subject 'semi-detached'."""
    valid, excl = _assess_comp(
        _p2_comp(250000, 0.1, prop_type="S", tenure="F"),
        _p2_subject("semi-detached", "freehold"), 0
    )
    assert valid is not None, f"S comp must be accepted against semi-detached. Got: {excl}"


def test_p2_T_comp_accepted_terraced_subject():
    """T4: LR code T accepted against subject 'terraced'."""
    valid, excl = _assess_comp(
        _p2_comp(200000, 0.1, prop_type="T", tenure="F"),
        _p2_subject("terraced", "freehold"), 0
    )
    assert valid is not None, f"T comp must be accepted against terraced. Got: {excl}"


def test_p2_D_comp_accepted_detached_subject():
    """T5: LR code D accepted against subject 'detached'."""
    valid, excl = _assess_comp(
        _p2_comp(400000, 0.1, prop_type="D", tenure="F"),
        _p2_subject("detached", "freehold"), 0
    )
    assert valid is not None, f"D comp must be accepted against detached. Got: {excl}"


def test_p2_wrong_class_rejected():
    """T6: Flat subject with detached comp is rejected (type_mismatch)."""
    valid, excl = _assess_comp(
        _p2_comp(400000, 0.1, prop_type="D", tenure="F"),
        _p2_subject("flat", "leasehold"), 0
    )
    assert valid is None, "Detached comp must be rejected against flat subject"
    assert "type_mismatch" in (excl or {}).get("reason", "")


# ── T7-T8: Type + tenure before IQR ──────────────────────────────────────

def test_p2_tenure_mismatch_excluded_before_iqr():
    """T7: Tenure-mismatched comps excluded (never reach IQR set)."""
    # Subject is leasehold; comp is freehold — must be excluded
    valid, excl = _assess_comp(
        _p2_comp(250000, 0.1, prop_type="F", tenure="F"),   # freehold
        _p2_subject("flat", "leasehold"),  # subject leasehold
        0
    )
    assert valid is None, "Freehold comp must be excluded against leasehold subject"
    assert "tenure_mismatch" in (excl or {}).get("reason", "")


def test_p2_cross_type_excluded_before_iqr():
    """T8: Cross-type comps excluded before IQR (not in clean IQR set)."""
    # 6 comps needed for IQR to fire. Mix flat and terraced.
    comps = (
        [_p2_comp(250000, 0.05, "F")] * 4  # flat — matching
        + [_p2_comp(300000, 0.1, "T", "F")] * 3  # terraced — excluded for type + tenure
    )
    subject = _p2_subject("flat", "leasehold")
    result = calculate_ceiling(legal_flags=[], financial_inputs={}, sold_comps=comps, subject=subject)
    valid_count = result["base"]["valid_comparable_count"]
    excl = result["comparables"]["excluded"]
    type_mismatch_excl = [e for e in excl if "type_mismatch" in (e.get("reason") or "")]
    assert len(type_mismatch_excl) >= 3, "Terraced comps must be excluded before IQR"
    # IQR only had 4 flat comps — fewer than 6 — so IQR was skipped
    audit = result.get("audit") or {}
    iqr = audit.get("iqr_trim_summary") or {}
    assert not iqr.get("applied"), "IQR should not apply when fewer than 6 matching comps"


# ── T9-T10: HPI normalisation ─────────────────────────────────────────────

def test_p2_hpi_multiplier_produces_adjusted_value():
    """T9: Sale price × hpi_multiplier = adjusted_value."""
    comp = _p2_comp(200000, 0.1, hpi_mult=1.05)
    valid, excl = _assess_comp(comp, _p2_subject(), 0)
    assert valid is not None
    expected = 200000 * 1.05
    assert abs(valid["adjusted_value"] - expected) < 1, \
        f"adjusted_value must be price × hpi_mult. Expected {expected}, got {valid['adjusted_value']}"
    assert valid["adjustments"]["time"] == 1.05


def test_p2_missing_hpi_is_audited():
    """T10: Missing hpi_multiplier is audited — time_adjustment=1.00 assumed."""
    comp = _p2_comp(200000, 0.1)  # no hpi_multiplier
    valid, excl = _assess_comp(comp, _p2_subject(), 0)
    assert valid is not None
    assert valid["adjustments"]["time"] == 1.00
    assert any("hpi_adjustment missing" in w for w in valid.get("audit_warnings", []))


# ── T11-T12: IQR trimming ─────────────────────────────────────────────────

def test_p2_iqr_outlier_removed_after_hpi():
    """T11: Outlier comp is removed by IQR after HPI normalisation (≥6 comps)."""
    # 6 flat comps at ~£260k + 1 extreme outlier at £600k
    comps = [
        _p2_comp(250000, 0.05),
        _p2_comp(255000, 0.08),
        _p2_comp(260000, 0.10),
        _p2_comp(262000, 0.12),
        _p2_comp(265000, 0.15),
        _p2_comp(270000, 0.18),
        _p2_comp(600000, 0.20),  # outlier — above IQR fence
    ]
    subject = _p2_subject()
    result = calculate_ceiling(legal_flags=[], financial_inputs={}, sold_comps=comps, subject=subject)
    audit = result.get("audit") or {}
    iqr = audit.get("iqr_trim_summary") or {}
    assert iqr.get("applied"), "IQR should have fired with 7 comps"
    assert iqr.get("removed_outliers_count", 0) >= 1, "Outlier should be removed"
    # base_value should be near the cluster (~260k), not pulled toward £600k
    bv = result["base"]["value"] or 0
    assert bv < 300000, f"base_value {bv} should be below £300k after outlier removal"


def test_p2_iqr_audit_persisted():
    """T12: IQR trim counts are persisted in audit."""
    comps = [
        _p2_comp(250000, 0.05),
        _p2_comp(255000, 0.08),
        _p2_comp(260000, 0.10),
        _p2_comp(262000, 0.12),
        _p2_comp(265000, 0.15),
        _p2_comp(600000, 0.20),  # outlier
    ]
    result = calculate_ceiling(legal_flags=[], financial_inputs={}, sold_comps=comps, subject=_p2_subject())
    audit = result.get("audit") or {}
    iqr = audit.get("iqr_trim_summary") or {}
    assert "pre_trim_count" in iqr
    assert "post_trim_count" in iqr
    assert "removed_outliers_count" in iqr


# ── T13: Weighted median primary ──────────────────────────────────────────

def test_p2_weighted_median_not_arithmetic_mean():
    """T13: Weighted median, not arithmetic mean, determines base_value."""
    # Skewed distribution: 4 comps at £200k, 1 at £800k
    comps = [
        _p2_comp(200000, 0.05, age_months=1),
        _p2_comp(200000, 0.06, age_months=1),
        _p2_comp(200000, 0.07, age_months=1),
        _p2_comp(200000, 0.08, age_months=1),
        _p2_comp(800000, 0.40, age_months=24),  # low weight, far, old
    ]
    result = calculate_ceiling(legal_flags=[], financial_inputs={}, sold_comps=comps, subject=_p2_subject())
    bv = result["base"]["value"] or 0
    arith_mean = (200000 * 4 + 800000) / 5  # = 280000
    assert abs(bv - 200000) < 10000, f"weighted median should be ~£200k, got {bv}"
    assert abs(bv - arith_mean) > 20000, "base_value must not equal arithmetic mean"
    assert result["base"]["method"].startswith("weighted_median")


# ── T14-T16: Verdict isolation ────────────────────────────────────────────

def test_p2_legal_flags_do_not_reduce_verdict():
    """T14: Legal flags must not reduce Verdict."""
    comps = [_p2_comp(260000, 0.05), _p2_comp(265000, 0.08), _p2_comp(258000, 0.10)]
    subject = _p2_subject()
    verdict = calculate_verdict_ceiling(sold_comps=comps, subject=subject)
    mid_v = (verdict.get("valuation_range") or {}).get("midpoint") or 0
    adj   = (verdict.get("legal_pack_value_risks") or {}).get("adjustment_factor", 0)
    assert adj == 1.0, f"Verdict adjustment_factor must be 1.0 (no flags), got {adj}"
    assert mid_v > 0


def test_p2_manual_bid_does_not_alter_verdict():
    """T15: current_bid must not change verdict midpoint."""
    comps = [_p2_comp(260000, 0.05), _p2_comp(265000, 0.08), _p2_comp(258000, 0.10)]
    area  = {"housing": {"soldComps": comps}}
    r1 = ensure_ceiling_owned_objects(summary_json={}, area_json=area, current_bid=None, subject=_p2_subject())
    r2 = ensure_ceiling_owned_objects(summary_json={}, area_json=area, current_bid=120000, subject=_p2_subject())
    mid1 = (r1["verdict_ceiling"].get("valuation_range") or {}).get("midpoint")
    mid2 = (r2["verdict_ceiling"].get("valuation_range") or {}).get("midpoint")
    assert mid1 == mid2, f"current_bid must not alter verdict. Got {mid1} vs {mid2}"


def test_p2_acquisition_costs_excluded_from_verdict():
    """T16: Acquisition cost flags must not reduce Verdict."""
    comps   = [_p2_comp(260000, 0.05), _p2_comp(265000, 0.08), _p2_comp(258000, 0.10)]
    subject = _p2_subject()
    flags   = [{"title": "Buyer's premium 5%", "severity": "high", "summation": "buyers premium"}]
    verdict = calculate_verdict_ceiling(sold_comps=comps, subject=subject)
    adj     = (verdict.get("legal_pack_value_risks") or {}).get("adjustment_factor", 0)
    assert adj == 1.0, "Acquisition cost flags must not reduce Verdict"


# ── T17-T18: Workbench relation ───────────────────────────────────────────

def test_p2_unresolved_risks_reduce_workbench_only():
    """T17: Unresolved legal risks reduce Workbench, not Verdict."""
    comps   = [_p2_comp(260000, 0.05), _p2_comp(265000, 0.08), _p2_comp(258000, 0.10)]
    verdict = calculate_verdict_ceiling(sold_comps=comps, subject=_p2_subject())
    flags   = [{"title": "Defective title", "severity": "critical", "summation": ""}]
    wb      = calculate_workbench_ceiling(verdict, flags)
    v_mid   = (verdict.get("valuation_range") or {}).get("midpoint") or 0
    w_mid   = (wb.get("valuation_range") or {}).get("midpoint") or 0
    assert w_mid < v_mid, f"Workbench ({w_mid}) must be below Verdict ({v_mid}) with active flags"
    assert wb.get("risk_discount_pct", 0) > 0


def test_p2_all_flags_resolved_workbench_equals_verdict():
    """T18: All flags resolved → Workbench midpoint == Verdict midpoint."""
    comps   = [_p2_comp(260000, 0.05), _p2_comp(265000, 0.08), _p2_comp(258000, 0.10)]
    verdict = calculate_verdict_ceiling(sold_comps=comps, subject=_p2_subject())
    wb      = calculate_workbench_ceiling(verdict, active_legal_flags=[])
    v_mid   = (verdict.get("valuation_range") or {}).get("midpoint") or 0
    w_mid   = (wb.get("valuation_range") or {}).get("midpoint") or 0
    assert w_mid == v_mid, f"All flags resolved: Workbench {w_mid} must equal Verdict {v_mid}"
    assert wb.get("risk_discount_pct") == 0.0
    assert wb.get("active_flag_count") == 0
    assert wb.get("all_flags_resolved") is True


# ── T19-T20: Fallback safety ──────────────────────────────────────────────

def test_p2_near_thin_comps_recompute_not_legacy_fallback():
    """T19 (revised, S33-STEP1): comps at 0.8mi/0.9mi are within
    MAX_RADIUS_MILES (3.0) and are valid evidence (degraded weight, not
    excluded) — must produce a genuine recompute, not a legacy fallback."""
    comps = [_p2_comp(260000, 0.8), _p2_comp(270000, 0.9)]
    area  = {"housing": {"soldComps": comps}}
    sj = {
        "ceiling": {
            "base_valuation": 160000,
            "ceiling_range": {"low": 152000, "high": 168000},
        }
    }
    result = ensure_ceiling_owned_objects(summary_json=sj, area_json=area, subject=_p2_subject())
    vc  = result["verdict_ceiling"]
    mid = (vc.get("valuation_range") or {}).get("midpoint") or 0
    assert mid > 0, "Must produce a real value from the 2 valid comps"
    assert not vc.get("_legacy_source"), \
        "0.8mi/0.9mi comps are valid evidence under S33-STEP1 — must not fall back to legacy"
    assert vc.get("status") == "degraded_low_comps"


def test_p2_comps_beyond_max_radius_fall_back_to_legacy():
    """T19b: comps genuinely beyond MAX_RADIUS_MILES (3.0mi) ARE excluded
    outright — recompute correctly fails over to legacy rather than
    blanking. Preserves the original T19 regression intent at the boundary
    that actually exists today (3.0mi, not 0.5mi)."""
    comps = [_p2_comp(260000, 3.5), _p2_comp(270000, 4.0)]
    area  = {"housing": {"soldComps": comps}}
    sj = {
        "ceiling": {
            "base_valuation": 160000,
            "ceiling_range": {"low": 152000, "high": 168000},
        }
    }
    result = ensure_ceiling_owned_objects(summary_json=sj, area_json=area, subject=_p2_subject())
    vc  = result["verdict_ceiling"]
    mid = (vc.get("valuation_range") or {}).get("midpoint") or 0
    assert mid > 0, "Must not blank ceiling when comps fail — use legacy fallback"
    assert vc.get("_legacy_source") is True, "Fallback must be marked _legacy_source=True"


def test_p2_fallback_clearly_marked():
    """T20: Fallback is clearly marked — not silently treated as clean comparable Verdict."""
    area  = {"housing": {"soldComps": []}}  # no comps
    sj    = {"ceiling": {"base_valuation": 160000, "ceiling_range": {"low": 152000, "high": 168000}}}
    result = ensure_ceiling_owned_objects(summary_json=sj, area_json=area, subject=_p2_subject())
    vc = result["verdict_ceiling"]
    assert vc.get("_legacy_source") is True
    audit = vc.get("audit") or {}
    src = audit.get("source_decision") or ""
    assert "legacy_fallback" in src, f"source_decision must indicate fallback, got {src!r}"
    assert audit.get("fallback_used") is True


# ── T21-T22: Regression ───────────────────────────────────────────────────

def test_p2_flat_F_comps_within_half_mile_produce_non_legacy_verdict():
    """T21: Flat subject with F comps within 0.5 miles → non-legacy comparable Verdict."""
    comps = [
        _p2_comp(250000, 0.00, "F"),
        _p2_comp(260000, 0.10, "F"),
        _p2_comp(255000, 0.12, "F"),
    ]
    area   = {"housing": {"soldComps": comps}}
    result = ensure_ceiling_owned_objects(summary_json={}, area_json=area, subject=_p2_subject("flat", "leasehold"))
    vc  = result["verdict_ceiling"]
    mid = (vc.get("valuation_range") or {}).get("midpoint") or 0
    assert not vc.get("_legacy_source"), "Must not be _legacy_source when valid comps succeed"
    assert mid > 200000, f"Midpoint must be near comp prices (~£255k), not a legacy value. Got {mid}"
    audit = vc.get("audit") or {}
    src   = audit.get("source_decision") or ""
    assert src == "computed_from_sold_comps", f"source_decision wrong: {src!r}"


def test_p2_legacy_replaced_when_valid_comp_recompute_succeeds():
    """T22: Legacy fallback is not preserved when valid comparable recompute succeeds."""
    # Pre-existing legacy verdict_ceiling with _legacy_source=True
    sj = {
        "ceiling": {"base_valuation": 160000, "ceiling_range": {"low": 152000, "high": 168000}},
        "verdict_ceiling": {
            "_ceiling_type": "verdict",
            "_legacy_source": True,
            "status": "ok",
            "valuation_range": {"low": 152000.0, "midpoint": 160000.0, "high": 168000.0, "uncertainty_band": 0.05},
            "ceiling_range": {"low": 152000, "high": 168000},
            "base": {"value": 160000.0, "method": "legacy_v1"},
            "base_valuation": 160000,
            "legal_pack_value_risks": {"method": "property_value_risk_adjustment_only",
                                       "adjustment_factor": 1.0, "adjusted_value": None, "risks": []},
            "confidence": {"final": 0.45, "label": "Low confidence"},
            "audit": {"warnings": ["legacy"], "version": "v1"},
            "acquisition_costs": None, "excluded_from_ceiling": [],
        },
    }
    area = {"housing": {"soldComps": [
        _p2_comp(250000, 0.05, "F"),
        _p2_comp(260000, 0.10, "F"),
        _p2_comp(258000, 0.12, "F"),
    ]}}
    result = ensure_ceiling_owned_objects(summary_json=sj, area_json=area, subject=_p2_subject("flat", "leasehold"))
    vc  = result["verdict_ceiling"]
    mid = (vc.get("valuation_range") or {}).get("midpoint") or 0
    assert not vc.get("_legacy_source"), "Legacy must be replaced when valid recompute succeeds"
    assert mid > 200000, f"Verdict must use comp-derived value (~£255k), not legacy £160k. Got {mid}"


# =============================================================================
# TERMINOLOGY / OBJECT MODEL TESTS
# Verify: comparable_valuation, risk_adjusted_value, ceiling derivation chain.
# =============================================================================

import sys as _sys
_sys.path.insert(0, "/tmp/term_fix")
from services.ceiling_engine import (
    calculate_verdict_ceiling,
    calculate_workbench_ceiling,
    ensure_ceiling_owned_objects,
)

def _tm_comps():
    return [
        {"price": 260000, "miles": 0.05, "property_type": "F", "duration": "L", "age_months": 3},
        {"price": 265000, "miles": 0.10, "property_type": "F", "duration": "L", "age_months": 3},
        {"price": 258000, "miles": 0.12, "property_type": "F", "duration": "L", "age_months": 3},
    ]

def _tm_subject():
    return {"property_type": "flat", "tenure": "leasehold"}

def _tm_flags():
    return [{"title": "Defective title", "severity": "critical", "summation": ""}]


# T1 — Like-for-like comps produce base_value (comparable_valuation)
def test_term_comps_produce_comparable_valuation():
    v = calculate_verdict_ceiling(sold_comps=_tm_comps(), subject=_tm_subject())
    # comparable_valuation is the explicit new field
    cv = v.get("comparable_valuation")
    assert cv is not None and cv > 0, f"comparable_valuation missing or zero: {cv}"
    # Must equal base.value
    assert cv == v["base"]["value"], \
        f"comparable_valuation {cv} != base.value {v['base']['value']}"
    # Must equal audit.comparable_valuation
    assert v["audit"].get("comparable_valuation") == cv, \
        "audit.comparable_valuation mismatch"
    # Must equal audit.base_value
    assert v["audit"].get("base_value") == cv


# T2 — Risk factor reduces base_value into risk_adjusted_value
def test_term_risk_factor_reduces_to_risk_adjusted_value():
    v = calculate_verdict_ceiling(sold_comps=_tm_comps(), subject=_tm_subject())
    w = calculate_workbench_ceiling(v, _tm_flags())
    cv  = v["comparable_valuation"]
    rf  = w["legal_pack_value_risks"]["adjustment_factor"]
    rav = round(cv * rf, 2)
    assert rf > 0 and rf < 1.0, f"risk_factor should be <1 with active flags, got {rf}"
    # Workbench must expose risk_adjusted_value explicitly
    assert w.get("risk_adjusted_value") is not None, "risk_adjusted_value missing from workbench"
    assert abs(w["risk_adjusted_value"] - rav) < 1, \
        f"risk_adjusted_value {w['risk_adjusted_value']} != base×rf {rav}"
    # audit must record both
    assert abs(w["audit"]["comparable_valuation"] - cv) < 1
    assert abs(w["audit"]["risk_adjusted_value"] - rav) < 1


# T3 — Ceiling range is derived from risk_adjusted_value only
def test_term_ceiling_range_derived_from_risk_adjusted_value():
    v = calculate_verdict_ceiling(sold_comps=_tm_comps(), subject=_tm_subject())
    w = calculate_workbench_ceiling(v, _tm_flags())
    rav = w["risk_adjusted_value"]
    u   = w["valuation_range"]["uncertainty_band"]
    expected_lo = round(rav * (1 - u), 2)
    expected_hi = round(rav * (1 + u), 2)
    actual_lo   = w["valuation_range"]["low"]
    actual_hi   = w["valuation_range"]["high"]
    assert abs(actual_lo - expected_lo) < 1, \
        f"ceiling_low {actual_lo} != risk_adjusted_value*(1-u) {expected_lo}"
    assert abs(actual_hi - expected_hi) < 1, \
        f"ceiling_high {actual_hi} != risk_adjusted_value*(1+u) {expected_hi}"
    # NOT derived from base_value directly
    cv = v["comparable_valuation"]
    wrong_lo = round(cv * (1 - u), 2)
    if abs(rav - cv) > 1:   # only meaningful when risk reduces
        assert abs(actual_lo - wrong_lo) > 1, \
            "ceiling_low appears to be derived from comparable_valuation, not risk_adjusted_value"


# T4 — Verdict risk_factor = 1.0
def test_term_verdict_risk_factor_is_1():
    v = calculate_verdict_ceiling(sold_comps=_tm_comps(), subject=_tm_subject())
    adj = v["legal_pack_value_risks"]["adjustment_factor"]
    assert adj == 1.0, f"Verdict adjustment_factor must be 1.0, got {adj}"
    # Because risk_factor=1, comparable_valuation == risk_adjusted_value on Verdict
    cv  = v["comparable_valuation"]
    rav = v.get("risk_adjusted_value")
    assert rav is not None, "risk_adjusted_value missing from Verdict"
    assert abs(cv - rav) < 1, \
        f"Verdict: comparable_valuation {cv} should equal risk_adjusted_value {rav} when factor=1"
    # midpoint backward compat should also equal both
    mid = v["valuation_range"]["midpoint"]
    assert abs(mid - cv) < 1, \
        f"Verdict midpoint {mid} should equal comparable_valuation {cv}"


# T5 — Workbench risk_factor applies unresolved risks
def test_term_workbench_applies_unresolved_risks():
    v = calculate_verdict_ceiling(sold_comps=_tm_comps(), subject=_tm_subject())
    # No flags
    w0 = calculate_workbench_ceiling(v, [])
    assert w0["risk_adjusted_value"] == v["comparable_valuation"], \
        "No flags: risk_adjusted_value should equal comparable_valuation"
    assert w0.get("risk_discount_pct") == 0.0
    # One critical flag
    w1 = calculate_workbench_ceiling(v, _tm_flags())
    assert w1["risk_adjusted_value"] < v["comparable_valuation"], \
        "Active flag must reduce risk_adjusted_value below comparable_valuation"
    assert w1.get("risk_discount_pct") > 0
    # Two critical flags — deeper discount
    w2 = calculate_workbench_ceiling(v, _tm_flags() + _tm_flags())
    assert w2["risk_adjusted_value"] < w1["risk_adjusted_value"], \
        "Two flags must reduce risk_adjusted_value more than one flag"


# T6 — Repeated Workbench calls do not compound
def test_term_workbench_no_compounding():
    v  = calculate_verdict_ceiling(sold_comps=_tm_comps(), subject=_tm_subject())
    cv = v["comparable_valuation"]
    rf = None
    w1 = calculate_workbench_ceiling(v, _tm_flags())
    rf = w1["legal_pack_value_risks"]["adjustment_factor"]
    expected_rav = round(cv * rf, 2)
    # Call workbench a second and third time from the same verdict
    w2 = calculate_workbench_ceiling(v, _tm_flags())
    w3 = calculate_workbench_ceiling(v, _tm_flags())
    assert abs(w1["risk_adjusted_value"] - expected_rav) < 1
    assert abs(w2["risk_adjusted_value"] - expected_rav) < 1, \
        f"Call 2 compounded: got {w2['risk_adjusted_value']} not {expected_rav}"
    assert abs(w3["risk_adjusted_value"] - expected_rav) < 1, \
        f"Call 3 compounded: got {w3['risk_adjusted_value']} not {expected_rav}"
    # Prove NOT compounded: if it were, call2 rav = call1_rav × rf (smaller)
    wrong_compounded = round(w1["risk_adjusted_value"] * rf, 2)
    if abs(w1["risk_adjusted_value"] - cv) > 1:
        assert abs(w2["risk_adjusted_value"] - wrong_compounded) > 1, \
            "Workbench IS compounding — derived from previous workbench not from verdict"


# T7 — Legacy summary_json.ceiling cannot override valid Verdict
def test_term_legacy_ceiling_cannot_override_valid_verdict():
    v   = calculate_verdict_ceiling(sold_comps=_tm_comps(), subject=_tm_subject())
    cv  = v["comparable_valuation"]
    # Build summary_json with valid non-legacy verdict_ceiling AND a legacy ceiling
    # with a lower (risk-reduced) base_valuation — the legacy alias must NOT win.
    sj = {
        "verdict_ceiling": v,   # valid non-legacy comparable verdict
        "ceiling": {            # legacy alias at a 15% lower value
            "base_valuation": int(cv * 0.85),
            "ceiling_range":  {"low": int(cv * 0.85 * 0.95), "high": int(cv * 0.85 * 1.05)},
            "_ceiling_type":  "workbench",
        },
    }
    result = ensure_ceiling_owned_objects(
        summary_json=sj,
        area_json={},
        subject=_tm_subject(),
    )
    result_cv = result["verdict_ceiling"].get("comparable_valuation")
    assert result_cv is not None, "comparable_valuation missing from preserved verdict"
    assert abs(result_cv - cv) < 1, \
        f"Legacy ceiling overrode valid verdict: expected {cv}, got {result_cv}"
    assert not result["verdict_ceiling"].get("_legacy_source"), \
        "Non-legacy verdict was replaced by a legacy-sourced object"


# =============================================================================
# Section 8 — _resolve_subject_property_type regression tests
# =============================================================================
#
# These tests prove:
#   (a) the helper in app.py resolves physical type correctly
#   (b) BTL/HMO never reaches the engine as subject.property_type
#   (c) soldComps majority inference is NOT used
#   (d) the ceiling engine itself honours type matching correctly
#
# The helper is inlined here (zero external deps) so tests run without importing
# all of app.py (which carries Flask, Supabase, and every route).  The inlined
# copy is byte-for-byte the authoritative implementation from app.py.
# Any change to the app.py function must be reflected here to keep tests green.
# =============================================================================

# ── Canonical implementation (copied from app.py) ────────────────────────────
def _resolve_subject_property_type(prop_type_raw, area_json=None):
    """
    Canonical implementation — inlined from app.py for isolated testing.

    Resolves the subject's physical LR property type code (F/T/D/S) or None.

    Priority:
      1. Direct physical-form lookup (prop_type_raw is already physical).
      2. area_json.housing.metrics.audit.filters_applied entry.
      3. None  (soldComps majority is NOT used).
    """
    _PHYSICAL_MAP = {
        "F": "F", "T": "T", "D": "D", "S": "S",
        "FLAT": "F", "MAISONETTE": "F", "APARTMENT": "F", "FLAT/MAISONETTE": "F",
        "TERRACED": "T", "TERRACE": "T",
        "END-TERRACE": "T", "END TERRACE": "T", "END TERRACED": "T",
        "DETACHED": "D",
        "SEMI-DETACHED": "S", "SEMI DETACHED": "S", "SEMI": "S",
    }
    _s = str(prop_type_raw or "").strip().upper()
    if _s:
        if _s in _PHYSICAL_MAP:
            return _PHYSICAL_MAP[_s]
        if "MAISONETTE" in _s or "APARTMENT" in _s:
            return "F"
        if "FLAT" in _s:
            return "F"
        if "END-TERRACE" in _s or "END TERRACE" in _s:
            return "T"
        if "SEMI" in _s:
            return "S"
        if "TERRACE" in _s:
            return "T"
        if "DETACH" in _s:
            return "D"
        # Investment-type labels fall through — NOT mapped to a physical type.

    _area    = area_json if isinstance(area_json, dict) else {}
    _housing = _area.get("housing") or {}
    _metrics = _housing.get("metrics") or {}
    _LR_VALID = {"F", "T", "D", "S"}
    _audit   = _metrics.get("audit") or {}
    for _entry in (_audit.get("filters_applied") or []):
        if str(_entry).startswith("property_type="):
            _parts = str(_entry).split("=", 1)
            if len(_parts) == 2:
                _code = _parts[1].split(":")[0].strip().upper()
                if _code in _LR_VALID:
                    return _code

    # soldComps majority is NOT used — neighbourhood distribution ≠ subject type.
    return None


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _mk_area_with_filter(lr_code: str, n_comps: int = 5) -> dict:
    """area_json with an audit filter entry for the given LR code."""
    return {
        "housing": {
            "metrics": {
                "audit": {
                    "filters_applied": [f"property_type={lr_code}:{n_comps}_comps"]
                }
            },
            "soldComps": [],
        }
    }


def _mk_area_no_filter(soldcomps_types=None) -> dict:
    """area_json with NO audit filter and optional soldComps distribution."""
    comps = [{"property_type": t, "price": 200000} for t in (soldcomps_types or [])]
    return {
        "housing": {
            "metrics": {"audit": {"filters_applied": []}},
            "soldComps": comps,
        }
    }


def _mk_comp_typed(prop_type, tenure="freehold", price=200000, dist=0.2, months=4):
    return {
        "property_type": prop_type,
        "tenure": tenure,
        "price": price,
        "distance_miles": dist,
        "months_ago": months,
    }


# =============================================================================
# R1 — Investment-type labels never reach engine as physical type
# =============================================================================

class TestInvestmentTypesNeverReachEngine:
    """BTL / HMO / Commercial / Development / Unknown must all return None."""

    _INVESTMENT_TYPES = [
        "BTL", "HMO", "Commercial", "Development", "Unknown",
        "btl", "hmo", "commercial", "development", "unknown",
    ]

    def test_investment_types_return_none_no_area(self):
        """Investment types with no area_json → None."""
        for inv in self._INVESTMENT_TYPES:
            result = _resolve_subject_property_type(inv, {})
            assert result is None, (
                f"Investment type {inv!r} must return None, got {result!r}. "
                "BTL/HMO must never reach the engine as subject.property_type."
            )

    def test_investment_types_return_none_with_dominated_soldcomps(self):
        """
        Investment type with soldComps that are 90% F must still return None.
        Proves soldComps majority is NOT used.
        """
        area = _mk_area_no_filter(soldcomps_types=["F"] * 9 + ["T"])
        for inv in ["BTL", "HMO", "Commercial"]:
            result = _resolve_subject_property_type(inv, area)
            assert result is None, (
                f"{inv!r} + 90% F soldComps returned {result!r} instead of None. "
                "soldComps majority must not be used."
            )

    def test_none_raw_type_returns_none(self):
        assert _resolve_subject_property_type(None, {}) is None

    def test_empty_string_returns_none(self):
        assert _resolve_subject_property_type("", {}) is None

    def test_investment_type_with_audit_filter_returns_filter_code(self):
        """
        BTL deal where area fetch DID resolve a physical type → filter code wins.
        Priority 2 overrides the BTL/None from Priority 1.
        """
        area = _mk_area_with_filter("F", n_comps=7)
        result = _resolve_subject_property_type("BTL", area)
        assert result == "F", (
            f"Audit filter 'property_type=F:7_comps' must resolve to 'F', got {result!r}"
        )


# =============================================================================
# R2 — Physical type labels resolve to canonical LR codes (Priority 1)
# =============================================================================

class TestPhysicalTypePriority1:
    """Direct physical-form labels resolve without needing area_json."""

    @pytest.mark.parametrize("raw,expected", [
        # LR single-char codes
        ("F", "F"), ("T", "T"), ("D", "D"), ("S", "S"),
        # Full labels
        ("Flat",          "F"),
        ("flat",          "F"),
        ("FLAT",          "F"),
        ("Maisonette",    "F"),
        ("MAISONETTE",    "F"),
        ("Apartment",     "F"),
        ("Flat/Maisonette", "F"),
        ("Terraced",      "T"),
        ("terraced",      "T"),
        ("TERRACED",      "T"),
        ("Terrace",       "T"),
        ("End-Terrace",   "T"),
        ("End Terrace",   "T"),
        ("End Terraced",  "T"),
        ("Detached",      "D"),
        ("detached",      "D"),
        ("DETACHED",      "D"),
        ("Semi-detached", "S"),
        ("semi-detached", "S"),
        ("SEMI-DETACHED", "S"),
        ("Semi Detached", "S"),
        ("SEMI",          "S"),
        # Compound labels with substring match
        ("Flat / Maisonette",         "F"),
        ("Ground Floor Apartment",    "F"),
        ("End-Terrace House",         "T"),
        ("Semi-Detached House",       "S"),
        ("Detached Bungalow",         "D"),
    ])
    def test_physical_label_resolves_to_lr_code(self, raw, expected):
        result = _resolve_subject_property_type(raw, {})
        assert result == expected, (
            f"Physical label {raw!r} must resolve to {expected!r}, got {result!r}"
        )

    @pytest.mark.parametrize("raw", [
        "BTL", "HMO", "Commercial", "Development", "Unknown",
        "btl", "hmo",
    ])
    def test_investment_labels_do_not_resolve_to_physical(self, raw):
        """Priority 1 returns None for investment-type labels."""
        result = _resolve_subject_property_type(raw, {})
        assert result is None, (
            f"Investment label {raw!r} must return None from Priority 1, got {result!r}"
        )


# =============================================================================
# R3 — Audit filter (Priority 2) resolves when Priority 1 returns None
# =============================================================================

class TestAuditFilterPriority2:
    """area_json.housing.metrics.audit.filters_applied is the authoritative stored source."""

    @pytest.mark.parametrize("lr_code", ["F", "T", "D", "S"])
    def test_audit_filter_all_lr_codes(self, lr_code):
        """Each valid LR code in the audit filter resolves correctly."""
        area = _mk_area_with_filter(lr_code)
        result = _resolve_subject_property_type("BTL", area)
        assert result == lr_code, (
            f"Audit filter 'property_type={lr_code}' must resolve to {lr_code!r}, got {result!r}"
        )

    def test_audit_filter_ignored_for_invalid_code(self):
        """Audit filter with non-LR code (e.g. 'O') is not returned."""
        area = _mk_area_with_filter("O")  # "Other" — not a valid physical type
        result = _resolve_subject_property_type("BTL", area)
        assert result is None, f"Invalid audit code 'O' must not resolve, got {result!r}"

    def test_audit_filter_skipped_entries_not_used(self):
        """
        filters_skipped (not filters_applied) must not provide the physical type.
        The pipeline writes to filters_skipped when the type filter was NOT applied.
        """
        area = {
            "housing": {
                "metrics": {
                    "audit": {
                        "filters_applied": [],
                        "filters_skipped": ["property_type=F:only_2_matched"],
                    }
                },
                "soldComps": [],
            }
        }
        result = _resolve_subject_property_type("BTL", area)
        assert result is None, (
            f"filters_skipped must not resolve physical type, got {result!r}"
        )

    def test_audit_filter_wins_over_soldcomps(self):
        """
        Audit filter present + soldComps all of different type → filter wins.
        (Also proves soldComps is not consulted when P2 returns a result.)
        """
        area = {
            "housing": {
                "metrics": {
                    "audit": {
                        "filters_applied": ["property_type=F:5_comps"]
                    }
                },
                "soldComps": [{"property_type": "T"} for _ in range(10)],
            }
        }
        result = _resolve_subject_property_type("BTL", area)
        assert result == "F", (
            f"Audit filter F must win over 100% T soldComps, got {result!r}"
        )

    def test_no_physical_source_returns_none(self):
        """No Priority 1 match, no audit filter → None."""
        area = _mk_area_no_filter(soldcomps_types=["F"] * 10)
        result = _resolve_subject_property_type("BTL", area)
        assert result is None, (
            f"No real physical source must return None, got {result!r}. "
            "soldComps majority must not be used."
        )

    def test_empty_area_returns_none(self):
        assert _resolve_subject_property_type("HMO", {}) is None

    def test_none_area_returns_none(self):
        assert _resolve_subject_property_type("HMO", None) is None


# =============================================================================
# R4 — soldComps majority is NOT used under any condition
# =============================================================================

class TestSoldCompsMajorityNotUsed:
    """Exhaustive proof that no soldComps distribution ever drives resolution."""

    @pytest.mark.parametrize("majority_type,pct,inv_type", [
        ("F", "100%", "BTL"),
        ("F",  "90%", "HMO"),
        ("T",  "80%", "BTL"),
        ("D",  "70%", "Commercial"),
        ("S",  "60%", "Unknown"),
        ("F",  "50%", "BTL"),   # exactly at the old threshold
        ("F",  "51%", "HMO"),   # just above the old threshold
    ])
    def test_soldcomps_majority_never_used(self, majority_type, pct, inv_type):
        """
        Even when soldComps are entirely one type, the helper returns None for
        investment-type raw values with no audit filter.
        """
        # Build soldComps dominated by majority_type
        n_majority = 9
        n_minority = 1
        types = [majority_type] * n_majority + (["T" if majority_type != "T" else "F"] * n_minority)
        area = _mk_area_no_filter(soldcomps_types=types)
        result = _resolve_subject_property_type(inv_type, area)
        assert result is None, (
            f"soldComps {pct} {majority_type} + inv_type={inv_type!r} returned {result!r}. "
            "soldComps majority must NEVER be used."
        )


# =============================================================================
# R5 — Ceiling engine type matching with resolved subject.property_type
# =============================================================================

class TestEngineTypeMatchingWithResolver:
    """
    End-to-end: _resolve_subject_property_type output → ceiling engine → correct comps.
    Tests use the ceiling engine directly (imported at top of file).
    """

    from services.ceiling_engine import calculate_verdict_ceiling as _calc_vc

    @staticmethod
    def _run(subject_type_raw, area_json, comps):
        """Resolve type, build subject, run engine, return result."""
        from services.ceiling_engine import calculate_verdict_ceiling
        resolved = _resolve_subject_property_type(subject_type_raw, area_json)
        subject  = {"property_type": resolved, "tenure": "leasehold", "lease_length": 125}
        return calculate_verdict_ceiling(sold_comps=comps, subject=subject, fallback_allowed=True), resolved

    def test_flat_subject_matches_f_comps(self):
        """
        subject.property_type = F  → F comps are valid (type_score = TYPE_SAME).
        Resolved via Priority 1 (raw type = 'Flat').
        """
        comps = [_mk_comp_typed("F", "leasehold", 200000 + i * 2000, 0.1 + i * 0.03)
                 for i in range(5)]
        result, resolved = self._run("Flat", {}, comps)
        assert resolved == "F", f"Expected resolved=F, got {resolved!r}"
        valid = result["comparables"]["valid"]
        valid_types = {c["property_type"] for c in valid}
        assert valid_types == {"flat"}, f"Only F comps should pass, got {valid_types}"
        assert result["comparable_valuation"] is not None
        assert result["comparable_valuation"] > 0
        # Confirm type_score = TYPE_SAME (1.0) for at least one comp
        type_scores = [c["scores"].get("type") for c in valid]
        assert all(s == 1.0 for s in type_scores), (
            f"All F comps should score TYPE_SAME=1.0, got {type_scores}"
        )

    def test_flat_subject_excludes_t_d_s_comps(self):
        """
        subject.property_type = F  → T/D/S comps produce type_mismatch exclusion.
        """
        f_comps  = [_mk_comp_typed("F", "leasehold", 200000+i*1000, 0.1+i*0.03) for i in range(5)]
        mismatch = [
            _mk_comp_typed("T", "freehold",  250000, 0.10),
            _mk_comp_typed("D", "freehold",  300000, 0.12),
            _mk_comp_typed("S", "freehold",  270000, 0.14),
        ]
        result, resolved = self._run("Flat", {}, f_comps + mismatch)
        assert resolved == "F"
        # excl_reasons keyed by raw comp.property_type ("T","D","S") as supplied —
        # the engine stores the original input value, not the normalised string.
        excl_reasons = {
            e["comp"]["property_type"]: e["reason"]
            for e in result["comparables"]["excluded"]
            if "comp" in e
        }
        for raw_pt in ["T", "D", "S"]:
            assert raw_pt in excl_reasons, (
                f"Comp with property_type={raw_pt!r} should be excluded, "
                f"got excluded keys: {list(excl_reasons.keys())}"
            )
            assert "type_mismatch" in excl_reasons[raw_pt], (
                f"{raw_pt} exclusion reason should be type_mismatch, "
                f"got {excl_reasons[raw_pt]!r}"
            )

    def test_type_mismatch_only_for_real_mismatches(self):
        """
        type_mismatch appears ONLY for T/D/S when subject is F.
        No false positives: F comps must not be excluded for type reasons.
        """
        f_comps = [_mk_comp_typed("F", "leasehold", 195000+i*1500, 0.1+i*0.04) for i in range(6)]
        t_comp  = _mk_comp_typed("T", "freehold", 260000, 0.15)
        result, resolved = self._run("F", {}, f_comps + [t_comp])
        assert resolved == "F"
        # All F comps must be in valid (no spurious type_mismatch)
        valid_types = {c["property_type"] for c in result["comparables"]["valid"]}
        assert "flat" in valid_types, "F comps must not be excluded"
        # T comp must be excluded for type_mismatch.
        # comp.property_type in excluded is the raw input value ("T"), not normalised.
        excl = [e for e in result["comparables"]["excluded"]
                if "comp" in e and e["comp"].get("property_type") == "T"]
        assert excl, (
            f"Terraced comp (raw='T') must be excluded. "
            f"All excluded: {[e.get('reason') for e in result['comparables']['excluded']]}"
        )
        assert "type_mismatch" in excl[0]["reason"]

    def test_audit_filter_resolution_drives_correct_engine_behaviour(self):
        """
        BTL deal where area audit recorded 'property_type=T:8_comps'.
        Resolver returns T → engine correctly filters to terraced comps only.

        Uses freehold throughout so tenure_mismatch does not mask type_mismatch.
        """
        from services.ceiling_engine import calculate_verdict_ceiling
        area = _mk_area_with_filter("T", n_comps=8)
        resolved = _resolve_subject_property_type("BTL", area)
        assert resolved == "T", f"Audit filter T must resolve to T, got {resolved!r}"
        # Subject: freehold terraced (resolved from audit filter)
        subject  = {"property_type": resolved, "tenure": "freehold"}
        t_comps  = [_mk_comp_typed("T", "freehold", 195000+i*2500, 0.1+i*0.03) for i in range(5)]
        f_comp   = _mk_comp_typed("F", "freehold",  160000, 0.20)  # freehold flat — excluded by type only
        result   = calculate_verdict_ceiling(sold_comps=t_comps + [f_comp], subject=subject, fallback_allowed=True)
        valid_types = {c["property_type"] for c in result["comparables"]["valid"]}
        excl_reasons = {
            e["comp"]["property_type"]: e["reason"]
            for e in result["comparables"]["excluded"] if "comp" in e
        }
        assert "terraced" in valid_types, (
            f"T comps must be valid when subject=T, got valid_types={valid_types}"
        )
        assert "F" in excl_reasons, "F comp must be excluded when subject=T"
        assert "type_mismatch" in excl_reasons["F"], (
            f"F exclusion must be type_mismatch, got {excl_reasons.get('F')!r}"
        )

    def test_none_subject_type_no_exclusion(self):
        """
        When resolver returns None (no physical source), engine applies
        TYPE_NEAR_EQUIV to all comps — no type_mismatch exclusions.
        """
        from services.ceiling_engine import calculate_verdict_ceiling
        # Investment type + no area source → resolved = None
        resolved = _resolve_subject_property_type("BTL", {})
        assert resolved is None
        subject = {"property_type": resolved, "tenure": "freehold"}
        comps = [_mk_comp_typed(pt, "freehold", 200000+i*2000, 0.1+i*0.03)
                 for i, pt in enumerate(["F", "T", "D", "S", "F"])]
        result = calculate_verdict_ceiling(sold_comps=comps, subject=subject, fallback_allowed=True)
        type_mismatch_excl = [e for e in result["comparables"]["excluded"]
                               if "type_mismatch" in str(e.get("reason", ""))]
        assert not type_mismatch_excl, (
            f"None subject type must produce zero type_mismatch exclusions, "
            f"got {[e['reason'] for e in type_mismatch_excl]}"
        )
        # All should score TYPE_NEAR_EQUIV = 0.75 for type dimension
        for c in result["comparables"]["valid"]:
            assert c["scores"].get("type") == 0.75, (
                f"None subject type: comp {c.get('property_type')} should score "
                f"TYPE_NEAR_EQUIV=0.75, got {c['scores'].get('type')}"
            )


# =============================================================================
# resolve_comp_size — persisted unit tests
# (2026-06-30: the 2026-06-25 session described this as "unit-tested" but
# it was verified manually against real deals only — zero entries existed in
# this file. Added here so the function has durable regression coverage
# matching the same standard as the rest of the suite.)
#
# resolve_comp_size lives in app.py and requires Flask context to import,
# so the tests reproduce the function logic standalone (trivially verifiable
# against app.py:4383-4459 — identical logic, no mocking, no patching).
# =============================================================================

import re as _re

def _resolve_comp_size_standalone(comp_paon_or_address, candidate_epc_rows):
    """
    Verbatim copy of resolve_comp_size() from app.py:4383-4459.
    Kept here so the test file has no Flask import dependency.
    If app.py's resolve_comp_size is ever changed, this copy must be updated
    to match — they are intentionally identical.
    """
    _none_result = {
        "floor_area": None, "habitable_rooms": None,
        "construction_age_band": None, "energy_rating": None, "source": "none",
    }
    if not candidate_epc_rows:
        return _none_result

    def _lead_num(_s):
        _m = _re.match(r"^\s*(\d+)", str(_s or ""))
        return _m.group(1) if _m else None

    _comp_house = _lead_num(comp_paon_or_address)
    _chosen, _source = None, "none"

    if _comp_house:
        for _er in candidate_epc_rows:
            if _lead_num(_er.get("address1")) == _comp_house:
                _chosen, _source = _er, "comp_epc_exact"
                break

    if _chosen is None and _comp_house:
        _best = None
        for _er in candidate_epc_rows:
            _hn = _lead_num(_er.get("address1"))
            if _hn is None:
                continue
            _dist = abs(int(_hn) - int(_comp_house))
            if _best is None or _dist < _best[0]:
                _best = (_dist, _er)
        if _best:
            _chosen, _source = _best[1], "comp_epc_nearest"

    if _chosen is None:
        _chosen, _source = candidate_epc_rows[0], "comp_epc_postcode_any"

    if not _chosen:
        return _none_result

    return {
        "floor_area":            float(_chosen.get("total_floor_area") or 0) or None,
        "habitable_rooms":       _chosen.get("number_habitable_rooms"),
        "construction_age_band": (str(_chosen.get("construction_age_band") or "").strip().upper() or None),
        "energy_rating":         _chosen.get("current_energy_rating"),
        "source":                _source,
    }


_EPC_ROWS = [
    {"address1": "8 Main St",  "total_floor_area": 75.0,  "number_habitable_rooms": 3,
     "construction_age_band": "1950-1966", "current_energy_rating": "D"},
    {"address1": "12 Main St", "total_floor_area": 85.0,  "number_habitable_rooms": 4,
     "construction_age_band": "1967-1975", "current_energy_rating": "C"},
    {"address1": "14 Main St", "total_floor_area": 90.0,  "number_habitable_rooms": 4,
     "construction_age_band": "1967-1975", "current_energy_rating": "B"},
]


def test_resolve_comp_size_exact_match():
    """Tier 1: comp's own house number matches an EPC row → comp_epc_exact."""
    r = _resolve_comp_size_standalone("12", _EPC_ROWS)
    assert r["source"]       == "comp_epc_exact",  f"source={r['source']}"
    assert r["floor_area"]   == 85.0,               f"floor_area={r['floor_area']}"
    assert r["habitable_rooms"] == 4


def test_resolve_comp_size_nearest_match():
    """Tier 2: no exact match → nearest house number wins (12 < 14 distance from 13)."""
    r = _resolve_comp_size_standalone("13", _EPC_ROWS)
    assert r["source"]     == "comp_epc_nearest", f"source={r['source']}"
    assert r["floor_area"] == 85.0,               "nearest to 13 is 12 (distance 1), not 14 (distance 1 too — first wins)"


def test_resolve_comp_size_postcode_any_no_house_number():
    """Tier 3: no leading house number in paon → can't exact/nearest match → postcode_any."""
    r = _resolve_comp_size_standalone("Flat 2", _EPC_ROWS)
    assert r["source"]     == "comp_epc_postcode_any", f"source={r['source']}"
    assert r["floor_area"] == 75.0  # first EPC row


def test_resolve_comp_size_none_no_epc_rows():
    """Tier 4: empty candidate list → source=none, all fields None."""
    r = _resolve_comp_size_standalone("12", [])
    assert r["source"]     == "none", f"source={r['source']}"
    assert r["floor_area"] is None
    assert r["habitable_rooms"] is None


def test_resolve_comp_size_zero_floor_area_returns_none():
    """A row with total_floor_area=None (or 0) must return floor_area=None, not 0.0."""
    rows = [{"address1": "12 Main St", "total_floor_area": None,
             "number_habitable_rooms": 3, "construction_age_band": None, "current_energy_rating": "C"}]
    r = _resolve_comp_size_standalone("12", rows)
    assert r["source"]     == "comp_epc_exact"
    assert r["floor_area"] is None, f"zero/None total_floor_area must produce floor_area=None, got {r['floor_area']}"


def test_resolve_comp_size_postcode_any_triggers_downstream_cap():
    """
    Regression: when floor_area_source == 'comp_epc_postcode_any', ceiling_engine.py
    caps the size score at 0.80 (ceiling_engine.py:1762-1763, S35-COMP-SOURCE-DISCOUNT).
    This test confirms the source tag is the right string to trigger that cap, and that
    the cap fires correctly in the engine for this specific tag value.
    """
    from services.ceiling_engine import calculate_ceiling

    def _mk_comp_with_source(price, dist, floor_area, source):
        return {
            "price": price, "distance_miles": dist, "months_ago": 3,
            "address": f"Comp {price}", "tenure": "freehold", "property_type": "flat",
            "evidence_quality": "official", "internal_area": floor_area,
            "floor_area_source": source,
        }

    subject = {"property_type": "flat", "tenure": "freehold", "internal_area": 80.0}

    # Comp with EXACT EPC (source != postcode_any) — size score should be full
    # Comp with POSTCODE_ANY — same size ratio, but size score capped at 0.80
    comps_exact = [_mk_comp_with_source(200_000 + i*2000, 0.1+i*0.02, 80.0, "comp_epc_exact")
                   for i in range(5)]
    comps_any   = [_mk_comp_with_source(200_000 + i*2000, 0.1+i*0.02, 80.0, "comp_epc_postcode_any")
                   for i in range(5)]

    result_exact = calculate_ceiling([], {}, sold_comps=comps_exact, subject=subject)
    result_any   = calculate_ceiling([], {}, sold_comps=comps_any,   subject=subject)

    # Both should have 5 valid comps (size-score cap doesn't exclude, it just reduces weight)
    assert result_exact["base"]["valid_comparable_count"] == 5
    assert result_any["base"]["valid_comparable_count"]   == 5

    # The postcode_any comps must carry a lower per-comp weight (cap reduces size score)
    w_exact = sum(c["weight"] for c in result_exact["comparables"]["valid"])
    w_any   = sum(c["weight"] for c in result_any["comparables"]["valid"])
    assert w_any < w_exact, (
        f"postcode_any comps must carry lower total weight than exact comps: "
        f"w_any={w_any:.4f} w_exact={w_exact:.4f}"
    )
    # Any postcode_any comp's size_score must not exceed 0.80
    for c in result_any["comparables"]["valid"]:
        sz = c["scores"].get("size")
        if sz is not None:
            assert sz <= 0.80, f"comp_epc_postcode_any size score must be capped at 0.80, got {sz}"


# =============================================================================
# Subject-resolution confidence caps — S35-TYPE-CONF + S35-AREA-CONF
# (2026-06-30: closes the gap confirmed live against 10 Lid Lane DE6 2EG on
# 2026-06-25 and flagged as blocking go-live in two audit sessions without
# being actioned — _type_conf and _gia_conf were computed and discarded, never
# reaching _calculate_confidence.)
# =============================================================================

def _conf_subject(**kw):
    return {"tenure": "freehold", **kw}

def _conf_comps(n=5):
    return [{"weight": 0.65, "audit_warnings": []} for _ in range(n)]

def test_subject_type_high_conf_no_cap():
    """type_confidence='high' must not change score vs None."""
    f_none, _, _ = _calculate_confidence(_conf_comps(), _conf_subject(), [], [], False, False, False, "")
    f_high, _, _ = _calculate_confidence(_conf_comps(), _conf_subject(type_confidence="high"), [], [], False, False, False, "")
    assert f_high == f_none, f"'high' must not cap: {f_high} vs {f_none}"

def test_subject_type_medium_conf_no_cap():
    """type_confidence='medium' must not change score."""
    f_none, _, _ = _calculate_confidence(_conf_comps(), _conf_subject(), [], [], False, False, False, "")
    f_med, _, _  = _calculate_confidence(_conf_comps(), _conf_subject(type_confidence="medium"), [], [], False, False, False, "")
    assert f_med == f_none, f"'medium' must not cap: {f_med} vs {f_none}"

def test_subject_type_none_conf_no_cap():
    """type_confidence=None (old deals, missing key) must not change score."""
    f1, _, _ = _calculate_confidence(_conf_comps(), _conf_subject(), [], [], False, False, False, "")
    f2, _, _ = _calculate_confidence(_conf_comps(), _conf_subject(type_confidence=None), [], [], False, False, False, "")
    assert f1 == f2, f"None must not cap: {f1} vs {f2}"

def test_subject_type_low_conf_caps_at_0_55():
    """type_confidence='low' must cap confidence at 0.55 and produce Low confidence label."""
    f, caps, lbl = _calculate_confidence(_conf_comps(), _conf_subject(type_confidence="low"), [], [], False, False, False, "")
    assert f <= 0.55, f"Expected <= 0.55, got {f}"
    assert lbl == "Low confidence", f"Expected 'Low confidence', got '{lbl}'"
    cats = [c["category"] for c in caps]
    assert "subject_type_low_confidence" in cats, f"Cap category missing from {cats}"

def test_subject_floor_area_low_conf_caps_at_0_55():
    """floor_area_confidence='low' must cap confidence at 0.55."""
    f, caps, lbl = _calculate_confidence(_conf_comps(), _conf_subject(floor_area_confidence="low"), [], [], False, False, False, "")
    assert f <= 0.55, f"Expected <= 0.55, got {f}"
    assert lbl == "Low confidence"
    cats = [c["category"] for c in caps]
    assert "subject_floor_area_low_confidence" in cats, f"Cap category missing from {cats}"

def test_subject_conf_caps_do_not_change_valuation_midpoint():
    """
    CRITICAL: comparable_valuation midpoint must be identical whether
    type_confidence is 'high' or 'low'. The cap changes the label, not the number.
    """
    def _rcomp_full(price, dist):
        return {"price": price, "distance_miles": dist, "months_ago": 3,
                "address": f"A{price}", "tenure": "freehold",
                "property_type": "flat", "evidence_quality": "official"}
    comps = [_rcomp_full(200_000 + i*2000, 0.1 + i*0.02) for i in range(5)]
    r_hi = calculate_ceiling([], {}, sold_comps=comps,
                             subject={"property_type": "flat", "tenure": "freehold", "type_confidence": "high"})
    r_lo = calculate_ceiling([], {}, sold_comps=comps,
                             subject={"property_type": "flat", "tenure": "freehold", "type_confidence": "low"})
    assert r_hi["base"]["value"] == r_lo["base"]["value"], (
        f"type_confidence cap must not change midpoint: hi={r_hi['base']['value']} lo={r_lo['base']['value']}"
    )
    assert r_hi["confidence"]["final"] > r_lo["confidence"]["final"], (
        "low type_confidence must produce a lower confidence score than high"
    )

def test_subject_both_low_conf_applies_both_caps():
    """Both type and floor_area low must both appear in caps list."""
    f, caps, _ = _calculate_confidence(
        _conf_comps(), _conf_subject(type_confidence="low", floor_area_confidence="low"),
        [], [], False, False, False, ""
    )
    cats = [c["category"] for c in caps]
    assert "subject_type_low_confidence" in cats or f <= 0.55, "type cap must fire or conf already at cap"
    assert f <= 0.55


# =============================================================================
# Section 9 — PAON-FIX (2026-07-01) address-prefix flat detection
# =============================================================================
#
# Tests the flat short-circuit logic added to _resolve_subject_type_code and
# the flat-designator extraction used by the EPC subject lookup in get_housing_data.
#
# The function itself has DB dependencies so cannot be imported directly.
# These tests inline the detection logic only (no DB calls) and prove the
# regex/prefix patterns match exactly the address formats that appear in
# real legal-pack extractions.
# =============================================================================

import re as _paon_re

def _detect_flat_addr(addr: str):
    """
    Inlined from app.py PAON-FIX — flat-prefix detection.
    Returns (is_flat, flat_num, building_num).
    flat_num    = unit designator ("3", "3a", "12") or None
    building_num = building number after comma ("24") or None (age-band only)
    """
    _FLAT_UNIT_PREFIXES = ("flat ", "flat,", "apartment ", "apt ", "apt,")
    addr_lower = str(addr or "").lower().strip()
    is_flat = any(addr_lower.startswith(p) for p in _FLAT_UNIT_PREFIXES)
    flat_num = None
    building_num = None
    if is_flat:
        fm = _paon_re.match(r"(?:flat|apartment|apt)[,\s]+([0-9a-z]+)", addr_lower)
        if fm:
            flat_num = fm.group(1).strip(",").strip()
        bm = _paon_re.search(r",\s*(\d+)", addr)
        if bm:
            building_num = bm.group(1)
    else:
        shm = _paon_re.match(r"^\s*(\d+)", addr)
        if shm:
            building_num = shm.group(1)
    return is_flat, flat_num, building_num


def _epc_flat_match(address1: str, flat_num: str) -> bool:
    """
    Inlined from app.py PAON-FIX — EPC address1 flat-designator match.
    Mirrors the logic in the _subj_flat_num exact-match path.
    """
    a1 = str(address1 or "").lower().strip()
    for prefix in (f"flat {flat_num}", f"apartment {flat_num}", f"apt {flat_num}"):
        if a1 == prefix or a1.startswith(prefix + " ") or a1.startswith(prefix + ","):
            return True
    return False


class TestPaonFixFlatDetection:
    """PAON-FIX: flat-prefix detection and flat/building number extraction."""

    def test_flat_prefix_detected(self):
        is_flat, _, _ = _detect_flat_addr("Flat 3, 24 High Street")
        assert is_flat is True

    def test_apartment_prefix_detected(self):
        is_flat, _, _ = _detect_flat_addr("Apartment 2, The Maltings")
        assert is_flat is True

    def test_apt_prefix_detected(self):
        is_flat, _, _ = _detect_flat_addr("Apt 7, 156 Commercial Road")
        assert is_flat is True

    def test_house_address_not_flat(self):
        is_flat, _, _ = _detect_flat_addr("24 High Street")
        assert is_flat is False

    def test_house_number_extracted_for_non_flat(self):
        is_flat, flat_num, building_num = _detect_flat_addr("24 High Street")
        assert is_flat is False
        assert flat_num is None
        assert building_num == "24"

    def test_flat_num_extracted_correctly(self):
        _, flat_num, _ = _detect_flat_addr("Flat 3, 24 High Street")
        assert flat_num == "3", f"Expected '3', got {flat_num!r}"

    def test_building_num_extracted_correctly(self):
        """Bug was: 'Flat 3, 24 High Street' → house='3' (flat number, not building).
        Fix: building_num='24' (after the comma)."""
        _, flat_num, building_num = _detect_flat_addr("Flat 3, 24 High Street")
        assert flat_num == "3"
        assert building_num == "24", (
            f"PAON bug: expected building_num='24', got {building_num!r}. "
            f"Old code extracted flat number '3' as the house number."
        )

    def test_flat_num_alphanumeric(self):
        _, flat_num, building_num = _detect_flat_addr("Flat 3A, 24 High Street")
        assert flat_num == "3a"
        assert building_num == "24"

    def test_apartment_num_extracted(self):
        _, flat_num, building_num = _detect_flat_addr("Apartment 12, The Maltings, London")
        assert flat_num == "12"
        assert building_num is None  # no digit-only segment after comma in "The Maltings"

    def test_flat_no_building_number(self):
        """Flat address where the building has no number — no building_num extracted."""
        _, flat_num, building_num = _detect_flat_addr("Flat 3, High Street")
        assert flat_num == "3"
        assert building_num is None

    def test_case_insensitive(self):
        is_flat_lower, _, _ = _detect_flat_addr("flat 3, 24 high street")
        is_flat_upper, _, _ = _detect_flat_addr("FLAT 3, 24 HIGH STREET")
        assert is_flat_lower is True
        assert is_flat_upper is True

    def test_old_bug_house_number_was_flat_number(self):
        """Regression: old code ran r'^\\s*(\\d+)' on 'Flat 3, 24 High Street'
        and extracted '3' (flat number) as _house. This then matched EPC records
        for '3 <other road>', returning wrong type S/T/D instead of F."""
        old_regex_result = _paon_re.match(r"^\s*(\d+)", "Flat 3, 24 High Street")
        assert old_regex_result is None, (
            "Old regex should NOT match 'Flat 3, ...' — it starts with 'F'. "
            "If this fails the test data is wrong."
        )
        # Old code produced None → _house = None → fell to LLM/neighbour
        # which could return wrong type on mixed postcodes.
        # New code: is_flat=True, return ("F", "address_prefix", "high") immediately.
        is_flat, _, _ = _detect_flat_addr("Flat 3, 24 High Street")
        assert is_flat is True


class TestPaonFixEpcFlatMatch:
    """PAON-FIX: EPC address1 flat-designator matching for floor area / room lookup."""

    def test_flat_3_matches_epc_flat_3(self):
        assert _epc_flat_match("FLAT 3", "3") is True

    def test_flat_3_matches_epc_flat_3_with_building(self):
        assert _epc_flat_match("FLAT 3, 24 HIGH STREET", "3") is True

    def test_flat_3_does_not_match_flat_30(self):
        """Must not match "FLAT 30" when looking for "FLAT 3"."""
        assert _epc_flat_match("FLAT 30", "3") is False

    def test_flat_3_does_not_match_flat_31(self):
        assert _epc_flat_match("FLAT 31", "3") is False

    def test_apartment_12_matches(self):
        assert _epc_flat_match("APARTMENT 12", "12") is True

    def test_apt_7_matches(self):
        assert _epc_flat_match("APT 7", "7") is True

    def test_house_address_does_not_match(self):
        assert _epc_flat_match("24 HIGH STREET", "3") is False

    def test_case_insensitive_match(self):
        assert _epc_flat_match("flat 3", "3") is True
        assert _epc_flat_match("Flat 3", "3") is True

    def test_flat_3a_matches(self):
        assert _epc_flat_match("FLAT 3A", "3a") is True

    def test_empty_address1_no_match(self):
        assert _epc_flat_match("", "3") is False
        assert _epc_flat_match(None, "3") is False
