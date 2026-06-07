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
    MIN_REQUIRED_COMPS,
    PREFERRED_COMPS,
    PRIMARY_RADIUS_MILES,
    MAX_TOTAL_VALUE_RISK_ADJ,
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
# TEST 1: 0.5-mile rule — comps outside 0.5 miles excluded from primary base
# =============================================================================

def test_outside_radius_excluded():
    comps = [
        _comp(200_000, 0.3, addr="A"),
        _comp(200_000, 0.4, addr="B"),
        _comp(200_000, 0.51, addr="C"),   # outside — must be excluded
        _comp(200_000, 1.0,  addr="D"),   # outside — must be excluded
    ]
    result = calculate_ceiling([], {}, sold_comps=comps, subject=_subject())
    # Only 2 comps inside radius — insufficient for full valuation but should appear in valid
    valid_addrs  = {c["address"] for c in result["comparables"]["valid"]}
    excl_addrs   = {e["comp"]["address"] for e in result["comparables"]["excluded"] if "comp" in e}
    assert "C" in excl_addrs, "Comp at 0.51 miles must be excluded"
    assert "D" in excl_addrs, "Comp at 1.0 miles must be excluded"
    assert "C" not in valid_addrs
    assert "D" not in valid_addrs
    assert result["base"]["valid_comparable_count"] == 2


def test_all_outside_radius_is_insufficient():
    comps = [_comp(200_000, 0.6, addr="X"), _comp(200_000, 0.9, addr="Y")]
    result = calculate_ceiling([], {}, sold_comps=comps, subject=_subject())
    assert result["status"] == "insufficient_evidence"
    assert result["valuation_range"]["midpoint"] is None


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
    assert result["base"]["method"] == "weighted_median_relational_comparables_0_5_mile"


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
    """legal_pack_value_risk_adjustment_factor = product(1 - value_adjustment_i)"""
    flags = [_flag("Defective title", "critical"), _flag("Short lease", "high")]
    risks = _process_legal_risks(flags)
    factor = _legal_pack_adjustment_factor(risks)
    # critical=0.10, high=0.06 → (1-0.10)×(1-0.06) = 0.90×0.94 = 0.846
    expected = (1 - 0.10) * (1 - 0.06)
    assert abs(factor - expected) < 0.001, f"Expected {expected} got {factor}"


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

    # capped at 0.20
    band_hi = _uncertainty_band(0, [
        {"reason": "tenure_uncertainty"},
        {"reason": "lease_uncertainty"},
        {"reason": "legal_pack_gaps"},
        {"reason": "evidence_quality"},
    ])
    assert band_hi <= 0.20


def test_distance_score_deterministic():
    assert _distance_score(0.05)  == 1.00
    assert _distance_score(0.10)  == 1.00
    assert _distance_score(0.15)  == 0.90
    assert _distance_score(0.25)  == 0.90
    assert _distance_score(0.30)  == 0.80
    assert _distance_score(0.50)  == 0.80
    assert _distance_score(0.51)  is None  # excluded
    assert _distance_score(1.00)  is None


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
    """Workbench midpoint = verdict_midpoint × product(1 - adj_i)."""
    comps   = _rpc_comps_5(200_000)
    verdict = calculate_verdict_ceiling(sold_comps=comps, subject=_subj())
    flags   = [_flag("Defective title", "critical"), _flag("Planning issue", "high")]
    wb      = calculate_workbench_ceiling(verdict, flags)

    vm = verdict["valuation_range"]["midpoint"]
    # critical=0.10, high=0.06 → factor = (1-0.10)×(1-0.06) = 0.846
    expected_factor = (1 - 0.10) * (1 - 0.06)
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
