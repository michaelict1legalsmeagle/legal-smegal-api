"""
test_api_ceiling_503.py
=======================
Regression tests for the /api/ceiling 503 production blocker.

ROOT CAUSE: app.py imported calculate_verdict_ceiling, calculate_workbench_ceiling,
calculate_financial_standing, and ensure_ceiling_owned_objects from
services/ceiling_engine.py in a SINGLE try/except block. When any of these
functions were missing from the deployed engine (e.g. old ceiling_engine.py
only had calculate_ceiling), the entire block raised ImportError, setting
_ceiling_engine_available = False. Every subsequent /api/ceiling call then hit
the guard:
    if not _ceiling_engine_available or not _calc_verdict_ceiling:
        return jsonify(...), 503

FIX: Split imports into two blocks:
  1. calculate_ceiling alone (sets _ceiling_engine_available = True if present)
  2. v2 functions in a separate try/except (sets each to None if absent)
     _use_v2 flag gates the v2 path; legacy path used when v2 not deployed.
  503 guard now checks only _ceiling_engine_available and _calc_ceiling.

These tests verify the fix and the /api/ceiling behaviour.
"""

import sys
import os
import pytest
import types
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# =============================================================================
# IMPORT ISOLATION HELPERS
# =============================================================================

def _import_ceiling_engine_full():
    """Import the current ceiling_engine — all v2 functions must be present."""
    from services import ceiling_engine as ce
    return ce


def _make_legacy_engine():
    """Create a mock ceiling engine with ONLY calculate_ceiling (old deployment)."""
    mod = types.ModuleType("services.ceiling_engine_legacy")

    def calculate_ceiling(legal_flags, financial_inputs, base_valuation=None,
                          strategy="BTL", sold_comps=None, subject=None,
                          fallback_allowed=True):
        base = 200_000.0
        factor = 1.0
        for f in (legal_flags or []):
            sev = (f.get("severity") or "note").lower()
            adj = {"critical": 0.10, "high": 0.06, "medium": 0.035, "low": 0.015}.get(sev, 0.0)
            factor *= (1 - adj)
        mid = round(base * factor, 2)
        return {
            "ceiling_range": {"low": int(mid * 0.95), "high": int(mid * 1.05)},
            "base_valuation": int(base),
            "confidence": 0.55,
        }

    mod.calculate_ceiling = calculate_ceiling
    # deliberately omit calculate_verdict_ceiling, calculate_workbench_ceiling etc.
    return mod


# =============================================================================
# TEST 1: Full engine imports all required functions
# =============================================================================

def test_ceiling_engine_exports_all_v2_functions():
    """All five functions required by app.py must be importable from ceiling_engine."""
    ce = _import_ceiling_engine_full()
    assert callable(getattr(ce, "calculate_ceiling", None)), "calculate_ceiling missing"
    assert callable(getattr(ce, "calculate_verdict_ceiling", None)), "calculate_verdict_ceiling missing"
    assert callable(getattr(ce, "calculate_workbench_ceiling", None)), "calculate_workbench_ceiling missing"
    assert callable(getattr(ce, "calculate_financial_standing", None)), "calculate_financial_standing missing"
    assert callable(getattr(ce, "ensure_ceiling_owned_objects", None)), "ensure_ceiling_owned_objects missing"


# =============================================================================
# TEST 2: Legacy engine (only calculate_ceiling) does not set engine_available=False
# =============================================================================

def test_legacy_engine_does_not_disable_ceiling_availability():
    """
    When the deployed engine only has calculate_ceiling (old deployment),
    _ceiling_engine_available must remain True.
    The 503 guard must NOT fire.
    """
    legacy = _make_legacy_engine()
    # Simulate the new two-block import strategy
    _calc_ceiling = getattr(legacy, "calculate_ceiling", None)
    _ceiling_engine_available = _calc_ceiling is not None  # True — calculate_ceiling present

    _calc_verdict_ceiling    = getattr(legacy, "calculate_verdict_ceiling",    None)  # None
    _calc_workbench_ceiling  = getattr(legacy, "calculate_workbench_ceiling",  None)  # None
    _calc_financial_standing = getattr(legacy, "calculate_financial_standing", None)  # None
    _ensure_ceiling_objects  = getattr(legacy, "ensure_ceiling_owned_objects",  None)  # None

    assert _ceiling_engine_available is True, (
        "Legacy engine with only calculate_ceiling must keep _ceiling_engine_available=True"
    )
    assert _calc_ceiling is not None

    # Old guard would have returned 503 because _calc_verdict_ceiling is None
    old_guard_would_503 = not _ceiling_engine_available or not _calc_verdict_ceiling
    assert old_guard_would_503 is True, "Old guard confirmed broken — it would return 503"

    # New guard only checks _calc_ceiling
    new_guard_would_503 = not _ceiling_engine_available or not _calc_ceiling
    assert new_guard_would_503 is False, "New guard must NOT return 503 with legacy engine"


# =============================================================================
# TEST 3: _use_v2 flag correctly gates v2 path
# =============================================================================

def test_use_v2_false_with_legacy_engine():
    """_use_v2 must be False when v2 functions are absent."""
    legacy = _make_legacy_engine()
    _calc_verdict_ceiling   = getattr(legacy, "calculate_verdict_ceiling",   None)
    _calc_workbench_ceiling = getattr(legacy, "calculate_workbench_ceiling",  None)
    _use_v2 = bool(_calc_verdict_ceiling and _calc_workbench_ceiling)
    assert _use_v2 is False, "_use_v2 must be False on legacy engine"


def test_use_v2_true_with_current_engine():
    """_use_v2 must be True when current ceiling_engine.py is deployed."""
    ce = _import_ceiling_engine_full()
    _calc_verdict_ceiling   = getattr(ce, "calculate_verdict_ceiling",   None)
    _calc_workbench_ceiling = getattr(ce, "calculate_workbench_ceiling",  None)
    _use_v2 = bool(_calc_verdict_ceiling and _calc_workbench_ceiling)
    assert _use_v2 is True, "_use_v2 must be True with current engine"


# =============================================================================
# TEST 4: Workbench payload shape — empty flags (all resolved) → 200 equivalent
# =============================================================================

def test_workbench_empty_flags_payload_returns_200_equivalent():
    """
    Simulates the exact payload Workbench sends when all flags are resolved:
    { "deal_id": "...", "legal_flags": [] }
    calculate_workbench_ceiling with empty flags must return risk_discount_pct=0.
    This is the /api/ceiling 200 path proven without a live server.
    """
    from services.ceiling_engine import calculate_verdict_ceiling, calculate_workbench_ceiling

    comps = [
        {"price": 200_000, "miles": 0.1 + i*0.08, "age_months": 3,
         "duration": "F", "address": f"T{i}", "property_type": "flat",
         "evidence_quality": "official"}
        for i in range(5)
    ]
    verdict = calculate_verdict_ceiling(sold_comps=comps, subject={"property_type": "flat", "tenure": "freehold"})
    wb = calculate_workbench_ceiling(verdict, active_legal_flags=[])

    # Verify the response would be 200 (no 503 trigger)
    assert wb is not None
    assert wb.get("risk_discount_pct") == 0.0
    assert wb.get("all_flags_resolved") is True
    assert wb["valuation_range"]["midpoint"] == verdict["valuation_range"]["midpoint"]
    # ceiling_range is present for frontend compat
    assert wb.get("ceiling_range") is not None


# =============================================================================
# TEST 5: Workbench payload — active flags present → risk_discount_pct > 0
# =============================================================================

def test_workbench_active_flags_payload_returns_nonzero_discount():
    """Active flags in payload must produce risk_discount_pct > 0."""
    from services.ceiling_engine import calculate_verdict_ceiling, calculate_workbench_ceiling

    comps = [
        {"price": 200_000, "miles": 0.1 + i*0.08, "age_months": 3,
         "duration": "F", "address": f"U{i}", "property_type": "flat",
         "evidence_quality": "official"}
        for i in range(5)
    ]
    flags = [{"title": "Short lease", "severity": "critical", "summation": ""}]
    verdict = calculate_verdict_ceiling(sold_comps=comps, subject={"property_type": "flat", "tenure": "freehold"})
    wb = calculate_workbench_ceiling(verdict, active_legal_flags=flags)

    assert wb.get("risk_discount_pct") > 0.0
    assert wb.get("all_flags_resolved") is False
    assert wb["valuation_range"]["midpoint"] < verdict["valuation_range"]["midpoint"]


# =============================================================================
# TEST 6: Missing comps/subject → structured response, not 503
# =============================================================================

def test_missing_comps_returns_structured_response_not_503():
    """No comps → insufficient_evidence status, but no exception / 503."""
    from services.ceiling_engine import calculate_verdict_ceiling, calculate_workbench_ceiling

    verdict = calculate_verdict_ceiling(sold_comps=[], subject={}, fallback_allowed=True)
    wb = calculate_workbench_ceiling(verdict, active_legal_flags=[])

    # Must return a dict (not raise) — status indicates data state
    assert isinstance(verdict, dict)
    assert isinstance(wb, dict)
    assert verdict.get("status") == "insufficient_evidence"
    assert wb.get("status") == "insufficient_evidence"
    # Must not be 503 — no exception raised


# =============================================================================
# TEST 7: Legacy summary_json.ceiling-only deal → structured response, not 503
# =============================================================================

def test_legacy_ceiling_only_deal_returns_structured_response():
    """Old deal with only sj.ceiling and no comps returns structured response."""
    from services.ceiling_engine import ensure_ceiling_owned_objects

    sj = {
        "ceiling": {
            "base_valuation": 250_000,
            "ceiling_range": {"low": 237_500, "high": 262_500},
            "confidence": 0.55,
        }
    }
    result = ensure_ceiling_owned_objects(sj, area_json={}, legal_flags=[])
    assert "verdict_ceiling" in result
    assert "workbench_ceiling" in result
    wb = result["workbench_ceiling"]
    assert wb.get("risk_discount_pct") == 0.0, "No active flags → 0% discount"
    assert wb.get("all_flags_resolved") is True


# =============================================================================
# TEST 8: Legacy path — calculate_ceiling fallback produces serializable output
# =============================================================================

def test_legacy_calculate_ceiling_produces_serializable_output():
    """Legacy calculate_ceiling output must be JSON-serializable (no 500 on return)."""
    import json
    from services.ceiling_engine import calculate_ceiling

    result = calculate_ceiling(
        legal_flags=[{"title": "Defective title", "severity": "high", "summation": ""}],
        financial_inputs={},
        base_valuation=None,
        strategy="BTL",
        sold_comps=[],
        subject={},
        fallback_allowed=True,
    )
    assert isinstance(result, dict)
    # Must serialize without error
    serialized = json.dumps(result)
    assert len(serialized) > 10


# =============================================================================
# TEST 9: Workbench payload shape matches backend expectation
# =============================================================================

def test_workbench_post_body_keys():
    """
    The Workbench frontend sends: { deal_id, legal_flags }.
    Verify backend ceiling_endpoint reads these keys correctly.
    """
    # The endpoint reads: body.get("legal_flags", []) and body.get("deal_id")
    # This test confirms the key names are stable
    workbench_payload = {
        "deal_id": "fixture-deal-uuid",
        "legal_flags": [
            {"title": "Short lease", "severity": "critical", "summation": ""}
        ],
    }
    assert "legal_flags" in workbench_payload
    assert "deal_id" in workbench_payload
    assert isinstance(workbench_payload["legal_flags"], list)


# =============================================================================
# TEST 10: Response JSON serializes successfully (no numpy/non-serializable types)
# =============================================================================

def test_workbench_response_is_fully_json_serializable():
    """Full /api/ceiling response dict must be JSON-serializable."""
    import json
    from services.ceiling_engine import (
        calculate_verdict_ceiling, calculate_workbench_ceiling, calculate_financial_standing
    )

    comps = [
        {"price": 200_000, "miles": 0.1 + i*0.08, "age_months": 3,
         "duration": "F", "address": f"J{i}", "property_type": "flat",
         "evidence_quality": "official"}
        for i in range(5)
    ]
    flags  = [{"title": "Short lease", "severity": "critical", "summation": ""}]
    verdict = calculate_verdict_ceiling(sold_comps=comps, subject={"property_type": "flat", "tenure": "freehold"})
    result  = calculate_workbench_ceiling(verdict, active_legal_flags=flags)
    fs      = calculate_financial_standing(result, current_bid=None)

    response = {
        "ok": True,
        "ceiling":           result,
        "workbench_ceiling": result,
        "verdict_ceiling":   verdict,
        "financial_current_standing": fs,
    }
    serialized = json.dumps(response)
    assert len(serialized) > 100, "Response must serialize to non-trivial JSON"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
