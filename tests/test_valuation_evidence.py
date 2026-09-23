"""
V-SSOT guards (2026-09-23): a valuation shown to a user must come from sold-price
evidence or user input — never from the guide price, a legacy base with an
invented band, cross-tenure comps the engine rejects, or an inverted new-build
flag; and every risk-adjusted figure must carry a TRUE calibration statement.
Each test names the register item it protects. Run: python3 -m pytest tests -q
"""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
APP = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()


# ── A1 / A2: calibration disclosure is registered and true ──────────────────
def test_a2_disclosure_is_expert_prior_not_outcome_validated():
    from services.ceiling_engine import get_calibration_disclosure
    d = get_calibration_disclosure()
    assert d["status"] == "expert_prior"
    assert "outcome-validated" not in d["summary"]
    assert "not yet validated against any completed auction outcome" in d["summary"]


def test_a1_route_fractions_registered_with_honest_source():
    from services.ceiling_engine import CALIBRATION_METADATA, _SEGMENT_RULES
    e = CALIBRATION_METADATA["segment_route_fractions"]
    assert e["calibration_status"] == "expert_prior"
    assert e["source_type"] == "unsourced_expert_judgement"
    assert e["values"]["rule_count"] == len(_SEGMENT_RULES)


def test_a2_disclosure_travels_with_every_workbench_result():
    from services.ceiling_engine import calculate_workbench_ceiling
    good = {"status": "ok", "comparable_valuation": 200000.0,
            "valuation_range": {"low": 190000.0, "midpoint": 200000.0, "high": 210000.0,
                                "uncertainty_band": 0.05}}
    wb = calculate_workbench_ceiling(verdict_ceiling=good,
                                     active_legal_flags=[{"title": "No Local Search in Pack", "severity": "missing"}])
    cal = wb["legal_pack_value_risks"]["calibration"]
    assert cal["status"] == "expert_prior"


# ── A4 / A5: no caller-supplied base, no legacy-band verdict ────────────────
def test_a4_request_base_valuation_is_ignored():
    assert re.search(r'^\s*base_val\s*=\s*None\s*$', APP, re.M)
    assert 'base_valuation=float(base_val) if base_val else None' not in APP


def test_a4_engine_without_comps_rules_insufficient_evidence():
    from services.ceiling_engine import calculate_verdict_ceiling
    r = calculate_verdict_ceiling(sold_comps=[], subject={}, base_valuation=None,
                                  strategy="BTL", fallback_allowed=True)
    assert r["status"] == "insufficient_evidence"
    assert not r.get("comparable_valuation")


def test_a5_no_legacy_verdict_with_invented_band():
    assert not re.search(r'["\']_legacy_source["\']\s*:\s*True', APP)
    assert '_ub    = 0.05' not in APP


# ── B1: Land Registry old_new — Y = new build, N = established ──────────────
def test_b1_new_build_is_Y():
    step4 = APP[APP.index("# ── STEP 4: NEW BUILD ROUTING"):APP.index("# ── STEP 5: TENURE FILTER")]
    assert 'if _subject_old_new == "Y":' in step4
    assert '== "Y"]' in step4
    assert 'if _subject_old_new == "N":\n            _nb_only' not in step4
    assert "subject_status_unknown_no_filter" in step4


# ── B2 / B3: one tenure rule, applied before the LIMIT ───────────────────────
def test_b3_tenure_filter_is_in_the_sql_before_limit():
    a = APP.index('_HETZNER_COMPS_SQL = """'); b = APP.index('"""', a + 25)
    sql = APP[a:b]
    assert "p.duration = %s::text" in sql
    assert sql.index("p.duration = %s::text") < sql.index("LIMIT %s")
    assert sql.count("%s") == 7
    assert APP.count("_pre_tenure, _pre_tenure, _pt_param, _pt_param") == 2


def test_b2_no_cross_tenure_fallback():
    assert "Tenure contamination: cross-tenure comps included" not in APP
    assert "def _resolve_subject_tenure(" in APP
