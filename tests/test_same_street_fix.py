"""
S-STREET-FIX guards (2026-10-03). The same-street credibility blend (S-STREET-BLEND,
15 Aug) never ran in production: 0 of 98 stored verdicts carried it. Two faults in
the _recompute_deal_ceiling wiring, either one fatal on its own:
  1. the token was substring-matched from the comps' majority property_type, which is
     a PPD code ("T", "F"...) - "t" never contains "terrac";
  2. the postcode was stripped with re.sub(r"\\\\s+") - a literal backslash-s, not
     whitespace - so "NN8 1SF" kept its space and missed the pcds_nospace lookup.
It was also only wired into one of the three verdict paths. These tests fail on the
pre-fix app.py and pass on the fixed one. Run: python3 -m pytest tests -q
"""
import ast
import os
import re
import sys

ROOT = os.environ.get("SS_FIX_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
APP = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()


def _load(calls):
    """Extract the two helpers from app.py and bind a recording stub for the
    database-backed _compute_same_street_blend."""
    tree = ast.parse(APP)
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "_same_street_token" in fns and "_same_street_for_subject" in fns, \
        "shared same-street helpers missing from app.py"

    def _stub(pc, tok, code):
        calls.append((pc, tok, code))
        return {"status": "admit", "value": 200000.0, "credibility": 0.5, "n": 4, "cv": 0.05}

    ns = {"re": re, "_compute_same_street_blend": _stub}
    for name in ("_same_street_token", "_same_street_for_subject"):
        exec(compile(ast.get_source_segment(APP, fns[name]), "app.py", "exec"), ns)
    return ns


def test_ppd_codes_map_to_tokens():
    ns = _load([])
    tok = ns["_same_street_token"]
    assert tok("T") == ("terraced", "T")
    assert tok("f") == ("flat", "F")
    assert tok("S") == ("semi", "S")
    assert tok("D") == ("detached", "D")


def test_text_types_still_map():
    tok = _load([])["_same_street_token"]
    assert tok("Semi-detached") == ("semi", "S")          # 'semi' wins over 'detach'
    assert tok("Flat/Maisonette") == ("flat", "F")
    assert tok("End Terrace") == ("terraced", "T")
    assert tok("Detached house") == ("detached", "D")
    assert tok("Apartment") == ("flat", "F")


def test_unknown_types_do_not_blend():
    tok = _load([])["_same_street_token"]
    for v in (None, "", "O", "Other", "Bungalow", "Land"):
        assert tok(v) == (None, None), v


def test_postcode_whitespace_is_stripped():
    calls = []
    ns = _load(calls)
    out = ns["_same_street_for_subject"]("nn8 1sf", "T")
    assert calls == [("NN81SF", "terraced", "T")]
    assert out["status"] == "admit"


def test_no_postcode_or_type_is_off_and_queries_nothing():
    calls = []
    ns = _load(calls)
    assert ns["_same_street_for_subject"](None, "T") == {"status": "off"}
    assert ns["_same_street_for_subject"]("NN8 1SF", "O") == {"status": "off"}
    assert calls == []


def test_all_three_verdict_paths_use_the_helper():
    # one definition + _recompute_deal_ceiling + /api/ceiling fallback + summarise_deal
    assert APP.count("_same_street_for_subject(") == 4
    tree = ast.parse(APP)
    fns = {n.name: ast.get_source_segment(APP, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    for fn in ("_recompute_deal_ceiling", "ceiling_endpoint", "summarise_deal"):
        assert "_same_street_for_subject(" in fns[fn], fn
        assert "_same_street_value" in fns[fn] and "_same_street_credibility" in fns[fn], fn


def test_old_broken_inline_logic_is_gone():
    assert '_ss_t  = str(_inferred_pt' not in APP
    assert '_re_ss.sub(r"\\\\s+"' not in APP


def test_engine_moves_the_valuation_when_fields_are_present():
    """End to end through the real engine: the fields the helper supplies change
    comparable_valuation (engine flag SAME_STREET_BLEND_ENABLED)."""
    from services import ceiling_engine as ce
    comps = [dict(property_type="T", duration="F", price=p, hpi_multiplier=1.0, miles=0.3,
                  age_months=4.0, floor_area=None) for p in
             (150000, 155000, 160000, 162000, 165000, 168000, 170000, 175000)]
    base = dict(property_type="T", tenure="Freehold")
    v0 = ce.calculate_verdict_ceiling(sold_comps=[dict(c) for c in comps], subject=dict(base))
    v1 = ce.calculate_verdict_ceiling(sold_comps=[dict(c) for c in comps],
                                      subject=dict(base, _same_street_value=200000.0,
                                                   _same_street_credibility=0.5))
    assert ce.SAME_STREET_BLEND_ENABLED is True
    assert v0["comparable_valuation"] and v1["comparable_valuation"]
    assert v1["comparable_valuation"] > v0["comparable_valuation"]
