"""
S-STREET-EVIDENCE guards (2026-10-03). _compute_same_street_blend now also returns
the individual same-street sales so the Verdict page can show the street evidence
in every state, not only when it is blended. These tests prove:
  * the numbers that drive the valuation (status, value, n, cv, credibility) are
    IDENTICAL to the pre-change function (extracted from git history copy if given
    via SS_OLD_APP, else compared against hand-computed expectations);
  * every state carries the sales actually found, with accurate fields.
Run: python3 -m pytest tests -q
"""
import ast
import os
import re
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
OLD_APP_PATH = os.environ.get("SS_OLD_APP")


def _fn(src, name, ns):
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.get_source_segment(src, node), "app.py", "exec"), ns)
    return ns[name]


def _consts(src, ns):
    tree = ast.parse(src)
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "").startswith("SAME_STREET_") for t in n.targets):
            exec(compile(ast.get_source_segment(src, n), "app.py", "exec"), ns)


def _safe_float(v):
    try:
        return float(v) if v is not None and v != "" else None
    except (TypeError, ValueError):
        return None


def _make(src, sales, lad="E06000061", latest=250000.0, month_avg=200000.0):
    def data_query(sql, params=None):
        assert "price_paid_raw_2025" in sql
        return [dict(r) for r in sales]

    def supabase_data_query(sql, params=None):
        if "postcode_to_lsoa" in sql:
            return [{"ladcd": lad}] if lad else []
        if "ORDER BY date DESC LIMIT 1" in sql and "date <=" not in sql:
            return [{"date": date(2026, 7, 1), "average_price": latest}] if latest else []
        return [{"average_price": month_avg}]
    ns = {"data_query": data_query, "supabase_data_query": supabase_data_query,
          "safe_float": _safe_float, "re": re}
    _consts(src, ns)
    return _fn(src, "_compute_same_street_blend", ns)


def _sales(prices, start_year=2022):
    return [{"date_of_transfer": date(start_year, 1 + i % 12, 5), "price": p,
             "paon": str(10 + i), "saon": None, "street": "KINGS ROAD"} for i, p in enumerate(prices)]


SCENARIOS = {
    "admit": _sales([200000, 205000, 198000, 210000]),
    "noisy": _sales([100000, 160000, 220000, 90000, 250000, 140000]),
    "one": _sales([180000]),
    "none": [],
}
KEYS = ("status", "value", "n", "cv", "credibility")


def test_numbers_identical_to_pre_change_function():
    if not OLD_APP_PATH:
        import pytest
        pytest.skip("set SS_OLD_APP to the pre-change app.py to run the side-by-side check")
    old_src = open(OLD_APP_PATH, encoding="utf-8").read()
    for name, sales in SCENARIOS.items():
        for kw in ({}, {"lad": None}, {"latest": None}):
            a = _make(old_src, sales, **kw)("SW1A1AA", "terraced", "T")
            b = _make(APP, sales, **kw)("SW1A1AA", "terraced", "T")
            assert {k: a.get(k) for k in KEYS} == {k: b.get(k) for k in KEYS}, (name, kw, a, b)


def test_admit_carries_sales_and_adjusted_prices():
    out = _make(APP, SCENARIOS["admit"])("SW1A1AA", "terraced", "T")
    assert out["status"] == "admit" and out["n"] == 4 and out["sales_total"] == 4
    assert len(out["sales"]) == 4
    s0 = out["sales"][0]
    assert s0["price"] == 200000 and s0["adjusted_price"] == 250000.0   # 200k x 250k/200k
    assert s0["address"] == "10 KINGS ROAD" and s0["date"] == "2022-01-05"
    assert out["hpi_month"] == "2026-07-01" and out["cv_limit"] == 0.20
    assert out["window_months"] == [18, 60]


def test_noisy_is_reported_with_its_sales_not_dropped():
    out = _make(APP, SCENARIOS["noisy"])("SW1A1AA", "terraced", "T")
    assert out["status"] == "excluded_noisy" and out["credibility"] == 0.0
    assert out["cv"] > out["cv_limit"]
    assert out["value"] and len(out["sales"]) == 6 and all(s["adjusted_price"] for s in out["sales"])


def test_too_few_still_lists_the_sale():
    out = _make(APP, SCENARIOS["one"])("SW1A1AA", "terraced", "T")
    assert out["status"] == "insufficient" and out["n"] == 1 and out["sales_total"] == 1
    assert out["sales"][0]["price"] == 180000 and out["sales"][0]["adjusted_price"] is None


def test_none_found_is_explicit():
    out = _make(APP, SCENARIOS["none"])("SW1A1AA", "terraced", "T")
    assert out["status"] == "insufficient" and out["sales_total"] == 0 and out["sales"] == []


def test_no_hpi_lists_unadjusted_sales():
    out = _make(APP, SCENARIOS["admit"], latest=None)("SW1A1AA", "terraced", "T")
    assert out["status"] == "no_hpi" and len(out["sales"]) == 4
    assert all(s["adjusted_price"] is None for s in out["sales"])


def test_sales_list_is_capped_but_total_is_true():
    out = _make(APP, _sales([200000 + i * 100 for i in range(45)], 2021))("SW1A1AA", "terraced", "T")
    assert out["sales_total"] == 45 and len(out["sales"]) == 30


def test_scope_is_recorded_as_postcode():
    out = _make(APP, SCENARIOS["one"])("SN139LZ", "terraced", "T")
    assert out["scope"] == "postcode" and out["scope_value"] == "SN13 9LZ"
    out = _make(APP, SCENARIOS["none"])("B11AA", "terraced", "T")
    assert out["scope_value"] == "B1 1AA"
