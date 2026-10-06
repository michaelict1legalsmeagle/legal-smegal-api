"""FIN-CORE backend tests (30 Sep 2026). Expected values worked by hand."""
import datetime as dt
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import area_history as ah  # noqa: E402


def _d(y, m=1):
    return dt.date(y, m, 1)


def test_decade_scenarios_hand_values():
    # Jan values: 2000=100, 2001=100, 2002=100, 2010=200, 2011=150, 2012=120
    s = [(_d(2000), 100.0), (_d(2001), 100.0), (_d(2002), 100.0),
         (_d(2010), 200.0), (_d(2011), 150.0), (_d(2012), 120.0)]
    out = ah.decade_scenarios(s)
    assert out["ok"] is True and out["windows"] == 3
    # last window ends 2012: 1.2 ** 0.1 − 1 = 1.8399%
    assert abs(out["last_decade"]["rate"] - 0.018399) < 1e-5
    assert out["last_decade"]["from"] == "2002-01-01" and out["last_decade"]["to"] == "2012-01-01"
    # worst is the same 2012 window; best is 2010: 2 ** 0.1 − 1 = 7.1773%
    assert abs(out["worst_decade"]["rate"] - 0.018399) < 1e-5
    assert abs(out["best_decade"]["rate"] - 0.071773) < 1e-5
    assert out["best_decade"]["to"] == "2010-01-01"


def test_decade_scenarios_too_short_is_null_not_guessed():
    out = ah.decade_scenarios([(_d(2020), 100.0), (_d(2021), 110.0)])
    assert out["ok"] is False and "10 years" in out["reason"]


def test_annualised_rent_matches_stored_north_northants_window():
    # uk_prms_monthly E06000061: index 100 (Jan 2015) → 170.3833 (Mar 2026), 134 months.
    # Hand: ln(1.703833)=0.53288; /11.1667 = 0.047720; e^x − 1 = 4.888%
    series = [(_d(2015, 1), 100.0)] + [(_d(2015 + (i // 12), 1 + i % 12), 120.0) for i in range(1, 134)] \
        + [(_d(2026, 3), 170.3833)]
    out = ah.annualised(series)
    assert out["ok"] is True and out["years"] == 11.17
    assert abs(out["rate"] - 0.04888) < 0.0005


def test_build_area_history_reads_both_tables_with_area_code():
    calls = []

    def q(sql, params):
        calls.append((sql, params))
        if "uk_hpi_monthly" in sql:
            return [{"date": "2000-01-01", "average_price": 100}, {"date": "2010-01-01", "average_price": 200}]
        return [{"period": "2015-01-01", "rent_index": 100}] + \
               [{"period": f"{2015 + i // 12}-{1 + i % 12:02d}-01", "rent_index": 110} for i in range(1, 13)] + \
               [{"period": "2016-02-01", "rent_index": 113}]
    out = ah.build_area_history("E06000061", q)
    assert all(p == ("E06000061",) for _, p in calls) and len(calls) == 2
    assert abs(out["hpi"]["last_decade"]["rate"] - (2 ** 0.1 - 1)) < 1e-9
    assert out["hpi"]["source"].startswith("UK House Price Index")
    assert out["rent"]["ok"] is True and out["rent"]["source"].startswith("ONS")


def test_rates_bench_exposes_comparison_series_and_strategy_staleness(monkeypatch):
    import rates_routes as rr
    from flask import Flask

    today = dt.date.today()
    fresh, old = (today - dt.timedelta(days=10)).isoformat(), (today - dt.timedelta(days=180)).isoformat()

    class _T:
        def __init__(self, rows): self.rows = rows
        def select(self, *_): return self
        def eq(self, *_): return self
        def execute(self):
            class R: pass
            r = R(); r.data = self.rows; return r

    lender_rows = [
        {"strategy": "btl", "name": "A", "rate_from": 4.5, "as_of": fresh, "tags": ["5yr"]},
        {"strategy": "hmo", "name": "B", "rate_from": 5.0, "as_of": old, "tags": ["5yr"]},
        {"strategy": "hmo", "name": "C", "rate_from": 5.1, "as_of": fresh, "tags": ["5yr"]},
    ]
    bench_rows = [
        {"series_code": "IUDBEDR", "rate_pct": 3.75, "as_of": fresh},
        {"series_code": "IUMB6RH", "rate_pct": 4.10, "as_of": fresh},
        {"series_code": "IUMAMNPY", "rate_pct": None, "as_of": None},
    ]

    class _SB:
        def table(self, name):
            return _T(lender_rows if name == "lender_rates" else bench_rows)

    monkeypatch.setattr(rr, "_sb", lambda: _SB())
    app = Flask(__name__)
    app.register_blueprint(rr.rates_bp)
    body = app.test_client().get("/api/rates").get_json()
    assert body["bench"]["bank_rate"] == 3.75 and body["bench"]["bank_rate_as_of"] == fresh
    assert body["bench"]["savings_2y"] == 4.1 and body["bench"]["savings_2y_as_of"] == fresh
    assert body["bench"]["gilt_10y"] is None  # not yet synced → null, never invented
    assert body["meta"]["strategies"]["btl"]["stale"] is False
    assert body["meta"]["strategies"]["hmo"]["stale"] is True   # oldest HMO row is 180 days
    assert body["meta"]["lender_stale"] is False                # legacy flag unchanged


def _load_calc():
    """app.py refuses to import without production secrets, so lift the pure
    functions it needs (safe_float, _FIN_PAGE_KEYS, _calculate_financials) out of
    its source and run them in isolation — the same code that ships."""
    import ast
    from typing import Any, Dict, Optional
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")).read()
    tree = ast.parse(src)
    want = {"safe_float", "_calculate_financials", "_FIN_PAGE_KEYS"}
    nodes = [n for n in tree.body
             if (isinstance(n, ast.FunctionDef) and n.name in want)
             or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in want for t in n.targets))]
    ns = {"Any": Any, "Dict": Dict, "Optional": Optional, "now_iso": lambda: "t", "math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "app.py", "exec"), ns)
    assert {"safe_float", "_calculate_financials", "_FIN_PAGE_KEYS"} <= set(ns)
    return ns["_calculate_financials"]


def test_calculate_financials_no_hidden_defaults_and_full_total():
    calc = _load_calc()
    r = calc({
        "purchase_price": 200000, "buyers_premium_pct": 3, "stamp_duty": 11500,
        "survey_cost": 600, "admin_fee": 350, "acquisition_insurance": 800, "legal_fees": 1800,
        "nation": "england", "_user_fields": ["purchase_price"],
    })
    # 200,000 + 6,000 + 11,500 + 1,800 + 600 + 350 + 800 = 221,050
    assert r["acquisition"]["total_acquisition"] == 221050
    assert r["inputs"]["nation"] == "england" and r["inputs"]["_user_fields"] == ["purchase_price"]
    r2 = calc({"purchase_price": 200000})
    # FIN-CORE-2: not entered stays missing (None) — neither £1,500 / 1% nor a stored 0
    assert r2["inputs"]["legal_fees"] is None and r2["inputs"]["maintenance_pct"] is None
    assert "legal_fees" in r2["missing_inputs"]


def test_whole_pound_amounts_over_10k_are_not_divided_by_100():
    calc = _load_calc()
    r = calc({"purchase_price": 200000, "stamp_duty": 11500, "renovation_cost": 25000})
    assert r["acquisition"]["stamp_duty"] == 11500        # was £115 via the pence heuristic
    assert r["acquisition"]["renovation_cost"] == 25000   # was £250


def test_rev5_lender_fee_pack_costs_and_bridging_count_in_total():
    calc = _load_calc()
    r = calc({"purchase_price": 200000, "stamp_duty": 11500, "lender_fee": 1495, "pack_costs": 1078.8, "bridging_cost": 0})
    # 200,000 + 11,500 + 1,495 + 1,078.80 = 214,073.80
    assert r["acquisition"]["total_acquisition"] == 214073.8
    assert r["inputs"]["lender_fee"] == 1495 and r["inputs"]["pack_costs"] == 1078.8
