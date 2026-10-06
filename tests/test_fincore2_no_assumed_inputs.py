"""FIN-CORE-2 (6 Oct 2026): the residential Financial Model stores no value the
user did not enter — no 0 for a blank, no 6% target yield, no 75% LTV, no
10-year hold. Expected figures worked by hand (live deal 564488a4, Park Vale)."""
import ast
import os
import re

from test_fincore_backend import _load_calc

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

# What Michael entered on 564488a4 (6 Oct 19:44), plus the lender terms then on
# the page and the transfer tax the page computed. Nothing else.
PARK_VALE = {
    "purchase_price": 245400, "monthly_rent": 1300, "stamp_duty": 14678,
    "ltv_pct": 75, "finance_rate_pct": 4.38, "lender_fee": 995, "nation": "england",
    "_user_fields": ["purchase_price", "monthly_rent"],
}


def test_park_vale_figures_match_hand_and_blanks_stay_blank():
    r = _load_calc()(dict(PARK_VALE))
    # 245,400 + 14,678 SDLT + 995 lender fee = 261,073
    assert r["acquisition"]["total_acquisition"] == 261073
    # debt 245,400 × 75% = 184,050; cash in 261,073 − 184,050 = 77,023 (the stored page figure)
    assert r["finance"]["loan_amount"] == 184050 and r["finance"]["equity"] == 77023
    for k in ("legal_fees", "survey_cost", "admin_fee", "acquisition_insurance", "buyers_premium_pct",
              "management_pct", "maintenance_pct", "insurance_pa", "void_weeks", "renovation_cost"):
        assert r["inputs"][k] is None, k            # was stored as 0
        assert k in r["missing_inputs"], k
    assert "target_yield" not in r["inputs"]         # was 6 on every row
    assert r["inputs"]["hold_years"] is None         # was 10
    assert r["model_version"] == "fin-core-2"
    assert "ltv_pct" not in r["missing_inputs"] and "finance_rate_pct" not in r["missing_inputs"]


def test_park_vale_cashflow_hand_value():
    r = _load_calc()(dict(PARK_VALE))
    # rent 15,600/yr, no costs entered → NOI 15,600; interest 184,050 × 4.38% = 8,061.39
    assert r["returns"]["noi"] == 15600
    assert r["finance"]["annual_interest"] == 8061.39
    assert r["returns"]["net_cashflow_pa"] == 7538.61       # = the page's stored 7,538.61
    assert r["returns"]["total_return"] is None              # needs a holding period


def test_no_ltv_means_no_debt_figures_not_a_cash_purchase():
    inp = {k: v for k, v in PARK_VALE.items() if k not in ("ltv_pct",)}
    r = _load_calc()(inp)
    assert r["finance"]["loan_amount"] is None and r["finance"]["equity"] is None
    assert r["finance"]["annual_interest"] is None
    assert r["returns"]["net_cashflow_pa"] is None and r["returns"]["cash_on_cash_pct"] is None
    assert "ltv_pct" in r["missing_inputs"]
    assert r["returns"]["gross_yield_pct"] == round(15600 / 245400 * 100, 2)   # needs no LTV


def test_ltv_zero_is_a_real_cash_purchase():
    inp = {**PARK_VALE, "ltv_pct": 0, "finance_rate_pct": None, "lender_fee": None}
    r = _load_calc()(inp)
    assert r["finance"]["loan_amount"] == 0 and r["finance"]["annual_interest"] == 0
    assert r["returns"]["net_cashflow_pa"] == 15600
    assert "finance_rate_pct" not in r["missing_inputs"] and "lender_fee" not in r["missing_inputs"]


def test_loan_without_rate_gives_no_cashflow_and_names_the_rate():
    inp = {k: v for k, v in PARK_VALE.items() if k not in ("finance_rate_pct", "lender_fee")}
    r = _load_calc()(inp)
    assert r["finance"]["loan_amount"] == 184050 and r["finance"]["annual_interest"] is None
    assert r["returns"]["net_cashflow_pa"] is None
    assert {"finance_rate_pct", "lender_fee"} <= set(r["missing_inputs"])


def test_no_rent_means_no_yield_not_zero():
    inp = {k: v for k, v in PARK_VALE.items() if k != "monthly_rent"}
    r = _load_calc()(inp)
    assert r["returns"]["gross_yield_pct"] is None and r["returns"]["noi"] is None
    assert "monthly_rent" in r["missing_inputs"]


def test_an_entered_zero_is_kept_and_not_listed():
    r = _load_calc()({**PARK_VALE, "legal_fees": 0, "survey_cost": 0, "occupancy_pct": 100, "void_weeks": 0})
    assert r["inputs"]["legal_fees"] == 0 and r["inputs"]["survey_cost"] == 0
    assert "legal_fees" not in r["missing_inputs"] and "void_weeks" not in r["missing_inputs"]


def test_holding_period_entered_gives_total_return():
    r = _load_calc()({**PARK_VALE, "hold_years": 10})
    assert r["returns"]["total_return"] == 75386.1          # 7,538.61 × 10


def _func_src(name):
    src = open(APP, encoding="utf-8").read()
    for n in ast.parse(src).body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.get_source_segment(src, n)
    raise AssertionError(name)


def test_new_deal_and_first_load_seed_no_model_values():
    for fn in ("create_deal", "get_financials"):
        body = _func_src(fn)
        assert not re.search(r'"target_yield"\s*:', body), fn
        assert not re.search(r'"ltv_pct"\s*:', body), fn
