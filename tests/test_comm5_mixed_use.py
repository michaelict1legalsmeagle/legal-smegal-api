"""COMM-5 (2026-10-06): mixed use as an indicative SUM OF PARTS.

Hand-checked worked example (ground-floor shop let at £9,000 at 7%, flats
your figure £250,000):
  shop  9,000 / 0.07           = £128,571.43
  total 128,571.43 + 250,000   = £378,571.43 (gross)
  SDLT on the net price, GOV.UK non-residential AND MIXED bands
  (0% to £150k, 2% to £250k, 5% above): net £370,544.20 ->
  2,000 + 5% x 120,544.20 = £8,027.21; net + SDLT = gross.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import services.commercial_valuation_engine as eng
import commercial_routes as cr

SHOP = {"label": "Ground-floor shop", "method": "income_producing_let",
        "passing_rent_pa": 9000, "market_rent_pa": 9000, "yield_pct": 7}
FLATS = {"label": "Flats above", "method": "entered_value", "value_gbp": 250000,
         "value_basis": "three comparable sales I checked"}
LOT = {"asset_class": "mixed_use", "tenure": "freehold", "nation": "england_ni"}


def test_worked_example():
    r = eng.calculate_commercial_ceiling({**LOT, "parts": [SHOP, FLATS]})
    assert r["status"] == "ok" and r["method"] == "mixed_use_sum_of_parts"
    assert r["comparable_valuation"] == 378571.43
    assert [p["value_gbp"] for p in r["parts"]] == [128571.43, 250000.0]
    pc = r["purchasers_costs"]
    assert pc["net_value_gbp"] == 370544.2 and pc["sdlt_gbp"] == 8027.21
    assert abs(pc["net_value_gbp"] + pc["sdlt_gbp"] - r["comparable_valuation"]) < 1
    assert "not the Market Value of the lot as a whole" in r["method_reasoning"]
    assert "not a RICS Red Book valuation" in r["method_reasoning"]
    assert r["not_rics_valuation"] is True


def test_profits_part_and_pack_costs_and_own_fees():
    pub = {"label": "Pub", "method": "trade_related", "fmop_pa": 40000, "profit_multiplier": 5}
    r = eng.calculate_commercial_ceiling({**LOT, "parts": [pub, FLATS], "purchaser_fees_pct": 1.5,
                                          "pack_costs": [{"basis": "fixed", "amount_gbp": 1500, "plus_vat": True}]})
    assert r["comparable_valuation"] == 450000.0                     # 40,000 x 5 + 250,000
    pc = r["purchasers_costs"]; n = pc["net_value_gbp"]
    assert abs(n + eng._sdlt_non_residential_england_ni(n) + 1800 + 0.015 * n - 450000) < 1


def test_every_missing_input_listed_and_no_figure():
    r = eng.calculate_commercial_ceiling({"asset_class": "mixed_use",
                                          "parts": [{"label": "Shop", "method": "income_producing_let"},
                                                    {"label": "Flats", "method": "entered_value", "value_gbp": 1},
                                                    {"label": "Yard"}]})
    g = r["audit"]["evidence_gaps"]
    assert r["status"] == "insufficient_evidence" and r["comparable_valuation"] is None
    assert sum(x.startswith("Tenure") for x in g) == 1               # lot-level, listed once
    assert sum(x.startswith("Shop:") for x in g) == 3                # rent, market rent, yield
    assert any(x.startswith("Flats: state where your figure comes from") for x in g)
    assert any(x.startswith("Yard: choose how this part is valued") for x in g)


def test_needs_two_parts_and_never_apportions():
    for parts in (None, []):
        r = eng.calculate_commercial_ceiling({**LOT, "parts": parts})
        assert r["status"] != "ok" and "at least two" in r["audit"]["evidence_gaps"][0]
    r = eng.calculate_commercial_ceiling({**LOT, "parts": [SHOP]})
    assert r["status"] != "ok" and r["audit"]["evidence_gaps"][0].startswith("Only 1 part is entered")


def test_leasehold_let_part_refused_not_guessed():
    r = eng.calculate_commercial_ceiling({**LOT, "tenure": "leasehold", "parts": [SHOP, FLATS]})
    assert r["status"] == "manual_review_required" and r["comparable_valuation"] is None


def test_unknown_nation_withholds_net_only():
    r = eng.calculate_commercial_ceiling({**LOT, "nation": None, "parts": [SHOP, FLATS]})
    assert r["status"] == "ok" and r["purchasers_costs"]["status"] == "unavailable"


def test_unsourced_rics_claim_removed():
    src = open(eng.__file__, encoding="utf-8").read()
    assert "RICS practice apportions" not in src and "typically needing a floor-area" not in src


def test_parts_sanitised_on_save():
    dirty = [{"label": "Shop" * 50, "method": "income_producing_let", "passing_rent_pa": "9000",
              "yield_pct": "abc", "evil": "<script>", "value_gbp": float("nan")}, "junk", {"label": "x"}] + [{}] * 20
    out = cr._clean_parts(dirty)
    assert len(out) == 11                                             # 12 kept, the non-dict dropped
    assert out[0] == {"label": ("Shop" * 50)[:80], "method": "income_producing_let", "passing_rent_pa": 9000.0}
    assert cr._clean_parts("not a list") == []


def test_fee_percentage_of_100_or_more_rejected_with_reason():
    # live 254367be stored purchaser_fees_pct 6200 (pounds typed into the % box)
    live = {"asset_class": "mixed_use", "purchaser_fees_pct": 6200,
            "parts": [{"label": "12000", "method": "trade_related", "fmop_pa": 1200, "profit_multiplier": 3}]}
    g = eng.calculate_commercial_ceiling(live)["audit"]["evidence_gaps"]
    assert g[0].startswith("Your own purchase fees are entered as 6200%") and g[1].startswith("Only 1 part")
    full = {**LOT, "parts": [SHOP, FLATS]}
    assert eng.calculate_commercial_ceiling({**full, "purchaser_fees_pct": 100})["status"] == "insufficient_evidence"
    assert eng.calculate_commercial_ceiling({**full, "purchaser_fees_pct": 99.9})["status"] == "ok"
    inv = {"asset_class": "income_producing_let", "tenure": "freehold", "nation": "england_ni",
           "passing_rent_pa": 9000, "market_rent_pa": 9000, "yield_pct": 7}
    assert eng.calculate_commercial_ceiling({**inv, "purchaser_fees_pct": 6200})["status"] == "insufficient_evidence"
    assert eng.calculate_commercial_ceiling({**inv, "purchaser_fees_pct": 1.5})["status"] == "ok"
