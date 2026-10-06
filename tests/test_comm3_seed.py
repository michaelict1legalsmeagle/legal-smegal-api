"""COMM-3 (2026-10-06): every residential seed value is removed from a
non-residential deal — the 8-value pre-V-NODEFAULT seed included — and
nothing the user entered is touched. Fixtures are the stored rows of live
deals (5 Oct 2026 query)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import residential_seed as rs

OLD8 = {"_seeded": True, "_seeded_at": "2026-09-07T22:22:40Z",          # live 50117369
        "inputs": {"ltv_pct": 75, "hold_years": 10, "legal_fees": 1500, "void_weeks": 2, "guide_price": None,
                   "target_yield": 6, "management_pct": 12, "maintenance_pct": 1, "finance_rate_pct": 5.14}}
NEW2 = {"_seeded": True, "_seeded_at": "2026-10-05T10:12:30Z",          # live 3ee5ee1b
        "inputs": {"ltv_pct": 75, "guide_price": None, "target_yield": 6}}


def test_eight_value_seed_fully_removed():
    assert rs.stripped(OLD8, "commercial")["inputs"] == {"guide_price": None}
    assert rs.stripped(NEW2, "mixed_use")["inputs"] == {"guide_price": None}


def test_original_not_mutated_and_metadata_kept():
    out = rs.stripped(OLD8, "commercial")
    assert out["_seeded_at"] == OLD8["_seeded_at"] and OLD8["inputs"]["ltv_pct"] == 75


def test_must_not_touch():
    assert rs.stripped(OLD8, "residential") is None                      # residential keeps its seed
    assert rs.stripped(OLD8, "") is None                                 # not yet classified
    assert rs.stripped({**OLD8, "ok": True}, "commercial") is None       # saved model
    assert rs.stripped({**OLD8, "inputs": {**OLD8["inputs"], "_user_fields": ["legal_fees"]}}, "commercial") is None
    assert rs.stripped({**OLD8, "inputs": {**OLD8["inputs"], "purchase_price": 300000}}, "commercial") is None
    assert rs.stripped({"inputs": {"ltv_pct": 75}}, "commercial") is None  # not a seed
    assert rs.stripped({"_seeded": True, "inputs": {"commercial": {}, "guide_price": None}}, "commercial") is None  # live 254367be, already clean
    out = rs.stripped({**OLD8, "inputs": {**OLD8["inputs"], "legal_fees": 1800}}, "commercial")
    assert out["inputs"] == {"guide_price": None, "legal_fees": 1800}   # a non-seed value stays


def test_deal_class_from_deal_type_for_old_deals():
    assert rs.deal_class({"deal_type": "Mixed Use", "summary_json": {}}) == "mixed_use"     # live 77042684
    assert rs.deal_class({"deal_type": "Commercial", "summary_json": None}) == "commercial" # live 4d83c8e0
    assert rs.deal_class({"deal_type": "Commercial", "summary_json": {"property": {"asset_class": "residential"}}}) == "residential"
    assert rs.deal_class({"deal_type": "Residential", "summary_json": {}}) == ""


# ── valuation method never assumed; every missing input listed at once ──────
import services.commercial_valuation_engine as eng


def test_no_asset_class_runs_no_method():
    for fi in ({}, {"asset_class": ""}, {"asset_class": "shop"},
               {"passing_rent_pa": 45000, "market_rent_pa": 45000, "yield_pct": 7, "tenure": "freehold"}):
        r = eng.calculate_commercial_ceiling(fi)
        assert r["status"] != "ok" and r.get("comparable_valuation") is None
        assert "valuation method depends on it" in r["audit"]["evidence_gaps"][0]


def test_all_missing_inputs_listed_together():
    gaps = eng.calculate_commercial_ceiling({"asset_class": "income_producing_let"})["audit"]["evidence_gaps"]
    assert [g.split(" ")[0:2] for g in gaps] == [["Tenure", "not"], ["No", "passing"], ["Market", "rent"], ["No", "yield"]]
    assert len(eng.calculate_commercial_ceiling({"asset_class": "trade_related"})["audit"]["evidence_gaps"]) == 2
    assert len(eng.calculate_commercial_ceiling({"asset_class": "development_site"})["audit"]["evidence_gaps"]) == 2
    assert len(eng.calculate_commercial_ceiling({"asset_class": "specialised_owner_occupied"})["audit"]["evidence_gaps"]) == 3
    # a leasehold is still refused outright (not a missing input)
    r = eng.calculate_commercial_ceiling({"asset_class": "income_producing_let", "tenure": "leasehold"})
    assert r["status"] == "manual_review_required"
