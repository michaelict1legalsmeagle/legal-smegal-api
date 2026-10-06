"""COMM-2 (2026-10-05): the commercial valuation assumes nothing.

  * no purchaser's-fee default (was 1.8%); own fees deducted only if entered;
  * special-conditions costs (pack_terms) deducted, VAT at 20% where stated;
  * market rent, tenure and nation never assumed;
  * tenure from the pack and nation from the postcode fill blanks, labelled,
    and are never stored as the user's own entries;
  * a commercial deal loses the residential Financial Model seed.

Regression (not in this file, run before delivery): 5,760 fully specified
input sets gave identical gross, net, SDLT, sensitivity, yields and
waterfall amounts on the old and new engine."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import services.commercial_valuation_engine as eng
import commercial_routes as cr
from fixtures_special_conditions import LOT73_SC

BASE = dict(asset_class="income_producing_let", passing_rent_pa=45000, market_rent_pa=45000,
            yield_pct=7.0, unexpired_term_years=5, tenure="freehold", nation="england_ni")


def _run(**kw):
    return eng.calculate_commercial_ceiling({**BASE, **kw})


def test_sdlt_matches_the_gov_uk_worked_example():
    # GOV.UK: freehold commercial property for £275,000 -> SDLT £3,250
    assert eng._sdlt_non_residential_england_ni(275_000) == 3_250


def test_no_fee_default_and_it_is_said():
    r = _run()
    pc = r["purchasers_costs"]
    assert pc["status"] == "ok" and pc["own_fees_entered"] is False and pc["fees_gbp"] is None
    net, gross = pc["net_value_gbp"], r["comparable_valuation"]
    assert abs(net + eng._sdlt_non_residential_england_ni(net) - gross) < 1      # only SDLT deducted
    assert any("own purchase fees" in a for a in r["audit"]["assumptions"])
    assert not hasattr(eng, "DEFAULT_PURCHASER_FEES_PCT")


def test_own_fees_when_entered():
    r = _run(purchaser_fees_pct=1.5)
    pc = r["purchasers_costs"]
    net = pc["net_value_gbp"]
    assert pc["own_fees_entered"] and pc["fees_pct"] == 1.5
    assert abs(net + eng._sdlt_non_residential_england_ni(net) + 0.015 * net - r["comparable_valuation"]) < 1


def test_lot73_special_conditions_costs_are_deducted():
    pack = [{"basis": "fixed", "amount_gbp": 1500, "plus_vat": True},
            {"basis": "fixed", "amount_gbp": 100, "plus_vat": False},
            {"basis": "fixed", "amount_gbp": 150, "plus_vat": True},
            {"basis": "percent_of_price", "percent": 1, "plus_vat": False}]
    r = _run(pack_costs=pack)
    pc = r["purchasers_costs"]
    net = pc["net_value_gbp"]
    expected_pack = 1500 * 1.2 + 100 + 150 * 1.2 + 0.01 * net        # £2,080 + 1% of price
    assert abs(pc["pack_costs_gbp"] - expected_pack) < 0.01
    assert abs(net + eng._sdlt_non_residential_england_ni(net) + expected_pack - r["comparable_valuation"]) < 1
    labels = [w["label"] for w in r["waterfall"]]
    assert any(l.startswith("Costs in the special conditions (4 items") for l in labels)


def test_percentage_minimum_applies():
    pack = [{"basis": "percent_of_price", "percent": 2.75, "minimum_gbp": 7000, "plus_vat": True}]
    assert eng._pack_cost_at(100_000, pack) == 7000 * 1.2           # 2.75% = £2,750 < £7,000 minimum
    assert abs(eng._pack_cost_at(1_000_000, pack) - 27_500 * 1.2) < 1e-6


def test_market_rent_tenure_nation_never_assumed():
    r = _run(market_rent_pa=None)
    assert r["status"] != "ok" and any("Market rent not entered" in g for g in r["audit"]["evidence_gaps"])
    r = _run(tenure=None)
    assert r["status"] != "ok" and any("Tenure not known" in g for g in r["audit"]["evidence_gaps"])
    r = _run(nation=None)
    assert r["status"] == "ok" and r["purchasers_costs"]["status"] == "unavailable"
    assert "Nation not known" in r["purchasers_costs"]["reason"]
    joined = " ".join(r["audit"]["assumptions"])
    assert "assumed equal" not in joined and "bands assumed" not in joined and "FREEHOLD assumed" not in joined


# ── routes: source-filled facts, never stored as the user's ──────────────────
class _Q:
    def __init__(self, sb, name): self.sb, self.name = sb, name
    def select(self, *a): return self
    def eq(self, *a): return self
    def single(self): return self
    def update(self, payload):
        self.sb.updates.append(payload); return self
    def execute(self):
        class R: pass
        r = R(); r.data = self.sb.docs if self.name == "documents" else None; return r


class _SB:
    def __init__(self, docs): self.docs, self.updates = docs, []
    def table(self, name): return _Q(self, name)


REG = {"file_name": "Lot_73_Register.pdf", "doc_type": "title_register",
       "extracted_text": "The Freehold land shown edged with red on the plan of the above Title"}
SC = {"file_name": "Lot_73_SC.pdf", "doc_type": "special_conditions", "extracted_text": LOT73_SC}


def _deal(stored=None, postcode="BT1 1AA"):
    return {"postcode": postcode, "summary_json": {},
            "financials_json": {"inputs": {"commercial": dict(stored or {}), "commercial_provenance": {}}}}


def test_blanks_filled_from_pack_and_postcode_labelled_not_stored():
    deal = _deal({"asset_class": "income_producing_let", "passing_rent_pa": 45000, "market_rent_pa": 45000, "yield_pct": 7})
    fi, pv, ctx = cr._effective_inputs(_SB([SC, REG]), "d", "u", deal)
    assert fi["tenure"] == "freehold" and pv["tenure"]["source"] == "extracted"
    assert "The Freehold land" in pv["tenure"]["citation"]
    assert fi["nation"] == "england_ni" and pv["nation"]["source"] == "postcode"
    assert len(fi["pack_costs"]) == 4                                  # £1,500, £100, £150, 1%
    assert [i["amount_gbp"] for i in ctx["pack_costs_not_deducted"]] == [200.0]   # notice to complete: conditional
    assert "tenure" not in deal["financials_json"]["inputs"]["commercial"]        # nothing written as the user's
    r = eng.calculate_commercial_ceiling(fi, provenance=pv)
    assert r["status"] == "ok" and r["evidence_tier"]["input_sources"]["tenure"] == "extracted"
    assert r["evidence_tier"]["input_sources"]["nation"] == "postcode"


def test_user_entries_win_over_sources():
    deal = _deal({"tenure": "leasehold", "nation": "wales"})
    fi, pv, ctx = cr._effective_inputs(_SB([SC, REG]), "d", "u", deal)
    assert fi["tenure"] == "leasehold" and fi["nation"] == "wales" and ctx["filled_from_sources"] == {}


def test_unresolved_postcode_leaves_nation_unknown(monkeypatch):
    monkeypatch.setattr(cr, "_nation_from_postcode", lambda pc: (None, "XX1 1XX: postcode did not resolve to a nation"))
    fi, pv, ctx = cr._effective_inputs(_SB([]), "d", "u", _deal(postcode="XX1 1XX"))
    assert "nation" not in fi and "did not resolve" in ctx["nation_unresolved"]


def test_nation_from_lsoa_codes(monkeypatch):
    import types
    fake = types.ModuleType("app")
    fake.resolve_lsoa_gss_from_postcode = lambda pc: ({"GL18 1JA": "E01022398", "CF10 1AA": "W01001900",
                                                      "AB11 8EN": "S01006500"}.get(pc), {})
    monkeypatch.setitem(sys.modules, "app", fake)
    assert cr._nation_from_postcode("GL18 1JA")[0] == "england_ni"
    assert cr._nation_from_postcode("CF10 1AA")[0] == "wales"
    assert cr._nation_from_postcode("AB11 8EN")[0] == "scotland"
    assert cr._nation_from_postcode("ZZ9 9ZZ")[0] is None


# ── residential seed removed from commercial deals ───────────────────────────
SEED = {"_seeded": True, "_seeded_at": "2026-10-05T10:36:44Z",
        "inputs": {"ltv_pct": 75, "guide_price": None, "target_yield": 6}}   # live row 254367be


def test_seed_removed_only_while_untouched():
    com = {"property": {"asset_class": "commercial"}}
    sb = _SB([]); deal = {"summary_json": com, "financials_json": {**SEED, "inputs": dict(SEED["inputs"])}}
    assert cr._heal_residential_seed(sb, "d", deal) is True
    assert sb.updates[0]["financials_json"]["inputs"] == {"guide_price": None}
    # COMM-3: a changed value is the user's and stays; untouched seed values go
    sb = _SB([])
    assert cr._heal_residential_seed(sb, "d", {"summary_json": com, "financials_json":
                                               {**SEED, "inputs": {**SEED["inputs"], "target_yield": 7}}}) is True
    assert sb.updates[0]["financials_json"]["inputs"] == {"guide_price": None, "target_yield": 7}
    for touched in ({**SEED, "inputs": {**SEED["inputs"], "_user_fields": ["target_yield"]}},  # user entered it
                    {**SEED, "inputs": {**SEED["inputs"], "purchase_price": 200000}},    # model in use
                    {**SEED, "ok": True}):                                               # saved model
        sb = _SB([])
        assert cr._heal_residential_seed(sb, "d", {"summary_json": com, "financials_json": touched}) is False and sb.updates == []
    sb = _SB([])   # pre-ROUTE-1 deal: class from deal_type (live 77042684 "Mixed Use", 4d83c8e0 "Commercial")
    assert cr._heal_residential_seed(sb, "d", {"deal_type": "Mixed Use", "summary_json": {},
                                                "financials_json": dict(SEED)}) is True
    sb = _SB([])   # a residential deal keeps its seed even if opened on the commercial page
    assert cr._heal_residential_seed(sb, "d", {"summary_json": {"property": {"asset_class": "residential"}},
                                                "financials_json": dict(SEED)}) is False and sb.updates == []


def test_app_strips_seed_for_non_residential_only():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"), encoding="utf-8").read()
    i = src.index("def _strip_residential_seed(")
    body = src[i:src.index("\ndef ", i + 10)]
    assert '("commercial", "mixed_use", "unclassified")' in body
    assert src.count("_strip_residential_seed(") == 4      # definition + analysis, reuse, classify

