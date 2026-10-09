"""PACK-COSTS-1 (9 Oct 2026) — buyer costs stated in the legal pack.

Root cause (verified on live data 9 Oct 2026): the report's buyer's premium, admin fee,
search and seller-legal-cost lines read only the LLM's special_conditions fields. On
HU9 3AQ (deal dd813528) the special conditions state "£121 plus VAT" (cl.36), a search
reimbursement (cl.34) and a conditional "£200 plus VAT" (cl.35), yet the report showed
none of them. pack_costs reads the cost clauses straight from the document text and
quotes them; the report shows those quotes as the authority.

Fixtures are verbatim excerpts of real special-conditions text (see the fixture's _note).
"""
import ast
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pack_costs as pc  # noqa: E402

FIX = json.load(open(os.path.join(ROOT, "tests", "fixtures", "pack_costs_excerpts.json"), encoding="utf-8"))
APP = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
READER = open(os.path.join(ROOT, "pack_reader.py"), encoding="utf-8").read()


def _run(key):
    f = FIX[key]
    return pc.find_costs([{"file_name": f["file_name"], "doc_type": f["doc_type"], "extracted_text": f["text"]}])


def _items(key):
    return [(i["category"], i["amount_gbp"], i["pct_of_price"], i["minimum_gbp"], i["vat"], i["conditional"], i["clause"])
            for i in _run(key)["items"]]


def test_hu9_cost_clauses_found_with_quotes():
    r = _run("hu9_3aq")
    assert _items("hu9_3aq") == [
        ("search_fees", None, None, None, None, True, "34"),
        ("seller_legal_costs", 200.0, None, None, "plus VAT", True, "35"),
        ("seller_legal_costs", 121.0, None, None, "plus VAT", False, "36"),
    ]
    for i in r["items"]:
        assert i["quote"]
        assert i["document"] == FIX["hu9_3aq"]["file_name"]
    assert r["deposit_terms"] == []
    assert r["version"] == pc.VERSION


def test_quotes_are_verbatim_text():
    norm = lambda s: " ".join(s.split())
    for key in ("hu9_3aq", "fee48", "pct275", "sixk", "admin1650", "bp5995", "search1000"):
        src = norm(FIX[key]["text"])
        for i in _run(key)["items"]:
            q = norm(i["quote"]).rstrip("…").strip()
            assert q[:80] in src, (key, q[:80])


def test_percentage_premium_with_minimum_and_deposit():
    assert _items("fee48") == [("buyers_premium", None, 4.8, 6000.0, "inc VAT", False, "21")]
    dep = _run("fee48")["deposit_terms"]
    assert len(dep) == 1 and dep[0]["pct_of_price"] == 5.0 and dep[0]["minimum_gbp"] == 5000.0


def test_written_numbers_and_notice_to_complete_is_conditional():
    assert _items("pct275") == [
        ("seller_legal_costs", None, 1.0, None, "plus VAT", True, "2"),
        ("admin_fee", None, 2.75, None, "plus VAT", False, "3"),
        ("seller_legal_costs", 1180.0, None, None, "plus VAT", False, "3"),
        ("seller_legal_costs", 450.0, None, None, "plus VAT", False, "3"),
    ]


def test_written_pounds_towards_premium():
    assert _items("sixk") == [("buyers_premium", 6900.0, None, None, None, False, "8")]
    assert _items("bp5995") == [("buyers_premium", 5995.0, None, None, None, False, "23")]


def test_search_and_admin():
    assert _items("search1000") == [("search_fees", 1000.0, None, None, None, False, "g")]
    assert _items("admin1650") == [
        ("search_fees", 483.58, None, None, None, False, "5"),
        ("admin_fee", 1650.0, None, None, "inc VAT", False, "6"),
    ]


def test_deposit_ten_percent_minimum():
    assert _run("dep10min")["items"] == []
    dep = _run("dep10min")["deposit_terms"]
    assert len(dep) == 1 and dep[0]["pct_of_price"] == 10.0 and dep[0]["minimum_gbp"] == 3000.0


def test_no_false_costs_from_non_cost_text():
    for key in ("service_charge_table", "generic_offer_text", "remediation"):
        r = _run(key)
        assert r["items"] == [] and r["deposit_terms"] == [], key


def test_empty_and_non_cost_documents():
    assert pc.find_costs([])["items"] == []
    r = pc.find_costs([{"file_name": "epc.pdf", "doc_type": "epc",
                        "extracted_text": "The Buyer shall pay the Seller £5,000 plus VAT."}])
    assert r["items"] == []


def test_backfill_fills_only_empty_fields():
    sc, ct = {}, {}
    assert pc.backfill_fields(sc, ct, _run("hu9_3aq")) == ["seller_legal_costs_gbp"]
    assert sc == {"seller_legal_costs_gbp": 121.0} and ct == {}       # conditional £200 not used

    sc, ct = {"buyers_premium_pct": 3.0}, {}
    filled = pc.backfill_fields(sc, ct, _run("fee48"))
    assert sc["buyers_premium_pct"] == 3.0 and "buyers_premium_pct" not in filled
    assert ct == {"deposit_pct": 5.0}

    sc, ct = {}, {}
    assert pc.backfill_fields(sc, ct, _run("pct275")) == []          # three legal-cost sums: ambiguous
    assert sc == {}

    sc, ct = {}, {}
    pc.backfill_fields(sc, ct, _run("admin1650"))
    assert sc == {"admin_fee_gbp": 1650.0, "search_fee_reimbursement": True}


# ── wiring locks ────────────────────────────────────────────────────────────────
def _func(src, name):
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.get_source_segment(src, n)
    raise AssertionError(name + " not found")


def test_pack_reader_attaches_stated_costs():
    assert "import pack_costs" in READER
    seg = _func(READER, "analyse_pack")
    i_merge = seg.index("merge_facts(")
    i_costs = seg.index("pack_costs.find_costs(documents)")
    assert i_costs > i_merge
    assert '"stated_costs"' in seg and "pack_costs.backfill_fields(" in seg


def test_get_deal_heals_old_deals():
    seg = _func(APP, "get_deal")
    assert "_maybe_attach_stated_costs(deal_id, deal)" in seg
    heal = _func(APP, "_maybe_attach_stated_costs")
    assert '"processing"' in heal                       # never while analysing
    assert '.eq("updated_at", deal["updated_at"])' in heal   # optimistic lock
    assert "_pack_costs.VERSION" in heal                # once per version


class _Res:
    def __init__(self, data): self.data = data


class _Q:
    def __init__(self, log): self.log = log
    def update(self, payload): self.log.append(("update", payload)); return self
    def eq(self, k, v): self.log.append(("eq", k, v)); return self
    def execute(self): return _Res([{"id": "x"}])


class _SB:
    def __init__(self): self.log = []
    def table(self, name): self.log.append(("table", name)); return _Q(self.log)


def _load_heal():
    """Run the real _maybe_attach_stated_costs source with stub globals (no Flask app)."""
    import logging
    src = _func(APP, "_maybe_attach_stated_costs")
    sb = _SB()
    g = {"Dict": dict, "Any": object, "_pack_costs": pc, "supabase": sb,
         "_json_sanitize": lambda x: x, "now_iso": lambda: "2026-10-09T00:00:00+00:00",
         "app": type("A", (), {"logger": logging.getLogger("t")})}
    exec(src, g)
    return g["_maybe_attach_stated_costs"], sb


def test_heal_persists_once_with_lock():
    heal, sb = _load_heal()
    f = FIX["hu9_3aq"]
    deal = {"status": "complete", "updated_at": "T0",
            "summary_json": {"special_conditions": {"buyers_premium_gbp": None}},
            "documents": [{"file_name": f["file_name"], "doc_type": f["doc_type"], "extracted_text": f["text"]}]}
    out = heal("dd813528", deal)
    sc = out["summary_json"]["special_conditions"]
    assert sc["stated_costs"]["version"] == pc.VERSION and len(sc["stated_costs"]["items"]) == 3
    assert ("eq", "updated_at", "T0") in sb.log
    n = len(sb.log)
    heal("dd813528", out)                       # already current: no second write
    assert len(sb.log) == n


def test_heal_skips_processing_and_no_docs():
    heal, sb = _load_heal()
    heal("a", {"status": "processing", "summary_json": {"special_conditions": {}},
               "documents": [{"extracted_text": "x"}]})
    heal("b", {"status": "complete", "summary_json": {"special_conditions": {}}, "documents": []})
    assert sb.log == []
