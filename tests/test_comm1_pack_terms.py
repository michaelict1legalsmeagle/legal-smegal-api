"""COMM-1 (2026-10-05): buyer costs and 'what you're buying' from the special
conditions; vacant possession only when stated about the sale; documents
about another property never read by the commercial routes.

Fixtures: tests/fixtures_special_conditions.py (stored text of live deals
3ee5ee1b Lot 73, 66217156 Haslemere, and the Lot 6 document)."""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pack_terms as pt
import commercial_extraction as ce
from fixtures_special_conditions import LOT73_SC, HASLEMERE_SC, LOT6_SC


def _sc(text, name="SC.pdf"):
    return [{"file_name": name, "doc_type": "special_conditions", "extracted_text": text}]


def _flat(t):
    return re.sub(r"\s+", " ", re.sub(r"=== PAGE \S+ ===", " ", t))


def _summary(items):
    return [(i["basis"], i.get("amount_gbp"), i.get("percent"), i.get("minimum_gbp"),
             i["plus_vat"], i.get("per"), i.get("when")) for i in items]


# ── buyer costs ──────────────────────────────────────────────────────────────
def test_lot73_costs_match_the_special_conditions():
    c = pt.extract_buyer_costs(_sc(LOT73_SC))
    assert _summary(c["items"]) == [
        ("fixed", 1500.0, None, None, True, None, "exchange"),      # cl.11
        ("fixed", 100.0, None, None, False, None, "completion"),    # cl.12
        ("fixed", 150.0, None, None, True, None, "completion"),     # cl.13
        ("percent_of_price", None, 1, None, False, None, None),     # cl.15
    ]
    assert c["fixed_total_gbp_ex_vat"] == 1750.0 and c["fixed_total_counts"] == 3
    assert len(c["contingent"]) == 1                                 # cl.14 notice to complete
    n = c["contingent"][0]
    assert n["amount_gbp"] == 200.0 and n["plus_vat"] and n.get("is_minimum")


def test_haslemere_costs_unstated_search_costs_and_percentage():
    c = pt.extract_buyer_costs(_sc(HASLEMERE_SC))
    assert _summary(c["items"]) == [
        ("not_stated", None, None, None, False, None, "completion"),
        ("percent_of_price", None, 2.5, None, True, None, "completion"),
    ]
    assert "local search" in c["items"][0]["as_written"]
    assert c["fixed_total_gbp_ex_vat"] is None                       # nothing fixed is stated
    assert [(i["amount_gbp"], i["plus_vat"], i["per"]) for i in c["contingent"]] == [
        (250.0, True, None), (62.2, False, "day")]


def test_lot6_costs_words_minimums_continuations_and_deposit():
    c = pt.extract_buyer_costs(_sc(LOT6_SC))
    s = _summary(c["items"])
    assert ("fixed", 640, None, None, False, None, "completion") in s       # "six hundred and forty pounds"
    assert ("fixed", 150.0, None, None, True, "transfer", "completion") in s
    assert ("percent_of_price", None, 2.75, 7000, True, None, None) in s    # min £7,000 belongs to the %
    assert ("fixed", 1440, None, None, True, None, None) in s               # "Plus an additional …" sentence
    assert s.count(("fixed", 1000.0, None, None, True, None, "completion")) == 2   # premium + admin fee
    assert sum(1 for x in s if x[0] == "not_stated") == 2                    # service charges; arrears
    assert c["fixed_total_gbp_ex_vat"] == 640 + 1440 + 1000 + 1000          # per-transfer £150 not totalled
    assert all(i.get("amount_gbp") != 8000 for i in c["items"] + c["contingent"])   # the deposit is not a cost
    assert sorted((i["amount_gbp"], i["per"]) for i in c["contingent"]) == [
        (250, None), (350.0, "hour"), (600.0, None), (1500, None)]


def test_every_quote_is_the_packs_own_wording():
    for text in (LOT73_SC, HASLEMERE_SC, LOT6_SC):
        flat = _flat(text)
        c = pt.extract_buyer_costs(_sc(text))
        b = pt.extract_buying(_sc(text))
        for i in c["items"] + c["contingent"]:
            assert i["quote"] in flat
        for f in b["facts"].values():
            assert f["quote"] in flat


def test_standard_auction_conditions_are_not_read():
    cac = ("COMMON AUCTION CONDITIONS (Edition 4)\nG1. The buyer must pay the deposit. "
           "The buyer must pay £250 plus VAT if the contract is not completed on time.")
    assert pt.extract_buyer_costs(_sc(cac, "Lot_9_Common_auction_conditions_EW.pdf"))["items"] == []
    assert pt.extract_buyer_costs([{"file_name": "Lease.pdf", "doc_type": "lease",
                                    "extracted_text": "The Tenant shall pay £5,000 plus VAT"}])["documents_read"] == []


def test_number_words():
    assert pt.words_to_number("one thousand, four hundred and forty") == 1440
    assert pt.words_to_number("fifteen hundred") == 1500
    assert pt.words_to_number("two point seventy five") == 2.75
    assert pt.words_to_number("two and a half") == 2.5
    assert pt.words_to_number("ten pence") is None


# ── what you're buying ───────────────────────────────────────────────────────
def test_lot73_buying_facts():
    reg = {"file_name": "Register.pdf", "doc_type": "title_register",
           "extracted_text": "The Freehold land shown edged with red on the plan of the above Title filed at the Registry"}
    b = pt.extract_buying(_sc(LOT73_SC) + [reg])
    f = b["facts"]
    assert f["property"]["value"] == "Highfields Residential Care Home, Culver Street, Newent, Gloucestershire, GL18 1JA"
    assert f["title_number"]["value"] == "GR205903"
    assert f["title_guarantee"]["value"] == "Full title guarantee"
    assert f["tenure"]["value"] == "Freehold" and f["tenure"]["file_name"] == "Register.pdf"
    assert f["completion"]["value"].startswith("Completion will take place 20 working days")
    assert b["not_stated"] == ["possession"]          # Lot 73's special conditions are silent on it


def test_conflicting_registers_give_no_tenure():
    regs = [{"file_name": "FH.pdf", "doc_type": "title_register",
             "extracted_text": "The Freehold land shown edged red. The Leasehold land shown edged blue."}]
    assert "tenure" in pt.extract_buying(regs)["not_stated"]


def test_haslemere_and_lot6_buying_facts():
    h = pt.extract_buying(_sc(HASLEMERE_SC))["facts"]
    assert h["property"]["value"] == "9-11 Junction Place, Haslemere, GU27 1LE"
    assert h["tenure"]["value"] == "Freehold"
    assert h["possession"]["value"] == "vacant possession subject to a lease/tenancy"
    assert h["completion"]["value"] == "AGREED COMPLETION DATE 18 NOVEMBER 2026"
    s6 = pt.extract_buying(_sc(LOT6_SC))["facts"]
    assert s6["title_guarantee"]["value"] == "Limited title guarantee"
    assert s6["possession"]["value"] == "vacant possession"


# ── vacant possession ────────────────────────────────────────────────────────
def _vp(text):
    m = re.search(r"vacant\s+possession|currently\s+vacant", text, re.I)
    return pt.possession_statement(text, m.start(), m.end())


def test_vacant_possession_only_when_stated_about_the_sale():
    # Lease wording of the kind that marked 60B and 59A vacant on 4 Oct
    # (yield-up covenant; rent-review assumption) — reconstructed, those deals
    # were deleted; the clause types are as recorded in that session.
    assert _vp("At the end of the Term to return the Property to the Landlord with vacant possession") is None
    assert _vp("the Property is let as a whole with vacant possession by a willing landlord") is None
    assert _vp("the words ', but otherwise with vacant possession on completion' are deleted") is None
    assert _vp("The Property is sold with vacant possession.") == "vacant possession"
    assert _vp("The sale is with vacant possession but subject to a long lease of the ground floor "
               "commercial unit") == "vacant possession subject to a lease/tenancy"
    assert _vp("The property is currently vacant.") == "vacant possession"


def _no_llm(monkeypatch):
    monkeypatch.setattr(ce, "extract_via_llm", lambda documents, already_found, subject_address=None: ({}, {}, []))


def test_let_lot_rent_is_no_longer_withheld(monkeypatch):
    _no_llm(monkeypatch)
    docs = [{"file_name": "Lease.pdf", "doc_type": "lease",
             "text": "The Tenant shall pay a rent of £29,500 per annum. At the end of the Term to return "
                     "the Property with vacant possession."},
            {"file_name": "SC.pdf", "doc_type": "special_conditions",
             "text": "The Property is sold subject to and with the benefit of the Lease."}]
    out = ce.extract_commercial_fields(docs)
    assert out["fields"].get("passing_rent_pa") == 29500.0
    assert not any("vacant possession" in g and "contradict" in g for g in out["evidence_gaps"])


def test_genuine_vacant_sale_with_a_rent_is_still_flagged(monkeypatch):
    _no_llm(monkeypatch)
    docs = [{"file_name": "SC.pdf", "doc_type": "special_conditions", "text": "The Property is sold with vacant possession."},
            {"file_name": "Old tenancy.pdf", "doc_type": "unknown", "text": "rent of £12,000 per annum"}]
    out = ce.extract_commercial_fields(docs)
    assert "passing_rent_pa" not in out["fields"]
    assert any("contradict" in g for g in out["evidence_gaps"])


def test_partly_let_sale_keeps_rent_and_says_so(monkeypatch):
    _no_llm(monkeypatch)
    docs = [{"file_name": "SC.pdf", "doc_type": "special_conditions", "text": HASLEMERE_SC},
            {"file_name": "Lease.pdf", "doc_type": "lease", "text": "a rent of £9,000 per annum"}]
    out = ce.extract_commercial_fields(docs)
    assert out["fields"].get("passing_rent_pa") == 9000.0
    assert any("subject to a lease" in g for g in out["evidence_gaps"])


# ── routes read only the lot's own documents ─────────────────────────────────
class _Q:
    def __init__(self, rows): self.rows = rows
    def select(self, *a): return self
    def eq(self, *a): return self
    def execute(self):
        class R: pass
        r = R(); r.data = self.rows; return r


class _SB:
    def __init__(self, rows): self.rows = rows
    def table(self, name): return _Q(self.rows)


def test_lot_documents_drop_other_property_files():
    import commercial_routes as cr
    rows = [{"file_name": "Lot_6_Special_conditions.docx", "doc_type": "special_conditions", "extracted_text": LOT6_SC},
            {"file_name": "Lot_73_SC.pdf", "doc_type": "special_conditions", "extracted_text": LOT73_SC},
            {"file_name": "scan.pdf", "doc_type": "unknown", "extracted_text": ""}]
    sj = {"pack_integrity": {"excluded_files": [{"file_name": "Lot_6_Special_conditions.docx", "postcode": "DN21 2DD"}]}}
    docs, excl = cr._lot_documents(_SB(rows), "d", "u", sj)
    assert [d["file_name"] for d in docs] == ["Lot_73_SC.pdf"] and excl == ["Lot_6_Special_conditions.docx"]
    costs = pt.extract_buyer_costs(docs)
    assert costs["fixed_total_gbp_ex_vat"] == 1750.0                   # no Lot 6 sums leak in
    docs2, excl2 = cr._lot_documents(_SB(rows), "d", "u", {})          # old deal: nothing excluded
    assert len(docs2) == 2 and excl2 == []


def test_lot73_non_cost_clauses_add_no_certain_costs():
    from fixtures_special_conditions import LOT73_SC_OTHER_CLAUSES
    c = pt.extract_buyer_costs(_sc(LOT73_SC + LOT73_SC_OTHER_CLAUSES))
    assert c["fixed_total_gbp_ex_vat"] == 1750.0
    assert [i["basis"] for i in c["items"]] == ["fixed", "fixed", "fixed", "percent_of_price"]
