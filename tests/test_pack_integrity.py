"""PACK-INTEG-1 (2026-10-04): documents about another property are excluded.

Texts below are the stored wording of live documents (deal 92f1d4c4 and the
party-address lines that a postcode-only rule wrongly caught in the 4 Oct
corpus run), cut to the lines that matter.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pack_integrity as pi
import pack_reader as pr

LOT6_SC = ("SPECIAL CONDITIONS\nThe property is known as 4 White Lion Yard, Gainsborough, DN21 2DD "
           "demised in the new long lease, a draft of which is in the legal pack.\n"
           "The completion date shall be 14 days from auction.\nThe buyer shall pay a £8,000 minimum deposit.")
LOT73_ENV = ("Site Address: Highfields Residential Home, Highfield House Nursing Home, Culver Street, "
             "Newent, GL18 1JA 25m scales Air Quality Index Generally Good\nRequest by: One Search, Cardiff CF10 4BZ")
LOT73_SC = ("SPECIAL CONDITIONS OF AUCTION SALE\nThe Property: Highfields Residential Care Home, Culver Street, "
            "Newent, Gloucestershire, GL18 1JA\nCompletion will take place 20 working days from the date of exchange of contracts.")
LOT73_REG = ("Title number GR205903\nA: Property Register\nThe Freehold land shown edged with red on the plan of the "
             "above Title filed at the Registry and being Highfields Residential Care Home, Culver Street, Newent "
             "(GL18 1JA).\nC: Charges Register\nProprietor: LLOYDS BANK PLC of 10 Fenchurch Avenue, London EC3M 5AG")
LOT73_LOCAL = "Local search\nCulver Street, Newent GL18 1JA\nForest of Dean District Council, Coleford GL16 8HG"
PNG_MAP = "BT Openreach map search. Plant records shown are indicative only."


def _d(name, text, dt="unknown"):
    return {"file_name": name, "doc_type": dt, "extracted_text": text}


LOT73_UPLOAD = [
    _d("Lot_6_Special_conditions.docx", LOT6_SC, "special_conditions"),
    _d("Lot_73_5._Enviromental.pdf", LOT73_ENV, "environmental"),
    _d("Lot_73_6._Local.pdf", LOT73_LOCAL, "local_auth_search"),
    _d("Lot_73_Official_Copy__Register__-_GR205903_redacted.pdf", LOT73_REG, "title_register"),
    _d("Lot_73_Special_conditions_of_auction_sale-_Culver_Street_redacted.pdf", LOT73_SC, "special_conditions"),
    _d("Lot_71_BT_Openreach_Map_Search8104623.1.png.pdf", PNG_MAP),
]


def test_lot73_upload_excludes_only_the_lot6_document():
    r = pi.check(LOT73_UPLOAD)
    assert r["status"] == "checked" and r["lot_postcode"] == "GL18 1JA"
    assert [x["file_name"] for x in r["excluded_files"]] == ["Lot_6_Special_conditions.docx"]
    x = r["excluded_files"][0]
    assert x["postcode"] == "DN21 2DD" and "White Lion Yard" in x["quote"]
    assert x["quote"] in LOT6_SC                       # the quote is the document's own wording
    assert r["named_for_other_lot"] == ["Lot_6_Special_conditions.docx",
                                        "Lot_71_BT_Openreach_Map_Search8104623.1.png.pdf"]


def test_party_addresses_are_not_property_addresses():
    # Lender / council / search-provider postcodes (all 65 false hits of a
    # postcode-only rule in the 4 Oct corpus run were of this kind).
    assert pi.property_postcodes(LOT73_REG) == [
        {"postcode": "GL18 1JA", "quote": pi.property_postcodes(LOT73_REG)[0]["quote"]}]
    assert pi.property_postcodes("PROPRIETOR: LLOYDS BANK PLC (Co. Regn. No. 2065) of 10 Fenchurch Avenue, London EC3M 5AG") == []
    assert pi.property_postcodes("TRAFFORD COUNCIL, TOWN HALL, TALBOT ROAD, STRETFORD, M32 0TH") == []


def test_lots_own_documents_with_only_other_postcodes_are_kept():
    # A register whose only postcode is the lender's, a gas certificate with the
    # engineer's address: no property postcode of their own -> never excluded.
    docs = [_d("SC.pdf", LOT73_SC), _d("Env.pdf", LOT73_ENV),
            _d("Register FH.pdf", "Proprietor: X LTD of 10 Fenchurch Avenue, London EC3M 5AG"),
            _d("Gas.pdf", "Engineer: 15 Gilbey Close, Wellingborough NN9 5YG")]
    r = pi.check(docs)
    assert r["lot_postcode"] == "GL18 1JA" and r["excluded_files"] == []


def test_same_district_neighbouring_postcode_is_kept():
    # Lot 34 (2C Talbot Road NN8 1SF): the estate transfer names NN8 1SG.
    docs = [_d("SC.pdf", "The Property: 2C Talbot Road, Wellingborough NN8 1SF"),
            _d("EPC.pdf", "Property address: 2C Talbot Road, Wellingborough, NN8 1SF"),
            _d("Transfer.pdf", "The Freehold land shown edged red on the plan being land at Ladywell Park, NN8 1SG")]
    assert pi.check(docs)["excluded_files"] == []


def test_portfolio_lot_linked_by_its_own_documents_is_kept():
    # Special conditions naming both properties link the second district.
    sc = ("The Property: 12 High Street, Grimsby DN31 1AA and 4 Low Road, Hull HU9 2PB\n"
          "Property known as 12 High Street, Grimsby DN31 1AA")
    docs = [_d("SC.pdf", sc), _d("EPC 12.pdf", "Property address: 12 High Street, Grimsby DN31 1AA"),
            _d("EPC 4.pdf", "Property address: 4 Low Road, Hull HU9 2PB")]
    r = pi.check(docs)
    assert r["lot_postcode"] == "DN31 1AA" and r["excluded_files"] == []


def test_tie_or_no_address_excludes_nothing():
    tie = [_d("A.pdf", "Property address: 1 A Street, Gainsborough DN21 2DD"),
           _d("B.pdf", "Property address: 2 B Street, Newent GL18 1JA")]
    r = pi.check(tie)
    assert r["status"] == "undetermined" and r["excluded_files"] == []
    r = pi.check([_d("A.pdf", PNG_MAP), _d("B.pdf", "Completion 20 working days.")])
    assert r["status"] == "no_property_address" and r["excluded_files"] == []


def test_scottish_document_numbering_is_not_a_lot_number():
    # Live deal 02707db6: "Lot__1_…", "Lot__2_…", "Lot__3_…" are document numbers.
    docs = [_d("Lot__1_Bundle.pdf", "x"), _d("Lot__2_Bundle.pdf", "y"), _d("Lot__3_Disp.pdf", "z")]
    assert pi.check(docs)["named_for_other_lot"] == []


def test_analyse_pack_never_sends_or_cites_the_excluded_document():
    sent = []

    def llm(system, prompt):
        sent.append(prompt)
        return {"flags": [
            {"severity": "critical", "title": "Seller may rescind",
             "evidence": "The completion date shall be 14 days from auction."},
            {"severity": "high", "title": "Lot 73 completion",
             "evidence": "Completion will take place 20 working days from the date of exchange of contracts."},
        ], "property": {}}

    r = pr.analyse_pack(LOT73_UPLOAD, llm)
    joined = "\n".join(sent)
    assert "White Lion Yard" not in joined and "Lot_6_Special_conditions.docx" not in joined
    titles = [f["title"] for f in r["flags"]]
    assert "Lot 73 completion" in titles
    assert "Seller may rescind" not in titles          # its quote exists only in the excluded document
    assert r["pack_integrity"]["excluded_files"][0]["file_name"] == "Lot_6_Special_conditions.docx"
    assert "excluded" not in r["pack_integrity"]       # indexes are internal
    assert r["read_coverage"]["documents_excluded_other_property"] == ["Lot_6_Special_conditions.docx"]
    assert r["read_coverage"]["documents_total"] == len(LOT73_UPLOAD) - 1
    assert r["pipeline_version"] == "fullread-3"


def test_upload_without_foreign_documents_is_unchanged():
    docs = [d for d in LOT73_UPLOAD if not d["file_name"].startswith("Lot_6_")]
    kept, r = pi.split(docs)
    assert kept == docs and r["excluded_files"] == []


def test_lot_document_whose_anchor_also_reaches_a_council_postcode_is_kept():
    # Lot 73 local search wording: "subject property only. If required ..."
    # is followed by the council's address. The document names the lot too.
    local = ("Search address: Highfields, Culver Street, Newent GL18 1JA\n"
             "The replies relate to the subject property only. If required, information relating to "
             "other properties can be supplied. Forest of Dean District Council, Coleford GL16 8HG")
    docs = [_d("SC.pdf", LOT73_SC), _d("Env.pdf", LOT73_ENV), _d("Local.pdf", local)]
    r = pi.check(docs)
    assert {x["postcode"] for x in pi.property_postcodes(local)} == {"GL18 1JA", "GL16 8HG"}
    assert r["excluded_files"] == []
