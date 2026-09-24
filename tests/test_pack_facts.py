"""
V-PACK guards (2026-09-24): subject facts read from the pack's own documents.
Fixtures are the stored text of the real Lot 34 pack (2C Talbot Road North),
deals ef885edd (23 Sep) and 7835cfa7 (24 Sep) — same PDFs, different text order.
Run: python3 -m pytest tests/test_pack_facts.py -q
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pack_facts import read_epc, read_register_tenures, read_rent_statements, resolve_pack_facts

ADDR, PC = "2c Talbot Road North, Wellingborough", "NN8 1SF"

EPC_23SEP = ("\n\n=== PAGE 1 ===\n\nEnergy performance certificate (EPC)\n2c, Talbot Road North\nWELLINGBOROUGH\n"
             "NN8 1SF\nEnergy rating\nB\nValid until:\n7 June 2028\nCertificate\nnumber: 0588-3022-7376-5288-5904\n"
             "Property type\nEnd-terrace house\nTotal floor area\n53 square metres\nRules on letting this property\n"
             "\n=== PAGE 6 ===\nAbout this assessment\nDate of assessment\n8 June 2018\n")
EPC_24SEP = ("\n\n=== PAGE 1 ===\n\nEnergy performance certificate (EPC)\n2c, Talbot Road North\nWELLINGBOROUGH\n"
             "NN8 1SF\nProperty type\nTotal floor area\nEnergy rating\nValid until:\n7 June 2028\nB\n"
             "Certificate 0588-3022-7376-5288-5904\nnumber:\nEnd-terrace house\n53 square metres\n"
             "Rules on letting this property\n")
STATEMENT = ("\n\n=== PAGE 1 ===\n\nInvoice No:\n68210\nInvoice Date:\nMarch 17, 2026\nVAT Reg. No:\nOur Ref:\n"
             "Your Ref:\n990 6841 80\n2091\nRE: 2C TALBOT ROAD NORTH, WELLINGBOROUGH\nRents received for the Period:\n"
             "15/03/2026-14/04/2026\n£850.00\nCommission on Collection\nVAT\n£60.00\n£12.00\nInland Revenue Annual Return\n")
REGISTER = ("A: Property Register\n1 (01.07.2021) The Freehold land shown edged with red on the plan of the above title filed"
            " at the Registry and being 2c Talbot Road North, Wellingborough (NN8 1SF).")


def test_epc_both_text_orders_give_the_same_facts():
    for t in (EPC_23SEP, EPC_24SEP):
        r = read_epc(t, PC, ADDR)
        assert r["type_code"] == "T" and r["property_type_label"] == "End-terrace house"
        assert r["floor_area_m2"] == 53.0
        assert r["certificate_number"] == "0588-3022-7376-5288-5904"
        assert r["address_match"] == "postcode+house_number"


def test_epc_for_another_address_is_rejected():
    assert read_epc(EPC_24SEP.replace("2c, Talbot", "4, Talbot"), PC, ADDR) is None
    assert read_epc(EPC_24SEP, "NN8 1SG", ADDR) is None


def test_register_tenure():
    assert read_register_tenures(REGISTER) == ["Freehold"]
    assert read_register_tenures("The Leasehold land shown") == ["Leasehold"]


def test_rent_statement_period_rent_and_commission():
    r = read_rent_statements(STATEMENT, ADDR)
    assert r == [{"period_start": "15/03/2026", "period_end": "14/04/2026", "rent_gbp": 850.0,
                  "commission_gbp": 60.0, "commission_vat_gbp": 12.0}]
    assert read_rent_statements(STATEMENT.replace("2C TALBOT", "9 TALBOT"), ADDR) == []


def test_resolver_end_to_end():
    docs = [
        {"file_name": "Lot_34_EPC.pdf", "doc_type": "epc", "extracted_text": EPC_24SEP, "extraction_status": "complete"},
        {"file_name": "Lot_34_Register_-_FH.pdf", "doc_type": "title_register", "extracted_text": REGISTER, "extraction_status": "complete"},
        {"file_name": "Lot_34_March_2026_-_Rent_Statement.pdf", "doc_type": "rent_statement", "extracted_text": STATEMENT, "extraction_status": "complete"},
        {"file_name": "Lot_34_Rent_Statements_-_redacted.pdf", "doc_type": "unknown", "extracted_text": None, "extraction_status": "empty"},
    ]
    f = resolve_pack_facts(docs, ADDR, PC)
    assert f["epc"]["type_code"] == "T" and f["epc"]["floor_area_m2"] == 53.0
    assert f["tenure"]["value"] == "Freehold"
    assert f["rent"]["monthly_rent_gbp"] == 850.0 and f["rent"]["months_evidenced"] == 1
    assert f["rent"]["agent_commission_pct_incl_vat"] == 8.5
    assert f["unread_documents"] == ["Lot_34_Rent_Statements_-_redacted.pdf"]


def test_conflicting_register_tenures_are_not_resolved():
    docs = [{"file_name": "a", "doc_type": "title_register", "extracted_text": "The Freehold land being 2c Talbot Road North", "extraction_status": "complete"},
            {"file_name": "b", "doc_type": "title_register", "extracted_text": "The Leasehold land being 2c Talbot Road North", "extraction_status": "complete"}]
    f = resolve_pack_facts(docs, ADDR, PC)
    assert f["tenure"] is None and f["tenure_ambiguous"]["values"] == ["Freehold", "Leasehold"]


def test_scottish_and_energy_report_area_formats():
    sc = ("Energy Performance Certificate (EPC)\n8 BRIMMOND PLACE, ABERDEEN, AB11 8EN\nDwelling type: Semi-detached house\n"
          "Date of certificate:\n30 January 2025\nTotal floor area:\n85 m2\nPrimary Energy Indicator:\n301 kWh/m2/year\n")
    r = read_epc(sc, "AB11 8EN", "8 Brimmond Place, Aberdeen")
    assert r["floor_area_m2"] == 85.0 and r["type_code"] == "S"
    assert read_epc("Energy performance certificate\nAB11 8EN 8 Brimmond\n301 kWh/m2/year", "AB11 8EN", "8 Brimmond Place")["floor_area_m2"] is None


def test_building_freehold_register_never_overrides_a_flat():   # live: 95b Woodside; Flat B, 18 Grosvenor Ave
    docs = [{"file_name": "Official_Copy_Register.pdf", "doc_type": "title_register",
             "extracted_text": "The Freehold land shown edged with red ... being 18 Grosvenor Avenue", "extraction_status": "complete"}]
    f = resolve_pack_facts(docs, "Flat B, 18 Grosvenor Avenue, Highbury", "N5 2NP")
    assert f["tenure"] is None and "building" in f["tenure_ambiguous"]["reason"]
    f2 = resolve_pack_facts(docs, "95b Woodside, London", "SW19 7BA", subject_is_flat=True)
    assert f2["tenure"] is None


def test_register_for_another_property_is_ignored():
    docs = [{"file_name": "r.pdf", "doc_type": "title_register",
             "extracted_text": "The Freehold land ... being 14 Talbot Road North", "extraction_status": "complete"}]
    assert resolve_pack_facts(docs, ADDR, PC)["tenure"] is None


def test_hmlr_stamp_only_text_routes_to_ocr():    # live: Lot 34 Transfer, 19 pages, 1,991 chars, all stamps
    from pack_facts import text_layer_is_unusable
    stamp = ("\n\n=== PAGE 1 ===\n\nThese are the notes referred to on the following official copy\nTitle Number NN359419\n"
             "The electronic official copy of the document follows this\nmessage.\nThis copy may not be the same size as the\n"
             "original.\nPlease note that this is the only official copy we will issue. We will not issue\na paper official copy.\n"
             + "".join(f"\n\n=== PAGE {i} ===\n\n This official copy is incomplete without the preceding notes page.\n" for i in range(2, 20)))
    assert text_layer_is_unusable(stamp, 19) is True
    assert text_layer_is_unusable("", 3) is True
    assert text_layer_is_unusable(EPC_24SEP, 1) is False
    assert text_layer_is_unusable(STATEMENT, 1) is False


def test_font_garbled_text_routes_to_ocr():
    from pack_facts import text_layer_is_unusable
    garbled = "\x02\x05\x11\x13\x1a" * 60 + " EPC "
    assert text_layer_is_unusable(garbled, 1) is True
    assert text_layer_is_unusable("Freehold land £850.00 – “quoted” • ok " * 10, 1) is False
