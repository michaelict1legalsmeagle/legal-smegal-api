"""
Guards for doc_classifier.classify_document (D-CLASSIFIER, 2026-09-23).

Fixtures are the opening text of the REAL Lot 12 pack (deal 3ca6f024…), with
personal names/addresses/refs redacted. Each case failed under the old app.py
classifier (shown in the comment) and must not regress.
Run: python3 -m pytest tests/test_doc_classifier.py -q   (from repo root)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from doc_classifier import classify_document as c

LOCAL_SEARCH = ("=== PAGE 1 === Prepared by OneSearch Direct No Roads, Footways, and Footpaths "
                "Maintained at Public Expense Roads Building Regulations Approval Planning Designations "
                "and Proposals Search Type: Land Charges Register and Local Search Enquiries "
                "Property: [REDACTED] Darlington Borough Council Town Hall Regulated Local Authority Search "
                "Land Charges Summary This search reveals 2 registration(s)")
DRAINAGE = ("=== PAGE 1 === Drainage and Water Search Property Address [REDACTED]. OneSearch. "
            "Records searched indicate Water undertaker: Northumbrian Water Water connection: Connected "
            "Caution - please refer to relevant question Land Registry Plans are Crown Copyright")
DEATH = ("=== PAGE 1 === D. Cert. S.R./R.B.D. CERTIFIED COPY Pursuant to the Births and CAUTION-Any "
         "person who falsifies any of the particulars OF AN ENTRY Deaths Registration Act 1953 DEATH "
         "Entry No. 17 Registration district [REDACTED] 1. Date and place of death Administrative area")
PROBATE = ("=== PAGE 1 === 0306/0000612 Grant of Probate High Court of Justice England and Wales "
           "Principal Registry of the Family Division HMCTS Probate [REDACTED] The Last Will and "
           "Testament of [REDACTED] (An official copy of which is available from the Court)")
REGISTER = ("=== PAGE 1 === The electronic official copy of the register follows this message. Please "
            "note that this is the only official copy we will issue. === PAGE 2 === Title number XX00000 "
            "A: Property Register ... B: Proprietorship Register Title absolute C: Charges Register")
PLAN = ("=== PAGE 1 === These are the notes referred to on the following official copy The electronic "
        "official copy of the title plan follows this message. Please note that this is the only "
        "official copy we will issue. This title is dealt with by the HM Land Registry, Durham Office.")
TA6 = ("=== PAGE 1 === TA6 Law Society TA6 (6th edition) Law Society Property Information Form (6th "
       "edition) (2025) 1. Property and seller details ... HM Land Registry title and/or title deeds")
EPC = ("=== PAGE 1 === Energy performance certificate (EPC) [REDACTED] Property type Total floor area "
       "Energy rating Valid until ... Rules on letting this property ... please")
SPECIAL = ("=== PAGE 1 === 1. Special Conditions of Sale – Lot [REDACTED] Auction Date. This Agreement "
           "incorporates the RICS Common Auction Conditions")


# ── live Lot 12 pack: (filename, text) -> expected   [old result] ──────────────
def test_local_search_by_text():            # was title_register
    assert c("Lot_12_07176229.pdf", LOCAL_SEARCH) == "local_auth_search"

def test_drainage_search_by_text():         # was title_register
    assert c("Lot_12_D03669859.pdf", DRAINAGE) == "environmental"

def test_death_certificate():               # was title_plan
    assert c("Lot_12_Certified_death_cert.pdf", DEATH) == "death_certificate"
    assert c("scan001.pdf", DEATH) == "death_certificate"          # text-only path

def test_probate():                         # was title_register
    assert c("Lot_12_Certified_Probate.pdf", PROBATE) == "probate"
    assert c("scan002.pdf", PROBATE) == "probate"

def test_title_plan_not_register():         # was title_register
    assert c("Lot_12_OC1-Title-Plan-DU50174.pdf", PLAN) == "title_plan"
    assert c("scan003.pdf", PLAN) == "title_plan"

def test_title_register():                  # correct before; must stay
    assert c("Lot_12_OC1-Register-DU50174.pdf", REGISTER) == "title_register"
    assert c("scan004.pdf", REGISTER) == "title_register"

def test_ta6_is_deed_bucket():              # was title_register
    assert c("Lot_12_TA6_-_Property_Information.pdf", TA6) == "deed"
    assert c("scan005.pdf", TA6) == "deed"

def test_epc():
    assert c("Lot_12_Energy_performance_certificate_EPC.pdf", EPC) == "epc"
    assert c("scan006.pdf", EPC) == "epc"

def test_special_conditions_by_text():
    assert c("Lot_12_Auction_Contract_-_15_Wolsingham_Terrace.pdf", SPECIAL) == "special_conditions"


# ── substring defects in the old classifier ───────────────────────────────────
def test_please_is_not_lease():
    assert c("scan.pdf", "Please note the following. Released on request.") == "unknown"

def test_ast_not_inside_words():
    assert c("Eastleigh_master_copy.pdf", "") == "unknown"
    assert c("Flat_2_AST.pdf", "") == "tenancy_ast"

def test_whitespace_split_heading():
    assert c("x.pdf", "Special\nConditions   of\n\nSale") == "special_conditions"

def test_filename_only_upload_path():       # OCR-bound uploads are typed on filename first
    assert c("Lot 4 Local Search.pdf", "") == "local_auth_search"
    assert c("EPC.pdf", "") == "epc"

def test_never_raises_and_unknown_default():
    assert c(None, None) == "unknown"
    assert c("", "") == "unknown"


# ── L4 audit regressions (live corpus, 2026-09-23) ────────────────────────────
def test_register_to_bid_is_not_title_register():
    assert c("Lot_11_REGISTER_TO_BID.pdf",
             "Register for remote bidding Before you start Please note you will require") == "unknown"

def test_epc_register_filename_is_epc():
    assert c("Lot_24_EPC-EPC_Register.pdf", "") == "epc"

def test_contracts_are_special_conditions():
    assert c("Lot_32_Sale_agreement.pdf", "AGREEMENT ... Property and Charges Register ...") == "special_conditions"
    assert c("Lot_22_AUCTION_Contract_-Auction_House.pdf", "") == "special_conditions"
    assert c("x.pdf", "CONTRACT (Incorporating the Common Auction Conditions (Fourth Edition) "
                      "... copy of the register") == "special_conditions"

def test_tr1_tr2_transfer_forms_are_deed():
    assert c("06._Transfer_of_Whole_printed.pdf", "") == "deed"
    assert c("scan.pdf", "HM Land Registry Transfer of whole of registered title(s) TR2") == "deed"

def test_coal_and_drainage_are_environmental():
    assert c("Lot_12_Coal_Search.pdf", "") == "environmental"
    assert c("CON29DW.pdf", "") == "environmental"
    assert c("Local_Search.pdf", "") == "local_auth_search"

def test_official_copy_lease_is_lease():
    assert c("Official_copy_lease_LA123.pdf", "") == "lease"


def test_rent_statements_are_typed():         # live Lot 34: was 'unknown'
    assert c("Lot_34_March_2026_-_Rent_Statement_-_Redacted.pdf", "") == "rent_statement"
    assert c("scan.pdf", "RE: 2C TALBOT ROAD NORTH Rents received for the Period: 15/03/2026-14/04/2026 £850.00") == "rent_statement"
