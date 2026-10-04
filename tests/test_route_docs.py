"""
ROUTE-1 + DOCS-1 guards (2026-10-04).

DOCS-1  every document in the pack is read, or named as not read with the reason.
        Live faults: the upload route required b"%PDF" for EVERY file (1,224 stored
        documents, 0 .docx — 60B / Lot 8 / Lot 10 / Lot 6 special conditions lost);
        a file that never reached the server was invisible (Lot 6: 6 selected,
        3 stored, analysed as "3/3 docs").
ROUTE-1 deals go to the right pipeline. Live fault: since 26 Sep the pack reader
        prompt produced type null / "investment", never "Commercial", so 60B, 59A
        and Lot 6 were valued on residential comps.

Run: python3 -m pytest tests -q
"""
import ast
import io
import os
import re
import sys
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
APP = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()

import asset_router as ar           # noqa: E402
import pack_reader as pr            # noqa: E402


# ── helpers extracted from app.py (no app import / no network) ───────────────
def _app_fns(*names):
    from werkzeug.utils import secure_filename
    from typing import List, Optional
    tree = ast.parse(APP)
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    ns = {"io": io, "os": os, "secure_filename": secure_filename,
          "Optional": Optional, "List": List,
          "_UPLOAD_EXTS": (".pdf", ".docx", ".txt"), "_UPLOAD_MAX_BYTES": 20 * 1024 * 1024}
    for name in names:
        assert name in fns, f"{name} missing from app.py"
        exec(compile(ast.get_source_segment(APP, fns[name]), "app.py", "exec"), ns)
    return ns


def _docx_bytes(text="SPECIAL CONDITIONS OF SALE"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml",
                   f'<w:document xmlns:w="w"><w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>')
    return buf.getvalue()


# ── DOCS-1: upload validation ────────────────────────────────────────────────
def test_upload_route_no_longer_requires_pdf_for_every_file():
    body = APP[APP.index("def upload_document"):APP.index("def list_documents")]
    assert 'if not file_bytes.startswith(b"%PDF"):' not in body
    assert "_file_signature_error(filename, file_bytes)" in body


def test_file_signature_by_type():
    chk = _app_fns("_file_signature_error")["_file_signature_error"]
    assert chk("Lot_6_Special_conditions.docx", _docx_bytes()) is None
    assert chk("pack.pdf", b"%PDF-1.7 ...") is None
    assert chk("notes.txt", b"plain text") is None
    assert chk("fake.docx", b"%PDF-1.7") is not None          # not a zip
    assert chk("zip.docx", _zip_without_document()) is not None
    assert chk("fake.pdf", b"PK\x03\x04") is not None
    assert chk("bin.txt", b"\x00\x01\x02") is not None
    assert chk("photo.jpg", b"\xff\xd8\xff") is not None


def _zip_without_document():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("other.xml", "<x/>")
    return buf.getvalue()


REAL_DOCX = [
    "/home/claude/packs/lot60b/Lot_60B_Special conditions of sale.docx",
    "/home/claude/packs/lot6/Lot_6_Special conditions.docx",
]


@pytest.mark.parametrize("path", REAL_DOCX)
def test_real_pack_docx_passes_and_is_read(path):
    if not os.path.exists(path):
        pytest.skip("real pack not present on this machine")
    chk = _app_fns("_file_signature_error")["_file_signature_error"]
    data = open(path, "rb").read()
    assert chk(os.path.basename(path), data) is None


# ── DOCS-1: manifest gap ─────────────────────────────────────────────────────
def test_manifest_names_every_file_not_received_with_reason():
    ns = _app_fns("_manifest_entry", "_manifest_not_received")
    files = [
        {"name": "Lot_6_4 White Lion Yard - lease plan - plan 1.pdf", "size": 1322220},
        {"name": "Lot_6_DRAFT Lease 4 white lion.pdf", "size": 514363},
        {"name": "Lot_6_Special conditions.docx", "size": 30979},
        {"name": "Licence.pdf", "size": 22881723},
        {"name": "Big.pdf", "size": 22881723, "parts": ["Big (part 1 of 2).pdf", "Big (part 2 of 2).pdf"]},
        {"name": "photo.jpg", "size": 1000},
    ]
    m = {"files": [ns["_manifest_entry"](f) for f in files]}
    stored = {"Lot_6_4_White_Lion_Yard_-_lease_plan_-_plan_1.pdf", "Big_part_1_of_2.pdf"}
    gaps = {g["name"]: g["reason"] for g in ns["_manifest_not_received"](m, stored)}
    assert "Lot_6_4 White Lion Yard - lease plan - plan 1.pdf" not in gaps   # stored under its secure name
    assert gaps["Lot_6_DRAFT Lease 4 white lion.pdf"] == "upload did not reach the server"
    assert gaps["Lot_6_Special conditions.docx"] == "upload did not reach the server"
    assert gaps["Licence.pdf"] == "over the 20 MB upload limit"
    assert gaps["Big.pdf"] == "1 of 2 parts did not reach the server"
    assert gaps["photo.jpg"].startswith("file type not supported")
    # complete pack -> no gaps
    stored_all = stored | {"Lot_6_DRAFT_Lease_4_white_lion.pdf", "Lot_6_Special_conditions.docx",
                           "Licence.pdf", "Big_part_2_of_2.pdf", "photo.jpg"}
    assert ns["_manifest_not_received"](m, stored_all) == []


def test_summarise_refuses_incomplete_pack_until_user_accepts():
    body = APP[APP.index("def summarise_deal"):APP.index("def _run_and_store")]
    i_gap = body.index('"error": "pack_incomplete"')
    i_reuse = body.index("_reproducibility_gate(deal_id, deal.data)")
    assert i_gap < i_reuse, "manifest check must run before an earlier analysis is reused"
    assert "None if _not_received else _reproducibility_gate" in body
    assert '"extraction_status": "not_received"' in body


# ── DOCS-1: the analysis names files not received ────────────────────────────
def _llm_factory(props_by_section):
    calls = {"i": 0}

    def llm(system, prompt):
        i = calls["i"]
        calls["i"] += 1
        return {"flags": [], "property": dict(props_by_section[min(i, len(props_by_section) - 1)]),
                "completion_terms": {}, "special_conditions": {}}
    return llm


DOCS = [
    {"file_name": "lease.pdf", "doc_type": "lease",
     "extracted_text": "THIS LEASE of the retail unit known as Unit 3 is granted to SIM Motorsport Ltd. " * 20},
    {"file_name": "Lot_6_Special conditions.docx", "doc_type": "special_conditions",
     "extracted_text": "", "extraction_status": "not_received",
     "not_read_reason": "upload did not reach the server"},
]


def test_not_received_file_is_named_not_counted_as_read():
    r = pr.analyse_pack(DOCS, _llm_factory([{"type": "Commercial"}]), section_chars=50_000)
    cov = r["read_coverage"]
    assert cov["documents_total"] == 2 and cov["documents_read_in_full"] == 1
    assert cov["documents_not_received"] == ["Lot_6_Special conditions.docx"]
    assert "Lot_6_Special conditions.docx (upload did not reach the server)" in r["viability_statement"]
    inv = pr.inventory_block(DOCS)
    assert "IN THE PACK BUT NOT RECEIVED (upload did not reach the server)" in inv


# ── ROUTE-1: prompt + version ────────────────────────────────────────────────
def test_prompt_restores_strategy_list_and_asks_asset_class():
    assert "type = investment strategy only if the pack states it (else null)" not in pr.PACK_SYSTEM
    assert "BTL/HMO/Flip/BRRR/SA/Commercial/Mixed Use/Other" in pr.PACK_SYSTEM
    assert '"asset_class": null, "asset_class_evidence": null' in pr.PACK_SYSTEM
    assert "tenure: as registered." in pr.PACK_SYSTEM          # residential tenure unchanged
    assert pr.PIPELINE_VERSION != "fullread-1"                  # old results never reused


# ── ROUTE-1: routing decisions on the live cases ─────────────────────────────
PACK = ("Unit 3 The Boathouse retail unit let to SIM Motorsport Ltd under a lease "
        "within Part II of the Landlord and Tenant Act 1954 the dwellinghouse known as Flat 2")
_words = " " + " ".join(re.findall(r"[a-z0-9]+", PACK.lower())) + " "


def _found(q):
    hit = " " + " ".join(re.findall(r"[a-z0-9]+", q.lower())) + " " in _words
    return {"lease"} if hit else set()


@pytest.mark.parametrize("sections,phys,expected", [
    # 60B / 59A as stored today: type "investment", physical "Other" -> never residential
    ([{"type": "investment"}], "Other", "unclassified"),
    # no class stated anywhere and not a house/flat -> ask
    ([{"type": None}], "Other", "unclassified"),
    # restored list: Commercial
    ([{"type": "Commercial"}, {"type": None}], "Other", "commercial"),
    # explicit class with a quote found in the pack
    ([{"asset_class": "commercial", "asset_class_evidence": "retail unit let to SIM Motorsport Ltd"}], "Other", "commercial"),
    # invented quote is ignored
    ([{"asset_class": "commercial", "asset_class_evidence": "a quote that is not in the pack"}], "Terraced", "residential"),
    # Lot 10 / 2C Talbot Road: house/flat with type null or "investment" stays residential
    ([{"type": None}], "Flat", "residential"),
    ([{"type": "investment"}], "Terraced", "residential"),
    ([{"type": "BTL"}], "Terraced", "residential"),
    # Mixed Use
    ([{"type": "Mixed Use"}], "Other", "mixed_use"),
    # sections disagree -> ask
    ([{"type": "BTL"}, {"type": "Commercial"}], "Terraced", "unclassified"),
])
def test_routing(sections, phys, expected):
    assert ar.resolve(sections, _found, physical_type=phys)["asset_class"] == expected


# Lot 6 live (deal 37d2c006, 4 Oct): the lot's own draft lease vs a 1954 conveyance
LOT6_DOCS = {
    "lease": " ".join(re.findall(r"[a-z0-9]+", "Permitted Use: as a single private dwelling.".lower())),
    "deed":  " ".join(re.findall(r"[a-z0-9]+", ("ALL THAT shop offices and disused dwellinghouse with the yard "
                                               "and outbuildings thereto adjoining").lower())),
}


def _lot6_source(q):
    qw = " " + " ".join(re.findall(r"[a-z0-9]+", q.lower())) + " "
    return {dt for dt, w in LOT6_DOCS.items() if qw in " " + w + " "}


def test_quoted_wording_outranks_bare_type_label():
    # Lot 65A live (deal a857c247): 3 sections quoted commercial wording, 1 bare "Mixed Use"
    q = {"Property type Retail/Financial and Professional Services": {"epc"}}
    sections = [{"asset_class": "commercial", "asset_class_evidence": "Property type Retail/Financial and Professional Services"},
                {"type": "Mixed Use"}, {"type": "Commercial"}]
    r = ar.resolve(sections, lambda x: q.get(x, set()), physical_type="Other")
    assert r["asset_class"] == "commercial" and r["reason"] == "pack_wording_agrees"


def test_conflicting_quoted_wording_still_asks():
    # Lot 6 live (deal 37d2c006): lease "single private dwelling" vs deeds "shop offices";
    # marketed as Commercial Property — the pack does not settle it.
    sections = [
        {"asset_class": "residential", "asset_class_evidence": "Permitted Use: as a single private dwelling."},
        {"asset_class": "commercial", "asset_class_evidence": "ALL THAT shop offices and disused dwellinghouse"},
    ]
    assert ar.resolve(sections, _lot6_source, physical_type="Flat")["asset_class"] == "unclassified"


def test_property_type_carries_the_class_every_gate_reads():
    p = ar.apply_to_property({"type": "investment"}, {"asset_class": "commercial"})
    assert p["type"] == "Commercial" and p["asset_class"] == "commercial"
    p = ar.apply_to_property({"type": "investment"}, {"asset_class": "residential"})
    assert p["type"] is None                                   # -> BTL fallback, as a null type today
    p = ar.apply_to_property({"type": "HMO"}, {"asset_class": "residential", "strategy": "HMO"})
    assert p["type"] == "HMO"
    p = ar.apply_to_property({}, {"asset_class": "unclassified"})
    assert p["type"] == "Unclassified"


def test_unclassified_is_gated_by_the_engine_and_not_by_the_verdict_keywords():
    from services.ceiling_engine import COMMERCIAL_DIVERSION_KEYWORDS
    assert "unclassified" in COMMERCIAL_DIVERSION_KEYWORDS
    verdict = open(os.path.join(ROOT, "..", "fe", "legalsmegal-verdict.html"), encoding="utf-8").read() \
        if os.path.exists(os.path.join(ROOT, "..", "fe", "legalsmegal-verdict.html")) else None
    if verdict is None:
        pytest.skip("frontend repo not alongside")
    kw = verdict[verdict.index("var _COMM_DIVERSION_KEYWORDS"):]
    kw = kw[:kw.index("];")]
    assert "unclassified" not in kw
    assert "legalsmegal-classify.html" in verdict


def test_pack_reader_routes_end_to_end():
    r = pr.analyse_pack(DOCS[:1], _llm_factory([{"type": "investment", "physical_type": "Other"}]),
                        section_chars=50_000)
    assert r["property"]["asset_class"] == "unclassified"
    assert r["property"]["type"] == "Unclassified"
    r = pr.analyse_pack(DOCS[:1], _llm_factory([{"type": "Commercial", "physical_type": "Other"}]),
                        section_chars=50_000)
    assert r["property"]["asset_class"] == "commercial" and r["property"]["type"] == "Commercial"
    assert r["asset_routing"]["reason"] == "sections_agree"

# ── DOCS-2: legacy Word .doc (Lot 71 "Replies to CPSE 2") ────────────────────
# A 9.7 KB .doc made with LibreOffice from three lines of text.
_DOC_FIXTURE_B64 = "0M8R4KGxGuEAAAAAAAAAAAAAAAAAAAAAOwADAP7/CQAGAAAAAAAAAAAAAAABAAAAEAAAAAAAAAAAEAAAAgAAAAEAAAD+////AAAAAAAAAAD////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////9//////////7///8EAAAABQAAAAYAAAAHAAAACAAAAAkAAAAKAAAACwAAAAwAAAANAAAADgAAAA8AAAD+////EQAAAP7//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////1IAbwBvAHQAIABFAG4AdAByAHkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWAAUA////////////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/v///wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD///////////////8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD+////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP///////////////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP7///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA////////////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/v///wAAAAAAAAAAAQAAAP7////+////BAAAAAUAAAAGAAAABwAAAAgAAAAJAAAACgAAAAsAAAAMAAAADQAAAA4AAAAPAAAAEAAAABEAAAASAAAAEwAAABQAAAAVAAAAFgAAABcAAAAYAAAAGQAAABoAAAAbAAAAHAAAAB0AAAAeAAAAHwAAACAAAAAhAAAAIgAAAP7///8kAAAAJQAAAP7///8nAAAAKAAAACkAAAAqAAAAKwAAACwAAAAtAAAALgAAAC8AAAAwAAAAMQAAADIAAAAzAAAANAAAADUAAAA2AAAANwAAADgAAAA5AAAAOgAAADsAAAA8AAAAPQAAAD4AAAA/AAAAQAAAAEEAAABCAAAAQwAAAEQAAABFAAAARgAAAEcAAABIAAAASQAAAEoAAABLAAAATAAAAE0AAABOAAAATwAAAFAAAABRAAAAUgAAAFMAAABUAAAAVQAAAFYAAABXAAAAWAAAAFkAAABaAAAAWwAAAFwAAABdAAAAXgAAAP7///9gAAAA/v////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////8BAP7/AwoAAP////8GCQIAAAAAAMAAAAAAAABGGAAAAE1pY3Jvc29mdCBXb3JkLURva3VtZW50AAoAAABNU1dvcmREb2MAEAAAAFdvcmQuRG9jdW1lbnQuOAD0ObJxAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAASABUACgABAFsADwAAAAAAAAAAAFoAABDx/wIAWgAAAAYATgBvAHIAbQBhAGwAAAALAAAAMSQAKiQBQSQAAC8AQioAT0oDAFFKAwBDShgAbUgJBHNICQRQSgQAbkgECHRIBAheSgUAYUoYAF9IOQQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEYA/h8BAAIBRgAAAAcASABlAGEAZABpAG4AZwAAAA0ADwATpPAAFKR4AAYkAQAYAE9KBgBRSgYAQ0ocAFBKBwBeSgUAYUocADQAQhABAAIBNAAAAAkAQgBvAGQAeQAgAFQAZQB4AHQAAAAQABAAEmQUAQEAE6QAABSkjAAAACAALxABARIBIAAAAAQATABpAHMAdAAAAAIAEQAEAF5KCABAACIQAQAiAUAAAAAHAEMAYQBwAHQAaQBvAG4AAAANABIAE6R4ABSkeAAMJAEAEgBDShgANggBXkoIAGFKGABdCAEmAP4fAQAyASYAAAAFAEkAbgBkAGUAeAAAAAUAEwAMJAEABABeSggAVgD+HwEAQgFWAAAAEQBQAHIAZQBmAG8AcgBtAGEAdAB0AGUAZAAgAFQAZQB4AHQAAAAKABQA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AAAAAAAIAADqCAAADgBDYW9sYW44MAAAAAAAAAAAAAAAAAAAAAAAAAkEFgAvDgAAAAAAAAAAAAB1AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAP//DwAFAAAAAQAAAP//DwAGAAAAAQAAAP//DwAAAAAAAAAAAAAAAAAAAAAAiAAAAAAA7gEAAAAAAADuAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADuAQAAFAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAgAADAAAAA4CAAAMAAAAAAAAAAAAAAA7AgAAMgMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAG0FAABiAgAAAAAAAAAAAAAmAgAAFQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaAgAADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACANkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAUwBQAEUAQwBJAEEATAAgAEMATwBOAEQASQBUAEkATwBOAFMAIABPAEYAIABTAEEATABFAA0AVABoAGUAIABCAHUAeQBlAHIAIABzAGgAYQBsAGwAIABwAGEAeQAgAGEAIABjAG8AbgB0AHIAaQBiAHUAdABpAG8AbgAgAG8AZgAgAHQAaAByAGUAZQAgAHQAaABvAHUAcwBhAG4AZAAgAHAAbwB1AG4AZABzAC4ADQBQAHIAZQBtAGkAdQBtACAAcABhAHkAYQBiAGwAZQAgAG8AbgAgAGUAeABjAGgAYQBuAGcAZQAuAA0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAA6AgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAIAAA2CAAAsAgAAOoIAAD1AAAAAAAAAAAAAAAA9QAAAAAAAAAAAAAAAPUAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKFAADJABhJAATpAAAFKQAAEEkAAADLQA8MABAUAAAQlACAB+wgi4gsMZBIbBuBCKwbgQjkG4EJJBu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"


def test_doc_signature_and_reader():
    import base64, struct, re as _re
    data = base64.b64decode(_DOC_FIXTURE_B64)
    tree = ast.parse(APP)
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    ns = {"io": io, "re": _re, "struct": struct, "Tuple": tuple}
    exec(compile(ast.get_source_segment(APP, fns["_extract_doc_text"]), "app.py", "exec"), ns)
    text, pages = ns["_extract_doc_text"](data)
    assert "SPECIAL CONDITIONS OF SALE" in text
    assert "contribution of three thousand pounds" in text
    chk = _app_fns("_file_signature_error")["_file_signature_error"]
    assert chk("replies.doc", data) is None
    assert chk("fake.doc", b"%PDF-1.4") is not None


def test_upload_route_accepts_doc_and_reads_it():
    body = APP[APP.index("def upload_document"):APP.index("def list_documents")]
    assert '".pdf", ".docx", ".doc", ".txt"' in body
    assert "_extract_doc_text(file_bytes)" in body
    assert "olefile" in open(os.path.join(ROOT, "requirements.txt")).read()


REAL_DOC = "/home/claude/packs/lot71/Lot_71_Replies to CPSE 2 - 15-27 Red Street Carmarthen(117264681) (1)(8107625.1).doc"


def test_real_pack_doc_is_read():
    if not os.path.exists(REAL_DOC):
        pytest.skip("real pack not present on this machine")
    import struct, re as _re
    tree = ast.parse(APP)
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    ns = {"io": io, "re": _re, "struct": struct, "Tuple": tuple}
    exec(compile(ast.get_source_segment(APP, fns["_extract_doc_text"]), "app.py", "exec"), ns)
    text, _ = ns["_extract_doc_text"](open(REAL_DOC, "rb").read())
    assert "Commercial Property Standard Enquiries" in text
    assert "15-27 Red Street, Carmarthen" in text
    assert "None so far as the Seller is aware" in text


# ── DOCS-3: replies to enquiries typed, and 'missing' flags checked against them ──
def test_enquiries_are_typed_not_lease():
    import doc_classifier as dc
    assert dc.classify_document("Lot_71_Replies to CPSE 2 - 15-27 Red Street.doc",
                                "1954 Act: means the Landlord and Tenant Act 1954 lease") == "enquiries"
    assert dc.classify_document("Lot_65A_CPSE.7.pdf", "Commercial Property Standard Enquiries") == "enquiries"
    assert dc.classify_document("Lot_7_Replies_to_req.pdf",
                                "Replies to Requisitions on Title (OYEZ 28B) lease") == "enquiries"
    # unchanged neighbours
    assert dc.classify_document("Lot_12_Special_Conditions_.pdf",
                                "replies to enquiries, requisitions") == "special_conditions"
    assert dc.classify_document("Lot_6_DRAFT Lease 4 white lion.pdf", "") == "lease"


def test_missing_flags_cleared_by_documents_that_were_read():
    import flag_evidence as fe
    groundsure = {"file_name": "Lot_65A_Groundsure_Screening.pdf", "doc_type": "environmental",
                  "extracted_text": ("Summary " * 600) + " Flood risk: no flood risks of significant concern. "
                                    "Coal mining: none identified."}
    cpse = {"file_name": "Lot_65A_CPSE.7.pdf", "doc_type": "enquiries",
            "extracted_text": "Commercial Property Standard Enquiries CPSE.7"}
    kinds = fe.inventory_kinds([groundsure, cpse])
    assert {"flood", "mining", "environmental", "enquiries"} <= kinds
    assert fe.missing_flag_kind({"title": "No flood risk search in document inventory"}) == "flood"
    assert fe.missing_flag_kind({"title": "CPSE replies not readable — seller enquiries unverified"}) == "enquiries"
    assert fe.missing_flag_kind({"title": "No coal or mining search in document inventory"}) == "mining"
    # chancel is genuinely absent from the screening -> no kind -> flag kept
    assert "chancel" not in kinds


def test_special_conditions_naming_searches_do_not_clear_missing_search_flags():
    import flag_evidence as fe
    sc = {"file_name": "Lot_59A_Special Conditions of Sale.pdf", "doc_type": "special_conditions",
          "extracted_text": "On completion the Buyer shall pay the Seller the cost of any local search, "
                            "drainage search, coal mining search or any other search; flood"}
    k = fe.doc_kinds(sc)
    assert not ({"local_search", "drainage", "mining", "flood", "environmental"} & k)
    assert "special_conditions" in k
    # a real search document still counts
    assert "local_search" in fe.doc_kinds({"file_name": "Lot_65A_Somerset Council Local Search (CON29R).pdf",
                                           "doc_type": "local_auth_search", "extracted_text": ""})


def test_lot_documents_showing_both_parts_are_mixed_use():
    # Lot 72 live (deal 2d41f2d4): votes as stored — 3 mixed_use, 1 commercial,
    # 1 residential from lot documents, 1 residential from title history.
    q = {"MU": {"special_conditions"}, "SHOP": {"lease"}, "FLAT": {"tenancy_ast"}, "DEED": {"deed"}}
    sections = [
        {"asset_class": "mixed_use", "asset_class_evidence": "MU"},
        {"asset_class": "commercial", "asset_class_evidence": "SHOP"},
        {"asset_class": "mixed_use", "asset_class_evidence": "MU"},
        {"asset_class": "residential", "asset_class_evidence": "FLAT"},
        {"asset_class": "residential", "asset_class_evidence": "DEED"},
        {"asset_class": "mixed_use", "asset_class_evidence": "MU"},
    ]
    r = ar.resolve(sections, lambda x: q.get(x, set()), physical_type="Other")
    assert r["asset_class"] == "mixed_use" and r["reason"] == "lot_parts_mixed"
    # shop lease + flat tenancy alone (no explicit mixed_use vote) -> mixed use
    r = ar.resolve(sections[1:2] + sections[3:4], lambda x: q.get(x, set()), physical_type="Other")
    assert r["asset_class"] == "mixed_use"
    # Lot 6 still asks: commercial wording only in title-history deeds
    assert ar.resolve([
        {"asset_class": "residential", "asset_class_evidence": "Permitted Use: as a single private dwelling."},
        {"asset_class": "commercial", "asset_class_evidence": "ALL THAT shop offices and disused dwellinghouse"},
    ], _lot6_source, physical_type="Flat")["asset_class"] == "unclassified"
