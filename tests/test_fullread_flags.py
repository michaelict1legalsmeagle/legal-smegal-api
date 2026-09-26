"""V-FULLREAD / V-FLAGS (2026-09-26): the reader sends every character; flags
are kept only with located evidence; score is computed; no placeholder flags."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import pack_reader as pr
import flag_evidence as fe

LONG = "\n".join(f"Paragraph {i}: the buyer shall comply with clause {i} of the transfer." for i in range(12000))
DOCS = [
    {"file_name": "Local Search.pdf", "doc_type": "local_auth_search", "extracted_text": LONG},
    {"file_name": "Special Conditions.pdf", "doc_type": "legal_pack",
     "extracted_text": "4. The Buyer shall pay the Seller's legal costs of £1,500 plus VAT on completion."},
    {"file_name": "scan-epc.pdf", "doc_type": "unknown", "extracted_text": ""},
]


def test_every_character_is_sent():
    secs, cov = pr.build_sections(DOCS, section_chars=50_000)
    assert cov["all_text_sent"] is True
    assert cov["chars_sent"] == cov["chars_extracted"] == sum(len(d["extracted_text"]) for d in DOCS)
    assert cov["documents_unreadable"] == ["scan-epc.pdf"]
    assert cov["sections"] > 1
    # special conditions go first
    assert secs[0]["text"].startswith("=== DOCUMENT: Special Conditions.pdf")
    # every stored character appears in some section
    joined = "".join(s["text"] for s in secs)
    for d in DOCS:
        for i in range(0, len(d["extracted_text"]), 997):
            assert d["extracted_text"][i:i + 40] in joined


def test_no_caps_left_in_live_paths():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "guest_routes.py"), encoding="utf-8").read()
    assert "HARD_CAP" not in src and "PER_DOC" not in src
    app = open(os.path.join(root, "app.py"), encoding="utf-8").read()
    i = app.index("def summarise_deal(")
    j = app.index("\n@app.route", i)
    body = app[i:j]
    assert "_HARD_CAP" not in body and "_PER_DOC" not in body
    assert "analyse_pack" in body
    assert "temperature=0.1" not in src


def _llm_factory(flags):
    def llm(system, prompt):
        assert "DOCUMENT INVENTORY" in prompt
        assert "scan-epc.pdf" in prompt and "UNREADABLE" in prompt
        return {"flags": list(flags), "property": {}, "completion_terms": {"completion_type": "working"},
                "special_conditions": {"special_conditions_present": True}}
    return llm


def test_verifier_keeps_located_drops_invented_and_false_missing():
    flags = [
        {"severity": "high", "title": "Seller legal costs", "evidence": "The Buyer shall pay the Seller's legal costs of £1,500 plus VAT"},
        {"severity": "critical", "title": "Invented clause", "evidence": "the buyer must pay a ransom of £9,999 to the seller"},
        {"severity": "missing", "title": "Missing Local Authority Search", "evidence": "Not in document inventory"},
        {"severity": "missing", "title": "Missing EPC", "evidence": "Not in document inventory"},
    ]
    r = pr.analyse_pack(DOCS, _llm_factory(flags), section_chars=50_000)
    titles = [f["title"] for f in r["flags"]]
    assert "Seller legal costs" in titles
    assert "Invented clause" not in titles
    assert "Missing Local Authority Search" not in titles      # a local search IS in the pack
    assert "Missing EPC" not in titles                          # scan-epc.pdf is present (unreadable)
    assert r["deal_score"] == 100 - 6
    assert r["flag_counts"] == {"critical": 0, "high": 1, "missing": 0, "note": 0}
    assert r["read_coverage"]["all_text_sent"] is True
    assert r["pipeline_version"] == pr.PIPELINE_VERSION


def test_empty_flag_list_is_valid_and_no_placeholder():
    r = pr.analyse_pack(DOCS, _llm_factory([]), section_chars=50_000)
    assert r["flags"] == []
    assert r["deal_score"] == 100
    assert "No issues were found" in r["viability_statement"]


def test_section_failure_fails_whole_analysis():
    calls = {"n": 0}
    def llm(system, prompt):
        calls["n"] += 1
        if "SECTION 2 OF" in prompt:
            raise RuntimeError("boom")
        return {"flags": []}
    with pytest.raises(pr.AnalysisIncomplete):
        pr.analyse_pack(DOCS, llm, section_chars=50_000)


def test_completion_type_only_when_stated():
    facts = pr.merge_facts([{"completion_terms": {"completion_type": "business"}}])
    assert facts["completion_terms"]["completion_type"] is None


def test_score_tariff_and_floor():
    many = [{"severity": "critical"}] * 20
    assert fe.compute_deal_score(many) == 0
    assert fe.compute_deal_score([{"severity": "missing"}, {"severity": "note"}]) == 95


def test_drainage_is_not_environmental():
    docs = [{"file_name": "CON29DW Drainage and Water Search.pdf", "doc_type": "unknown", "extracted_text": "drainage"}]
    ver = fe.verify_flags([{"severity": "missing", "title": "Missing environmental search"}], docs)
    assert len(ver["flags"]) == 1
    ver = fe.verify_flags([{"severity": "missing", "title": "Missing drainage search"}], docs)
    assert len(ver["flags"]) == 0


def test_storage_truncation_is_reported():
    docs = [{"file_name": "huge.pdf", "doc_type": "unknown",
             "extracted_text": "x" * 100 + "\n[LEGALSMEGAL: TEXT TRUNCATED AT STORAGE — DOCUMENT NOT READ IN FULL]"}]
    _, cov = pr.build_sections(docs)
    assert cov["documents_truncated_at_storage"] == ["huge.pdf"]
    assert cov["documents_read_in_full"] == 0
