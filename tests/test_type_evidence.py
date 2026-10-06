"""TYPE-EVID-1 (2026-10-06): a residential strategy label (BTL/HMO/Flip/BRRR/SA)
is kept only when the pack states it — quoted in type_evidence and that quote
found in the pack text. Otherwise property.type is "Residential".

Live origin (L3, 6 Oct): deal 564488a4 (15 Park Vale Close) stored
property.type = deal_type = "Flip" with no quote; the prompt asked the model
for "what the buyer intends to do", which a legal pack never states. The
label was shown on the dashboard and Deal Report as if read from the pack.

Routing is NOT changed: commercial / mixed-use / unclassified handling and the
section votes are exactly as ROUTE-1..4 (the Commercial / Mixed Use prompt
wording restored word for word on 4 Oct is guarded below).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import asset_router as ar  # noqa: E402
import pack_reader as pr   # noqa: E402

PACK = {"title_register": "the property known as 15 park vale close semi detached house",
        "tenancy_ast": "the property is licensed as a house in multiple occupation hmo licence granted 2024"}


def _src(q):
    q = q.lower()
    return {dt for dt, t in PACK.items() if q in t}


RES_QUOTE = {"asset_class": "residential",
             "asset_class_evidence": "the property known as 15 Park Vale Close"}


# ── the live failure: a guessed strategy with no quote ───────────────────────
def test_guessed_flip_without_a_quote_becomes_residential():
    r = ar.resolve([{**RES_QUOTE, "type": "Flip"}], _src, physical_type="Semi-Detached")
    assert r["asset_class"] == "residential"
    assert r["strategy"] is None and r["strategy_evidence"] is None
    p = ar.apply_to_property({"type": "Flip"}, r)
    assert p["type"] == "Residential" and p["type_evidence"] is None


def test_quote_not_found_in_the_pack_is_not_evidence():
    r = ar.resolve([{**RES_QUOTE, "type": "Flip",
                     "type_evidence": "suitable for refurbishment and resale"}], _src)
    assert r["strategy"] is None
    assert ar.apply_to_property({}, r)["type"] == "Residential"


def test_stated_and_found_strategy_is_kept_with_its_quote():
    q = "hmo licence granted 2024"
    r = ar.resolve([{**RES_QUOTE, "type": "HMO", "type_evidence": q}], _src)
    assert r["strategy"] == "HMO" and r["strategy_evidence"] == q
    p = ar.apply_to_property({}, r)
    assert p["type"] == "HMO" and p["type_evidence"] == q


def test_physical_type_fallback_writes_residential_not_null():
    r = ar.resolve([], _src, physical_type="Terraced")
    assert r["reason"] == "no_vote_residential_physical_type"
    assert ar.apply_to_property({}, r)["type"] == "Residential"


# ── routing unchanged ────────────────────────────────────────────────────────
def test_unquoted_residential_type_still_votes_residential():
    # the section vote from a type label is unchanged (ROUTE-1 fallback)
    r = ar.resolve([{"type": "BTL"}], _src, physical_type="Other")
    assert r["asset_class"] == "residential" and r["reason"] == "sections_agree"
    assert ar.apply_to_property({}, r)["type"] == "Residential"


def test_commercial_mixed_unclassified_unchanged_and_carry_no_type_evidence():
    for cls, t in (("commercial", "Commercial"), ("mixed_use", "Mixed Use"),
                   ("unclassified", "Unclassified")):
        p = ar.apply_to_property({"type": "BTL", "type_evidence": "x"}, {"asset_class": cls})
        assert p["type"] == t and p["type_evidence"] is None


def test_residential_label_trips_no_commercial_gate():
    from services.ceiling_engine import COMMERCIAL_DIVERSION_KEYWORDS
    assert not any(kw in "residential" for kw in COMMERCIAL_DIVERSION_KEYWORDS)
    assert ar.normalise_strategy("Residential") is None
    assert ar.class_from_type("Residential") is None


# ── prompt ───────────────────────────────────────────────────────────────────
def test_prompt_no_longer_asks_what_the_buyer_intends():
    assert "what the buyer intends to do" not in pr.PACK_SYSTEM
    assert '"type_evidence": null' in pr.PACK_SYSTEM
    assert "never infer a strategy" in pr.PACK_SYSTEM


def test_commercial_and_mixed_use_wording_kept_word_for_word():
    # 26 Sep regression guard: losing this wording stopped all commercial routing.
    assert ("Use 'Mixed Use' if the title/lot contains BOTH a commercial element "
            "(retail/office/industrial/leisure unit) AND a residential element "
            "(flat(s) above a shop, etc) — do not force this into BTL/HMO/Commercial "
            "when both are genuinely present.") in pr.PACK_SYSTEM
    assert ("Use 'Commercial' for a purely non-residential unit (retail, office, "
            "industrial, warehouse, leisure) with no residential element.") in pr.PACK_SYSTEM


def test_pipeline_version_bumped_so_old_guesses_are_not_reused():
    assert pr.PIPELINE_VERSION == "fullread-4"


# ── end to end through analyse_pack with a stubbed model ─────────────────────
def test_analyse_pack_stores_residential_when_the_model_guesses_flip():
    docs = [{"file_name": "register.pdf", "doc_type": "title_register",
             "extracted_text": "The property known as 15 Park Vale Close, Halstead CO9 3DS. Semi-detached house."}]

    def llm(system, prompt):
        return {"flags": [], "property": {
            "postcode": "CO9 3DS", "type": "Flip", "type_evidence": None,
            "asset_class": "residential",
            "asset_class_evidence": "The property known as 15 Park Vale Close",
            "physical_type": "Semi-Detached"}}

    r = pr.analyse_pack(docs, llm)
    assert r["property"]["asset_class"] == "residential"
    assert r["property"]["type"] == "Residential"
    assert r["property"]["type_evidence"] is None
    assert r["pipeline_version"] == "fullread-4"


# ── user reclassify keeps only an evidenced strategy (app.py, code check) ────
def test_reclassify_endpoint_requires_type_evidence_to_keep_a_strategy():
    src = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
    i = src.index("keep_strategy = (_ar.normalise_strategy(prop.get(\"type\"))")
    block = src[i:i + 700]
    assert 'prop.get("type_evidence")' in block
    assert '"strategy_evidence"' in block
