"""
asset_router.py — ROUTE-1 (2026-10-04)

Decides which pipeline a deal belongs to: residential, commercial, mixed use,
or unclassified (the user is asked). Pure functions, no I/O — unit-tested in
tests/test_asset_router.py.

WHY (evidence, 4 Oct 2026):
  * pack_reader's V-FULLREAD prompt (26 Sep) told the model
    "type = investment strategy only if the pack states it (else null)".
    It had no allowed list, so since 26 Sep the model returned null (4 deals)
    or the word "investment" (3 deals) and never "Commercial" / "Mixed Use".
    The commercial gate (ceiling_engine COMMERCIAL_DIVERSION_KEYWORDS) and the
    Verdict redirect match on those words, so every commercial upload since
    26 Sep was valued on residential comps (60B £263,000; 59A £100,968).
  * Before 26 Sep the enum prompt produced BTL x107, Commercial x2,
    Mixed Use x1, HMO x1, Flip/BRRR x1 — no observed misroute.

RULE (fail closed):
  Each analysed section votes once:
    1. its explicit `asset_class` if it is one of the allowed values AND its
       `asset_class_evidence` quote is found in the pack text, else
    2. the class implied by its `type` if `type` is one of the allowed
       strategy values (the pre-26-Sep classifier, restored).
  * every vote agrees            -> that class
  * votes disagree               -> votes that rest only on TITLE-HISTORY documents
    (old conveyances, assents, epitomes, registers, plans, searches) are set
    aside; if the remaining LOT votes (special conditions, the lease/transfer
    being granted, tenancies, EPC, survey) agree -> that class.
    Live case (Lot 6, deal 37d2c006, 4 Oct): the draft lease of the lot says
    "Permitted Use: as a single private dwelling"; the 1954 conveyance of the
    whole of 41-43 Lord Street says "shop offices and disused dwellinghouse".
    The lot is the dwelling — the conveyance describes the superior title.
  * still disagree               -> "unclassified" (user is asked)
  * no votes, physical_type is a house/flat type -> "residential"
    (preserves today's behaviour for residential packs whose type is null)
  * no votes otherwise           -> "unclassified" (user is asked)

Nothing here values anything. "unclassified" is matched by ceiling_engine's
COMMERCIAL_DIVERSION_KEYWORDS so no residential valuation runs for it.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

ASSET_CLASSES = ("residential", "commercial", "mixed_use")
UNCLASSIFIED = "unclassified"

# Strategy labels the platform already uses (stored deal_type values + the
# pre-26-Sep prompt's list). Matching is case-insensitive.
_RESIDENTIAL_STRATEGIES = {
    "btl": "BTL", "hmo": "HMO", "flip": "Flip", "brrr": "BRRR",
    "flip/brrr": "Flip/BRRR", "sa": "SA", "serviced accommodation": "SA",
}
_COMMERCIAL_TYPE = {"commercial": "Commercial"}
_MIXED_TYPE = {"mixed use": "Mixed Use", "mixed-use": "Mixed Use"}

RESIDENTIAL_PHYSICAL_TYPES = {"flat", "detached", "semi-detached", "terraced"}

# Documents that record the history / context of the title rather than the lot
# being sold. Their description of "the property" is often the superior or
# whole building (Lot 6: a 1954 conveyance of shops and a dwellinghouse).
TITLE_HISTORY_DOC_TYPES = {"deed", "title_register", "freehold", "title_plan",
                           "local_auth_search"}

# property.type written for each final class (the signal every existing gate reads)
TYPE_FOR_CLASS = {"commercial": "Commercial", "mixed_use": "Mixed Use",
                  UNCLASSIFIED: "Unclassified"}

LABEL = {"residential": "Residential", "commercial": "Commercial",
         "mixed_use": "Mixed use", UNCLASSIFIED: "Not yet classified"}


def normalise_asset_class(value) -> Optional[str]:
    v = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return v if v in ASSET_CLASSES else None


def normalise_strategy(value) -> Optional[str]:
    """Canonical strategy label, or None for anything outside the allowed list
    (e.g. the free-text 'investment' the 26-Sep prompt produced)."""
    v = str(value or "").strip().lower()
    if not v:
        return None
    if v in _RESIDENTIAL_STRATEGIES:
        return _RESIDENTIAL_STRATEGIES[v]
    if v in _COMMERCIAL_TYPE:
        return _COMMERCIAL_TYPE[v]
    if v in _MIXED_TYPE:
        return _MIXED_TYPE[v]
    return None


def class_from_type(value) -> Optional[str]:
    s = normalise_strategy(value)
    if s is None:
        return None
    if s == "Commercial":
        return "commercial"
    if s == "Mixed Use":
        return "mixed_use"
    return "residential"


def _role(doc_types) -> str:
    dts = {d for d in (doc_types or []) if d}
    return "title_history" if dts and dts <= TITLE_HISTORY_DOC_TYPES else "lot"


def section_vote(prop: Dict, quote_source: Callable[[str], set],
                 section_doc_types=None) -> Optional[Dict]:
    """One section's vote: {'class', 'basis', 'evidence', 'role'} or None.
    quote_source(q) -> set of doc_types whose text contains q (empty = not found)."""
    prop = prop or {}
    ac = normalise_asset_class(prop.get("asset_class"))
    ev = (prop.get("asset_class_evidence") or "").strip()
    if ac and ev:
        found_in = quote_source(ev) or set()
        if found_in:
            return {"class": ac, "basis": "evidence_quote", "evidence": ev,
                    "role": _role(found_in)}
    tc = class_from_type(prop.get("type"))
    if tc:
        return {"class": tc, "basis": "type", "evidence": None,
                "type": normalise_strategy(prop.get("type")),
                "role": _role(section_doc_types)}
    return None


def resolve(section_props: List[Dict], quote_source: Callable[[str], set],
            physical_type: Optional[str] = None,
            section_doc_types: Optional[List] = None) -> Dict:
    """Decide the deal's class from every section's property block.
    section_doc_types[i] = doc_types in section i (for votes without a quote)."""
    sdt = list(section_doc_types or [])
    votes = [v for v in (section_vote(p, quote_source, sdt[i] if i < len(sdt) else None)
                         for i, p in enumerate(section_props or [])) if v]
    classes = sorted({v["class"] for v in votes})
    lot_classes = sorted({v["class"] for v in votes if v.get("role") == "lot"})
    evidence = [v["evidence"] for v in votes if v.get("evidence") and v.get("role") == "lot"] \
        + [v["evidence"] for v in votes if v.get("evidence") and v.get("role") != "lot"]
    ev_classes = sorted({v["class"] for v in votes if v["basis"] == "evidence_quote"})
    if len(classes) == 1:
        cls, reason = classes[0], "sections_agree"
    elif len(ev_classes) == 1:
        # ROUTE-3 (4 Oct, Lot 65A deal a857c247): quoted pack wording outranks an
        # unquoted type label. Three sections quoted commercial wording (EPC
        # "Retail/Financial and Professional Services"); one section's bare
        # type said "Mixed Use" with no quote -> the user was asked needlessly.
        cls, reason = ev_classes[0], "pack_wording_agrees"
    elif len(classes) > 1:
        # Quoted wording itself conflicts (Lot 6: the lease to be granted says
        # "single private dwelling", the title deeds say "shop offices") — the
        # pack does not settle it, so the user is asked.
        cls, reason = UNCLASSIFIED, "sections_disagree"
    elif str(physical_type or "").strip().lower() in RESIDENTIAL_PHYSICAL_TYPES:
        cls, reason = "residential", "no_vote_residential_physical_type"
    else:
        cls, reason = UNCLASSIFIED, "no_vote"
    strategy = None
    if cls == "residential":
        for v in votes:
            if v["class"] == "residential" and v.get("type"):
                strategy = v["type"]
                break
    return {
        "asset_class": cls,
        "reason": reason,
        "votes": [{"class": v["class"], "basis": v["basis"], "role": v.get("role")} for v in votes],
        "classes_voted": classes,
        "evidence": evidence[:3],
        "strategy": strategy,
    }


def apply_to_property(prop: Dict, resolution: Dict, source: str = "pack") -> Dict:
    """Write the decision onto the property block. property.type becomes the
    signal every existing gate reads: Commercial / Mixed Use / Unclassified for
    those classes; for residential a canonical strategy or None (None falls back
    to BTL in the valuation paths, exactly as a null type does today)."""
    prop = dict(prop or {})
    cls = resolution.get("asset_class") or UNCLASSIFIED
    prop["asset_class"] = cls
    prop["asset_class_source"] = source
    prop["asset_class_reason"] = resolution.get("reason")
    prop["asset_class_evidence"] = list(resolution.get("evidence") or [])
    if cls in TYPE_FOR_CLASS:
        prop["type"] = TYPE_FOR_CLASS[cls]
    else:
        prop["type"] = resolution.get("strategy") or (
            normalise_strategy(prop.get("type")) if class_from_type(prop.get("type")) == "residential" else None)
    return prop
