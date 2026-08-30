# services/legal_analysis.py
# TWO-STAGE LEGAL PACK ANALYSIS PIPELINE
#
# Stage 1 — Extraction: forensic, verbatim, atomic findings from document text
# Stage 2 — Classification: findings → flags, JIS, deal score, summary schema
#
# Both stages call llm_json() which uses OPENROUTER_MODEL env var.
# Recommended: anthropic/claude-sonnet-4-6

import os
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── TIMEOUT ─────────────────────────────────────────────────
# Legal packs can be large — allow generous timeout for both stages
ANALYSIS_TIMEOUT = int(os.getenv("ANALYSIS_TIMEOUT_SECONDS", "120"))


# ── STAGE 1 PROMPT — EXTRACTION ─────────────────────────────
# Your forensic extraction prompt — verbatim, citation-backed, atomic
STAGE_1_SYSTEM = """You are a UK property legal analyst performing STRICT, audit-grade extraction from auction legal pack documents.

MISSION: Extract ONLY explicitly stated, verifiable findings. Output must be citation-backed, minimal, and machine-safe.

ZERO-TOLERANCE RULES:
* NO inference. NO assumptions. NO interpretation.
* ONLY include information explicitly stated in the document.
* If uncertain → EXCLUDE.
* If partially supported → EXCLUDE.
* If citation is incomplete → EXCLUDE.

MANDATORY INTERNAL VALIDATION (DO NOT OUTPUT THIS PROCESS):
For EACH candidate finding, you MUST internally verify:
1. TEXT MATCH CHECK: The evidence must be an exact verbatim substring of the document.
2. CLAIM BOUNDARY CHECK: The claim must not extend beyond what the evidence explicitly states.
3. CITATION CHECK: Document name, clause (if present), and page number must match the source exactly.
4. FIELD JUSTIFICATION CHECK:
   - magnitude must appear explicitly in the evidence or adjacent text
   - consequence must be explicitly stated, not implied
If ANY check fails → DISCARD the finding.

FINDING DEFINITION (STRICT):
Each finding must contain:
* claim: minimal factual statement (no interpretation)
* evidence: exact verbatim quote (no edits, no paraphrasing — copy exact words, max 40 words)
* magnitude: explicitly stated numeric/date/quantitative data or null
* consequence: explicitly stated outcome/obligation/penalty or null
* severity: "critical" | "high" | "note"
  - critical: legal obligation, financial penalty, time constraint that could cause forfeiture or loss
  - high: restriction, covenant, or right that materially affects use or value
  - note: relevant but low-risk factual finding
* citation:
  * document: exact document name
  * clause: exact clause reference or null
  * page: integer page number

ATOMICITY:
* One finding = one fact
* Split aggressively — do NOT group multiple facts

DEDUPLICATION:
* If multiple passages support the same fact, keep ONLY the most explicit instance

EXTRACTION THRESHOLD — only extract findings that meet at least ONE of:
* legal obligation
* financial amount
* time constraint or deadline
* restriction, covenant, or right
* penalty, risk, or liability
Ignore trivial or purely descriptive content.

OUTPUT (STRICT JSON ONLY — no prose, no markdown):
{
  "findings": [
    {
      "claim": "string",
      "evidence": "string",
      "magnitude": "string | null",
      "consequence": "string | null",
      "severity": "critical | high | note",
      "citation": {
        "document": "string",
        "clause": "string | null",
        "page": number
      }
    }
  ]
}

FAIL-SAFE: If no findings pass ALL checks, return: { "findings": [] }"""


# ── STAGE 2 PROMPT — CLASSIFICATION ─────────────────────────
# Takes Stage 1 findings JSON and produces the full summary page schema
STAGE_2_SYSTEM = """You are a UK property legal analyst. You will receive a JSON array of verified legal findings extracted from auction documents.

Your task is to classify these findings into a structured summary for property investors.

RULES:
* Work ONLY from the findings provided — do not add external knowledge
* Every flag must map directly to one or more findings
* Deal score starts at 100. Deduct: critical = 12pts, high = 6pts, missing_document = 4pts
* Completion terms must be extracted from findings only — null if not found in findings

OUTPUT (STRICT JSON ONLY — no prose, no markdown):
{
  "deal_score": number,
  "property": {
    "address": "string or null — full UK property address including postcode if present. Look for: lot description, property address, title, subject property, premises. Extract any address-like text.",
    "postcode": "string or null — UK postcode e.g. B1 1AA",
    "type": "HMO | BTL | Commercial | Development | Unknown",
    "tenure": "Freehold | Leasehold | Unknown",
    "lease_years": number or null,
    "lot_number": "string or null — look for Lot followed by a number",
    "guide_price_pence": number or null
  },
  "completion_terms": {
    "deposit_pct": number or null,
    "deposit_refundable": boolean or null,
    "completion_days": number or null,
    "completion_type": "working | calendar | unknown",
    "buyers_premium_pct": number or null,
    "vacant_possession": boolean or null,
    "deposit_amount_pence": number or null
  },
  "flags": [
    {
      "severity": "critical | high | missing | note",
      "title": "string — one line, max 12 words",
      "summation": "string — one sentence, factual, clause-referenced",
      "risk_category": "string — EXACTLY one value from RISK CATEGORY VOCABULARY below, or 'uncategorised'",
      "source_document": "string",
      "source_clause": "string or null",
      "source_page": number or null,
      "legal_risk_weight": number
    }
  ],
  "flag_counts": {
    "critical": number,
    "high": number,
    "missing": number,
    "note": number
  },
  "viability_statement": "string — 2-3 sentences, factual, no verdict, no recommendation",
  "solicitor_questions": ["string"]
}

RISK CATEGORY VOCABULARY (closed list — S42, 2026-07-05):
local_authority_search, environmental_search, former_landfill, structural_issues_disclaimer, will_not_answer_buyer_enquiries, mining, chancel_repair, road_adoption, common_drains, tree_and_shrub, road_making, restrictive_covenant, indemnity_insurance, no_epc, epc_rating, drainage, absent_landlord, hmo_licen, ground_rent_escalation, alter_access, multi_property_title, groundwater_flooding, surface_water_flooding, flood_risk, radon, japanese_knotweed, building_regulations_completion, planning_enforcement, lawful_development_certificate, eicr, gas_safety, possessory_title, flying_freehold, short_lease, service_charge, notice_to_complete, forfeiture, sitting_tenant, buyer_s_premium, seller_s_costs, nearby_planning, overage, vat_uncertain, ta6, party_wall, absence_of_easement, right_of_way_burden, shared_easement, listed_building_on_adjacent, listed_building_consent, conservation_area, smoke_control, air_quality_management, absolute_no_mortgage, no_disputes, vacant_possession_occupier_letter, epc_valid_until, designated_primarily_residential, furniture_and_effects_may_remain, 21_day_completion, limited_title_guarantee, cannot_assign_contract, sold_as_seen, right_to_buy_covenant, buyer_bears_risk_from_exchange, discrepancy_in_seller_s_legal_fees, climateindex, special_conditions_of_sale_not_present, conveyance_document_incomplete, overhanging_structure, rooms_over_passageway, boundary_moved, epc_present_rating_appears_low, tenant_covenants_indemnity, no_buildings_insurance_details, deemed_to_have_full, title_covers_three_properties, nationally_significant, public_sewer_within, trust_corporation_restriction, indemnifies_seller_for_all_covenant, completion_date_fixed

RISK CATEGORY RULES:
* risk_category MUST be exactly one slug from the vocabulary above, chosen by meaning against the flag's evidence — NOT by matching words in your own title.
* If no category genuinely fits, use "uncategorised". NEVER guess or force the nearest-looking slug: a wrong category silently misprices the flag downstream; "uncategorised" is honest and reviewable.
* The vocabulary is a fixed contract with the pricing engine. Never invent new slugs, never pluralise, never reword.

SCORING RULES:
* legal_risk_weight: 1-10. Critical findings that override contract obligations = 10. Financial exposure = 8-9. Restrictions = 6-7. Procedural = 4-5.
* solicitor_questions: 3-5 specific questions derived directly from the flags. Each question should name the specific clause or document.
* viability_statement: state the flag count and what is resolvable. Never say "recommend" or "advise". Never give a verdict."""


# ── STAGE 3 PROMPT — PACK-LEVEL CROSS-DOCUMENT & STATUTORY REVIEW ──
# Stages 1 and 2 are deliberately single-document and forbid inference/external
# knowledge. That is correct for verbatim extraction but it is BLIND to risks that
# only appear when two documents are compared, or when a stated fact triggers a
# well-established statutory consequence. Stage 3 fills exactly that gap, under
# strict evidence discipline so it cannot fabricate.
STAGE_3_SYSTEM = """You are a UK property legal analyst performing PACK-LEVEL cross-document and statutory review of an auction legal pack. This stage exists to catch material risks that a single-document, no-inference scan cannot see.

You MAY compare facts across documents and apply established UK conveyancing law. You operate under STRICT evidence discipline:

EVIDENCE DISCIPLINE (non-negotiable):
* Every flag MUST quote the exact verbatim text from EACH document it relies on, and name that document. If you cannot quote the triggering fact(s) from the documents provided, DO NOT emit the flag.
* You connect STATED facts. You never invent a fact that is not written in the documents.
* A statutory-consequence flag fires ONLY when its triggering fact is explicitly present in the pack (e.g. only raise a build-over point if a document explicitly states a public sewer/lateral drain is within the property boundary).
* If a comparison is impossible because a document is absent (e.g. no title register present), DO NOT flag — silence is correct.
* No verdicts, no recommendations. State the risk factually and the specific action a buyer's solicitor should take.
* When in doubt, EXCLUDE. A false alarm that scares a buyer off a sound deal is as harmful as a miss.

REVIEW TASKS — check each; emit a flag only where the evidence is present in the documents:

A) SELLER vs REGISTERED PROPRIETOR (cross-document):
Compare the SELLER named in the special conditions / contract against the REGISTERED PROPRIETOR(S) named in the title register Proprietorship Register (the "PROPRIETOR:" entry). If the named seller is NOT among the registered proprietors, this is CRITICAL — the buyer would contract to buy from a party not yet on the title. Corroborate with any "pending application" / unregistered-transfer evidence in the pack. In summation, quote BOTH the seller line and the proprietor line. Category: seller_not_registered_proprietor.

B) REGISTRATION-BLOCKING RESTRICTIONS (title register):
Identify RESTRICTION entries that can block the buyer's own registration: (i) third-party CONSENT restrictions ("no disposition ... without the consent of [named party]") — category registration_consent_restriction; (ii) SETTLEMENT/trust-compliance restrictions requiring a certificate or statutory declaration of compliance with a named settlement/deed — category settlement_compliance_restriction; (iii) Form A / two-trustee (overreaching) restrictions — category trust_corporation_restriction. Where a consent or settlement restriction is present, emit ONE flag stating registration can be blocked until each is satisfied, quoting each restriction; severity critical.

C) STATUTORY OVERLAYS (a stated fact triggers a statutory consequence) — only where the fact is explicit:
* Public sewer/lateral drain within the boundary (drainage search) OR a covenant barring building near sewers/drains -> building or extending may require a build-over agreement under Building Regulations Part H4. Category: public_sewer_within.
* Coal-mining search verdict of "potential risk" / "action required" / within the boundary -> coal mining subsidence risk; a visual mining survey is advised. Category: mining.
* Recent building works recorded with a missing Building Regulations completion certificate -> building-control enforcement exposure (extended to 10 years for works completed after 1 October 2023 by the Building Safety Act 2022). Category: building_safety_act_enforcement.
* Highway not confirmed maintainable at public expense -> possible unadopted-road / private maintenance liability; a highways search is advised. Category: road_adoption.

D) DERIVED FINANCIAL / DELIVERABILITY (combine stated terms) — only where the terms are stated:
* If the sale is UNCONDITIONAL with a short completion window (about 28 days or fewer) AND the title carries a registration or mortgageability risk identified in A or B, flag that mainstream finance is unlikely to complete in time and bridging may be required, with deposit and buyer's-fee forfeiture exposure on failure. Category: unconditional_completion_finance.
* If the seller is not yet the registered proprietor because an already-completed transfer merely awaits registration, note that SDLT sub-sale relief is unlikely to apply (this is two chargeable transactions) — for the buyer's tax adviser to confirm. Category: sub_sale_sdlt.

ALLOWED risk_category slugs (choose EXACTLY one per flag, or "uncategorised" if none genuinely fits):
seller_not_registered_proprietor, registration_consent_restriction, settlement_compliance_restriction, trust_corporation_restriction, public_sewer_within, mining, building_safety_act_enforcement, building_regulations_completion, road_adoption, unconditional_completion_finance, sub_sale_sdlt, restrictive_covenant, indemnifies_seller_for_all_covenant, limited_title_guarantee, buyer_bears_risk_from_exchange, notice_to_complete, uncategorised

OUTPUT (STRICT JSON ONLY — no prose, no markdown):
{
  "flags": [
    {
      "severity": "critical | high | note",
      "title": "string — one line, max 12 words",
      "summation": "string — one sentence that quotes or names the specific evidence from each document relied on",
      "risk_category": "one slug from the allowed list above, or uncategorised",
      "source_document": "string — primary document, or 'multiple: X + Y' for a cross-document finding",
      "source_clause": "string or null",
      "source_page": number or null,
      "legal_risk_weight": number
    }
  ]
}
FAIL-SAFE: if nothing qualifies with evidence, return { "flags": [] }."""


# ── DOCUMENT TYPE DISPLAY NAMES ──────────────────────────────
DOC_TYPE_LABELS = {
    "legal_pack":         "Auctioneer's legal pack",
    "special_conditions": "Special conditions of sale",
    "addendum":           "Addendum / amendments",
    "title_register":     "Title register",
    "title_plan":         "Title plan",
    "local_auth_search":  "Local authority search",
    "lease":              "Lease",
    "epc":                "EPC certificate",
    "survey":             "Survey / structural report",
    "auction_tcs":        "Auction house T&Cs",
    "freehold":           "Freehold title confirmation",
    "deed":               "Deed — transfer/conveyance",
    "tenancy_ast":        "Tenancy agreements / ASTs",
    "unknown":            "Document",
}

CRITICAL_DOC_TYPES = {
    "legal_pack", "special_conditions", "addendum",
    "title_register", "freehold"
}

IMPORTANT_DOC_TYPES = {
    "title_plan", "local_auth_search", "deed", "lease"
}


# ── PACK COMPLETENESS ────────────────────────────────────────
def build_pack_completeness(documents: List[Dict]) -> Dict:
    """Build document pack completeness summary from uploaded documents."""
    present_types = {d.get("doc_type") for d in documents if d.get("doc_type") != "unknown"}

    all_types = list(DOC_TYPE_LABELS.keys())
    all_types.remove("unknown")

    items = []
    for doc_type in all_types:
        present = doc_type in present_types
        if doc_type in CRITICAL_DOC_TYPES:
            severity = "critical"
        elif doc_type in IMPORTANT_DOC_TYPES:
            severity = "important"
        else:
            severity = "optional"

        items.append({
            "doc_type":  doc_type,
            "label":     DOC_TYPE_LABELS[doc_type],
            "present":   present,
            "severity":  severity,
        })

    present_count = sum(1 for i in items if i["present"])
    total = len(items)
    completeness_pct = round((present_count / total) * 100) if total > 0 else 0

    # Missing critical docs become flags
    missing_critical_flags = [
        {
            "severity":         "missing",
            "title":            f"Missing: {DOC_TYPE_LABELS[d]}",
            "summation":        f"{DOC_TYPE_LABELS[d]} not uploaded. Information from this document is unavailable.",
            "source_document":  "Not uploaded",
            "source_clause":    None,
            "source_page":      None,
            "legal_risk_weight": 8 if d in CRITICAL_DOC_TYPES else 5,
        }
        for d in CRITICAL_DOC_TYPES
        if d not in present_types
    ]

    return {
        "items":              items,
        "present_count":      present_count,
        "total":              total,
        "completeness_pct":   completeness_pct,
        "missing_critical":   missing_critical_flags,
    }


# ── MAIN PIPELINE ────────────────────────────────────────────
# ── REGEX-FIRST ENTITY EXTRACTION ───────────────────────────
import re as _re

def _extract_postcode(text: str) -> Optional[str]:
    """Extract UK postcode using regex. Handles all UK postcode formats."""
    pattern = r'([A-Z]{1,2}[0-9]{1,2}[A-Z]?\s*[0-9][A-Z]{2})'
    matches = _re.findall(pattern, text.upper())
    if not matches:
        return None
    # Filter out common false positives
    false_positives = {'WM428', 'WM700', 'MM126', 'WM527'}
    valid = [m.strip() for m in matches if not any(fp in m for fp in false_positives)]
    if not valid:
        return None
    from collections import Counter
    return Counter(valid).most_common(1)[0][0]


def _extract_address_regex(documents: List[Dict]) -> Dict:
    """
    Structured regex extraction of property address.
    Tries multiple document types and patterns in priority order.
    Returns dict with address, postcode, lot_number fields.
    """
    result = {}

    # Combine all extracted text for searching
    all_text = ""
    for doc in documents:
        text = (doc.get("extracted_text") or "").strip()
        if text:
            all_text += f"\n{text}"

    # ── Pattern 1: Local authority search — "Matter:" line followed by address
    # Format: "Prepared for: [solicitor]\n[address]\n[postcode]"
    # Or: "Property: [address]"
    for pattern in [
        r'[Pp]roperty[:\s]+([A-Z0-9][^\n]{5,80}(?:Birmingham|Wolverhampton|Manchester|Leeds|London|Bristol|Sheffield|Liverpool|Coventry|Leicester)[^\n]{0,50})',
        r'[Ss]ubject [Pp]roperty[:\s]+([A-Z0-9][^\n]{5,100})',
        r'[Pp]remises[:\s]+([A-Z0-9][^\n]{5,100})',
    ]:
        m = _re.search(pattern, all_text)
        if m:
            result['address'] = m.group(1).strip()
            break

    # ── Pattern 2: Land Registry — "being [address]" after WEST MIDLANDS etc
    # "The Freehold/Leasehold land... and being [address]"
    area_pattern = r'(?:WEST MIDLANDS|EAST MIDLANDS|GREATER MANCHESTER|WEST YORKSHIRE|SOUTH YORKSHIRE|MERSEYSIDE|TYNE AND WEAR)[^\n]*\n'
    area_match = _re.search(area_pattern, all_text)
    if area_match and not result.get('address'):
        # Look for address in next 2000 chars
        after_area = all_text[area_match.end():area_match.end()+2000]
        being_match = _re.search(
            r'being\s+(?:land at\s+|known as\s+|situate at\s+)?'
            r'([A-Z0-9][^.]{5,150}(?:Road|Street|Lane|Avenue|Drive|Close|Way|Crescent|Grove|Place|Court|Gardens|Terrace|Hill|Rise|View|Walk|Mews)[^.]{0,80})',
            after_area, _re.IGNORECASE
        )
        if being_match:
            result['address'] = being_match.group(1).strip()

    # ── Pattern 3: Lot number from any document
    lot_patterns = [
        r'[Ll]ot\s+[Nn]o\.?\s*:?\s*(\d+)',
        r'[Ll]ot\s+(\d+)\s*[-:]',
        r'LOT\s+(\d+)',
    ]
    for pattern in lot_patterns:
        m = _re.search(pattern, all_text)
        if m:
            result['lot_number'] = m.group(1)
            break

    # ── Pattern 4: Postcode — always try regex regardless
    postcode = _extract_postcode(all_text[:80000])
    if postcode:
        result['postcode'] = postcode

    # ── Pattern 5: Tenure from title register
    if _re.search(r'[Ff]reehold', all_text[:20000]):
        result['tenure'] = 'Freehold'
    elif _re.search(r'[Ll]easehold', all_text[:20000]):
        result['tenure'] = 'Leasehold'

    # ── Pattern 6: Lease term
    lease_match = _re.search(r'[Tt]erm\s+of\s+(\d+)\s+years', all_text[:30000])
    if lease_match:
        result['lease_years'] = int(lease_match.group(1))

    # Filter out false positives — search companies, solicitors, lenders
    FALSE_POSITIVE_KEYWORDS = [
        'infotrack', 'devall law', 'lawful', 'solicitor', 'conveyancer',
        'limited', 'ltd', 'plc', 'llp', 'llc', 'bank', 'mortgage',
        'prepared for', 'prepared by', 'waterloo road', 'fleet street',
        'chancery lane', 'legal services', 'law firm', 'chambers',
        'national westminster', 'barclays', 'halifax', 'nationwide',
        'Birmingham City Council', 'city council', 'district council',
    ]
    if result.get('address'):
        addr_lower = result['address'].lower()
        if any(kw.lower() in addr_lower for kw in FALSE_POSITIVE_KEYWORDS):
            logger.info(f"Address rejected as false positive: {result['address']}")
            result.pop('address', None)

    logger.info(f"Regex extraction: address={result.get('address')}, postcode={result.get('postcode')}, lot={result.get('lot_number')}")
    return result


def run_document_summary(
    documents: List[Dict],
    llm_json_fn: Any,
) -> Dict:
    """
    Full two-stage summary pipeline.

    Args:
        documents: list of document dicts with keys:
                   doc_type, file_name, extracted_text, page_count
        llm_json_fn: the llm_json() function from llm_openrouter

    Returns:
        Complete summary dict matching the summary page schema
    """

    # ── Concatenate document text ──
    combined_text = _build_combined_text(documents)

    if not combined_text.strip():
        return _empty_summary("No text could be extracted from the uploaded documents.")

    # ── Address pre-extraction — regex first, then LLM ──
    address_data = {}

    # Stage 0: Regex extraction — fast, deterministic, no LLM cost
    try:
        address_data = _extract_address_regex(documents)
        logger.info(f"Regex address data: {address_data}")
    except Exception as e:
        logger.warning(f"Regex extraction failed: {e}")

    # Stage 0b: LLM address extraction — only if regex missed the address
    try:
        if not address_data.get('address'):
            # Priority 1: local_auth_search and special_conditions always have clean address at top
            # Priority 2: title_register, title_plan, legal_pack
            priority_1 = {"local_auth_search", "special_conditions", "environmental"}
            priority_2 = {"title_register", "title_plan", "legal_pack"}

            priority_text = ""
            for doc in documents:
                if doc.get("doc_type") in priority_1 and doc.get("extracted_text"):
                    priority_text += f"\n=== {doc.get('doc_type')} ({doc.get('file_name','')}) ===\n"
                    priority_text += (doc.get("extracted_text") or "")[:5000]
            for doc in documents:
                if doc.get("doc_type") in priority_2 and doc.get("extracted_text"):
                    priority_text += f"\n=== {doc.get('doc_type')} ({doc.get('file_name','')}) ===\n"
                    priority_text += (doc.get("extracted_text") or "")[:5000]
            addr_input = (priority_text.strip() or combined_text)[:60000]

            addr_result = llm_json_fn(
            system="""Extract property identification from UK HM Land Registry title register documents.
Return ONLY valid JSON — no prose, no markdown:
{
  "address": "full UK property address including street number, street name, town/city — or null",
  "postcode": "UK postcode e.g. B15 2QT — or null",
  "lot_number": "lot number digits only — or null",
  "tenure": "Freehold or Leasehold or Unknown",
  "lease_years": number or null,
  "property_type": "HMO or BTL or Commercial or Development or Unknown",
  "guide_price_pence": number or null
}

LAND REGISTRY FORMAT — the address appears in this exact pattern:
The document contains "A: Property Register" or "A Property Register" section.
Under it is an administrative area line e.g. "WEST MIDLANDS : BIRMINGHAM"
Then a numbered entry like:
  "1 (date) The Freehold land shown edged with red on the plan of the above Title 
   filed at the Registry and being [THE ADDRESS YOU NEED]."
OR:
  "1 (date) The Leasehold land shown edged... being [ADDRESS]"
OR paragraphs containing "known as", "situate at", "situate and being", "being land at"

EXTRACT the street address from within these numbered paragraphs.
The administrative area (WEST MIDLANDS : BIRMINGHAM) gives you the city.
Combine the street address with the city to form the full address.

ALSO check: lot description lines, special conditions property description, 
any line containing a house number followed by a street name.""",
            prompt=f"Extract property identification:\n\n{addr_input}",
            temperature=0.1,
            )
            address_data = addr_result
            logger.info(f"Address extracted: {address_data.get('address')}")

            # Merge LLM result with regex data — LLM fills gaps regex missed
            if not address_data.get("postcode") and addr_result.get("postcode"):
                address_data["postcode"] = addr_result.get("postcode")
            if not address_data.get("tenure") and addr_result.get("tenure"):
                address_data["tenure"] = addr_result.get("tenure")
            if not address_data.get("lot_number") and addr_result.get("lot_number"):
                address_data["lot_number"] = addr_result.get("lot_number")

    except Exception as e:
        logger.warning(f"Address pre-extraction failed: {e}")

    # ── Stage 1 — Extraction ──
    logger.info(f"Stage 1: extracting findings from {len(combined_text):,} chars")
    # Truncate intelligently — keep first 60k chars (cover pages, key clauses)
    # and last 20k chars (often contains schedules and special conditions)
    if len(combined_text) > 120000:
        truncated = combined_text[:80000] + "\n\n[...middle section truncated...]\n\n" + combined_text[-30000:]
    else:
        truncated = combined_text

    try:
        stage1_result = llm_json_fn(
            system=STAGE_1_SYSTEM,
            prompt=f"Extract all qualifying findings from these auction documents:\n\n{truncated}",
            temperature=0.1,
        )
        findings = stage1_result.get("findings", [])
        logger.info(f"Stage 1 complete: {len(findings)} findings extracted")
    except Exception as e:
        logger.error(f"Stage 1 failed: {e}")
        findings = []

    # ── Stage 2 — Classification ──
    logger.info(f"Stage 2: classifying {len(findings)} findings")
    stage2_result = {}
    if findings:
        try:
            findings_json = json.dumps({"findings": findings}, indent=2)
            stage2_result = llm_json_fn(
                system=STAGE_2_SYSTEM,
                prompt=f"Classify these verified findings into the summary schema:\n\n{findings_json}",
                temperature=0.1,
            )
            logger.info("Stage 2 complete")
        except Exception as e:
            logger.error(f"Stage 2 failed: {e}")
            stage2_result = {}

    # ── Build pack completeness ──
    pack = build_pack_completeness(documents)

    # ── Merge missing document flags into stage 2 flags ──
    flags = stage2_result.get("flags", [])
    flags = flags + pack["missing_critical"]

    # ── Stage 3 — Pack-level cross-document & statutory reasoning ──
    # Catches risks the single-document, no-inference stages cannot see:
    # seller vs registered proprietor, registration-blocking restrictions,
    # statutory overlays (build-over / coal / building-safety / road adoption),
    # and derived finance/deliverability. Evidence-grounded (see STAGE_3_SYSTEM);
    # additive and fully optional — a failure never blocks the summary.
    stage3_flags: List[Dict] = []
    if combined_text.strip():
        try:
            logger.info("Stage 3: pack-level cross-document review")
            stage3_result = llm_json_fn(
                system=STAGE_3_SYSTEM,
                prompt=("Perform a pack-level cross-document and statutory review of these "
                        "auction documents. Emit flags only where the evidence is present:\n\n"
                        + truncated),
                temperature=0.1,
            )
            raw3 = stage3_result.get("flags", []) if isinstance(stage3_result, dict) else []
            # De-duplicate against stage-2 flags so we never double-count the same risk.
            _seen = {((f.get("risk_category") or "").strip().lower(),
                      (f.get("title") or "").strip().lower()[:40]) for f in flags}
            for f in raw3:
                if not isinstance(f, dict):
                    continue
                key = ((f.get("risk_category") or "").strip().lower(),
                       (f.get("title") or "").strip().lower()[:40])
                if key in _seen:
                    continue
                _seen.add(key)
                f.setdefault("source_clause", None)
                f.setdefault("source_page", None)
                stage3_flags.append(f)
            logger.info(f"Stage 3 complete: {len(stage3_flags)} pack-level flag(s)")
        except Exception as e:
            logger.error(f"Stage 3 failed: {e}")
            stage3_flags = []
    flags = flags + stage3_flags

    # Recount after merge
    flag_counts = {
        "critical": sum(1 for f in flags if f.get("severity") == "critical"),
        "high":     sum(1 for f in flags if f.get("severity") == "high"),
        "missing":  sum(1 for f in flags if f.get("severity") == "missing"),
        "note":     sum(1 for f in flags if f.get("severity") == "note"),
    }

    # Recalculate deal score including missing doc deductions and stage-3 flags.
    # Stage 2 already deducted its own flags from deal_score; apply the same
    # tariff (critical 12 / high 6) to the additive stage-3 pack-level flags.
    base_score = stage2_result.get("deal_score", 100)
    missing_deduction = flag_counts["missing"] * 4
    stage3_deduction = sum(
        12 if f.get("severity") == "critical" else 6 if f.get("severity") == "high" else 0
        for f in stage3_flags
    )
    deal_score = max(0, base_score - missing_deduction - stage3_deduction)

    # ── Merge address pre-extraction into property ──
    prop = stage2_result.get("property", {}) or {}
    if not prop.get("address") and address_data.get("address"):
        prop["address"]     = address_data.get("address")
        prop["postcode"]    = address_data.get("postcode")
        prop["lot_number"]  = prop.get("lot_number") or address_data.get("lot_number")
        prop["tenure"]      = prop.get("tenure") or address_data.get("tenure", "Unknown")
        prop["lease_years"] = prop.get("lease_years") or address_data.get("lease_years")
        prop["type"]        = prop.get("type") or address_data.get("property_type", "Unknown")
        prop["guide_price_pence"] = prop.get("guide_price_pence") or address_data.get("guide_price_pence")

    # ── Assemble final output ──
    return {
        "ok":                True,
        "deal_score":        deal_score,
        "property":          prop,
        "completion_terms":  stage2_result.get("completion_terms", {}),
        "flags":             flags,
        "flag_counts":       flag_counts,
        "pack_completeness": pack,
        "viability_statement": stage2_result.get("viability_statement", ""),
        "solicitor_questions": stage2_result.get("solicitor_questions", []),
        "findings_count":    len(findings),
        "documents_processed": len(documents),
    }


# ── HELPERS ──────────────────────────────────────────────────
def _build_combined_text(documents: List[Dict]) -> str:
    """Concatenate all document text with clear section headers.
    
    Per-document limit: 15,000 chars (approx 10-12 pages of legal text).
    Priority docs (special_conditions, addendum, title_register) get 25,000 chars.
    Total cap: 120,000 chars to prevent memory issues on large packs.
    """
    PRIORITY_TYPES = {"special_conditions", "addendum", "title_register", "legal_pack",
                      "local_auth_search"}
    PER_DOC_LIMIT  = 25000  # chars per priority document
    STD_DOC_LIMIT  = 12000  # chars per standard document
    TOTAL_LIMIT    = 120000 # total chars across all documents

    parts = []
    total_chars = 0

    # Process priority documents first
    priority_docs = [d for d in documents if d.get("doc_type") in PRIORITY_TYPES]
    standard_docs = [d for d in documents if d.get("doc_type") not in PRIORITY_TYPES]

    for doc in priority_docs + standard_docs:
        if total_chars >= TOTAL_LIMIT:
            break
        text = (doc.get("extracted_text") or "").strip()
        if not text:
            continue
        doc_type = doc.get("doc_type", "unknown")
        limit = PER_DOC_LIMIT if doc_type in PRIORITY_TYPES else STD_DOC_LIMIT
        text = text[:limit]
        label = DOC_TYPE_LABELS.get(doc_type, "Document")
        filename = doc.get("file_name", "")
        pages = doc.get("page_count", 0)
        header = f"\n\n{'='*60}\nDOCUMENT: {label}\nFILE: {filename}\nPAGES: {pages}\n{'='*60}\n"
        chunk = header + text
        parts.append(chunk)
        total_chars += len(chunk)

    return "\n".join(parts)


def _empty_summary(reason: str) -> Dict:
    return {
        "ok":                   False,
        "error":                reason,
        "deal_score":           0,
        "property":             {},
        "completion_terms":     {},
        "flags":                [],
        "flag_counts":          {"critical": 0, "high": 0, "missing": 0, "note": 0},
        "pack_completeness":    {"items": [], "present_count": 0, "total": 0, "completeness_pct": 0, "missing_critical": []},
        "viability_statement":  "",
        "solicitor_questions":  [],
        "findings_count":       0,
        "documents_processed":  0,
    }
