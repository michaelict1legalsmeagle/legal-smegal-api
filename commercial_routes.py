"""
commercial_routes.py — LegalSmegal Commercial Brief pipeline
================================================================
Separate blueprint for Commercial-strategy deals. Registered alongside
guest_bp in app.py; does not modify or call anything in app.py's
residential ceiling flow.

Why this exists:
  services/ceiling_engine.py gates Commercial deals out with
  status="manual_review_required" (S-COMM-GATE) rather than producing a
  wrong residential-comp number. This blueprint is where those deals land
  instead — a genuinely separate valuation path using
  services/commercial_valuation_engine.py (RICS Investment Method).

Flow:
  1. GET  /api/commercial/<deal_id>          — fetch current commercial
     valuation for a deal (computes from stored inputs; insufficient_evidence
     if no inputs yet saved).
  2. POST /api/commercial/<deal_id>/inputs   — save/update user-supplied
     commercial inputs (rent, yield, lease term) and recompute.
  3. POST /api/commercial/<deal_id>/extract  — S-COMM-P1 (2026-07-11): read
     the deal's uploaded legal-pack documents and populate commercial
     fields from them where the pack actually supports it, tagged
     "extracted" with a page citation via commercial_extraction.py. Never
     overwrites a field a person has already entered by hand — see that
     route's docstring below.

Auth: mirrors app.py's require_auth pattern via a lazy import of
get_user_id_from_request — deferred to call time to avoid a circular
import with app.py (which imports this blueprint at startup).

Storage: commercial inputs are stored at
  deals.financials_json.inputs.commercial
This is an additive, separate key from the residential fields already
stored at financials_json.inputs (strategy, tenure, lease_length, etc.) —
no existing field is read, renamed, or overwritten.
"""

import logging

from flask import Blueprint, request, jsonify

from services.commercial_valuation_engine import calculate_commercial_ceiling
from commercial_extraction import extract_commercial_fields, EXTRACTABLE_FIELDS
import pack_terms

commercial_bp = Blueprint("commercial", __name__)
logger = logging.getLogger(__name__)

# Fields accepted from the client for the commercial inputs form.
# Anything else in the POST body is ignored — this is an explicit allow-list,
# not a passthrough, so unrelated deal fields can never be touched via this route.
_ALLOWED_COMMERCIAL_FIELDS = {
    # Shared
    "asset_class",
    # Investment Method (income_producing_let)
    "passing_rent_pa",
    "market_rent_pa",
    "yield_pct",
    "term_yield_pct",
    "reversion_yield_pct",
    "top_slice_yield_pct",
    "unexpired_term_years",
    "wault_years",
    "wault_to_break_years",
    "tenant_name",
    "rent_review_basis",
    "nation",
    "purchaser_fees_pct",
    "void_months",
    "rent_free_months",
    "tenure",
    "yield_basis",
    # Profits Method (trade_related)
    "fmop_pa",
    "profit_multiplier",
    "fmt_pa",
    # Residual Method (development_site)
    "gdv",
    "build_costs_gbp",
    "professional_fees_gbp",
    "professional_fees_pct_of_build",
    "finance_cost_gbp",
    "interest_rate_pct",
    "build_period_years",
    "contingency_gbp",
    "contingency_pct_of_build",
    "developer_profit_gbp",
    "developer_profit_pct_of_gdv",
    # DRC / Contractor's Method (specialised_owner_occupied)
    "land_value_gbp",
    "gross_replacement_cost_gbp",
    "depreciation_pct",
}


def _get_supabase():
    """Lazy import — avoids circular import with app.py at module load time."""
    from app import supabase
    return supabase


def _get_user_id():
    """Lazy import — same reason as _get_supabase."""
    from app import get_user_id_from_request
    return get_user_id_from_request()


def _lot_documents(supabase, deal_id, user_id, summary_json):
    """COMM-1 (2026-10-05): the deal's readable documents, minus any the
    pack-integrity check found to be about another property (PACK-INTEG-1,
    summary_json.pack_integrity.excluded_files). Returns (documents, excluded
    names). Each document: {file_name, doc_type, text, extracted_text}."""
    excluded = {x.get("file_name") for x in
                ((summary_json or {}).get("pack_integrity") or {}).get("excluded_files") or []
                if isinstance(x, dict)}
    rows = (
        supabase.table("documents")
        .select("file_name,doc_type,extracted_text")
        .eq("deal_id", deal_id)
        .eq("user_id", user_id)
        .execute()
    ).data or []
    docs = []
    for d in rows:
        txt = d.get("extracted_text") or ""
        if not txt.strip() or d.get("file_name") in excluded:
            continue
        docs.append({"file_name": d.get("file_name") or "unknown", "doc_type": d.get("doc_type"),
                     "text": txt, "extracted_text": txt})
    return docs, sorted(n for n in excluded if n)


def _nation_from_postcode(postcode):
    """COMM-2: the SDLT/LTT/LBTT nation from the deal postcode. postcodes.io
    LSOA code: E… England, W… Wales, S… Scotland; a BT postcode is Northern
    Ireland (SDLT). Returns (nation, citation) or (None, reason)."""
    pc = (postcode or "").strip().upper()
    if not pc:
        return None, "no postcode on the deal"
    if pc.startswith("BT"):
        return "england_ni", f"{pc}: Northern Ireland postcode (SDLT applies)"
    try:
        from app import resolve_lsoa_gss_from_postcode
        lsoa, _meta = resolve_lsoa_gss_from_postcode(pc)
    except Exception as exc:
        logger.warning("[commercial] postcode lookup failed for %s: %s", pc, exc)
        return None, f"{pc}: postcode lookup failed"
    code = (lsoa or "").strip().upper()[:1]
    nation = {"E": "england_ni", "W": "wales", "S": "scotland"}.get(code)
    if not nation:
        return None, f"{pc}: postcode did not resolve to a nation"
    return nation, f"{pc}: area code {lsoa} (postcodes.io)"


def _effective_inputs(supabase, deal_id, user_id, deal):
    """COMM-2 (2026-10-05): the inputs the engine values on, built from:
      * what is stored for the deal (user-entered or extracted earlier);
      * where the user has entered nothing, facts from source data only —
        tenure from the lot's special conditions / title registers
        (pack_terms, with the quote), nation from the deal postcode;
      * the costs the special conditions make the buyer pay (pack_costs).
    Source-filled values are NEVER written to the stored inputs, so they are
    never mistaken for the user's own entries. Returns (inputs, provenance,
    context) where context is shown to the user."""
    fins = deal.get("financials_json") or {}
    stored = dict((fins.get("inputs") or {}).get("commercial") or {})
    prov = dict((fins.get("inputs") or {}).get("commercial_provenance") or {})
    fi, pv = dict(stored), dict(prov)
    ctx = {"filled_from_sources": {}, "pack_costs_deducted": [], "pack_costs_not_deducted": [],
           "documents_excluded_other_property": []}
    try:
        docs, excl = _lot_documents(supabase, deal_id, user_id, deal.get("summary_json"))
    except Exception as exc:
        logger.error("[commercial] documents fetch failed for %s: %s", deal_id, exc)
        docs, excl = [], []
    ctx["documents_excluded_other_property"] = excl
    buying = pack_terms.extract_buying(docs)
    costs = pack_terms.extract_buyer_costs(docs)

    if fi.get("tenure") in (None, ""):
        t = (buying.get("facts") or {}).get("tenure")
        if t:
            fi["tenure"] = t["value"].lower()
            pv["tenure"] = {"source": "extracted",
                            "citation": f'{t["file_name"]}: "{t["quote"]}"'}
            ctx["filled_from_sources"]["tenure"] = {"value": t["value"], "source": "legal pack",
                                                    "quote": t["quote"], "file_name": t["file_name"]}
    if fi.get("nation") in (None, ""):
        postcode = deal.get("postcode") or ((deal.get("summary_json") or {}).get("property") or {}).get("postcode")
        nation, cite = _nation_from_postcode(postcode)
        if nation:
            fi["nation"] = nation
            pv["nation"] = {"source": "postcode", "citation": cite}
            ctx["filled_from_sources"]["nation"] = {"value": nation, "source": "deal postcode", "quote": cite}
        else:
            ctx["nation_unresolved"] = cite

    deducted, not_deducted = [], []
    for it in costs.get("items") or []:
        if it.get("per") or it["basis"] == "not_stated" or (
                it["basis"] == "fixed" and it.get("amount_gbp") is None) or (
                it["basis"] == "percent_of_price" and it.get("percent") is None):
            not_deducted.append(dict(it, reason=("charged per " + it["per"]) if it.get("per")
                                     else "amount not stated in the pack"))
        else:
            deducted.append(it)
    for it in costs.get("contingent") or []:
        not_deducted.append(dict(it, reason="only if something happens (e.g. late completion)"))
    fi["pack_costs"] = [{k: it.get(k) for k in ("basis", "amount_gbp", "percent", "minimum_gbp", "plus_vat")}
                        for it in deducted]
    ctx["pack_costs_deducted"] = deducted
    ctx["pack_costs_not_deducted"] = not_deducted
    return fi, pv, ctx


def _heal_residential_seed(supabase, deal_id, deal):
    """COMM-2: a commercial deal must not carry the residential Financial
    Model's creation seed (target_yield 6, ltv_pct 75 — app.create_deal).
    Removed only while the model is still the untouched seed; any saved
    residential model is left alone. Returns True if healed."""
    cls = (((deal.get("summary_json") or {}).get("property") or {}).get("asset_class") or "").lower()
    if not cls:   # deals analysed before ROUTE-1 carry the class in deal_type only
        dt = (deal.get("deal_type") or "").strip().lower()
        cls = {"commercial": "commercial", "mixed use": "mixed_use", "mixed-use": "mixed_use"}.get(dt, "")
    if cls not in ("commercial", "mixed_use", "unclassified"):
        return False          # residential (or not yet analysed): its seed is its own
    fins = deal.get("financials_json") or {}
    inp = fins.get("inputs") or {}
    if not (fins.get("_seeded") and inp.get("target_yield") == 6.0 and inp.get("ltv_pct") == 75.0
            and not fins.get("ok") and not inp.get("purchase_price")):
        return False
    cleaned = dict(fins)
    cleaned["inputs"] = {k: v for k, v in inp.items() if k not in ("target_yield", "ltv_pct")}
    try:
        supabase.table("deals").update({"financials_json": cleaned}).eq("id", deal_id).execute()
        deal["financials_json"] = cleaned
        return True
    except Exception as exc:
        logger.warning("[commercial] seed heal failed for %s: %s", deal_id, exc)
        return False


def _respond(result, deal_id, stored, ctx):
    result["deal_id"] = deal_id
    result["stored_inputs"] = stored          # the form shows only what is stored
    result["sources"] = ctx
    return result


@commercial_bp.route("/api/commercial/<deal_id>", methods=["GET"])
def get_commercial_valuation(deal_id):
    if request.method == "OPTIONS":
        return "", 200

    user_id = _get_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorised — valid JWT required"}), 401

    supabase = _get_supabase()
    if not supabase:
        return jsonify({"error": "Database not configured"}), 503

    try:
        row = (
            supabase.table("deals")
            .select("id,user_id,financials_json,deal_type,summary_json,postcode")
            .eq("id", deal_id)
            .single()
            .execute()
        )
    except Exception as exc:
        logger.warning("[commercial] deal fetch failed for %s: %s", deal_id, exc)
        return jsonify({"error": "Deal not found"}), 404

    deal = row.data or {}
    if not deal:
        return jsonify({"error": "Deal not found"}), 404
    if deal.get("user_id") != user_id:
        return jsonify({"error": "Unauthorised"}), 403

    _heal_residential_seed(supabase, deal_id, deal)
    fi, pv, ctx = _effective_inputs(supabase, deal_id, user_id, deal)
    stored = ((deal.get("financials_json") or {}).get("inputs") or {}).get("commercial") or {}
    result = calculate_commercial_ceiling(fi, provenance=pv)
    return jsonify(_respond(result, deal_id, stored, ctx)), 200


@commercial_bp.route("/api/commercial/<deal_id>/inputs", methods=["POST", "OPTIONS"])
def save_commercial_inputs(deal_id):
    if request.method == "OPTIONS":
        return "", 200

    user_id = _get_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorised — valid JWT required"}), 401

    supabase = _get_supabase()
    if not supabase:
        return jsonify({"error": "Database not configured"}), 503

    body = request.get_json(silent=True) or {}
    incoming = {k: v for k, v in body.items() if k in _ALLOWED_COMMERCIAL_FIELDS}

    try:
        row = (
            supabase.table("deals")
            .select("id,user_id,financials_json,summary_json,postcode")
            .eq("id", deal_id)
            .single()
            .execute()
        )
    except Exception as exc:
        logger.warning("[commercial] deal fetch failed for %s: %s", deal_id, exc)
        return jsonify({"error": "Deal not found"}), 404

    deal = row.data or {}
    if not deal:
        return jsonify({"error": "Deal not found"}), 404
    if deal.get("user_id") != user_id:
        return jsonify({"error": "Unauthorised"}), 403

    fins = deal.get("financials_json") or {}
    inputs = fins.get("inputs") or {}
    existing_commercial = inputs.get("commercial") or {}
    merged_commercial = {**existing_commercial, **incoming}

    inputs["commercial"] = merged_commercial

    # v2.4 provenance contract: every field arriving from the browser form
    # is stamped user_entered, SERVER-SIDE — the client cannot assert
    # "extracted" (any provenance in the request body is ignored: it is not
    # in the allow-list). Fields the user did NOT touch keep whatever
    # provenance they had, so extraction-pipeline citations survive until
    # the person overrides that field, at which point the override is
    # honestly re-stamped as user-entered.
    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()
    provenance = inputs.get("commercial_provenance") or {}
    for field in incoming:
        provenance[field] = {"source": "user_entered", "at": now_iso}
    inputs["commercial_provenance"] = provenance
    fins["inputs"] = inputs

    # v2.3: compute BEFORE persisting so the audit snapshot of this
    # computation is stored in the same single write as the inputs —
    # institutional record-keeping (what was computed, when, by which
    # engine version). Additive key; nothing else reads it yet.
    # COMM-2: value on stored inputs + source-filled facts + pack costs.
    fi, pv, ctx = _effective_inputs(supabase, deal_id, user_id, {**deal, "financials_json": fins})
    result = calculate_commercial_ceiling(fi, provenance=pv)
    _respond(result, deal_id, merged_commercial, ctx)

    outputs = fins.get("outputs") or {}
    outputs["commercial_last"] = {
        "computed_at":         datetime.now(timezone.utc).isoformat(),
        "engine_version":      (result.get("audit") or {}).get("version"),
        "status":              result.get("status"),
        "method":              result.get("method"),
        "capital_value_gross": result.get("comparable_valuation"),
        "net_value_gbp":       (result.get("purchasers_costs") or {}).get("net_value_gbp"),
        "evidence_tier":       (result.get("evidence_tier") or {}).get("tier"),
    }
    fins["outputs"] = outputs

    try:
        supabase.table("deals").update({"financials_json": fins}).eq("id", deal_id).execute()
    except Exception as exc:
        logger.error("[commercial] failed to persist inputs for %s: %s", deal_id, exc)
        return jsonify({"error": "Failed to save commercial inputs"}), 500

    return jsonify(result), 200


@commercial_bp.route("/api/commercial/<deal_id>/extract", methods=["POST", "OPTIONS"])
def extract_commercial_inputs(deal_id):
    """S-COMM-P1 (2026-07-11): populate commercial fields from the deal's
    uploaded legal-pack documents, via commercial_extraction.py.

    NEVER OVERWRITES A USER-ENTERED VALUE. If a field already has
    provenance "user_entered", extraction leaves it untouched even if it
    finds a different value in the documents — a person's deliberate entry
    is not silently replaced by an automated read. A field with no
    existing provenance, or with existing provenance "extracted" (i.e. a
    previous extraction run), IS updated — extraction re-running against a
    freshly uploaded document should be able to fill in what it finds.

    Returns the same shape as GET/POST .../inputs (the recomputed
    valuation), plus an "extraction" key summarising what happened:
      {
        "fields_written": [...],
        "fields_skipped_user_entered": [...],
        "evidence_gaps": [...],
      }
    """
    if request.method == "OPTIONS":
        return "", 200

    user_id = _get_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorised — valid JWT required"}), 401

    supabase = _get_supabase()
    if not supabase:
        return jsonify({"error": "Database not configured"}), 503

    try:
        row = (
            supabase.table("deals")
            .select("id,user_id,financials_json,summary_json,address,postcode")
            .eq("id", deal_id)
            .single()
            .execute()
        )
    except Exception as exc:
        logger.warning("[commercial] deal fetch failed for %s: %s", deal_id, exc)
        return jsonify({"error": "Deal not found"}), 404

    deal = row.data or {}
    if not deal:
        return jsonify({"error": "Deal not found"}), 404
    if deal.get("user_id") != user_id:
        return jsonify({"error": "Unauthorised"}), 403

    try:
        # COMM-1: documents about another property (PACK-INTEG-1) are never read here.
        documents, excluded_files = _lot_documents(supabase, deal_id, user_id, deal.get("summary_json"))
    except Exception as exc:
        logger.error("[commercial] failed to fetch documents for %s: %s", deal_id, exc)
        return jsonify({"error": "Could not fetch documents"}), 500

    if not documents:
        return jsonify({
            "error": "no_text_extracted",
            "detail": "No documents with extracted text found for this deal — nothing to extract from.",
        }), 400

    extraction = extract_commercial_fields(documents, subject_address=deal.get("address"))

    fins = deal.get("financials_json") or {}
    inputs = fins.get("inputs") or {}
    existing_commercial = inputs.get("commercial") or {}
    existing_provenance = inputs.get("commercial_provenance") or {}

    fields_written = []
    fields_skipped = []
    merged_commercial = dict(existing_commercial)
    provenance = dict(existing_provenance)

    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()

    for field, value in extraction["fields"].items():
        if field not in EXTRACTABLE_FIELDS:
            continue  # defensive — extract_commercial_fields already filters this, but never trust a single layer
        current_prov = provenance.get(field)
        current_source = current_prov.get("source") if isinstance(current_prov, dict) else None
        if current_source == "user_entered":
            fields_skipped.append(field)
            continue
        merged_commercial[field] = value
        provenance[field] = {
            "source": "extracted",
            "citation": extraction["provenance"][field]["citation"],
            "at": now_iso,
        }
        fields_written.append(field)

    inputs["commercial"] = merged_commercial
    inputs["commercial_provenance"] = provenance
    fins["inputs"] = inputs

    fi, pv, ctx = _effective_inputs(supabase, deal_id, user_id, {**deal, "financials_json": fins})
    result = calculate_commercial_ceiling(fi, provenance=pv)
    _respond(result, deal_id, merged_commercial, ctx)
    result["extraction"] = {
        "fields_written":              fields_written,
        "fields_skipped_user_entered": fields_skipped,
        "evidence_gaps":               extraction["evidence_gaps"],
        "documents_excluded_other_property": excluded_files,
    }

    outputs = fins.get("outputs") or {}
    outputs["commercial_last"] = {
        "computed_at":         now_iso,
        "engine_version":      (result.get("audit") or {}).get("version"),
        "status":              result.get("status"),
        "method":              result.get("method"),
        "capital_value_gross": result.get("comparable_valuation"),
        "net_value_gbp":       (result.get("purchasers_costs") or {}).get("net_value_gbp"),
        "evidence_tier":       (result.get("evidence_tier") or {}).get("tier"),
    }
    fins["outputs"] = outputs

    try:
        supabase.table("deals").update({"financials_json": fins}).eq("id", deal_id).execute()
    except Exception as exc:
        logger.error("[commercial] failed to persist extraction for %s: %s", deal_id, exc)
        return jsonify({"error": "Failed to save extracted commercial inputs"}), 500

    return jsonify(result), 200


@commercial_bp.route("/api/commercial/<deal_id>/pack-terms", methods=["GET"])
def get_pack_terms(deal_id):
    """COMM-1 (2026-10-05): what the buyer is buying and what they pay on top
    of the price, read from this lot's special conditions (pack_terms.py).
    Read-only: computed from the stored document text on every call, nothing
    written. Every item carries the pack's own wording and file name; a fact
    the pack does not state is listed in buying.not_stated."""
    user_id = _get_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorised — valid JWT required"}), 401
    supabase = _get_supabase()
    if not supabase:
        return jsonify({"error": "Database not configured"}), 503
    try:
        row = (supabase.table("deals").select("id,user_id,summary_json")
               .eq("id", deal_id).single().execute())
    except Exception as exc:
        logger.warning("[commercial] deal fetch failed for %s: %s", deal_id, exc)
        return jsonify({"error": "Deal not found"}), 404
    deal = row.data or {}
    if not deal:
        return jsonify({"error": "Deal not found"}), 404
    if deal.get("user_id") != user_id:
        return jsonify({"error": "Unauthorised"}), 403
    try:
        documents, excluded_files = _lot_documents(supabase, deal_id, user_id, deal.get("summary_json"))
    except Exception as exc:
        logger.error("[commercial] failed to fetch documents for %s: %s", deal_id, exc)
        return jsonify({"error": "Could not fetch documents"}), 500
    prop = (deal.get("summary_json") or {}).get("property") or {}
    return jsonify({
        "deal_id": deal_id,
        "buying": pack_terms.extract_buying(documents),
        "costs": pack_terms.extract_buyer_costs(documents),
        "asset_class": {"value": prop.get("asset_class"), "reason": prop.get("asset_class_reason"),
                        "quotes": prop.get("asset_class_evidence") or []},
        "documents_excluded_other_property": excluded_files,
    }), 200
