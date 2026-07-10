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
            .select("id,user_id,financials_json,deal_type,summary_json")
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
    commercial_inputs = (fins.get("inputs") or {}).get("commercial") or {}
    provenance = (fins.get("inputs") or {}).get("commercial_provenance") or {}

    result = calculate_commercial_ceiling(commercial_inputs, provenance=provenance)
    result["deal_id"] = deal_id
    return jsonify(result), 200


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
            .select("id,user_id,financials_json")
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
    result = calculate_commercial_ceiling(merged_commercial, provenance=provenance)
    result["deal_id"] = deal_id

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
