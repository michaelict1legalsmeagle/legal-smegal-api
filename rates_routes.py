"""
rates_routes.py — LegalSmegal lender-rate API (serves Table 1 + Table 2)

Register in app.py alongside the existing blueprints (one line):

    from rates_routes import rates_bp; app.register_blueprint(rates_bp)

GET /api/rates  (PUBLIC — lender rates are the same for everyone, not user data,
so no JWT is required; this keeps the frontend swap trivial.)

Returns lender_rates grouped into the EXACT shape the existing frontend
LENDER_DATA object uses, plus the BoE benchmark and two honesty signals:
  • meta.lender_stale  — true if the newest lender row is older than the threshold
  • cross_check.flagged — named lenders whose rate sits implausibly far from the
                          BoE 5yr/75% benchmark (auto sanity-check, so a drifted
                          manual row flags ITSELF for free).

FAILURE CONTRACT: on any DB error we return 503 with empty lender arrays and an
error string. The frontend then shows "unavailable" — never stale-as-live,
never fabricated.
"""

import os
import datetime as dt
from flask import Blueprint, jsonify

rates_bp = Blueprint("rates_bp", __name__)

LENDER_STALE_DAYS = 45          # lender rows older than this are flagged stale
BENCH_STALE_DAYS = 45           # BoE is monthly
CROSS_CHECK_TOLERANCE_PP = 1.5  # flag a mortgage rate > benchmark + this many points

_STRATEGIES = ["btl", "hmo", "bridging", "brrrRefi", "sa", "land"]


def _sb():
    from supabase import create_client
    url = (os.getenv("SUPABASE_URL") or "").strip()
    key = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY") or "").strip()
    if not url or not key:
        raise RuntimeError("Supabase credentials not configured")
    return create_client(url, key)


def _row_to_obj(r: dict) -> dict:
    """Map a lender_rates DB row to the frontend object shape (only include keys
    the original object carried, so the render is byte-for-byte compatible)."""
    o = {
        "name": r["name"],
        "maxLTV": _num(r.get("max_ltv")),
        "rateFrom": _num(r.get("rate_from")),
        "arrangeFee": _num(r.get("arrange_fee")),
        "icr": _num(r.get("icr")),
        "personal": bool(r.get("personal")),
        "company": bool(r.get("company")),
        "hmoBlock": bool(r.get("hmo_block")),
        "notes": r.get("notes") or "",
        "tags": r.get("tags") or [],
        "asOf": (str(r.get("as_of"))[:10] if r.get("as_of") else None),
        "source": r.get("source") or "",
    }
    if r.get("hmo_premium") is not None:
        o["hmoPremium"] = _num(r.get("hmo_premium"))
    if r.get("short_lease_ok"):
        o["shortLeaseOK"] = True
    if r.get("is_bridging"):
        o["bridging"] = True
    if r.get("is_land"):
        o["land"] = True
    if (r.get("fee_type") or "gbp") == "pct":
        o["feeType"] = "pct"
    return o


def _num(v):
    if v is None:
        return None
    try:
        f = float(v)
        return int(f) if f == int(f) else f
    except (TypeError, ValueError):
        return None


def _age_days(as_of_str) -> int:
    try:
        d = dt.date.fromisoformat(str(as_of_str)[:10])
        return (dt.date.today() - d).days
    except (TypeError, ValueError):
        return 10**6  # unknown date => treat as very stale


@rates_bp.route("/api/rates", methods=["GET"])
def get_rates():
    try:
        sb = _sb()
        lender_rows = (sb.table("lender_rates").select("*").eq("active", True).execute()).data or []
        bench_rows = (sb.table("bench_rates").select("*").execute()).data or []
    except Exception as e:  # noqa: BLE001 — honest failure, no fabrication
        return jsonify({
            "error": "Lender rates temporarily unavailable",
            "detail": str(e)[:200],
            "lenders": {s: [] for s in _STRATEGIES},
        }), 503

    # Group lenders by strategy in the exact frontend shape.
    lenders = {s: [] for s in _STRATEGIES}
    newest_as_of = None
    for r in lender_rows:
        strat = r.get("strategy")
        if strat in lenders:
            lenders[strat].append(_row_to_obj(r))
            ao = str(r.get("as_of") or "")[:10]
            if ao and (newest_as_of is None or ao > newest_as_of):
                newest_as_of = ao

    # Benchmark block.
    bench = {"bank_rate": None, "fix2_75": None, "fix5_75": None, "as_of": None, "stale": True}
    bench_as_of = None
    for b in bench_rows:
        code, rate = b.get("series_code"), _num(b.get("rate_pct"))
        if code == "IUDBEDR":
            bench["bank_rate"] = rate
        elif code == "IUMBV34":
            bench["fix2_75"] = rate
        elif code == "IUMBV42":
            bench["fix5_75"] = rate
        ao = str(b.get("as_of") or "")[:10]
        if ao and (bench_as_of is None or ao > bench_as_of):
            bench_as_of = ao
    if bench_as_of:
        bench["as_of"] = bench_as_of
        bench["stale"] = _age_days(bench_as_of) > BENCH_STALE_DAYS

    # Cross-check: flag mortgage BTL rows implausibly far above the BoE 5yr/75%.
    flagged = []
    ref = bench["fix5_75"]
    if ref is not None:
        for o in lenders["btl"]:
            if o.get("feeType") == "pct":
                continue  # bridging-style, not comparable
            rf = o.get("rateFrom")
            if isinstance(rf, (int, float)) and rf > ref + CROSS_CHECK_TOLERANCE_PP:
                flagged.append(o["name"])

    lender_stale = (newest_as_of is None) or (_age_days(newest_as_of) > LENDER_STALE_DAYS)
    rates_updated = _fmt_month(newest_as_of) if newest_as_of else "unavailable"

    return jsonify({
        "lenders": lenders,
        "rates_updated": rates_updated,
        "meta": {
            "lender_as_of": newest_as_of,
            "lender_stale": lender_stale,
            "threshold_days": LENDER_STALE_DAYS,
        },
        "bench": bench,
        "cross_check": {
            "flagged": flagged,
            "benchmark_5yr_75": ref,
            "tolerance_pp": CROSS_CHECK_TOLERANCE_PP,
        },
    })


def _fmt_month(iso_date: str) -> str:
    try:
        d = dt.date.fromisoformat(iso_date[:10])
        return d.strftime("%b %Y")
    except (TypeError, ValueError):
        return "unavailable"


# ============================================================================
# RATE CANDIDATES — human-gated approval board (governance: no auto-write of
# discovered values to the live panel; every live rate is human-confirmed).
# These endpoints are ADMIN-ONLY, gated by the ADMIN_TOKEN env var, checked as
# the 'X-Admin-Token' header. /api/rates itself stays public and unchanged.
# ============================================================================

def _admin_ok(req) -> bool:
    want = (os.getenv("ADMIN_TOKEN") or "").strip()
    got = (req.headers.get("X-Admin-Token") or "").strip()
    return bool(want) and got == want


@rates_bp.route("/api/rate-candidates", methods=["GET"])
def list_candidates():
    """Pending candidates for review. Admin-only."""
    from flask import request
    if not _admin_ok(request):
        return jsonify({"error": "unauthorised"}), 401
    try:
        sb = _sb()
        rows = (sb.table("rate_candidates").select("*")
                .eq("status", "pending").order("detected_at", desc=True)
                .execute()).data or []
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": "candidates unavailable", "detail": str(e)[:200]}), 503
    return jsonify({"pending": rows, "count": len(rows)})


@rates_bp.route("/api/rate-candidates/<int:cid>/approve", methods=["POST"])
def approve_candidate(cid):
    """Apply a candidate to lender_rates (human gate). Optional JSON body
    {"value": <override>} lets the reviewer correct the proposed number against
    the source before it goes live. Admin-only."""
    from flask import request
    if not _admin_ok(request):
        return jsonify({"error": "unauthorised"}), 401
    try:
        sb = _sb()
        cand = (sb.table("rate_candidates").select("*").eq("id", cid)
                .eq("status", "pending").limit(1).execute()).data
        if not cand:
            return jsonify({"error": "candidate not found or already decided"}), 404
        c = cand[0]
        body = request.get_json(silent=True) or {}
        # Reviewer may override the machine-proposed value after checking source.
        value = body.get("value", c.get("proposed_value"))
        field = c.get("field")  # e.g. 'rate_from' | 'max_ltv' | 'icr'
        allowed = {"rate_from", "max_ltv", "icr", "arrange_fee"}
        if field not in allowed:
            return jsonify({"error": f"field '{field}' not approvable"}), 400
        as_of = c.get("source_date") or dt.date.today().isoformat()
        # Update the live row, stamped with the SOURCE date + link (provenance).
        sb.table("lender_rates").update({
            field: value,
            "as_of": as_of,
            "source": c.get("source_url") or "human-approved",
            "updated_at": dt.datetime.utcnow().isoformat(),
        }).eq("strategy", c.get("strategy")).eq("name", c.get("lender")).execute()
        sb.table("rate_candidates").update({
            "status": "approved", "decided_at": dt.datetime.utcnow().isoformat(),
            "approved_value": value,
        }).eq("id", cid).execute()
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": "approve failed", "detail": str(e)[:200]}), 500
    return jsonify({"ok": True, "applied": {"lender": c.get("lender"),
                    "field": field, "value": value, "as_of": as_of}})


@rates_bp.route("/api/rate-candidates/<int:cid>/reject", methods=["POST"])
def reject_candidate(cid):
    """Dismiss a candidate without touching lender_rates. Admin-only."""
    from flask import request
    if not _admin_ok(request):
        return jsonify({"error": "unauthorised"}), 401
    try:
        sb = _sb()
        sb.table("rate_candidates").update({
            "status": "rejected", "decided_at": dt.datetime.utcnow().isoformat(),
        }).eq("id", cid).eq("status", "pending").execute()
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": "reject failed", "detail": str(e)[:200]}), 500
    return jsonify({"ok": True})
