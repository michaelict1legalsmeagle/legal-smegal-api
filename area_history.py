"""
area_history.py — AREA-HIST (30 Sep 2026)

GET /api/deals/<deal_id>/area-history

Serves the evidence the Financials 10-year projection uses, computed from the
stored official series for the deal's own local authority:

  * UK House Price Index (HM Land Registry / ONS, OGL) — table uk_hpi_monthly
    (Supabase; see legalsmegal-data-topology). Three scenario rates:
      last_decade  — annualised change over the most recent 10 years
      worst_decade — the lowest annualised change of any 10-year window
      best_decade  — the highest annualised change of any 10-year window
    Each comes with its window dates, so the page states exactly what it is.
  * ONS Price Index of Private Rents (OGL) — table uk_prms_monthly. Annualised
    change in the rent index over the full stored window, with dates.

Nothing is estimated: if the series is missing or too short, the field is null
with a reason, and the page shows "—".
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, jsonify, request

area_history_bp = Blueprint("area_history", __name__)

HPI_SOURCE = "UK House Price Index (HM Land Registry / ONS), table uk_hpi_monthly"
PRMS_SOURCE = "ONS Price Index of Private Rents, table uk_prms_monthly"


def _as_date(v: Any) -> Optional[date]:
    if v is None:
        return None
    if isinstance(v, date):
        return v
    try:
        return date.fromisoformat(str(v)[:10])
    except ValueError:
        return None


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f > 0 else None


def decade_scenarios(series: List[Tuple[date, float]]) -> Dict[str, Any]:
    """series: (month date, value) ascending. Pairs each month with the same month
    10 years earlier and annualises: (end/start) ** (1/10) - 1."""
    by_date = {d: v for d, v in series}
    windows = []
    for d, v in series:
        try:
            start = d.replace(year=d.year - 10)
        except ValueError:
            continue
        s = by_date.get(start)
        if s:
            windows.append((d, start, (v / s) ** 0.1 - 1.0))
    if not windows:
        return {"ok": False, "reason": "fewer than 10 years of monthly data for this area"}

    def pack(w):
        end, start, rate = w
        return {"rate": rate, "from": start.isoformat(), "to": end.isoformat()}

    last = max(windows, key=lambda w: w[0])
    worst = min(windows, key=lambda w: w[2])
    best = max(windows, key=lambda w: w[2])
    return {"ok": True, "last_decade": pack(last), "worst_decade": pack(worst),
            "best_decade": pack(best), "windows": len(windows)}


def annualised(series: List[Tuple[date, float]]) -> Dict[str, Any]:
    if len(series) < 13:
        return {"ok": False, "reason": "fewer than 13 months of data for this area"}
    (d0, v0), (d1, v1) = series[0], series[-1]
    years = ((d1.year - d0.year) * 12 + (d1.month - d0.month)) / 12.0
    if years <= 0:
        return {"ok": False, "reason": "no time span"}
    return {"ok": True, "rate": (v1 / v0) ** (1.0 / years) - 1.0,
            "from": d0.isoformat(), "to": d1.isoformat(), "years": round(years, 2)}


def build_area_history(area_code: str, query) -> Dict[str, Any]:
    """query(sql, params) -> list[dict]. Separated from Flask for testing."""
    out: Dict[str, Any] = {"ok": True, "area_code": area_code}
    hpi_rows = query(
        "SELECT date, average_price FROM public.uk_hpi_monthly WHERE area_code = %s ORDER BY date ASC LIMIT 1000",
        (area_code,),
    ) or []
    hpi = sorted(((_as_date(r.get("date")), _num(r.get("average_price"))) for r in hpi_rows),
                 key=lambda t: t[0] or date.min)
    hpi = [(d, v) for d, v in hpi if d and v]
    h = decade_scenarios(hpi)
    h["source"] = HPI_SOURCE
    if hpi:
        h["series_from"], h["series_to"] = hpi[0][0].isoformat(), hpi[-1][0].isoformat()
    out["hpi"] = h

    prms_rows = query(
        "SELECT period, rent_index FROM public.uk_prms_monthly WHERE area_code = %s ORDER BY period ASC LIMIT 1000",
        (area_code,),
    ) or []
    prms = sorted(((_as_date(r.get("period")), _num(r.get("rent_index"))) for r in prms_rows),
                  key=lambda t: t[0] or date.min)
    prms = [(d, v) for d, v in prms if d and v]
    r = annualised(prms)
    r["source"] = PRMS_SOURCE
    out["rent"] = r
    return out


def register(require_auth, supabase_client_getter, query):
    """Wire the route with the app's own auth decorator, Supabase client and
    supabase_data_query (passed in to avoid a circular import)."""

    @area_history_bp.route("/api/deals/<deal_id>/area-history", methods=["GET"])
    @require_auth
    def get_area_history(deal_id: str):
        sb = supabase_client_getter()
        if not sb:
            return jsonify({"ok": False, "error": "Database unavailable"}), 503
        try:
            row = sb.table("deals").select("area_json").eq("id", deal_id) \
                .eq("user_id", request.user_id).single().execute()
        except Exception:
            return jsonify({"ok": False, "error": "Deal not found"}), 404
        area = ((row.data or {}).get("area_json") or {}) if row else {}
        code = (area.get("area_code") or "").strip()
        if not code:
            return jsonify({"ok": False, "reason": "deal has no local-authority code (area data not fetched)"}), 200
        try:
            return jsonify(build_area_history(code, query)), 200
        except Exception as e:  # never invent: report the failure
            return jsonify({"ok": False, "reason": "area history query failed: " + type(e).__name__}), 200

    return area_history_bp
