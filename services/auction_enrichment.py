"""
services/auction_enrichment.py
────────────────────────────────────────────────────────────────────────────
Phase 2 contextual enrichment for discovered auction listings.

DESIGN CONTRACTS
  Stateless:    no globals mutated. Per-run LAD cache passed explicitly.
  Isolated:     every step catches its own exceptions. Failure ≠ abort.
  Additive:     writes only to investment_json, enrichment_status,
                enrichment_confidence, enriched_at. No other columns touched.
  Non-blocking: enrichment failure never affects listing visibility.
  No LLM:       all inference is deterministic rule-based Python.
  Traceable:    every signal references its source and confidence.

ENRICHMENT STEPS (sequential, each independently fenced)
  1. Postcode resolve  → lad_code, lsoa_code        [Hetzner]
  2. HPI benchmark     → regional avg price, YoY    [Supabase RPC]
  3. Rental benchmark  → regional avg rent, YoY     [Supabase]
  4. Sold comps        → count, avg, median          [Hetzner, tiered]
  5. EPC lookup        → rating, rooms, floor area   [Hetzner]
  6. Yield estimate    → gross_yield_pct             [computed]
  7. Contextual inference → signals[], summary       [deterministic rules]

ENTRY POINTS
  enrich_listing(listing, supabase_client, hetzner_url, lad_cache={}) -> dict
  enrich_pass(supabase_client, hetzner_url, limit=100) -> dict

ROLLBACK
  Drop investment_json, enrichment_status, enrichment_confidence, enriched_at
  columns from auction_listings. No other schema touched.
"""

from __future__ import annotations

import logging
import os
import re
import statistics
import time
from datetime import datetime, timezone
from typing import Any, Optional

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

ENRICH_LIMIT_DEFAULT = 100       # max listings per enrichment pass
COMPS_MIN_USEFUL     = 3        # minimum comps before widening to next tier
COMPS_LIMIT_PER_TIER = 50       # row cap per comps query
HETZNER_TIMEOUT      = 15       # seconds — psycopg connect_timeout

# Minimum recent price for a valid comparable (filters noise/data errors)
COMPS_MIN_PRICE = 5_000
COMPS_MAX_PRICE = 5_000_000

# Yield signal thresholds (percentage points above benchmark)
YIELD_STRONG_PP  = 2.0   # >200bps above benchmark → positive signal
YIELD_WEAK_PP    = -1.0  # >100bps below benchmark → caution signal

# Price vs comps thresholds
PRICE_BELOW_COMPS_THRESHOLD = 0.85   # guide < 85% of comps avg → signal
PRICE_ABOVE_COMPS_THRESHOLD = 1.15   # guide > 115% of comps avg → caution

# Price vs HPI thresholds
PRICE_DEEP_DISCOUNT_HPI = 0.70   # guide < 70% of HPI avg → strong signal

# Rental growth threshold
RENTAL_GROWTH_STRONG = 5.0      # >5% YoY → positive signal

# Confidence model weights
CONF_BASE          = 0.50
CONF_COMPS_HIGH    = 0.15   # ≥10 comps
CONF_COMPS_MED     = 0.08   # ≥3 comps
CONF_COMPS_LOW     = 0.02   # <3 comps but exists
CONF_RENTAL        = 0.10
CONF_HPI           = 0.08
CONF_EPC           = 0.05
CONF_PEN_NO_LAD    = 0.35   # cascading failure — most steps depend on this
CONF_PEN_NO_HPI    = 0.10
CONF_PEN_NO_RENTAL = 0.10
CONF_PEN_NO_COMPS  = 0.08
CONF_PEN_NO_PRICE  = 0.15   # no guide price → yield meaningless


# ── Utilities ─────────────────────────────────────────────────────────────────

def _safe_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return None if f != f else f   # guard NaN
    except Exception:
        return None


def _norm_postcode(pc: str) -> str:
    """'wa1 3ea' → 'WA1 3EA'"""
    return " ".join(pc.strip().upper().split()) if isinstance(pc, str) else ""


def _norm_postcode_nospace(pc: str) -> str:
    """'WA1 3EA' → 'WA13EA'"""
    return re.sub(r"\s+", "", pc.strip().upper()) if isinstance(pc, str) else ""


def _postcode_tiers(postcode: str) -> tuple[str, str, str]:
    """
    Return (exact_nospace, sector_prefix, district_prefix) for tiered comps.

    UK inward code is always 3 chars (digit + 2 letters).
    e.g. 'WA1 3EA' → nospace='WA13EA', sector='WA13', district='WA1'
    """
    nospace = _norm_postcode_nospace(postcode)
    sector   = nospace[:-2] if len(nospace) > 3 else nospace
    district = nospace[:-3] if len(nospace) > 3 else nospace
    return nospace, sector, district


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Hetzner connection ─────────────────────────────────────────────────────────

def _hetzner_query(hetzner_url: str, sql: str, params=()) -> list[dict]:
    """Execute a SELECT on Hetzner. Returns [] on any failure."""
    if not hetzner_url:
        return []
    try:
        import psycopg
        from psycopg.rows import dict_row
        with psycopg.connect(
            hetzner_url,
            row_factory=dict_row,
            connect_timeout=HETZNER_TIMEOUT
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        log.warning("[ENRICH] Hetzner query failed: %s", e)
        return []


# ── Step 1: Postcode → LAD code ───────────────────────────────────────────────

def _step_postcode(postcode: str, hetzner_url: str) -> dict:
    """
    Resolve postcode to LAD code + LSOA.

    Strategy (in order):
      1. postcodes.io — free public API, no credentials, full UK coverage.
         Returns LAD E-codes in the correct format for uk_hpi_monthly queries.
      2. Hetzner postcode_to_lsoa — fallback when DATA_DATABASE_URL is set.

    Upstream callers only need 'lad_code'. All other fields are supplementary.
    """
    result: dict = {"ok": False}
    pc = _norm_postcode(postcode)
    if not pc:
        result["error"] = "no_postcode"
        return result

    # ── Primary: postcodes.io (zero credentials, ~100ms) ─────────────────────
    try:
        import requests as _req
        pc_nospace = re.sub(r"\s+", "", pc)
        resp = _req.get(
            f"https://api.postcodes.io/postcodes/{pc_nospace}",
            timeout=6,
        )
        if resp.status_code == 200:
            data = resp.json().get("result") or {}
            codes = data.get("codes") or {}
            lad_code = codes.get("admin_district") or ""
            if lad_code.startswith(("E", "W", "S", "N")):   # valid ONS E/W/S/N code
                return {
                    "ok":        True,
                    "lad_code":  lad_code,
                    "lad_name":  data.get("admin_district") or None,
                    "lsoa_code": codes.get("lsoa") or None,
                    "lsoa_name": None,
                    "source":    "postcodes.io",
                }
        elif resp.status_code == 404:
            # Postcode genuinely not found — don't fall through to Hetzner
            return {"ok": False, "error": "postcode_not_found"}
    except Exception as e:
        log.debug("[ENRICH] postcodes.io failed for %s: %s", pc, e)

    # ── Fallback: Hetzner postcode_to_lsoa ───────────────────────────────────
    if not hetzner_url:
        return {"ok": False, "error": "postcode_not_resolved"}
    try:
        rows = _hetzner_query(
            hetzner_url,
            "SELECT ladcd, ladnm, lsoa11cd, lsoa11nm FROM public.postcode_to_lsoa "
            "WHERE pcds = %s LIMIT 1",
            (pc,)
        )
        if rows:
            r = rows[0]
            return {
                "ok":        True,
                "lad_code":  str(r.get("ladcd") or "").strip() or None,
                "lad_name":  str(r.get("ladnm") or "").strip() or None,
                "lsoa_code": str(r.get("lsoa11cd") or "").strip() or None,
                "lsoa_name": str(r.get("lsoa11nm") or "").strip() or None,
                "source":    "hetzner",
            }
        return {"ok": False, "error": "postcode_not_found"}
    except Exception as e:
        return {"ok": False, "error": f"hetzner_error: {type(e).__name__}"}


# ── Step 2: HPI benchmark ─────────────────────────────────────────────────────

def _step_hpi(lad_code: str, supabase_client, cache: dict) -> dict:
    """
    Regional HPI benchmark via get_hpi_benchmark RPC (SECURITY DEFINER).
    Falls back to England aggregate (E92000001) if LAD not found.
    Results cached per LAD code to avoid N+1 on same area.
    """
    cache_key = f"hpi:{lad_code}"
    if cache_key in cache:
        return cache[cache_key]

    result: dict = {"ok": False}

    def _rpc(code: str) -> Optional[dict]:
        try:
            res = supabase_client.rpc(
                "get_hpi_benchmark", {"p_area_code": code}
            ).execute()
            rows = res.data if hasattr(res, "data") and isinstance(res.data, list) else []
            if rows:
                r = rows[0]
                avg = _safe_float(r.get("average_price"))
                yoy = _safe_float(r.get("annual_change"))
                if avg:
                    return {"avg_price": avg, "yoy_pct": yoy}
        except Exception as e:
            log.debug("[ENRICH] HPI RPC failed for %s: %s", code, e)
        return None

    regional = _rpc(lad_code)
    if regional:
        result.update({
            "ok":                True,
            "regional_avg_price": regional["avg_price"],
            "regional_yoy_pct":   regional["yoy_pct"],
            "fallback_used":      False,
        })
    else:
        # England aggregate fallback
        national = _rpc("E92000001")
        if national:
            result.update({
                "ok":                True,
                "regional_avg_price": national["avg_price"],
                "regional_yoy_pct":   national["yoy_pct"],
                "fallback_used":      True,
                "note":               "LAD not found — England aggregate used",
            })
        else:
            result["error"] = "hpi_rpc_failed"

    cache[cache_key] = result
    return result


# ── Step 3: Rental benchmark ──────────────────────────────────────────────────

def _step_rental(lad_code: str, supabase_client, cache: dict) -> dict:
    """
    Regional rental benchmark from uk_prms_monthly.
    Returns avg monthly rent and YoY growth for the LAD.
    """
    cache_key = f"rental:{lad_code}"
    if cache_key in cache:
        return cache[cache_key]

    result: dict = {"ok": False}
    try:
        res = supabase_client.table("uk_prms_monthly") \
            .select("period,rent_price_gbp,rent_yoy_pct") \
            .eq("area_code", lad_code) \
            .order("period", desc=True) \
            .limit(3) \
            .execute()
        rows = res.data if hasattr(res, "data") and isinstance(res.data, list) else []

        if not rows:
            # Try England aggregate
            res2 = supabase_client.table("uk_prms_monthly") \
                .select("period,rent_price_gbp,rent_yoy_pct") \
                .eq("area_code", "E92000001") \
                .order("period", desc=True) \
                .limit(3) \
                .execute()
            rows = res2.data if hasattr(res2, "data") and isinstance(res2.data, list) else []
            fallback = True
        else:
            fallback = False

        if rows:
            rents = [_safe_float(r.get("rent_price_gbp")) for r in rows
                     if r.get("rent_price_gbp") is not None]
            yoys  = [_safe_float(r.get("rent_yoy_pct")) for r in rows
                     if r.get("rent_yoy_pct") is not None]
            result.update({
                "ok":              True,
                "avg_rent_gbp":    round(sum(rents) / len(rents)) if rents else None,
                "latest_yoy_pct":  round(yoys[0], 2) if yoys else None,
                "fallback_used":   fallback,
            })
        else:
            result["error"] = "no_rental_data"

    except Exception as e:
        result["error"] = f"query_error: {type(e).__name__}"

    cache[cache_key] = result
    return result


# ── Step 4: Sold comps ────────────────────────────────────────────────────────

def _step_comps(postcode: str, hetzner_url: str) -> dict:
    """
    Sold comparables from Hetzner price_paid_raw_2025.
    Tiered postcode matching: exact → sector → district.
    Stops widening once ≥ COMPS_MIN_USEFUL results found.
    """
    result: dict = {"ok": False}
    exact, sector, district = _postcode_tiers(postcode)

    tiers = [
        ("exact",    "postcode_nospace = %s", exact),
        ("sector",   "postcode_nospace LIKE %s", sector + "%"),
        ("district", "postcode_nospace LIKE %s", district + "%"),
    ]

    for tier_name, where_clause, param in tiers:
        try:
            rows = _hetzner_query(
                hetzner_url,
                f"""SELECT price, date_of_transfer
                    FROM public.price_paid_raw_2025
                    WHERE {where_clause}
                      AND price BETWEEN %s AND %s
                      AND date_of_transfer >= '2022-01-01'
                    ORDER BY date_of_transfer DESC
                    LIMIT {COMPS_LIMIT_PER_TIER}""",
                (param, COMPS_MIN_PRICE, COMPS_MAX_PRICE)
            )
        except Exception as e:
            result["error"] = f"comps_query_error: {type(e).__name__}"
            break

        prices = [int(r["price"]) for r in rows
                  if r.get("price") and COMPS_MIN_PRICE <= int(r["price"]) <= COMPS_MAX_PRICE]

        if len(prices) >= COMPS_MIN_USEFUL or tier_name == "district":
            if prices:
                result.update({
                    "ok":          True,
                    "count":       len(prices),
                    "avg_price":   round(sum(prices) / len(prices)),
                    "median_price": round(statistics.median(prices)),
                    "min_price":   min(prices),
                    "max_price":   max(prices),
                    "tier":        tier_name,
                })
            else:
                result["error"] = "no_comps_found"
            break

    return result


# ── Step 5: EPC lookup ────────────────────────────────────────────────────────

def _step_epc(postcode: str, hetzner_url: str) -> dict:
    """
    EPC data from Hetzner epc_certificates.
    Most-recent certificate for exact postcode.
    Scotland postcodes will return no data (EPC covers England+Wales only).
    """
    result: dict = {"ok": False}
    pc = _norm_postcode(postcode)
    if not pc:
        result["error"] = "no_postcode"
        return result
    try:
        rows = _hetzner_query(
            hetzner_url,
            """SELECT current_energy_rating, number_habitable_rooms,
                      total_floor_area, property_type
               FROM public.epc_certificates
               WHERE postcode = %s
               ORDER BY lodgement_date DESC LIMIT 1""",
            (pc,)
        )
        if rows:
            r = rows[0]
            result.update({
                "ok":            True,
                "rating":        str(r.get("current_energy_rating") or "").strip().upper() or None,
                "rooms":         int(r["number_habitable_rooms"]) if r.get("number_habitable_rooms") else None,
                "floor_area_sqm": _safe_float(r.get("total_floor_area")),
                "property_type": str(r.get("property_type") or "").strip() or None,
            })
        else:
            result["error"] = "no_epc_record"
    except Exception as e:
        result["error"] = f"query_error: {type(e).__name__}"
    return result


# ── Step 6: Yield estimate ────────────────────────────────────────────────────

def _step_yield(guide_price: Optional[float], rental: dict) -> dict:
    """
    Gross yield estimate. Uses guide price as cost basis.
    Rental from step 3 (regional benchmark).
    Pure computation — no DB call.
    """
    result: dict = {"ok": False}
    rent = _safe_float(rental.get("avg_rent_gbp")) if rental.get("ok") else None
    price = _safe_float(guide_price)

    if price and rent and price > 0:
        gross = round((rent * 12) / price * 100, 2)
        result.update({
            "ok":                 True,
            "gross_yield_pct":    gross,
            "monthly_rent_used":  round(rent),
            "annual_rent_est":    round(rent * 12),
            "basis":              "guide_price_x_regional_rent",
        })
    elif rent and not price:
        result["error"] = "no_guide_price"
    elif price and not rent:
        result["error"] = "no_rental_benchmark"
    else:
        result["error"] = "insufficient_data"
    return result


# ── Step 7: Contextual inference ──────────────────────────────────────────────

def _step_inference(
    guide_price: Optional[float],
    hpi: dict,
    rental: dict,
    comps: dict,
    epc: dict,
    yield_est: dict,
) -> dict:
    """
    Deterministic rule-based signal generation.
    No LLM. Each signal references its source.
    Returns: signals[], summary, confidence, confidence_band.
    """
    signals: list[dict] = []

    # ── Yield vs regional benchmark ──────────────────────────────────────────
    _yld_ok  = yield_est.get("gross_yield_pct") is not None
    _hpi_ok  = bool(hpi.get("regional_avg_price"))
    _rent_ok = bool(rental.get("avg_rent_gbp"))
    _comp_ok = bool(comps.get("avg_price"))
    _epc_ok  = bool(epc.get("rating"))

    if _yld_ok and _hpi_ok and _rent_ok:
        gross = yield_est["gross_yield_pct"]
        # Derive benchmark yield from HPI price + rental avg
        rental_gbp = _safe_float(rental.get("avg_rent_gbp"))
        hpi_price  = _safe_float(hpi.get("regional_avg_price"))
        if rental_gbp and hpi_price and hpi_price > 0:
            bench_yield = round((rental_gbp * 12) / hpi_price * 100, 2)
            delta = round(gross - bench_yield, 2)
            if delta >= YIELD_STRONG_PP:
                signals.append({
                    "id":         "yield_above_benchmark",
                    "direction":  "positive",
                    "label":      "Gross yield above regional benchmark",
                    "text":       (
                        f"Estimated {gross:.1f}% gross yield vs "
                        f"{bench_yield:.1f}% regional benchmark "
                        f"(+{delta:.1f}pp). Regional rent applied to guide price. "
                        f"Verify local rent achievability."
                    ),
                    "source":     "uk_prms_monthly + guide_price",
                })
            elif delta <= YIELD_WEAK_PP:
                signals.append({
                    "id":         "yield_below_benchmark",
                    "direction":  "negative",
                    "label":      "Gross yield below regional benchmark",
                    "text":       (
                        f"Estimated {gross:.1f}% gross yield vs "
                        f"{bench_yield:.1f}% regional benchmark "
                        f"({delta:.1f}pp). Verify rent assumptions or price sensitivity."
                    ),
                    "source":     "uk_prms_monthly + guide_price",
                })

    # ── Price vs sold comps ───────────────────────────────────────────────────
    if _comp_ok and guide_price:
        comp_avg = comps["avg_price"]
        ratio = guide_price / comp_avg if comp_avg > 0 else None
        if ratio is not None:
            pct_below = round((1 - ratio) * 100, 1)
            if ratio < PRICE_BELOW_COMPS_THRESHOLD:
                signals.append({
                    "id":         "price_below_comps",
                    "direction":  "positive",
                    "label":      "Guide price below comparable evidence",
                    "text":       (
                        f"Guide price £{int(guide_price):,} is {pct_below:.0f}% below "
                        f"{comps['count']} comparables (avg £{comp_avg:,}) "
                        f"at {comps['tier']} level. "
                        f"Investigate condition, tenure and legal pack."
                    ),
                    "source":     f"price_paid_raw_2025 ({comps['tier']} level)",
                })
            elif ratio > PRICE_ABOVE_COMPS_THRESHOLD:
                pct_above = round((ratio - 1) * 100, 1)
                signals.append({
                    "id":         "price_above_comps",
                    "direction":  "negative",
                    "label":      "Guide price above comparable evidence",
                    "text":       (
                        f"Guide price £{int(guide_price):,} is {pct_above:.0f}% above "
                        f"{comps['count']} comparables (avg £{comp_avg:,}) "
                        f"at {comps['tier']} level. Limited discount to market."
                    ),
                    "source":     f"price_paid_raw_2025 ({comps['tier']} level)",
                })

    # ── Price vs HPI deep discount ────────────────────────────────────────────
    if _hpi_ok and guide_price and not _comp_ok:
        # Only fire this if we don't have comps (would be redundant)
        hpi_price = _safe_float(hpi.get("regional_avg_price"))
        if hpi_price and guide_price < hpi_price * PRICE_DEEP_DISCOUNT_HPI:
            pct = round((1 - guide_price / hpi_price) * 100, 1)
            signals.append({
                "id":         "price_deep_discount_hpi",
                "direction":  "positive",
                "label":      "Guide price substantially below HPI average",
                "text":       (
                    f"Guide price £{int(guide_price):,} is {pct:.0f}% below "
                    f"regional HPI average £{int(hpi_price):,}. "
                    f"No sold comparables available — investigate condition and legal pack."
                ),
                "source":     "uk_hpi_monthly (regional)",
            })

    # ── EPC MEES risk ─────────────────────────────────────────────────────────
    if _epc_ok:
        rating = epc["rating"]
        if rating in ("F", "G"):
            signals.append({
                "id":         "epc_mees_risk",
                "direction":  "negative",
                "label":      "MEES compliance risk",
                "text":       (
                    f"EPC rating {rating} — below minimum E threshold for residential letting. "
                    f"Upgrade works required before re-letting. Estimated £3,000–£18,000 "
                    f"depending on property. Factor into acquisition cost."
                ),
                "source":     "epc_certificates",
            })
        elif rating in ("D", "E"):
            signals.append({
                "id":         "epc_upgrade_opportunity",
                "direction":  "neutral",
                "label":      "EPC upgrade opportunity",
                "text":       (
                    f"EPC rating {rating} — one grade below current minimum for MEES compliance. "
                    f"Upgrade to C or above may improve letting appeal and future-proof compliance."
                ),
                "source":     "epc_certificates",
            })

    # ── Rental growth signal ──────────────────────────────────────────────────
    if _rent_ok and rental.get("latest_yoy_pct") is not None:
        yoy = rental["latest_yoy_pct"]
        if yoy > RENTAL_GROWTH_STRONG:
            signals.append({
                "id":         "rental_growth_strong",
                "direction":  "positive",
                "label":      "Rental growth above national average",
                "text":       (
                    f"Local rents growing at {yoy:.1f}% YoY — "
                    f"above the ~3–4% national average. "
                    f"Signals tightening supply or strong demand. Verify local data."
                ),
                "source":     "uk_prms_monthly",
            })

    # ── Thin comp market warning ──────────────────────────────────────────────
    if _comp_ok and comps["count"] < COMPS_MIN_USEFUL:
        signals.append({
            "id":         "thin_comp_market",
            "direction":  "negative",
            "label":      "Limited comparable evidence",
            "text":       (
                f"Only {comps['count']} sold comparables at {comps['tier']} level. "
                f"Pricing confidence reduced — verify with local agent."
            ),
            "source":     f"price_paid_raw_2025 ({comps['tier']} level)",
        })
    elif not _comp_ok and not _hpi_ok:
        signals.append({
            "id":         "insufficient_evidence",
            "direction":  "negative",
            "label":      "Insufficient market data",
            "text":       "Insufficient market data for pricing assessment in this postcode.",
            "source":     "enrichment",
        })

    # ── Confidence score ──────────────────────────────────────────────────────
    conf = _compute_confidence(
        hpi_ok=_hpi_ok,
        rental_ok=_rent_ok,
        comps_ok=_comp_ok,
        comps_count=comps.get("count") if _comp_ok else None,
        epc_ok=_epc_ok,
        has_guide_price=bool(guide_price),
    )
    band = (
        "high"   if conf >= 0.70 else
        "medium" if conf >= 0.40 else
        "low"
    )

    # ── Summary line ─────────────────────────────────────────────────────────
    positive = [s for s in signals if s["direction"] == "positive"]
    negative = [s for s in signals if s["direction"] == "negative"]
    neutral  = [s for s in signals if s["direction"] == "neutral"]

    if positive and negative:
        summary = f"{positive[0]['label']}; {negative[0]['label']}. {band.capitalize()} confidence."
    elif positive:
        summary = f"{positive[0]['label']}." + (
            f" {positive[1]['label']}." if len(positive) > 1 else ""
        ) + f" {band.capitalize()} confidence."
    elif negative:
        summary = f"{negative[0]['label']}." + (
            f" {negative[1]['label']}." if len(negative) > 1 else ""
        ) + f" {band.capitalize()} confidence."
    elif neutral:
        summary = f"{neutral[0]['label']}. {band.capitalize()} confidence."
    else:
        summary = f"No significant asymmetries detected. {band.capitalize()} confidence."

    # Truncate to keep summary scannable
    if len(summary) > 140:
        summary = summary[:137] + "..."

    return {
        "ok":               True,
        "signals":          signals,
        "summary":          summary,
        "confidence":       conf,
        "confidence_band":  band,
    }


def _compute_confidence(
    hpi_ok: bool,
    rental_ok: bool,
    comps_ok: bool,
    comps_count: Optional[int],
    epc_ok: bool,
    has_guide_price: bool,
) -> float:
    score = CONF_BASE
    if hpi_ok:    score += CONF_HPI
    if rental_ok: score += CONF_RENTAL
    if epc_ok:    score += CONF_EPC
    if comps_ok:
        if (comps_count or 0) >= 10: score += CONF_COMPS_HIGH
        elif (comps_count or 0) >= 3: score += CONF_COMPS_MED
        else:                         score += CONF_COMPS_LOW
    if not hpi_ok:          score -= CONF_PEN_NO_HPI
    if not rental_ok:       score -= CONF_PEN_NO_RENTAL
    if not comps_ok:        score -= CONF_PEN_NO_COMPS
    if not has_guide_price: score -= CONF_PEN_NO_PRICE
    return round(max(0.10, min(0.90, score)), 2)


# ── Core enrichment function ───────────────────────────────────────────────────


# ── Step 8: Ceiling engine wrapper ────────────────────────────────────────────
def _step_ceiling(guide_price: Optional[float], comps: dict, rental: dict,
                  strategy: str = "BTL") -> dict:
    """
    Additive wrapper around ceiling_engine.calculate_ceiling.
    At discovery stage there are no legal flags — the ceiling reflects
    structural auction discount applied to comps-anchored base only.
    Result is stored as a quick-reference bid ceiling range for the card.
    Full ceiling (with legal flags) runs separately after legal pack analysis.
    """
    result: dict = {"ok": False}
    try:
        from services.ceiling_engine import calculate_ceiling as _calc_ceiling
    except ImportError:
        result["error"] = "ceiling_engine_unavailable"
        return result

    fins: dict = {}
    if comps and comps.get("avg_price") and float(comps["avg_price"]) > 5_000:
        fins["comps_avg_value"] = float(comps["avg_price"])
    if rental and rental.get("avg_rent_gbp") and float(rental["avg_rent_gbp"]) > 0:
        fins["monthly_rent"] = float(rental["avg_rent_gbp"])

    if not fins:
        result["error"] = "insufficient_inputs"
        return result

    try:
        out = _calc_ceiling(
            legal_flags      = [],          # no flags at discovery stage
            financial_inputs = fins,
            base_valuation   = None,
            strategy         = strategy,
        )
        if out.get("error"):
            result["error"] = out["error"]
            return result

        cr = out.get("ceiling_range", {})
        result.update({
            "ok":               True,
            "ceiling_low":      cr.get("low"),
            "ceiling_high":     cr.get("high"),
            "base_valuation":   out.get("base_valuation"),
            "base_method":      out.get("base_method"),
            "strategy":         out.get("strategy_used"),
            "risk_discount_pct": out.get("risk_discount_pct"),
            "confidence":       out.get("confidence"),
            "note":             "No legal flags applied — discovery ceiling only.",
        })
    except Exception as e:
        result["error"] = f"ceiling_error: {type(e).__name__}: {str(e)[:120]}"
    return result


# ── Step 9: Planning context ────────────────────────────────────────────────
def _step_planning(postcode: str) -> dict:
    """
    Queries planning.data.gov.uk for planning constraints near the postcode.
    No credentials required. Compressed output only — no raw payloads.

    Datasets queried:
      - article-4-direction : HMO and development restrictions
      - conservation-area   : affects alterations, extensions
    """
    result: dict = {"ok": False}
    pc = _norm_postcode(postcode)
    if not pc:
        result["error"] = "no_postcode"
        return result

    pc_q = re.sub(r"\s+", "%20", pc)   # URL-safe postcode

    BASE = "https://www.planning.data.gov.uk/entity.json"
    DATASETS = [
        ("article-4-direction", "article_4"),
        ("conservation-area",   "conservation_area"),
    ]

    try:
        import requests as _req
        planning: dict = {"source": "planning.data.gov.uk"}

        for dataset, key in DATASETS:
            try:
                r = _req.get(
                    f"{BASE}?postcode={pc_q}&dataset={dataset}&limit=5",
                    timeout=8,
                )
                if r.status_code == 200:
                    entities = r.json().get("entities", [])
                    planning[key] = len(entities) > 0
                    if planning[key] and dataset == "article-4-direction":
                        # Store first article 4 name for context
                        names = [e.get("name") or e.get("reference", "")
                                 for e in entities if e.get("name") or e.get("reference")]
                        if names:
                            planning["article_4_ref"] = names[0]
                else:
                    planning[key] = None   # API unavailable for this dataset
            except Exception:
                planning[key] = None

        result.update(planning)
        result["ok"] = any(
            planning.get(k) is not None for _, k in DATASETS
        )

    except Exception as e:
        result["error"] = f"planning_api_error: {type(e).__name__}"

    return result


def _fetch_lot_image(source_url: str) -> "str | None":
    """Fetch og:image from a lot detail page. Called once when image_url is null."""
    if not source_url:
        return None
    try:
        import requests as _req
        r = _req.get(source_url, timeout=5,
                     headers={"User-Agent": "Mozilla/5.0 (compatible; LegalSmegal/1.0)"},
                     allow_redirects=True)
        if r.status_code != 200:
            return None
        text = r.text
        for prop in ('property="og:image"', 'name="og:image"', "property='og:image'"):
            idx = text.find(prop)
            if idx == -1:
                continue
            tag_start = text.rfind("<meta", 0, idx)
            if tag_start == -1:
                continue
            tag_end = text.find(">", idx)
            meta_html = text[tag_start:tag_end + 1]
            for prefix in ('content="', "content='"):
                ci = meta_html.find(prefix)
                if ci == -1:
                    continue
                ci += len(prefix)
                quote = prefix[-1]
                end = meta_html.find(quote, ci)
                url = meta_html[ci:end].strip()
                if url.startswith("http"):
                    return url
    except Exception:
        pass
    return None


def enrich_listing(
    listing: dict,
    supabase_client,
    hetzner_url: str,
    lad_cache: Optional[dict] = None,
) -> dict:
    """
    Enrich one auction listing through all 7 steps.

    Args:
        listing:         auction_listings row dict (must include id, postcode, guide_price)
        supabase_client: authenticated Supabase client
        hetzner_url:     DATA_DATABASE_URL connection string
        lad_cache:       shared per-run dict for benchmark caching across listings

    Returns:
        investment_json dict — also contains 'enrichment_status' and
        'enrichment_confidence' for writing to the DB columns.
    """
    if lad_cache is None:
        lad_cache = {}

    listing_id  = listing.get("id", "unknown")
    postcode    = str(listing.get("postcode") or "").strip()
    guide_price = _safe_float(listing.get("guide_price"))

    _t_start = time.time()
    _image_url = listing.get("image_url") or None
    if not _image_url:
        _image_url = _fetch_lot_image(listing.get("source_url") or "")
        if _image_url:
            log.info("[ENRICH:%s] lot image fetched from detail page", listing_id)
    log.info("[ENRICH:%s] ── Start ─── postcode:%s guide:%s", listing_id, postcode, guide_price)

    steps_completed: list[str] = []
    steps_failed:    list[str] = []
    step_timing:     dict = {}
    result: dict = {}

    # Step 1 — Postcode resolve (prerequisite for all Hetzner+benchmark steps)
    _t = time.time()
    step1 = _step_postcode(postcode, hetzner_url)
    step_timing["postcode"] = round(time.time() - _t, 2)
    if step1.get("ok"):
        steps_completed.append("postcode")
        lad_code = step1["lad_code"]
        result["postcode"] = {
            "lad_code":  lad_code,
            "lad_name":  step1.get("lad_name"),
            "lsoa_code": step1.get("lsoa_code"),
        }
    else:
        steps_failed.append("postcode")
        lad_code = None
        result["postcode"] = {"error": step1.get("error")}
        log.warning("[ENRICH:%s] step:postcode FAIL err=%s t=%.2fs", listing_id, step1.get("error"), step_timing.get("postcode",0))

    # Step 2 — HPI benchmark
    if lad_code:
        _t = time.time()
        step2 = _step_hpi(lad_code, supabase_client, lad_cache)
        step_timing["hpi"] = round(time.time() - _t, 2)
        if step2.get("ok"):
            steps_completed.append("hpi")
            result["hpi"] = {
                "regional_avg_price": step2["regional_avg_price"],
                "regional_yoy_pct":   step2.get("regional_yoy_pct"),
                "fallback_used":      step2.get("fallback_used", False),
                "source":             "uk_hpi_monthly (get_hpi_benchmark RPC)",
            }
        else:
            steps_failed.append("hpi")
            result["hpi"] = {"error": step2.get("error")}
    else:
        steps_failed.append("hpi")
        result["hpi"] = {"error": "skipped_no_lad_code"}

    # Step 3 — Rental benchmark
    if lad_code:
        _t = time.time()
        step3 = _step_rental(lad_code, supabase_client, lad_cache)
        step_timing["rental"] = round(time.time() - _t, 2)
        if step3.get("ok"):
            steps_completed.append("rental")
            result["rental"] = {
                "avg_rent_gbp":    step3["avg_rent_gbp"],
                "latest_yoy_pct":  step3.get("latest_yoy_pct"),
                "fallback_used":   step3.get("fallback_used", False),
                "source":          "uk_prms_monthly",
            }
        else:
            steps_failed.append("rental")
            result["rental"] = {"error": step3.get("error")}
    else:
        steps_failed.append("rental")
        result["rental"] = {"error": "skipped_no_lad_code"}
        step3 = {}

    # Step 4 — Sold comps (Hetzner, tiered)
    if postcode and hetzner_url:
        _t = time.time()
        step4 = _step_comps(postcode, hetzner_url)
        step_timing["comps"] = round(time.time() - _t, 2)
        if step4.get("ok"):
            steps_completed.append("comps")
            result["comps"] = {
                "count":        step4["count"],
                "avg_price":    step4["avg_price"],
                "median_price": step4["median_price"],
                "min_price":    step4["min_price"],
                "max_price":    step4["max_price"],
                "tier":         step4["tier"],
                "source":       "price_paid_raw_2025",
            }
        else:
            steps_failed.append("comps")
            result["comps"] = {"error": step4.get("error", "no_comps")}
    else:
        steps_failed.append("comps")
        result["comps"] = {"error": "skipped_no_postcode_or_hetzner"}
        step4 = {}

    # Step 5 — EPC lookup
    if postcode and hetzner_url:
        _t = time.time()
        step5 = _step_epc(postcode, hetzner_url)
        step_timing["epc"] = round(time.time() - _t, 2)
        if step5.get("ok"):
            steps_completed.append("epc")
            result["epc"] = {
                "rating":         step5["rating"],
                "rooms":          step5.get("rooms"),
                "floor_area_sqm": step5.get("floor_area_sqm"),
                "property_type":  step5.get("property_type"),
                "source":         "epc_certificates",
            }
        else:
            steps_failed.append("epc")
            result["epc"] = {"error": step5.get("error")}
    else:
        steps_failed.append("epc")
        result["epc"] = {"error": "skipped_no_postcode_or_hetzner"}
        step5 = {}

    # Step 6 — Yield estimate
    _t = time.time()
    step6 = _step_yield(guide_price, step3)
    step_timing["yield"] = round(time.time() - _t, 3)
    if step6.get("ok"):
        steps_completed.append("yield")
        result["yield_estimate"] = {
            "gross_yield_pct":   step6["gross_yield_pct"],
            "monthly_rent_used": step6["monthly_rent_used"],
            "annual_rent_est":   step6["annual_rent_est"],
            "basis":             step6["basis"],
        }
    else:
        steps_failed.append("yield")
        result["yield_estimate"] = {"error": step6.get("error")}

    # Step 7 — Contextual inference
    _t = time.time()
    step7 = _step_inference(
        guide_price = guide_price,
        hpi         = result.get("hpi", {}),
        rental      = result.get("rental", {}),
        comps       = result.get("comps", {}),
        epc         = result.get("epc", {}),
        yield_est   = result.get("yield_estimate", {}),
    )
    step_timing["inference"] = round(time.time() - _t, 3)
    if step7.get("ok"):
        steps_completed.append("inference")
        result["inference"] = {
            "signals":          step7["signals"],
            "summary":          step7["summary"],
            "confidence":       step7["confidence"],
            "confidence_band":  step7["confidence_band"],
        }
    else:
        steps_failed.append("inference")
        result["inference"] = {"error": "inference_failed"}

    # Step 8 — Ceiling engine (discovery-mode: no legal flags)
    _t = time.time()
    step8 = _step_ceiling(
        guide_price = guide_price,
        comps       = result.get("comps", {}),
        rental      = result.get("rental", {}),
    )
    step_timing["ceiling"] = round(time.time() - _t, 2)
    if step8.get("ok"):
        steps_completed.append("ceiling")
        result["ceiling"] = {
            "ceiling_low":      step8["ceiling_low"],
            "ceiling_high":     step8["ceiling_high"],
            "base_valuation":   step8["base_valuation"],
            "base_method":      step8["base_method"],
            "strategy":         step8["strategy"],
            "risk_discount_pct": step8["risk_discount_pct"],
            "confidence":       step8["confidence"],
            "note":             step8["note"],
        }
        log.info("[ENRICH:%s] step:ceiling ok low=%s high=%s t=%.2fs",
                 listing_id, step8["ceiling_low"], step8["ceiling_high"],
                 step_timing["ceiling"])
    else:
        steps_failed.append("ceiling")
        result["ceiling"] = {"error": step8.get("error")}
        log.warning("[ENRICH:%s] step:ceiling FAIL err=%s t=%.2fs",
                    listing_id, step8.get("error"), step_timing["ceiling"])

    # Step 9 — Planning context (article 4, conservation area)
    if postcode:
        _t = time.time()
        step9 = _step_planning(postcode)
        step_timing["planning"] = round(time.time() - _t, 2)
        if step9.get("ok"):
            steps_completed.append("planning")
            result["planning"] = {
                k: v for k, v in step9.items()
                if k not in ("ok",)
            }
            log.info("[ENRICH:%s] step:planning ok a4=%s ca=%s t=%.2fs",
                     listing_id, step9.get("article_4"), step9.get("conservation_area"),
                     step_timing["planning"])
        else:
            steps_failed.append("planning")
            result["planning"] = {"error": step9.get("error")}
            log.warning("[ENRICH:%s] step:planning FAIL err=%s t=%.2fs",
                        listing_id, step9.get("error"), step_timing.get("planning", 0))
    else:
        steps_failed.append("planning")
        result["planning"] = {"error": "no_postcode"}

    # ── Determine final status and confidence ─────────────────────────────────
    if not steps_failed:
        enrichment_status = "complete"
    elif not steps_completed:
        enrichment_status = "failed"
    else:
        enrichment_status = "partial"

    confidence = step7.get("confidence") if step7.get("ok") else None

    # ── Assemble investment_json ───────────────────────────────────────────────
    _total_s = round(time.time() - _t_start, 1)

    investment_json = {
        "enriched_at":       _now_iso(),
        "steps_completed":   steps_completed,
        "steps_failed":      steps_failed,
        "step_timing":       step_timing,
        "duration_s":        _total_s,
        **result,
    }

    enrichment_error = None
    if steps_failed:
        enrichment_error = {
            "failed_steps": steps_failed,
            "step_errors": {s: result.get(s, {}).get("error") for s in steps_failed if result.get(s, {}).get("error")},
            "recorded_at": _now_iso(),
        }

    log.info(
        "[ENRICH:%s] %s — steps_ok: %s steps_fail: %s confidence: %s",
        listing_id, enrichment_status.upper(),
        steps_completed, steps_failed, confidence,
    )

    log.info("[ENRICH:%s] ── Done %s ─── ok:%s fail:%s conf:%s dur:%.1fs",
             listing_id, enrichment_status.upper(),
             steps_completed, steps_failed, confidence, _total_s)

    return {
        "listing_id":            listing_id,
        "image_url":             _image_url,
        "investment_json":       investment_json,
        "enrichment_status":     enrichment_status,
        "enrichment_confidence": confidence,
        "enrichment_error":      enrichment_error,
    }


# ── Enrichment pass (batch entry point) ───────────────────────────────────────

def enrich_pass(
    supabase_client,
    hetzner_url: str,
    limit: int = ENRICH_LIMIT_DEFAULT,
) -> dict:
    """
    Process up to `limit` active listings where enrichment_status is pending or failed.
    Processes newest first so the discovery feed gains context immediately.

    Returns a summary dict for scan log reporting.
    """
    run_start = time.time()
    log.info("[ENRICH] ════ Enrichment pass starting (limit=%d) ════", limit)

    # Fetch listings needing enrichment
    try:
        # Include 'enriching': listing was being processed when job was killed.
        # Picked up on next run for clean retry.
        res = supabase_client.table("auction_listings") \
            .select("id,postcode,guide_price,auction_house,source_url,image_url") \
            .eq("status", "active") \
            .in_("enrichment_status", ["pending", "failed", "enriching"]) \
            .order("first_seen_at", desc=True) \
            .limit(limit) \
            .execute()
        listings = res.data or []
    except Exception as e:
        log.error("[ENRICH] Failed to fetch listings for enrichment: %s", e)
        return {"ok": False, "error": str(e), "enriched": 0, "failed": 0}

    if not listings:
        log.info("[ENRICH] No listings pending enrichment — pass complete")
        return {"ok": True, "enriched": 0, "failed": 0, "skipped": 0, "duration_s": 0.0}

    log.info("[ENRICH] %d listings to enrich", len(listings))

    # Per-run cache: benchmarks shared across listings in same LAD
    lad_cache: dict = {}
    enriched = failed = 0

    for listing in listings:
        listing_id = listing.get("id", "unknown")
        try:
            try:
                supabase_client.table("auction_listings").update({"enrichment_status":"enriching"}).eq("id", listing_id).execute()
            except Exception as _se:
                log.warning("[ENRICH:%s] Could not set enriching: %s", listing_id, _se)

            enrich_result = enrich_listing(
                listing        = listing,
                supabase_client = supabase_client,
                hetzner_url    = hetzner_url,
                lad_cache      = lad_cache,
            )

            _upd = {
                "investment_json":       enrich_result["investment_json"],
                "enrichment_status":     enrich_result["enrichment_status"],
                "enrichment_confidence": enrich_result["enrichment_confidence"],
                "enriched_at":           enrich_result["investment_json"]["enriched_at"],
                "enrichment_error":      enrich_result.get("enrichment_error"),
            }
            if enrich_result.get("image_url"):
                _upd["image_url"] = enrich_result["image_url"]
            update_res = supabase_client.table("auction_listings").update(_upd).eq("id", listing_id).execute()
            rows_written = len(update_res.data or [])
            if not rows_written:
                log.error("[ENRICH:%s] DB write returned 0 rows — RLS or ID mismatch", listing_id)
                failed += 1
            else:
                log.info("[ENRICH:%s] Persisted status:%s conf:%s rows:%d",
                         listing_id, enrich_result["enrichment_status"],
                         enrich_result["enrichment_confidence"], rows_written)
                enriched += 1

        except Exception as e:
            log.error("[ENRICH:%s] Unexpected error: %s", listing_id, e, exc_info=True)
            # Mark as failed so it's retried next run, but don't re-raise
            try:
                supabase_client.table("auction_listings").update({
                    "enrichment_status": "failed",
                    "investment_json":   {"error": str(e)[:200], "enriched_at": _now_iso()},
                    "enrichment_error":  {"exception": str(e)[:500], "recorded_at": _now_iso()},
                }).eq("id", listing_id).execute()
            except Exception:
                pass
            failed += 1

    duration = round(time.time() - run_start, 1)
    log.info(
        "[ENRICH] ════ Pass complete ════ enriched: %d failed: %d duration: %.1fs",
        enriched, failed, duration,
    )
    return {
        "ok":         True,
        "enriched":   enriched,
        "failed":     failed,
        "total":      len(listings),
        "duration_s": duration,
    }
