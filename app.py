from flask import Flask, request, jsonify
import requests
import os
import time
import json
import re
import math
import random
import csv
from flask_cors import CORS
from supabase import create_client, Client
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
from pathlib import Path
import jwt as pyjwt
import io
import psycopg
from psycopg.rows import dict_row
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# --- Solicitor Q&A (bounded clarification) ---
try:
    from services.ceiling_engine import calculate_ceiling as _calc_ceiling
    _ceiling_engine_available = True
except ImportError:
    _calc_ceiling = None
    _ceiling_engine_available = False
    import logging as _log
    _log.getLogger(__name__).warning("[ceiling] ceiling_engine not available — ceiling skipped")
try:
    from services.solicitor_qa_engine import clarify_flag  # type: ignore
except Exception:
    clarify_flag = None  # type: ignore
try:
    from services.llm_openrouter import llm_json, _openrouter_chat, _extract_json, _normalize_messages  # type: ignore

    def llm_json_raw(*, system=None, prompt=None, temperature=0.1):
        """Like llm_json() but without the score/summary contract validation.
        Returns the parsed JSON directly — used for our custom analysis prompts."""
        try:
            msg_list = _normalize_messages(system=system, prompt=prompt, messages=None)
            content = _openrouter_chat(msg_list, temperature=float(temperature))
            parsed = _extract_json(content)

            if parsed is None:
                try:
                    parsed = json.loads(content)
                except Exception:
                    return {
                        "ok": False,
                        "error": "invalid_json",
                        "raw": content[:500]
                    }

            return parsed

        except Exception as e:
            return {
                "ok": False,
                "error": "llm_failure",
                "message": str(e)
            }

except Exception:
    # Do not crash the whole API if the optional LLM helper is missing/mispackaged.
    def llm_json(*args, **kwargs):  # type: ignore
        return {"ok": False, "error": "llm_helper_unavailable"}
    def llm_json_raw(*args, **kwargs):  # type: ignore
        return {"ok": False, "error": "llm_helper_unavailable"}
# --- Guaranteed Trends fallback (UI hard contract) ---
try:
    # Repo may place this module in backend/services; keep import forgiving.
    from guaranteed_trends import get_guaranteed_market_trends  # type: ignore
except Exception:
    get_guaranteed_market_trends = None  # type: ignore


# --- Solicitor Q&A (bounded clarification) ---
try:
    # Prefer services package layout; keep import forgiving.
    from services.solicitor_qa_engine import answer_flag  # type: ignore
except Exception:
    answer_flag = None  # type: ignore

app = Flask(__name__)

# CORS: wildcard origin + supports_credentials=True is rejected by all modern browsers.
# Use an explicit allowlist. Add CORS_ORIGINS env var on Render if you add more origins.
_CORS_ORIGINS = [
    o.strip() for o in
    (os.getenv("CORS_ORIGINS", "https://legalsmegal-frontend.onrender.com,http://localhost:3000,http://localhost:5173") or "").split(",")
    if o.strip()
]
CORS(
    app,
    resources={r"/*": {"origins": _CORS_ORIGINS}},
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GOOGLE_MAPS_API_KEY = (os.getenv("GOOGLE_MAPS_API_KEY") or "").strip()

SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip()
ENVIRONMENT = (os.getenv("ENVIRONMENT") or "development").strip().lower()
DEV_BYPASS_LIMITS = ENVIRONMENT != "production"  # Set ENVIRONMENT=production in Render to enforce limits
SUPABASE_SERVICE_ROLE_KEY = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
SUPABASE_KEY_FALLBACK = (os.getenv("SUPABASE_KEY") or "").strip()
SUPABASE_KEY = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY_FALLBACK

_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SECONDS = int(os.getenv("MARKET_INSIGHTS_CACHE_TTL_SECONDS", "21600"))

APP_CACHE_BUSTER = (os.getenv("APP_CACHE_BUSTER", "") or "").strip()

MARKET_CONTRACT_MODE = (os.getenv("MARKET_CONTRACT_MODE", "0").strip().lower() in {"1", "true", "yes", "on"})

HTTP_USER_AGENT = os.getenv(
    "HTTP_USER_AGENT",
    "LegalSmegal/1.0 (market-insights; contact=admin@example.com)"
)

MAX_CRIMES = int(os.getenv("MAX_CRIMES", "300"))
MAX_OSM_NAMES = int(os.getenv("MAX_OSM_NAMES", "250"))
DEFAULT_OSM_RADIUS = int(os.getenv("OSM_RADIUS_METERS", "1200"))

MIN_VERIFIED = float(os.getenv("MIN_VERIFIED_CONFIDENCE", "0.95"))

SCHOOLS_PROVIDER = os.getenv("SCHOOLS_PROVIDER", "").strip().lower()
BROADBAND_PROVIDER = os.getenv("BROADBAND_PROVIDER", "").strip().lower()

HOUSING_PROVIDER = os.getenv("HOUSING_PROVIDER", "supabase_rpc").strip().lower()
HOUSING_RPC_NAME = os.getenv("HOUSING_RPC_NAME", "housing_comps_v1").strip()
HOUSING_MAX_LIMIT = int(os.getenv("HOUSING_MAX_LIMIT", "200"))
HOUSING_DEFAULT_LIMIT = int(os.getenv("HOUSING_DEFAULT_LIMIT", "100"))
HOUSING_DEFAULT_RADIUS_MILES = float(os.getenv("HOUSING_DEFAULT_RADIUS_MILES", "3"))
HOUSING_CONFIDENCE_VALUE = float(os.getenv("HOUSING_CONFIDENCE_VALUE", "0.96"))

HOUSING_ENRICH_LATLNG = (os.getenv("HOUSING_ENRICH_LATLNG", "1").strip().lower() in {"1", "true", "yes", "on"})
HOUSING_ENRICH_BATCH_LIMIT = int(os.getenv("HOUSING_ENRICH_BATCH_LIMIT", "10"))

SCHOOLS_SUPABASE_VIEW = os.getenv("SCHOOLS_SUPABASE_VIEW", "schools_by_district").strip()
SCHOOLS_SUPABASE_FALLBACK_TABLE = os.getenv("SCHOOLS_SUPABASE_FALLBACK_TABLE", "schools_clean_v2").strip()

SCHOOLS_MAX_RESULTS = int(os.getenv("SCHOOLS_MAX_RESULTS", "20"))
SCHOOLS_CONFIDENCE_VALUE = float(os.getenv("SCHOOLS_CONFIDENCE_VALUE", "0.90"))

BROADBAND_SUPABASE_TABLE = os.getenv("BROADBAND_SUPABASE_TABLE", "").strip()
BROADBAND_MAX_RESULTS = int(os.getenv("BROADBAND_MAX_RESULTS", "5"))
BROADBAND_CONFIDENCE_VALUE = float(os.getenv("BROADBAND_CONFIDENCE_VALUE", "0.90"))

NOMIS_ENABLED = (os.getenv("NOMIS_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"})
NOMIS_DATASET_ID = os.getenv("NOMIS_DATASET_ID", "NM_2023_1").strip()
NOMIS_DEFAULT_GEOGRAPHY = os.getenv("NOMIS_DEFAULT_GEOGRAPHY", "").strip()
NOMIS_TIMEOUT = int(os.getenv("NOMIS_TIMEOUT", "20"))
NOMIS_FREQ = os.getenv("NOMIS_FREQ", "A").strip()

NOMIS_TS003_DIM = os.getenv("NOMIS_TS003_DIM", "c2021_hhcomp_15").strip()
NOMIS_TS003_CATS = os.getenv(
    "NOMIS_TS003_CATS",
    "1001,1,2,1002,1003,4,5,6,1004,7,8,9,1005,10,11,1006,12,1007,13,14"
).strip()

NOMIS_TS044_DIM = os.getenv("NOMIS_TS044_DIM", "").strip()
NOMIS_TS044_CATS = os.getenv("NOMIS_TS044_CATS", "").strip()

NOMIS_TS054_DIM = os.getenv("NOMIS_TS054_DIM", "").strip()
NOMIS_TS054_CATS = os.getenv("NOMIS_TS054_CATS", "").strip()

POSTCODES_IO_TIMEOUT = int(os.getenv("POSTCODES_IO_TIMEOUT", "10"))
POSTCODES_IO_CACHE_TTL_SECONDS = int(os.getenv("POSTCODES_IO_CACHE_TTL_SECONDS", "2592000"))
_GEO_CACHE: Dict[str, Dict[str, Any]] = {}

GEOCODE_CACHE_TABLE = os.getenv("GEOCODE_CACHE_TABLE", "geocode_cache").strip()
GEOCODE_BATCH_LIMIT = int(os.getenv("GEOCODE_BATCH_LIMIT", "10"))

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("🟢 Supabase enabled. URL:", SUPABASE_URL)
else:
    print("🔴 Supabase env vars not set. Supabase features are DISABLED.")

# ── Hetzner data connection ────────────────────────────────────────
DATA_DATABASE_URL = os.environ.get(
    "DATA_DATABASE_URL",
    "postgresql://legalsmegal:Thesixkids68@159.69.27.104:5432/legalsmegal_data"
)

def get_data_conn():
    """Get a psycopg v3 connection to Hetzner data database."""
    return psycopg.connect(DATA_DATABASE_URL, row_factory=dict_row)

def data_query(sql: str, params=None) -> list:
    """Execute a SELECT query on Hetzner and return list of dicts."""
    try:
        with psycopg.connect(DATA_DATABASE_URL, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params or ())
                return [dict(r) for r in cur.fetchall()]
    except Exception as e:
        print(f"[DATA_QUERY ERROR] {e}")
        return []

print("🟢 Hetzner data connection configured:", DATA_DATABASE_URL.split("@")[1])

# ── Supabase direct Postgres connection (for uk_hpi_monthly, uk_prms_monthly) ──
# Set SUPABASE_DB_URL in Render env to:
# postgresql://postgres.[ref]:[password]@aws-0-eu-west-2.pooler.supabase.com:6543/postgres
SUPABASE_DB_URL = os.environ.get("SUPABASE_DB_URL", "")

def supabase_data_query(sql: str, params=None) -> list:
    """
    Query Supabase tables (uk_hpi_monthly, uk_prms_monthly).
    Primary:  direct psycopg via SUPABASE_DB_URL.
    Fallback: Supabase REST client — works when pooler DNS fails on Render.
    """
    import re as _re2

    if SUPABASE_DB_URL:
        try:
            with psycopg.connect(SUPABASE_DB_URL, row_factory=dict_row, connect_timeout=8) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params or ())
                    return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            print(f"[SUPABASE_DATA_QUERY] direct postgres failed ({e}) — REST fallback")

    if not supabase:
        return []
    try:
        sql_u = sql.upper()
        tm = _re2.search(r"FROM\s+public\.(\w+)", sql, _re2.IGNORECASE)
        if not tm:
            return []
        table = tm.group(1)
        cm = _re2.search(r"SELECT\s+(.+?)\s+FROM", sql, _re2.IGNORECASE | _re2.DOTALL)
        cols_raw = cm.group(1).strip() if cm else "*"
        raw_cols, alias_map = [], {}
        for c in cols_raw.split(","):
            c = c.strip()
            am = _re2.match(r"(\w+)\s+AS\s+(\w+)", c, _re2.IGNORECASE)
            if am:
                raw_cols.append(am.group(1))
                alias_map[am.group(1)] = am.group(2)
            else:
                raw_cols.append(c)
        q = supabase.table(table).select(",".join(raw_cols))
        if "WHERE" in sql_u and "AREA_CODE" in sql_u and params:
            q = q.eq("area_code", params[0])
        if "SELECT MAX" in sql_u:
            # Try "period" (uk_prms_monthly) then "date" (uk_hpi_monthly)
            rows = []
            for _om in ["period", "date"]:
                try:
                    _max_rows = q.order(_om, desc=True).limit(500).execute().data or []
                    if _max_rows:   # truthy — only break when data returned, not on []
                        rows = _max_rows
                        break
                except Exception:
                    continue
            for row in rows:
                for orig, alias in alias_map.items():
                    if orig in row:
                        row[alias] = row[orig]
            return rows
        else:
            desc = "DESC" in sql_u
            lm = _re2.search(r"LIMIT\s+(\d+)", sql, _re2.IGNORECASE)
            lim = int(lm.group(1)) if lm else 500
            # Try ordering by "period" first (uk_prms_monthly), then "date" (uk_hpi_monthly)
            for _order_col in ["period", "date", "month"]:
                try:
                    rows = q.order(_order_col, desc=desc).limit(lim).execute().data or []
                    if rows:   # only break when data actually returned — [] is falsy
                        break
                except Exception:
                    rows = []
                    continue
            for row in rows:
                for orig, alias in alias_map.items():
                    if orig in row:
                        row[alias] = row[orig]
            return rows
        rows = q.execute().data or []
        for row in rows:
            for orig, alias in alias_map.items():
                if orig in row:
                    row[alias] = row[orig]
        return rows
    except Exception as e:
        print(f"[SUPABASE_DATA_QUERY REST ERROR] {e}")
        return []


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _fetch_ons_rent_yoy_series(months: int = 60):
    """Deterministic ONS Rent YoY time series fetch.

    Verified Supabase RPC signature:
      rpc_ons_rent_yoy_series(p_region text, p_months integer)

    We force p_region='UK' to guarantee data exists.
    Returns points shaped for Card 2 UI:
      [{ "period": "YYYY-MM-01", "rent_yoy_pct": float, "value": float }, ...]
    """
    try:
        if not supabase:
            return []
    except Exception:
        return []

    try:
        res = supabase.rpc(
            "rpc_ons_rent_yoy_series",
            {"p_region": "UK", "p_months": int(months)},
        ).execute()
        rows = res.data or []
    except Exception:
        return []

    out = []
    for r in rows:
        period = r.get("period") or r.get("month") or r.get("date")
        yoy = r.get("rent_yoy_pct") or r.get("yoy_pct") or r.get("value")
        if period is None or yoy is None:
            continue
        try:
            yoy_f = float(yoy)
        except Exception:
            continue
        out.append({
            "period": str(period)[:10],
            "rent_yoy_pct": yoy_f,
            "value": yoy_f,
        })
    return out



def normalize_postcode(pc: str) -> str:
    if not isinstance(pc, str):
        return ""
    return " ".join(pc.strip().upper().split())


def normalize_postcode_nospace(pc: str) -> str:
    if not isinstance(pc, str):
        return ""
    return re.sub(r"\s+", "", pc.strip().upper())


def postcode_district(pc: str) -> str:
    pc = normalize_postcode(pc)
    if not pc:
        return ""
    return pc.split(" ")[0] if " " in pc else pc


def is_digits_only(s: Any) -> bool:
    return isinstance(s, str) and s.strip().isdigit()


def cache_get(key: str) -> Optional[Dict[str, Any]]:
    hit = _CACHE.get(key)
    if not hit:
        return None
    if time.time() - hit.get("_cached_at", 0) > CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return hit.get("value")


def cache_set(key: str, value: Dict[str, Any]) -> None:
    _CACHE[key] = {"_cached_at": time.time(), "value": value}


def geo_cache_get(key: str) -> Optional[Dict[str, Any]]:
    hit = _GEO_CACHE.get(key)
    if not hit:
        return None
    if time.time() - hit.get("_cached_at", 0) > POSTCODES_IO_CACHE_TTL_SECONDS:
        _GEO_CACHE.pop(key, None)
        return None
    return hit.get("value")


def geo_cache_set(key: str, value: Dict[str, Any]) -> None:
    _GEO_CACHE[key] = {"_cached_at": time.time(), "value": value}


def safe_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
        if f != f:
            return None
        return f
    except Exception:
        return None


def safe_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _first_source_url(sources: Any) -> str:
    if isinstance(sources, list) and sources:
        s0 = sources[0]
        if isinstance(s0, dict):
            u = s0.get("url")
            if isinstance(u, str) and u.strip():
                return u.strip()
    return ""


def metric_ok(summary: str, value: Any, sources: list, retrieved_at: str, confidence: float) -> Dict[str, Any]:
    cv = float(confidence) if confidence is not None else 0.0
    return {
        "status": "ok",
        "summary": summary or "",
        "value": value,
        "metrics": {},
        "sources": sources or [],
        "sourceUrl": _first_source_url(sources),
        "retrievedAtISO": retrieved_at,
        "confidenceValue": cv,
        "needsEvidence": False if (cv and cv > 0) else True,
    }


def metric_missing_provider(summary: str, sources: list, retrieved_at: str, extra_metrics: Optional[dict] = None) -> Dict[str, Any]:
    return {
        "status": "missing_provider",
        "summary": summary or "",
        "value": None,
        "metrics": extra_metrics or {},
        "sources": sources or [],
        "sourceUrl": _first_source_url(sources),
        "retrievedAtISO": retrieved_at,
        "confidenceValue": 0.0,
        "needsEvidence": True,
    }


def metric_unavailable(summary: str, sources: list, retrieved_at: str, extra_metrics: Optional[dict] = None) -> Dict[str, Any]:
    return {
        "status": "unavailable",
        "summary": summary or "",
        "value": None,
        "metrics": extra_metrics or {},
        "sources": sources or [],
        "sourceUrl": _first_source_url(sources),
        "retrievedAtISO": retrieved_at,
        "confidenceValue": 0.0,
        "needsEvidence": True,
    }

# -------------------------------
# HARD CONTRACT: MARKET TRENDS
# Always present in API payloads.
# Uses evidence already computed inside housing.metrics.
# Anything we present as "real" must meet MIN_VERIFIED.
# -------------------------------

def build_market_trends(housing_metric: Dict[str, Any]) -> Dict[str, Any]:
    retrieved = now_iso()

    base_unavailable = {
        "status": "unavailable",
        "confidenceValue": 0.0,
        "summary": "Market trends not computable from available evidence.",
        "signals": None,
        "retrievedAtISO": retrieved,
    }

    if not isinstance(housing_metric, dict):
        base_unavailable["summary"] = "Housing metric unavailable; market trends not computable."
        return base_unavailable

    metrics = housing_metric.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}

    momentum = metrics.get("pricingPowerSoldCompsMomentum")
    if not isinstance(momentum, dict):
        return base_unavailable

    headline = momentum.get("headline")
    reason = momentum.get("reason")

    summary = ""
    if isinstance(headline, str) and headline.strip():
        summary = headline.strip()
    elif isinstance(reason, str) and reason.strip():
        summary = reason.strip()

    cv = float(momentum.get("confidenceValue") or 0.0)
    status = str(momentum.get("status") or "unknown")
    retrieved_at = str(momentum.get("retrievedAtISO") or retrieved)

    # Hard contract: only "real" if meets MIN_VERIFIED
    if cv >= float(MIN_VERIFIED):
        return {
            "status": status,
            "confidenceValue": cv,
            "summary": summary,
            "signals": momentum,
            "retrievedAtISO": retrieved_at,
        }

    # Below threshold: suppress but keep evidence attached
    return {
        "status": "suppressed",
        "confidenceValue": 0.0,
        "summary": summary or "Trend signal below minimum confidence threshold.",
        "signals": momentum,
        "retrievedAtISO": retrieved_at,
    }


def normalize_trends_payload(trends: Any) -> Any:
    """Make trends series maximally compatible with frontend expectations.

    The UI may look for a generic numeric field like `value`.
    Our backend uses domain-specific keys (e.g. `price_change_pct`, `rental_demand_index`).
    This normalizer adds `value` alongside the domain key without changing meaning.
    """
    if not isinstance(trends, dict):
        return trends
    signals = trends.get("signals")
    if not isinstance(signals, dict):
        return trends

    # priceGrowth points
    try:
        pg = signals.get("priceGrowth")
        hd = (pg or {}).get("historicalData") if isinstance(pg, dict) else None
        if isinstance(hd, list):
            for p in hd:
                if not isinstance(p, dict):
                    continue
                if "value" not in p:
                    if "price_change_pct" in p and isinstance(p.get("price_change_pct"), (int, float)):
                        p["value"] = p.get("price_change_pct")
                    elif "average_price" in p and isinstance(p.get("average_price"), (int, float)):
                        p["value"] = p.get("average_price")
    except Exception:
        pass

    # rentalDemand points
    try:
        rd = signals.get("rentalDemand")
        hd = (rd or {}).get("historicalData") if isinstance(rd, dict) else None
        if isinstance(hd, list):
            for p in hd:
                if not isinstance(p, dict):
                    continue
                if "value" not in p:
                    if "rental_demand_index" in p and isinstance(p.get("rental_demand_index"), (int, float)):
                        p["value"] = p.get("rental_demand_index")
                    elif "index_value" in p and isinstance(p.get("index_value"), (int, float)):
                        p["value"] = p.get("index_value")
    except Exception:
        pass

    return trends


def ensure_market_trends(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload

    # NOTE: Historical name. This function now enforces BOTH:
    # - marketTrends (legacy)
    # - trends (time-series contract consumed by Market Trends UI)

    la = payload.get("localAreaAnalysis")
    la = la if isinstance(la, dict) else {}

    housing = la.get("housing")
    housing = housing if isinstance(housing, dict) else {}

    if "marketTrends" not in payload:
        payload["marketTrends"] = build_market_trends(housing)

    # HARD CONTRACT: trends must always exist (never missing).
    if "trends" not in payload:
        pc = normalize_postcode(payload.get("postcode", "") or "")
        if callable(get_guaranteed_market_trends):
            payload["trends"] = get_guaranteed_market_trends(pc)
        else:
            payload["trends"] = {
                "status": "unavailable",
                "summary": "Trends provider not configured.",
                "confidenceValue": 0.0,
                "signals": None,
                "source": "none",
                "retrievedAtISO": now_iso(),
            }

    # Normalize series points so the frontend can always read numeric values.
    try:
        payload["trends"] = normalize_trends_payload(payload.get("trends"))
    except Exception:
        pass

    
    # Card 2 (Rent Growth YoY): ensure historical series exists for legacy UI path.
    # UI reads: marketTrends.rentalDemand.historicalData
    try:
        series = _fetch_ons_rent_yoy_series(months=60)
        mt = payload.get("marketTrends")
        if isinstance(mt, dict):
            rd = mt.get("rentalDemand")
            if not isinstance(rd, dict):
                rd = {}
                mt["rentalDemand"] = rd
            rd["historicalData"] = series
    except Exception:
        pass

# Ultra-defensive: some UI builds still read `marketTrends` for charts.
    # If `marketTrends` has no usable series but `trends` does, promote `trends`.
    try:
        mt = payload.get("marketTrends")
        tr = payload.get("trends")
        mt_series = (((mt or {}).get("signals") or {}).get("priceGrowth") or {}).get("historicalData")
        tr_series = (((tr or {}).get("signals") or {}).get("priceGrowth") or {}).get("historicalData")
        if (not isinstance(mt_series, list) or len(mt_series) < 2) and isinstance(tr_series, list) and len(tr_series) >= 2:
            payload["marketTrends"] = tr
    except Exception:
        pass
    return payload


def _trend_from_yoy(yoy: Optional[float]) -> str:
    if yoy is None:
        return "Stable"
    if yoy > 0.5:
        return "Increasing"
    if yoy < -0.5:
        return "Decreasing"
    return "Stable"


def _to_ym(v: Any) -> str:
    """Convert RPC `period` (date/str) to YYYY-MM."""
    if isinstance(v, str):
        s = v.strip()
        if len(s) >= 7:
            return s[:7]
        return ""
    try:
        if hasattr(v, "strftime"):
            return v.strftime("%Y-%m")
    except Exception:
        pass
    return ""



# ----------------------------
# UK HPI CSV fallback (Option 1)
# ----------------------------
# Expected file: Average-prices-2025-11.csv (user-provided)
# Columns (case-insensitive): Date, Region_Name, Area_Code, Average_Price, Monthly_Change, Annual_Change, Average_Price_SA
HPI_CSV_PATH = (os.getenv("HPI_CSV_PATH") or "Average-prices-2025-11.csv").strip()

def _normalise_csv_header(h: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (h or "").strip().lower())

def _safe_float(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return None
        return float(s)
    except Exception:
        return None

def _to_period_yyyy_mm(date_str: str):
    """
    Accepts date like '1968-04-01' or '1968-04' and returns 'YYYY-MM' or None.
    """
    s = (date_str or "").strip()
    if not s:
        return None
    # common: YYYY-MM-DD
    m = re.match(r"^(\d{4})-(\d{2})", s)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}"

def _load_hpi_csv_rows(path: str):
    """
    Returns: (by_area_code: dict[str, list[dict]], by_region_name: dict[str, list[dict]])
    Cached in-process so we don't re-read per request.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        return {}, {}

    by_code = {}
    by_region = {}

    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}, {}
        # map headers to canonical
        field_map = {name: _normalise_csv_header(name) for name in reader.fieldnames}

        for raw in reader:
            row = {field_map.get(k, k): v for k, v in (raw or {}).items()}
            period = _to_period_yyyy_mm(row.get("date") or row.get("date_"))
            if not period:
                continue
            area_code = (row.get("area_code") or "").strip()
            region_name = (row.get("region_name") or "").strip()

            avg_price = _safe_float(row.get("average_price"))
            annual_change = _safe_float(row.get("annual_change"))
            monthly_change = _safe_float(row.get("monthly_change"))
            avg_price_sa = _safe_float(row.get("average_price_sa"))

            rec = {
                "period": period,
                "avg_price": avg_price,
                "avg_price_sa": avg_price_sa,
                "annual_change": annual_change,
                "monthly_change": monthly_change,
                "area_code": area_code,
                "region_name": region_name,
            }

            if area_code:
                by_code.setdefault(area_code, []).append(rec)
            if region_name:
                by_region.setdefault(region_name.lower(), []).append(rec)

    # sort oldest->newest for determinism
    for d in (by_code, by_region):
        for k, rows in d.items():
            rows.sort(key=lambda r: r.get("period") or "")

    return by_code, by_region

# simple in-process cache
_HPI_CSV_CACHE = {"path": None, "by_code": None, "by_region": None}

def _get_hpi_csv_index():
    global _HPI_CSV_CACHE
    path = HPI_CSV_PATH
    if _HPI_CSV_CACHE["path"] == path and _HPI_CSV_CACHE["by_code"] is not None:
        return _HPI_CSV_CACHE["by_code"], _HPI_CSV_CACHE["by_region"]

    by_code, by_region = _load_hpi_csv_rows(path)
    _HPI_CSV_CACHE = {"path": path, "by_code": by_code, "by_region": by_region}
    return by_code, by_region

def _build_trends_from_csv(area_code: str = "", region_name: str = ""):
    by_code, by_region = _get_hpi_csv_index()

    rows = []
    if area_code:
        rows = by_code.get(area_code, []) or []
    if (not rows) and region_name:
        rows = by_region.get(region_name.strip().lower(), []) or []

    # If we still have nothing, fall back to England if present (better than empty).
    if (not rows) and ("england" in by_region):
        rows = by_region.get("england", []) or []

    # Build annual % change series (QoY proxy for "price_change_pct")
    series = []
    for r in rows:
        chg = r.get("annual_change")
        period = r.get("period")
        if period and isinstance(chg, (int, float)):
            series.append({"period": period, "price_change_pct": float(chg)})

    # keep last 120 months for payload size sanity (10y)
    if len(series) > 120:
        series = series[-120:]

    # Snapshot
    latest = series[-1]["price_change_pct"] if series else None
    trend = "Stable"
    if isinstance(latest, (int, float)):
        if latest > 1:
            trend = "Increasing"
        elif latest < -1:
            trend = "Decreasing"

    commentary = (
        f"Annual % change derived from UK HPI CSV ({Path(HPI_CSV_PATH).name})."
        if series
        else f"UK HPI CSV loaded ({Path(HPI_CSV_PATH).name}) but no usable annual change points for this area."
    )

    return {
        "confidenceValue": 0,  # numeric evidence exists; confidence is a label upstream, not a hide gate
        "returnedAtISO": datetime.utcnow().isoformat() + "Z",
        "signals": {
            "priceGrowth": {
                "trend": trend,
                "percentage": (f"{latest:.2f}%" if isinstance(latest, (int, float)) else ""),
                "commentary": commentary,
                "historicalData": series,
            },
            "rentalDemand": {
                "trend": "Medium",
                "commentary": "Rental demand series not wired in (CSV option 1 is price only).",
                "historicalData": [],
            },
            "futureOutlook": {
                "prediction": "Positive" if trend == "Increasing" else "Negative" if trend == "Decreasing" else "Neutral",
                "commentary": "Rule-of-thumb outlook from annual change direction (replace with model later).",
            },
            "notes": commentary,
        },
        "status": "ok" if len(series) >= 2 else "snapshot",
        "source": "hpi_csv",
    }



# ----------------------------
# Private Rents (YoY) optional series (Card 2)
# ----------------------------
# If you provide a Supabase RPC named `rpc_private_rents_yoy_series` that returns rows with:
#   - period (date or YYYY-MM)
#   - annual_change (YoY %, numeric)
# this backend will automatically populate the `rentalDemand.historicalData` series for Card 2.
#
# If the RPC doesn't exist (or fails), Card 2 remains empty and the UI will show "Series: none returned".
PRIVATE_RENTS_RPC_NAME = os.getenv("PRIVATE_RENTS_RPC_NAME", "rpc_private_rents_yoy_series").strip()

# Optional direct table fallback (if the data is loaded into a table rather than exposed via an RPC)
PRIVATE_RENTS_TABLE = os.getenv("PRIVATE_RENTS_TABLE", "private_rents_yoy").strip()

def _fetch_private_rents_yoy_series(area_code: str, months: int) -> List[Dict[str, Any]]:
    """
    Card 2 (Rent Growth YoY): pull private rent YoY % series from Supabase table `ons_private_rents_yoy`.

    Expected columns in the table (as loaded by your CSV):
      - date (date)
      - rent_yoy_pct (numeric)
      - avg_rent_gbp (numeric)
      - region (text)   (may be 'uk', 'england', etc.)

    Some deployments store an admin-code instead (e.g., 'area_code'). We defensively try both.
    """
    try:
        sb = supabase  # created once at module import
        if not sb:
            return []

        # Try to filter by area_code first (if column exists), else fall back to region.
        # Supabase will return a 400 if we reference a non-existent column; catch and retry.
        def _query(filter_col: str) -> List[Dict[str, Any]]:
            q = (
                sb.table("ons_private_rents_yoy")
                .select("date,region,avg_rent_gbp,rent_yoy_pct")
                .eq(filter_col, area_code.lower())
                .order("date", desc=True)
                .limit(months)
            )
            res = q.execute()
            rows = getattr(res, "data", None) or []
            return rows

        rows: List[Dict[str, Any]] = []
        try:
            rows = _query("area_code")
        except Exception:
            rows = _query("region")

        if not rows:
            # final fallback: UK aggregate (keeps card stable if caller passes a LAD code)
            try:
                rows = (
                    sb.table("ons_private_rents_yoy")
                    .select("date,region,avg_rent_gbp,rent_yoy_pct")
                    .eq("region", "uk")
                    .order("date", desc=True)
                    .limit(months)
                    .execute()
                ).data or []
            except Exception:
                rows = []

        rows = list(reversed(rows))  # oldest -> newest

        series: List[Dict[str, Any]] = []
        for r in rows:
            dt = r.get("date")
            yoy = _safe_float(r.get("rent_yoy_pct"))
            avg = _safe_float(r.get("avg_rent_gbp"))

            if dt is None or yoy is None:
                continue

            # Frontend expects `value` for charts/tables (like card 1), so provide it.
            series.append(
                {
                    "period": str(dt)[:10],  # YYYY-MM-DD
                    "value": yoy,
                    "rent_yoy_pct": yoy,
                    "avg_rent_gbp": avg,
                    "region": (r.get("region") or "").lower() or "unknown",
                    "source": "ONS private rents (YoY)",
                }
            )

        return series
    except Exception:
        return []

def build_trends_from_uk_hpi(
    postcode: str,
    area_code: str,
    months: int,
    property_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the `trends` payload (time series) from Supabase UK HPI RPCs.

    - Always returns a dict.
    - If RPC fails or returns < 2 usable points, falls back to guaranteed trends.
    """
    pc = normalize_postcode(postcode or "")
    ac = (area_code or "").strip()
    m = int(months) if isinstance(months, int) and months > 0 else 24
    m = max(2, min(m, 240))

    # Fallback helper
    def _fallback(reason: str) -> Dict[str, Any]:
        # Option 1 (CSV) takes priority when present: it provides real numeric series.
        try:
            csv_out = _build_trends_from_csv(area_code=ac)
            # accept even snapshot (>=1 point), but charts need >=2
            if isinstance(csv_out, dict) and isinstance((csv_out.get("signals") or {}).get("priceGrowth", {}).get("historicalData"), list):
                # annotate why we fell back (without suppressing UI)
                try:
                    (csv_out.setdefault("signals", {}).setdefault("notes", ""))
                    csv_out["signals"]["notes"] = f"{csv_out['signals'].get('notes','').strip()} (fallback: {reason})".strip()
                except Exception:
                    pass
                return csv_out
        except Exception:
            pass

        # Guaranteed fallback (last resort)
        if callable(get_guaranteed_market_trends):
            out = get_guaranteed_market_trends(pc)
            try:
                out["summary"] = f"{out.get('summary','').strip()} ({reason})".strip()
            except Exception:
                pass
            return out
        return {
            "status": "unavailable",
            "summary": reason,
            "confidenceValue": 0.0,
            "signals": None,
            "source": "none",
            "retrievedAtISO": now_iso(),
        }

    if not supabase:
        return _fallback("Supabase not configured")
    if not ac:
        return _fallback("No area_code provided")

    fn = "rpc_uk_hpi_series"
    params: Dict[str, Any] = {"p_area_code": ac, "p_months": m}
    src = "hpi_area"

    pt = (property_type or "").strip()
    if pt:
        fn = "rpc_uk_hpi_series_by_type"
        params = {"p_area_code": ac, "p_property_type": pt, "p_months": m}
        src = "hpi_area_by_type"

    try:
        res = supabase.rpc(fn, params).execute()
        rows = res.data if hasattr(res, "data") else None
        if not isinstance(rows, list):
            rows = []
    except Exception as e:
        return _fallback(f"HPI RPC failed: {str(e)}")

    # Convert to series (oldest->newest) using annual_change as price_change_pct
    series: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        period = _to_ym(r.get("period"))
        yoy = safe_float(r.get("annual_change"))
        if period and yoy is not None:
            v = float(yoy)
            # Include a generic `value` field to maximize frontend compatibility.
            series.append({"period": period, "price_change_pct": v, "value": v})

    series.sort(key=lambda x: x["period"])

    if len(series) < 2:
        return _fallback("Insufficient HPI series points")

    latest_yoy = safe_float(series[-1].get("price_change_pct"))
    trend = _trend_from_yoy(latest_yoy)

    # Start from guaranteed payload so rentalDemand/futureOutlook always exist.
    base = get_guaranteed_market_trends(pc) if callable(get_guaranteed_market_trends) else {
        "status": "ok",
        "summary": "Market trends provided.",
        "confidenceValue": 0.95,
        "signals": {
            "priceGrowth": {},
            "rentalDemand": {},
            "futureOutlook": {},
        },
        "source": "national",
    }

    base["status"] = "ok"
    base["summary"] = "Local UK HPI trends provided (area-level series)."
    # INVARIANT: no hardcoded confidenceValue. Derive from series quality.
    _series_recency_months = 0
    try:
        from datetime import datetime as _dtt
        _latest_period = series[-1]["period"] if series else ""
        if _latest_period:
            _lp = _dtt.strptime(_latest_period + "-01", "%Y-%m-%d")
            _series_recency_months = max(0, (_dtt.utcnow() - _lp).days // 30)
    except Exception:
        pass
    _hpi_conf = round(min(0.55, max(0.15,
        0.55 - max(0, _series_recency_months - 3) * 0.01  # decay by 1% per month stale
    )), 2)
    base["confidenceValue"] = _hpi_conf
    base["confidence_basis"] = "hpi_series_recency_heuristic_uncalibrated"
    base["is_synthetic"] = False
    base["ui_display_permitted"] = True
    base.pop("suppression_reason", None)  # clear synthetic marker when real data present
    base["source"] = src
    # Reset futureOutlook — synthetic narrative must not persist when real data exists
    base.setdefault("signals", {})["futureOutlook"] = {
        "rating": "Neutral",
        "narrative": "Future outlook based on HPI trend direction only. Not a forecast.",
        "historicalData": [],
        "source": "Land Registry HPI (area-level series)",
    }
    base["retrievedAtISO"] = now_iso()
    base.setdefault("signals", {})
    base["signals"]["priceGrowth"] = {
        "trend": trend,
        "percentage": f"{float(latest_yoy):.2f}%" if latest_yoy is not None else "0.00%",
        "commentary": "UK HPI annual change series (area-level).",
        "historicalData": series,
    }
    # Card 2: Private rents (YoY) series if available
    try:
        rent_series = _fetch_private_rents_yoy_series(ac, m)
        if isinstance(rent_series, list) and len(rent_series) >= 2:
            latest_r = safe_float(rent_series[-1].get("value"))
            base["signals"]["rentalDemand"] = {
                "trend": "Medium",
                "commentary": "Private rents YoY series (area-level), via Supabase RPC.",
                "historicalData": rent_series,
            }
            if latest_r is not None:
                base["signals"]["rentalDemand"]["percentage"] = f"{float(latest_r):.2f}%"
    except Exception:
        pass

    return base


@app.after_request
def inject_market_trends(response):
    try:
        # Trigger for BOTH routes: /market-insights and /market_insights (with or without trailing slash)
        p = (request.path or "").rstrip("/")
        if not (p.endswith("/market-insights") or p.endswith("/market_insights")):
            return response

        if not getattr(response, "is_json", False):
            return response

        payload = response.get_json(silent=True)
        if not isinstance(payload, dict):
            return response

        # Ensure both snapshot cards and historical series are always present when available.
        # We MUST not leave the UI with trends === null / missing.
        if ("marketTrends" not in payload) or (payload.get("marketTrends") is None) or ("trends" not in payload):
            payload = ensure_market_trends(payload)
            response.set_data(json.dumps(payload))
            response.mimetype = "application/json"

        return response

    except Exception:
        return response



def _http_get_json(url: str, params: Optional[dict] = None, headers: Optional[dict] = None, timeout: int = 20) -> Tuple[int, Any]:
    h = {"User-Agent": HTTP_USER_AGENT}
    if headers:
        h.update(headers)
    r = requests.get(url, params=params or {}, headers=h, timeout=timeout)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, None


def _http_get_json_raw(url: str, params: Optional[dict] = None, timeout: int = 20) -> Any:
    h = {"User-Agent": HTTP_USER_AGENT}
    r = requests.get(url, params=params or {}, headers=h, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _http_post_text(url: str, data: bytes, headers: Optional[dict] = None, timeout: int = 30) -> Tuple[int, str]:
    h = {"User-Agent": HTTP_USER_AGENT}
    if headers:
        h.update(headers)
    r = requests.post(url, data=data, headers=h, timeout=timeout)
    return r.status_code, r.text or ""


def _norm_geocode_query(s: str) -> str:
    s = (s or "").strip().upper()
    s = re.sub(r"\s+", " ", s)
    return s

def _google_geocode_one(query: str, api_key: str) -> Dict[str, Any]:
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    r = requests.get(url, params={"address": query, "key": api_key}, timeout=12)
    r.raise_for_status()
    data = r.json()

    status = data.get("status")
    if status != "OK":
        return {"ok": False, "error": f"google_status={status}"}

    results = data.get("results") or []
    if not results:
        return {"ok": False, "error": "no_results"}

    loc = (results[0].get("geometry") or {}).get("location") or {}
    lat = loc.get("lat")
    lng = loc.get("lng")
    if lat is None or lng is None:
        return {"ok": False, "error": "missing_lat_lng"}

    return {"ok": True, "lat": float(lat), "lng": float(lng)}


def _row_has_latlng(r: Dict[str, Any]) -> bool:
    lat = safe_float(r.get("lat"))
    lng = safe_float(r.get("lng"))
    return (lat is not None) and (lng is not None)


def _build_comp_geocode_query(r: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k in ("address", "town", "postcode"):
        v = r.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return _norm_geocode_query(", ".join(parts))


def _enrich_housing_rows_with_latlng(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    meta = {
        "enabled": HOUSING_ENRICH_LATLNG,
        "attempted": 0,
        "filled": 0,
        "failed": 0,
        "notes": "",
    }

    if not HOUSING_ENRICH_LATLNG:
        meta["notes"] = "Enrichment disabled (HOUSING_ENRICH_LATLNG=0)."
        return rows, meta

    if not rows or not isinstance(rows, list):
        meta["notes"] = "No rows to enrich."
        return rows, meta

    if not GOOGLE_MAPS_API_KEY:
        meta["notes"] = "GOOGLE_MAPS_API_KEY not set; cannot enrich."
        return rows, meta

    if not supabase:
        meta["notes"] = "Supabase not configured; cannot use geocode cache."
        return rows, meta

    queries: List[str] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if _row_has_latlng(r):
            continue
        q = _build_comp_geocode_query(r)
        if not q:
            continue
        queries.append(q)

    if not queries:
        meta["notes"] = "All rows already contain lat/lng (or lacked address data)."
        return rows, meta

    seen = set()
    uniq_queries: List[str] = []
    for q in queries:
        if q in seen:
            continue
        uniq_queries.append(q)
        seen.add(q)

    uniq_queries = uniq_queries[:max(1, HOUSING_ENRICH_BATCH_LIMIT)]
    meta["attempted"] = len(uniq_queries)

    cached_map: Dict[str, Dict[str, Any]] = {}
    try:
        res = supabase.table(GEOCODE_CACHE_TABLE).select("query,lat,lng").in_("query", uniq_queries).execute()
        rows_cached = res.data if hasattr(res, "data") else []
        if isinstance(rows_cached, list):
            for row in rows_cached:
                if isinstance(row, dict) and row.get("query"):
                    cached_map[str(row["query"]).upper()] = row
    except Exception:
        cached_map = {}

    resolved: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    for q in uniq_queries:
        qk = q.upper()
        hit = cached_map.get(qk)
        if hit:
            lat = safe_float(hit.get("lat"))
            lng = safe_float(hit.get("lng"))
            if lat is not None and lng is not None:
                resolved[q] = (lat, lng)
                meta["filled"] += 1
                continue

        try:
            g = _google_geocode_one(q, GOOGLE_MAPS_API_KEY)
            if not g.get("ok"):
                meta["failed"] += 1
                continue

            lat = safe_float(g.get("lat"))
            lng = safe_float(g.get("lng"))
            if lat is None or lng is None:
                meta["failed"] += 1
                continue

            try:
                supabase.table(GEOCODE_CACHE_TABLE).upsert(
                    {"query": q, "lat": lat, "lng": lng, "provider": "google"},
                    on_conflict="query",
                ).execute()
            except Exception:
                pass

            resolved[q] = (lat, lng)
            meta["filled"] += 1
            time.sleep(0.05)

        except Exception:
            meta["failed"] += 1

    for r in rows:
        if not isinstance(r, dict):
            continue
        if _row_has_latlng(r):
            continue
        q = _build_comp_geocode_query(r)
        if not q:
            continue
        if q in resolved:
            lat, lng = resolved[q]
            r["lat"] = lat
            r["lng"] = lng

    meta["notes"] = "Enrichment completed (cache + google)."
    return rows, meta


@app.route("/adapters/geocode/batch", methods=["POST"])
def adapter_geocode_batch():
    payload = request.get_json(silent=True) or {}
    queries = payload.get("queries") or []
    if not isinstance(queries, list):
        return jsonify({"status": "error", "error": "queries must be a list"}), 400

    queries = [q for q in queries if isinstance(q, str) and q.strip()]
    queries = queries[:max(1, GEOCODE_BATCH_LIMIT)]

    if not GOOGLE_MAPS_API_KEY:
        return jsonify({"status": "error", "error": "GOOGLE_MAPS_API_KEY not set"}), 500
    if not supabase:
        return jsonify({"status": "error", "error": "Supabase not configured (geocode cache requires Supabase)"}), 500
    if not queries:
        return jsonify({"status": "ok", "results": [], "failed": []})

    normalized = [_norm_geocode_query(q) for q in queries]
    seen = set()
    normalized_unique: List[str] = []
    for q in normalized:
        if q in seen:
            continue
        normalized_unique.append(q)
        seen.add(q)

    results: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    cached_map: Dict[str, Dict[str, Any]] = {}
    try:
        res = supabase.table(GEOCODE_CACHE_TABLE).select("query,lat,lng").in_("query", normalized_unique).execute()
        rows = res.data if hasattr(res, "data") else []
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and row.get("query"):
                    cached_map[str(row["query"]).upper()] = row
    except Exception:
        cached_map = {}

    for q in normalized_unique:
        hit = cached_map.get(q.upper())
        if hit:
            results.append({"query": q, "lat": hit.get("lat"), "lng": hit.get("lng"), "cached": True})
            continue

        try:
            g = _google_geocode_one(q, GOOGLE_MAPS_API_KEY)
            if not g.get("ok"):
                failed.append({"query": q, "error": g.get("error", "geocode_failed")})
                continue

            lat = g["lat"]
            lng = g["lng"]

            try:
                supabase.table(GEOCODE_CACHE_TABLE).upsert(
                    {"query": q, "lat": lat, "lng": lng, "provider": "google"},
                    on_conflict="query",
                ).execute()
            except Exception:
                pass

            results.append({"query": q, "lat": lat, "lng": lng, "cached": False})
            time.sleep(0.05)
        except Exception as e:
            failed.append({"query": q, "error": f"exception:{str(e)}"})

    return jsonify({"status": "ok", "results": results, "failed": failed})

def resolve_lsoa_gss_from_postcode(postcode: str) -> Tuple[Optional[str], Dict[str, Any]]:
    retrieved = now_iso()
    pc = normalize_postcode(postcode)
    pc_key = normalize_postcode_nospace(pc)

    meta = {
        "provider": "postcodes.io",
        "retrievedAtISO": retrieved,
        "postcode": pc,
        "sources": [{"label": "postcodes.io", "url": "https://postcodes.io/"}],
        "notes": "",
        "cache": {"hit": False, "ttlSeconds": POSTCODES_IO_CACHE_TTL_SECONDS},
        "lat": None,
        "lng": None,
    }

    if not pc_key:
        meta["notes"] = "No postcode provided."
        return None, meta

    cache_key = f"geo::{pc_key}"
    cached = geo_cache_get(cache_key)
    if isinstance(cached, dict):
        lsoa_cached = cached.get("lsoa_gss")
        if isinstance(lsoa_cached, str) and lsoa_cached.strip():
            meta["cache"]["hit"] = True
            meta["notes"] = "Resolved from geo cache."
            meta["lat"] = safe_float(cached.get("lat"))
            meta["lng"] = safe_float(cached.get("lng"))
            meta["area_code"] = (cached.get("area_code") or "").strip() or None
            return lsoa_cached.strip(), meta

    url = f"https://api.postcodes.io/postcodes/{pc_key}"
    try:
        status, payload = _http_get_json(url, timeout=POSTCODES_IO_TIMEOUT)

        if isinstance(payload, dict) and isinstance(payload.get("status"), int) and payload.get("status") != 200:
            meta["notes"] = f"postcodes.io payload status {payload.get('status')}: {payload.get('error', '')}".strip()
            return None, meta

        if status != 200 or not isinstance(payload, dict):
            meta["notes"] = f"postcodes.io returned HTTP {status}"
            return None, meta

        result = payload.get("result") if isinstance(payload.get("result"), dict) else None
        if not isinstance(result, dict):
            meta["notes"] = "postcodes.io returned no result object."
            return None, meta

        meta["lat"] = safe_float(result.get("latitude"))
        meta["lng"] = safe_float(result.get("longitude"))

        codes = result.get("codes") if isinstance(result.get("codes"), dict) else {}
        lsoa_gss = codes.get("lsoa")
        lsoa_gss = lsoa_gss.strip() if isinstance(lsoa_gss, str) and lsoa_gss.strip() else None
        # Admin district code (LAD) used by UK HPI tables (e.g., E06000001).
        area_code = (codes.get("admin_district") or "").strip() if isinstance(codes, dict) else ""
        meta["area_code"] = area_code or None

        if not lsoa_gss:
            meta["notes"] = "postcodes.io result missing codes.lsoa (GSS)."
            return None, meta

        geo_cache_set(cache_key, {"lsoa_gss": lsoa_gss, "lat": meta["lat"], "lng": meta["lng"], "area_code": area_code or ""})
        meta["notes"] = "Resolved LSOA GSS (+ coords) from postcodes.io."
        return lsoa_gss, meta

    except Exception as e:
        meta["notes"] = f"postcodes.io exception: {str(e)}"
        return None, meta


def fetch_nomis_jsonstat(dataset_id: str, params: dict) -> dict:
    base = f"https://www.nomisweb.co.uk/api/v01/dataset/{dataset_id}.jsonstat.json"
    payload = _http_get_json_raw(base, params=params, timeout=NOMIS_TIMEOUT)

    if not isinstance(payload, dict):
        raise ValueError("Nomis payload is not a dict")

    if payload.get("class") == "dataset":
        return {"dataset": payload}

    if "dataset" in payload:
        return payload

    raise ValueError(f"Nomis returned unsupported JSON-stat shape: keys={list(payload.keys())}")


def parse_jsonstat_single_dimension(jsonstat: dict) -> Dict[str, Any]:
    ds = jsonstat.get("dataset") or {}
    dim = ds.get("dimension") or {}

    dim_ids = ds.get("id") or dim.get("id") or []
    if not isinstance(dim_ids, list) or not dim_ids:
        if isinstance(dim, dict):
            dim_ids = [k for k in dim.keys() if isinstance(k, str)]
        else:
            dim_ids = []

    exclude = {"date", "time", "geography", "measures", "freq"}
    candidate_dims = [d for d in dim_ids if isinstance(d, str) and d.lower() not in exclude]
    main_dim = candidate_dims[0] if candidate_dims else None

    if not main_dim and isinstance(dim, dict):
        for d in dim_ids:
            if not isinstance(d, str) or d.lower() in exclude:
                continue
            d_obj = dim.get(d)
            cat = (d_obj or {}).get("category") if isinstance(d_obj, dict) else None
            if isinstance(cat, dict) and isinstance(cat.get("index"), (list, dict)):
                main_dim = d
                break

    if not main_dim:
        raise ValueError(
            f"Could not infer main dimension from JSON-stat. dim_ids={dim_ids}, dim_keys={list(dim.keys()) if isinstance(dim, dict) else type(dim)}"
        )

    d_obj = dim.get(main_dim) if isinstance(dim, dict) else {}
    cat = (d_obj or {}).get("category") if isinstance(d_obj, dict) else {}
    labels = cat.get("label") or {}
    index = cat.get("index")

    if isinstance(index, list):
        codes = index
    elif isinstance(index, dict):
        codes = [k for k, _ in sorted(index.items(), key=lambda kv: kv[1])]
    else:
        codes = list(labels.keys()) if isinstance(labels, dict) else []

    values = ds.get("value")
    if not isinstance(values, list):
        raise ValueError("JSON-stat 'value' is not a list (unexpected for this query shape).")

    items = []
    total_val = 0
    for i, code in enumerate(codes):
        lab = labels.get(code) if isinstance(labels, dict) else str(code)
        v = values[i] if i < len(values) else None
        iv = safe_int(v)
        if iv is None:
            iv = 0
        items.append({"code": str(code), "label": str(lab) if lab is not None else str(code), "value": iv})
        total_val += iv

    return {"items": items, "total": total_val, "dimensionId": main_dim}


def get_nomis_table(label: str, dimension: str, categories: str, geography: str) -> Dict[str, Any]:
    retrieved = now_iso()
    sources = [{"label": "Nomis API (ONS)", "url": "https://www.nomisweb.co.uk/api/v01/help"}]

    if not NOMIS_ENABLED:
        return metric_missing_provider("Nomis disabled. Set NOMIS_ENABLED=1.", sources, retrieved)

    if not geography:
        return metric_unavailable(
            "Nomis requires a geography id. Provide a numeric geography id or set NOMIS_DEFAULT_GEOGRAPHY.",
            sources,
            retrieved,
        )

    if not is_digits_only(geography):
        return metric_unavailable(
            "Nomis geography must be a numeric Nomis geography id for this dataset (NM_2023_1).",
            sources,
            retrieved,
            extra_metrics={"label": label, "geography": geography},
        )

    if not dimension or not categories:
        return metric_missing_provider(
            f"{label} not configured. Set env for its dimension/categories (e.g. NOMIS_TS054_DIM and NOMIS_TS054_CATS).",
            sources,
            retrieved,
            extra_metrics={"label": label, "dimension": dimension, "categories": categories, "geography": geography},
        )

    if "..." in categories or "…" in categories:
        return metric_unavailable(
            f"{label} categories contain an ellipsis (truncated copy). Use full category list (no ...).",
            sources,
            retrieved,
            extra_metrics={"dimension": dimension, "categories": categories},
        )

    try:
        params = {
            "date": "latest",
            "geography": geography,
            "freq": NOMIS_FREQ,
            dimension: categories,
            "measures": "20100",
        }
        js = fetch_nomis_jsonstat(NOMIS_DATASET_ID, params)
        parsed = parse_jsonstat_single_dimension(js)

        bullets = [f"• {it['label']}: {it['value']}" for it in parsed["items"]]
        summary = f"{label} (Nomis) — total: {parsed['total']}"

        out = metric_ok(summary, bullets, sources, retrieved, 0.92)
        out["metrics"] = {
            "provider": "nomis",
            "dataset": NOMIS_DATASET_ID,
            "label": label,
            "geography": geography,
            "dimensionId": parsed.get("dimensionId"),
            "total": parsed.get("total"),
            "items": parsed.get("items"),
            "params": params,
        }
        return out

    except Exception as e:
        return metric_unavailable(f"{label} fetch/parse failed: {str(e)}", sources, retrieved)


def map_property_type_label(v: Any) -> str:
    s = (str(v or "").strip())
    if not s:
        return ""
    up = s.upper()
    if up == "D":
        return "Detached"
    if up == "S":
        return "Semi-detached"
    if up == "T":
        return "Terraced"
    if up == "F":
        return "Flat/Maisonette"
    if up == "O":
        return "Other"

    low = s.lower()
    if "semi" in low:
        return "Semi-detached"
    if "terr" in low:
        return "Terraced"
    if "detach" in low and "semi" not in low:
        return "Detached"
    if "flat" in low or "maison" in low:
        return "Flat/Maisonette"

    return "Other"


def build_housing_charts_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    price_bins = [
        ("<£100k", 0, 100_000),
        ("£100–150k", 100_000, 150_000),
        ("£150–200k", 150_000, 200_000),
        ("£200–300k", 200_000, 300_000),
        ("£300k+", 300_000, None),
    ]
    dist_bins = [
        ("0–0.5 mi", 0.0, 0.5),
        ("0.5–1 mi", 0.5, 1.0),
        ("1–2 mi", 1.0, 2.0),
        ("2–3 mi", 2.0, 3.0),
        ("3+ mi", 3.0, None),
    ]

    price_counts = {lab: 0 for (lab, _, _) in price_bins}
    dist_counts = {lab: 0 for (lab, _, _) in dist_bins}

    type_counts: Dict[str, int] = {
        "Terraced": 0,
        "Semi-detached": 0,
        "Detached": 0,
        "Flat/Maisonette": 0,
        "Other": 0,
    }

    for r in rows:
        if not isinstance(r, dict):
            continue
        p = safe_int(r.get("price"))
        m = safe_float(r.get("miles"))
        t = map_property_type_label(r.get("property_type"))

        if isinstance(p, int):
            for lab, lo, hi in price_bins:
                if hi is None and p >= lo:
                    price_counts[lab] += 1
                    break
                if hi is not None and lo <= p < hi:
                    price_counts[lab] += 1
                    break

        if isinstance(m, float):
            for lab, lo, hi in dist_bins:
                if hi is None and m >= lo:
                    dist_counts[lab] += 1
                    break
                if hi is not None and lo <= m < hi:
                    dist_counts[lab] += 1
                    break

        if t:
            type_counts[t] = type_counts.get(t, 0) + 1

    charts = {
        "priceBands": {
            "title": "Price bands (sold comps)",
            "bins": [{"label": lab, "value": int(price_counts[lab])} for (lab, _, _) in price_bins],
        },
        "propertyTypes": {
            "title": "Property types (sold comps)",
            "bins": [
                {"label": lab, "value": int(type_counts.get(lab, 0))}
                for lab in ["Terraced", "Semi-detached", "Detached", "Flat/Maisonette", "Other"]
            ],
        },
        "distanceBands": {
            "title": "Distance bands (sold comps)",
            "bins": [{"label": lab, "value": int(dist_counts[lab])} for (lab, _, _) in dist_bins],
        },
    }
    return charts


def _parse_date_any(v: Any) -> Optional[datetime]:
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1]
        s = s.replace(" ", "T")
        return datetime.fromisoformat(s)
    except Exception:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except Exception:
            return None


def _evidence_grade(n: int) -> str:
    if n >= 20:
        return "strong"
    if n >= 10:
        return "moderate"
    if n >= 3:
        return "thin"
    return "minimal"


# -------------------------------
# Trend Card #1 (GATED, EVIDENCE-FIRST)
# Pricing Power (Sold Comps Momentum)
# -------------------------------

def _quantile_float(values: List[float], q: float) -> Optional[float]:
    vs = sorted([float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))])
    if not vs:
        return None
    q = max(0.0, min(1.0, float(q)))
    idx = int(round((len(vs) - 1) * q))
    idx = max(0, min(len(vs) - 1, idx))
    return float(vs[idx])


def _median_float(values: List[float]) -> Optional[float]:
    vs = sorted([float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))])
    if not vs:
        return None
    mid = len(vs) // 2
    if len(vs) % 2 == 1:
        return float(vs[mid])
    return float((vs[mid - 1] + vs[mid]) / 2.0)


def _ols_slope(xs: List[float], ys: List[float]) -> Optional[float]:
    if not xs or not ys or len(xs) != len(ys) or len(xs) < 3:
        return None
    x_mean = sum(xs) / float(len(xs))
    y_mean = sum(ys) / float(len(ys))
    num = 0.0
    den = 0.0
    for x, y in zip(xs, ys):
        dx = x - x_mean
        num += dx * (y - y_mean)
        den += dx * dx
    if den == 0.0:
        return None
    return num / den


def _bootstrap_slope_ci_width(xs: List[float], ys: List[float], iters: int = 200) -> Optional[float]:
    n = len(xs)
    if n < 10:
        return None

    seed = int(n * 1000 + (xs[0] if xs else 0.0))
    rnd = random.Random(seed)
    slopes: List[float] = []
    iters = max(50, int(iters))

    for _ in range(iters):
        bx: List[float] = []
        by: List[float] = []
        for _j in range(n):
            i = rnd.randrange(0, n)
            bx.append(xs[i])
            by.append(ys[i])
        s = _ols_slope(bx, by)
        if s is not None and math.isfinite(float(s)):
            slopes.append(float(s))

    if len(slopes) < 30:
        return None

    lo = _quantile_float(slopes, 0.05)
    hi = _quantile_float(slopes, 0.95)
    if lo is None or hi is None:
        return None
    return float(hi - lo)


def build_pricing_power_sold_comps_momentum(rows: List[Dict[str, Any]], radius_miles: Optional[float]) -> Dict[str, Any]:
    retrieved = now_iso()
    window_months = 12

    obs: List[Tuple[datetime, float, str]] = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        dt = _parse_date_any(r.get("date"))
        pr = safe_float(r.get("price"))
        if dt is None or pr is None or pr <= 0:
            continue
        cid = str(r.get("id") or r.get("uid") or r.get("transaction_id") or r.get("uprn") or "").strip()
        obs.append((dt, float(pr), cid))

    if not obs:
        return {
            "status": "unavailable",
            "confidenceValue": 0.0,
            "reason": "No dated sold comps available for momentum analysis.",
            "window": {"months": window_months, "radiusMiles": radius_miles},
            "sample": {"n": 0, "dateMin": None, "dateMax": None},
            "evidence": {"method": "ols-log-price-regression", "outlierRule": "trim p5..p95", "compsUsed": []},
            "retrievedAtISO": retrieved,
        }

    obs.sort(key=lambda t: t[0])
    date_min = obs[0][0]
    date_max = obs[-1][0]

    cutoff = date_max - timedelta(days=int(window_months * 30.5))
    obs_w = [(d, p, cid) for (d, p, cid) in obs if d >= cutoff]

    if len(obs_w) < 10:
        return {
            "status": "suppressed",
            "confidenceValue": 0.0,
            "reason": f"Insufficient sold comps in last {window_months} months (n={len(obs_w)}).",
            "window": {"months": window_months, "radiusMiles": radius_miles},
            "sample": {"n": len(obs_w), "dateMin": date_min.date().isoformat(), "dateMax": date_max.date().isoformat()},
            "evidence": {"method": "ols-log-price-regression", "outlierRule": "trim p5..p95", "compsUsed": []},
            "retrievedAtISO": retrieved,
        }

    prices = [p for (_d, p, _cid) in obs_w]
    p5 = _quantile_float(prices, 0.05)
    p95 = _quantile_float(prices, 0.95)
    if p5 is None or p95 is None or p95 <= p5:
        p5, p95 = None, None

    filtered: List[Tuple[datetime, float, str]] = []
    for d, p, cid in obs_w:
        if p <= 0:
            continue
        if p5 is not None and p95 is not None:
            if p < p5 or p > p95:
                continue
        filtered.append((d, p, cid))

    n = len(filtered)
    if n < 10:
        return {
            "status": "suppressed",
            "confidenceValue": 0.0,
            "reason": f"Too few observations after outlier trimming (n={n}).",
            "window": {"months": window_months, "radiusMiles": radius_miles},
            "sample": {"n": n, "dateMin": date_min.date().isoformat(), "dateMax": date_max.date().isoformat()},
            "evidence": {"method": "ols-log-price-regression", "outlierRule": "trim p5..p95", "compsUsed": [c for (_d, _p, c) in filtered if c][:250]},
            "retrievedAtISO": retrieved,
        }

    t0 = filtered[0][0]
    xs: List[float] = []
    ys: List[float] = []
    for d, p, _cid in filtered:
        days = float((d - t0).days)
        xs.append(days)
        ys.append(math.log(float(p)))

    slope = _ols_slope(xs, ys)
    if slope is None:
        return {
            "status": "suppressed",
            "confidenceValue": 0.0,
            "reason": "Could not compute momentum slope (degenerate data).",
            "window": {"months": window_months, "radiusMiles": radius_miles},
            "sample": {"n": n, "dateMin": date_min.date().isoformat(), "dateMax": date_max.date().isoformat()},
            "evidence": {"method": "ols-log-price-regression", "outlierRule": "trim p5..p95", "compsUsed": [c for (_d, _p, c) in filtered if c][:250]},
            "retrievedAtISO": retrieved,
        }

    momentum_annualized_pct = (math.exp(float(slope) * 365.0) - 1.0) * 100.0

    last_dt = filtered[-1][0]
    r0 = last_dt - timedelta(days=90)
    p0 = last_dt - timedelta(days=180)
    recent_vals = [p for (d, p, _cid) in filtered if d >= r0]
    prev_vals = [p for (d, p, _cid) in filtered if (d < r0 and d >= p0)]

    med_recent = _median_float(recent_vals)
    med_prev = _median_float(prev_vals)
    recent_shift_pct: Optional[float] = None
    if med_recent is not None and med_prev is not None and med_prev > 0:
        recent_shift_pct = ((med_recent - med_prev) / float(med_prev)) * 100.0

    months_seen = len(set((d.year, d.month) for (d, _p, _cid) in filtered))
    recent_120 = len([1 for (d, _p, _cid) in filtered if d >= (last_dt - timedelta(days=120))])
    ciw = _bootstrap_slope_ci_width(xs, ys, iters=200)

    reasons: List[str] = []
    if n < 40:
        reasons.append(f"n<40 (n={n})")
    if months_seen < 6:
        reasons.append(f"months_seen<6 (months_seen={months_seen})")
    if recent_120 < 10:
        reasons.append(f"recent_120<10 (recent_120={recent_120})")
    if ciw is None:
        reasons.append("bootstrap_ci_unavailable")
    else:
        if float(ciw) > 0.10:
            reasons.append(f"ci_width_too_wide (ciw={float(ciw):.4f})")

    headline = "Flat"
    if momentum_annualized_pct > 5.0:
        headline = "Upward pressure"
    elif momentum_annualized_pct < -5.0:
        headline = "Downward pressure"
    elif abs(momentum_annualized_pct) < 2.0:
        headline = "Flat"
    else:
        headline = "Slightly moving"

    base = {
        "headline": headline,
        "momentumAnnualizedPct": float(momentum_annualized_pct),
        "recentMedianShiftPct": float(recent_shift_pct) if recent_shift_pct is not None else None,
        "unit": "£",
        "window": {"months": window_months, "radiusMiles": radius_miles},
        "sample": {
            "n": int(n),
            "dateMin": date_min.date().isoformat(),
            "dateMax": date_max.date().isoformat(),
            "monthsSeen": int(months_seen),
            "recent120Count": int(recent_120),
        },
        "evidence": {
            "method": "ols-log-price-regression",
            "outlierRule": "trim p5..p95",
            "bootstrapIters": 200,
            "bootstrapSlopeCIWidth": float(ciw) if ciw is not None else None,
            "compsUsed": [c for (_d, _p, c) in filtered if c][:250],
            "notes": [],
        },
        "retrievedAtISO": retrieved,
    }

    if not reasons:
        return {**base, "status": "ok", "confidenceValue": float(MIN_VERIFIED)}

    return {**base, "status": "suppressed", "confidenceValue": 0.0, "reason": "Gating failed: " + ", ".join(reasons)}


def _pricing_power_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    pairs: List[Tuple[datetime, int]] = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        dt = _parse_date_any(r.get("date"))
        pr = safe_int(r.get("price"))
        if dt is None or pr is None:
            continue
        pairs.append((dt, pr))

    if not pairs:
        return {
            "status": "unavailable",
            "trend": "unknown",
            "windowDays": 365,
            "evidenceGrade": "minimal",
            "counts": {"total": 0, "recent": 0, "previous": 0},
            "medians": {"recent": None, "previous": None},
            "pctChange": None,
            "confidence": 0.0,
            "notes": "No dated sale prices available in comps for pricing power analysis.",
        }

    pairs.sort(key=lambda x: x[0])
    latest_dt = pairs[-1][0]
    earliest_dt = pairs[0][0]

    recent_from = latest_dt - timedelta(days=365)
    prev_from = latest_dt - timedelta(days=730)

    recent = [p for (d, p) in pairs if d >= recent_from]
    prev = [p for (d, p) in pairs if (d < recent_from and d >= prev_from)]

    n_total = len(pairs)
    n_recent = len(recent)
    n_prev = len(prev)

    def _median_int(values: List[int]) -> Optional[int]:
        vs = sorted([v for v in values if isinstance(v, int)])
        if not vs:
            return None
        mid = len(vs) // 2
        if len(vs) % 2 == 1:
            return vs[mid]
        return int((vs[mid - 1] + vs[mid]) / 2)

    med_recent = _median_int(recent)
    med_prev = _median_int(prev)

    pct_change: Optional[float] = None
    if isinstance(med_recent, int) and isinstance(med_prev, int) and med_prev > 0:
        pct_change = ((med_recent - med_prev) / float(med_prev)) * 100.0

    trend = "unknown"
    if pct_change is not None:
        if pct_change > 2.0:
            trend = "up"
        elif pct_change < -2.0:
            trend = "down"
        else:
            trend = "flat"

    grade = _evidence_grade(n_total)
    conf = 0.0
    if grade == "strong":
        conf = 0.92
    elif grade == "moderate":
        conf = 0.87
    elif grade == "thin":
        conf = 0.80
    else:
        conf = 0.70

    notes = ""
    if n_prev < 3:
        notes = "Limited comparison window: fewer than 3 sales in the prior 12 months."

    return {
        "status": "ok",
        "trend": trend,
        "windowDays": 365,
        "evidenceGrade": grade,
        "counts": {"total": n_total, "recent": n_recent, "previous": n_prev},
        "medians": {"recent": med_recent, "previous": med_prev},
        "pctChange": pct_change,
        "dateRange": {"earliest": earliest_dt.date().isoformat(), "latest": latest_dt.date().isoformat()},
        "confidence": conf,
        "notes": notes,
    }


def build_market_contract_stub(postcode: str, lat: Optional[float], lng: Optional[float], nomis_geo: str) -> Dict[str, Any]:
    retrieved = now_iso()

    housing_sold = [
        {
            "price": 149950,
            "date": "2025-10-29",
            "property_type": "Terraced",
            "miles": 0.84,
            "address": "PARROT ROW",
            "town": "ABERTILLERY",
            "postcode": "NP13 3AH",
            "lat": None,
            "lng": None,
        },
        {
            "price": 85000,
            "date": "2025-10-24",
            "property_type": "Terraced",
            "miles": 0.20,
            "address": "GLADSTONE STREET",
            "town": "ABERTILLERY",
            "postcode": "NP13 3HJ",
            "lat": None,
            "lng": None,
        },
    ]
    housing_charts = build_housing_charts_from_rows(housing_sold)

    housing_metric = metric_ok(
        summary="Contract mode: deterministic housing comps + charts.",
        value=housing_sold,
        sources=[{"label": "Contract stub", "url": ""}],
        retrieved_at=retrieved,
        confidence=1.0,
    )
    housing_metric["soldComps"] = housing_sold
    housing_metric["charts"] = housing_charts
    housing_metric["metrics"] = housing_metric.get("metrics") or {}
    housing_metric["metrics"]["pricingPower"] = _pricing_power_from_rows(housing_sold)
    housing_metric["metrics"]["pricingPowerSoldCompsMomentum"] = build_pricing_power_sold_comps_momentum(
        housing_sold, HOUSING_DEFAULT_RADIUS_MILES
    )

    census_stub = metric_ok(
        summary="Contract mode: census TS003 placeholder.",
        value=[
            "• One-person household: 0",
            "• Couple household: 0",
            "• Family household: 0",
        ],
        sources=[{"label": "Contract stub", "url": ""}],
        retrieved_at=retrieved,
        confidence=1.0,
    )
    census_stub["metrics"] = {
        "provider": "stub",
        "dataset": "stub",
        "label": "Household composition (TS003)",
        "geography": nomis_geo,
        "dimensionId": "stub_dim",
        "total": 0,
        "items": [],
        "params": {},
    }

    results = {
        "postcode": postcode,
        "location": {
            "lat": lat,
            "lng": lng,
            "geocodeMeta": {"provider": "stub", "notes": "Contract mode: no geocoding performed."},
            "lsoaMeta": {"provider": "stub", "notes": "Contract mode: no LSOA lookup performed."},
            "nomisGeography": nomis_geo,
        },
        "localAreaAnalysis": {
            "retrievedAtISO": retrieved,
            "postcode": postcode,
            "schools": metric_missing_provider("Contract mode: schools not wired.", [{"label": "Contract stub", "url": ""}], retrieved),
            "housing": housing_metric,
            "transport": metric_missing_provider("Contract mode: transport not wired.", [{"label": "Contract stub", "url": ""}], retrieved),
            "amenities": metric_missing_provider("Contract mode: amenities not wired.", [{"label": "Contract stub", "url": ""}], retrieved),
            "crime": metric_missing_provider("Contract mode: crime not wired.", [{"label": "Contract stub", "url": ""}], retrieved),
            "broadband": metric_missing_provider("Contract mode: broadband not wired.", [{"label": "Contract stub", "url": ""}], retrieved),
            "census": {
                "ts003": census_stub,
                "ts044": metric_missing_provider("Contract mode: TS044 not wired.", [{"label": "Contract stub", "url": ""}], retrieved),
                "ts054": metric_missing_provider("Contract mode: TS054 not wired.", [{"label": "Contract stub", "url": ""}], retrieved),
            },
        },
        "comparableProperties": {
            "forSale": [],
            "sourceUrl": "",
            "sources": [],
            "retrievedAtISO": retrieved,
            "confidenceValue": 0.0,
            "status": "missing_provider",
            "summary": "Contract mode: live listings provider not configured.",
        },
    }
    return results


def nspl_lookup_latlng(postcode: str) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    meta = {
        "retrievedAtISO": now_iso(),
        "sources": [],
        "referenceLinks": [],
        "notes": "",
        "provider": "nspl_lookup",
    }

    pc_key = normalize_postcode_nospace(postcode)
    if not pc_key:
        meta["notes"] = "No postcode provided."
        return None, None, meta

    if not supabase:
        meta["notes"] = "Supabase not configured; cannot query nspl_lookup."
        meta["provider"] = "nspl_lookup_unavailable"
        meta["sources"] = [{"label": "Supabase", "url": "https://supabase.com/"}]
        return None, None, meta

    try:
        rows = data_query(
            "SELECT lat, lng FROM public.nspl_postcodes WHERE pcd_nospace = %s LIMIT 1",
            (pc_key,)
        )
        if not isinstance(rows, list) or not rows:
            meta["notes"] = f"NSPL lookup returned no rows for {pc_key}."
            meta["sources"] = [{"label": "Hetzner (nspl_postcodes)", "url": f"{SUPABASE_URL}"}]
            return None, None, meta

        lat = safe_float(rows[0].get("lat"))
        lng = safe_float(rows[0].get("lng"))
        if lat is None or lng is None:
            meta["notes"] = "NSPL lookup returned invalid coordinates."
            meta["sources"] = [{"label": "Hetzner (nspl_postcodes)", "url": f"{SUPABASE_URL}"}]
            return None, None, meta

        meta["notes"] = "Resolved from NSPL."
        meta["sources"] = [{"label": "Hetzner (nspl_postcodes)", "url": f"{SUPABASE_URL}"}]
        return lat, lng, meta

    except Exception as e:
        meta["notes"] = f"NSPL lookup exception: {str(e)}"
        meta["sources"] = [{"label": "Hetzner (nspl_postcodes)", "url": f"{SUPABASE_URL}"}]
        return None, None, meta


def geocode_postcode(postcode: str) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    pc = normalize_postcode(postcode)
    meta = {
        "retrievedAtISO": now_iso(),
        "sources": [{"label": "OpenStreetMap Nominatim", "url": "https://nominatim.openstreetmap.org/"}],
        "referenceLinks": [],
        "notes": "",
        "provider": "nominatim",
    }

    if not pc:
        meta["notes"] = "No postcode provided."
        return None, None, meta

    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": pc, "format": "json", "limit": 1, "addressdetails": 0}
        status, payload = _http_get_json(url, params=params, timeout=15)
        if status != 200:
            meta["notes"] = f"Nominatim error: HTTP {status}"
            return None, None, meta

        if not isinstance(payload, list) or not payload:
            meta["notes"] = "Nominatim returned no results for postcode."
            return None, None, meta

        item = payload[0] or {}
        lat = safe_float(item.get("lat"))
        lng = safe_float(item.get("lon"))
        if lat is None or lng is None:
            meta["notes"] = "Nominatim returned invalid coordinates."
            return None, None, meta

        return lat, lng, meta
    except Exception as e:
        meta["notes"] = f"Nominatim exception: {str(e)}"
        return None, None, meta


def summarise_counts(title: str, counts: Dict[str, int], top_names: Optional[list] = None) -> str:
    parts = []
    for k in sorted(counts.keys()):
        parts.append(f"{k}: {counts[k]}")
    headline = f"{title}: " + (", ".join(parts) if parts else "no results.")
    if top_names:
        names = [n for n in top_names if isinstance(n, str) and n.strip()][:6]
        if names:
            headline += "\n• Examples: " + ", ".join(names)
    return headline


def get_crime_data(lat: Optional[float], lng: Optional[float]) -> Dict[str, Any]:
    retrieved = now_iso()
    docs_url = "https://data.police.uk/docs/"
    base_sources = [{"label": "UK Police Data API docs", "url": docs_url}]

    if lat is None or lng is None:
        return metric_unavailable(
            "Crime data not available: postcode could not be resolved to coordinates.",
            base_sources,
            retrieved,
        )

    url = f"https://data.police.uk/api/crimes-street/all-crime?lat={lat}&lng={lng}"
    try:
        status, crimes = _http_get_json(url, timeout=20)
        if status != 200 or not isinstance(crimes, list):
            crimes = []

        counts: Dict[str, int] = {}
        for c in crimes:
            cat = (c or {}).get("category") or "unknown"
            counts[cat] = counts.get(cat, 0) + 1

        summary = summarise_counts("Crimes (street-level)", counts)
        bounded = crimes[:MAX_CRIMES]

        sources = [
            {"label": "UK Police Data API (crimes-street)", "url": url},
            {"label": "UK Police Data API docs", "url": docs_url},
        ]

        out = metric_ok(
            summary if crimes else "No crime records returned for this location/time window.",
            bounded,
            sources,
            retrieved,
            MIN_VERIFIED if len(crimes) > 0 else 0.0,
        )
        out["metrics"] = {
            "total": len(crimes),
            "categories": counts,
            "radius_hint": "Police API uses a fixed area around the point; see documentation.",
        }
        return out

    except Exception as e:
        return metric_unavailable(
            f"Crime data fetch failed: {str(e)}",
            base_sources,
            retrieved,
        )


def overpass_query(lat: float, lng: float, selectors: str) -> Dict[str, Any]:
    q = f"""
[out:json];
(
  {selectors}
);
out body;
""".strip()

    status, text = _http_post_text(
        "https://overpass-api.de/api/interpreter",
        data=q.encode("utf-8"),
        headers={"Content-Type": "text/plain"},
        timeout=30,
    )
    if status != 200 or not text:
        return {"elements": []}
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
        return {"elements": []}
    except Exception:
        return {"elements": []}


def get_transport_data(lat: Optional[float], lng: Optional[float]) -> Dict[str, Any]:
    retrieved = now_iso()
    base_sources = [
        {"label": "OpenStreetMap (Overpass API)", "url": "https://overpass-api.de/"},
        {"label": "OpenStreetMap", "url": "https://www.openstreetmap.org"},
    ]

    if lat is None or lng is None:
        return metric_unavailable(
            "Transport data not available: postcode could not be resolved to coordinates.",
            base_sources,
            retrieved,
        )

    radius = DEFAULT_OSM_RADIUS
    selectors = f"""
nwr["railway"="station"](around:{radius},{lat},{lng});
nwr["railway"="tram_stop"](around:{radius},{lat},{lng});
nwr["highway"="bus_stop"](around:{radius},{lat},{lng});
nwr["public_transport"="stop_position"](around:{radius},{lat},{lng});
nwr["public_transport"="platform"](around:{radius},{lat},{lng});
""".strip()

    try:
        payload = overpass_query(lat, lng, selectors)
        elements = payload.get("elements", []) if isinstance(payload, dict) else []
        if not isinstance(elements, list):
            elements = []

        counts: Dict[str, int] = {"stations": 0, "tram_stops": 0, "public_transport": 0, "bus_stops": 0}
        named_stations: List[str] = []
        named_tram: List[str] = []
        named_bus: List[str] = []

        for e in elements:
            tags = (e or {}).get("tags") or {}
            if not isinstance(tags, dict):
                continue

            name = tags.get("name")
            nm = name.strip() if isinstance(name, str) and name.strip() else ""

            if tags.get("railway") == "station":
                counts["stations"] += 1
                if nm:
                    named_stations.append(nm)
            elif tags.get("railway") == "tram_stop":
                counts["tram_stops"] += 1
                if nm:
                    named_tram.append(nm)

            # OSM bus stop tagging varies by region:
            # highway=bus_stop (traditional), public_transport=stop_position or platform (modern schema)
            # West Midlands / Birmingham primarily uses public_transport schema
            is_bus = (
                tags.get("highway") == "bus_stop" or
                (tags.get("public_transport") in ("stop_position", "platform") and
                 tags.get("bus") in ("yes", "designated")) or
                (tags.get("public_transport") in ("stop_position", "platform") and
                 tags.get("railway") not in ("station", "tram_stop")) or
                tags.get("highway") == "bus_stop"
            )
            if is_bus:
                counts["bus_stops"] += 1
                if nm:
                    named_bus.append(nm)

        def _dedup(xs: List[str]) -> List[str]:
            seen = set()
            out = []
            for x in xs:
                if x not in seen:
                    out.append(x)
                    seen.add(x)
            return out

        named_stations = _dedup(named_stations)[:6]
        named_tram = _dedup(named_tram)[:6]
        named_bus = _dedup(named_bus)[:8]

        bullets: List[str] = []
        bullets.append(
            f"• Rail: {counts['stations']} station(s) within ~{radius}m"
            + (f" (e.g., {', '.join(named_stations)})" if named_stations else "")
        )
        if counts["tram_stops"] > 0:
            bullets.append(
                f"• Tram: {counts['tram_stops']} stop(s) within ~{radius}m"
                + (f" (e.g., {', '.join(named_tram)})" if named_tram else "")
            )
        bullets.append(
            f"• Bus: {counts['bus_stops']} stop(s) within ~{radius}m"
            + (f" (e.g., {', '.join(named_bus)})" if named_bus else "")
        )

        if not elements or (counts["stations"] + counts["tram_stops"] + counts["bus_stops"]) == 0:
            # Try wider radius (2500m) before giving up — Birmingham/West Midlands OSM coverage can be sparse
            wider_selectors = f"""
nwr["railway"="station"](around:2500,{lat},{lng});
nwr["railway"="tram_stop"](around:2500,{lat},{lng});
nwr["public_transport"](around:2500,{lat},{lng});
nwr["highway"="bus_stop"](around:2500,{lat},{lng});
""".strip()
            try:
                wider_payload = overpass_query(lat, lng, wider_selectors)
                wider_elements = wider_payload.get("elements", []) if isinstance(wider_payload, dict) else []
                wider_counts: Dict[str, int] = {"stations": 0, "tram_stops": 0, "bus_stops": 0}
                wider_named_bus: List[str] = []
                wider_named_stations: List[str] = []
                for we in wider_elements:
                    wtags = (we or {}).get("tags") or {}
                    if not isinstance(wtags, dict):
                        continue
                    wname = wtags.get("name")
                    wnm = wname.strip() if isinstance(wname, str) and wname.strip() else ""
                    if wtags.get("railway") == "station":
                        wider_counts["stations"] += 1
                        if wnm: wider_named_stations.append(wnm)
                    is_wbus = (
                        wtags.get("highway") == "bus_stop" or
                        (wtags.get("public_transport") in ("stop_position", "platform") and
                         wtags.get("railway") not in ("station", "tram_stop")) or
                        wtags.get("highway") == "bus_stop"
                    )
                    if is_wbus:
                        wider_counts["bus_stops"] += 1
                        if wnm: wider_named_bus.append(wnm)
                total_wider = wider_counts["stations"] + wider_counts["bus_stops"]
                if total_wider > 0:
                    wbullets = []
                    wbullets.append(f"• Rail: {wider_counts['stations']} station(s) within ~2500m" + (f" (e.g., {', '.join(wider_named_stations[:3])})" if wider_named_stations else ""))
                    wbullets.append(f"• Bus: {wider_counts['bus_stops']} stop(s) within ~2500m" + (f" (e.g., {', '.join(wider_named_bus[:5])})" if wider_named_bus else ""))
                    wsummary = "Transport (OSM within ~2.5km):\n" + "\n".join(wbullets)
                    wout = metric_ok(wsummary, wbullets, base_sources, retrieved, 0.75)
                    wout["metrics"] = {
                        "radiusMeters": 2500,
                        "counts": wider_counts,
                        "sample": {"stations": wider_named_stations[:3], "tram": [], "bus": wider_named_bus[:8]},
                    }
                    return wout
            except Exception:
                pass
            return metric_ok("No transport features returned from OSM for this area.", [], base_sources, retrieved, 0.0)

        summary = "Transport (OSM within ~1.2km):\n" + "\n".join(bullets)

        out = metric_ok(summary, bullets, base_sources, retrieved, 0.90)
        out["metrics"] = {
            "radiusMeters": radius,
            "counts": counts,
            "sample": {"stations": named_stations, "tram": named_tram, "bus": named_bus},
        }
        return out

    except Exception as e:
        return metric_unavailable(
            f"Transport data fetch failed: {str(e)}",
            base_sources,
            retrieved,
        )


def get_amenities_data(lat: Optional[float], lng: Optional[float]) -> Dict[str, Any]:
    retrieved = now_iso()
    base_sources = [
        {"label": "OpenStreetMap (Overpass API)", "url": "https://overpass-api.de/"},
        {"label": "OpenStreetMap", "url": "https://www.openstreetmap.org"},
    ]

    if lat is None or lng is None:
        return metric_unavailable(
            "Amenities data not available: postcode could not be resolved to coordinates.",
            base_sources,
            retrieved,
        )

    radius = DEFAULT_OSM_RADIUS
    selectors = f"""
nwr["amenity"](around:{radius},{lat},{lng});
nwr["shop"](around:{radius},{lat},{lng});
nwr["leisure"](around:{radius},{lat},{lng});
nwr["tourism"](around:{radius},{lat},{lng});
""".strip()

    try:
        payload = overpass_query(lat, lng, selectors)
        elements = payload.get("elements", []) if isinstance(payload, dict) else []
        if not isinstance(elements, list):
            elements = []

        buckets: Dict[str, Dict[str, Any]] = {
            "foodDrink": {"count": 0, "top": []},
            "shopping": {"count": 0, "top": []},
            "healthcare": {"count": 0, "top": []},
            "education": {"count": 0, "top": []},
            "leisure": {"count": 0, "top": []},
            "services": {"count": 0, "top": []},
            "other": {"count": 0, "top": []},
        }

        food_amenities = {"restaurant", "cafe", "pub", "bar", "fast_food", "food_court", "ice_cream", "biergarten"}
        health_amenities = {"hospital", "clinic", "doctors", "dentist", "pharmacy", "veterinary"}
        edu_amenities = {"school", "college", "university", "kindergarten", "childcare", "library"}
        service_amenities = {
            "bank", "atm", "post_office", "parcel_locker", "police", "fire_station",
            "townhall", "community_centre", "courthouse", "place_of_worship"
        }

        def _bucket_for(tags: Dict[str, Any]) -> str:
            a = tags.get("amenity")
            s = tags.get("shop")
            l = tags.get("leisure")
            t = tags.get("tourism")

            if isinstance(a, str):
                if a in food_amenities:
                    return "foodDrink"
                if a in health_amenities:
                    return "healthcare"
                if a in edu_amenities:
                    return "education"
                if a in service_amenities:
                    return "services"
                if a in {"cinema", "theatre", "arts_centre", "gym", "sports_centre", "swimming_pool", "park"}:
                    return "leisure"

            if isinstance(s, str):
                return "shopping"
            if isinstance(l, str) or isinstance(t, str):
                return "leisure"

            return "other"

        def _push_top(bucket_key: str, name: str) -> None:
            if not name:
                return
            arr = buckets[bucket_key]["top"]
            if name not in arr:
                arr.append(name)

        for e in elements:
            tags = (e or {}).get("tags") or {}
            if not isinstance(tags, dict):
                continue

            bk = _bucket_for(tags)
            buckets[bk]["count"] += 1

            nm = tags.get("name")
            name = nm.strip() if isinstance(nm, str) and nm.strip() else ""
            _push_top(bk, name)

        for k in buckets.keys():
            buckets[k]["top"] = buckets[k]["top"][:6]

        total = sum(int(buckets[k]["count"]) for k in buckets.keys())
        if total == 0:
            return metric_ok("No amenities returned from OSM for this area.", [], base_sources, retrieved, 0.0)

        bullets: List[str] = []

        def _line(label: str, key: str) -> None:
            c = buckets[key]["count"]
            tops = buckets[key]["top"]
            if c and c > 0:
                bullets.append(f"• {label}: {c}" + (f" (e.g., {', '.join(tops)})" if tops else ""))

        _line("Food & drink", "foodDrink")
        _line("Shopping", "shopping")
        _line("Healthcare", "healthcare")
        _line("Education", "education")
        _line("Leisure", "leisure")
        _line("Services", "services")
        if buckets["other"]["count"] >= 10:
            bullets.append(f"• Other mapped POIs: {buckets['other']['count']}")

        summary = f"Amenities (OSM within ~{radius}m): {total} mapped places.\n" + "\n".join(bullets)
        out = metric_ok(summary, bullets, base_sources, retrieved, 0.90)
        out["metrics"] = {"radiusMeters": radius, "total": total, "buckets": buckets}
        return out

    except Exception as e:
        return metric_unavailable(
            f"Amenities data fetch failed: {str(e)}",
            base_sources,
            retrieved,
        )


def get_schools_data(postcode: str) -> Dict[str, Any]:
    retrieved = now_iso()
    pc = normalize_postcode(postcode)
    district = postcode_district(pc)

    if not pc:
        return metric_unavailable(
            "Schools data not available: no postcode provided.",
            [{"label": "DfE Find and Compare Schools", "url": "https://www.compare-school-performance.service.gov.uk/"}],
            retrieved,
        )

    # ── Hetzner path (primary) ────────────────────────────────────────────────
    # schools_ofsted table on Hetzner: 9,781 rows, England only
    # Columns: urn, school_name, postcode, ofsted_rating, school_type, phase, la_name
    try:
        lat_lng_rows = data_query(
            "SELECT lat, lng FROM public.nspl_postcodes WHERE pcd_nospace = %s LIMIT 1",
            (pc.replace(" ", ""),)
        )
        if lat_lng_rows:
            slat = lat_lng_rows[0].get("lat")
            slng = lat_lng_rows[0].get("lng")
        else:
            slat, slng = None, None

        if slat and slng:
            school_rows = data_query(
                """
                SELECT s.urn, s.school_name, s.postcode, s.ofsted_rating,
                       s.la_name,
                       ROUND(
                         ST_Distance(
                           ST_MakePoint(n.lng, n.lat)::geography,
                           ST_MakePoint(%s, %s)::geography
                         ) / 1609.34, 2
                       ) AS miles
                FROM public.schools_ofsted s
                JOIN public.nspl_postcodes n ON n.pcd_nospace = REPLACE(s.postcode, ' ', '')
                WHERE ST_Distance(
                    ST_MakePoint(n.lng, n.lat)::geography,
                    ST_MakePoint(%s, %s)::geography
                ) <= 4828
                ORDER BY miles ASC
                LIMIT 10
                """,
                (slng, slat, slng, slat)
            )
        else:
            school_rows = data_query(
                """SELECT urn, school_name, postcode, ofsted_rating, la_name
                   FROM public.schools_ofsted
                   WHERE postcode ILIKE %s
                   LIMIT 10""",
                (f"{district}%",)
            )

        if school_rows:
            # Normalise numeric Ofsted codes to text — applies to both Hetzner and Supabase paths
            _OFSTED_NORM = {"1": "Outstanding", "2": "Good",
                            "3": "Requires Improvement", "4": "Inadequate"}
            for _sr in school_rows:
                _raw_rating = str(_sr.get("ofsted_rating") or "").strip()
                if _raw_rating in _OFSTED_NORM:
                    _sr["ofsted_rating"] = _OFSTED_NORM[_raw_rating]
                # Sanitise school name — replace numeric URN with "School"
                _sn = str(_sr.get("school_name") or "").strip()
                if not _sn or _sn.isdigit():
                    _sr["school_name"] = "School"
            sources = [{"label": "Ofsted / DfE", "url": "https://reports.ofsted.gov.uk/"}]
            out = metric_ok(
                f"{len(school_rows)} schools within 3mi of {pc}.",
                school_rows, sources, retrieved, 0.85,
            )
            out["metrics"] = {"provider": "hetzner", "count": len(school_rows)}
            return out
    except Exception as _se:
        print(f"[WARN] Schools Hetzner query failed: {_se}")

    # ── Supabase auto-fallback: schools_clean_v2 (no env var required) ──────
    # schools_clean_v2 schema is unknown — SELECT * and map whatever columns exist
    _OFSTED_CODE_MAP = {"1": "Outstanding", "2": "Good",
                        "3": "Requires Improvement", "4": "Inadequate"}
    if supabase:
        # Always use SELECT * — the table schema may differ from expected columns
        # (e.g. --schools job never run, or table created with different schema)
        for _cols in [
            "*",  # always try * first — handles any schema
            "urn,school_name,postcode,ofsted_rating",
            "urn,name,postcode,ofsted_rating",
            "urn,postcode",
        ]:
            try:
                _q = supabase.table("schools_clean_v2").select(_cols)
                # Try postcode ilike first; fallback to postcode_district eq
                try:
                    res_sb = _q.ilike("postcode", f"{district}%").limit(SCHOOLS_MAX_RESULTS).execute()
                except Exception:
                    res_sb = supabase.table("schools_clean_v2").select(_cols).eq("postcode_district", district).limit(SCHOOLS_MAX_RESULTS).execute()
                sb_rows = res_sb.data if hasattr(res_sb, "data") and isinstance(res_sb.data, list) else []
                if not sb_rows:
                    break  # Table empty — no point trying other column sets
                for r in sb_rows:
                    # Normalise name variants to school_name
                    if "name" in r and "school_name" not in r:
                        r["school_name"] = r["name"]
                    elif "EstablishmentName" in r and "school_name" not in r:
                        r["school_name"] = r["EstablishmentName"]
                    # Use "School" not URN as name fallback — URNs are meaningless to investors
                    _raw_name = r.get("school_name") or r.get("name") or r.get("EstablishmentName")
                    if not _raw_name or str(_raw_name).strip().isdigit():
                        r["school_name"] = "School"
                    else:
                        r["school_name"] = str(_raw_name).strip()
                    r.setdefault("miles", None)
                    # Normalise numeric Ofsted codes to text
                    raw_r = str(r.get("ofsted_rating") or "").strip()
                    if raw_r in _OFSTED_CODE_MAP:
                        r["ofsted_rating"] = _OFSTED_CODE_MAP[raw_r]
                sources_sb = [{"label": "Ofsted / DfE", "url": "https://reports.ofsted.gov.uk/"}]
                out_sb = metric_ok(
                    f"{len(sb_rows)} schools found for district {district}.",
                    sb_rows, sources_sb, retrieved, SCHOOLS_CONFIDENCE_VALUE,
                )
                out_sb["metrics"] = {"provider": "supabase_schools_clean_v2", "district": district, "count": len(sb_rows)}
                return out_sb
            except Exception as _sb_se:
                print(f"[WARN] Schools Supabase fallback ({_cols}): {_sb_se}")
                continue

    if SCHOOLS_PROVIDER == "supabase":
        if not supabase:
            return metric_unavailable(
                "Schools provider set to supabase but Supabase is not configured on server.",
                [{"label": "Supabase", "url": "https://supabase.com/"}],
                retrieved,
                extra_metrics={"postcode": pc},
            )

        sources = [{"label": "Supabase (schools)", "url": f"{SUPABASE_URL}" if SUPABASE_URL else "https://supabase.com/"}]

        try:
            cols_view = "postcode_district,urn,name,establishment_type,phase,local_authority,town,postcode,status,telephone,website"
            res = (
                supabase.table(SCHOOLS_SUPABASE_VIEW)
                .select(cols_view)
                .eq("postcode_district", district)
                .limit(SCHOOLS_MAX_RESULTS)
                .execute()
            )
            rows = res.data if hasattr(res, "data") else None
            if not isinstance(rows, list):
                rows = []

            for r in rows:
                if isinstance(r, dict):
                    r.setdefault("postcode_district", postcode_district(r.get("postcode", "") or ""))

            if rows:
                out = metric_ok(
                    f"Schools found for postcode district {district}: {len(rows)}.",
                    rows,
                    sources,
                    retrieved,
                    SCHOOLS_CONFIDENCE_VALUE,
                )
                out["metrics"] = {"provider": "supabase", "mode": "district", "district": district, "limit": SCHOOLS_MAX_RESULTS}
                return out

        except Exception:
            pass

        try:
            cols_tbl = "urn,name,postcode,phase,establishment_type,local_authority,town,status,telephone,website"
            res2 = (
                supabase.table(SCHOOLS_SUPABASE_FALLBACK_TABLE)
                .select(cols_tbl)
                .ilike("postcode", f"{district}%")
                .limit(SCHOOLS_MAX_RESULTS)
                .execute()
            )
            rows2 = res2.data if hasattr(res2, "data") else None
            if not isinstance(rows2, list):
                rows2 = []

            for r in rows2:
                if isinstance(r, dict):
                    r["postcode_district"] = postcode_district(r.get("postcode", "") or "")

            if rows2:
                out = metric_ok(
                    f"Schools found for postcode district {district}: {len(rows2)}.",
                    rows2,
                    sources,
                    retrieved,
                    SCHOOLS_CONFIDENCE_VALUE,
                )
                out["metrics"] = {"provider": "supabase", "mode": "district_fallback", "district": district, "limit": SCHOOLS_MAX_RESULTS}
                return out

            return metric_unavailable(
                f"No schools returned for postcode district {district}.",
                sources,
                retrieved,
                extra_metrics={"district": district},
            )

        except Exception as e2:
            return metric_unavailable(
                f"Schools query failed: {str(e2)}",
                [{"label": "Supabase", "url": "https://supabase.com/"}],
                retrieved,
                extra_metrics={"postcode": pc, "district": district},
            )

    return metric_missing_provider(
        "Schools provider not configured. Set SCHOOLS_PROVIDER=supabase.",
        [
            {"label": "Ofsted reports", "url": "https://reports.ofsted.gov.uk/"},
            {"label": "DfE Find and Compare Schools", "url": "https://www.compare-school-performance.service.gov.uk/"},
        ],
        retrieved,
        extra_metrics={"postcode": pc, "district": district},
    )




def get_gp_data(postcode: str) -> Dict[str, Any]:
    """NHS Service Search API — nearest GP practices to a postcode. Free, no key required."""
    retrieved = now_iso()
    sources = [{"label": "NHS Service Search", "url": "https://www.nhs.uk/service-search/"}]
    pc = normalize_postcode(postcode)
    if not pc:
        return metric_unavailable("GP data not available: no postcode.", sources, retrieved)
    try:
        url = f"https://api.nhs.uk/service-search/search?api-version=2&search={pc.replace(' ', '+')}&filter=OrganisationTypeId eq 'GPS'&top=5&orderby=geo.distance(Geocode, geography%27POINT({0} {1})%27)"
        # Use simpler endpoint that doesn't require coordinates
        url = f"https://api.nhs.uk/service-search/search-postcodes-and-places?api-version=1&search={pc.replace(' ','%20')}"
        status, payload = _http_get_json(url, timeout=10)
        if status == 200 and isinstance(payload, dict):
            results = payload.get("value") or payload.get("results") or []
            if results:
                gps = [r for r in results if (r.get("OrganisationTypeId") or "") == "GPS"][:5]
                if gps:
                    out = metric_ok(
                        f"{len(gps)} GP practice(s) found near {pc}.",
                        gps,
                        sources,
                        retrieved,
                        0.8,
                    )
                    out["metrics"] = {"count": len(gps), "postcode": pc}
                    return out
    except Exception as e:
        print(f"[WARN] GP data fetch failed: {e}")
    return metric_unavailable(f"GP data not available for {pc}.", sources, retrieved)


def _get_lad_code_for_postcode(postcode: str) -> Optional[str]:
    """Look up LAD code from postcode_to_lsoa table."""
    try:
        pc = normalize_postcode(postcode)
        rows = data_query(
            "SELECT ladcd FROM public.postcode_to_lsoa WHERE pcds = %s LIMIT 1",
            (pc,)
        )
        if rows and rows[0].get("ladcd"):
            return str(rows[0]["ladcd"]).strip()
    except Exception:
        pass
    return None


def _get_rental_trend(lad_code: str) -> Dict[str, Any]:
    """
    Pull rent_yoy_pct series from uk_prms_monthly for the LAD.
    Returns direction: Increasing / Stable / Declining and 36-month series.
    """
    try:
        rows = supabase_data_query(
            "SELECT period, rent_yoy_pct, rent_price_gbp FROM public.uk_prms_monthly WHERE area_code = %s ORDER BY period ASC LIMIT 48",
            (lad_code,)
        )
        if not rows:
            return {"direction": "Unknown", "series": [], "latest_yoy": None}

        series = []
        for r in rows:
            yoy = safe_float(r.get("rent_yoy_pct"))
            idx = safe_float(r.get("rent_index"))
            period = str(r.get("period") or r.get("date") or "")[:10]
            if period:
                series.append({
                    "period":        period,
                    "rent_index":    idx,
                    "rent_yoy_pct":  yoy,
                    "rent_price_gbp": r.get("rent_price_gbp"),
                })

        # Use last 3 readings with yoy data for direction
        yoy_vals = [s["rent_yoy_pct"] for s in series if s["rent_yoy_pct"] is not None]
        latest_yoy = yoy_vals[-1] if yoy_vals else None
        avg_yoy    = sum(yoy_vals[-6:]) / len(yoy_vals[-6:]) if len(yoy_vals) >= 2 else latest_yoy

        if avg_yoy is None:
            direction = "Unknown"
        elif avg_yoy > 1.0:
            direction = "Increasing"
        elif avg_yoy < -1.0:
            direction = "Declining"
        else:
            direction = "Stable"

        return {"direction": direction, "series": series, "latest_yoy": latest_yoy}
    except Exception as e:
        print(f"[WARN] Rental trend fetch failed for {lad_code}: {e}")
        return {"direction": "Unknown", "series": [], "latest_yoy": None}


def _get_population_trend(lad_code: str) -> Dict[str, Any]:
    """
    Pull population series from ons_population for the LAD.
    Returns direction: Growing / Stable / Declining and year series.
    """
    try:
        rows = data_query(
            "SELECT year, population FROM public.ons_population WHERE lad_code = %s ORDER BY year ASC",
            (lad_code,)
        )
        if not rows or len(rows) < 2:
            return {"direction": "Unknown", "series": rows, "change_pct": None}

        series = [{"year": r["year"], "population": r["population"]} for r in rows]

        # Use last 5 years for trend
        recent = series[-5:]
        start_pop = recent[0]["population"]
        end_pop   = recent[-1]["population"]
        change_pct = ((end_pop - start_pop) / start_pop * 100) if start_pop else 0

        if change_pct > 1.0:
            direction = "Growing"
        elif change_pct < -1.0:
            direction = "Declining"
        else:
            direction = "Stable"

        return {"direction": direction, "series": series, "change_pct": round(change_pct, 2)}
    except Exception as e:
        print(f"[WARN] Population trend fetch failed for {lad_code}: {e}")
        return {"direction": "Unknown", "series": [], "change_pct": None}


def _get_national_rental_benchmark() -> Dict[str, Any]:
    """
    National rental growth benchmark from uk_prms_monthly.
    Uses area_code E92000001 (England) or aggregates all LADs.
    Returns latest YoY % and 12-month average.
    """
    try:
        rows = supabase_data_query(
            "SELECT period, rent_yoy_pct FROM public.uk_prms_monthly WHERE area_code = %s ORDER BY period DESC LIMIT 48",
            ("E92000001",)
        )
        if not rows:
            rows2 = supabase_data_query(
                "SELECT rent_yoy_pct FROM public.uk_prms_monthly WHERE period = (SELECT MAX(period) FROM public.uk_prms_monthly) LIMIT 500"
            )
            vals = [safe_float(r.get("rent_yoy_pct")) for r in rows2 if r.get("rent_yoy_pct") is not None]
            avg = round(sum(vals) / len(vals), 2) if vals else None
            return {"latest_yoy": avg, "avg_12m": avg}
        yoys = [safe_float(r.get("rent_yoy_pct")) for r in rows if r.get("rent_yoy_pct") is not None]
        latest = yoys[0] if yoys else None
        avg12  = round(sum(yoys) / len(yoys), 2) if yoys else None
        return {"latest_yoy": latest, "avg_12m": avg12}
    except Exception as e:
        print(f"[WARN] National rental benchmark failed: {e}")
        return {"latest_yoy": None, "avg_12m": None}


def _get_national_price_benchmark() -> Dict[str, Any]:
    """
    National average sold price from uk_hpi_monthly.
    Uses get_hpi_benchmark RPC (SECURITY DEFINER) — proven to work on Render
    where direct supabase.table().select() for uk_hpi_monthly returns [] via REST.
    """
    if supabase:
        try:
            res = supabase.rpc("get_hpi_benchmark", {"p_area_code": "E92000001"}).execute()
            rows = res.data if hasattr(res, "data") and isinstance(res.data, list) else []
            if rows:
                r = rows[0]
                return {
                    "avg_price": safe_float(r.get("average_price")),
                    "yoy_pct":   safe_float(r.get("annual_change")),
                }
        except Exception as e:
            print(f"[WARN] National price benchmark RPC failed: {e}")
    return {"avg_price": None, "yoy_pct": None}


def _get_regional_price_benchmark(lad_code: str) -> Dict[str, Any]:
    """
    Regional average sold price from uk_hpi_monthly for the LAD.
    Uses get_hpi_benchmark RPC (SECURITY DEFINER) — bypasses the REST table path
    which silently returns [] even with GRANT applied (proven on production).
    Falls back to England aggregate if LAD not found.
    """
    if not supabase:
        return {"avg_price": None, "yoy_pct": None}

    def _rpc_lookup(code: str):
        try:
            res = supabase.rpc("get_hpi_benchmark", {"p_area_code": code}).execute()
            rows = res.data if hasattr(res, "data") and isinstance(res.data, list) else []
            if rows:
                r = rows[0]
                return {
                    "avg_price": safe_float(r.get("average_price")),
                    "yoy_pct":   safe_float(r.get("annual_change")),
                }
        except Exception as e:
            print(f"[WARN] HPI RPC lookup failed for {code}: {e}")
        return None

    result = _rpc_lookup(lad_code)
    if result and result.get("avg_price"):
        return result

    print(f"[HPI] LAD {lad_code} not found via RPC — trying England aggregate")
    for eng_code in ("E92000001", "K02000001"):
        result = _rpc_lookup(eng_code)
        if result and result.get("avg_price"):
            print(f"[HPI] Using {eng_code} aggregate for {lad_code}")
            return result

    return {"avg_price": None, "yoy_pct": None}


def _get_regional_rental_benchmark(lad_code: str) -> Dict[str, Any]:
    """
    Regional rental growth from uk_prms_monthly for LAD.
    Returns latest YoY % and direction.
    """
    try:
        rows = supabase_data_query(
            "SELECT period, rent_yoy_pct, rent_price_gbp FROM public.uk_prms_monthly WHERE area_code = %s ORDER BY period DESC LIMIT 3",
            (lad_code,)
        )
        if not rows:
            return {"latest_yoy": None, "avg_rent_gbp": None}
        yoys = [safe_float(r.get("rent_yoy_pct")) for r in rows if r.get("rent_yoy_pct") is not None]
        rents = [safe_float(r.get("rent_price_gbp")) for r in rows if r.get("rent_price_gbp") is not None]
        return {
            "latest_yoy":   round(yoys[0], 2) if yoys else None,
            "avg_rent_gbp": round(sum(rents) / len(rents), 0) if rents else None,
        }
    except Exception as e:
        print(f"[WARN] Regional rental benchmark failed for {lad_code}: {e}")
        return {"latest_yoy": None, "avg_rent_gbp": None}


def _get_transaction_liquidity(postcode: str, lad_code: str) -> Dict[str, Any]:
    """
    Transaction liquidity: count of sold transactions per quarter
    within postcode district vs LAD average.
    Sourced from price_paid_raw_2025.
    """
    try:
        district = postcode.split()[0] if postcode else ""
        if not district:
            return {"local_qtly": None, "lad_qtly": None}
        # Local: last 12 months in postcode district
        import datetime as _dt
        cutoff = (_dt.datetime.utcnow() - _dt.timedelta(days=365)).strftime("%Y-%m-%d")
        pp_rows = data_query(
            "SELECT COUNT(*) AS cnt FROM public.price_paid_raw_2025 WHERE postcode ILIKE %s AND date_of_transfer >= %s",
            (f"{district}%", cutoff)
        )
        local_count = int(pp_rows[0].get("cnt", 0)) if pp_rows else 0
        local_qtly  = round((local_count or 0) / 4, 0)
        return {
            "local_qtly":  local_qtly,
            "local_annual": local_count,
            "district":   district,
        }
    except Exception as e:
        print(f"[WARN] Transaction liquidity failed for {postcode}: {e}")
        return {"local_qtly": None, "local_annual": None, "district": None}


def _get_yield_benchmarks(lad_code: str) -> Dict[str, Any]:
    """
    Compute gross yield benchmarks from existing tables.
    yield = (latest_monthly_rent * 12) / avg_price * 100
    Local (LAD), regional (from HPI), national (England).
    """
    try:
        results = {}

        # Local yield — LAD rent + LAD price
        rent_rows = supabase_data_query(
            "SELECT rent_price_gbp FROM public.uk_prms_monthly WHERE area_code = %s ORDER BY period DESC LIMIT 1",
            (lad_code,)
        )
        local_rent = safe_float((rent_rows[0].get("rent_price_gbp") if rent_rows else None))

        price_rows = supabase_data_query(
            "SELECT average_price FROM public.uk_hpi_monthly WHERE area_code = %s ORDER BY date DESC LIMIT 1",
            (lad_code,)
        )
        local_price = safe_float((price_rows[0].get("average_price") if price_rows else None))

        if local_rent and local_price and local_price > 0:
            results["local_yield"] = round((local_rent * 12 / local_price) * 100, 2)
        else:
            results["local_yield"] = None

        # National yield — England E92000001
        nat_rent_rows = supabase_data_query(
            "SELECT rent_price_gbp FROM public.uk_prms_monthly WHERE area_code = %s ORDER BY period DESC LIMIT 1",
            ("E92000001",)
        )
        nat_rent = safe_float((nat_rent_rows[0].get("rent_price_gbp") if nat_rent_rows else None))

        nat_price_rows = supabase_data_query(
            "SELECT average_price FROM public.uk_hpi_monthly WHERE area_code = %s ORDER BY date DESC LIMIT 1",
            ("E92000001",)
        )
        nat_price = safe_float((nat_price_rows[0].get("average_price") if nat_price_rows else None))

        if nat_rent and nat_price and nat_price > 0:
            results["national_yield"] = round((nat_rent * 12 / nat_price) * 100, 2)
        else:
            results["national_yield"] = None

        results["local_rent_gbp"]    = local_rent
        results["local_price_gbp"]   = local_price
        results["national_rent_gbp"] = nat_rent
        results["source"] = "ONS PRMS · Land Registry HPI"
        return results

    except Exception as e:
        print(f"[WARN] Yield benchmark failed for {lad_code}: {e}")
        return {"local_yield": None, "national_yield": None}


def _get_imd_for_lsoa(lsoa_code: str) -> Dict[str, Any]:
    """
    IMD deprivation decile for an LSOA.
    Reads from lsoa_imd table if loaded, returns None if not yet available.
    Table: lsoa_imd (lsoa_code, imd_rank, imd_decile, lsoa_name)
    """
    if not lsoa_code:
        return {"decile": None, "rank": None}
    try:
        rows = data_query(
            "SELECT imd_rank, imd_decile FROM public.lsoa_imd WHERE lsoa_code = %s LIMIT 1",
            (lsoa_code,)
        )
        _ = []
        if rows:
            return {
                "decile": rows[0].get("imd_decile"),
                "rank":   rows[0].get("imd_rank"),
                "source": "MHCLG IMD 2019",
            }
        return {"decile": None, "rank": None}
    except Exception as e:
        print(f"[WARN] IMD lookup failed for {lsoa_code}: {e}")
        return {"decile": None, "rank": None}


def _get_census_private_rent_pct(lsoa_code: str) -> Optional[float]:
    """
    Return % private rented for LSOA.
    Primary:  Hetzner census_tenure_lsoa table.
    Fallback: ONS Census 2021 open API — no key required.
    """
    if not lsoa_code:
        return None

    # Primary: Hetzner
    try:
        rows = data_query(
            "SELECT private_rent_pct FROM public.census_tenure_lsoa WHERE lsoa_code = %s LIMIT 1",
            (lsoa_code,)
        )
        if rows:
            return float(rows[0]["private_rent_pct"])
    except Exception as e:
        print(f"[WARN] _get_census_private_rent_pct hetzner: {e}")

    # Fallback: ONS Census 2021 API (TS054 Tenure — open, no key needed)
    try:
        url = (
            f"https://api.beta.ons.gov.uk/v1/datasets/TS054/editions/2021/versions/1"
            f"/observations?area-type=lsoa&areas={lsoa_code}"
        )
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            observations = (r.json() or {}).get("observations") or []
            total = 0
            private_rent = 0
            for obs in observations:
                dims = obs.get("dimensions") or {}
                tenure = str((dims.get("Tenure of household (5 categories)") or {}).get("label") or "").lower()
                val = int(obs.get("observation") or 0)
                total += val
                if "private rented" in tenure:
                    private_rent += val
            if total > 0:
                pct = round((private_rent / total) * 100, 1)
                print(f"[INFO] ONS TS054 for {lsoa_code}: {pct}% private rented")
                return pct
    except Exception as e:
        print(f"[WARN] _get_census_private_rent_pct ONS API: {e}")

    return None


def build_area_inference(area_data: Dict[str, Any], postcode: str) -> Dict[str, Any]:
    """
    7-step inference engine per spec.
    Reads all signals from area_data + Supabase.
    Returns structured inference: trajectory, chart_data, drivers, demand_supply, provenance.
    Never raises — returns partial result on any failure.
    """
    try:
        # ── STEP 1: FEATURE EXTRACTION ─────────────────────────────
        area_code = str(area_data.get("area_code") or "").strip()

        # Price trend — from trends.signals.priceGrowth
        trends    = area_data.get("trends") or {}
        signals   = (trends.get("signals") or {})
        pg        = signals.get("priceGrowth") or {}
        price_yoy = safe_float(pg.get("percentage", "").replace("%", "")) if isinstance(pg.get("percentage"), str) else safe_float(pg.get("percentage"))
        price_trend_dir = pg.get("trend") or "Stable"  # Increasing / Stable / Decreasing
        price_history   = pg.get("historicalData") or []

        # Transaction volume — from housing metrics
        housing   = area_data.get("housing") or {}
        h_metrics = housing.get("metrics") or {}
        tx_count  = safe_float(h_metrics.get("transaction_count") or h_metrics.get("count") or 0)

        # Crime trend
        crime     = area_data.get("crime") or {}
        c_metrics = crime.get("metrics") or {}
        crime_trend = str(c_metrics.get("trend") or crime.get("summary") or "").lower()
        crime_rising = any(w in crime_trend for w in ["rising", "increasing", "up", "higher"])
        crime_falling = any(w in crime_trend for w in ["falling", "decreasing", "down", "lower", "low"])

        # Amenities
        amenities   = area_data.get("amenities") or {}
        a_metrics   = amenities.get("metrics") or {}
        amenity_count = int(safe_float(a_metrics.get("total") or a_metrics.get("count") or 0) or 0)

        # EPC distribution
        epc       = area_data.get("epc") or {}
        e_metrics = epc.get("metrics") or {}
        epc_dominant = str(e_metrics.get("dominant_rating") or "").upper()
        epc_score    = safe_float(e_metrics.get("efficiency_score") or 0) or 0

        # Planning
        planning    = area_data.get("planning") or {}
        p_metrics   = planning.get("metrics") or {}
        plan_recent = int(safe_float(p_metrics.get("recent_24m") or 0) or 0)
        plan_total  = int(safe_float(p_metrics.get("total") or 0) or 0)
        plan_new_build = int(safe_float(p_metrics.get("new_build") or 0) or 0)

        # Rental trend from Supabase
        lad_code = _get_lad_code_for_postcode(postcode) or area_code
        rental   = _get_rental_trend(lad_code)
        rental_dir = rental["direction"]   # Increasing / Stable / Declining / Unknown
        rental_series = rental["series"]

        # Population trend from Supabase
        pop      = _get_population_trend(lad_code)
        pop_dir  = pop["direction"]        # Growing / Stable / Declining / Unknown
        pop_series = pop["series"]

        # ── BENCHMARKS: national + regional comparisons ───────────
        nat_price    = _get_national_price_benchmark()
        reg_price    = _get_regional_price_benchmark(lad_code)
        nat_rental   = _get_national_rental_benchmark()
        reg_rental   = _get_regional_rental_benchmark(lad_code)
        liquidity    = _get_transaction_liquidity(postcode, lad_code)

        # Crime index: crimes per 1000 population vs national avg
        # ONS 2023: ~82 crimes per 1000 population nationally (England & Wales)
        crime_total  = safe_float((area_data.get("crime") or {}).get("metrics", {}).get("total") or 0) or 0
        pop_latest   = safe_float(pop.get("latest_value") or 0) or 10000
        local_crime_rate   = (crime_total / pop_latest * 1000) if pop_latest > 0 else 0
        national_crime_rate = 82.0  # ONS Crime Survey England and Wales 2023
        crime_index  = round(local_crime_rate / national_crime_rate, 2) if national_crime_rate > 0 else None

        # Comp avg from housing for price benchmarking
        housing_metrics = (area_data.get("housing") or {}).get("metrics") or {}
        comp_avg    = safe_float(housing_metrics.get("avg") or housing_metrics.get("average_price") or 0) or None
        comp_count  = safe_float(housing_metrics.get("count") or tx_count or 0) or 0

        # Yield benchmarks — local vs national (computable from existing tables)
        yield_data   = _get_yield_benchmarks(lad_code)

        # IMD deprivation — requires lsoa_imd table to be loaded
        lsoa_code    = str(area_data.get("lsoa_gss") or "").strip()
        imd_data     = _get_imd_for_lsoa(lsoa_code)

        # Census 2021 tenure — private rent % (Hetzner primary, ONS API fallback)
        _census_private_rent_pct = _get_census_private_rent_pct(lsoa_code) if lsoa_code else None

        # ── STEP 2: SIGNAL MAPPING ────────────────────────────────
        # Demand Pressure
        rental_up   = rental_dir == "Increasing"
        rental_down = rental_dir == "Declining"
        pop_up      = pop_dir == "Growing"
        pop_down    = pop_dir == "Declining"
        rental_unknown = rental_dir == "Unknown"
        pop_unknown    = pop_dir == "Unknown"

        if rental_up and pop_up:
            demand_signal = "Increasing"
        elif rental_down or pop_down:
            demand_signal = "Weakening"
        elif rental_up or pop_up:
            demand_signal = "Increasing"
        elif rental_unknown and pop_unknown:
            # INVARIANT: "Stable" must not be used when both source datasets are unavailable.
            # "Stable" implies data-backed stability. "Insufficient Data" is honest.
            demand_signal = "Insufficient Data"
        else:
            demand_signal = "Stable"

        # Growth Vector
        amenity_positive = amenity_count > 20
        planning_active  = plan_recent >= 5

        if amenity_positive and planning_active:
            growth_signal = "Expanding"
        elif amenity_positive or planning_active:
            growth_signal = "Emerging"
        else:
            growth_signal = "Flat"

        # Market Response
        price_up   = price_trend_dir == "Increasing"
        price_down = price_trend_dir == "Decreasing"
        vol_up     = tx_count > 10

        if price_up and vol_up:
            market_signal = "Confirming"
        elif price_up and not vol_up:
            market_signal = "Fragile"
        elif price_down:
            market_signal = "Diverging"
        else:
            market_signal = "Neutral"

        # Risk Drag
        epc_poor = epc_dominant in ("E", "F", "G") or epc_score < 3.5
        epc_good = epc_dominant in ("A", "B", "C") or epc_score >= 4.5

        if crime_rising and epc_poor:
            risk_signal = "Suppressing"
        elif crime_falling and epc_good:
            risk_signal = "Minimal"
        else:
            risk_signal = "Neutral"

        # ── STEP 3: AREA TRAJECTORY ───────────────────────────────
        # Score: demand(2) + growth(1) + market(2) - risk(2)
        score = 0
        # "Insufficient Data" maps to 0 (neutral) but trajectory label must reflect data absence
        score += {"Increasing": 2, "Stable": 0, "Weakening": -2, "Insufficient Data": 0}.get(demand_signal, 0)
        score += {"Expanding": 2, "Emerging": 1, "Flat": 0}.get(growth_signal, 0)
        score += {"Confirming": 2, "Fragile": 1, "Neutral": 0, "Diverging": -2}.get(market_signal, 0)
        score += {"Minimal": 1, "Neutral": 0, "Suppressing": -2}.get(risk_signal, 0)

        # INVARIANT: Must not label trajectory without minimum data coverage
        # Use fetch_status to detect data absence vs genuine 0 crimes
        # crime_total = 0 in a safe rural area is real data, not missing data
        _crime_data_present = (
            (area_data.get("crime") or {}).get("fetch_status") not in
            ("unavailable", "error", None, "")
            and (area_data.get("crime") or {}).get("status") not in ("unavailable", "error")
        )
        _data_signals_present = sum([
            rental_dir != "Unknown",
            pop_dir != "Unknown",
            price_yoy is not None,
            _crime_data_present,
        ])
        _data_availability_pct = _data_signals_present / 4.0

        if _data_availability_pct < 0.5:
            # Fewer than 2 of 4 core signals available — trajectory is not determinable
            trajectory = "INSUFFICIENT DATA"
        elif score >= 4:
            trajectory = "EXPANDING"
        elif score >= 1:
            trajectory = "STABLE"
        elif score >= -1:
            trajectory = "CONSTRAINED"
        else:
            trajectory = "DECLINING"

        # ── STEP 4: CHART DATA (indexed, base=100) ────────────────
        # Price line — from uk_hpi historicalData (already YoY %)
        # Convert YoY % series to index (base=100 at start)
        price_index = []
        idx_val = 100.0
        for pt in sorted(price_history, key=lambda x: x.get("period", "")):
            yoy = safe_float(pt.get("price_change_pct") or pt.get("value") or 0) or 0
            idx_val = idx_val * (1 + yoy / 100)
            price_index.append({"period": pt.get("period"), "value": round(idx_val, 2)})
        # Trim to last 36 months
        price_index = price_index[-36:] if len(price_index) > 36 else price_index

        # Demand line — rent_index from uk_prms_monthly (already base=100)
        demand_index = [
            {"period": s["period"], "value": round(float(s["rent_index"]), 2)}
            for s in rental_series if s.get("rent_index") is not None
        ][-36:]

        # Growth line — planning applications indexed (base=100 at first period)
        # Use amenity count as a flat signal if planning sparse
        growth_index = []
        if plan_total > 0:
            growth_base = max(plan_total / 12, 1)
            for i, s in enumerate(rental_series[-36:]):
                growth_index.append({"period": s["period"], "value": round(100 * (1 + (plan_recent / max(plan_total, 1) - 0.5)), 2)})
        else:
            growth_index = [{"period": s["period"], "value": 100.0} for s in rental_series[-36:]]

        chart_data = {
            "price":   price_index,
            "demand":  demand_index,
            "growth":  growth_index,
        }

        # ── STEP 5: DRIVER BULLETS — benchmarked, sourced, no verdicts ──
        # Moneyball principle: every number benchmarked. Data tells the story.
        drivers = []

        def _fmt_gbp(v):
            if v is None: return None
            return f"£{int(v):,}"

        def _fmt_pct(v):
            if v is None: return None
            return f"{v:+.1f}%"

        # Rental growth — local vs regional vs national
        if rental.get("latest_yoy") is not None:
            local_r  = rental["latest_yoy"]
            reg_r    = reg_rental.get("latest_yoy")
            nat_r    = nat_rental.get("latest_yoy")
            parts    = [f"Rental {_fmt_pct(local_r)} YoY"]
            if reg_r is not None: parts.append(f"reg {_fmt_pct(reg_r)}")
            if nat_r is not None: parts.append(f"nat {_fmt_pct(nat_r)}")
            parts.append("· ONS PRMS")
            sign = "+" if local_r > 0 else "-"
            drivers.append({"sign": sign, "text": " · ".join(parts)})

        # Price growth — local vs national
        if price_yoy is not None:
            nat_py = nat_price.get("yoy_pct")
            parts  = [f"Price {_fmt_pct(price_yoy)} YoY"]
            if nat_py is not None: parts.append(f"nat {_fmt_pct(nat_py)}")
            parts.append("· Land Registry HPI")
            sign = "+" if price_yoy > 0 else "-"
            drivers.append({"sign": sign, "text": " · ".join(parts)})

        # Average sold price — local vs regional vs national
        if comp_avg and comp_avg > 0:
            reg_p = reg_price.get("avg_price")
            nat_p = nat_price.get("avg_price")
            parts = [f"Avg sold {_fmt_gbp(comp_avg)}"]
            if reg_p: parts.append(f"reg {_fmt_gbp(reg_p)}")
            if nat_p: parts.append(f"nat {_fmt_gbp(nat_p)}")
            parts.append(f"· {int(comp_count)} transactions · Land Registry")
            drivers.append({"sign": "~", "text": " · ".join(parts)})

        # Crime index — local rate vs national benchmark
        if crime_index is not None:
            nat_label = f"{national_crime_rate:.0f}/1000 national avg"
            local_label = f"{local_crime_rate:.1f}/1000 local"
            index_label = f"{crime_index:.2f}× national"
            sign = "-" if crime_index > 1.2 else ("+" if crime_index < 0.8 else "~")
            drivers.append({
                "sign": sign,
                "text": f"Crime {local_label} · {index_label} · {nat_label} · Police.uk"
            })
        elif crime_total > 0:
            drivers.append({
                "sign": "~" if crime_total < 300 else "-",
                "text": f"Crime {int(crime_total)}/yr · Police.uk"
            })

        # Population trend — with magnitude
        if pop.get("change_pct") is not None:
            chg = pop["change_pct"]
            parts = [f"Population {_fmt_pct(chg)} over 5yr"]
            parts.append("· ONS Mid-Year Estimates")
            sign = "+" if chg > 2 else ("-" if chg < -2 else "~")
            drivers.append({"sign": sign, "text": " · ".join(parts)})

        # Transaction liquidity
        if liquidity.get("local_qtly") is not None:
            qtly = int(liquidity["local_qtly"])
            ann  = int(liquidity.get("local_annual") or 0)
            dist = liquidity.get("district", "")
            drivers.append({
                "sign": "~",
                "text": f"Transactions {ann}/yr ({qtly}/qtr) in {dist} · Land Registry"
            })

        # Yield benchmark — local vs national
        if yield_data.get("local_yield"):
            local_y  = yield_data["local_yield"]
            nat_y    = yield_data.get("national_yield")
            parts    = [f"Gross yield {local_y:.1f}%"]
            if nat_y: parts.append(f"nat {nat_y:.1f}%")
            parts.append("· ONS PRMS · Land Registry HPI")
            sign = "+" if (nat_y and local_y > nat_y) else ("~" if nat_y else "~")
            drivers.append({"sign": sign, "text": " · ".join(parts)})

        # IMD deprivation decile (when lsoa_imd table is loaded)
        if imd_data.get("decile") is not None:
            decile = imd_data["decile"]
            sign   = "-" if decile <= 2 else ("+" if decile >= 8 else "~")
            drivers.append({
                "sign": sign,
                "text": f"IMD decile {decile}/10 · {imd_data.get('source', 'MHCLG IMD')}"
            })

        # Planning activity
        if plan_recent >= 3:
            sign = "+" if plan_recent >= 10 else "~"
            drivers.append({
                "sign": sign,
                "text": f"Planning {plan_recent} applications in 24m · PlanningAlerts"
            })

        # EPC profile
        if epc_dominant:
            sign = "+" if epc_good else ("-" if epc_poor else "~")
            drivers.append({
                "sign": sign,
                "text": f"EPC dominant {epc_dominant} · MHCLG"
            })

        # Sort: negative first (risk), then data points, then positive
        # Keep max 6 drivers — enough for full picture without noise
        neg = [d for d in drivers if d["sign"] == "-"]
        pos = [d for d in drivers if d["sign"] == "+"]
        neu = [d for d in drivers if d["sign"] == "~"]
        drivers = (neg + pos + neu)[:6]

        if not drivers:
            drivers.append({"sign": "~", "text": "Insufficient data to compute area drivers."})

        # ── STEP 6: DEMAND VS SUPPLY ───────────────────────────────
        # Demand/supply: data classification only, no interpretive language
        if demand_signal == "Increasing" and market_signal in ("Confirming", "Fragile"):
            demand_supply = f"Demand signal: {demand_signal} · Market: {market_signal}"
        elif demand_signal == "Weakening" or market_signal == "Diverging":
            demand_supply = f"Demand signal: {demand_signal} · Market: {market_signal}"
        else:
            demand_supply = f"Demand signal: {demand_signal} · Market: {market_signal}"

        # ── STEP 7: FINAL OUTPUT ───────────────────────────────────
        return {
            "inference": {
                "trajectory":     trajectory,
                "score":          score,
                "signals": {
                    "demand":   demand_signal,
                    "growth":   growth_signal,
                    "market":   market_signal,
                    "risk":     risk_signal,
                },
                "chart_data":     chart_data,
                "drivers":        drivers,
                "demand_supply":  demand_supply,
                "benchmarks": {
                    "price": {
                        "local":    comp_avg,
                        "regional": reg_price.get("avg_price"),
                        "national": nat_price.get("avg_price"),
                        "local_yoy":    price_yoy,
                        "regional_yoy": reg_price.get("yoy_pct"),
                        "national_yoy": nat_price.get("yoy_pct"),
                        "source": "Land Registry HPI",
                    },
                    "rental": {
                        "local_yoy":    rental.get("latest_yoy"),
                        "regional_yoy": reg_rental.get("latest_yoy"),
                        "national_yoy": nat_rental.get("latest_yoy"),
                        "regional_rent_gbp": reg_rental.get("avg_rent_gbp"),
                        "source": "ONS PRMS",
                    },
                    "crime": {
                        "local_total":      int(crime_total),
                        "local_rate_per_1000": round(local_crime_rate, 2),
                        "national_rate_per_1000": national_crime_rate,
                        "crime_index":      crime_index,
                        "source": "Police.uk",
                    },
                    "liquidity": {
                        "annual_transactions": liquidity.get("local_annual"),
                        "quarterly_transactions": liquidity.get("local_qtly"),
                        "district": liquidity.get("district"),
                        "source": "Land Registry",
                    },
                    "population": {
                        "change_pct_5yr": pop.get("change_pct"),
                        "direction":      pop_dir,
                        "source": "ONS Mid-Year Estimates",
                    },
                    "yield": {
                        "local_pct":    yield_data.get("local_yield"),
                        "national_pct": yield_data.get("national_yield"),
                        "local_rent_gbp":    yield_data.get("local_rent_gbp"),
                        "local_price_gbp":   yield_data.get("local_price_gbp"),
                        "source": "ONS PRMS · Land Registry HPI",
                    },
                    "deprivation": {
                        "imd_decile": imd_data.get("decile"),
                        "imd_rank":   imd_data.get("rank"),
                        "source": imd_data.get("source", "MHCLG IMD 2019"),
                    },
                    "census": {
                        "private_rent_pct": _census_private_rent_pct,
                        "source": "ONS Census 2021 TS054",
                    },
                },
                "provenance":     "Land Registry · ONS Population · ONS PRMS · OSM · PlanningAlerts · Police.uk · MHCLG EPC · ONS Census 2021",
                "lad_code":       lad_code,
                "computed_at":    now_iso(),
                "data_availability_pct": round(_data_availability_pct, 2),
                "data_signals_present": _data_signals_present,
            }
        }

    except Exception as e:
        print(f"[ERROR] build_area_inference failed for {postcode}: {e}")
        return {
            "inference": {
                "trajectory":    "UNKNOWN",
                "score":         0,
                "signals":       {},
                "chart_data":    {"price": [], "demand": [], "growth": []},
                "drivers":       [],
                "demand_supply": "Insufficient data.",
                "provenance":    "Sources: Land Registry · ONS · OSM · Police · EPC · PlanningAlerts",
                "error":         str(e),
                "computed_at":   now_iso(),
            }
        }


def get_epc_data(postcode: str) -> Dict[str, Any]:
    """
    EPC data from Hetzner epc_certificates table.
    Returns rating distribution + dominant rating + habitable rooms for subject postcode.
    """
    retrieved = now_iso()
    try:
        pc = normalize_postcode(postcode)
        if not pc:
            return {"fetch_status": "skipped", "metrics": {}}

        # Get EPC data for the exact postcode and nearby (district level)
        district = pc.split()[0] if ' ' in pc else pc[:3]

        # Subject property EPC
        subject_rows = data_query(
            """SELECT current_energy_rating, number_habitable_rooms, total_floor_area,
                      property_type, built_form
               FROM public.epc_certificates
               WHERE postcode = %s
               ORDER BY lodgement_date DESC LIMIT 5""",
            (pc,)
        )

        # District-level distribution for area profile
        dist_rows = data_query(
            """SELECT current_energy_rating, COUNT(*) AS cnt
               FROM public.epc_certificates
               WHERE postcode LIKE %s
               AND current_energy_rating IS NOT NULL
               GROUP BY 1 ORDER BY 1""",
            (f"{district}%",)
        )

        rating_counts = {r["current_energy_rating"]: int(r["cnt"]) for r in dist_rows if r.get("current_energy_rating")}
        total = sum(rating_counts.values())
        dominant = max(rating_counts, key=rating_counts.get) if rating_counts else None

        # Subject property details
        subject_rating = None
        subject_rooms  = None
        subject_area   = None
        if subject_rows:
            subject_rating = subject_rows[0].get("current_energy_rating")
            subject_rooms  = subject_rows[0].get("number_habitable_rooms")
            subject_area   = subject_rows[0].get("total_floor_area")

        return {
            "fetch_status": "ok",
            "retrieved_at": retrieved,
            "source": "Hetzner epc_certificates · MHCLG EPC Register",
            "metrics": {
                "total_certificates": total,
                "dominant_rating": dominant,
                "rating_counts": rating_counts,
                "subject_rating": subject_rating,
                "subject_habitable_rooms": int(subject_rooms) if subject_rooms else None,
                "subject_floor_area": float(subject_area) if subject_area else None,
            }
        }

    except Exception as e:
        print(f"[WARN] get_epc_data failed for {postcode}: {e}")
        return {"fetch_status": "error", "error": str(e), "metrics": {}}


def get_planning_data(lat: Optional[float], lng: Optional[float], postcode: str = "") -> Dict[str, Any]:
    """
    PlanningAlerts.org API — free, no key required.
    Returns planning applications within 3-mile (4827m) radius in last 24 months.
    Returns count, types and recent activity for inference engine growth signal.
    """
    retrieved = now_iso()
    sources = [{"label": "PlanningAlerts.org", "url": "https://www.planningalerts.org.uk/"}]

    if lat is None or lng is None:
        return metric_unavailable("Planning data unavailable: coordinates not resolved.", sources, retrieved)

    try:
        # PlanningAlerts API — radius in metres, returns recent applications
        url = "https://www.planningalerts.org.uk/api/v2/applications"
        params = {
            "lat":    lat,
            "lng":    lng,
            "radius": 4827,  # 3 miles in metres
            "count":  100,
        }
        status, payload = _http_get_json(url, params=params, timeout=15)

        if status == 200 and isinstance(payload, dict):
            applications = payload.get("applications") or []
            if not applications:
                out = metric_ok(
                    f"No planning applications found within 3 miles of {postcode}.",
                    [], sources, retrieved, 0.7
                )
                out["metrics"] = {"total": 0, "new_build": 0, "change_of_use": 0, "other": 0}
                return out

            # Classify by type
            new_build     = 0
            change_of_use = 0
            other         = 0
            recent_24m    = 0
            cutoff        = datetime.utcnow() - timedelta(days=730)  # 24 months

            for app in applications:
                desc = str(app.get("description") or "").lower()
                date_str = app.get("date_scraped") or app.get("on_notice_from") or ""
                try:
                    app_date = datetime.fromisoformat(date_str[:10])
                    if app_date >= cutoff:
                        recent_24m += 1
                except Exception:
                    pass

                if any(kw in desc for kw in ["new dwelling", "new build", "erection of", "residential development"]):
                    new_build += 1
                elif any(kw in desc for kw in ["change of use", "conversion", "permitted development"]):
                    change_of_use += 1
                else:
                    other += 1

            total = len(applications)
            summary = (
                f"{total} planning applications within 3 miles. "
                f"{recent_24m} in last 24 months. "
                f"New build: {new_build}, Change of use: {change_of_use}, Other: {other}."
            )

            out = metric_ok(summary, applications[:10], sources, retrieved, 0.8)
            out["metrics"] = {
                "total":          total,
                "recent_24m":     recent_24m,
                "new_build":      new_build,
                "change_of_use":  change_of_use,
                "other":          other,
            }
            return out

        elif status == 404:
            out = metric_ok(
                f"No planning applications found within 3 miles of {postcode}.",
                [], sources, retrieved, 0.7
            )
            out["metrics"] = {"total": 0, "new_build": 0, "change_of_use": 0, "other": 0}
            return out
        else:
            return metric_unavailable(f"Planning API returned status {status}.", sources, retrieved)

    except Exception as e:
        print(f"[WARN] Planning fetch failed for {lat},{lng}: {e}")

    return metric_unavailable("Planning data temporarily unavailable.", sources, retrieved)


def get_flood_risk(lat: Optional[float], lng: Optional[float], postcode: str = "") -> Dict[str, Any]:
    """
    Environment Agency Flood Map for Planning API — free, no key required.
    Returns flood risk zone (1=low, 2=medium, 3=high) for given coordinates.
    Zone 1: <0.1% annual probability. Zone 2: 0.1-1%. Zone 3: >1%.
    """
    retrieved = now_iso()
    sources = [{"label": "Environment Agency Flood Map", "url": "https://environment.data.gov.uk/flood-monitoring/"}]

    if lat is None or lng is None:
        return metric_unavailable("Flood risk data not available: coordinates not resolved.", sources, retrieved)

    try:
        # EA Flood Zone endpoint — returns flood zone polygons containing the point
        url = f"https://environment.data.gov.uk/flood-monitoring/id/floodAreas?lat={lat}&long={lng}&dist=0.5"
        status, payload = _http_get_json(url, timeout=10)

        if status == 200 and isinstance(payload, dict):
            items = payload.get("items") or []
            if isinstance(items, list) and len(items) > 0:
                # Find highest risk zone
                zones = []
                for item in items:
                    label = str(item.get("label","") or item.get("notation","") or "")
                    if "3" in label or "high" in label.lower():
                        zones.append(3)
                    elif "2" in label or "medium" in label.lower():
                        zones.append(2)
                    elif "1" in label or "low" in label.lower():
                        zones.append(1)
                max_zone = max(zones) if zones else 1
                zone_desc = {
                    1: "Zone 1 — annual flood probability below 0.1%. Minimal insurance impact.",
                    2: "Zone 2 — annual flood probability 0.1%–1%. Standard flood insurance recommended.",
                    3: "Zone 3 — annual flood probability above 1%. Significant insurance cost implications. May affect mortgage availability."
                }.get(max_zone, "Zone 1")
                out = metric_ok(
                    zone_desc,
                    [{"zone": max_zone, "areas": len(items)}],
                    sources, retrieved, 0.9
                )
                out["metrics"] = {"zone": max_zone, "flood_areas": len(items)}
                return out
            else:
                # No flood areas — Zone 1 (minimal risk)
                out = metric_ok(
                    "Zone 1 — no flood risk areas recorded at this location. Annual probability below 0.1%.",
                    [{"zone": 1, "areas": 0}],
                    sources, retrieved, 0.85
                )
                out["metrics"] = {"zone": 1, "flood_areas": 0}
                return out
    except Exception as e:
        print(f"[WARN] Flood risk fetch failed for {lat},{lng}: {e}")

    return metric_unavailable(
        f"Flood risk data temporarily unavailable. Verify via Environment Agency Flood Map before bidding.",
        sources, retrieved
    )


def get_broadband_data(postcode: str) -> Dict[str, Any]:
    retrieved = now_iso()
    pc = normalize_postcode(postcode)
    district = postcode_district(pc)

    if not pc:
        return metric_unavailable(
            "Broadband data not available: no postcode provided.",
            [{"label": "Ofcom", "url": "https://www.ofcom.org.uk/"}],
            retrieved,
        )

    if BROADBAND_PROVIDER == "supabase":
        if not supabase:
            return metric_unavailable(
                "Broadband provider set to supabase but Supabase is not configured on server.",
                [{"label": "Supabase", "url": "https://supabase.com/"}],
                retrieved,
                extra_metrics={"postcode": pc},
            )

        if not BROADBAND_SUPABASE_TABLE:
            return metric_missing_provider(
                "Broadband provider set to supabase, but BROADBAND_SUPABASE_TABLE is not set.",
                [{"label": "Ofcom", "url": "https://www.ofcom.org.uk/"}],
                retrieved,
                extra_metrics={"postcode": pc},
            )

        try:
            q = (
                supabase.table(BROADBAND_SUPABASE_TABLE)
                .select("*")
                .eq("postcode", pc)
                .limit(BROADBAND_MAX_RESULTS)
            )
            res = q.execute()
            rows = res.data if hasattr(res, "data") else None
            if not isinstance(rows, list):
                rows = []

            if not rows and district:
                q2 = (
                    supabase.table(BROADBAND_SUPABASE_TABLE)
                    .select("*")
                    .ilike("postcode", f"{district}%")
                    .limit(BROADBAND_MAX_RESULTS)
                )
                res2 = q2.execute()
                rows = res2.data if hasattr(res2, "data") else rows
                if not isinstance(rows, list):
                    rows = []

            summary = (
                f"Broadband records found for {pc}: {len(rows)}."
                if rows else
                f"No broadband records found for {pc}."
            )

            sources = [{"label": "Supabase (broadband)", "url": f"{SUPABASE_URL}" if SUPABASE_URL else "https://supabase.com/"}]
            out = metric_ok(summary, rows, sources, retrieved, BROADBAND_CONFIDENCE_VALUE if rows else 0.0)
            out["metrics"] = {"provider": "supabase", "district": district, "limit": BROADBAND_MAX_RESULTS}
            return out

        except Exception as e:
            return metric_unavailable(
                f"Broadband query failed: {str(e)}",
                [{"label": "Supabase", "url": "https://supabase.com/"}],
                retrieved,
                extra_metrics={"postcode": pc, "district": district},
            )

    return metric_missing_provider(
        "Broadband provider not configured. Set BROADBAND_PROVIDER=supabase and BROADBAND_SUPABASE_TABLE=... ",
        [
            {"label": "Ofcom", "url": "https://www.ofcom.org.uk/"},
            {"label": "ThinkBroadband", "url": "https://www.thinkbroadband.com/"},
        ],
        retrieved,
        extra_metrics={"postcode": pc, "district": district},
    )


def _median_int(values: List[int]) -> Optional[int]:
    vs = sorted([v for v in values if isinstance(v, int)])
    if not vs:
        return None
    mid = len(vs) // 2
    if len(vs) % 2 == 1:
        return vs[mid]
    return int((vs[mid - 1] + vs[mid]) / 2)


# ── TASK 2: density-aware radius compression ────────────────────────────────
# Measured: a 3-mile radius in dense London postcode areas pulls 13k-14k
# transactions and the comps RPC times out cold (NW9/E14/N1/E1 verified).
# Compress the START radius for these areas BEFORE RPC execution. Only the
# London postcode areas are included — that is the measured timeout zone;
# other areas are left on the configured default (no evidence they time out).
#
# Radius set to 0.5 mile (805 m) per cold-cache measurement: 1.0 mile still
# returned 2.3-3.3 s for E14/N1/E1 (above the sub-2s bar); 0.5 mile delivers
# NW9 676 ms, N1 820 ms, E14 1,535 ms (cold reads present) — all sub-2s.
_DENSE_POSTCODE_AREAS = frozenset({"E", "EC", "N", "NW", "SE", "SW", "W", "WC"})
_DENSE_RADIUS_MILES = 0.5

def _density_tier_radius(pc: str) -> Optional[float]:
    """Compressed start radius (miles) for dense-market postcodes, else None
    (None => use the configured default). Decided before RPC execution."""
    if not pc:
        return None
    _m = re.match(r"^([A-Z]{1,2})", str(pc).upper().strip())
    _area = _m.group(1) if _m else ""
    if _area in _DENSE_POSTCODE_AREAS:
        return _DENSE_RADIUS_MILES
    return None


def get_housing_data(postcode: str, radius_miles: Optional[float] = None, limit: Optional[int] = None, property_type: Optional[str] = None, guide_price: Optional[float] = None, subject_tenure_hint: Optional[str] = None) -> Dict[str, Any]:
    retrieved = now_iso()
    pc = normalize_postcode(postcode)

    sources = [
        {"label": "HM Land Registry (Price Paid)", "url": "https://www.gov.uk/government/collections/price-paid-data"},
        {"label": "Supabase (Postgres)", "url": f"{SUPABASE_URL}" if SUPABASE_URL else "https://supabase.com/"},
    ]

    if not pc:
        return metric_unavailable("Housing data not available: no postcode provided.", sources, retrieved)

    r_miles = radius_miles if isinstance(radius_miles, (int, float)) and radius_miles > 0 else HOUSING_DEFAULT_RADIUS_MILES
    # Task 2 — density-aware radius compression. Applied ONLY when the caller
    # did not pass an explicit radius (debug endpoints still override freely).
    if not (isinstance(radius_miles, (int, float)) and radius_miles > 0):
        _tier_r = _density_tier_radius(pc)
        if _tier_r is not None:
            r_miles = _tier_r
    lim = limit if isinstance(limit, int) and limit > 0 else HOUSING_DEFAULT_LIMIT

    lim = max(1, min(int(lim), HOUSING_MAX_LIMIT))
    r_miles = max(0.25, min(float(r_miles), 10.0))

    if HOUSING_PROVIDER != "supabase_rpc":
        return metric_missing_provider(
            "Housing provider not configured. Set HOUSING_PROVIDER=supabase_rpc.",
            sources,
            retrieved,
            extra_metrics={"postcode": pc, "radius_miles": r_miles, "limit": lim},
        )

    if not supabase:
        return metric_unavailable(
            "Housing provider set to supabase_rpc but Supabase is not configured on server.",
            [{"label": "Supabase", "url": "https://supabase.com/"}],
            retrieved,
            extra_metrics={"postcode": pc, "radius_miles": r_miles, "limit": lim},
        )

    payload = {"in_postcode": pc, "in_radius_miles": r_miles, "in_limit": lim}

    try:
        res = supabase.rpc(HOUSING_RPC_NAME, payload).execute()
        rows = res.data if hasattr(res, "data") else None
        if not isinstance(rows, list):
            rows = []

        # ── T-1: dedupe identical PPD entries post-RPC ──────────────────────
        # A single conveyance can appear as multiple Price Paid Data rows
        # (separate title entries — flat + parking, etc.) sharing identical
        # (postcode, price, date_of_transfer). They are not two distinct
        # transactions. Dedupe at source so every downstream statistic
        # (mean, median, IQR, similarity, area-normalisation) operates on
        # unique transactions only.
        _raw_rpc_count = len(rows)
        if rows:
            _seen_keys: set = set()
            _deduped: list = []
            for _r in rows:
                _rd = _r if isinstance(_r, dict) else {}
                _k = (
                    _rd.get("postcode"),
                    _rd.get("price"),
                    str(_rd.get("date_of_transfer") or ""),
                )
                if _k in _seen_keys:
                    continue
                _seen_keys.add(_k)
                _deduped.append(_r)
            rows = _deduped
        _duplicates_removed = _raw_rpc_count - len(rows)

        # ── S-2: PPD Category-B exclusion ──────────────────────────────────
        # HM Land Registry classifies Price Paid records as:
        #   Cat-A = standard arm's-length residential sale (institutional input)
        #   Cat-B = non-standard: repossession, BTL portfolio transfer,
        #           sub-market sale, transfer to corporate vehicle, etc.
        #           (NOT comparable evidence)
        # Cat-A is the only category an institutional valuer would admit.
        # Surfaced by S-1 (RPC now returns ppd_category_type); this patch is
        # the first consumer of that field. Null treated as A (literal spec:
        # drop only where ppd_category_type == 'B').
        _cat_b_excluded_count = 0
        if rows:
            _cat_a_only: list = []
            for _r in rows:
                _rd = _r if isinstance(_r, dict) else {}
                _cat = str(_rd.get("ppd_category_type") or "").upper()
                if _cat == "B":
                    _cat_b_excluded_count += 1
                    continue
                _cat_a_only.append(_r)
            rows = _cat_a_only

        if not rows:
            out = metric_unavailable(
                f"No sold comparables returned within {r_miles} miles for {pc}.",
                sources,
                retrieved,
                extra_metrics={"postcode": pc, "radius_miles": r_miles, "limit": lim, "rpc": HOUSING_RPC_NAME},
            )
            out["metrics"]["payload"] = payload
            return out

        # ══════════════════════════════════════════════════════════════════════
        # SIMILARITY-WEIGHTED COMPARABLE ENGINE
        # ══════════════════════════════════════════════════════════════════════
        # Methodology: property-similarity filtering + dimensional scoring
        # + HPI temporal normalisation + floor-area price normalisation
        # + IQR outlier rejection + adaptive radius
        # + explicit uncertainty disclosure
        #
        # All dimensions derived from verified data sources:
        #   property_type   → price_paid_raw_2025.property_type
        #   floor_area      → epc_certificates.total_floor_area
        #   habitable_rooms → epc_certificates.number_habitable_rooms
        #   tenure          → price_paid_raw_2025.duration
        #   old_new         → price_paid_raw_2025.old_new
        #   age_band        → epc_certificates.construction_age_band
        #   HPI multiplier  → uk_hpi_monthly.annual_change (LAD level)
        #   miles           → housing_comps_v1 RPC (haversine from subject)
        # ══════════════════════════════════════════════════════════════════════

        import datetime as _dt

        # ── AUDIT TRAIL ─────────────────────────────────────────────────────
        # Every decision recorded for lineage disclosure
        _audit: dict = {
            "postcode": pc,
            "raw_rpc_count": _raw_rpc_count,
            "duplicates_removed": _duplicates_removed,
            "cat_b_excluded_count": _cat_b_excluded_count,
            "initial_rpc_count": len(rows),
            "radius_miles": r_miles,
            "filters_applied": [],
            "filters_skipped": [],
            "outliers_rejected": 0,
            "hpi_adjusted_count": 0,
            "area_normalised_count": 0,
            "tenure_resolved_count": 0,
            "new_build_resolved_count": 0,
            "final_comp_count": 0,
            "methodology_degraded": False,
            "insufficient_evidence": False,
            "warnings": [],
        }

        # ── SUBJECT PROPERTY DATA COLLECTION ────────────────────────────────
        _pc_norm = normalize_postcode(postcode)

        # Subject: EPC data (single lookup, reused across all dimensions)
        _subject_epc = {}
        try:
            _sepc = data_query(
                """SELECT number_habitable_rooms, total_floor_area,
                          construction_age_band, current_energy_rating, uprn
                   FROM public.epc_certificates
                   WHERE postcode = %s
                   AND (number_habitable_rooms IS NOT NULL OR total_floor_area IS NOT NULL)
                   ORDER BY lodgement_date DESC LIMIT 1""",
                (_pc_norm,)
            )
            if _sepc:
                _subject_epc = _sepc[0]
        except Exception as _e:
            _audit["warnings"].append(f"subject_epc_lookup_failed: {_e}")

        _subject_rooms      = int(_subject_epc.get("number_habitable_rooms") or 0) or None
        _subject_area       = float(_subject_epc.get("total_floor_area") or 0) or None
        _subject_band       = str(_subject_epc.get("construction_age_band") or "").strip().upper() or None
        _subject_uprn       = str(_subject_epc.get("uprn") or "").strip() or None

        # ── I-1: Subject tenure resolution (authoritative hierarchy) ────────
        # Resolution precedence:
        #   1. Legal-pack / LLM-extracted tenure (caller-supplied hint sourced
        #      from summary_json.property.tenure — the canonical persisted
        #      LLM extraction from the legal pack)
        #   2. (The brief's tier 2 — persisted summary_json.property.tenure —
        #      is identical in source to tier 1: the caller passes either the
        #      fresh LLM extraction OR the persisted value into the same
        #      `subject_tenure_hint` parameter. Function consumes both
        #      identically; precedence is the caller's concern.)
        #   3. price_paid_raw_2025 postcode majority-vote proxy (fallback,
        #      preserved verbatim from prior implementation)
        #   4. Unresolved (None) — downstream Step-4 emits the canonical
        #      "subject_tenure_unknown: tenure filter skipped" warning
        #
        # The legal pack is the authoritative title-document source under any
        # RICS-aligned framework; the postcode majority-vote is a proxy that
        # silently fails on rural / sparsely-sold postcodes where
        # price_paid_raw_2025 has no rows for the subject postcode. This
        # block consults the hint first so the proxy is consulted only when
        # the legal pack provided no admissible value.
        _subject_tenure = None
        if subject_tenure_hint:
            _h = str(subject_tenure_hint).strip().upper()
            if _h in ("F", "FH", "FREEHOLD"):
                _subject_tenure = "F"
            elif _h in ("L", "LH", "LEASEHOLD"):
                _subject_tenure = "L"
            # Other values ("Unknown", "Mixed", "Commonhold", etc.) fall
            # through to tier 3 — conservative posture, since the Step-4
            # filter expects F/L only.
        if _subject_tenure is None:
            # Tier 3: postcode majority-vote proxy (preserved verbatim)
            try:
                _tr = data_query(
                    """SELECT duration, COUNT(*) AS cnt
                       FROM public.price_paid_raw_2025
                       WHERE postcode = %s AND duration IN ('F','L')
                       GROUP BY duration ORDER BY cnt DESC LIMIT 1""",
                    (_pc_norm,)
                )
                if _tr:
                    _subject_tenure = str(_tr[0].get("duration") or "").upper() or None
            except Exception as _e:
                _audit["warnings"].append(f"subject_tenure_lookup_failed: {_e}")
        # Tier 4: _subject_tenure remains None → the existing Step-4 guard
        # emits the canonical "subject_tenure_unknown: tenure filter skipped"
        # warning and the tenure-match similarity component at Step 8 stays
        # dormant for this deal (unchanged behaviour for unresolved cases).

        # Subject: new-build status from price_paid
        _subject_old_new = None
        try:
            _nbr = data_query(
                """SELECT old_new FROM public.price_paid_raw_2025
                   WHERE postcode = %s AND old_new IN ('Y','N')
                   ORDER BY date_of_transfer DESC LIMIT 1""",
                (_pc_norm,)
            )
            if _nbr:
                _subject_old_new = str(_nbr[0]["old_new"]).upper()
        except Exception as _e:
            _audit["warnings"].append(f"subject_new_build_lookup_failed: {_e}")

        # Subject: LAD code for HPI temporal adjustment
        _lad_code_for_hpi = _get_lad_code_for_postcode(_pc_norm)
        _hpi_yoy = None  # latest annual_change for this LAD
        if _lad_code_for_hpi:
            try:
                _hpi_rows = supabase_data_query(
                    "SELECT annual_change FROM public.uk_hpi_monthly WHERE area_code = %s ORDER BY date DESC LIMIT 1",
                    (_lad_code_for_hpi,)
                )
                if _hpi_rows:
                    _hpi_yoy = safe_float(_hpi_rows[0].get("annual_change"))
            except Exception as _e:
                _audit["warnings"].append(f"hpi_yoy_lookup_failed: {_e}")

        # ── STEP 1: PROPERTY TYPE FILTER ────────────────────────────────────
        pt_filter = (property_type or "").strip().upper()
        if pt_filter and pt_filter in ("D", "S", "T", "F", "O"):
            _pt_matched = [r for r in rows if str(r.get("property_type") or "").upper() == pt_filter]
            if len(_pt_matched) >= 5:
                rows = _pt_matched
                _audit["filters_applied"].append(f"property_type={pt_filter}:{len(rows)}_comps")
            else:
                _audit["filters_skipped"].append(f"property_type={pt_filter}:only_{len(_pt_matched)}_matched")
                _audit["methodology_degraded"] = True
                _audit["warnings"].append(f"Cross-type contamination: insufficient {pt_filter} comps ({len(_pt_matched)}), using all property types")

        # ── STEP 2: EPC ENRICHMENT ───────────────────────────────────────────
        # Enrich comps with EPC data via UPRN (deterministic) then postcode fallback
        _band_order = ["A","B","C","D","E","F","G","H","I","J","K","L"]
        _uprn_enriched_count = 0

        for _r in rows:
            if _r.get("habitable_rooms") is not None and _r.get("floor_area") is not None:
                continue  # already enriched by prior UPRN path
            _comp_pc = normalize_postcode(str(_r.get("postcode") or ""))
            if not _comp_pc:
                continue
            try:
                _cepc = data_query(
                    """SELECT number_habitable_rooms, total_floor_area,
                              construction_age_band, current_energy_rating
                       FROM public.epc_certificates
                       WHERE postcode = %s
                       AND (number_habitable_rooms IS NOT NULL OR total_floor_area IS NOT NULL)
                       ORDER BY lodgement_date DESC LIMIT 1""",
                    (_comp_pc,)
                )
                if _cepc:
                    _r["habitable_rooms"]       = _cepc[0].get("number_habitable_rooms")
                    _r["floor_area"]            = float(_cepc[0].get("total_floor_area") or 0) or None
                    _r["construction_age_band"] = str(_cepc[0].get("construction_age_band") or "").strip().upper() or None
                    _r["energy_rating"]         = _cepc[0].get("current_energy_rating")
                    _uprn_enriched_count += 1
            except Exception:
                pass

        # ── STEP 3: HABITABLE ROOM FILTER ───────────────────────────────────
        if _subject_rooms:
            _room_matched = [
                r for r in rows
                if r.get("habitable_rooms") is not None
                and abs(int(r["habitable_rooms"]) - _subject_rooms) <= 1
            ]
            if len(_room_matched) >= 5:
                rows = _room_matched
                _audit["filters_applied"].append(f"habitable_rooms={_subject_rooms}±1:{len(rows)}_comps")
            else:
                _audit["filters_skipped"].append(f"habitable_rooms:only_{len(_room_matched)}_matched")
                _audit["methodology_degraded"] = True
                _audit["warnings"].append(f"Room-size contamination: insufficient ±1-room matches ({len(_room_matched)}), bedroom filter skipped")
        else:
            _audit["warnings"].append("subject_rooms_unknown: bedroom filter skipped — EPC not lodged or not found")

        # ── STEP 4: NEW BUILD ROUTING ────────────────────────────────────────
        if _subject_old_new == "N":
            _nb_only = [r for r in rows if str(r.get("old_new") or "").upper() == "N"]
            if len(_nb_only) >= 5:
                rows = _nb_only
                _audit["filters_applied"].append(f"new_build_preferred:{len(rows)}_nb_comps")
            else:
                _audit["filters_skipped"].append(f"new_build_preferred:only_{len(_nb_only)}_nb_comps")
                _audit["warnings"].append(f"New-build subject: only {len(_nb_only)} new-build comps available, using mixed stock (ceiling may be understated)")
        elif _subject_old_new == "Y":
            _established = [r for r in rows if str(r.get("old_new") or "").upper() != "N"]
            if len(_established) >= 5:
                rows = _established
                _audit["filters_applied"].append(f"new_build_excluded:{len(rows)}_established_comps")
            else:
                _audit["filters_skipped"].append("new_build_excluded:insufficient_established")
        else:
            # Unknown: soft exclude new builds if majority are established
            _established = [r for r in rows if str(r.get("old_new") or "").upper() != "N"]
            if len(_established) >= 5:
                rows = _established

        # ── STEP 5: TENURE FILTER ────────────────────────────────────────────
        if _subject_tenure in ("F", "L"):
            _tenure_matched = [r for r in rows if str(r.get("duration") or "").upper() == _subject_tenure]
            if len(_tenure_matched) >= 5:
                rows = _tenure_matched
                _audit["filters_applied"].append(f"tenure={'Freehold' if _subject_tenure=='F' else 'Leasehold'}:{len(rows)}_comps")
            else:
                _audit["filters_skipped"].append(f"tenure:only_{len(_tenure_matched)}_matched")
                _audit["warnings"].append(f"Tenure contamination: cross-tenure comps included ({len(_tenure_matched)} same-tenure comps insufficient)")
        else:
            _audit["warnings"].append("subject_tenure_unknown: tenure filter skipped")

        # ── STEP 6: CONSTRUCTION AGE BAND FILTER ─────────────────────────────
        if _subject_band and _subject_band in _band_order:
            _sub_idx = _band_order.index(_subject_band)
            _age_matched = [
                r for r in rows
                if (r.get("construction_age_band") in _band_order
                    and abs(_band_order.index(r["construction_age_band"]) - _sub_idx) <= 2)
                or r.get("construction_age_band") is None  # include unknown-age comps
            ]
            if len(_age_matched) >= 5:
                rows = _age_matched
                _audit["filters_applied"].append(f"age_band={_subject_band}±2:{len(rows)}_comps")
            else:
                _audit["filters_skipped"].append(f"age_band:only_{len(_age_matched)}_matched")
        else:
            _audit["warnings"].append("subject_age_band_unknown: age filter skipped — EPC construction age not found")

        # ── STEP 7: IQR OUTLIER REJECTION ────────────────────────────────────
        _prices_raw = sorted([r.get("price") or 0 for r in rows if r.get("price")])
        if len(_prices_raw) >= 6:
            _n = len(_prices_raw)
            _q1 = _prices_raw[_n // 4]
            _q3 = _prices_raw[(3 * _n) // 4]
            _iqr = _q3 - _q1
            _fence_lo = _q1 - 1.5 * _iqr
            _fence_hi = _q3 + 1.5 * _iqr
            _iqr_filtered = [r for r in rows if _fence_lo <= (r.get("price") or 0) <= _fence_hi]
            if len(_iqr_filtered) >= 5:
                _audit["outliers_rejected"] = len(rows) - len(_iqr_filtered)
                rows = _iqr_filtered
                if _audit["outliers_rejected"] > 0:
                    _audit["filters_applied"].append(f"iqr_outlier_rejection:{_audit['outliers_rejected']}_removed")
            else:
                _audit["warnings"].append(f"IQR rejection skipped: only {len(_iqr_filtered)} comps survive fence lo={_fence_lo:.0f} hi={_fence_hi:.0f}")

        # ── STEP 8: HPI TEMPORAL NORMALISATION ───────────────────────────────
        # Adjusts historical nominal prices to current-equivalent using LAD-level HPI
        # Formula: adjusted = nominal × (1 + hpi_yoy/100)^(age_months/12)
        # Rationale: a comp from 18 months ago at £350k in a +3.5% YoY area
        #            has current-equivalent value of £350k × 1.035^1.5 ≈ £368k
        # Directionally valid; residual error from non-linear appreciation not addressed
        _now = _dt.datetime.utcnow()
        _hpi_adjusted_count = 0
        for _r in rows:
            _tx_date_str = str(_r.get("date_of_transfer") or "")[:10]
            _age_months = None
            try:
                _tx_dt = _dt.datetime.strptime(_tx_date_str, "%Y-%m-%d")
                _age_months = (_now - _tx_dt).days / 30.44
            except Exception:
                pass
            _r["age_months"] = _age_months
            _r["nominal_price"] = _r.get("price")
            if _hpi_yoy is not None and _age_months is not None and _age_months > 3:
                _age_years = _age_months / 12.0
                _hpi_mult = (1.0 + _hpi_yoy / 100.0) ** _age_years
                _adj = int((_r.get("price") or 0) * _hpi_mult)
                _r["hpi_adjusted_price"] = _adj
                _r["hpi_multiplier"] = round(_hpi_mult, 4)
                _hpi_adjusted_count += 1
            else:
                _r["hpi_adjusted_price"] = _r.get("price")
                _r["hpi_multiplier"] = 1.0
        _audit["hpi_adjusted_count"] = _hpi_adjusted_count
        if _hpi_yoy is None:
            _audit["warnings"].append("hpi_temporal_adjustment_skipped: no LAD HPI data — nominal prices used")

        # ── STEP 9: SIMILARITY SCORING ───────────────────────────────────────
        # Weighted score: recency × proximity × type-match × room-match × tenure-match × age-match
        # All weights are explicit and disclosed in comp output
        def _similarity_score(_r: dict) -> float:
            _s = 1.0
            _components = {}

            # Recency weight (applied to HPI-adjusted prices — ordering by recency still valid)
            _am = _r.get("age_months")
            if _am is not None:
                if _am <= 6:    _rw = 1.00
                elif _am <= 12: _rw = 0.90
                elif _am <= 18: _rw = 0.78
                else:           _rw = 0.62
            else:
                _rw = 0.65
            _s *= _rw
            _components["recency"] = round(_rw, 2)

            # Proximity weight: linear decay
            _mi = safe_float(_r.get("miles"))
            if isinstance(_mi, float) and _mi >= 0:
                _pw = max(0.10, 1.0 - (_mi / r_miles) * 0.65)
            else:
                _pw = 0.50
            _s *= _pw
            _components["proximity"] = round(_pw, 2)

            # Property type exact match
            if pt_filter and str(_r.get("property_type") or "").upper() == pt_filter:
                _s *= 1.10
                _components["type_match"] = 1.10
            else:
                _components["type_match"] = 1.00

            # Room similarity (if subject rooms known)
            if _subject_rooms and _r.get("habitable_rooms") is not None:
                _rd = abs(int(_r["habitable_rooms"]) - _subject_rooms)
                if _rd == 0:   _rmw = 1.00
                elif _rd == 1: _rmw = 0.88
                else:          _rmw = 0.70
                _s *= _rmw
                _components["room_match"] = round(_rmw, 2)
            else:
                _components["room_match"] = None

            # Tenure match
            if _subject_tenure and _r.get("duration"):
                if str(_r["duration"]).upper() == _subject_tenure:
                    _s *= 1.05
                    _components["tenure_match"] = 1.05
                else:
                    _s *= 0.82
                    _components["tenure_match"] = 0.82
            else:
                _components["tenure_match"] = None

            # New build alignment
            _ron = str(_r.get("old_new") or "").upper()
            if _subject_old_new and _ron:
                if _ron == _subject_old_new:
                    _s *= 1.05
                    _components["build_match"] = 1.05
                else:
                    _s *= 0.80
                    _components["build_match"] = 0.80
            else:
                _components["build_match"] = None

            _r["_similarity_score"] = round(_s, 4)
            _r["_score_components"] = _components
            return _s

        rows_scored = sorted(rows, key=_similarity_score, reverse=True)
        rows = rows_scored[:10]

        # ── STEP 10: FLOOR-AREA PRICE NORMALISATION ───────────────────────────
        # Uses HPI-adjusted prices as the base for normalisation
        # Formula: normalised_price = hpi_adjusted_price × (subject_area / comp_area)
        # Only applied when >= 5 comps have known floor area
        _normalised_prices = []
        _area_normalised_count = 0
        if _subject_area and _subject_area > 0:
            for _r in rows:
                _comp_area = _r.get("floor_area")
                _hpi_adj = _r.get("hpi_adjusted_price") or _r.get("price")
                if _comp_area and _comp_area > 0 and _hpi_adj:
                    _norm_factor = _subject_area / _comp_area
                    _norm_price = int(_hpi_adj * _norm_factor)
                    _r["price_normalised"] = _norm_price
                    _r["normalisation_factor"] = round(_norm_factor, 3)
                    _normalised_prices.append(_norm_price)
                    _area_normalised_count += 1
                else:
                    _r["price_normalised"] = _hpi_adj
                    _r["normalisation_factor"] = None
        else:
            for _r in rows:
                _r["price_normalised"] = _r.get("hpi_adjusted_price") or _r.get("price")
                _r["normalisation_factor"] = None
            _audit["warnings"].append("floor_area_normalisation_skipped: subject floor area unknown")

        _use_normalised = len(_normalised_prices) >= 5
        _audit["area_normalised_count"] = _area_normalised_count
        if not _use_normalised and _subject_area:
            _audit["warnings"].append(f"floor_area_normalisation_skipped: only {_area_normalised_count}/10 comps have floor area data")

        # ── S-3: post-normalisation outlier bounds ──────────────────────────
        # Linear £/m² area-normalisation can produce extreme synthetic prices
        # in prime central London where comparable floor areas vary by an
        # order of magnitude. The Step 9 IQR rejection operates on nominal
        # prices, so a post-normalisation [0.5×, 2×] band around the
        # normalised median protects downstream median / mean / variance
        # statistics from arithmetic artefacts of the normalisation function.
        # Acts on price_normalised which Step 10 guarantees is set on every
        # row (area-normalised, HPI-adjusted, or nominal fallback).
        _post_norm_excluded = 0
        if rows:
            _norm_vals: list = []
            for _r in rows:
                if not isinstance(_r, dict):
                    continue
                _pn = safe_int(_r.get("price_normalised"))
                if isinstance(_pn, int) and _pn > 0:
                    _norm_vals.append(_pn)
            _norm_median = _median_int(_norm_vals)
            if _norm_median and _norm_median > 0:
                _lower_bound = 0.5 * _norm_median
                _upper_bound = 2.0 * _norm_median
                _bounded: list = []
                for _r in rows:
                    if not isinstance(_r, dict):
                        _bounded.append(_r)
                        continue
                    _pn = safe_int(_r.get("price_normalised"))
                    if not isinstance(_pn, int) or _pn <= 0:
                        # Defensive: keep rows without a positive normalised
                        # price. Step 9 IQR has already handled nominal
                        # outliers; S-3 drops only on normalised excess.
                        _bounded.append(_r)
                        continue
                    if _pn < _lower_bound or _pn > _upper_bound:
                        _post_norm_excluded += 1
                        continue
                    _bounded.append(_r)
                rows = _bounded
        _audit["post_normalisation_excluded_count"] = _post_norm_excluded

        # ── STEP 11: COMPUTE PRICES FOR MEDIAN ──────────────────────────────
        prices: list = []
        ptypes: dict = {}
        miles_list: list = []
        _norm_coverage_pct = round(_area_normalised_count / max(len(rows), 1) * 100, 1)

        for r in rows:
            if not isinstance(r, dict):
                continue
            # Price hierarchy: normalised > HPI-adjusted > nominal
            if _use_normalised and r.get("price_normalised") is not None:
                pr = safe_int(r.get("price_normalised"))
            elif r.get("hpi_adjusted_price") is not None:
                pr = safe_int(r.get("hpi_adjusted_price"))
            else:
                pr = safe_int(r.get("price"))
            if isinstance(pr, int) and pr > 0:
                prices.append(pr)
            pt = map_property_type_label(r.get("property_type"))
            if pt:
                ptypes[pt] = ptypes.get(pt, 0) + 1
            mi = safe_float(r.get("miles"))
            if isinstance(mi, float):
                miles_list.append(mi)

        _audit["final_comp_count"] = len(rows)

        # ── S-5 (part B): tenure / new-build resolution audit counters ──────
        # S-1 surfaced duration + old_new on every comp. Where these are
        # non-null, the engine's Step-3 / Step-4 filters and Step-8 similarity
        # components had real signal to operate on. Counting per-comp
        # resolution here makes the institutional purity of the comp set
        # auditable from area_json downstream (Verdict caveats, Workbench
        # confidence display).
        _tenure_resolved = 0
        _new_build_resolved = 0
        for _r in rows:
            if not isinstance(_r, dict):
                continue
            _d = str(_r.get("duration") or "").upper()
            if _d in ("F", "L"):
                _tenure_resolved += 1
            _n = str(_r.get("old_new") or "").upper()
            if _n in ("Y", "N"):
                _new_build_resolved += 1
        _audit["tenure_resolved_count"] = _tenure_resolved
        _audit["new_build_resolved_count"] = _new_build_resolved

        # ── STEP 12: INSUFFICIENT EVIDENCE GATE ─────────────────────────────
        # Refuse high-confidence valuation when evidence is materially inadequate
        _price_variance_pct = 0.0
        if len(prices) >= 3:
            _sorted_prices = sorted(prices)
            _med_p = _sorted_prices[len(_sorted_prices) // 2]
            _iqr_p = _sorted_prices[int(len(_sorted_prices)*0.75)] - _sorted_prices[int(len(_sorted_prices)*0.25)]
            _price_variance_pct = (_iqr_p / max(_med_p, 1)) * 100

        _evidence_insufficient = (
            len(rows) < 3
            or (len(rows) < 5 and _audit["methodology_degraded"])
            or _price_variance_pct > 60
        )
        if _evidence_insufficient:
            _audit["insufficient_evidence"] = True
            _audit["warnings"].append(
                f"INSUFFICIENT_COMPARABLE_EVIDENCE: {len(rows)} comps, "
                f"variance={_price_variance_pct:.0f}%, degraded={_audit['methodology_degraded']}"
            )

        med = _median_int(prices)
        # T-3: arithmetic mean on the same price stream the median operates on.
        # Until now metrics.average_price held the median (mis-labelled), so the
        # two keys always returned the same value. Computing the mean here makes
        # the two reported statistics distinct and label-honest. Same input
        # stream, different statistic — no cross-stream mixing. The persist
        # ceiling's base-statistic choice is unchanged (tracked as D4/D-8).
        _mean_val = (sum(prices) / len(prices)) if prices else None
        avg = int(round(_mean_val)) if _mean_val is not None else None
        min_m = min(miles_list) if miles_list else None
        max_m = max(miles_list) if miles_list else None

        pt_parts = [f"{k}:{ptypes[k]}" for k in sorted(ptypes.keys())]
        pt_str = ", ".join(pt_parts) if pt_parts else "n/a"

        # ── COMP OUTPUT ENRICHMENT ────────────────────────────────────────────
        # Every comp exposes full lineage: nominal, HPI-adjusted, normalised, factors, score
        for r in rows:
            r["_lineage"] = {
                "nominal_price":             r.get("nominal_price") or r.get("price"),
                "hpi_adjusted_price":         r.get("hpi_adjusted_price"),
                "hpi_multiplier":             r.get("hpi_multiplier"),
                "hpi_yoy_applied":            _hpi_yoy,
                "price_normalised":          r.get("price_normalised"),
                "normalisation_factor":      r.get("normalisation_factor"),
                "subject_floor_area_m2":     _subject_area,
                "comp_floor_area_m2":        r.get("floor_area"),
                "price_used_for_median":     (
                    "normalised" if (_use_normalised and r.get("normalisation_factor") is not None)
                    else "hpi_adjusted" if (_hpi_yoy is not None and r.get("age_months", 0) > 3)
                    else "nominal"
                ),
                "similarity_score":          r.get("_similarity_score"),
                "score_components":          r.get("_score_components"),
            }

        _methodology_label = (
            "similarity_weighted_normalised"
            if _use_normalised and _hpi_yoy is not None
            else "similarity_weighted_hpi_adjusted"
            if _hpi_yoy is not None
            else "similarity_weighted_area_normalised"
            if _use_normalised
            else "similarity_weighted_nominal"
        )

        _summary_parts = [f"{len(rows)} comps within {r_miles}mi"]
        if _use_normalised:
            _summary_parts.append(f"area-normalised ({_norm_coverage_pct:.0f}% coverage)")
        if _hpi_yoy is not None:
            _summary_parts.append(f"HPI-adjusted ({_hpi_yoy:+.1f}% YoY)")
        if _audit["outliers_rejected"]:
            _summary_parts.append(f"{_audit['outliers_rejected']} outliers rejected")
        if _audit["methodology_degraded"]:
            _summary_parts.append("⚠ methodology degraded — see warnings")
        if _evidence_insufficient:
            _summary_parts.append("⚠ INSUFFICIENT COMPARABLE EVIDENCE")

        summary = ". ".join(_summary_parts) + f". Median: {'£{:,}'.format(med) if med else 'n/a'}. Types: {pt_str}."

        # latlng enrichment metadata: no lat/lng enrichment step exists in this
        # function, so this is an inert placeholder. Defined here to prevent a
        # NameError at the out["metrics"] assignment below (was previously
        # referenced as `enrich_meta` without ever being defined, which caused
        # get_housing_data to discard valid comps via the except handler).
        enrich_meta = None

        out = metric_ok(summary, rows, sources, retrieved, HOUSING_CONFIDENCE_VALUE)
        out["metrics"] = {
            "provider":                    "supabase_rpc",
            "rpc":                         HOUSING_RPC_NAME,
            "postcode":                    pc,
            "radius_miles":                r_miles,
            "limit":                       lim,
            "count":                       len(rows),
            "median_price":                med,
            "avg":                         avg,   # T-3: arithmetic mean (was: median, mislabelled)
            "average_price":               avg,   # T-3: arithmetic mean (was: median, mislabelled)
            "area_normalisation_applied":  _use_normalised,
            "area_normalisation_coverage": _area_normalised_count,
            "hpi_adjusted_count":          _hpi_adjusted_count,
            "hpi_yoy_applied":             _hpi_yoy,
            "outliers_rejected":           _audit.get("outliers_rejected", 0),
            "methodology_degraded":        _audit.get("methodology_degraded", False),
            "insufficient_evidence":       _evidence_insufficient,
            "methodology":                 _methodology_label,
            "warnings":                    _audit.get("warnings", []),
            "audit":                       _audit,
            "price_variance_pct":          round(_price_variance_pct, 1),
            "min_miles":                   min_m,
            "max_miles":                   max_m,
            "property_type_counts":        ptypes,
            "latlngEnrichment":            enrich_meta,
            "payload":                     payload,
        }

        charts = build_housing_charts_from_rows(rows)

        try:
            t_total = sum(
                int(b.get("value") or 0)
                for b in (charts.get("propertyTypes") or {}).get("bins", [])
                if isinstance(b, dict)
            )
            if t_total != len(rows):
                print("⚠️ propertyTypes bins total mismatch:", {"binsTotal": t_total, "rows": len(rows), "sampleType": (rows[0] or {}).get("property_type")})
        except Exception:
            pass

        out["metrics"]["pricingPower"] = _pricing_power_from_rows(rows)
        out["metrics"]["pricingPowerSoldCompsMomentum"] = build_pricing_power_sold_comps_momentum(rows, r_miles)

        out["soldComps"] = rows
        out["charts"] = charts
        return out

    except Exception as e:
        msg = str(e) or "Unknown error"
        out = metric_missing_provider(
            f"Housing RPC not available. Create Supabase function '{HOUSING_RPC_NAME}' then retry. Error: {msg}",
            sources,
            retrieved,
            extra_metrics={"postcode": pc, "radius_miles": r_miles, "limit": lim, "rpc": HOUSING_RPC_NAME},
        )
        out["metrics"]["housingRpcError"] = msg
        out["metrics"]["payload"] = payload
        return out


@app.route("/adapters/geo", methods=["GET"])
def adapter_geo():
    postcode = normalize_postcode(request.args.get("postcode", "") or "")
    lsoa_gss, meta = resolve_lsoa_gss_from_postcode(postcode)
    return jsonify({
        "status": "ok" if lsoa_gss else "unavailable",
        "postcode": postcode,
        "lsoa_gss": lsoa_gss,
        "meta": meta,
    })


@app.route("/adapters/schools", methods=["GET"])
def adapter_schools():
    postcode = normalize_postcode(request.args.get("postcode", "") or "")
    return jsonify(get_schools_data(postcode))


@app.route("/adapters/broadband", methods=["GET"])
def adapter_broadband():
    postcode = normalize_postcode(request.args.get("postcode", "") or "")
    return jsonify(get_broadband_data(postcode))


@app.route("/adapters/housing/comps", methods=["GET"])
def adapter_housing_comps():
    postcode = normalize_postcode(request.args.get("postcode", "") or "")
    radius_miles = safe_float(request.args.get("radius_miles"))
    limit = safe_int(request.args.get("limit"))
    return jsonify(get_housing_data(postcode, radius_miles=radius_miles, limit=limit))


@app.route("/adapters/nomis", methods=["GET"])
def adapter_nomis():
    table_raw = (request.args.get("table", "") or "").strip()
    postcode = normalize_postcode(request.args.get("postcode", "") or "")
    geography = (request.args.get("geography", "") or "").strip()

    if ("postcode=" in table_raw) and (not postcode):
        parts = table_raw.split("postcode=", 1)
        table_raw = parts[0]
        postcode = normalize_postcode(parts[1]) if len(parts) > 1 else ""

    table = table_raw.lower().strip()

    if not geography:
        geography = NOMIS_DEFAULT_GEOGRAPHY

    if table == "ts003":
        return jsonify(get_nomis_table("Household composition (TS003)", NOMIS_TS003_DIM, NOMIS_TS003_CATS, geography))
    if table == "ts044":
        return jsonify(get_nomis_table("Accommodation type (TS044)", NOMIS_TS044_DIM, NOMIS_TS044_CATS, geography))
    if table == "ts054":
        return jsonify(get_nomis_table("Tenure (TS054)", NOMIS_TS054_DIM, NOMIS_TS054_CATS, geography))

    return jsonify(metric_unavailable(
        "Unknown Nomis table. Use table=ts003|ts044|ts054",
        [{"label": "Nomis API", "url": "https://www.nomisweb.co.uk/api/v01/help"}],
        now_iso()
    ))


@app.route("/market-insights", methods=["POST"])
@app.route("/market_insights", methods=["POST"])  # alias for any legacy/underscore callers
def market_insights():
    data = request.get_json(silent=True) or {}
    postcode = normalize_postcode(data.get("postcode", "") or "")
    lat = safe_float(data.get("lat"))
    lng = safe_float(data.get("lng"))

    # Optional UK HPI inputs (used to build `trends` time series).
    # Frontend may send any of these; accept the common variants.
    area_code = (data.get("area_code") or data.get("areaCode") or data.get("hpiAreaCode") or "")
    area_code = str(area_code).strip()
    # If frontend didn’t pass an area_code, we will try to derive it after postcode lookup.

    months = safe_int(data.get("months"))
    months = int(months) if isinstance(months, int) and months > 0 else 24
    property_type = (data.get("property_type") or data.get("propertyType") or "")
    property_type = str(property_type).strip() or None

    force_refresh = bool(data.get("forceRefresh") is True)

    def _finalize(payload: Dict[str, Any]) -> Any:
        """Hard-contract finalization for this endpoint."""
        payload = ensure_market_trends(payload)
        return jsonify(payload)

    if MARKET_CONTRACT_MODE:
        nomis_geo = (NOMIS_DEFAULT_GEOGRAPHY or "stub").strip()
        results = build_market_contract_stub(postcode, lat, lng, nomis_geo)
        # Contract mode must still satisfy the hard Trends contract.
        results["trends"] = build_trends_from_uk_hpi(postcode, area_code, months, property_type=property_type)
        payload = {
            **results,
            "_contract": {"mode": True},
            "_cache": {"hit": False, "ttlSeconds": CACHE_TTL_SECONDS},
        }
        return _finalize(payload)

    lsoa_gss = ""
    lsoa_meta = None
    if postcode:
        lsoa_gss, lsoa_meta = resolve_lsoa_gss_from_postcode(postcode)

    # If frontend didn’t pass an area_code, derive it from postcode lookup (postcodes.io codes.admin_district).
    if not area_code and isinstance(lsoa_meta, dict):
        area_code = str(lsoa_meta.get("area_code") or "").strip()

    if (lat is None or lng is None) and isinstance(lsoa_meta, dict):
        if lat is None:
            lat = safe_float(lsoa_meta.get("lat"))
        if lng is None:
            lng = safe_float(lsoa_meta.get("lng"))

    nomis_geo = (NOMIS_DEFAULT_GEOGRAPHY or "").strip()

    cache_key = (
        f"market-insights::{postcode}::{lat or ''}::{lng or ''}::{nomis_geo or ''}::rpc={HOUSING_RPC_NAME}::bust={APP_CACHE_BUSTER}"
        if postcode else
        f"market-insights::no-postcode::rpc={HOUSING_RPC_NAME}::bust={APP_CACHE_BUSTER}"
    )

    if not force_refresh:
        cached = cache_get(cache_key)
        if cached:
            try:
                h = (cached.get("localAreaAnalysis") or {}).get("housing") or {}
                hv = h.get("value") or []
                if h.get("status") == "ok" and isinstance(hv, list) and len(hv) > 0:
                    payload = {
                        **cached,
                        "_cache": {"hit": True, "ttlSeconds": CACHE_TTL_SECONDS},
                    }
                    return _finalize(payload)
            except Exception:
                pass

    try:
        geo_meta = None

        if (lat is None or lng is None) and postcode:
            lat2, lng2, nspl_meta = nspl_lookup_latlng(postcode)
            if lat2 is not None and lng2 is not None:
                lat, lng, geo_meta = lat2, lng2, nspl_meta
            else:
                lat, lng, geo_meta = geocode_postcode(postcode)

        local_area = {
            "retrievedAtISO": now_iso(),
            "postcode": postcode,
            "schools": get_schools_data(postcode),
            "housing": get_housing_data(postcode),
            "transport": get_transport_data(lat, lng),
            "amenities": get_amenities_data(lat, lng),
            "crime": get_crime_data(lat, lng),
            "broadband": get_broadband_data(postcode),
            "gp": get_gp_data(postcode),
            "flood": get_flood_risk(lat, lng, postcode),
            "census": {
                "ts003": get_nomis_table("Household composition (TS003)", NOMIS_TS003_DIM, NOMIS_TS003_CATS, nomis_geo),
                "ts044": get_nomis_table("Accommodation type (TS044)", NOMIS_TS044_DIM, NOMIS_TS044_CATS, nomis_geo),
                "ts054": get_nomis_table("Tenure (TS054)", NOMIS_TS054_DIM, NOMIS_TS054_CATS, nomis_geo),
                "private_rent_pct": _get_census_private_rent_pct(lsoa_gss),
            },
        }

        results = {
            "postcode": postcode,
            "location": {
                "lat": lat,
                "lng": lng,
                "geocodeMeta": geo_meta,
                "lsoaMeta": lsoa_meta,
                "nomisGeography": nomis_geo,
            },
            "localAreaAnalysis": local_area,
            "comparableProperties": {
                "forSale": [],
                "sourceUrl": "",
                "sources": [],
                "retrievedAtISO": now_iso(),
                "confidenceValue": 0.0,
                "status": "missing_provider",
                "summary": "Comparable properties provider not configured in this build.",
            },
        }

        # UK HPI trends (area-level series). Falls back to guaranteed trends if unavailable.
        results["trends"] = build_trends_from_uk_hpi(postcode, area_code, months, property_type=property_type)

        # Ensure hard-contract fields are present BEFORE caching.
        results = ensure_market_trends(results)

        if not force_refresh:
            try:
                h = (results.get("localAreaAnalysis") or {}).get("housing") or {}
                hv = h.get("value") or []
                if h.get("status") == "ok" and isinstance(hv, list) and len(hv) > 0:
                    cache_set(cache_key, results)
            except Exception:
                pass

        payload = {
            **results,
            "_cache": {"hit": False, "ttlSeconds": CACHE_TTL_SECONDS, "forceRefresh": force_refresh},
        }
        return _finalize(payload)

    except Exception as e:
        print("❌ Error in /market-insights:", str(e))
        payload = {
            "error": str(e),
            "postcode": postcode,
            "_cache": {"hit": False, "ttlSeconds": CACHE_TTL_SECONDS, "forceRefresh": force_refresh},
        }
        return _finalize(payload), 500






@app.route("/qa/clarify", methods=["POST"])
def qa_clarify():
    """Bounded solicitor-style clarification for a single triage flag.

    Expected JSON body:
      - flag_id: str (required)
      - question: str (optional)

    Returns a stable, UI-friendly object. If the engine is not available,
    returns 501 so the UI can degrade gracefully.
    """
    if clarify_flag is None:
        return (
            jsonify({
                "error": "not_implemented",
                "message": "Solicitor Q&A engine is not available in this deployment.",
            }),
            501,
        )

    payload = request.get_json(silent=True) or {}
    flag_id = payload.get("flag_id")
    question = payload.get("question")

    if not flag_id or not isinstance(flag_id, str):
        return (
            jsonify(
                {
                    "error": "invalid_request",
                    "message": "flag_id (string) is required.",
                }
            ),
            400,
        )

    try:
        result = clarify_flag(flag_id.strip(), question if isinstance(question, str) else None)
        return jsonify(result)
    except Exception as e:
        app.logger.exception("qa_clarify failed")
        return (
            jsonify(
                {
                    "error": "qa_clarify_failed",
                    "message": "Unable to generate clarification at this time.",
                    "details": str(e),
                }
            ),
            500,
        )


@app.route("/llm/json", methods=["POST"])
def llm_json_route():
    """
    Dual-mode endpoint:
      - application/json  (legacy clients)
      - multipart/form-data (frontend FormData uploads)

    Required field: prompt (non-empty). Accepts prompt from:
      - JSON body: { prompt } or { options: { prompt } }
      - multipart: form field "prompt" or JSON in form field "options" containing { prompt }

    Contract (always returned, even on failure):
      { score:number, summary:string, positives:string[], risks:string[] }
    """
    system = ""
    prompt = None
    options = {}

    # --- JSON mode (legacy) ---
    if request.is_json:
        payload = request.get_json(force=True, silent=True) or {}
        system = payload.get("system", "") or ""
        prompt = payload.get("prompt")
        maybe_opts = payload.get("options")
        if isinstance(maybe_opts, dict):
            options = maybe_opts
            if not prompt:
                prompt = maybe_opts.get("prompt")

    # --- multipart / form mode (frontend FormData) ---
    else:
        system = (request.form.get("system") or "").strip()
        prompt = request.form.get("prompt")

        raw_options = request.form.get("options")
        if raw_options:
            try:
                parsed = json.loads(raw_options)
                if isinstance(parsed, dict):
                    options = parsed
                else:
                    options = {"_raw_options": parsed}
            except Exception:
                options = {"_raw_options": raw_options}

        if not prompt and isinstance(options, dict):
            prompt = options.get("prompt")

    prompt_str = (str(prompt).strip() if prompt is not None else "")
    if not prompt_str:
        return jsonify({"error": "prompt is required"}), 400

    # Deterministic health check (no LLM call)
    if prompt_str.lower() in ("ping", "health", "healthcheck"):
        return jsonify({"score": 0, "summary": "pong", "positives": [], "risks": []}), 200

    try:
        result = llm_json(system=str(system), prompt=prompt_str)

        # Hard guard: if upstream returns unexpected shape, normalise here (never 500 the client)
        if not isinstance(result, dict):
            raise ValueError("LLM returned non-object")

        score = result.get("score", 0)
        summary = result.get("summary", "")
        positives = result.get("positives", [])
        risks = result.get("risks", [])

        if not isinstance(score, (int, float)):
            score = 0
        if not isinstance(summary, str):
            summary = ""
        if not isinstance(positives, list):
            positives = []
        if not isinstance(risks, list):
            risks = []

        return jsonify({"score": score, "summary": summary, "positives": positives, "risks": risks}), 200

    except Exception as e:
        app.logger.exception("llm_json failed")

        # Never strand the UI with a 500. Return contract-compliant payload + error context.
        return (
            jsonify(
                {
                    "score": 0,
                    "summary": "LLM unavailable. Returning safe fallback.",
                    "positives": [],
                    "risks": [],
                    "_error": "llm_failed",
                    "_details": str(e),
                }
            ),
            200,
        )



# ============================================================
# PDF UPLOAD PIPELINE
# ============================================================
import jwt as pyjwt
import anthropic as _anthropic
#   import io
#   try:
#       import pdfplumber
#   except ImportError:
#       pdfplumber = None
#
# ============================================================

# ── JWT VALIDATION ──────────────────────────────────────────
SUPABASE_JWT_SECRET = (os.getenv("SUPABASE_JWT_SECRET") or "").strip()

def get_user_id_from_request() -> Optional[str]:
    """Extract and validate Supabase JWT from Authorization header.
    Returns user_id (UUID string) or None if invalid/missing."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    token = auth_header[7:]
    if not SUPABASE_JWT_SECRET or not token:
        return None
    try:
        payload = pyjwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated"
        )
        return payload.get("sub")
    except Exception:
        return None


def require_auth(f):
    """Decorator — returns 401 if no valid JWT.
    OPTIONS preflight requests are always passed through so flask-cors
    can attach the correct Access-Control-* headers without an auth gate."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        # CORS preflight — never carries an Authorization header by spec.
        # Let flask-cors handle it; do NOT auth-gate it.
        if request.method == "OPTIONS":
            return f(*args, **kwargs)
        user_id = get_user_id_from_request()
        if not user_id:
            return jsonify({"error": "Unauthorised — valid JWT required"}), 401
        request.user_id = user_id
        return f(*args, **kwargs)
    return decorated


# ── AI EXPLAIN — Flag workbench Ask AI proxy ────────────────
_anthropic_client = None

def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set on Render")
        _anthropic_client = _anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def _llm_json_anthropic(*, system: str, prompt: str, temperature: float = 0.1) -> dict:
    """Run a JSON-returning LLM call via Anthropic claude-sonnet.

    Drop-in replacement for llm_json_raw() that uses the Anthropic client
    already configured via ANTHROPIC_API_KEY — avoiding the OpenRouter dependency
    for the legal analysis pipeline.

    Returns a parsed dict. Raises ValueError if the model returns non-JSON.
    Raises RuntimeError if ANTHROPIC_API_KEY is not set.
    """
    import re as _re
    import json as _json

    client = _get_anthropic_client()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=20000,
        temperature=float(temperature),
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    content     = message.content[0].text if message.content else ""
    stop_reason = getattr(message, "stop_reason", "unknown")
    usage_in    = getattr(getattr(message, "usage", None), "input_tokens",  0)
    usage_out   = getattr(getattr(message, "usage", None), "output_tokens", 0)

    # ── ALWAYS print to stdout — visible in Render logs regardless of log level ──
    print(f"[LLM] stop_reason={stop_reason} tokens_in={usage_in} tokens_out={usage_out} content_len={len(content)}", flush=True)

    if stop_reason == "max_tokens":
        print(f"[LLM] WARNING: TRUNCATED at max_tokens=16000. Last 300 chars: {content[-300:]!r}", flush=True)

    # Print the FULL raw LLM response before any processing.
    # This is the ground truth — if flags are missing, this tells us why.
    print(f"[LLM] RAW RESPONSE START >>>", flush=True)
    print(content[:3000], flush=True)  # first 3000 chars
    if len(content) > 3000:
        print(f"[LLM] ... ({len(content) - 3000} more chars) ...", flush=True)
        print(content[-500:], flush=True)  # last 500 chars
    print(f"[LLM] RAW RESPONSE END <<<", flush=True)

    # Try direct parse first
    try:
        result = _json.loads(content.strip())
        print(f"[LLM] Parsed OK (direct). flags={len(result.get('flags') or [])}", flush=True)
        return result
    except Exception:
        pass

    # Strip markdown fences
    cleaned = _re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(),
                      flags=_re.IGNORECASE | _re.MULTILINE).strip()
    try:
        result = _json.loads(cleaned)
        print(f"[LLM] Parsed OK (stripped fences). flags={len(result.get('flags') or [])}", flush=True)
        return result
    except Exception:
        pass

    # Bracket-counter extraction — find first { to its matching closing }
    # More robust than greedy r"(\{.*\})" which breaks on trailing prose
    # after a valid JSON block or on multiple top-level objects.
    def _extract_json_object(text):
        start = text.find('{')
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None  # unmatched brackets — truncated response

    candidate = _extract_json_object(cleaned)
    if candidate:
        try:
            result = _json.loads(candidate)
            print(f"[LLM] Parsed OK (bracket-counter). flags={len(result.get('flags') or [])}", flush=True)
            return result
        except Exception as parse_err:
            print(f"[LLM] Bracket-counter extracted {len(candidate)} chars but json.loads failed: {parse_err}", flush=True)

    # All parse attempts failed
    print(f"[LLM] ALL PARSE ATTEMPTS FAILED. stop_reason={stop_reason} tokens_out={usage_out}", flush=True)
    raise ValueError(
        f"Anthropic model returned non-JSON. "
        f"stop_reason={stop_reason} tokens_out={usage_out} "
        f"first_300={content[:300]!r}"
    )


@app.route("/api/ai-explain", methods=["POST"])
@require_auth
def ai_explain():
    """Flag workbench Ask AI proxy. POST {prompt} -> {text}. Bearer token required."""
    body = request.get_json(silent=True) or {}
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400
    if len(prompt) > 4000:
        return jsonify({"error": "prompt too long (max 4000 chars)"}), 400
    try:
        client = _get_anthropic_client()
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=(
                "You are a concise UK property auction legal analyst. "
                "You explain legal pack issues to investors in plain English. "
                "Give specific cost estimates in \u00a3 where relevant. "
                "Never give legal advice. "
                "Keep responses to 120 words maximum."
            ),
            messages=[{"role": "user", "content": prompt}]
        )
        text = message.content[0].text if message.content else ""
        return jsonify({"text": text})
    except _anthropic.AuthenticationError:
        app.logger.error("Anthropic auth failed — check ANTHROPIC_API_KEY on Render")
        return jsonify({"error": "AI service authentication failed — check ANTHROPIC_API_KEY on Render"}), 500
    except _anthropic.RateLimitError:
        return jsonify({"error": "AI rate limit reached — please try again"}), 429
    except _anthropic.APIError as e:
        app.logger.exception("Anthropic API error in ai_explain")
        return jsonify({"error": f"AI service error: {str(e)}"}), 500
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        app.logger.exception("Unexpected error in ai_explain")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


# ── PDF TEXT EXTRACTION ─────────────────────────────────────
DOCUMENT_PATTERNS: Dict[str, List[str]] = {
    "legal_pack":          ["legal pack", "auction pack", "lot information", "information pack",
                            "document archive", "pack archive"],
    "special_conditions":  ["special conditions", "special condition of sale", "conditions of sale"],
    "addendum":            ["addendum", "day of sale", "lot amendment", "amendment notice",
                            "late amendment", "revised conditions", "updated conditions",
                            "pre auction notice", "vendor notice"],
    "title_register":      ["title register", "hm land registry", "official copy of register",
                            "land registry", "register of title", "property register",
                            "proprietorship register", "charges register", "official copy"],
    "title_plan":          ["title plan", "filed plan", "ordnance survey map",
                            "administrative area", "title number"],
    "local_auth_search":   ["local authority search", "con29", "llc1", "local land charges",
                            "city council", "district council", "borough council",
                            "regulated local authority", "enquiries of the local authority"],
    "environmental":       ["groundsure", "homebuyer environmental", "environmental search",
                            "flood risk", "ground risk", "chancel",
                            "drainage search", "severn trent", "thames water",
                            "anglian water", "water search", "regulated drainage",
                            "combined drainage", "utilities search"],
    "lease":               ["lease", "underlease", "sublease", "leasehold land",
                            "lease dated", "term of years"],
    "epc":                 ["energy performance", "epc", "energy certificate",
                            "domestic energy", "energy rating"],
    "survey":              ["structural survey", "building survey", "rics survey",
                            "condition report", "level 2", "level 3", "homebuyer report"],
    "auction_tcs":         ["auction terms", "auctioneer terms", "conditions of auction"],
    "freehold":            ["freehold", "absolute freehold", "possessory freehold"],
    "deed":                ["transfer deed", "conveyance", "tr1", "deed of",
                            "ta6", "ta10", "seller property", "fittings and contents",
                            "property information form"],
    "tenancy_ast":         ["assured shorthold", "tenancy agreement", "rental agreement"],
}

def detect_document_type(filename: str, text: str) -> str:
    combined = (filename + " " + (text or "")[:3000]).lower()
    combined = re.sub(r"[_\-.]", " ", combined)
    for doc_type, patterns in DOCUMENT_PATTERNS.items():
        if any(p in combined for p in patterns):
            return doc_type
    return "unknown"


def extract_pdf_text(file_bytes: bytes) -> Tuple[str, int]:
    """Extract text from PDF bytes. Returns (text, page_count).
    
    Strategy:
    1. pymupdf (fitz) — best on complex layouts, tables, multi-column
    2. pdfplumber — fallback, good on standard text PDFs
    3. Empty string — if both fail (scanned/image PDFs)
    
    Checks text density per page — if very low, likely scanned.
    """
    # ── Attempt 1: pymupdf ──
    try:
        import fitz  # pymupdf
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page_count = len(doc)
        text_parts = []
        for page in doc:
            text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            if text and text.strip():
                text_parts.append(text)
        doc.close()
        combined = "\n\n".join(text_parts)
        # Check text density — if meaningful text extracted, use it
        if combined.strip() and len(combined) > page_count * 50:
            app.logger.info(f"pymupdf extracted {len(combined):,} chars from {page_count} pages")
            return combined, page_count
        app.logger.info(f"pymupdf low yield ({len(combined)} chars) — trying pdfplumber")
    except ImportError:
        app.logger.info("pymupdf not available — trying pdfplumber")
    except Exception as e:
        app.logger.warning(f"pymupdf failed: {e} — trying pdfplumber")

    # ── Attempt 2: pdfplumber ──
    if pdfplumber is not None:
        try:
            text_parts = []
            page_count = 0
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    # Try standard extraction first
                    page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                    if not page_text:
                        # Try with looser tolerances for complex layouts
                        page_text = page.extract_text(x_tolerance=8, y_tolerance=8)
                    if page_text and page_text.strip():
                        text_parts.append(page_text)
            combined = "\n\n".join(text_parts)
            if combined.strip():
                app.logger.info(f"pdfplumber extracted {len(combined):,} chars from {page_count} pages")
                return combined, page_count
        except Exception as e:
            app.logger.warning(f"pdfplumber failed: {e}")

    # ── Both failed — likely scanned PDF ──
    app.logger.warning("Both extraction methods failed — PDF may be scanned/image-based")
    return "", page_count if 'page_count' in dir() else 0



# ── DEALS ───────────────────────────────────────────────────
@app.route("/api/deals", methods=["POST"])
@require_auth
def create_deal():
    """Create a new deal record. Called before upload begins.
    Body: { deal_name, postcode?, lot_number?, guide_price?, deal_type?, auction_date? }
    Returns: { deal_id }"""
    if not supabase:
        return jsonify({"error": "Database unavailable"}), 503

    data = request.get_json(silent=True) or {}
    # deal_name is optional — address is extracted from documents during analysis
    from datetime import datetime as _dt
    deal_name = (data.get("deal_name") or "").strip() or f"Deal — {_dt.now().strftime('%d %b %Y')}" 

    try:
        result = supabase.table("deals").insert({
            "user_id":      request.user_id,
            "deal_name":    deal_name,
            "title":        deal_name,
            "postcode":     (data.get("postcode") or "").strip().upper() or None,
            "lot_number":   (data.get("lot_number") or "").strip() or None,
            "guide_price":  data.get("guide_price"),
            "deal_type":    (data.get("deal_type") or "").strip() or None,
            "auction_date": data.get("auction_date") or None,
            "status":       "active",
        }).execute()

        deal_id = result.data[0]["id"]
        return jsonify({"ok": True, "deal_id": deal_id}), 201

    except Exception as e:
        app.logger.exception("create_deal failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/profile", methods=["GET"])
@require_auth
def get_profile():
    """Return the current user plan and usage for tier gating."""
    if not supabase:
        return jsonify({"plan": "free", "summaries_used": 0, "analyses_used": 0}), 200
    try:
        result = supabase.table("profiles") \
            .select("plan, summaries_used, analyses_used, usage_reset_date") \
            .eq("id", request.user_id) \
            .single() \
            .execute()
        profile = result.data or {}
        return jsonify({
            "plan":             profile.get("plan", "free"),
            "summaries_used":   profile.get("summaries_used", 0),
            "analyses_used":    profile.get("analyses_used", 0),
            "usage_reset_date": profile.get("usage_reset_date"),
        }), 200
    except Exception as e:
        app.logger.warning(f"get_profile failed: {e}")
        return jsonify({"plan": "free", "summaries_used": 0, "analyses_used": 0}), 200


@app.route("/api/deals", methods=["GET"])
@require_auth
def list_deals():
    """List all deals for the authenticated user, newest first."""
    if not supabase:
        return jsonify({"error": "Database unavailable"}), 503
    try:
        result = supabase.table("deals") \
            .select("*") \
            .eq("user_id", request.user_id) \
            .neq("status", "archived") \
            .order("created_at", desc=True) \
            .execute()
        return jsonify({"ok": True, "deals": result.data}), 200
    except Exception as e:
        app.logger.exception("list_deals failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/deals/<deal_id>", methods=["GET"])
@require_auth
def get_deal(deal_id: str):
    """Get a single deal by ID — user must own it."""
    if not supabase:
        return jsonify({"error": "Database unavailable"}), 503
    try:
        result = supabase.table("deals") \
            .select("*, documents(*)") \
            .eq("id", deal_id) \
            .eq("user_id", request.user_id) \
            .single() \
            .execute()
        if not result.data:
            return jsonify({"error": "Deal not found"}), 404
        deal = result.data
        # area_json is fetched via POST /api/deals/:id/area (verdict and area pages both trigger this)
        # Do NOT launch a second background thread here — it races with the POST /area fetch
        # and produces inconsistent area_json (missing property_type, guide_price, gp data).
        return jsonify({"ok": True, "deal": deal}), 200
    except Exception as e:
        app.logger.exception("get_deal failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/deals/<deal_id>", methods=["PATCH"])
@require_auth
def update_deal(deal_id: str):
    """Update deal fields. Body: any subset of deal columns."""
    if not supabase:
        return jsonify({"error": "Database unavailable"}), 503
    data = request.get_json(silent=True) or {}
    # Whitelist updatable fields — never allow user_id to be changed
    allowed = {
        "deal_name", "title", "postcode", "lot_number", "guide_price",
        "deal_type", "auction_date", "status", "bid_ceiling", "hammer_price", "outcome",
        "completion_period", "completion_deadline", "completion_actions",
        "summary_json", "analysis_json", "area_json", "financials_json",
        "deal_score", "address",
    }
    updates = {k: v for k, v in data.items() if k in allowed}
    if not updates:
        return jsonify({"error": "No valid fields to update"}), 400
    updates["updated_at"] = now_iso()
    try:
        result = supabase.table("deals") \
            .update(updates) \
            .eq("id", deal_id) \
            .eq("user_id", request.user_id) \
            .execute()
        return jsonify({"ok": True, "deal": result.data[0] if result.data else {}}), 200
    except Exception as e:
        app.logger.exception("update_deal failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/deals/<deal_id>", methods=["DELETE"])
@require_auth
def delete_deal(deal_id: str):
    """Archive (soft-delete) a deal. Sets status=archived so it disappears from dashboard
    but data is retained. Pass ?hard=1 to permanently delete."""
    if not supabase:
        return jsonify({"error": "Database unavailable"}), 503

    hard_delete = request.args.get("hard", "").lower() in ("1", "true")

    # Verify ownership first
    try:
        check = supabase.table("deals") \
            .select("id") \
            .eq("id", deal_id) \
            .eq("user_id", request.user_id) \
            .single() \
            .execute()
        if not check.data:
            return jsonify({"error": "Deal not found"}), 404
    except Exception:
        return jsonify({"error": "Deal not found"}), 404

    try:
        if hard_delete:
            # Permanently delete — cascades to documents via FK
            supabase.table("deals") \
                .delete() \
                .eq("id", deal_id) \
                .eq("user_id", request.user_id) \
                .execute()
            return jsonify({"ok": True, "deleted": True, "deal_id": deal_id}), 200
        else:
            # Soft delete — archive
            supabase.table("deals") \
                .update({"status": "archived", "updated_at": now_iso()}) \
                .eq("id", deal_id) \
                .eq("user_id", request.user_id) \
                .execute()
            return jsonify({"ok": True, "archived": True, "deal_id": deal_id}), 200
    except Exception as e:
        app.logger.exception("delete_deal failed")
        return jsonify({"error": str(e)}), 500


# ── DOCUMENT UPLOAD ─────────────────────────────────────────
# Hard cap: 20MB. Render free plan has 512MB RAM; pymupdf can 3-5× a PDF in
# memory during extraction — a 50MB PDF could exhaust the worker and cause 502.
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB — Flask rejects larger before any read

@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({"error": "File exceeds 20MB limit. Split your legal pack into smaller documents."}), 413

@app.route("/api/documents/upload", methods=["OPTIONS"])
def upload_options():
    """Explicit OPTIONS handler so CORS preflight always gets a 200, not a 502."""
    return "", 200

@app.route("/api/documents/upload", methods=["POST"])
@require_auth
def upload_document():
    """Upload a PDF legal pack document.
    Multipart form: file (PDF), deal_id (required).
    Returns: { document_id, doc_type, page_count, extraction_status }"""
    if not supabase:
        return jsonify({"error": "Database unavailable"}), 503

    deal_id = (request.form.get("deal_id") or "").strip()
    if not deal_id:
        return jsonify({"error": "deal_id is required"}), 400

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are accepted"}), 400

    # Verify deal belongs to this user
    try:
        deal_check = supabase.table("deals") \
            .select("id") \
            .eq("id", deal_id) \
            .eq("user_id", request.user_id) \
            .single() \
            .execute()
        if not deal_check.data:
            return jsonify({"error": "Deal not found or access denied"}), 403
    except Exception:
        return jsonify({"error": "Deal verification failed"}), 403

    # Read file bytes — Flask has already enforced MAX_CONTENT_LENGTH above
    try:
        file_bytes = file.read()
        file_size = len(file_bytes)
    except Exception as e:
        return jsonify({"error": f"File read failed: {e}"}), 400

    # Belt-and-braces size check (in case MAX_CONTENT_LENGTH was bypassed)
    MAX_SIZE = 20 * 1024 * 1024
    if file_size > MAX_SIZE:
        return jsonify({"error": "File exceeds 20MB limit. Split your legal pack into smaller documents."}), 413

    # Extract text — catch any OOM / crash gracefully
    try:
        extracted_text, page_count = extract_pdf_text(file_bytes)
    except Exception as e:
        app.logger.warning(f"PDF extraction failed: {e} — storing without text")
        extracted_text, page_count = "", 0
    extraction_status = "complete" if extracted_text else "empty"

    # Detect document type
    doc_type = detect_document_type(filename, extracted_text)

    # Store in Supabase Storage — non-fatal if this fails
    storage_path = f"{request.user_id}/{deal_id}/{filename}"
    try:
        supabase.storage.from_("legal-packs").upload(
            path=storage_path,
            file=file_bytes,
            file_options={"content-type": "application/pdf", "upsert": "true"}
        )
    except Exception as e:
        app.logger.warning(f"Storage upload failed: {e} — continuing without storage")
        storage_path = f"upload_failed/{filename}"

    # Free memory before the DB write
    del file_bytes

    # Create document record in database
    try:
        doc_result = supabase.table("documents").insert({
            "deal_id":           deal_id,
            "user_id":           request.user_id,
            "doc_type":          doc_type,
            "file_name":         filename,
            "storage_path":      storage_path,
            "file_size_bytes":   file_size,
            "page_count":        page_count,
            "extracted_text":    extracted_text[:500000] if extracted_text else None,
            "extraction_status": extraction_status,
        }).execute()

        document_id = doc_result.data[0]["id"]

        return jsonify({
            "ok":                True,
            "document_id":       document_id,
            "doc_type":          doc_type,
            "page_count":        page_count,
            "file_size_bytes":   file_size,
            "extraction_status": extraction_status,
            "has_text":          bool(extracted_text),
        }), 201

    except Exception as e:
        app.logger.exception("document insert failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/documents/<deal_id>", methods=["GET"])
@require_auth
def list_documents(deal_id: str):
    """List all documents for a deal."""
    if not supabase:
        return jsonify({"error": "Database unavailable"}), 503
    try:
        result = supabase.table("documents") \
            .select("id, doc_type, file_name, page_count, file_size_bytes, extraction_status, created_at") \
            .eq("deal_id", deal_id) \
            .eq("user_id", request.user_id) \
            .order("created_at") \
            .execute()
        return jsonify({"ok": True, "documents": result.data}), 200
    except Exception as e:
        app.logger.exception("list_documents failed")
        return jsonify({"error": str(e)}), 500


# ── USAGE ───────────────────────────────────────────────────
@app.route("/api/usage", methods=["GET"])
@require_auth
def get_usage():
    """Get current month usage for authenticated user."""
    if not supabase:
        return jsonify({"error": "Database unavailable"}), 503
    try:
        profile = supabase.table("profiles") \
            .select("plan, summaries_used, analyses_used, usage_reset_date") \
            .eq("id", request.user_id) \
            .single() \
            .execute()

        if not profile.data:
            return jsonify({"error": "Profile not found"}), 404

        p = profile.data
        plan = p.get("plan", "starter")

        PLAN_LIMITS = {
            "free":         {"summaries": 0,  "analyses": 0},
            "report":       {"summaries": 1,  "analyses": 0},
            "starter":      {"summaries": 3,  "analyses": 999},
            "professional": {"summaries": 10, "analyses": 999},
            "portfolio":    {"summaries": 999, "analyses": 999},
            "enterprise":   {"summaries": 999, "analyses": 999},
        }
        limits = PLAN_LIMITS.get(plan, PLAN_LIMITS["starter"])

        return jsonify({
            "ok": True,
            "plan":              plan,
            "summaries_used":    p.get("summaries_used", 0),
            "summaries_limit":   limits["summaries"],
            "analyses_used":     p.get("analyses_used", 0),
            "analyses_limit":    limits["analyses"],
            "reset_date":        p.get("usage_reset_date"),
        }), 200

    except Exception as e:
        app.logger.exception("get_usage failed")
        return jsonify({"error": str(e)}), 500


# ── DOCUMENT SUMMARY ────────────────────────────────────────
@app.route("/api/deals/<deal_id>/summarise", methods=["POST"])
@require_auth
def summarise_deal(deal_id: str):
    """Run two-stage document summary for a deal.
    Reads all uploaded documents from Supabase, runs LLM pipeline,
    stores result in deals.summary_json, returns full summary."""
    if not supabase:
        return jsonify({"error": "Database unavailable"}), 503

    # Verify deal ownership
    try:
        deal = supabase.table("deals") \
            .select("id, deal_name, summary_json") \
            .eq("id", deal_id) \
            .eq("user_id", request.user_id) \
            .single() \
            .execute()
        if not deal.data:
            return jsonify({"error": "Deal not found"}), 404
    except Exception as e:
        return jsonify({"error": "Deal not found"}), 404

    # Return cached result immediately if already analysed AND flags are populated.
    # IMPORTANT: if cached flags=[] (broken previous run), we must re-run the analysis.
    # An empty flags array with a deal_score means the previous run was truncated.
    existing = deal.data.get("summary_json")
    if (existing
            and existing.get("deal_score") is not None
            and isinstance(existing.get("flags"), list)
            and len(existing.get("flags", [])) > 0):
        return jsonify({"ok": True, "status": "complete", **existing}), 200
    # If we reach here: either no summary yet, OR summary exists but flags=[] (re-run needed)

    # Check usage allowance
    try:
        profile = supabase.table("profiles") \
            .select("plan, summaries_used, usage_reset_date") \
            .eq("id", request.user_id) \
            .single() \
            .execute()

        if profile.data:
            p = profile.data
            plan = p.get("plan", "starter")
            used = p.get("summaries_used", 0)
            limits = {"free": 1, "starter": 5, "professional": 20, "enterprise": 999}
            limit = limits.get(plan, 1)

            if not DEV_BYPASS_LIMITS and used >= limit:
                return jsonify({
                    "error": "summary_limit_reached",
                    "used": used,
                    "limit": limit,
                    "plan": plan,
                }), 402
    except Exception as e:
        app.logger.warning(f"Usage check failed: {e} — proceeding")

    # Fetch all documents with extracted text
    try:
        docs_result = supabase.table("documents") \
            .select("doc_type, file_name, extracted_text, page_count, extraction_status") \
            .eq("deal_id", deal_id) \
            .eq("user_id", request.user_id) \
            .execute()
        documents = docs_result.data or []
    except Exception as e:
        return jsonify({"error": f"Could not fetch documents: {e}"}), 500

    print(f"DEBUG: Found {len(documents)} documents for deal {deal_id}", flush=True)

    if not documents:
        return jsonify({"error": "No documents found for this deal"}), 400

    # Single combined LLM call — fast path
    try:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(__file__))
        # Build prioritised text inline — bypass external service
        _PRIORITY = ['special_conditions','addendum','title_register','lease',
                     'title_plan','deed','freehold','tenancy_ast',
                     'local_auth_search','environmental','epc','survey','auction_tcs','unknown']
        _docs_sorted = sorted(documents,
            key=lambda d: _PRIORITY.index(d.get('doc_type','unknown'))
                          if d.get('doc_type','unknown') in _PRIORITY else 99)
        _parts = []
        _total = 0
        _HARD_CAP = 40000   # ~10k tokens — target <20s LLM response
        _PER_DOC  = 6000    # max per document
        for _doc in _docs_sorted:
            _txt = (_doc.get('extracted_text') or '').strip()
            if not _txt:
                continue
            _label = f"=== {_doc.get('doc_type','unknown').upper()}: {_doc.get('file_name','')} ===\n"
            _capped = _txt[:_PER_DOC] + ('\n[...truncated...]' if len(_txt) > _PER_DOC else '')
            _chunk  = _label + _capped + '\n\n'
            if _total + len(_chunk) > _HARD_CAP:
                _rem = _HARD_CAP - _total - len(_label) - 20
                if _rem > 300:
                    _parts.append(_label + _txt[:_rem] + '\n[...truncated...]\n\n')
                break
            _parts.append(_chunk)
            _total += len(_chunk)
        truncated = ''.join(_parts)

        # ── Database context verification — log what we're actually sending ──
        docs_with_text = sum(1 for d in documents if (d.get('extracted_text') or '').strip())
        app.logger.info(
            f"[summarise] deal={deal_id} docs_total={len(documents)} "
            f"docs_with_text={docs_with_text} "
            f"truncated_chars={len(truncated)} "
            f"doc_types={[d.get('doc_type','?') for d in documents]}"
        )
        # Print to stdout so it's visible in Render logs regardless of log level
        print(
            f"[summarise] PROMPT SIZE: {len(truncated)} chars | "
            f"{docs_with_text}/{len(documents)} docs have text | "
            f"first_200: {truncated[:200]!r}",
            flush=True
        )

        # ── Guard: refuse to call LLM with empty text ──
        # If every document has extraction_status='empty' (scanned PDFs), truncated="" here.
        # A blank prompt to the LLM produces garbage or an API error. Surface it clearly.
        if not truncated.strip():
            docs_with_text = sum(1 for d in documents if (d.get('extracted_text') or '').strip())
            return jsonify({
                "error": "no_text_extracted",
                "detail": (
                    f"All {len(documents)} documents appear to be image-based or scanned PDFs. "
                    f"No text could be extracted ({docs_with_text} of {len(documents)} had text). "
                    "Please ensure your PDFs contain selectable text, not scanned images."
                ),
                "documents_count": len(documents),
                "docs_with_text": docs_with_text,
            }), 400

        COMBINED_SYSTEM = """You are a UK auction property legal analyst. Your job is to FIND EVERY RISK in this legal pack. Be aggressive and thorough — an investor's money is at stake.

Return ONLY valid JSON. No prose, no markdown fences. Exactly this structure (flags MUST come first):
{
  "flags": [
    {
      "severity": "critical|high|missing|note",
      "title": "specific risk title — max 10 words",
      "summation": "one sentence: what this means for the investor",
      "evidence": "verbatim quote from document — max 30 words",
      "implication": "financial or legal impact — max 20 words",
      "action": "what investor must do — max 15 words",
      "source_document": "document filename",
      "source_clause": "clause number or null",
      "source_page": null,
      "legal_risk_weight": 7
    }
  ],
  "flag_counts": {"critical": 0, "high": 0, "missing": 0, "note": 0},
  "deal_score": 0,
  "viability_statement": "2-3 sentences: investor verdict",
  "property": {"address": "full address", "postcode": "postcode", "lot_number": "lot", "type": "BTL/HMO/Commercial/etc", "tenure": "Freehold/Leasehold", "lease_years": null, "guide_price_pence": null},
  "completion_terms": {"deposit_pct": null, "deposit_refundable": null, "completion_days": null, "completion_type": "working", "buyers_premium_pct": null, "vacant_possession": null},
  "special_conditions": {
    "buyers_premium_pct": null,
    "buyers_premium_gbp": null,
    "admin_fee_gbp": null,
    "vat_elected": false,
    "seller_legal_costs_gbp": null,
    "search_fee_reimbursement": false,
    "completion_days": null,
    "deposit_pct": null,
    "non_refundable_deposit": false,
    "conditional_sale": false,
    "overage_clause": false,
    "addendum_present": false,
    "addendum_date": null,
    "addendum_notes": null,
    "unusual_clauses": [],
    "true_cost_additions_notes": null,
    "special_conditions_present": false,
    "special_conditions_missing": false
  },
  "pack_completeness": {"completeness_pct": 0, "present_count": 0, "total": 13},
  "documents_processed": 0
}

FLAG EXTRACTION RULES — YOU MUST FOLLOW ALL OF THEM:
1. NEVER return an empty flags array. Every legal pack has risks. If a pack seems clean, flag what is MISSING.
2. Flag EVERY one of these if present: restrictive covenants, chancel repair, mining/subsidence, flood risk, Japanese knotweed, Article 4 directions, HMO licensing, short lease (<85 years), ground rent escalation, service charge >£2500/yr, absent landlord, possessory title, missing searches, auction clauses (non-refundable deposit, 28-day completion, buyers premium), tenancy issues (sitting tenant, AST expiry, rent arrears), planning enforcement notices.
3. Flag MISSING documents: if Special Conditions, Title Register, Local Search, Environmental Search, EPC are absent — each is a MISSING flag.
4. Minimum flags: generate at least 1 flag per document that contains a clause. Aim for 10-20 flags total.
5. Scoring: Start at 100. Deduct critical=-12, high=-6, missing=-4, note=-1.
6. Keep evidence quotes SHORT (max 30 words) — critical for fitting all flags within token budget.
7. The flags array MUST be complete before flag_counts. Do not close the JSON until all flags are written.

FEW-SHOT EXAMPLE — this is exactly what one flag object must look like:
{"severity": "critical", "title": "Missing Local Authority Search", "summation": "No local search in pack — planning restrictions and enforcement notices unknown.", "evidence": "Document not present in legal pack", "implication": "Unknown planning restrictions could prevent intended use", "action": "Order local search before bidding — allow 5-10 working days", "source_document": "Not present", "source_clause": null, "source_page": null, "legal_risk_weight": 9}

Another example (informational note):
{"severity": "note", "title": "Freehold Title Verified", "summation": "Property held as absolute freehold with no charges registered.", "evidence": "Absolute freehold title confirmed in register entry A", "implication": "No ground rent or service charge obligations", "action": "Verify no covenants restrict intended use", "source_document": "title_register.pdf", "source_clause": "A: Property Register", "source_page": 1, "legal_risk_weight": 1}

A blank flags array is a SYSTEM FAILURE. Minimum 3 flags required even for a clean pack.

SPECIAL CONDITIONS EXTRACTION — populate the special_conditions object:
1. buyers_premium_pct/gbp: extract any buyer's premium or administration fee stated in special conditions
2. vat_elected: true if the property is elected for VAT (makes purchase price +20%)
3. seller_legal_costs_gbp: any amount buyer must pay toward seller's legal costs
4. completion_days: extract actual completion period (28 = standard, <28 = non-standard red flag)
5. non_refundable_deposit: true if deposit described as non-refundable beyond exchange
6. addendum_present: true if any addendum, amendment notice, or day-of-sale notice is in the pack
7. addendum_date: date of addendum if present
8. unusual_clauses: list any clauses that are non-standard or investor-unfavourable
9. special_conditions_missing: true if no Special Conditions of Sale document is present in pack
10. true_cost_additions_notes: plain English summary of all costs above hammer price"""


        # Run LLM in background thread — return immediately, frontend polls for result
        import threading as _t

        # Capture request context vars before thread (Flask request doesn't survive threads)
        _user_id = request.user_id
        _deal_id = deal_id

        def _run_and_store():
            try:
                result = _llm_json_anthropic(
                    system=COMBINED_SYSTEM,
                    prompt=f"Analyse this auction legal pack:\n\n{truncated}",
                    temperature=0.1,
                )

                # ── Schema enforcement: guarantee frontend contract is always met ──
                # flags must be a list (never null/missing — workbench reads data.flags || [])
                if not isinstance(result.get("flags"), list):
                    result["flags"] = []

                # ── Minimum-flag guarantee ──
                # If the LLM processed real documents but returned zero flags, something went wrong.
                # Inject a system note so the workbench is never blank and the user knows
                # analysis ran. This is a safety net — the prompt changes above should prevent this.
                if len(result["flags"]) == 0 and len(documents) > 0:
                    docs_with_text = sum(1 for d in documents if (d.get("extracted_text") or "").strip())
                    print(
                        f"DEBUG: Found {len(documents)} documents for deal {_deal_id} "
                        f"({docs_with_text} with text). LLM returned 0 flags — injecting system note.",
                        flush=True
                    )
                    result["flags"] = [{
                        "severity": "note",
                        "title": "Analysis complete — no specific flags raised",
                        "summation": (
                            f"The LLM analysed {len(documents)} documents "
                            f"({docs_with_text} with extracted text) and identified no specific risk flags. "
                            "This may indicate a clean pack, or that text extraction was limited. "
                            "Always have a solicitor review before bidding."
                        ),
                        "evidence":    "System generated — no clause evidence",
                        "implication": "No automated flags does not guarantee a clean legal pack",
                        "action":      "Commission independent solicitor review",
                        "source_document": "System",
                        "source_clause":   None,
                        "source_page":     None,
                        "legal_risk_weight": 1,
                    }]

                # ALWAYS recompute flag_counts from the actual flags array.
                # The LLM sometimes returns mismatched counts vs the array contents,
                # especially if the response was near the token limit.
                result["flag_counts"] = {
                    "critical": sum(1 for f in result["flags"] if (f.get("severity") or "").lower() == "critical"),
                    "high":     sum(1 for f in result["flags"] if (f.get("severity") or "").lower() == "high"),
                    "missing":  sum(1 for f in result["flags"] if (f.get("severity") or "").lower() == "missing"),
                    "note":     sum(1 for f in result["flags"] if (f.get("severity") or "").lower() == "note"),
                }

                # ── CEILING ENGINE ────────────────────────────────────────────────────
                # Only runs if ceiling_engine.py is available in services/
                if _ceiling_engine_available and _calc_ceiling:
                 try:
                    _deal_row = supabase.table("deals").select(
                        "financials_json,area_json,guide_price,deal_type"
                    ).eq("id", _deal_id).single().execute()
                    _deal_data = _deal_row.data or {}

                    _fins = (_deal_data.get("financials_json") or {})
                    _fins_inputs = _fins.get("inputs") or _fins or {}

                    # Inject comps avg from area_json if available
                    _area = _deal_data.get("area_json") or {}
                    _housing = _area.get("housing") or {}
                    _comps = _housing.get("soldComps") or _housing.get("value") or []
                    _comp_prices = [c.get("price") for c in _comps if c.get("price")]
                    if _comp_prices and not _fins_inputs.get("comps_avg_value"):
                        _fins_inputs["comps_avg_value"] = round(
                            sum(_comp_prices) / len(_comp_prices)
                        )

                    _strategy = (
                        _fins_inputs.get("strategy")
                        or _deal_data.get("deal_type")
                        or result.get("property", {}).get("type")
                        or "BTL"
                    )
                    # Normalise strategy string
                    _strategy_map = {
                        "hmo": "HMO", "btl": "BTL", "flip": "Flip",
                        "brrr": "BRRR", "sa": "Serviced Accommodation",
                        "serviced": "Serviced Accommodation",
                    }
                    _strategy = _strategy_map.get(
                        str(_strategy).lower(), str(_strategy)
                    )

                    result["ceiling"] = _calc_ceiling(
                        legal_flags=result["flags"],
                        financial_inputs=_fins_inputs,
                        base_valuation=None,
                        strategy=_strategy,
                    )
                    app.logger.info(
                        f"[ceiling] deal={_deal_id} strategy={_strategy} "
                        f"range={result['ceiling'].get('ceiling_range')} "
                        f"confidence={result['ceiling'].get('confidence')}"
                    )
                 except Exception as _ce:
                    app.logger.warning(f"[ceiling] Ceiling calculation failed for {_deal_id}: {_ce}")
                    result["ceiling"] = None
                # ─────────────────────────────────────────────────────────────────────

                app.logger.info(
                    f"[summarise] deal={_deal_id} score={result.get('deal_score')} "
                    f"flags={len(result['flags'])} counts={result['flag_counts']}"
                )
                print(
                    f"[summarise] RESULT deal={_deal_id} score={result.get('deal_score')} "
                    f"flags={len(result['flags'])} counts={result['flag_counts']}",
                    flush=True
                )
                # deal_score must be a number
                if result.get("deal_score") is None:
                    result["deal_score"] = 50  # safe fallback — signals analysis ran
                # property must be a dict
                if not isinstance(result.get("property"), dict):
                    result["property"] = {}
                # completion_terms must be a dict
                if not isinstance(result.get("completion_terms"), dict):
                    result["completion_terms"] = {}
                if not isinstance(result.get("special_conditions"), dict):
                    result["special_conditions"] = {}
                # pack_completeness must be a dict
                if not isinstance(result.get("pack_completeness"), dict):
                    result["pack_completeness"] = {"completeness_pct": 0, "present_count": 0, "total": 13}

                result["documents_processed"] = result.get("documents_processed") or len(documents)
                prop = result.get("property") or {}

                # Extract guide price from LLM output and normalise to pounds
                # LLM stores in guide_price_pence but value is sometimes pounds not pence
                _raw_gp = None
                _gpp = prop.get("guide_price_pence")
                _gp  = prop.get("guide_price")
                if _gp and float(_gp) > 5000:
                    _raw_gp = round(float(_gp))
                elif _gpp and float(_gpp) > 100000:
                    _raw_gp = round(float(_gpp) / 100)  # genuine pence
                elif _gpp and float(_gpp) > 5000:
                    _raw_gp = round(float(_gpp))  # LLM gave pounds not pence

                update_payload = {
                    "summary_json": result,
                    "deal_score":   result.get("deal_score"),
                    "status":       "analysed",
                    "updated_at":   now_iso(),
                    "address":      prop.get("address"),
                    "postcode":     prop.get("postcode") or None,
                    "deal_type":    prop.get("type"),
                }
                if _raw_gp:
                    update_payload["guide_price"] = _raw_gp

                supabase.table("deals").update(update_payload).eq("id", _deal_id).execute()
                # Increment usage counter
                try:
                    prof = supabase.table("profiles").select("summaries_used").eq("id", _user_id).single().execute()
                    used = (prof.data or {}).get("summaries_used", 0)
                    supabase.table("profiles").update({"summaries_used": used + 1}).eq("id", _user_id).execute()
                except Exception:
                    pass
                app.logger.info(f"Background analysis complete for deal {_deal_id}")
            except Exception as e:
                import traceback as _tb
                # anthropic SDK exceptions often have empty str(e) — extract all fields
                error_type  = type(e).__name__
                error_str   = str(e)
                error_repr  = repr(e)
                # anthropic-specific: status_code, response body, request_id
                status_code = getattr(e, 'status_code', None)
                body        = getattr(e, 'body', None) or getattr(e, 'response', None)
                request_id  = getattr(e, 'request_id', None)
                tb_short    = _tb.format_exc()[-600:]

                # Build a non-empty error message from whatever is available
                error_msg = (
                    error_str
                    or (f"{error_type}: status={status_code} body={body}" if status_code else "")
                    or error_repr
                    or tb_short
                    or "Unknown error in analysis thread"
                )
                detail_msg = (
                    f"type={error_type} | status={status_code} | body={body} | "
                    f"request_id={request_id} | repr={error_repr}"
                )
                app.logger.error(
                    f"[summarise] THREAD FAILED deal={_deal_id} | {detail_msg}\n{tb_short}"
                )
                try:
                    supabase.table("deals").update({
                        "status": "error",
                        "summary_json": {
                            "error":       error_msg,
                            "error_detail": detail_msg,
                            "deal_score":  None,
                            "flags":       [],
                            "flag_counts": {"critical": 0, "high": 0, "missing": 0, "note": 0},
                        },
                        "updated_at": now_iso(),
                    }).eq("id", _deal_id).execute()
                except Exception as _se:
                    app.logger.error(f"[summarise] Could not write error state to DB: {_se}")

        thread = _t.Thread(target=_run_and_store, daemon=True)
        thread.start()

        # Return immediately — frontend polls /api/deals/<id> for summary_json
        return jsonify({"ok": True, "status": "processing", "deal_id": deal_id}), 202

    except Exception as e:
        app.logger.exception("Summarise setup failed")
        return jsonify({"error": str(e)}), 500

    prop = {}  # unreachable but keeps linter happy
    # Write to analyses table — matches actual Supabase schema
    # Columns: id, user_id, created_at, updated_at, analysis_data (jsonb), analysis_name (text), analysis (jsonb)
    try:
        supabase.table("analyses").insert({
            "user_id":       request.user_id,
            "analysis_name": f"Summary — {deal.data.get('deal_name') or deal_id[:8]}",
            "analysis_data": {
                "deal_id":           deal_id,
                "analysis_type":     "summary",
                "deal_score":        summary.get("deal_score"),
                "flag_counts":       summary.get("flag_counts") or {},
                "property":          prop,
                "completion_terms":  summary.get("completion_terms") or {},
                "viability_statement": summary.get("viability_statement") or "",
                "findings_count":    summary.get("findings_count", 0),
                "documents_processed": summary.get("documents_processed", 0),
            },
            "analysis": summary,
        }).execute()
    except Exception as e:
        app.logger.warning(f"Could not write to analyses table: {e}")

    # Record usage
    try:
        supabase.table("profiles").update({
            "summaries_used": (profile.data.get("summaries_used", 0) + 1)
            if profile and profile.data else 1
        }).eq("id", request.user_id).execute()

        supabase.table("usage_events").insert({
            "user_id":    request.user_id,
            "event_type": "summary",
            "deal_id":    deal_id,
            "amount_pence": 0,
        }).execute()
    except Exception as e:
        app.logger.warning(f"Usage recording failed: {e}")

    return jsonify(summary), 200


# ── FULL ANALYSIS ───────────────────────────────────────────
FULL_ANALYSIS_SYSTEM = """You are a UK property legal analyst. Analyse the auction documents provided.
Every finding must be directly evidenced from document text.
Do not infer, suggest, or recommend. State only what the documents explicitly state.
Return ONLY valid JSON — no prose, no markdown fences.

{
  "deal_score": number,
  "adjusted_score": number,
  "adjusted_score_rationale": string,
  "jis_findings": [
    {
      "number": number,
      "title": "string — one line, max 12 words, investor-facing",
      "severity": "critical | high | opportunity",
      "finding": "string — what the document states",
      "evidence": "string — exact verbatim quote from document, max 40 words",
      "implication": "string — magnitude and consequence, no inference beyond document",
      "action": "string — what to do, not a recommendation",
      "source_document": "string",
      "source_clause": "string or null",
      "source_page": number or null
    }
  ],
  "flags": [
    {
      "severity": "critical | high | missing | note",
      "title": "string — one line, max 12 words",
      "summation": "string — one sentence, factual, clause-referenced",
      "source_document": "string",
      "source_clause": "string or null",
      "source_page": number or null,
      "legal_risk_weight": number
    }
  ],
  "solicitor_questions": ["string"],
  "flag_counts": {
    "critical": number,
    "high": number,
    "missing": number,
    "note": number
  },
  "viability_statement": "string — 2-3 sentences, factual, no verdict"
}

SCORING: Start at 100. Deduct: critical=12, high=6, missing=4.
adjusted_score: score if resolvable flags resolved.
jis_findings: 5-10 findings, most material issues first.
flags: all issues including missing documents."""


@app.route("/api/deals/<deal_id>/analyse", methods=["POST"])
@require_auth
def analyse_deal(deal_id: str):
    """Run full legal analysis for a deal.
    Uses all uploaded document text. Stores result in deals.analysis_json."""
    if not supabase:
        return jsonify({"error": "Database unavailable"}), 503

    # Verify ownership
    try:
        deal = supabase.table("deals")             .select("id, deal_name, analysis_json")             .eq("id", deal_id)             .eq("user_id", request.user_id)             .single().execute()
        if not deal.data:
            return jsonify({"error": "Deal not found"}), 404
    except Exception:
        return jsonify({"error": "Deal not found"}), 404

    # Return cached analysis if available and not forced refresh
    force = request.args.get("force", "").lower() in ("1", "true")
    if deal.data.get("analysis_json") and not force:
        return jsonify(deal.data["analysis_json"]), 200

    # Fetch documents
    try:
        docs = supabase.table("documents")             .select("doc_type, file_name, extracted_text, page_count")             .eq("deal_id", deal_id)             .eq("user_id", request.user_id)             .execute()
        documents = docs.data or []
    except Exception as e:
        return jsonify({"error": f"Could not fetch documents: {e}"}), 500

    if not documents:
        return jsonify({"error": "No documents found for this deal"}), 400

    # Build combined text
    try:
        from legal_analysis import _build_combined_text, DOC_TYPE_LABELS
        combined = _build_combined_text(documents)
    except Exception:
        parts = []
        for doc in documents:
            text = (doc.get("extracted_text") or "").strip()
            if text:
                label = doc.get('doc_type', 'doc')
                fname = doc.get('file_name', '')
                parts.append(f"=== {label} ({fname}) ===\n{text[:15000]}")
        combined = "\n\n".join(parts)

    if not combined.strip():
        return jsonify({"error": "No text extracted from documents"}), 400

    # Truncate
    if len(combined) > 100000:
        truncated = combined[:70000] + "\n\n[...truncated...]\n\n" + combined[-20000:]
    else:
        truncated = combined

    # Run LLM
    try:
        result = _llm_json_anthropic(
            system=FULL_ANALYSIS_SYSTEM,
            prompt="Analyse these auction documents and return the full analysis JSON:\n\n" + truncated,
            temperature=0.1,
        )
    except Exception as e:
        app.logger.exception("Full analysis LLM failed")
        return jsonify({"error": str(e)}), 500

    # Add summary flags from deal if available
    try:
        deal_full = supabase.table("deals")             .select("summary_json, deal_score")             .eq("id", deal_id).single().execute()
        if deal_full.data and deal_full.data.get("summary_json"):
            summary = deal_full.data["summary_json"]
            # Merge missing doc flags from summary if not in analysis
            summary_flags = summary.get("flags", [])
            missing_flags = [f for f in summary_flags if f.get("severity") == "missing"]
            analysis_flags = result.get("flags", [])
            existing_titles = {f.get("title","") for f in analysis_flags}
            for mf in missing_flags:
                if mf.get("title","") not in existing_titles:
                    analysis_flags.append(mf)
            result["flags"] = analysis_flags
    except Exception:
        pass

    # Store result in deals table
    try:
        supabase.table("deals").update({
            "analysis_json": result,
            "deal_score":    result.get("deal_score"),
            "updated_at":    now_iso(),
        }).eq("id", deal_id).execute()
    except Exception as e:
        app.logger.warning(f"Could not store analysis in deals: {e}")

    # Write to analyses table — matches actual Supabase schema
    try:
        fc = result.get("flag_counts") or {}
        deal_name_short = deal_id[:8]
        try:
            dn = supabase.table("deals").select("deal_name").eq("id", deal_id).single().execute()
            deal_name_short = (dn.data or {}).get("deal_name") or deal_id[:8]
        except Exception:
            pass
        supabase.table("analyses").insert({
            "user_id":       request.user_id,
            "analysis_name": f"Full Analysis — {deal_name_short}",
            "analysis_data": {
                "deal_id":           deal_id,
                "analysis_type":     "full_analysis",
                "deal_score":        result.get("deal_score"),
                "adjusted_score":    result.get("adjusted_score"),
                "flag_counts":       fc,
                "viability_statement": result.get("viability_statement") or "",
            },
            "analysis": result,
        }).execute()
    except Exception as e:
        app.logger.warning(f"Could not write to analyses table: {e}")

    # Record usage
    try:
        supabase.table("usage_events").insert({
            "user_id":    request.user_id,
            "event_type": "analysis",
            "deal_id":    deal_id,
            "amount_pence": 0,
        }).execute()
    except Exception:
        pass

    return jsonify(result), 200


# ── AUTH PROFILE ─────────────────────────────────────────────
@app.route("/api/auth/me", methods=["GET"])
@require_auth
def get_me():
    """Get current user profile."""
    if not supabase:
        return jsonify({"error": "Database unavailable"}), 503
    try:
        result = supabase.table("profiles") \
            .select("*") \
            .eq("id", request.user_id) \
            .single() \
            .execute()
        if not result.data:
            return jsonify({"error": "Profile not found"}), 404
        # Never return sensitive fields
        profile = result.data
        profile.pop("stripe_customer_id", None)
        profile.pop("stripe_subscription_id", None)
        return jsonify({"ok": True, "profile": profile}), 200
    except Exception as e:
        app.logger.exception("get_me failed")
        return jsonify({"error": str(e)}), 500


# ── FINANCIAL MODEL ──────────────────────────────────────────

def _calculate_financials(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate property investment yields from raw inputs. All GBP in £ (float)."""
    def _gbp(v: Any) -> Optional[float]:
        f = safe_float(v)
        if f is None:
            return None
        if f > 10_000 and isinstance(v, int):
            return f / 100.0
        return f

    def _pct(v: Any) -> Optional[float]:
        f = safe_float(v)
        if f is None:
            return None
        return max(0.0, min(100.0, f))

    purchase_price      = _gbp(inputs.get("purchase_price"))
    guide_price         = _gbp(inputs.get("guide_price"))
    renovation_cost     = _gbp(inputs.get("renovation_cost")) or 0.0
    monthly_rent        = _gbp(inputs.get("monthly_rent"))
    annual_rent         = _gbp(inputs.get("annual_rent"))
    void_weeks          = safe_float(inputs.get("void_weeks")) or 0.0
    management_pct      = _pct(inputs.get("management_pct")) or 0.0
    service_charge_pa   = _gbp(inputs.get("service_charge_pa")) or 0.0
    ground_rent_pa      = _gbp(inputs.get("ground_rent_pa")) or 0.0
    insurance_pa        = _gbp(inputs.get("insurance_pa")) or 0.0
    maintenance_pct     = _pct(inputs.get("maintenance_pct")) or 1.0
    buyers_premium_pct  = _pct(inputs.get("buyers_premium_pct")) or 0.0
    stamp_duty          = _gbp(inputs.get("stamp_duty")) or 0.0
    legal_fees          = _gbp(inputs.get("legal_fees")) or 1500.0
    survey_cost         = _gbp(inputs.get("survey_cost")) or 0.0
    finance_rate_pct    = _pct(inputs.get("finance_rate_pct")) or 0.0
    ltv_pct             = _pct(inputs.get("ltv_pct")) or 0.0
    target_yield        = _pct(inputs.get("target_yield")) or 6.0
    exit_price          = _gbp(inputs.get("exit_price"))
    hold_years          = safe_float(inputs.get("hold_years")) or 5.0

    if purchase_price is None:
        return {"ok": False, "error": "purchase_price is required"}

    buyers_premium      = purchase_price * (buyers_premium_pct / 100.0)
    total_acquisition   = purchase_price + buyers_premium + stamp_duty + legal_fees + survey_cost
    total_invested      = total_acquisition + renovation_cost

    if annual_rent is None and monthly_rent is not None:
        annual_rent = monthly_rent * 12.0
    if annual_rent is None:
        annual_rent = 0.0

    void_weeks_pa       = min(float(void_weeks), 52.0)
    occupied_weeks      = 52.0 - void_weeks_pa
    void_adj_rent_pa    = annual_rent * (occupied_weeks / 52.0)

    management_cost_pa  = void_adj_rent_pa * (management_pct / 100.0)
    maintenance_cost_pa = purchase_price * (maintenance_pct / 100.0)
    total_expenses_pa   = (management_cost_pa + maintenance_cost_pa +
                           service_charge_pa + ground_rent_pa + insurance_pa)

    loan_amount         = purchase_price * (ltv_pct / 100.0) if ltv_pct > 0 else 0.0
    annual_interest     = loan_amount * (finance_rate_pct / 100.0)
    equity              = total_invested - loan_amount

    noi                 = void_adj_rent_pa - total_expenses_pa
    net_cashflow_pa     = noi - annual_interest

    gross_yield_pct     = (annual_rent / purchase_price * 100.0)  if purchase_price > 0 else None
    net_yield_pct       = (noi / total_invested * 100.0)          if total_invested > 0 else None
    cash_on_cash_pct    = (net_cashflow_pa / equity * 100.0)      if (equity and equity > 0) else None

    max_bid_gross       = (annual_rent / (target_yield / 100.0))  if (annual_rent > 0 and target_yield > 0) else None
    fixed_exp           = service_charge_pa + ground_rent_pa + insurance_pa
    rent_after_mgmt     = void_adj_rent_pa * (1.0 - management_pct / 100.0)
    denominator         = (target_yield / 100.0) + (maintenance_pct / 100.0)
    max_bid_net         = ((rent_after_mgmt - fixed_exp) / denominator) if denominator > 0 else None

    total_rent_received = net_cashflow_pa * hold_years
    capital_gain        = (exit_price - purchase_price) if exit_price else None
    total_return        = (total_rent_received + capital_gain) if capital_gain is not None else total_rent_received
    simple_roi_pct      = (total_return / total_invested * 100.0)       if total_invested > 0 else None
    annualised_roi_pct  = (simple_roi_pct / hold_years)                 if (simple_roi_pct is not None and hold_years > 0) else None
    payback_years       = (total_invested / noi)                        if noi > 0 else None

    flags = []
    if gross_yield_pct is not None and gross_yield_pct < 5.0:
        flags.append({"type": "warning", "msg": f"Gross yield {gross_yield_pct:.1f}% is below 5% threshold"})
    if gross_yield_pct is not None and gross_yield_pct >= 8.0:
        flags.append({"type": "positive", "msg": f"Strong gross yield of {gross_yield_pct:.1f}%"})
    if net_cashflow_pa is not None and net_cashflow_pa < 0:
        flags.append({"type": "critical", "msg": "Net cashflow is negative after finance costs"})
    if guide_price and purchase_price > guide_price * 1.15:
        flags.append({"type": "warning", "msg": f"Purchase {((purchase_price/guide_price)-1)*100:.0f}% above guide price"})
    if max_bid_gross and purchase_price > max_bid_gross:
        flags.append({"type": "warning", "msg": f"Price exceeds max bid for {target_yield}% target yield"})

    def _r2(v: Optional[float]) -> Optional[float]:
        return round(v, 2) if v is not None else None

    return {
        "ok": True,
        "inputs": {
            "purchase_price": _r2(purchase_price), "guide_price": _r2(guide_price),
            "renovation_cost": _r2(renovation_cost), "annual_rent": _r2(annual_rent),
            "monthly_rent": _r2(annual_rent / 12.0) if annual_rent else None,
            "void_weeks": _r2(void_weeks), "management_pct": _r2(management_pct),
            "service_charge_pa": _r2(service_charge_pa), "ground_rent_pa": _r2(ground_rent_pa),
            "insurance_pa": _r2(insurance_pa), "maintenance_pct": _r2(maintenance_pct),
            "buyers_premium_pct": _r2(buyers_premium_pct), "stamp_duty": _r2(stamp_duty),
            "legal_fees": _r2(legal_fees), "survey_cost": _r2(survey_cost),
            "finance_rate_pct": _r2(finance_rate_pct), "ltv_pct": _r2(ltv_pct),
            "target_yield": _r2(target_yield), "exit_price": _r2(exit_price), "hold_years": _r2(hold_years),
        },
        "acquisition": {
            "buyers_premium": _r2(buyers_premium), "stamp_duty": _r2(stamp_duty),
            "legal_fees": _r2(legal_fees), "survey_cost": _r2(survey_cost),
            "total_acquisition": _r2(total_acquisition), "renovation_cost": _r2(renovation_cost),
            "total_invested": _r2(total_invested),
        },
        "income": {
            "gross_annual_rent": _r2(annual_rent), "void_weeks_pa": _r2(void_weeks_pa),
            "void_adj_rent_pa": _r2(void_adj_rent_pa), "monthly_net_rent": _r2(void_adj_rent_pa / 12.0),
        },
        "expenses": {
            "management_cost_pa": _r2(management_cost_pa), "maintenance_cost_pa": _r2(maintenance_cost_pa),
            "service_charge_pa": _r2(service_charge_pa), "ground_rent_pa": _r2(ground_rent_pa),
            "insurance_pa": _r2(insurance_pa), "total_expenses_pa": _r2(total_expenses_pa),
        },
        "finance": {
            "loan_amount": _r2(loan_amount), "ltv_pct": _r2(ltv_pct),
            "annual_interest": _r2(annual_interest), "equity": _r2(equity),
        },
        "returns": {
            "noi": _r2(noi), "net_cashflow_pa": _r2(net_cashflow_pa),
            "net_cashflow_pm": _r2(net_cashflow_pa / 12.0),
            "gross_yield_pct": _r2(gross_yield_pct), "net_yield_pct": _r2(net_yield_pct),
            "cash_on_cash_pct": _r2(cash_on_cash_pct), "payback_years": _r2(payback_years),
            "simple_roi_pct": _r2(simple_roi_pct), "annualised_roi_pct": _r2(annualised_roi_pct),
            "total_return": _r2(total_return), "capital_gain": _r2(capital_gain),
        },
        "max_bid": {
            "gross_target": _r2(max_bid_gross), "net_target": _r2(max_bid_net),
            "target_yield_pct": _r2(target_yield),
        },
        "flags": flags,
        "calculated_at": now_iso(),
    }


@app.route("/api/deals/<deal_id>/financials", methods=["GET"])
@require_auth
def get_financials(deal_id: str):
    """Retrieve saved financial model for a deal, seeded with guide price if not yet saved."""
    if not supabase:
        return jsonify({"error": "Database unavailable"}), 503
    try:
        result = supabase.table("deals") \
            .select("financials_json, guide_price, summary_json") \
            .eq("id", deal_id).eq("user_id", request.user_id).single().execute()
        if not result.data:
            return jsonify({"error": "Deal not found"}), 404

        financials = result.data.get("financials_json") or {}
        if not financials:
            guide = result.data.get("guide_price")
            summary = result.data.get("summary_json") or {}
            terms = summary.get("completion_terms") or {}
            prop  = summary.get("property") or {}
            gpp   = prop.get("guide_price_pence")
            financials = {
                "_seeded": True,
                "inputs": {
                    "purchase_price":    float(guide) if guide else None,
                    "guide_price":       gpp / 100.0 if gpp else (float(guide) if guide else None),
                    "buyers_premium_pct": terms.get("buyers_premium_pct"),
                    "monthly_rent":      None,
                    "renovation_cost":   None,
                    "target_yield":      6.0,
                    "ltv_pct":           0.0,
                    "finance_rate_pct":  0.0,
                    "management_pct":    10.0,
                    "maintenance_pct":   1.0,
                    "legal_fees":        1500.0,
                    "void_weeks":        2.0,
                    "hold_years":        5.0,
                }
            }
        return jsonify({"ok": True, "financials": financials}), 200
    except Exception as e:
        app.logger.exception("get_financials failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/deals/<deal_id>/financials", methods=["POST"])
@require_auth
def save_financials(deal_id: str):
    """
    Save financial inputs + calculate yields. Persists to deals.financials_json.
    Body: { purchase_price, monthly_rent, renovation_cost, ... }
    """
    if not supabase:
        return jsonify({"error": "Database unavailable"}), 503
    try:
        deal = supabase.table("deals") \
            .select("id, guide_price, summary_json") \
            .eq("id", deal_id).eq("user_id", request.user_id).single().execute()
        if not deal.data:
            return jsonify({"error": "Deal not found"}), 404
    except Exception:
        return jsonify({"error": "Deal not found"}), 404

    data = request.get_json(silent=True) or {}

    # Seed defaults from deal metadata when not provided
    if not data.get("guide_price") and not data.get("purchase_price"):
        gp = deal.data.get("guide_price")
        if gp:
            data.setdefault("guide_price", float(gp))
    if not data.get("buyers_premium_pct"):
        try:
            bpp = (deal.data.get("summary_json") or {}).get("completion_terms", {}).get("buyers_premium_pct")
            if bpp:
                data.setdefault("buyers_premium_pct", float(bpp))
        except Exception:
            pass

    result = _calculate_financials(data)
    if not result.get("ok"):
        return jsonify(result), 400

    try:
        supabase.table("deals").update({
            "financials_json": result,
            "updated_at":      now_iso(),
        }).eq("id", deal_id).execute()
    except Exception as e:
        app.logger.warning(f"Could not persist financials: {e}")

    return jsonify(result), 200


# ── DASHBOARD ────────────────────────────────────────────────

@app.route("/api/dashboard", methods=["GET"])
@require_auth
def get_dashboard():
    """Aggregated dashboard stats for the authenticated user's deals."""
    if not supabase:
        return jsonify({"error": "Database unavailable"}), 503
    try:
        result = supabase.table("deals") \
            .select("id, deal_name, title, address, postcode, status, deal_score, "
                    "guide_price, auction_date, deal_type, created_at, updated_at, "
                    "summary_json, financials_json, analysis_json, area_json") \
            .eq("user_id", request.user_id) \
            .neq("status", "archived") \
            .order("created_at", desc=True) \
            .execute()
        deals = result.data or []
    except Exception as e:
        app.logger.exception("dashboard fetch failed")
        return jsonify({"error": str(e)}), 500

    total_deals      = len(deals)
    scored           = [d for d in deals if d.get("deal_score") is not None]
    avg_score        = (sum(d["deal_score"] for d in scored) / len(scored)) if scored else None

    with_fin         = [d for d in deals if d.get("financials_json") and (d["financials_json"] or {}).get("ok")]
    gy_vals = [d["financials_json"]["returns"]["gross_yield_pct"] for d in with_fin
               if (d["financials_json"].get("returns") or {}).get("gross_yield_pct") is not None]
    ny_vals = [d["financials_json"]["returns"]["net_yield_pct"] for d in with_fin
               if (d["financials_json"].get("returns") or {}).get("net_yield_pct") is not None]
    eq_vals = [d["financials_json"]["finance"]["equity"] for d in with_fin
               if (d["financials_json"].get("finance") or {}).get("equity") and d["financials_json"]["finance"]["equity"] > 0]

    total_critical = total_high = total_missing = 0
    for d in deals:
        fc = (d.get("summary_json") or {}).get("flag_counts") or {}
        total_critical += int(fc.get("critical", 0))
        total_high     += int(fc.get("high", 0))
        total_missing  += int(fc.get("missing", 0))

    status_counts: Dict[str, int] = {}
    for d in deals:
        st = d.get("status") or "active"
        status_counts[st] = status_counts.get(st, 0) + 1

    today_str = datetime.utcnow().date().isoformat()
    cutoff    = (datetime.utcnow() + timedelta(days=30)).date().isoformat()
    upcoming  = []
    for d in deals:
        ad = d.get("auction_date")
        if ad and isinstance(ad, str) and today_str <= ad[:10] <= cutoff:
            prop        = ((d.get("summary_json") or {}).get("property") or
                          (d.get("summary_json") or {}).get("prop") or {})
            prop_addr   = prop.get("address") or ""
            raw_name    = d.get("deal_name") or d.get("title") or ""
            is_fallback = raw_name.startswith("Deal") and ("—" in raw_name or "-" in raw_name)
            display     = prop_addr if (is_fallback and prop_addr) else (raw_name or prop_addr or d.get("address") or d.get("postcode") or "")
            upcoming.append({
                "deal_id":     d["id"],
                "deal_name":   display,
                "address":     prop_addr or d.get("address") or d.get("postcode", ""),
                "auction_date": ad[:10],
                "deal_score":  d.get("deal_score"),
                "guide_price": d.get("guide_price"),
            })
    upcoming.sort(key=lambda x: x["auction_date"])

    recent_deals = []
    for d in deals[:10]:
        fin = d.get("financials_json") or {}
        ret = (fin.get("returns") or {})
        recent_deals.append({
            "deal_id":         d["id"],
            "deal_name":       d.get("deal_name") or d.get("title", ""),
            "address":         d.get("address") or "",
            "postcode":        d.get("postcode") or "",
            "deal_score":      d.get("deal_score"),
            "guide_price":     d.get("guide_price"),
            "auction_date":    d.get("auction_date"),
            "deal_type":       d.get("deal_type"),
            "status":          d.get("status", "active"),
            "outcome":         d.get("outcome"),
            "hammer_price":    d.get("hammer_price"),
            "hammer_date":     d.get("hammer_date"),
            "completion_period": d.get("completion_period") or 28,
            "completion_actions": d.get("completion_actions") or [],
            "created_at":      d.get("created_at"),
            "gross_yield_pct": ret.get("gross_yield_pct"),
            "net_yield_pct":   ret.get("net_yield_pct"),
            "net_cashflow_pm": ret.get("net_cashflow_pm"),
            "has_analysis":    bool(d.get("analysis_json")),
            "has_financials":  bool(fin.get("ok")),
            "flag_counts":     (d.get("summary_json") or {}).get("flag_counts") or {},
        })

    return jsonify({
        "ok": True,
        "summary": {
            "total_deals":            total_deals,
            "avg_deal_score":         round(avg_score, 1) if avg_score is not None else None,
            "deals_analysed":         len(scored),
            "deals_with_financials":  len(with_fin),
            "avg_gross_yield_pct":    round(sum(gy_vals)/len(gy_vals), 2) if gy_vals else None,
            "avg_net_yield_pct":      round(sum(ny_vals)/len(ny_vals), 2) if ny_vals else None,
            "total_equity_deployed":  round(sum(eq_vals), 2) if eq_vals else None,
            "total_critical_flags":   total_critical,
            "total_high_flags":       total_high,
            "total_missing_flags":    total_missing,
            "upcoming_auction_count": len(upcoming),
            "status_counts":          status_counts,
        },
        "recent_deals":  recent_deals,
        "upcoming":      upcoming,
        "retrieved_at":  now_iso(),
    }), 200


# ── HPI DATA DIAGNOSTIC ENDPOINT ─────────────────────────────
@app.route("/api/test-hpi", methods=["GET", "OPTIONS"])
def test_hpi_endpoint():
    """
    GET /api/test-hpi
    Diagnostic: verify uk_hpi_monthly is populated and queryable.
    Returns sample area_codes and whether England aggregate exists.
    NO AUTH REQUIRED — read-only diagnostic only.
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200
    results = {}
    # Test 1: query for England aggregate
    for code in ("E92000001", "England", "United Kingdom"):
        rows = supabase_data_query(
            "SELECT average_price, annual_change FROM public.uk_hpi_monthly WHERE area_code = %s ORDER BY date DESC LIMIT 1",
            (code,)
        )
        results[code] = rows[0] if rows else None
    # Test 2: query for Darlington LAD
    for code in ("E06000005", "Darlington"):
        rows = supabase_data_query(
            "SELECT average_price, annual_change FROM public.uk_hpi_monthly WHERE area_code = %s ORDER BY date DESC LIMIT 1",
            (code,)
        )
        results[code] = rows[0] if rows else None
    # Test 3: count total rows
    count_rows = supabase_data_query(
        "SELECT COUNT(*) AS total FROM public.uk_hpi_monthly",
        ()
    )
    results["total_rows"] = count_rows[0].get("total") if count_rows else "QUERY FAILED"
    return jsonify({"ok": True, "hpi_test": results}), 200


# ── AREA INTELLIGENCE ─────────────────────────────────────────



# ── CEILING ENGINE — live endpoint for Flag Workbench ────────────────────────
def _apply_audit_confidence_cap(ceiling: dict, area_data: dict) -> dict:
    """T-4 — mechanical confidence cap shared by Path A (_recompute_deal_ceiling)
    and Path B (/api/ceiling). Caps ceiling.confidence at 0.4 when the engine's
    own area audit declares any of:
      - insufficient_evidence
      - a tenure-filter skip
      - price_variance_pct > 50
    Mutates `ceiling` in place and returns it. Safe on legacy `area_data`
    without an audit block (returns unchanged). Never raises."""
    if not isinstance(ceiling, dict):
        return ceiling
    try:
        _m = ((area_data or {}).get("housing") or {}).get("metrics") or {}
        _a = _m.get("audit") if isinstance(_m.get("audit"), dict) else None
        _ins = bool(
            (_a or {}).get("insufficient_evidence") if _a is not None
            else _m.get("insufficient_evidence")
        )
        _skips_raw = (_a or {}).get("filters_skipped") if _a is not None else None
        _skips = _skips_raw if isinstance(_skips_raw, list) else []
        _tenure_skipped = any(
            isinstance(s, str) and s.startswith("tenure:") for s in _skips
        )
        _var_raw = _m.get("price_variance_pct")
        try:
            _var = float(_var_raw) if _var_raw is not None else None
        except (TypeError, ValueError):
            _var = None
        _high_var = _var is not None and _var > 50.0

        if _ins or _tenure_skipped or _high_var:
            _cap = 0.4
            _cur_raw = ceiling.get("confidence")
            try:
                _cur = float(_cur_raw) if _cur_raw is not None else None
            except (TypeError, ValueError):
                _cur = None
            if _cur is None or _cur > _cap:
                ceiling["confidence"] = _cap
                _reasons = []
                if _ins:            _reasons.append("insufficient_evidence")
                if _tenure_skipped: _reasons.append("tenure_filter_skipped")
                if _high_var:       _reasons.append(f"price_variance_pct={_var:.1f}")
                ceiling["confidence_capped_reasons"] = _reasons
                ceiling["confidence_capped_at"] = _cap
    except Exception as _err:
        print(f"[ceiling-cap skipped] {_err}")
    return ceiling


@app.route("/api/ceiling", methods=["POST", "OPTIONS"])
@require_auth
def ceiling_endpoint():
    """
    POST /api/ceiling
    Accept legal flags + financial inputs, return a bid ceiling range.
    Used by the Flag Workbench for live ceiling recalculation as flags
    are resolved.

    Body (JSON):
      {
        "legal_flags":       [...],       # array of flag dicts
        "financial_inputs":  {...},       # rent, yield, comps etc
        "base_valuation":    165000,      # optional RICS/AVM figure
        "strategy":          "BTL",       # BTL | HMO | Flip | BRRR | SA
        "deal_id":           "uuid"       # optional — to pull fins from DB
      }
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200

    try:
        body = request.get_json(silent=True) or {}

        legal_flags      = body.get("legal_flags", [])
        financial_inputs = body.get("financial_inputs", {})
        base_val         = body.get("base_valuation")
        strategy         = body.get("strategy", "BTL")
        deal_id          = body.get("deal_id")

        # D4 — captured during DB enrichment for the shared T-4 confidence cap.
        # None when no deal_id or no DB row (cap is then a no-op).
        _area_data_for_cap = None

        # If deal_id provided, enrich financial_inputs from DB
        if deal_id and supabase:
            try:
                row = supabase.table("deals").select(
                    "financials_json,area_json,deal_type,guide_price"
                ).eq("id", deal_id).eq("user_id", request.user_id).single().execute()

                if row.data:
                    d = row.data
                    fins = (d.get("financials_json") or {})
                    fins_inp = fins.get("inputs") or fins or {}

                    # Merge DB inputs — body overrides DB
                    merged = {**fins_inp, **financial_inputs}

                    # Normalise yield: financials_json stores as pct (7.0),
                    # ceiling_engine expects decimal (0.07).
                    for _yk in ("target_yield", "target_gross_yield"):
                        _yv = merged.get(_yk)
                        if _yv is not None:
                            try:
                                _yf = float(_yv)
                                if _yf > 1.0:
                                    merged[_yk] = round(_yf / 100.0, 6)
                            except (TypeError, ValueError):
                                pass

                    # ── D4 CANONICALIZATION ─────────────────────────────────
                    # /api/ceiling no longer derives its own comps_avg_value.
                    # The canonical base lives in summary_json.ceiling, written
                    # by _recompute_deal_ceiling (Path A) from the persisted
                    # soldComps. Path B (this endpoint, Workbench live) reads
                    # that canonical base and varies ONLY legal_flags.
                    #
                    # No fallback cascade. The pre-D4 cascade computed five
                    # different bases (median-normalised soldComps, housing
                    # metrics.median_price, summary_json avg_sold_price,
                    # guide × 1.15, HPI regional/national/local, England RPC)
                    # — that cascade is exactly what produced the Verdict /
                    # Workbench valuation divergence on b724a3ee.
                    #
                    # If canonical base is missing, financial_inputs is left
                    # without comps_avg_value; _calc_ceiling then returns the
                    # same base_method=none degraded ceiling that Path A
                    # produces in the same state. Structurally identical.
                    area = d.get("area_json") or {}
                    _area_data_for_cap = area  # capture for the shared T-4 cap below
                    _sj = d.get("summary_json") or {}
                    _canon_ceiling = _sj.get("ceiling") if isinstance(_sj, dict) else None
                    _canon_ceiling = _canon_ceiling if isinstance(_canon_ceiling, dict) else {}
                    _canon_base_raw = _canon_ceiling.get("base_valuation")
                    try:
                        _canon_base_f = float(_canon_base_raw) if _canon_base_raw is not None else None
                    except (TypeError, ValueError):
                        _canon_base_f = None

                    if _canon_base_f and _canon_base_f > 5000 and not merged.get("comps_avg_value"):
                        merged["comps_avg_value"]  = int(round(_canon_base_f))
                        merged["comps_source"]     = "canonical_persisted_base"
                        merged["comps_base_method"] = _canon_ceiling.get("base_method")

                    print(
                        f"[ceiling] deal={deal_id} "
                        f"canonical_base={('£' + format(merged['comps_avg_value'], ',')) if merged.get('comps_avg_value') else 'MISSING'} "
                        f"base_method={_canon_ceiling.get('base_method')} "
                        f"strategy={strategy}"
                    )

                    financial_inputs = merged

                    if not strategy or strategy == "BTL":
                        strategy = d.get("deal_type") or strategy or "BTL"

            except Exception as _de:
                app.logger.warning(f"[ceiling] DB enrich failed for {deal_id}: {_de}")

        if not isinstance(legal_flags, list):
            return jsonify({"error": "legal_flags must be an array"}), 400

        if not _ceiling_engine_available or not _calc_ceiling:
            return jsonify({"error": "ceiling_engine not available on this deployment"}), 503

        result = _calc_ceiling(
            legal_flags=legal_flags,
            financial_inputs=financial_inputs,
            base_valuation=float(base_val) if base_val else None,
            strategy=str(strategy),
        )

        # D4 — Path B (/api/ceiling) applies the same T-4 confidence cap as
        # Path A (_recompute_deal_ceiling). Both paths thus report identical
        # confidence semantics derived from the same persisted audit block.
        # _area_data_for_cap was set during DB enrichment; None otherwise.
        result = _apply_audit_confidence_cap(result, _area_data_for_cap)

        # Store ceiling to DB so dashboard can show yield/cashflow
        # without requiring Financial Model to be visited first
        if deal_id and result and supabase:
            try:
                _cr = result.get("ceiling_range") or {}
                _low  = _cr.get("low")
                _high = _cr.get("high")
                _mid  = round((_low + _high) / 2) if _low and _high else None
                if _mid and _mid > 5000:
                    # Update financials_json.ceiling_snapshot for dashboard reads
                    _existing_fin = merged or {}
                    _monthly_rent = _existing_fin.get("monthly_rent")
                    _gy = round((_monthly_rent * 12 / _mid) * 100, 2) if _monthly_rent and _mid else None
                    supabase.table("deals").update({
                        "bid_ceiling": _mid,
                        "updated_at": now_iso(),
                    }).eq("id", deal_id).eq("user_id", request.user_id).execute()
                    print(f"[ceiling] Stored bid_ceiling £{_mid:,} for {deal_id}")
            except Exception as _se:
                print(f"[ceiling] Store to DB failed: {_se}")

        return jsonify({"ok": True, "ceiling": result}), 200

    except Exception as e:
        app.logger.exception("[ceiling] /api/ceiling failed")
        return jsonify({"error": str(e)}), 500

@app.route("/api/deals/<deal_id>/area", methods=["GET"])
@require_auth
def get_area(deal_id: str):
    """Retrieve saved area intelligence for a deal."""
    if not supabase:
        return jsonify({"error": "Database unavailable"}), 503
    try:
        result = supabase.table("deals") \
            .select("area_json, postcode, address") \
            .eq("id", deal_id).eq("user_id", request.user_id).single().execute()
        if not result.data:
            return jsonify({"error": "Deal not found"}), 404
        area = result.data.get("area_json")
        return jsonify({
            "ok":       True,
            "area":     area,
            "postcode": result.data.get("postcode") or "",
            "has_data": bool(area),
        }), 200
    except Exception as e:
        app.logger.exception("get_area failed")
        return jsonify({"error": str(e)}), 500


def _recompute_deal_ceiling(deal_id: str, area_data: dict):
    """
    D1 — recompute summary_json.ceiling from area_json comps and persist it.

    The ceiling is normally computed during /api/analyze, which runs before
    area_json exists; it therefore persists no_base_valuation and is never
    refreshed. This helper recomputes the ceiling once area_json.housing
    comps are available and re-persists ONLY the summary_json.ceiling key.

    `area_data` is the area_json dict (freshly fetched or cached). Returns the
    new ceiling dict, or None if it could not be computed / was not persisted.
    Safe to call from any flow; a no-op when there are no comps. No formula
    change — this calls the same _calc_ceiling used by /api/analyze.
    """
    if not (_ceiling_engine_available and _calc_ceiling and supabase):
        return None
    try:
        _housing = (area_data or {}).get("housing") or {}
        _comps = _housing.get("soldComps") or _housing.get("value") or []
        _comp_prices = [c.get("price") for c in _comps
                        if isinstance(c, dict) and c.get("price")]
        if not _comp_prices:
            return None  # nothing to recompute from — leave summary_json untouched

        # Fresh read so the field-merge and optimistic lock use current state.
        _row = supabase.table("deals").select(
            "summary_json, financials_json, deal_type, updated_at"
        ).eq("id", deal_id).single().execute()
        _d = _row.data or {}
        _summary = _d.get("summary_json")
        if not isinstance(_summary, dict):
            return None

        _fins = _d.get("financials_json") or {}
        _fins_inputs = dict(_fins.get("inputs") or _fins or {})
        if not _fins_inputs.get("comps_avg_value"):
            _fins_inputs["comps_avg_value"] = round(
                sum(_comp_prices) / len(_comp_prices)
            )

        _strategy_map = {
            "hmo": "HMO", "btl": "BTL", "flip": "Flip",
            "brrr": "BRRR", "sa": "Serviced Accommodation",
            "serviced": "Serviced Accommodation",
        }
        _strategy = (
            _fins_inputs.get("strategy")
            or _d.get("deal_type")
            or (_summary.get("property") or {}).get("type")
            or "BTL"
        )
        _strategy = _strategy_map.get(str(_strategy).lower(), str(_strategy))

        _ceiling = _calc_ceiling(
            legal_flags=_summary.get("flags") or [],
            financial_inputs=_fins_inputs,
            base_valuation=None,
            strategy=_strategy,
        )

        # T-4 (centralised under D4) — same cap helper as /api/ceiling.
        _apply_audit_confidence_cap(_ceiling, area_data)

        # Merge: replace ONLY the ceiling key, preserve every other field.
        _new_summary = dict(_summary)
        _new_summary["ceiling"] = _ceiling

        # Optimistic-lock write — reject if the row changed since our read.
        _res = supabase.table("deals").update({
            "summary_json": _new_summary,
            "updated_at":   now_iso(),
        }).eq("id", deal_id).eq("updated_at", _d.get("updated_at")).execute()

        if not _res.data:
            print(f"[ceiling-recompute STALE_WRITE_REJECTED] {deal_id} — newer update existed")
            return None
        print(f"[ceiling-recompute OK] {deal_id} "
              f"base_method={_ceiling.get('base_method')} "
              f"range={_ceiling.get('ceiling_range')}")
        return _ceiling
    except Exception as _e:
        print(f"[ceiling-recompute ERROR] {deal_id}: {_e}")
        return None


@app.route("/api/deals/<deal_id>/area", methods=["POST"])
@require_auth
def save_area(deal_id: str):
    """
    Fetch & persist area intelligence for a deal's postcode.
    Body (optional): { postcode, forceRefresh }

    Architecture: returns immediately with cached data if available.
    If no cache (or forceRefresh), fires a background thread to fetch all
    external APIs and writes result to area_json. Frontend polls GET endpoint.
    This avoids Render's 60s load-balancer timeout killing sync workers.
    """
    if not supabase:
        return jsonify({"error": "Database unavailable"}), 503

    try:
        deal = supabase.table("deals") \
            .select("id, postcode, area_json, summary_json") \
            .eq("id", deal_id).eq("user_id", request.user_id).single().execute()
        if not deal.data:
            return jsonify({"error": "Deal not found"}), 404
    except Exception:
        return jsonify({"error": "Deal not found"}), 404

    body          = request.get_json(silent=True) or {}
    postcode      = normalize_postcode(body.get("postcode") or deal.data.get("postcode") or "")
    force_refresh = bool(body.get("forceRefresh") or body.get("force_refresh"))

    # Return cached data immediately if available and not force-refreshing
    if deal.data.get("area_json") and not force_refresh:
        cached = deal.data["area_json"]
        # Don't return error-marker cache as valid data
        if cached.get("fetch_status") != "error":
            # D1 — heal a stale/missing summary_json.ceiling when the cached
            # area_json already carries comps. Backgrounded so this cache
            # response still returns immediately.
            _sj_cached   = deal.data.get("summary_json") or {}
            _ceil_cached = _sj_cached.get("ceiling")
            _ceil_stale  = (
                not _ceil_cached
                or not isinstance(_ceil_cached, dict)
                or _ceil_cached.get("error") == "no_base_valuation"
                or (_ceil_cached.get("ceiling_range") or {}).get("low") is None
            )
            _housing_cached = (cached or {}).get("housing") or {}
            _has_comps_cached = bool(
                _housing_cached.get("soldComps") or _housing_cached.get("value")
            )
            if _ceil_stale and _has_comps_cached:
                import threading as _tc
                _tc.Thread(
                    target=_recompute_deal_ceiling,
                    args=(deal_id, cached),
                    daemon=True,
                ).start()
            # If cached data is missing inference, patch it in a background thread
            if not cached.get("inference") and cached.get("postcode"):
                _cached_ref  = cached
                _deal_id_ref = deal_id
                _pc_ref      = cached["postcode"]
                def _patch_inference():
                    # INVARIANT: must not overwrite newer area_json writes (Invariant 3)
                    # Use optimistic lock: only write if updated_at matches snapshot
                    try:
                        _snap = supabase.table("deals").select("updated_at").eq("id", _deal_id_ref).single().execute()
                        _snap_ts = (_snap.data or {}).get("updated_at")
                    except Exception as _se:
                        print(f"[_patch_inference] Snapshot read failed for {_deal_id_ref}: {_se}")
                        return
                    try:
                        inference_result = build_area_inference(_cached_ref, _pc_ref)
                        _cached_ref.update(inference_result)
                        try:
                            _pct = (
                                (_cached_ref.get("inference") or {})
                                .get("benchmarks", {})
                                .get("census", {})
                                .get("private_rent_pct")
                            )
                            if _pct is not None:
                                _cached_ref.setdefault("census", {})["private_rent_pct"] = _pct
                        except Exception:
                            pass
                        # Conditional write — reject if superseded
                        _result = supabase.table("deals").update({
                            "area_json":  _cached_ref,
                            "updated_at": now_iso(),
                        }).eq("id", _deal_id_ref).eq("updated_at", _snap_ts).execute()
                        if _result.data:
                            print(f"[_patch_inference OK] {_deal_id_ref}")
                        else:
                            print(f"[_patch_inference STALE_WRITE_REJECTED] {_deal_id_ref} — newer update existed")
                    except Exception as _e:
                        print(f"[_patch_inference ERROR] {_deal_id_ref}: {_e}")
                import threading as _ti
                _ti.Thread(target=_patch_inference, daemon=True).start()
            return jsonify({
                "ok":       True,
                "area":     cached,
                "postcode": postcode,
                "cached":   True,
                "fetching": False,
            }), 200

    if not postcode:
        return jsonify({"error": "postcode is required (set on deal or pass in body)"}), 400

    # Capture for background thread — Flask request context does not survive threads
    _deal_id  = deal_id
    _postcode = postcode
    # Extract property type from summary_json for matched comps filtering
    _summary     = deal.data.get("summary_json") or {}
    _prop        = _summary.get("property") or {}
    _prop_type   = str(_prop.get("type") or "").strip().upper()
    # Map common labels to Land Registry codes
    _pt_map = {
        "DETACHED": "D", "SEMI-DETACHED": "S", "SEMI DETACHED": "S",
        "TERRACED": "T", "FLAT": "F", "MAISONETTE": "F", "APARTMENT": "F",
        "D": "D", "S": "S", "T": "T", "F": "F"
    }
    _prop_type_code = _pt_map.get(_prop_type) or None
    _raw_gp = _prop.get("guide_price_pence") or _prop.get("guide_price") or 0
    try:
        _gp_num = float(_raw_gp)
        _guide_price_gbp = _gp_num / 100 if _gp_num > 1_000_000 else _gp_num
        if _guide_price_gbp < 5000 or _guide_price_gbp > 100_000_000:
            _guide_price_gbp = None
    except Exception:
        _guide_price_gbp = None

    def _fetch_and_store():
        try:
            lsoa_gss, lsoa_meta = resolve_lsoa_gss_from_postcode(_postcode)
            lat       = safe_float((lsoa_meta or {}).get("lat"))
            lng       = safe_float((lsoa_meta or {}).get("lng"))
            area_code = str((lsoa_meta or {}).get("area_code") or "").strip()

            if lat is None or lng is None:
                lat, lng, _ = nspl_lookup_latlng(_postcode)
            if lat is None or lng is None:
                lat, lng, _ = geocode_postcode(_postcode)

            area_data = {
                "postcode":     _postcode,
                "lsoa_gss":     lsoa_gss,
                "lat":          lat,
                "lng":          lng,
                "area_code":    area_code,
                "housing":      get_housing_data(_postcode, property_type=_prop_type_code, guide_price=_guide_price_gbp, subject_tenure_hint=_prop.get("tenure")),
                "crime":        get_crime_data(lat, lng),
                "transport":    get_transport_data(lat, lng),
                "amenities":    get_amenities_data(lat, lng),
                "schools":      get_schools_data(_postcode),
                "broadband":    get_broadband_data(_postcode),
                "gp":           get_gp_data(_postcode),
                "flood":        get_flood_risk(lat, lng, _postcode),
                "epc":          get_epc_data(_postcode),
                "planning":     get_planning_data(lat, lng, _postcode),
                "trends":       build_trends_from_uk_hpi(_postcode, area_code, 24),
                "fetched_at":   now_iso(),
                "fetch_status": "complete",
            }

            # ── INFERENCE ENGINE ─────────────────────────────────
            inference_result = build_area_inference(area_data, _postcode)
            area_data.update(inference_result)

            # Write census.private_rent_pct to TOP-LEVEL area_json.census so
            # frontend can read it at area_json.census.private_rent_pct.
            # (build_area_inference buries it at inference.benchmarks.census)
            try:
                _pct = (
                    (area_data.get("inference") or {})
                    .get("benchmarks", {})
                    .get("census", {})
                    .get("private_rent_pct")
                )
                if _pct is not None:
                    area_data.setdefault("census", {})["private_rent_pct"] = _pct
            except Exception:
                pass

            supabase.table("deals").update({
                "area_json":  area_data,
                "updated_at": now_iso(),
            }).eq("id", _deal_id).execute()

            # D1 — area_json now carries fresh comps; recompute & persist the
            # ceiling that /api/analyze could not compute before area existed.
            _recompute_deal_ceiling(_deal_id, area_data)

            print(f"✅ Area data fetched and stored for deal {_deal_id} ({_postcode})")

        except Exception as exc:
            print(f"❌ Background area fetch failed for {_deal_id}: {exc}")
            try:
                supabase.table("deals").update({
                    "area_json": {
                        "postcode":     _postcode,
                        "fetch_status": "error",
                        "fetch_error":  str(exc),
                        "fetched_at":   now_iso(),
                    },
                    "updated_at": now_iso(),
                }).eq("id", _deal_id).execute()
            except Exception:
                pass

    # Mark as fetching in DB immediately — polls can detect live fetch vs dead thread
    try:
        supabase.table("deals").update({
            "area_json":  {
                "postcode":     postcode,
                "fetch_status": "fetching",
                "fetched_at":   now_iso(),
            },
            "updated_at": now_iso(),
        }).eq("id", _deal_id).execute()
    except Exception:
        pass  # Non-fatal — background thread will overwrite on success

    import threading as _t
    _t.Thread(target=_fetch_and_store, daemon=True).start()

    # Return 202 immediately — frontend polls GET /api/deals/:id/area
    return jsonify({
        "ok":       True,
        "area":     None,
        "postcode": postcode,
        "cached":   False,
        "fetching": True,
        "message":  "Area data is being fetched in the background.",
    }), 202


# ── AUCTION BRIEF ─────────────────────────────────────────────

AUCTION_BRIEF_SYSTEM = """You are a UK property auction analyst. Generate a concise investor auction brief from the deal data provided.

Return ONLY valid JSON — no prose, no markdown fences.

{
  "headline": "string — one compelling line summarising the deal",
  "property_snapshot": {
    "address": "string or null",
    "type": "string or null",
    "tenure": "string or null",
    "guide_price_display": "string — e.g. £95,000",
    "lot_number": "string or null"
  },
  "deal_verdict": "string — 2-3 sentences, factual, no recommendation. State what documents show.",
  "top_risks": [
    { "title": "string", "detail": "string — one sentence" }
  ],
  "top_opportunities": [
    { "title": "string", "detail": "string — one sentence" }
  ],
  "financials_snapshot": {
    "purchase_price_display": "string or null",
    "gross_yield_display": "string or null",
    "net_cashflow_pm_display": "string or null",
    "total_invested_display": "string or null"
  },
  "key_legal_flags": [
    { "severity": "critical|high|missing", "title": "string", "summation": "string" }
  ],
  "solicitor_actions": ["string"],
  "auction_checklist": [
    { "item": "string", "status": "done|pending|unknown" }
  ]
}

top_risks and top_opportunities: max 3 each.
key_legal_flags: max 5, critical/high only.
solicitor_actions: max 5 specific actions from the flag data.
auction_checklist: standard pre-auction items with status inferred from data."""


@app.route("/api/deals/<deal_id>/auction-brief", methods=["GET"])
@require_auth
def get_auction_brief(deal_id: str):
    """
    Generate auction brief for a deal using LLM + stored deal data.
    Combines summary_json + analysis_json + financials_json.
    ?force=1 to regenerate (otherwise generates fresh each call — no caching yet).
    """
    if not supabase:
        return jsonify({"error": "Database unavailable"}), 503
    try:
        deal = supabase.table("deals") \
            .select("id, deal_name, title, address, postcode, guide_price, "
                    "auction_date, deal_type, deal_score, status, "
                    "summary_json, analysis_json, financials_json") \
            .eq("id", deal_id).eq("user_id", request.user_id).single().execute()
        if not deal.data:
            return jsonify({"error": "Deal not found"}), 404
    except Exception:
        return jsonify({"error": "Deal not found"}), 404

    d          = deal.data
    summary    = d.get("summary_json") or {}
    analysis   = d.get("analysis_json") or {}
    financials = d.get("financials_json") or {}
    prop       = summary.get("property") or {}
    fin_ret    = (financials.get("returns") or {})
    fin_acq    = (financials.get("acquisition") or {})

    def _fmt_gbp(v: Any) -> Optional[str]:
        f = safe_float(v)
        return f"£{f:,.0f}" if f is not None else None

    def _fmt_pct(v: Any) -> Optional[str]:
        f = safe_float(v)
        return f"{f:.1f}%" if f is not None else None

    context = {
        "deal_name":     d.get("deal_name") or d.get("title", ""),
        "deal_score":    d.get("deal_score"),
        "address":       d.get("address") or prop.get("address") or "",
        "postcode":      d.get("postcode") or prop.get("postcode") or "",
        "guide_price":   d.get("guide_price"),
        "auction_date":  d.get("auction_date"),
        "deal_type":     d.get("deal_type") or prop.get("type", ""),
        "property":      prop,
        "completion_terms": summary.get("completion_terms") or {},
        "flags":         (summary.get("flags") or [])[:10],
        "flag_counts":   summary.get("flag_counts") or {},
        "viability_statement": summary.get("viability_statement") or "",
        "solicitor_questions": summary.get("solicitor_questions") or [],
        "financials": {
            "purchase_price":  safe_float((financials.get("inputs") or {}).get("purchase_price")),
            "gross_yield_pct": fin_ret.get("gross_yield_pct"),
            "net_yield_pct":   fin_ret.get("net_yield_pct"),
            "net_cashflow_pm": fin_ret.get("net_cashflow_pm"),
            "total_invested":  fin_acq.get("total_invested"),
        },
        "analysis_flags":  (analysis.get("flags") or [])[:10],
        "jis_findings":    (analysis.get("jis_findings") or [])[:5],
    }

    try:
        brief = _llm_json_anthropic(
            system=AUCTION_BRIEF_SYSTEM,
            prompt=f"Generate auction brief from this deal data:\n\n{json.dumps(context, indent=2)}",
            temperature=0.2,
        )
    except Exception as e:
        app.logger.exception("auction_brief LLM failed")
        return jsonify({"error": f"Brief generation failed: {e}"}), 500

    # Fill display values if LLM left them blank
    try:
        pp = safe_float((financials.get("inputs") or {}).get("purchase_price")) or d.get("guide_price")
        fs = brief.setdefault("financials_snapshot", {})
        if not fs.get("purchase_price_display"):
            fs["purchase_price_display"] = _fmt_gbp(pp)
        if not fs.get("gross_yield_display"):
            fs["gross_yield_display"] = _fmt_pct(fin_ret.get("gross_yield_pct"))
        if not fs.get("net_cashflow_pm_display"):
            ncf = fin_ret.get("net_cashflow_pm")
            if ncf is not None:
                fs["net_cashflow_pm_display"] = f"£{ncf:,.0f}/mo"
        if not fs.get("total_invested_display"):
            fs["total_invested_display"] = _fmt_gbp(fin_acq.get("total_invested"))
    except Exception:
        pass

    brief["brief_generated_at"] = now_iso()
    brief["deal_id"]             = deal_id
    return jsonify(brief), 200



# ── SSE STREAMING SUMMARISE ─────────────────────────────────
# Streams real-time progress events as the LLM pipeline runs.
# Frontend connects with EventSource — each event is a JSON line.
# Event types: progress | finding | complete | error | limit
#
# Usage: GET /api/deals/<deal_id>/summarise/stream
# Auth:  Bearer token in Authorization header (EventSource can't set headers
#        natively, so we accept token as a query param too: ?token=<jwt>)

@app.route("/api/deals/<deal_id>/summarise/stream", methods=["GET"])
def summarise_stream(deal_id: str):
    """SSE endpoint — streams summary pipeline progress to the frontend."""
    import json as _json
    from flask import Response, stream_with_context

    # Auth — accept token from header OR query param (EventSource limitation)
    auth_header = request.headers.get("Authorization", "")
    token = auth_header[7:] if auth_header.startswith("Bearer ") else request.args.get("token", "")
    user_id = None
    if token and SUPABASE_JWT_SECRET:
        try:
            payload = pyjwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], audience="authenticated")
            user_id = payload.get("sub")
        except Exception:
            pass
    if not user_id:
        def err():
            yield "data: " + _json.dumps({"type": "error", "msg": "Unauthorised"}) + "\n\n"
        return Response(stream_with_context(err()), mimetype="text/event-stream")

    def generate():
        def ev(type_, **kwargs):
            return "data: " + _json.dumps({"type": type_, **kwargs}) + "\n\n"

        try:
            if not supabase:
                yield ev("error", msg="Database unavailable")
                return

            # ── Check deal ownership ──
            try:
                deal = supabase.table("deals").select("id, deal_name, summary_json")                     .eq("id", deal_id).eq("user_id", user_id).single().execute()
                if not deal.data:
                    yield ev("error", msg="Deal not found")
                    return
            except Exception:
                yield ev("error", msg="Deal not found")
                return

            # ── Check usage ──
            try:
                profile = supabase.table("profiles")                     .select("plan, summaries_used, usage_reset_date")                     .eq("id", user_id).single().execute()
                if profile.data and not DEV_BYPASS_LIMITS:
                    p = profile.data
                    plan = p.get("plan", "starter")
                    used = p.get("summaries_used", 0)
                    limits = {"free": 1, "starter": 5, "professional": 20, "enterprise": 999}
                    limit = limits.get(plan, 1)
                    if used >= limit:
                        yield ev("limit", used=used, limit=limit, plan=plan)
                        return
            except Exception as e:
                app.logger.warning(f"Usage check failed in stream: {e}")

            # ── Return cached summary if available ──
            if deal.data.get("summary_json"):
                yield ev("progress", msg="Loading existing analysis…", step=1, total=6)
                import time as _time
                _time.sleep(0.3)
                summary = deal.data["summary_json"]
                yield ev("complete", summary=summary)
                return

            # ── Fetch documents ──
            yield ev("progress", msg="Connecting to document store…", step=1, total=6)
            try:
                docs_result = supabase.table("documents")                     .select("doc_type, file_name, extracted_text, page_count, extraction_status")                     .eq("deal_id", deal_id).eq("user_id", user_id).execute()
                documents = docs_result.data or []
            except Exception as e:
                yield ev("error", msg=f"Could not fetch documents: {e}")
                return

            if not documents:
                yield ev("error", msg="No documents found for this deal")
                return

            # ── Emit document inventory ──
            total_pages = sum(d.get("page_count") or 0 for d in documents)
            yield ev("progress",
                     msg=f"Loaded {len(documents)} documents — {total_pages} pages",
                     step=2, total=6, doc_count=len(documents), page_count=total_pages)

            # Emit each doc type found
            DOC_LABELS = {
                "special_conditions": "Special Conditions of Sale",
                "addendum": "Addendum",
                "title_register": "Title Register",
                "title_plan": "Title Plan",
                "local_auth_search": "Local Authority Search",
                "lease": "Lease",
                "epc": "EPC Certificate",
                "legal_pack": "Legal Pack",
                "environmental": "Environmental Search",
                "freehold": "Freehold Title",
                "deed": "Transfer Deed",
                "tenancy_ast": "Tenancy Agreement",
                "survey": "Survey Report",
                "auction_tcs": "Auction T&Cs",
                "unknown": "Document",
            }
            for doc in documents:
                label = DOC_LABELS.get(doc.get("doc_type", "unknown"), "Document")
                pages = doc.get("page_count") or 0
                yield ev("document", doc_type=doc.get("doc_type"), label=label, pages=pages,
                         filename=doc.get("file_name", ""))

            # ── Stage 1: LLM extraction ──
            yield ev("progress", msg="Reading clauses and identifying legal obligations…", step=3, total=6)

            try:
                import sys as _sys
                import os as _os
                import threading as _threading
                import time as _time
                _sys.path.insert(0, _os.path.dirname(__file__))
                try:
                    from services.legal_analysis import run_document_summary, _build_combined_text
                except ImportError:
                    from legal_analysis import run_document_summary, _build_combined_text

                # Build combined text from all documents
                # Smart prioritised text — legal docs first, searches capped
                PRIORITY = ['special_conditions','addendum','title_register','lease',
                            'title_plan','deed','freehold','tenancy_ast',
                            'local_auth_search','environmental','epc','survey','auction_tcs','unknown']
                docs_sorted = sorted(documents,
                    key=lambda d: PRIORITY.index(d.get('doc_type','unknown'))
                                  if d.get('doc_type','unknown') in PRIORITY else 99)
                parts = []
                total = 0
                HARD_CAP = 55000
                PER_DOC_CAP = 8000
                for doc in docs_sorted:
                    text = (doc.get('extracted_text') or '').strip()
                    if not text: continue
                    label = f"=== {doc.get('doc_type','unknown').upper()}: {doc.get('file_name','')} ===\n"
                    capped = text[:PER_DOC_CAP] + ('\n[...truncated...]' if len(text) > PER_DOC_CAP else '')
                    chunk = label + capped + '\n\n'
                    if total + len(chunk) > HARD_CAP:
                        remaining = HARD_CAP - total - len(label) - 30
                        if remaining > 200:
                            parts.append(label + text[:remaining] + '\n[...truncated...]\n\n')
                        break
                    parts.append(chunk)
                    total += len(chunk)
                truncated = ''.join(parts)
                char_count = len(truncated)
                yield ev("progress", msg=f"Prepared {char_count:,} characters from {len(documents)} documents", step=3, total=4)

                # ── Single LLM call — extract + classify + score in one pass ──
                yield ev("progress", msg="Reading clauses and identifying risks…", step=4, total=4)

                COMBINED_SYSTEM = """You are a UK auction property legal analyst. Your job is to FIND EVERY RISK in this legal pack. Be aggressive and thorough — an investor's money is at stake.

Return ONLY valid JSON. No prose, no markdown fences. Exactly this structure (flags MUST come first):
{
  "flags": [
    {
      "severity": "critical|high|missing|note",
      "title": "specific risk title — max 10 words",
      "summation": "one sentence: what this means for the investor",
      "evidence": "verbatim quote from document — max 30 words",
      "implication": "financial or legal impact — max 20 words",
      "action": "what investor must do — max 15 words",
      "source_document": "document filename",
      "source_clause": "clause number or null",
      "source_page": null,
      "legal_risk_weight": 7
    }
  ],
  "flag_counts": {"critical": 0, "high": 0, "missing": 0, "note": 0},
  "deal_score": 0,
  "viability_statement": "2-3 sentences: investor verdict",
  "property": {"address": "full address", "postcode": "postcode", "lot_number": "lot", "type": "BTL/HMO/Commercial/etc", "tenure": "Freehold/Leasehold", "lease_years": null, "guide_price_pence": null},
  "completion_terms": {"deposit_pct": null, "deposit_refundable": null, "completion_days": null, "completion_type": "working", "buyers_premium_pct": null, "vacant_possession": null},
  "pack_completeness": {"completeness_pct": 0, "present_count": 0, "total": 13},
  "documents_processed": 0
}

FLAG EXTRACTION RULES — YOU MUST FOLLOW ALL OF THEM:
1. NEVER return an empty flags array. Every legal pack has risks. If a pack seems clean, flag what is MISSING.
2. Flag EVERY one of these if present: restrictive covenants, chancel repair, mining/subsidence, flood risk, Japanese knotweed, Article 4 directions, HMO licensing, short lease (<85 years), ground rent escalation, service charge >£2500/yr, absent landlord, possessory title, missing searches, auction clauses (non-refundable deposit, 28-day completion, buyers premium), tenancy issues (sitting tenant, AST expiry, rent arrears), planning enforcement notices.
3. Flag MISSING documents: if Special Conditions, Title Register, Local Search, Environmental Search, EPC are absent — each is a MISSING flag.
4. Minimum flags: generate at least 1 flag per document that contains a clause. Aim for 10-20 flags total.
5. Scoring: Start at 100. Deduct critical=-12, high=-6, missing=-4, note=-1.
6. Keep evidence quotes SHORT (max 30 words) — critical for fitting all flags within token budget.
7. The flags array MUST be complete before flag_counts. Do not close the JSON until all flags are written."""


                _res = {}
                def _run_analysis():
                    try:
                        _res["data"] = _llm_json_anthropic(
                            system=COMBINED_SYSTEM,
                            prompt=f"Analyse this auction legal pack and return the complete JSON summary:\n\n{truncated}",
                            temperature=0.1,
                        )
                    except Exception as e:
                        _res["error"] = str(e)

                t = _threading.Thread(target=_run_analysis, daemon=True)
                t.start()
                elapsed = 0
                _feed_items = [
                    (15, "Reading Special Conditions and addenda…"),
                    (25, "Checking title register for encumbrances…"),
                    (35, "Identifying non-standard contractual clauses…"),
                    (45, "Extracting verbatim evidence for each finding…"),
                    (55, "Cross-referencing document types…"),
                    (65, "Calculating deal score…"),
                    (75, "Building risk report…"),
                    (85, "Finalising analysis…"),
                ]
                feed_idx = 0
                while t.is_alive():
                    _time.sleep(5)
                    elapsed += 5
                    # Emit realistic feed items timed to elapsed seconds
                    while feed_idx < len(_feed_items) and elapsed >= _feed_items[feed_idx][0]:
                        yield ev("progress", msg=_feed_items[feed_idx][1], step=4, total=4)
                        feed_idx += 1
                    # Heartbeat to keep connection alive
                    yield ev("heartbeat", elapsed=elapsed, msg=f"Analysing… ({elapsed}s)")
                t.join()

                if "error" in _res:
                    raise Exception(_res["error"])

                llm_result = _res.get("data", {})

                # Stream individual findings as teasers
                for f in (llm_result.get("flags") or [])[:8]:
                    sev = (f.get("severity") or "note").lower()
                    title = f.get("title", "")
                    if title:
                        yield ev("finding", severity=sev, claim=title)

                yield ev("progress", msg="Calculating deal score and finalising report…", step=4, total=4)

                # Use LLM result directly as summary — no second pipeline call
                summary = llm_result
                # ── Schema enforcement: guarantee frontend contract ──
                if not isinstance(summary.get("flags"), list):
                    summary["flags"] = []
                # ALWAYS recompute flag_counts from actual flags array
                summary["flag_counts"] = {
                    "critical": sum(1 for f in summary["flags"] if (f.get("severity") or "").lower() == "critical"),
                    "high":     sum(1 for f in summary["flags"] if (f.get("severity") or "").lower() == "high"),
                    "missing":  sum(1 for f in summary["flags"] if (f.get("severity") or "").lower() == "missing"),
                    "note":     sum(1 for f in summary["flags"] if (f.get("severity") or "").lower() == "note"),
                }
                if summary.get("deal_score") is None:
                    summary["deal_score"] = 50
                if not isinstance(summary.get("property"), dict):
                    summary["property"] = {}
                if not isinstance(summary.get("completion_terms"), dict):
                    summary["completion_terms"] = {}
                if not summary.get("documents_processed"):
                    summary["documents_processed"] = len(documents)

            except Exception as e:
                app.logger.exception("SSE pipeline failed")
                yield ev("error", msg=f"Analysis failed: {str(e)}")
                return

            # ── Persist ──
            prop = summary.get("property") or {}
            try:
                supabase.table("deals").update({
                    "summary_json": summary,
                    "deal_score":   summary.get("deal_score"),
                    "updated_at":   now_iso(),
                    "address":      prop.get("address"),
                    "postcode":     prop.get("postcode") or None,
                    "deal_type":    prop.get("type"),
                }).eq("id", deal_id).execute()
            except Exception as e:
                app.logger.warning(f"Could not persist summary: {e}")

            # Record usage
            try:
                profile_data = profile.data if profile and profile.data else {}
                supabase.table("profiles").update({
                    "summaries_used": (profile_data.get("summaries_used", 0) + 1)
                }).eq("id", user_id).execute()
                supabase.table("usage_events").insert({
                    "user_id": user_id, "event_type": "summary",
                    "deal_id": deal_id, "amount_pence": 0,
                }).execute()
            except Exception as e:
                app.logger.warning(f"Usage recording failed: {e}")

            yield ev("complete", summary=summary)

        except Exception as e:
            app.logger.exception("SSE outer error")
            yield "data: " + _json.dumps({"type": "error", "msg": str(e)}) + "\n\n"

    response = Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": ",".join(_CORS_ORIGINS),
            "Access-Control-Allow-Headers": "Authorization, Content-Type",
        }
    )
    return response

@app.route("/api/test-llm", methods=["GET"])
def test_llm():
    """Diagnostic endpoint — tests Anthropic client directly. No auth required.
    Returns the exact error if something is wrong with the key or model."""
    result = {"ok": False, "model": "claude-sonnet-4-6", "error": None, "detail": None}
    try:
        client = _get_anthropic_client()
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=50,
            system="Reply with valid JSON only.",
            messages=[{"role": "user", "content": "Return: {\"ok\": true}"}],
        )
        text = msg.content[0].text if msg.content else ""
        result["ok"] = True
        result["response"] = text
        result["stop_reason"] = msg.stop_reason
    except Exception as e:
        result["error"]        = type(e).__name__
        result["detail"]       = str(e) or repr(e)
        result["status_code"]  = getattr(e, 'status_code', None)
        result["body"]         = str(getattr(e, 'body', None) or '')
        result["request_id"]   = getattr(e, 'request_id', None)
    return jsonify(result)


@app.route("/api/client-config", methods=["GET", "OPTIONS"])
def client_config():
    """Public, unauthenticated client runtime config.

    Returns browser-safe public config values sourced from server environment.
    Currently exposes MAPBOX_PUBLIC_TOKEN (a URL-restricted Mapbox public token,
    prefix 'pk.'). Returns null when the env var is unset so the frontend can
    surface a clear configuration error rather than guessing.
    """
    if request.method == "OPTIONS":
        return ("", 204)
    return jsonify({
        "mapbox_token": os.environ.get("MAPBOX_PUBLIC_TOKEN")
    }), 200


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Legal Smegal API Final",
        "status": "active",
        "supabaseEnabled": bool(supabase),
        "routes": {
            "POST /market-insights": "{ 'postcode': 'EC3A 5DE', 'forceRefresh': true }  // optional: lat/lng",
            "POST /api/ai-explain": "{ 'prompt': '...' } — Bearer token required — Ask AI proxy for flag workbench",
            "POST /adapters/geocode/batch": "{ 'queries': ['PARROT ROW, ABERTILLERY, NP13 3AH', ...] }",
            "GET /adapters/geo?postcode=EC1A%201BB": "debug postcode -> LSOA(GSS) + coords",
            "GET /adapters/nomis?table=ts003&geography=2092957699": "Nomis TS003 (requires numeric geography id)",
            "GET /adapters/nomis?table=ts054&geography=2092957699": "Nomis TS054 (requires env dims/cats + numeric geography)",
            "GET /adapters/nomis?table=ts044&geography=2092957699": "Nomis TS044 (requires env dims/cats + numeric geography)",
            "GET /adapters/schools?postcode=EC3A%205DE": "debug schools adapter",
            "GET /adapters/broadband?postcode=EC3A%205DE": "debug broadband adapter",
            "GET /adapters/housing/comps?postcode=EC3A%205DE&radius_miles=3&limit=20": "debug housing sold comps (RPC)",
        },
        "envHints": {
            "ANTHROPIC_API_KEY": "REQUIRED for /api/ai-explain (Ask AI in flag workbench)",
            "APP_CACHE_BUSTER": "change this value to force refresh of cached /market-insights payloads",
            "MARKET_CONTRACT_MODE": "set to 1 to force deterministic UI-safe payload (bypasses cache/providers)",
            "SUPABASE_URL": "required for supabase providers",
            "SUPABASE_SERVICE_ROLE_KEY": "preferred (server-only). SUPABASE_KEY also supported as fallback.",
            "GOOGLE_MAPS_API_KEY": "required for /adapters/geocode/batch AND housing lat/lng enrichment",
            "GEOCODE_CACHE_TABLE": "defaults to geocode_cache",
            "GEOCODE_BATCH_LIMIT": "defaults to 10",
            "HOUSING_ENRICH_LATLNG": "defaults to 1; ensures sold comps include lat/lng for map pins",
            "HOUSING_ENRICH_BATCH_LIMIT": "defaults to 10; max comps to enrich per request",
            "SCHOOLS_PROVIDER": "set to 'supabase' to enable",
            "BROADBAND_PROVIDER": "set to 'supabase' to enable",
            "BROADBAND_SUPABASE_TABLE": "e.g. broadband_by_postcode",
            "NSPL": "requires view/table public.nspl_lookup(pcd_nospace, lat, lng) for postcode->coords",
            "HOUSING_PROVIDER": "set to 'supabase_rpc' to enable sold comps",
            "HOUSING_RPC_NAME": "defaults to 'housing_comps_v1'",
            "NOMIS_ENABLED": "1 to enable",
            "NOMIS_DEFAULT_GEOGRAPHY": "REQUIRED for Nomis with this dataset unless you supply a numeric geography id",
            "NOMIS_FREQ": "REQUIRED for NM_2023_1 (e.g. A)",
            "NOMIS_TS003_DIM": "defaults to c2021_hhcomp_15 (must match dataset id)",
            "NOMIS_TS054_DIM/NOMIS_TS054_CATS": "paste from TS054 copy-address URL",
            "NOMIS_TS044_DIM/NOMIS_TS044_CATS": "paste from TS044 copy-address URL",
        }
    })



@app.route("/api/auction-triangulation", methods=["GET"])
def auction_triangulation():
    guide = request.args.get("guide", type=float)
    postcode = request.args.get("postcode", type=str)

    result = supabase.rpc('get_auction_triangulation', {
        'input_guide': guide,
        'input_postcode': postcode
    }).execute()

    if hasattr(result, 'error') and result.error:
        return jsonify({"error": result.error}), 500

    return jsonify(result.data)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FORENSIC DIAGNOSTIC ENDPOINTS — TEMPORARY OBSERVABILITY LAYER
# Purpose: runtime evidence only. No business logic. No side effects. No masking.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.route("/api/diag/runtime-health", methods=["GET", "OPTIONS"])
def diag_runtime_health():
    """
    Forensic runtime health check. NO AUTH — read-only, no PII, no deal data.
    Tests every database path independently with timing. Reports exact failure text.
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200

    import socket as _socket

    out = {"runtime": {}, "connectivity": {}, "tables": {}, "rpc": {}}

    # ── RUNTIME IDENTITY ───────────────────────────────────────────────────
    try:
        out["runtime"]["hostname"] = _socket.gethostname()
    except Exception as _e:
        out["runtime"]["hostname"] = f"ERROR: {_e}"
    out["runtime"]["database_url_configured"]     = bool(DATA_DATABASE_URL)
    out["runtime"]["supabase_url_configured"]      = bool(SUPABASE_URL)
    out["runtime"]["supabase_key_configured"]      = bool(SUPABASE_KEY)
    out["runtime"]["supabase_db_url_configured"]   = bool(SUPABASE_DB_URL)
    out["runtime"]["supabase_client_initialised"]  = supabase is not None
    out["runtime"]["housing_provider"]             = HOUSING_PROVIDER
    out["runtime"]["housing_rpc_name"]             = HOUSING_RPC_NAME

    # ── HETZNER CONNECTIVITY ───────────────────────────────────────────────
    _t0 = time.time()
    try:
        with psycopg.connect(DATA_DATABASE_URL, row_factory=dict_row, connect_timeout=8) as _c:
            with _c.cursor() as _cur:
                _cur.execute("SELECT 1 AS ping")
                _cur.fetchone()
        out["connectivity"]["hetzner"] = {"ok": True, "latency_ms": round((time.time()-_t0)*1000,1), "error": None}
    except Exception as _e:
        out["connectivity"]["hetzner"] = {"ok": False, "latency_ms": round((time.time()-_t0)*1000,1), "error": str(_e)}

    # ── SUPABASE DIRECT POSTGRES (SUPABASE_DB_URL) ────────────────────────
    if SUPABASE_DB_URL:
        _t0 = time.time()
        try:
            with psycopg.connect(SUPABASE_DB_URL, row_factory=dict_row, connect_timeout=8) as _c:
                with _c.cursor() as _cur:
                    _cur.execute("SELECT 1 AS ping")
                    _cur.fetchone()
            out["connectivity"]["supabase_direct_postgres"] = {"ok": True, "latency_ms": round((time.time()-_t0)*1000,1), "error": None}
        except Exception as _e:
            out["connectivity"]["supabase_direct_postgres"] = {"ok": False, "latency_ms": round((time.time()-_t0)*1000,1), "error": str(_e)}
    else:
        out["connectivity"]["supabase_direct_postgres"] = {"ok": False, "latency_ms": None, "error": "SUPABASE_DB_URL not set"}

    # ── SUPABASE REST (PostgREST via supabase-py) ─────────────────────────
    if supabase:
        _t0 = time.time()
        try:
            supabase.table("deals").select("id").limit(1).execute()
            out["connectivity"]["supabase_rest"] = {"ok": True, "latency_ms": round((time.time()-_t0)*1000,1), "error": None}
        except Exception as _e:
            out["connectivity"]["supabase_rest"] = {"ok": False, "latency_ms": round((time.time()-_t0)*1000,1), "error": str(_e)}
    else:
        out["connectivity"]["supabase_rest"] = {"ok": False, "latency_ms": None, "error": "Supabase client not initialised"}

    # ── HETZNER TABLES ────────────────────────────────────────────────────
    for _tbl in ("price_paid_raw_2025", "epc_certificates", "nspl_postcodes"):
        try:
            _r = data_query(f"SELECT COUNT(*) AS cnt FROM public.{_tbl}")
            out["tables"][_tbl] = {"database": "hetzner", "exists": True, "row_count": int((_r[0].get("cnt") or 0)) if _r else 0, "error": None}
        except Exception as _e:
            out["tables"][_tbl] = {"database": "hetzner", "exists": False, "row_count": None, "error": str(_e)}

    # ── SUPABASE TABLES ───────────────────────────────────────────────────
    for _tbl, _sql in [
        ("uk_hpi_monthly",  "SELECT COUNT(*) AS cnt FROM public.uk_hpi_monthly"),
        ("uk_prms_monthly", "SELECT COUNT(*) AS cnt FROM public.uk_prms_monthly"),
    ]:
        try:
            _r = supabase_data_query(_sql) if supabase else []
            _cnt = int((_r[0].get("cnt") or 0)) if _r else 0
            out["tables"][_tbl] = {
                "database": "supabase",
                "exists": bool(_r),
                "row_count": _cnt,
                "error": None if _r else "Empty result — GRANT missing or table empty",
            }
        except Exception as _e:
            out["tables"][_tbl] = {"database": "supabase", "exists": False, "row_count": None, "error": str(_e)}

    # ── CRITICAL CHECK: does price_paid_raw_2025 exist on Supabase? ───────
    # If not, housing_comps_v1 will always return [].
    if supabase:
        try:
            _pp_sb = supabase.table("price_paid_raw_2025").select("paon").limit(1).execute()
            _found = len(_pp_sb.data or []) > 0
            out["tables"]["price_paid_raw_2025_on_supabase"] = {
                "database": "supabase", "sample_row_found": _found,
                "error": None,
                "diagnosis": "OK — housing_comps_v1 has data" if _found else "0 rows — housing_comps_v1 always returns []",
            }
        except Exception as _e:
            out["tables"]["price_paid_raw_2025_on_supabase"] = {
                "database": "supabase", "sample_row_found": False,
                "error": str(_e),
                "diagnosis": "Not accessible on Supabase — housing_comps_v1 has no data source",
            }

    # ── HPI ENGLAND SPOT CHECK ────────────────────────────────────────────
    try:
        _er = supabase_data_query(
            "SELECT average_price, annual_change FROM public.uk_hpi_monthly WHERE area_code = %s ORDER BY date DESC LIMIT 1",
            ("E92000001",)
        ) if supabase else []
        out["tables"]["uk_hpi_monthly_england_spot"] = {
            "area_code": "E92000001", "value": _er[0] if _er else None, "queryable": bool(_er),
        }
    except Exception as _e:
        out["tables"]["uk_hpi_monthly_england_spot"] = {"area_code": "E92000001", "value": None, "queryable": False, "error": str(_e)}

    # ── HOUSING_COMPS_V1 LIVE PROBE ───────────────────────────────────────
    if supabase:
        _t0 = time.time()
        try:
            _rpc_res = supabase.rpc(HOUSING_RPC_NAME, {"in_postcode": "DL3 0PL", "in_radius_miles": 3.0, "in_limit": 5}).execute()
            _rows = _rpc_res.data if hasattr(_rpc_res, "data") and isinstance(_rpc_res.data, list) else None
            out["rpc"][HOUSING_RPC_NAME] = {
                "test_postcode": "DL3 0PL", "latency_ms": round((time.time()-_t0)*1000,1),
                "rows_returned": len(_rows) if _rows is not None else None,
                "sample": (_rows[:2] if _rows else []),
                "diagnosis": (
                    "RPC returned rows — price_paid present on Supabase" if _rows
                    else "RPC returned 0 rows — price_paid absent on Supabase" if _rows is not None
                    else "RPC returned unexpected non-list result"
                ),
            }
        except Exception as _e:
            out["rpc"][HOUSING_RPC_NAME] = {
                "test_postcode": "DL3 0PL", "latency_ms": round((time.time()-_t0)*1000,1),
                "rows_returned": None, "sample": [], "error": str(_e),
                "diagnosis": f"RPC threw exception: {_e}",
            }
    else:
        out["rpc"][HOUSING_RPC_NAME] = {"rows_returned": None, "error": "Supabase client not initialised"}

    return jsonify({"ok": True, "diag": out, "generated_at": now_iso()}), 200


@app.route("/api/diag/deal-trace/<deal_id>", methods=["GET", "OPTIONS"])
@require_auth
def diag_deal_trace(deal_id: str):
    """
    Forensic valuation trace for a specific deal. AUTH REQUIRED.
    Dry-runs the ceiling fallback chain + probes comparables + checks HPI benchmarks.
    No mutations. Reports exact value and failure reason at every step.
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200

    if not supabase:
        return jsonify({"error": "Supabase client not initialised"}), 503

    # ── LOAD DEAL ─────────────────────────────────────────────────────────
    try:
        _dr = supabase.table("deals").select(
            "id,address,postcode,guide_price,deal_type,bid_ceiling,updated_at,"
            "financials_json,area_json,summary_json"
        ).eq("id", deal_id).eq("user_id", request.user_id).single().execute()
    except Exception as _e:
        return jsonify({"error": f"Deal load failed: {_e}"}), 500

    if not _dr.data:
        return jsonify({"error": "Deal not found"}), 404

    d        = _dr.data
    area     = d.get("area_json") or {}
    fins_inp = (d.get("financials_json") or {}).get("inputs") or (d.get("financials_json") or {})
    housing  = area.get("housing") or {}
    inf      = area.get("inference") or {}
    b_price  = (inf.get("benchmarks") or {}).get("price") or {}
    b_rental = (inf.get("benchmarks") or {}).get("rental") or {}
    summary  = d.get("summary_json") or {}

    out = {
        "deal_id": deal_id,
        "valuation_state": {
            "address":                 d.get("address"),
            "postcode":                d.get("postcode"),
            "guide_price":             d.get("guide_price"),
            "deal_type":               d.get("deal_type"),
            "stored_bid_ceiling":      d.get("bid_ceiling"),
            "area_json_present":       bool(area),
            "area_fetch_status":       area.get("fetch_status"),
            "area_fetched_at":         area.get("fetched_at"),
            "area_code":               area.get("area_code"),
            "inference_present":       bool(inf),
            "benchmarks_present":      bool(inf.get("benchmarks")),
            "financials_present":      bool(fins_inp),
            "monthly_rent_input":      fins_inp.get("monthly_rent"),
            "target_yield_input":      fins_inp.get("target_yield") or fins_inp.get("target_gross_yield"),
            "updated_at":              d.get("updated_at"),
        },
        "ceiling_trace": {},
        "comparables":   {},
        "hpi_benchmarks":{},
        "scenario_modelling": {},
        "architectural_findings": [],
    }

    # ── CEILING FALLBACK CHAIN — DRY RUN (mirrors app.py:6785-6865 exactly) ──
    _base = None

    # Step 0: soldComps
    comps = housing.get("soldComps") or housing.get("value") or []
    _cprices = []
    for _c in comps:
        for _k in ("price_normalised", "hpi_adjusted_price", "price"):
            _v = _c.get(_k)
            if _v:
                try:
                    _vi = int(float(_v))
                    if _vi > 5000:
                        _cprices.append(_vi)
                        break
                except (TypeError, ValueError):
                    pass
    if _cprices:
        _s = sorted(_cprices)
        _n = len(_s)
        _base = _s[_n//2] if _n % 2 else (_s[_n//2-1] + _s[_n//2]) // 2
    out["ceiling_trace"]["step_0_sold_comps"] = {
        "comp_count": len(comps), "usable_prices": len(_cprices), "value": _base, "fired": _base is not None,
        "failure_reason": None if _base else (
            "soldComps empty — housing_comps_v1 returned 0 rows" if not comps else "no valid prices in comps"
        ),
    }

    # Fallback 1: housing.metrics.median_price
    _f1 = None
    if _base is None:
        _mp = (housing.get("metrics") or {}).get("median_price")
        if _mp:
            try:
                _f1v = float(_mp)
                if _f1v > 5000:
                    _f1 = int(_f1v)
                    _base = _f1
            except (TypeError, ValueError):
                pass
    out["ceiling_trace"]["fallback_1_median_price"] = {
        "raw_value": (housing.get("metrics") or {}).get("median_price"),
        "value": _f1, "fired": _f1 is not None,
        "failure_reason": None if _f1 else "median_price null — derives from comps which are empty",
    }

    # Fallback 2: summary_json avg_sold_price
    _f2 = None
    if _base is None:
        _sp = (summary.get("property") or {}).get("avg_sold_price") or (summary.get("area") or {}).get("avg_sold_price")
        if _sp:
            try:
                _f2v = float(_sp)
                if _f2v > 5000:
                    _f2 = int(_f2v)
                    _base = _f2
            except (TypeError, ValueError):
                pass
    out["ceiling_trace"]["fallback_2_summary_avg_sold"] = {
        "raw_value": (summary.get("property") or {}).get("avg_sold_price"),
        "value": _f2, "fired": _f2 is not None,
        "failure_reason": None if _f2 else "avg_sold_price not in summary_json",
    }

    # Fallback 3: guide_price × 1.15
    _f3 = None
    if _base is None:
        _gp = d.get("guide_price")
        if _gp:
            try:
                _gpv = float(_gp)
                if _gpv > 5000:
                    _f3 = round(_gpv * 1.15)
                    _base = _f3
            except (TypeError, ValueError):
                pass
    out["ceiling_trace"]["fallback_3_guide_price"] = {
        "guide_price_raw": d.get("guide_price"),
        "value": _f3, "fired": _f3 is not None,
        "failure_reason": None if _f3 else "guide_price null or not extracted",
    }

    # Fallback 4: inference.benchmarks.price
    _f4, _f4_key = None, None
    if _base is None:
        for _pk in ("regional", "national", "local"):
            _pv = b_price.get(_pk)
            if _pv and isinstance(_pv, (int, float)) and float(_pv) > 5000:
                _f4 = int(float(_pv))
                _f4_key = _pk
                _base = _f4
                break
    out["ceiling_trace"]["fallback_4_hpi_benchmark"] = {
        "available": {"regional": b_price.get("regional"), "national": b_price.get("national"), "local": b_price.get("local")},
        "key_used": _f4_key, "value": _f4, "fired": _f4 is not None,
        "failure_reason": None if _f4 else (
            "no area_json inference — not fetched or pre-GRANT build" if not inf
            else "all benchmark price keys are null/zero"
        ),
    }

    # Fallback 5: live supabase_data_query
    _f5, _f5_code, _f5_err = None, None, None
    if _base is None and supabase:
        for _code in ("E92000001", "England", "United Kingdom", "K02000001"):
            try:
                _r = supabase_data_query(
                    "SELECT average_price FROM public.uk_hpi_monthly WHERE area_code = %s ORDER BY date DESC LIMIT 1",
                    (_code,)
                )
                if _r:
                    _v = safe_float(_r[0].get("average_price"))
                    if _v and _v > 5000:
                        _f5 = int(_v)
                        _f5_code = _code
                        _base = _f5
                        break
            except Exception as _fe:
                _f5_err = str(_fe)
                break
    out["ceiling_trace"]["fallback_5_hpi_live"] = {
        "area_code_used": _f5_code, "value": _f5, "fired": _f5 is not None,
        "error": _f5_err,
        "failure_reason": None if _f5 else (
            _f5_err or "supabase_data_query returned [] for all England codes — GRANT missing or table empty"
        ),
    }

    out["ceiling_trace"]["resolved"] = {
        "base_valuation": _base,
        "ceiling_will_compute": _base is not None and _base > 5000,
        "failure_reason": None if _base else "All fallbacks exhausted — no base valuation",
    }

    # ── LIVE COMPARABLES PROBE ────────────────────────────────────────────
    _pc = d.get("postcode") or ""
    if supabase and _pc:
        _t0 = time.time()
        try:
            _rr = supabase.rpc(HOUSING_RPC_NAME, {"in_postcode": _pc, "in_radius_miles": 3.0, "in_limit": 10}).execute()
            _rows = _rr.data if hasattr(_rr, "data") and isinstance(_rr.data, list) else None
            _rprices = [safe_float(r.get("price")) for r in (_rows or []) if safe_float(r.get("price")) and safe_float(r.get("price")) > 5000]  # type: ignore
            out["comparables"] = {
                "rpc": HOUSING_RPC_NAME, "postcode": _pc, "radius_miles": 3.0,
                "latency_ms": round((time.time()-_t0)*1000,1),
                "rows_returned": len(_rows) if _rows is not None else None,
                "prices_found": len(_rprices),
                "median_price": sorted(_rprices)[len(_rprices)//2] if _rprices else None,
                "avg_price": round(sum(_rprices)/len(_rprices)) if _rprices else None,
                "sample": (_rows[:3] if _rows else []),
                "diagnosis": (
                    f"{len(_rows)} rows — comparables available" if _rows
                    else "0 rows — price_paid absent on Supabase" if _rows is not None
                    else "unexpected result"
                ),
            }
        except Exception as _e:
            out["comparables"] = {"rpc": HOUSING_RPC_NAME, "postcode": _pc, "error": str(_e)}
    else:
        out["comparables"] = {"rpc": HOUSING_RPC_NAME, "error": "No postcode on deal or Supabase not available"}

    # ── HPI BENCHMARKS: stored vs live ────────────────────────────────────
    _lad = str(area.get("area_code") or "").strip()
    out["hpi_benchmarks"] = {
        "area_code": _lad,
        "stored_inference": {
            "price_regional": b_price.get("regional"),
            "price_national": b_price.get("national"),
            "price_local":    b_price.get("local"),
            "rental_regional_gbp": b_rental.get("regional_rent_gbp"),
        },
        "live_lad_query":     None,
        "live_england_query": None,
    }
    if _lad and supabase:
        try:
            _lr = supabase_data_query(
                "SELECT date, average_price, annual_change FROM public.uk_hpi_monthly WHERE area_code = %s ORDER BY date DESC LIMIT 1",
                (_lad,)
            )
            out["hpi_benchmarks"]["live_lad_query"] = _lr[0] if _lr else "empty"
        except Exception as _e:
            out["hpi_benchmarks"]["live_lad_query"] = f"ERROR: {_e}"
    if supabase:
        try:
            _er = supabase_data_query(
                "SELECT date, average_price, annual_change FROM public.uk_hpi_monthly WHERE area_code = %s ORDER BY date DESC LIMIT 1",
                ("E92000001",)
            )
            out["hpi_benchmarks"]["live_england_query"] = _er[0] if _er else "empty — GRANT missing"
        except Exception as _e:
            out["hpi_benchmarks"]["live_england_query"] = f"ERROR: {_e}"

    # ── SCENARIO MODELLING INPUTS ─────────────────────────────────────────
    _regional_rent = b_rental.get("regional_rent_gbp")
    _comp_avg = round(sum([safe_float(c.get("price")) for c in comps if safe_float(c.get("price")) and safe_float(c.get("price")) > 5000]) /  # type: ignore
                      len([c for c in comps if safe_float(c.get("price")) and safe_float(c.get("price")) > 5000])) if comps else None
    _implied_rent_from_comps = round((_comp_avg * 0.065) / 12) if _comp_avg else None
    out["scenario_modelling"] = {
        "implied_rent_from_comps":    _implied_rent_from_comps,
        "regional_rent_benchmark_gbp": _regional_rent,
        "hpi_price_available":        bool(b_price.get("regional") or b_price.get("national")),
        "scenario_would_render":      bool(_implied_rent_from_comps or _regional_rent),
        "frontend_fix_required":      True,  # b-scope bug + _impliedRent fallback — fix in outputs/
        "failure_reason": (
            None if (_implied_rent_from_comps or _regional_rent)
            else "No implied rent source: comps empty, no rental benchmark in area_json. "
                 "Fetch/re-fetch area for this deal after GRANT was applied."
        ),
    }

    # ── ARCHITECTURAL FINDINGS ─────────────────────────────────────────────
    findings = []
    if out["comparables"].get("rows_returned") == 0:
        findings.append({"severity": "CRITICAL", "component": "housing_comps_v1",
            "finding": "RPC returned 0 rows — price_paid_raw_2025 absent on Supabase",
            "impact": "Ceiling Fallbacks 0+1 permanently fail. All comp-backed valuation impossible."})
    if not b_price.get("regional") and not b_price.get("national"):
        findings.append({"severity": "HIGH", "component": "inference.benchmarks.price",
            "finding": "No regional/national HPI price in stored area_json",
            "impact": "Fallback 4 cannot fire. area_json built pre-GRANT or not yet fetched post-GRANT."})
    if not _regional_rent:
        findings.append({"severity": "HIGH", "component": "inference.benchmarks.rental",
            "finding": "No regional rent benchmark in area_json",
            "impact": "Scenario modelling has no rental input. Deploy legalsmegal-area.html fix."})
    if not area:
        findings.append({"severity": "CRITICAL", "component": "area_json",
            "finding": "area_json is NULL — not yet fetched",
            "impact": "Fallback 4 dead. No rental benchmark. Scenario blank."})
    out["architectural_findings"] = findings

    return jsonify({"ok": True, "trace": out, "generated_at": now_iso()}), 200


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUCTION INTELLIGENCE — PHASE 1 ROUTES
# Additive only. Zero modification to any route above this block.
# Append this block at the end of app.py, before the if __name__ == "__main__"
# guard.
#
# Routes:
#   GET  /api/auction/sources              — list active auction sources
#   GET  /api/auction/listings             — paginated listing feed
#   GET  /api/auction/listings/<id>        — single listing detail
#   POST /api/auction/listings/<id>/convert — convert listing to deal
#   POST /api/auction/scan                 — admin-only manual scan trigger
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AUCTION_SCAN_SECRET = os.environ.get("AUCTION_SCAN_SECRET", "").strip()


@app.route("/api/auction/sources", methods=["GET", "OPTIONS"])
@require_auth
def auction_sources_list():
    """Return all active auction source records (name, slug, last_scanned_at)."""
    if request.method == "OPTIONS":
        return jsonify({}), 200
    if not supabase:
        return jsonify({"error": "Database unavailable"}), 503
    try:
        res = supabase.table("auction_sources") \
            .select("id,name,slug,listings_url,scrape_method,active,last_scanned_at") \
            .eq("active", True) \
            .order("name") \
            .execute()
        return jsonify({"ok": True, "sources": res.data or []}), 200
    except Exception as e:
        app.logger.exception("auction_sources_list failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/auction/listings", methods=["GET", "OPTIONS"])
@require_auth
def auction_listings_list():
    """
    Paginated listing feed.

    Query params:
      page     int  — 1-indexed, default 1
      per_page int  — default 24, max 100
      source   str  — filter by auction_sources.slug
      status   str  — filter by status (default: active)
      min_price, max_price — guide_price range filter
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200
    if not supabase:
        return jsonify({"error": "Database unavailable"}), 503

    try:
        page     = max(1, int(request.args.get("page", 1)))
        per_page = min(100, max(1, int(request.args.get("per_page", 24))))
        status   = request.args.get("status", "active")
        source_slug = request.args.get("source", "").strip()
        min_price   = safe_float(request.args.get("min_price"))
        max_price   = safe_float(request.args.get("max_price"))

        offset = (page - 1) * per_page

        q = supabase.table("auction_listings") \
            .select(
                "id,source_id,source_url,auction_house,lot_number,address,postcode,"
                "guide_price,auction_date,property_type,legal_pack_url,"
                "status,converted_deal_id,first_seen_at,last_seen_at,image_url,"
                "investment_json,enrichment_status,enrichment_confidence,"
                "auction_sources!auction_listings_source_id_fkey(slug,name)",
                count="exact"
            ) \
            .eq("status", status) \
            .order("auction_date", desc=False, nullsfirst=False) \
            .order("first_seen_at", desc=True) \
            .range(offset, offset + per_page - 1)

        if source_slug:
            # Filter via join — get source_id first
            src = supabase.table("auction_sources") \
                .select("id").eq("slug", source_slug).maybe_single().execute()
            if src.data:
                q = q.eq("source_id", src.data["id"])

        if min_price is not None:
            q = q.gte("guide_price", min_price)
        if max_price is not None:
            q = q.lte("guide_price", max_price)

        res = q.execute()
        listings = res.data or []
        total    = res.count or 0

        return jsonify({
            "ok":        True,
            "listings":  listings,
            "total":     total,
            "page":      page,
            "per_page":  per_page,
            "pages":     max(1, -(-total // per_page)),  # ceiling division
        }), 200

    except Exception as e:
        app.logger.exception("auction_listings_list failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/auction/listings/<listing_id>", methods=["GET", "OPTIONS"])
@require_auth
def auction_listing_detail(listing_id: str):
    """Return a single listing with full detail."""
    if request.method == "OPTIONS":
        return jsonify({}), 200
    if not supabase:
        return jsonify({"error": "Database unavailable"}), 503
    try:
        res = supabase.table("auction_listings") \
            .select("*,auction_sources!auction_listings_source_id_fkey(slug,name)") \
            .eq("id", listing_id) \
            .single() \
            .execute()
        if not res.data:
            return jsonify({"error": "Listing not found"}), 404
        return jsonify({"ok": True, "listing": res.data}), 200
    except Exception as e:
        app.logger.exception("auction_listing_detail failed for %s", listing_id)
        return jsonify({"error": str(e)}), 500




@app.route("/api/auction/listings/<listing_id>/infer", methods=["POST", "OPTIONS"])
@require_auth
def auction_listing_infer(listing_id):
    """
    POST /api/auction/listings/<listing_id>/infer
    On-demand Phase 9 LLM inference for a single discovery listing.
    Result cached in investment_json.llm_inference.
    Pass ?force=true to bypass cache and regenerate.
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200
    if not supabase:
        return jsonify({"error": "Database unavailable"}), 503

    force = request.args.get("force", "").lower() in ("1", "true", "yes")

    try:
        # Fetch listing
        res = supabase.table("auction_listings")             .select(
                "id,address,postcode,guide_price,property_type,auction_date,"
                "auction_house,source_url,investment_json,enrichment_status"
            )             .eq("id", listing_id)             .maybe_single()             .execute()

        if not res.data:
            return jsonify({"error": "listing_not_found"}), 404

        listing = res.data
        inv     = listing.get("investment_json") or {}

        # Return cached inference unless force=true
        if not force and inv.get("llm_inference") and not inv["llm_inference"].get("error"):
            return jsonify({
                "listing_id":    listing_id,
                "llm_inference": inv["llm_inference"],
                "cached":        True,
            }), 200

        # Require basic enrichment before running LLM
        if listing.get("enrichment_status") not in ("complete", "partial"):
            return jsonify({
                "error":  "listing_not_enriched",
                "status": listing.get("enrichment_status"),
            }), 422

        # Run LLM inference
        from services.auction_inference import run_inference
        result = run_inference(listing=listing, inv=inv)

        if result.get("error"):
            app.logger.error("[LLM_INFER] %s: %s", listing_id, result["error"])
            return jsonify({"error": result["error"]}), 500

        # Cache result in investment_json.llm_inference
        inv["llm_inference"] = result
        supabase.table("auction_listings")             .update({"investment_json": inv})             .eq("id", listing_id)             .execute()

        return jsonify({
            "listing_id":    listing_id,
            "llm_inference": result,
            "cached":        False,
        }), 200

    except Exception as e:
        app.logger.exception("[LLM_INFER] endpoint error for %s", listing_id)
        return jsonify({"error": str(e)}), 500


@app.route("/api/auction/enrichment/debug/<listing_id>", methods=["GET", "OPTIONS"])
@require_auth
def enrichment_debug(listing_id):
    """
    GET /api/auction/enrichment/debug/<listing_id>
    Returns enrichment state for one listing — step results, timing,
    populated fields, errors. Temporary observability endpoint.
    Remove once enrichment pipeline is confirmed stable.
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200
    if not supabase:
        return jsonify({"error": "Database unavailable"}), 503
    try:
        res = supabase.table("auction_listings")             .select(
                "id,address,postcode,guide_price,auction_house,"
                "enrichment_status,enrichment_confidence,enriched_at,"
                "investment_json,enrichment_error"
            )             .eq("id", listing_id)             .maybe_single()             .execute()
        if not res.data:
            return jsonify({"error": "listing_not_found"}), 404

        l   = res.data
        inv = l.get("investment_json") or {}

        return jsonify({
            "listing_id":          listing_id,
            "address":             l.get("address"),
            "postcode":            l.get("postcode"),
            "guide_price":         l.get("guide_price"),
            "auction_house":       l.get("auction_house"),
            "enrichment_status":   l.get("enrichment_status"),
            "enrichment_confidence": l.get("enrichment_confidence"),
            "enriched_at":         l.get("enriched_at"),
            "steps_completed":     inv.get("steps_completed", []),
            "steps_failed":        inv.get("steps_failed", []),
            "step_timing":         inv.get("step_timing", {}),
            "duration_s":          inv.get("duration_s"),
            "populated_fields": {
                "postcode_lad":    bool((inv.get("postcode") or {}).get("lad_code")),
                "hpi_avg_price":   bool((inv.get("hpi") or {}).get("regional_avg_price")),
                "rental_avg":      bool((inv.get("rental") or {}).get("avg_rent_gbp")),
                "comps_count":     bool((inv.get("comps") or {}).get("count")),
                "epc_rating":      bool((inv.get("epc") or {}).get("rating")),
                "yield_pct":       bool((inv.get("yield_estimate") or {}).get("gross_yield_pct")),
                "inference_summary": bool((inv.get("inference") or {}).get("summary")),
                "inference_signals": len((inv.get("inference") or {}).get("signals", [])),
            },
            "step_errors": {
                s: (inv.get(s) or {}).get("error")
                for s in ["postcode", "hpi", "rental", "comps", "epc", "yield_estimate"]
                if (inv.get(s) or {}).get("error")
            },
            "enrichment_error":    l.get("enrichment_error"),
        }), 200
    except Exception as e:
        app.logger.exception("enrichment_debug failed for %s", listing_id)
        return jsonify({"error": str(e)}), 500


@app.route("/api/auction/listings/<listing_id>/convert", methods=["POST", "OPTIONS"])
@require_auth
def auction_listing_convert(listing_id: str):
    """
    Convert an auction listing to a deal in the existing pipeline.

    Idempotent: if the listing is already converted, returns the existing
    deal_id without creating a duplicate.

    Flow:
      1. Load listing
      2. Check not already converted (idempotency gate)
      3. POST to existing deal creation logic (reuse inline, avoid HTTP self-call)
      4. Mark listing as converted
      5. Return {deal_id} — frontend redirects to upload page

    Does NOT:
      - Upload a legal pack
      - Run LLM analysis
      - Modify any existing deal
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200
    if not supabase:
        return jsonify({"error": "Database unavailable"}), 503

    try:
        # ── Load listing ─────────────────────────────────────────────────
        listing_res = supabase.table("auction_listings") \
            .select("id,status,converted_deal_id,address,postcode,lot_number,"
                    "guide_price,auction_date,property_type,auction_house,source_url") \
            .eq("id", listing_id) \
            .single() \
            .execute()

        if not listing_res.data:
            return jsonify({"error": "Listing not found"}), 404

        listing = listing_res.data

        # ── Idempotency gate ─────────────────────────────────────────────
        if listing.get("converted_deal_id"):
            app.logger.info(
                "[auction/convert] Listing %s already converted → deal %s",
                listing_id, listing["converted_deal_id"]
            )
            return jsonify({
                "ok":               True,
                "deal_id":          listing["converted_deal_id"],
                "already_converted": True,
            }), 200

        # ── Build deal name ───────────────────────────────────────────────
        address = listing.get("address") or ""
        lot_num = listing.get("lot_number") or ""
        house   = listing.get("auction_house") or ""

        if address:
            deal_name = address
            if lot_num:
                deal_name = f"Lot {lot_num} — {address}"
        elif lot_num:
            deal_name = f"{house} Lot {lot_num}" if house else f"Lot {lot_num}"
        else:
            deal_name = f"{house} listing" if house else "Auction listing"

        # Truncate to reasonable length
        deal_name = deal_name[:120]

        # ── Create deal (reuse Supabase insert directly) ──────────────────
        deal_row = {
            "user_id":            request.user_id,
            "deal_name":          deal_name,
            "title":              deal_name,
            "address":            listing.get("address"),
            "postcode":           listing.get("postcode"),
            "lot_number":         listing.get("lot_number"),
            "guide_price":        listing.get("guide_price"),
            "deal_type":          None,   # user sets after upload
            "auction_date":       listing.get("auction_date"),
            "status":             "active",
            "source_listing_id":  listing_id,
        }

        deal_res = supabase.table("deals").insert(deal_row).execute()

        if not deal_res.data or not deal_res.data[0].get("id"):
            app.logger.error("[auction/convert] Deal insert returned no id for listing %s", listing_id)
            return jsonify({"error": "Deal creation failed"}), 500

        deal_id = deal_res.data[0]["id"]

        # ── Mark listing as converted ─────────────────────────────────────
        try:
            supabase.table("auction_listings").update({
                "status":             "converted",
                "converted_deal_id":  deal_id,
            }).eq("id", listing_id).execute()
        except Exception as mark_e:
            # Non-fatal: deal was created successfully. Log the marking failure.
            app.logger.warning(
                "[auction/convert] Deal %s created but could not mark listing %s as converted: %s",
                deal_id, listing_id, mark_e
            )

        app.logger.info(
            "[auction/convert] Listing %s → deal %s (user %s)",
            listing_id, deal_id, request.user_id
        )

        return jsonify({
            "ok":               True,
            "deal_id":          deal_id,
            "already_converted": False,
        }), 201

    except Exception as e:
        app.logger.exception("auction_listing_convert failed for listing %s", listing_id)
        return jsonify({"error": str(e)}), 500


@app.route("/api/auction/scan", methods=["POST", "OPTIONS"])
@require_auth
def auction_manual_scan():
    """
    Admin-only: trigger a scan of one or all auction sources.
    Requires X-Scan-Secret header matching AUCTION_SCAN_SECRET env var.

    Body (optional): {"slug": "allsop"}  — omit to scan all active sources.

    Note: this runs synchronously in the request thread.
    For large source sets, the cron job is preferred.
    Use this for testing and one-off manual refreshes only.
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200
    if not supabase:
        return jsonify({"error": "Database unavailable"}), 503

    # Admin gate — shared secret check
    if not AUCTION_SCAN_SECRET:
        return jsonify({"error": "Manual scan not configured — set AUCTION_SCAN_SECRET env var"}), 503

    provided = request.headers.get("X-Scan-Secret", "").strip()
    if provided != AUCTION_SCAN_SECRET:
        app.logger.warning("[auction/scan] Unauthorised scan attempt by user %s", request.user_id)
        return jsonify({"error": "Forbidden"}), 403

    try:
        from services.auction_scraper import scrape_source as _scrape
    except ImportError as e:
        return jsonify({"error": f"Scraper not available: {e}"}), 503

    data = request.get_json(silent=True) or {}
    target_slug = (data.get("slug") or "").strip()

    # Load sources
    q = supabase.table("auction_sources") \
        .select("id,name,slug,listings_url,scrape_method,selectors") \
        .eq("active", True)
    if target_slug:
        q = q.eq("slug", target_slug)

    sources_res = q.execute()
    sources = sources_res.data or []

    if not sources:
        return jsonify({"error": f"No active source found{' for slug: ' + target_slug if target_slug else ''}"}), 404

    results = []
    total_new = 0
    total_updated = 0

    for source in sources:
        slug       = source.get("slug", "unknown")
        t0         = time.time()
        status     = "ok"
        error_msg  = None
        listings   = []
        new_c = updated_c = 0

        try:
            _meta: dict = {}
            listings = _scrape(source, meta=_meta)
            new_c, updated_c = _upsert_auction_listings(supabase, listings)
            if not listings and status == "ok":
                status = "partial"
                error_msg = _meta.get("partial_reason", "zero_listings")
        except Exception as exc:
            status    = "failed"
            error_msg = str(exc)[:500]
            app.logger.error("[auction/scan:%s] %s", slug, exc, exc_info=True)

        duration = round(time.time() - t0, 2)

        try:
            supabase.table("auction_scan_log").insert({
                "source_id":        source["id"],
                "source_slug":      slug,
                "status":           status,
                "duration_s":       duration,
                "listings_found":   len(listings),
                "listings_new":     new_c,
                "listings_updated": updated_c,
                "error_msg":        error_msg,
            }).execute()
        except Exception as log_e:
            app.logger.warning("[auction/scan] Log write failed for %s: %s", slug, log_e)

        try:
            supabase.table("auction_sources").update({
                "last_scanned_at": now_iso()
            }).eq("id", source["id"]).execute()
        except Exception:
            pass

        total_new     += new_c
        total_updated += updated_c
        results.append({
            "slug":     slug,
            "status":   status,
            "found":    len(listings),
            "new":      new_c,
            "updated":  updated_c,
            "duration_s": duration,
            "error":    error_msg,
        })

    return jsonify({
        "ok":           True,
        "sources_scanned": len(results),
        "total_new":    total_new,
        "total_updated": total_updated,
        "results":      results,
    }), 200


def _upsert_auction_listings(supabase_client, listings: list) -> tuple:
    """
    Shared upsert logic used by both manual scan and (future) async paths.
    Returns (new_count, updated_count).
    """
    if not listings:
        return 0, 0

    source_urls = [l["source_url"] for l in listings]
    try:
        existing_res = supabase_client.table("auction_listings") \
            .select("source_url") \
            .in_("source_url", source_urls) \
            .execute()
        existing = {r["source_url"] for r in (existing_res.data or [])}
    except Exception:
        existing = set()

    from datetime import datetime, timezone
    now_ts = datetime.now(timezone.utc).isoformat()
    new_c = updated_c = 0

    for listing in listings:
        is_new = listing["source_url"] not in existing
        row = {k: v for k, v in listing.items() if not k.startswith("_")}
        row["last_seen_at"] = now_ts
        if is_new:
            row["first_seen_at"] = now_ts

        _preserve = {"guide_price", "legal_pack_url", "auction_date",
                     "address", "postcode", "property_type", "tenure", "lot_number"}
        try:
            if is_new:
                supabase_client.table("auction_listings").insert(row).execute()
                new_c += 1
            else:
                _update = {k: v for k, v in row.items()
                           if v is not None or k not in _preserve}
                supabase_client.table("auction_listings")                     .update(_update)                     .eq("source_url", listing.get("source_url"))                     .execute()
                updated_c += 1
        except Exception as e:
            app.logger.warning("[auction/upsert] Failed for %s: %s", listing.get("source_url"), e)

    return new_c, updated_c


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
