# app.py

from flask import Flask, request, jsonify
import requests
import os
import time
import json
import re
from flask_cors import CORS
from supabase import create_client, Client
from typing import Dict, Any, Optional, Tuple, List

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ---- Supabase env hardening (support legacy names) ----
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip()
SUPABASE_SERVICE_ROLE_KEY = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
SUPABASE_KEY_FALLBACK = (os.getenv("SUPABASE_KEY") or "").strip()
SUPABASE_KEY = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY_FALLBACK

# ----------------------------
# Caching (in-memory, TTL)
# ----------------------------
_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SECONDS = int(os.getenv("MARKET_INSIGHTS_CACHE_TTL_SECONDS", "21600"))  # default 6 hours

# Identify ourselves to public services (Nominatim requires a proper UA)
HTTP_USER_AGENT = os.getenv(
    "HTTP_USER_AGENT",
    "LegalSmegal/1.0 (market-insights; contact=admin@example.com)"
)

# Safety + stability: bound payload sizes
MAX_CRIMES = int(os.getenv("MAX_CRIMES", "300"))
MAX_OSM_NAMES = int(os.getenv("MAX_OSM_NAMES", "250"))
DEFAULT_OSM_RADIUS = int(os.getenv("OSM_RADIUS_METERS", "1200"))

# Confidence rules
MIN_VERIFIED = float(os.getenv("MIN_VERIFIED_CONFIDENCE", "0.95"))

# Provider toggles
SCHOOLS_PROVIDER = os.getenv("SCHOOLS_PROVIDER", "").strip().lower()          # "supabase" or ""
BROADBAND_PROVIDER = os.getenv("BROADBAND_PROVIDER", "").strip().lower()      # "supabase" or ""

# Housing provider (Supabase RPC)
HOUSING_PROVIDER = os.getenv("HOUSING_PROVIDER", "supabase_rpc").strip().lower()   # "supabase_rpc" or ""
HOUSING_RPC_NAME = os.getenv("HOUSING_RPC_NAME", "housing_comps_v1").strip()
HOUSING_MAX_LIMIT = int(os.getenv("HOUSING_MAX_LIMIT", "50"))
HOUSING_DEFAULT_LIMIT = int(os.getenv("HOUSING_DEFAULT_LIMIT", "20"))
HOUSING_DEFAULT_RADIUS_MILES = float(os.getenv("HOUSING_DEFAULT_RADIUS_MILES", "3"))
HOUSING_CONFIDENCE_VALUE = float(os.getenv("HOUSING_CONFIDENCE_VALUE", "0.96"))

# Supabase-backed adapters
SCHOOLS_SUPABASE_VIEW = os.getenv("SCHOOLS_SUPABASE_VIEW", "schools_by_district").strip()
SCHOOLS_SUPABASE_FALLBACK_TABLE = os.getenv("SCHOOLS_SUPABASE_FALLBACK_TABLE", "schools_clean_v2").strip()

SCHOOLS_MAX_RESULTS = int(os.getenv("SCHOOLS_MAX_RESULTS", "20"))
SCHOOLS_CONFIDENCE_VALUE = float(os.getenv("SCHOOLS_CONFIDENCE_VALUE", "0.90"))

BROADBAND_SUPABASE_TABLE = os.getenv("BROADBAND_SUPABASE_TABLE", "").strip()
BROADBAND_MAX_RESULTS = int(os.getenv("BROADBAND_MAX_RESULTS", "5"))
BROADBAND_CONFIDENCE_VALUE = float(os.getenv("BROADBAND_CONFIDENCE_VALUE", "0.90"))

# ----------------------------
# Nomis (Census 2021) API
# ----------------------------
NOMIS_ENABLED = (os.getenv("NOMIS_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"})
NOMIS_DATASET_ID = os.getenv("NOMIS_DATASET_ID", "NM_2023_1").strip()

# Keep as a fallback only (debug / emergency). UK-wide uses postcode->LSOA.
NOMIS_DEFAULT_GEOGRAPHY = os.getenv("NOMIS_DEFAULT_GEOGRAPHY", "").strip()
NOMIS_TIMEOUT = int(os.getenv("NOMIS_TIMEOUT", "20"))

# TS003 preset from your example URL
NOMIS_TS003_DIM = os.getenv("NOMIS_TS003_DIM", "c2021_hhcomp_15").strip()
NOMIS_TS003_CATS = os.getenv(
    "NOMIS_TS003_CATS",
    "1001,1,2,1002,1003,4...6,1004,7...9,1005,10,11,1006,12,1007,13,14"
).strip()

# TS044 / TS054 are configurable.
# Set these to EXACTLY what appears in your copy-address URL (dimension name + category list).
NOMIS_TS044_DIM = os.getenv("NOMIS_TS044_DIM", "").strip()
NOMIS_TS044_CATS = os.getenv("NOMIS_TS044_CATS", "").strip()

NOMIS_TS054_DIM = os.getenv("NOMIS_TS054_DIM", "").strip()
NOMIS_TS054_CATS = os.getenv("NOMIS_TS054_CATS", "").strip()

# ----------------------------
# Postcode -> LSOA (GSS) via postcodes.io (UK-wide)
# ----------------------------
POSTCODES_IO_TIMEOUT = int(os.getenv("POSTCODES_IO_TIMEOUT", "10"))
POSTCODES_IO_CACHE_TTL_SECONDS = int(os.getenv("POSTCODES_IO_CACHE_TTL_SECONDS", "2592000"))  # 30 days
_GEO_CACHE: Dict[str, Dict[str, Any]] = {}

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("🟢 Supabase enabled. URL:", SUPABASE_URL)
else:
    print("🔴 Supabase env vars not set. Supabase features are DISABLED.")


# ----------------------------
# Helpers
# ----------------------------
def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def normalize_postcode(pc: str) -> str:
    """UK canonical display format: uppercase, single internal spaces."""
    if not isinstance(pc, str):
        return ""
    return " ".join(pc.strip().upper().split())


def normalize_postcode_nospace(pc: str) -> str:
    """UK canonical lookup key: uppercase, remove all whitespace."""
    if not isinstance(pc, str):
        return ""
    return re.sub(r"\s+", "", pc.strip().upper())


def postcode_district(pc: str) -> str:
    pc = normalize_postcode(pc)
    if not pc:
        return ""
    return pc.split(" ")[0] if " " in pc else pc


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
        if f != f:  # NaN
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


def resolve_lsoa_gss_from_postcode(postcode: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Returns (lsoa_gss, meta)
    lsoa_gss looks like 'E0100....' (England/Wales), 'S0100....' (Scotland), etc.
    """
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
    }

    if not pc_key:
        meta["notes"] = "No postcode provided."
        return None, meta

    cache_key = f"lsoa_gss::{pc_key}"
    cached = geo_cache_get(cache_key)
    if cached and isinstance(cached.get("lsoa_gss"), str) and cached["lsoa_gss"].strip():
        meta["cache"]["hit"] = True
        meta["notes"] = "Resolved from geo cache."
        return cached["lsoa_gss"].strip(), meta

    url = f"https://api.postcodes.io/postcodes/{pc_key}"
    try:
        status, payload = _http_get_json(url, timeout=POSTCODES_IO_TIMEOUT)
        if status != 200 or not isinstance(payload, dict):
            meta["notes"] = f"postcodes.io returned HTTP {status}"
            return None, meta

        result = payload.get("result") if isinstance(payload.get("result"), dict) else None
        if not isinstance(result, dict):
            meta["notes"] = "postcodes.io returned no result object."
            return None, meta

        codes = result.get("codes") if isinstance(result.get("codes"), dict) else {}
        lsoa_gss = codes.get("lsoa")
        lsoa_gss = lsoa_gss.strip() if isinstance(lsoa_gss, str) and lsoa_gss.strip() else None
        if not lsoa_gss:
            meta["notes"] = "postcodes.io result missing codes.lsoa (GSS)."
            return None, meta

        geo_cache_set(cache_key, {"lsoa_gss": lsoa_gss})
        meta["notes"] = "Resolved LSOA GSS from postcodes.io."
        return lsoa_gss, meta

    except Exception as e:
        meta["notes"] = f"postcodes.io exception: {str(e)}"
        return None, meta


# ----------------------------
# Nomis (JSON-stat) — UK-wide census tables
# ----------------------------
def fetch_nomis_jsonstat(dataset_id: str, params: dict) -> dict:
    base = f"https://www.nomisweb.co.uk/api/v01/dataset/{dataset_id}.jsonstat.json"
    payload = _http_get_json_raw(base, params=params, timeout=NOMIS_TIMEOUT)
    if not isinstance(payload, dict) or "dataset" not in payload:
        raise ValueError("Nomis returned unexpected payload (missing 'dataset').")
    return payload


def parse_jsonstat_single_dimension(jsonstat: dict) -> Dict[str, Any]:
    ds = jsonstat.get("dataset") or {}
    dim = ds.get("dimension") or {}
    dim_ids = dim.get("id") or []
    if not isinstance(dim_ids, list):
        dim_ids = []

    exclude = {"date", "time", "geography", "measures"}
    candidate_dims = [d for d in dim_ids if isinstance(d, str) and d.lower() not in exclude]

    main_dim = candidate_dims[0] if candidate_dims else None
    if not main_dim:
        for d in dim_ids:
            cat = (((dim.get(d) or {}).get("category")) or {})
            if isinstance(cat.get("index"), (list, dict)):
                main_dim = d
                break

    if not main_dim:
        raise ValueError("Could not infer main dimension from JSON-stat.")

    cat = ((dim.get(main_dim) or {}).get("category")) or {}
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
            "Nomis requires a geography id. Provide a postcode (preferred) or set NOMIS_DEFAULT_GEOGRAPHY for fallback.",
            sources,
            retrieved,
        )

    if not dimension or not categories:
        return metric_missing_provider(
            f"{label} not configured. Set env for its dimension/categories (e.g. NOMIS_TS054_DIM and NOMIS_TS054_CATS).",
            sources,
            retrieved,
            extra_metrics={"label": label, "dimension": dimension, "categories": categories, "geography": geography},
        )

    try:
        params = {
            "date": "latest",
            "geography": geography,
            dimension: categories,
            "measures": "20100",
        }
        js = fetch_nomis_jsonstat(NOMIS_DATASET_ID, params)
        parsed = parse_jsonstat_single_dimension(js)

        bullets = [f"• {it['label']}: {it['value']}" for it in parsed["items"]]
        summary = f"{label} (Nomis) — total households: {parsed['total']}"

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


# ----------------------------
# NSPL (Supabase view) postcode -> lat/lng
# Requires: public.nspl_lookup(pcd_nospace, lat, lng)
# ----------------------------
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
        res = (
            supabase.table("nspl_lookup")
            .select("lat,lng")
            .eq("pcd_nospace", pc_key)
            .limit(1)
            .execute()
        )

        rows = res.data if hasattr(res, "data") else None
        if not isinstance(rows, list) or not rows:
            meta["notes"] = f"NSPL lookup returned no rows for {pc_key}."
            meta["sources"] = [{"label": "Supabase (nspl_lookup)", "url": f"{SUPABASE_URL}"}]
            return None, None, meta

        lat = safe_float(rows[0].get("lat"))
        lng = safe_float(rows[0].get("lng"))
        if lat is None or lng is None:
            meta["notes"] = "NSPL lookup returned invalid coordinates."
            meta["sources"] = [{"label": "Supabase (nspl_lookup)", "url": f"{SUPABASE_URL}"}]
            return None, None, meta

        meta["notes"] = "Resolved from NSPL."
        meta["sources"] = [{"label": "Supabase (nspl_lookup)", "url": f"{SUPABASE_URL}"}]
        return lat, lng, meta

    except Exception as e:
        meta["notes"] = f"NSPL lookup exception: {str(e)}"
        meta["sources"] = [{"label": "Supabase (nspl_lookup)", "url": f"{SUPABASE_URL}"}]
        return None, None, meta


# ----------------------------
# Geocoding (Nominatim) fallback only
# ----------------------------
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


# ----------------------------
# Summaries
# ----------------------------
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


# ----------------------------
# Crime (UK Police)
# ----------------------------
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


# ----------------------------
# OSM (Overpass) Utilities
# ----------------------------
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


# ----------------------------
# ✅ TRANSPORT
# ----------------------------
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
nwr["public_transport"](around:{radius},{lat},{lng});
nwr["highway"="bus_stop"](around:{radius},{lat},{lng});
""".strip()

    try:
        payload = overpass_query(lat, lng, selectors)
        elements = payload.get("elements", []) if isinstance(payload, dict) else []
        if not isinstance(elements, list):
            elements = []

        counts: Dict[str, int] = {"stations": 0, "tram_stops": 0, "public_transport": 0, "bus_stops": 0}
        named_stations: List[str] = []
        named_tram: List[str] = []
        named_pt: List[str] = []
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

            if "public_transport" in tags:
                counts["public_transport"] += 1
                if nm:
                    named_pt.append(nm)

            if tags.get("highway") == "bus_stop":
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
        bullets.append(f"• Rail: {counts['stations']} station(s) within ~{radius}m" + (f" (e.g., {', '.join(named_stations)})" if named_stations else ""))
        if counts["tram_stops"] > 0:
            bullets.append(f"• Tram: {counts['tram_stops']} stop(s) within ~{radius}m" + (f" (e.g., {', '.join(named_tram)})" if named_tram else ""))
        bullets.append(f"• Bus: {counts['bus_stops']} stop(s) within ~{radius}m" + (f" (e.g., {', '.join(named_bus)})" if named_bus else ""))

        if not elements or (counts["stations"] + counts["tram_stops"] + counts["bus_stops"] + counts["public_transport"]) == 0:
            return metric_ok("No transport features returned from OSM for this area.", [], base_sources, retrieved, 0.0)

        summary = "Transport (OSM within ~1.2km):\n" + "\n".join(bullets)

        out = metric_ok(summary, bullets, base_sources, retrieved, 0.90)
        out["metrics"] = {"radiusMeters": radius, "counts": counts, "sample": {"stations": named_stations, "tram": named_tram, "bus": named_bus}}
        return out

    except Exception as e:
        return metric_unavailable(
            f"Transport data fetch failed: {str(e)}",
            base_sources,
            retrieved,
        )


# ----------------------------
# ✅ AMENITIES
# ----------------------------
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
        service_amenities = {"bank", "atm", "post_office", "parcel_locker", "police", "fire_station", "townhall", "community_centre", "courthouse", "place_of_worship"}

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


# ----------------------------
# Schools (Supabase adapter)
# ----------------------------
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


# ----------------------------
# Broadband (Supabase adapter - optional)
# ----------------------------
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


# ----------------------------
# Housing (SOLD COMPS via Supabase RPC)
# ----------------------------
def _median_int(values: List[int]) -> Optional[int]:
    vs = sorted([v for v in values if isinstance(v, int)])
    if not vs:
        return None
    mid = len(vs) // 2
    if len(vs) % 2 == 1:
        return vs[mid]
    return int((vs[mid - 1] + vs[mid]) / 2)


def get_housing_data(postcode: str, radius_miles: Optional[float] = None, limit: Optional[int] = None) -> Dict[str, Any]:
    retrieved = now_iso()
    pc = normalize_postcode(postcode)

    sources = [
        {"label": "HM Land Registry (Price Paid)", "url": "https://www.gov.uk/government/collections/price-paid-data"},
        {"label": "Supabase (Postgres)", "url": f"{SUPABASE_URL}" if SUPABASE_URL else "https://supabase.com/"},
    ]

    if not pc:
        return metric_unavailable("Housing data not available: no postcode provided.", sources, retrieved)

    r_miles = radius_miles if isinstance(radius_miles, (int, float)) and radius_miles > 0 else HOUSING_DEFAULT_RADIUS_MILES
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

    try:
        payload = {"postcode": pc, "radius_miles": r_miles, "limit_n": lim}
        res = supabase.rpc(HOUSING_RPC_NAME, payload).execute()
        rows = res.data if hasattr(res, "data") else None
        if not isinstance(rows, list):
            rows = []

        if not rows:
            return metric_unavailable(
                f"No sold comparables returned within {r_miles} miles for {pc}.",
                sources,
                retrieved,
                extra_metrics={"postcode": pc, "radius_miles": r_miles, "limit": lim, "rpc": HOUSING_RPC_NAME},
            )

        prices: List[int] = []
        ptypes: Dict[str, int] = {}
        miles_list: List[float] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            pr = safe_int(r.get("price"))
            if isinstance(pr, int):
                prices.append(pr)
            pt = (r.get("property_type") or "").strip()
            if pt:
                ptypes[pt] = ptypes.get(pt, 0) + 1
            mi = safe_float(r.get("miles"))
            if isinstance(mi, float):
                miles_list.append(mi)

        med = _median_int(prices)
        min_m = min(miles_list) if miles_list else None
        max_m = max(miles_list) if miles_list else None

        pt_parts = [f"{k}:{ptypes[k]}" for k in sorted(ptypes.keys())]
        pt_str = ", ".join(pt_parts) if pt_parts else "n/a"

        summary = f"{len(rows)} sold comparables within {r_miles} miles. Median price: {med if med is not None else 'n/a'}. Types: {pt_str}."

        out = metric_ok(summary, rows, sources, retrieved, HOUSING_CONFIDENCE_VALUE)
        out["metrics"] = {
            "provider": "supabase_rpc",
            "rpc": HOUSING_RPC_NAME,
            "postcode": pc,
            "radius_miles": r_miles,
            "limit": lim,
            "count": len(rows),
            "median_price": med,
            "min_miles": min_m,
            "max_miles": max_m,
            "property_type_counts": ptypes,
        }
        return out

    except Exception as e:
        msg = str(e) or "Unknown error"
        return metric_missing_provider(
            f"Housing RPC not available. Create Supabase function '{HOUSING_RPC_NAME}' then retry. Error: {msg}",
            sources,
            retrieved,
            extra_metrics={"postcode": pc, "radius_miles": r_miles, "limit": lim, "rpc": HOUSING_RPC_NAME},
        )


# ----------------------------
# Route: Adapter endpoints
# ----------------------------
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
    table = (request.args.get("table", "") or "").strip().lower()
    postcode = normalize_postcode(request.args.get("postcode", "") or "")
    geography = (request.args.get("geography", "") or "").strip()

    # Prefer UK-wide: postcode -> LSOA(GSS). Allow manual geography override for debugging.
    if not geography and postcode:
        lsoa_gss, _meta = resolve_lsoa_gss_from_postcode(postcode)
        geography = lsoa_gss or ""

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


# ----------------------------
# Route: /market-insights
# ----------------------------
@app.route("/market-insights", methods=["POST"])
def market_insights():
    data = request.get_json(silent=True) or {}
    postcode = normalize_postcode(data.get("postcode", "") or "")
    lat = safe_float(data.get("lat"))
    lng = safe_float(data.get("lng"))

    # UK-wide Nomis geography derived from postcode (LSOA GSS)
    lsoa_gss = ""
    lsoa_meta = None
    if postcode:
        lsoa_gss, lsoa_meta = resolve_lsoa_gss_from_postcode(postcode)
    nomis_geo = lsoa_gss or NOMIS_DEFAULT_GEOGRAPHY

    cache_key = f"market-insights::{postcode}::{lat or ''}::{lng or ''}::{nomis_geo or ''}" if postcode else "market-insights::no-postcode"
    cached = cache_get(cache_key)
    if cached:
        return jsonify({**cached, "_cache": {"hit": True, "ttlSeconds": CACHE_TTL_SECONDS}})

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
            "census": {
                "ts003": get_nomis_table("Household composition (TS003)", NOMIS_TS003_DIM, NOMIS_TS003_CATS, nomis_geo),
                "ts044": get_nomis_table("Accommodation type (TS044)", NOMIS_TS044_DIM, NOMIS_TS044_CATS, nomis_geo),
                "ts054": get_nomis_table("Tenure (TS054)", NOMIS_TS054_DIM, NOMIS_TS054_CATS, nomis_geo),
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

        cache_set(cache_key, results)
        return jsonify({**results, "_cache": {"hit": False, "ttlSeconds": CACHE_TTL_SECONDS}})

    except Exception as e:
        print("❌ Error in /market-insights:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Legal Smegal API Final",
        "status": "active",
        "supabaseEnabled": bool(supabase),
        "routes": {
            "POST /market-insights": "{ 'postcode': 'EC3A 5DE' }  // optional: lat/lng",
            "GET /adapters/geo?postcode=B1%201AA": "debug postcode -> LSOA(GSS)",
            "GET /adapters/nomis?table=ts003&postcode=B1%201AA": "Nomis TS003 UK-wide via postcode",
            "GET /adapters/nomis?table=ts054&postcode=B1%201AA": "Nomis TS054 (requires env dims/cats)",
            "GET /adapters/nomis?table=ts044&postcode=B1%201AA": "Nomis TS044 (requires env dims/cats)",
            "GET /adapters/schools?postcode=EC3A%205DE": "debug schools adapter",
            "GET /adapters/broadband?postcode=EC3A%205DE": "debug broadband adapter",
            "GET /adapters/housing/comps?postcode=EC3A%205DE&radius_miles=3&limit=20": "debug housing sold comps (RPC)",
        },
        "envHints": {
            "SUPABASE_URL": "required for supabase providers",
            "SUPABASE_SERVICE_ROLE_KEY": "preferred (server-only). SUPABASE_KEY also supported as fallback.",
            "SCHOOLS_PROVIDER": "set to 'supabase' to enable",
            "BROADBAND_PROVIDER": "set to 'supabase' to enable",
            "BROADBAND_SUPABASE_TABLE": "e.g. broadband_by_postcode",
            "NSPL": "requires view/table public.nspl_lookup(pcd_nospace, lat, lng) for postcode->coords",
            "HOUSING_PROVIDER": "set to 'supabase_rpc' to enable sold comps",
            "HOUSING_RPC_NAME": "defaults to 'housing_comps_v1'",
            "NOMIS_ENABLED": "1 to enable",
            "NOMIS_DEFAULT_GEOGRAPHY": "fallback only (optional)",
            "NOMIS_TS054_DIM/NOMIS_TS054_CATS": "paste from TS054 copy-address URL",
            "NOMIS_TS044_DIM/NOMIS_TS044_CATS": "paste from TS044 copy-address URL",
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
