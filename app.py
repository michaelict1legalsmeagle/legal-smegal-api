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

# Some deployments used SUPABASE_KEY previously; accept as fallback.
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

# Supabase-backed adapters
# ✅ Default to the district view you created: public.schools_by_district
SCHOOLS_SUPABASE_VIEW = os.getenv("SCHOOLS_SUPABASE_VIEW", "schools_by_district").strip()
SCHOOLS_MAX_RESULTS = int(os.getenv("SCHOOLS_MAX_RESULTS", "20"))
SCHOOLS_CONFIDENCE_VALUE = float(os.getenv("SCHOOLS_CONFIDENCE_VALUE", "0.90"))

BROADBAND_SUPABASE_TABLE = os.getenv("BROADBAND_SUPABASE_TABLE", "").strip()  # e.g. "broadband_by_postcode"
BROADBAND_MAX_RESULTS = int(os.getenv("BROADBAND_MAX_RESULTS", "5"))
BROADBAND_CONFIDENCE_VALUE = float(os.getenv("BROADBAND_CONFIDENCE_VALUE", "0.90"))

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


def safe_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
        if f != f:  # NaN
            return None
        return f
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


def _http_post_text(url: str, data: bytes, headers: Optional[dict] = None, timeout: int = 30) -> Tuple[int, str]:
    h = {"User-Agent": HTTP_USER_AGENT}
    if headers:
        h.update(headers)
    r = requests.post(url, data=data, headers=h, timeout=timeout)
    return r.status_code, r.text or ""


# ----------------------------
# NSPL (Supabase view) postcode -> lat/lng
# Requires: public.nspl_lookup(pcd_nospace, lat, lng)
# ----------------------------
def nspl_lookup_latlng(postcode: str) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    """
    Returns (lat, lng, meta). Meta mirrors geocode meta shape used by existing code.
    """
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
node["public_transport"](around:{radius},{lat},{lng});
node["railway"="station"](around:{radius},{lat},{lng});
node["highway"="bus_stop"](around:{radius},{lat},{lng});
"""
    try:
        payload = overpass_query(lat, lng, selectors)
        elements = payload.get("elements", []) if isinstance(payload, dict) else []
        if not isinstance(elements, list):
            elements = []

        counts: Dict[str, int] = {"public_transport": 0, "stations": 0, "bus_stops": 0}
        names: List[str] = []
        for e in elements:
            tags = (e or {}).get("tags") or {}
            if not isinstance(tags, dict):
                continue
            if "public_transport" in tags:
                counts["public_transport"] += 1
            if tags.get("railway") == "station":
                counts["stations"] += 1
            if tags.get("highway") == "bus_stop":
                counts["bus_stops"] += 1
            nm = tags.get("name")
            if isinstance(nm, str) and nm.strip():
                names.append(nm.strip())

        summary = summarise_counts("Transport (OSM within ~1.2km)", counts, top_names=names)
        value = names[: min(MAX_OSM_NAMES, 200)]

        out = metric_ok(
            summary if elements else "No transport features returned from OSM for this area.",
            value,
            base_sources,
            retrieved,
            0.90 if elements else 0.0,
        )
        out["metrics"] = {"radiusMeters": radius, "counts": counts, "totalElements": len(elements)}
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
node["amenity"](around:{radius},{lat},{lng});
node["shop"](around:{radius},{lat},{lng});
node["leisure"](around:{radius},{lat},{lng});
"""
    try:
        payload = overpass_query(lat, lng, selectors)
        elements = payload.get("elements", []) if isinstance(payload, dict) else []
        if not isinstance(elements, list):
            elements = []

        counts: Dict[str, int] = {}
        names: List[str] = []
        for e in elements:
            tags = (e or {}).get("tags") or {}
            if not isinstance(tags, dict):
                continue

            bucket = None
            if "amenity" in tags:
                bucket = f"amenity:{tags.get('amenity')}"
            elif "shop" in tags:
                bucket = f"shop:{tags.get('shop')}"
            elif "leisure" in tags:
                bucket = f"leisure:{tags.get('leisure')}"
            if bucket:
                counts[bucket] = counts.get(bucket, 0) + 1

            nm = tags.get("name")
            if isinstance(nm, str) and nm.strip():
                names.append(nm.strip())

        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
        top_counts = {k: v for k, v in top}

        summary = summarise_counts("Amenities (OSM within ~1.2km)", top_counts, top_names=names)
        value = names[:MAX_OSM_NAMES]

        out = metric_ok(
            summary if elements else "No amenities returned from OSM for this area.",
            value,
            base_sources,
            retrieved,
            0.90 if elements else 0.0,
        )
        out["metrics"] = {"radiusMeters": radius, "topCategories": top_counts, "totalElements": len(elements)}
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

        # ✅ Query the district-indexed view by postcode_district (exact + fast).
        # NOTE: schools_by_district has columns:
        # postcode_district, urn, name, establishment_type, phase, local_authority, town, postcode, status, telephone, website
        try:
            cols = "postcode_district,urn,name,establishment_type,phase,local_authority,town,postcode,status,telephone,website"
            q = (
                supabase.table(SCHOOLS_SUPABASE_VIEW)
                .select(cols)
                .eq("postcode_district", district)
                .limit(SCHOOLS_MAX_RESULTS)
            )
            res = q.execute()
            rows = res.data if hasattr(res, "data") else None
            if not isinstance(rows, list):
                rows = []

            if rows:
                summary = (
                    f"Schools found for postcode district {district}: {len(rows)} "
                    f"(from Supabase view {SCHOOLS_SUPABASE_VIEW})."
                )
                confidence = SCHOOLS_CONFIDENCE_VALUE
                status = "ok"
                needs_evidence = False
            else:
                summary = (
                    f"No schools returned for postcode district {district} "
                    f"(Supabase view {SCHOOLS_SUPABASE_VIEW}). Data coverage may be incomplete."
                )
                confidence = 0.0
                status = "unavailable"
                needs_evidence = True

            sources = [{"label": "Supabase (schools view)", "url": f"{SUPABASE_URL}"}]

            out = metric_ok(summary, rows, sources, retrieved, confidence)
            # Override for empty case: don't lie with "ok" when it's a coverage gap.
            out["status"] = status
            out["needsEvidence"] = needs_evidence
            out["metrics"] = {
                "provider": "supabase",
                "view": SCHOOLS_SUPABASE_VIEW,
                "district": district,
                "limit": SCHOOLS_MAX_RESULTS,
                "queryMode": "eq(postcode_district)",
                "coverageLevel": "district",
            }
            return out

        except Exception as e:
            return metric_unavailable(
                f"Schools Supabase query failed: {str(e)}",
                [{"label": "Supabase", "url": "https://supabase.com/"}],
                retrieved,
                extra_metrics={"postcode": pc, "district": district, "view": SCHOOLS_SUPABASE_VIEW},
            )

    # Default: missing provider
    return metric_missing_provider(
        "Schools provider not configured. Set SCHOOLS_PROVIDER=supabase and SCHOOLS_SUPABASE_VIEW=schools_by_district (or configure an API provider).",
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

            # Fallback: try district prefix if exact postcode not found (optional)
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
                f"Broadband records found for {pc}: {len(rows)} (from Supabase table {BROADBAND_SUPABASE_TABLE})."
                if rows else
                f"No broadband records found for {pc} in Supabase table {BROADBAND_SUPABASE_TABLE}."
            )

            sources = [{"label": "Supabase (broadband table)", "url": f"{SUPABASE_URL}"}]
            out = metric_ok(summary, rows, sources, retrieved, BROADBAND_CONFIDENCE_VALUE if rows else 0.0)
            out["metrics"] = {
                "provider": "supabase",
                "table": BROADBAND_SUPABASE_TABLE,
                "district": district,
                "limit": BROADBAND_MAX_RESULTS,
            }
            return out

        except Exception as e:
            return metric_unavailable(
                f"Broadband Supabase query failed: {str(e)}",
                [{"label": "Supabase", "url": "https://supabase.com/"}],
                retrieved,
                extra_metrics={"postcode": pc, "district": district, "table": BROADBAND_SUPABASE_TABLE},
            )

    return metric_missing_provider(
        "Broadband provider not configured. Set BROADBAND_PROVIDER=supabase and BROADBAND_SUPABASE_TABLE=broadband_by_postcode (or configure a provider).",
        [
            {"label": "Ofcom", "url": "https://www.ofcom.org.uk/"},
            {"label": "ThinkBroadband", "url": "https://www.thinkbroadband.com/"},
        ],
        retrieved,
        extra_metrics={"postcode": pc, "district": district},
    )


# ----------------------------
# Housing (baseline stub + Land Registry sold comps already elsewhere in your codebase)
# ----------------------------
def get_housing_data(postcode: str) -> Dict[str, Any]:
    retrieved = now_iso()
    pc = normalize_postcode(postcode)
    return metric_missing_provider(
        "Housing market provider not configured. Integrate sales/rents/listings API and return evidence-linked comps.",
        [
            {"label": "UK House Price Index (GOV.UK)", "url": "https://www.gov.uk/government/collections/uk-house-price-index-reports"},
            {"label": "HM Land Registry (open data)", "url": "https://landregistry.data.gov.uk/"},
        ],
        retrieved,
        extra_metrics={"postcode": pc} if pc else {},
    )


# ----------------------------
# Route: Adapter endpoints (useful for debugging)
# ----------------------------
@app.route("/adapters/schools", methods=["GET"])
def adapter_schools():
    postcode = normalize_postcode(request.args.get("postcode", "") or "")
    return jsonify(get_schools_data(postcode))


@app.route("/adapters/broadband", methods=["GET"])
def adapter_broadband():
    postcode = normalize_postcode(request.args.get("postcode", "") or "")
    return jsonify(get_broadband_data(postcode))


# ----------------------------
# Route: /market-insights
# ----------------------------
@app.route("/market-insights", methods=["POST"])
def market_insights():
    data = request.get_json(silent=True) or {}
    postcode = normalize_postcode(data.get("postcode", "") or "")
    lat = safe_float(data.get("lat"))
    lng = safe_float(data.get("lng"))

    cache_key = f"market-insights::{postcode}::{lat or ''}::{lng or ''}" if postcode else "market-insights::no-postcode"
    cached = cache_get(cache_key)
    if cached:
        return jsonify({**cached, "_cache": {"hit": True, "ttlSeconds": CACHE_TTL_SECONDS}})

    try:
        geo_meta = None

        # ✅ Prefer NSPL (Supabase) for postcode → coords, fallback to Nominatim only if needed
        if (lat is None or lng is None) and postcode:
            lat2, lng2, nspl_meta = nspl_lookup_latlng(postcode)
            if lat2 is not None and lng2 is not None:
                lat, lng, geo_meta = lat2, lng2, nspl_meta
            else:
                # Fallback to Nominatim if NSPL not available or missing row
                lat, lng, geo_meta = geocode_postcode(postcode)

        # Local area analysis: schools/broadband do NOT depend on geocode now.
        local_area = {
            "retrievedAtISO": now_iso(),
            "postcode": postcode,
            "schools": get_schools_data(postcode),
            "housing": get_housing_data(postcode),
            "transport": get_transport_data(lat, lng),
            "amenities": get_amenities_data(lat, lng),
            "crime": get_crime_data(lat, lng),
            "broadband": get_broadband_data(postcode),
        }

        results = {
            "postcode": postcode,
            "location": {"lat": lat, "lng": lng, "geocodeMeta": geo_meta},
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
            "GET /adapters/schools?postcode=EC3A%205DE": "debug schools adapter",
            "GET /adapters/broadband?postcode=EC3A%205DE": "debug broadband adapter",
        },
        "envHints": {
            "SUPABASE_URL": "required for supabase providers",
            "SUPABASE_SERVICE_ROLE_KEY": "preferred (server-only). SUPABASE_KEY also supported as fallback.",
            "SCHOOLS_PROVIDER": "set to 'supabase' to enable",
            "SCHOOLS_SUPABASE_VIEW": "e.g. schools_by_district (recommended)",
            "BROADBAND_PROVIDER": "set to 'supabase' to enable",
            "BROADBAND_SUPABASE_TABLE": "e.g. broadband_by_postcode",
            "NSPL": "requires view public.nspl_lookup(pcd_nospace, lat, lng) for postcode->coords",
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
