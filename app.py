from flask import Flask, request, jsonify
import requests
import os
import time
from flask_cors import CORS
from supabase import create_client, Client
from typing import Dict, Any, Optional, Tuple

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Simple in-memory cache (postcode-keyed) with TTL
_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SECONDS = int(os.getenv("MARKET_INSIGHTS_CACHE_TTL_SECONDS", "21600"))  # default 6 hours

# Identify ourselves to public services (Nominatim requires a proper UA)
HTTP_USER_AGENT = os.getenv(
    "HTTP_USER_AGENT",
    "LegalSmegal/1.0 (market-insights; contact=admin@example.com)"
)

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("🟢 Supabase enabled. Key prefix:", SUPABASE_KEY[:20])
else:
    print("🔴 Supabase env vars not set. Supabase features (save-analysis, logs) are DISABLED locally.")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def normalize_postcode(pc: str) -> str:
    if not isinstance(pc, str):
        return ""
    return " ".join(pc.strip().upper().split())


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


def geocode_postcode(postcode: str) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    """
    Postcode -> lat/lng via Nominatim (OpenStreetMap).
    We DO NOT invent coords. If geocode fails, lat/lng remain None.
    """
    pc = normalize_postcode(postcode)
    meta = {
        "retrievedAtISO": now_iso(),
        "sources": [
            {"label": "OpenStreetMap Nominatim", "url": "https://nominatim.openstreetmap.org/"}
        ],
        "referenceLinks": [],
        "notes": "",
    }

    if not pc:
        meta["notes"] = "No postcode provided."
        return None, None, meta

    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": pc, "format": "json", "limit": 1, "addressdetails": 0}
        resp = requests.get(url, params=params, headers={"User-Agent": HTTP_USER_AGENT}, timeout=15)
        if resp.status_code != 200:
            meta["notes"] = f"Nominatim error: HTTP {resp.status_code}"
            return None, None, meta

        payload = resp.json() if resp.text else []
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
    if lat is None or lng is None:
        return {
            "summary": "Crime data not available: postcode could not be geocoded to coordinates.",
            "metrics": {},
            "sources": [{"label": "UK Police Data API", "url": "https://data.police.uk/docs/"}],
            "retrievedAtISO": retrieved,
            "confidenceValue": 0.0,
            "needsEvidence": True,
        }

    url = f"https://data.police.uk/api/crimes-street/all-crime?lat={lat}&lng={lng}"
    try:
        resp = requests.get(url, headers={"User-Agent": HTTP_USER_AGENT}, timeout=20)
        crimes = resp.json() if resp.status_code == 200 else []
        if not isinstance(crimes, list):
            crimes = []

        # Count categories
        counts: Dict[str, int] = {}
        for c in crimes:
            cat = (c or {}).get("category") or "unknown"
            counts[cat] = counts.get(cat, 0) + 1

        # A simple summary that is factual: counts only, no claims.
        summary = summarise_counts("Crimes (street-level)", counts)

        return {
            "summary": summary if crimes else "No crime records returned for this location/time window.",
            "metrics": {
                "total": len(crimes),
                "categories": counts,
                "radius_hint": "Police API uses a fixed area around the point; see documentation.",
            },
            "sources": [
                {"label": "UK Police Data API (crimes-street)", "url": url},
                {"label": "UK Police Data API docs", "url": "https://data.police.uk/docs/"},
            ],
            "retrievedAtISO": retrieved,
            "confidenceValue": 0.95 if len(crimes) > 0 else 0.0,
            "needsEvidence": False if len(crimes) > 0 else True,
        }
    except Exception as e:
        return {
            "summary": f"Crime data fetch failed: {str(e)}",
            "metrics": {},
            "sources": [{"label": "UK Police Data API", "url": "https://data.police.uk/docs/"}],
            "retrievedAtISO": retrieved,
            "confidenceValue": 0.0,
            "needsEvidence": True,
        }


def overpass_query(lat: float, lng: float, radius_m: int, selectors: str) -> Dict[str, Any]:
    q = f"""
[out:json];
(
  {selectors}
);
out body;
"""
    r = requests.post(
        "https://overpass-api.de/api/interpreter",
        data=q.encode("utf-8"),
        headers={"User-Agent": HTTP_USER_AGENT},
        timeout=30,
    )
    return r.json() if r.status_code == 200 else {"elements": []}


def get_transport_data(lat: Optional[float], lng: Optional[float]) -> Dict[str, Any]:
    retrieved = now_iso()
    if lat is None or lng is None:
        return {
            "summary": "Transport data not available: postcode could not be geocoded to coordinates.",
            "metrics": {},
            "sources": [
                {"label": "OpenStreetMap (Overpass)", "url": "https://overpass-api.de/"},
                {"label": "OpenStreetMap", "url": "https://www.openstreetmap.org"},
            ],
            "retrievedAtISO": retrieved,
            "confidenceValue": 0.0,
            "needsEvidence": True,
        }

    radius = 1200
    selectors = f"""
node["public_transport"](around:{radius},{lat},{lng});
node["railway"="station"](around:{radius},{lat},{lng});
node["highway"="bus_stop"](around:{radius},{lat},{lng});
"""
    try:
        payload = overpass_query(lat, lng, radius, selectors)
        elements = payload.get("elements", []) if isinstance(payload, dict) else []
        if not isinstance(elements, list):
            elements = []

        counts: Dict[str, int] = {"public_transport": 0, "stations": 0, "bus_stops": 0}
        names = []
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

        return {
            "summary": summary if elements else "No transport features returned from OSM for this area.",
            "metrics": {"radiusMeters": radius, "counts": counts},
            "sources": [
                {"label": "OpenStreetMap (Overpass API)", "url": "https://overpass-api.de/"},
                {"label": "OpenStreetMap", "url": "https://www.openstreetmap.org"},
            ],
            "retrievedAtISO": retrieved,
            "confidenceValue": 0.90 if elements else 0.0,
            "needsEvidence": False if elements else True,
        }
    except Exception as e:
        return {
            "summary": f"Transport data fetch failed: {str(e)}",
            "metrics": {},
            "sources": [
                {"label": "OpenStreetMap (Overpass)", "url": "https://overpass-api.de/"},
                {"label": "OpenStreetMap", "url": "https://www.openstreetmap.org"},
            ],
            "retrievedAtISO": retrieved,
            "confidenceValue": 0.0,
            "needsEvidence": True,
        }


def get_amenities_data(lat: Optional[float], lng: Optional[float]) -> Dict[str, Any]:
    retrieved = now_iso()
    if lat is None or lng is None:
        return {
            "summary": "Amenities data not available: postcode could not be geocoded to coordinates.",
            "metrics": {},
            "sources": [
                {"label": "OpenStreetMap (Overpass)", "url": "https://overpass-api.de/"},
                {"label": "OpenStreetMap", "url": "https://www.openstreetmap.org"},
            ],
            "retrievedAtISO": retrieved,
            "confidenceValue": 0.0,
            "needsEvidence": True,
        }

    radius = 1200
    selectors = f"""
node["amenity"](around:{radius},{lat},{lng});
node["shop"](around:{radius},{lat},{lng});
node["leisure"](around:{radius},{lat},{lng});
"""
    try:
        payload = overpass_query(lat, lng, radius, selectors)
        elements = payload.get("elements", []) if isinstance(payload, dict) else []
        if not isinstance(elements, list):
            elements = []

        counts: Dict[str, int] = {}
        names = []
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

        # Compress counts into top categories
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
        top_counts = {k: v for k, v in top}

        summary = summarise_counts("Amenities (OSM within ~1.2km)", top_counts, top_names=names)

        return {
            "summary": summary if elements else "No amenities returned from OSM for this area.",
            "metrics": {"radiusMeters": radius, "topCategories": top_counts, "totalElements": len(elements)},
            "sources": [
                {"label": "OpenStreetMap (Overpass API)", "url": "https://overpass-api.de/"},
                {"label": "OpenStreetMap", "url": "https://www.openstreetmap.org"},
            ],
            "retrievedAtISO": retrieved,
            "confidenceValue": 0.90 if elements else 0.0,
            "needsEvidence": False if elements else True,
        }
    except Exception as e:
        return {
            "summary": f"Amenities data fetch failed: {str(e)}",
            "metrics": {},
            "sources": [
                {"label": "OpenStreetMap (Overpass)", "url": "https://overpass-api.de/"},
                {"label": "OpenStreetMap", "url": "https://www.openstreetmap.org"},
            ],
            "retrievedAtISO": retrieved,
            "confidenceValue": 0.0,
            "needsEvidence": True,
        }


def get_schools_data(postcode: str) -> Dict[str, Any]:
    retrieved = now_iso()
    pc = normalize_postcode(postcode)
    # No invented schools. Provider integration required.
    return {
        "summary": "Schools data provider not configured. Add an integration (e.g., DfE / Ofsted datasets or a commercial provider) to populate current nearby schools.",
        "metrics": {"postcode": pc} if pc else {},
        "sources": [
            {"label": "Ofsted reports", "url": "https://reports.ofsted.gov.uk/"},
            {"label": "DfE Find and Compare Schools", "url": "https://www.compare-school-performance.service.gov.uk/"},
        ],
        "retrievedAtISO": retrieved,
        "confidenceValue": 0.0,
        "needsEvidence": True,
    }


def get_broadband_data(postcode: str) -> Dict[str, Any]:
    retrieved = now_iso()
    pc = normalize_postcode(postcode)
    # No invented broadband. Provider integration required.
    return {
        "summary": "Broadband data provider not configured. Add an integration (e.g., ThinkBroadband/Ofcom datasets or a commercial checker) to populate current speeds and availability.",
        "metrics": {"postcode": pc} if pc else {},
        "sources": [
            {"label": "Ofcom", "url": "https://www.ofcom.org.uk/"},
            {"label": "ThinkBroadband", "url": "https://www.thinkbroadband.com/"},
        ],
        "retrievedAtISO": retrieved,
        "confidenceValue": 0.0,
        "needsEvidence": True,
    }


def get_housing_data(postcode: str) -> Dict[str, Any]:
    retrieved = now_iso()
    pc = normalize_postcode(postcode)
    # No invented housing figures. Provider integration required for "current market" signals.
    return {
        "summary": "Housing data provider not configured. For investor-grade 'current market' signals, integrate a property data API (sales/rents/listings) and return evidence-linked comps.",
        "metrics": {"postcode": pc} if pc else {},
        "sources": [
            {"label": "UK House Price Index (GOV.UK)", "url": "https://www.gov.uk/government/collections/uk-house-price-index-reports"},
            {"label": "HM Land Registry", "url": "https://landregistry.data.gov.uk/"},
        ],
        "retrievedAtISO": retrieved,
        "confidenceValue": 0.0,
        "needsEvidence": True,
    }


def get_zoopla_comps(postcode: str) -> Any:
    """
    No fake comps. Until a real provider is configured, return an empty list.
    Frontend expects an array of comparable items (or []), not a made-up pack.
    """
    pc = normalize_postcode(postcode)
    return []


@app.route("/market-insights", methods=["POST"])
def market_insights():
    data = request.get_json() or {}
    postcode = normalize_postcode(data.get("postcode", "") or "")

    # Accept optional lat/lng. If missing, geocode from postcode.
    lat = safe_float(data.get("lat"))
    lng = safe_float(data.get("lng"))

    cache_key = f"market-insights::{postcode}" if postcode else "market-insights::no-postcode"
    cached = cache_get(cache_key)
    if cached:
        return jsonify({**cached, "_cache": {"hit": True, "ttlSeconds": CACHE_TTL_SECONDS}})

    try:
        geo_meta = None
        if (lat is None or lng is None) and postcode:
            lat, lng, geo_meta = geocode_postcode(postcode)

        local_area = {
            "schools": get_schools_data(postcode),
            "housing": get_housing_data(postcode),
            "transport": get_transport_data(lat, lng),
            "amenities": get_amenities_data(lat, lng),
            "crime": get_crime_data(lat, lng),
            "broadband": get_broadband_data(postcode),
        }

        results = {
            "postcode": postcode,
            "location": {
                "lat": lat,
                "lng": lng,
                "geocodeMeta": geo_meta,
            },
            "localAreaAnalysis": local_area,
            "comparableProperties": get_zoopla_comps(postcode),
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
            "POST /market-insights": "{ 'postcode': 'B1 1AA' }  // optional: lat/lng"
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
