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


def _http_post_text(url: str, data: bytes, headers: Optional[dict] = None, timeout: int = 30) -> Tuple[int, str]:
    h = {"User-Agent": HTTP_USER_AGENT}
    if headers:
        h.update(headers)
    r = requests.post(url, data=data, headers=h, timeout=timeout)
    return r.status_code, r.text or ""


# ----------------------------
# Distance (miles) helper
# ----------------------------
def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # no external deps; stable and fast
    from math import radians, sin, cos, sqrt, atan2
    R = 3958.7613  # earth radius in miles
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


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
def overpass_query(lat: float, lng: float, selectors: str, out_mode: str = "body") -> Dict[str, Any]:
    # out_mode: "body" (fast) or "center" (adds center for ways/relations)
    out_stmt = "out body;" if out_mode == "body" else "out center;"
    q = f"""
[out:json][timeout:25];
(
  {selectors}
);
{out_stmt}
""".strip()

    status, text = _http_post_text(
        "https://overpass-api.de/api/interpreter",
        data=q.encode("utf-8"),
        headers={"Content-Type": "text/plain"},
        timeout=35,
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
    radius_roads = max(2500, min(6000, int(radius * 5)))  # roads need a wider catchment

    selectors = f"""
node["railway"="station"](around:{radius},{lat},{lng});
way["railway"="station"](around:{radius},{lat},{lng});
relation["railway"="station"](around:{radius},{lat},{lng});

node["public_transport"="platform"](around:{radius},{lat},{lng});
node["highway"="bus_stop"](around:{radius},{lat},{lng});

node["highway"="motorway_junction"](around:{radius_roads},{lat},{lng});
way["highway"~"motorway|trunk|primary"](around:{radius_roads},{lat},{lng});
"""

    try:
        payload = overpass_query(lat, lng, selectors, out_mode="center")
        elements = payload.get("elements", []) if isinstance(payload, dict) else []
        if not isinstance(elements, list):
            elements = []

        # Extract useful points (nodes have lat/lon; ways/relations have center.{lat,lon} with out center)
        def elem_latlon(e: dict) -> Optional[Tuple[float, float]]:
            if not isinstance(e, dict):
                return None
            la = safe_float(e.get("lat"))
            lo = safe_float(e.get("lon"))
            if la is not None and lo is not None:
                return la, lo
            c = e.get("center") if isinstance(e.get("center"), dict) else None
            if c:
                la2 = safe_float(c.get("lat"))
                lo2 = safe_float(c.get("lon"))
                if la2 is not None and lo2 is not None:
                    return la2, lo2
            return None

        # Buckets
        stations: List[Tuple[float, str]] = []        # (miles, name)
        bus_stops = 0
        platforms = 0
        motorway_junctions: List[Tuple[float, str]] = []
        roads: Dict[str, set] = {"motorway": set(), "trunk": set(), "primary": set()}

        for e in elements:
            if not isinstance(e, dict):
                continue
            tags = e.get("tags") if isinstance(e.get("tags"), dict) else {}
            ll = elem_latlon(e)
            dist = haversine_miles(lat, lng, ll[0], ll[1]) if ll else None

            if tags.get("railway") == "station":
                name = tags.get("name") or tags.get("ref") or "Station"
                if isinstance(name, str) and dist is not None:
                    stations.append((dist, name.strip()))

            if tags.get("highway") == "bus_stop":
                bus_stops += 1

            if tags.get("public_transport") == "platform":
                platforms += 1

            if tags.get("highway") == "motorway_junction":
                name = tags.get("ref") or tags.get("name") or "Motorway junction"
                if isinstance(name, str) and dist is not None:
                    motorway_junctions.append((dist, name.strip()))

            hw = tags.get("highway")
            if hw in ("motorway", "trunk", "primary"):
                ref = tags.get("ref") or ""
                name = tags.get("name") or ""
                label = ""
                if isinstance(ref, str) and ref.strip():
                    label = ref.strip()
                elif isinstance(name, str) and name.strip():
                    label = name.strip()
                if label:
                    roads[hw].add(label)

        # Curate output (value MUST be string[] so frontend renders bullets nicely)
        lines: List[str] = []

        if stations:
            stations.sort(key=lambda x: x[0])
            nearest = stations[0]
            others = [n for _, n in stations[1:5] if isinstance(n, str)]
            rail_line = f"Rail: nearest station ~{nearest[0]:.2f} mi ({nearest[1]})"
            if others:
                rail_line += f"; also nearby: {', '.join(others[:3])}"
            lines.append(rail_line)

        # Roads (ref/name lists)
        motorways = sorted(list(roads["motorway"]))[:6]
        trunks = sorted(list(roads["trunk"]))[:8]
        primaries = sorted(list(roads["primary"]))[:10]

        if motorways:
            lines.append(f"Road: Motorways: {', '.join(motorways)}")
        if trunks:
            lines.append(f"Road: Trunk routes: {', '.join(trunks)}")
        if primaries:
            # try to avoid huge spam
            lines.append(f"Road: Primary routes: {', '.join(primaries[:6])}{'…' if len(primaries) > 6 else ''}")

        if motorway_junctions:
            motorway_junctions.sort(key=lambda x: x[0])
            j = motorway_junctions[0]
            lines.append(f"Motorway junction: ~{j[0]:.2f} mi ({j[1]})")

        if bus_stops:
            density = "high" if bus_stops >= 40 else "medium" if bus_stops >= 15 else "low"
            lines.append(f"Bus: {bus_stops} stops within ~{int(radius)}m (density: {density}).")

        if platforms and platforms > 0:
            lines.append(f"Public transport platforms: {platforms} within ~{int(radius)}m.")

        counts = {
            "stations": len(stations),
            "bus_stops": bus_stops,
            "platforms": platforms,
            "motorway_junctions": len(motorway_junctions),
        }

        if not lines:
            summary = "No transport features returned from OSM for this area."
            out = metric_ok(summary, [], base_sources, retrieved, 0.0)
            out["metrics"] = {"radiusMeters": radius, "radiusRoadsMeters": radius_roads, "counts": counts, "totalElements": len(elements)}
            return out

        summary = "• " + "\n• ".join(lines)
        out = metric_ok(summary, lines[:12], base_sources, retrieved, 0.90)
        out["metrics"] = {"radiusMeters": radius, "radiusRoadsMeters": radius_roads, "counts": counts, "totalElements": len(elements)}
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
way["amenity"](around:{radius},{lat},{lng});
relation["amenity"](around:{radius},{lat},{lng});

node["shop"](around:{radius},{lat},{lng});
way["shop"](around:{radius},{lat},{lng});
relation["shop"](around:{radius},{lat},{lng});

node["leisure"](around:{radius},{lat},{lng});
way["leisure"](around:{radius},{lat},{lng});
relation["leisure"](around:{radius},{lat},{lng});

node["tourism"](around:{radius},{lat},{lng});
way["tourism"](around:{radius},{lat},{lng});
relation["tourism"](around:{radius},{lat},{lng});
"""

    try:
        payload = overpass_query(lat, lng, selectors, out_mode="center")
        elements = payload.get("elements", []) if isinstance(payload, dict) else []
        if not isinstance(elements, list):
            elements = []

        def elem_latlon(e: dict) -> Optional[Tuple[float, float]]:
            if not isinstance(e, dict):
                return None
            la = safe_float(e.get("lat"))
            lo = safe_float(e.get("lon"))
            if la is not None and lo is not None:
                return la, lo
            c = e.get("center") if isinstance(e.get("center"), dict) else None
            if c:
                la2 = safe_float(c.get("lat"))
                lo2 = safe_float(c.get("lon"))
                if la2 is not None and lo2 is not None:
                    return la2, lo2
            return None

        # Bucket mapping (high-signal, human readable)
        FOOD_AMENITY = {"restaurant", "cafe", "bar", "pub", "fast_food", "food_court", "biergarten"}
        HEALTH_AMENITY = {"pharmacy", "hospital", "doctors", "dentist", "clinic", "veterinary"}
        SERVICE_AMENITY = {"bank", "atm", "post_office", "library", "community_centre", "townhall"}
        LEISURE_KEYS = {"park", "fitness_centre", "sports_centre", "pitch", "playground", "cinema", "theatre"}

        # Hold nearest named examples per bucket
        buckets: Dict[str, List[Tuple[float, str]]] = {
            "Food & drink": [],
            "Shopping": [],
            "Healthcare": [],
            "Services": [],
            "Leisure": [],
            "Other": [],
        }
        counts: Dict[str, int] = {k: 0 for k in buckets.keys()}

        # Track "essentials" distances if present
        nearest_supermarket: Optional[Tuple[float, str]] = None
        nearest_convenience: Optional[Tuple[float, str]] = None
        nearest_pharmacy: Optional[Tuple[float, str]] = None

        for e in elements:
            if not isinstance(e, dict):
                continue
            tags = e.get("tags") if isinstance(e.get("tags"), dict) else {}
            if not isinstance(tags, dict):
                continue

            ll = elem_latlon(e)
            dist = haversine_miles(lat, lng, ll[0], ll[1]) if ll else None

            amenity = tags.get("amenity")
            shop = tags.get("shop")
            leisure = tags.get("leisure")
            tourism = tags.get("tourism")
            name = tags.get("name") or tags.get("brand") or ""

            # decide bucket
            bucket = "Other"
            if isinstance(amenity, str) and amenity in FOOD_AMENITY:
                bucket = "Food & drink"
            elif isinstance(amenity, str) and amenity in HEALTH_AMENITY:
                bucket = "Healthcare"
            elif isinstance(amenity, str) and amenity in SERVICE_AMENITY:
                bucket = "Services"
            elif isinstance(leisure, str) and leisure in LEISURE_KEYS:
                bucket = "Leisure"
            elif isinstance(shop, str) and shop.strip():
                bucket = "Shopping"
            elif isinstance(amenity, str) and amenity.strip():
                # non-mapped amenity still usually useful
                bucket = "Services" if amenity in {"police", "fire_station"} else "Other"
            elif isinstance(tourism, str) and tourism.strip():
                bucket = "Leisure"

            counts[bucket] = counts.get(bucket, 0) + 1

            if isinstance(name, str) and name.strip() and dist is not None:
                buckets[bucket].append((dist, name.strip()))

            # essentials tracking
            if isinstance(shop, str) and dist is not None:
                if shop == "supermarket" and isinstance(name, str) and name.strip():
                    cand = (dist, name.strip())
                    if nearest_supermarket is None or cand[0] < nearest_supermarket[0]:
                        nearest_supermarket = cand
                if shop == "convenience" and isinstance(name, str) and name.strip():
                    cand = (dist, name.strip())
                    if nearest_convenience is None or cand[0] < nearest_convenience[0]:
                        nearest_convenience = cand

            if isinstance(amenity, str) and amenity == "pharmacy" and dist is not None and isinstance(name, str) and name.strip():
                cand = (dist, name.strip())
                if nearest_pharmacy is None or cand[0] < nearest_pharmacy[0]:
                    nearest_pharmacy = cand

        total = sum(counts.values())

        if total == 0:
            out = metric_ok("No amenities returned from OSM for this area.", [], base_sources, retrieved, 0.0)
            out["metrics"] = {"radiusMeters": radius, "buckets": counts, "total": 0}
            return out

        # Build high-end, human output (string[] so UI renders bullets, not JSON)
        def top_names_for(bucket_name: str, n: int = 4) -> List[str]:
            xs = buckets.get(bucket_name, [])
            xs.sort(key=lambda x: x[0])
            outn: List[str] = []
            for d, nm in xs[:n]:
                outn.append(f"{nm} (~{d:.2f} mi)")
            return outn

        lines: List[str] = []
        # headline essentials
        essentials: List[str] = []
        if nearest_supermarket:
            essentials.append(f"Nearest supermarket: {nearest_supermarket[1]} (~{nearest_supermarket[0]:.2f} mi)")
        if nearest_convenience:
            essentials.append(f"Nearest convenience: {nearest_convenience[1]} (~{nearest_convenience[0]:.2f} mi)")
        if nearest_pharmacy:
            essentials.append(f"Nearest pharmacy: {nearest_pharmacy[1]} (~{nearest_pharmacy[0]:.2f} mi)")
        if essentials:
            lines.extend(essentials)

        # bucket summaries
        order = ["Food & drink", "Shopping", "Healthcare", "Services", "Leisure", "Other"]
        for b in order:
            c = counts.get(b, 0)
            if c <= 0:
                continue
            tops = top_names_for(b, 4)
            if tops:
                lines.append(f"{b}: {c} (e.g., {', '.join(tops)})")
            else:
                lines.append(f"{b}: {c}")

        summary = "• " + "\n• ".join(lines[:12])
        out = metric_ok(summary, lines[:12], base_sources, retrieved, 0.90)
        out["metrics"] = {"radiusMeters": radius, "buckets": counts, "total": total}
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

    # Bounds
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

    # RPC contract expected:
    # public.housing_comps_v1(postcode text, radius_miles numeric, limit_n int)
    # returns rows: date_of_transfer, price, property_type, street, town_city, postcode, miles
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
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
