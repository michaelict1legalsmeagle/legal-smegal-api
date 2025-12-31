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


# ======================================================
# AMENDED: Transport (high-end, aggregated, not street list)
# ======================================================
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

    # ---- scoped helpers (keep change local) ----
    def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        # Earth radius (miles)
        R = 3958.7613
        from math import radians, sin, cos, asin, sqrt

        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        return R * c

    def _overpass_center(selectors: str, timeout_sec: int = 40) -> Dict[str, Any]:
        # out center => ways get a representative point
        q = f"""
[out:json][timeout:25];
(
  {selectors}
);
out center;
""".strip()

        status, text = _http_post_text(
            "https://overpass-api.de/api/interpreter",
            data=q.encode("utf-8"),
            headers={"Content-Type": "text/plain"},
            timeout=timeout_sec,
        )
        if status != 200 or not text:
            return {"elements": []}
        try:
            payload = json.loads(text)
            return payload if isinstance(payload, dict) else {"elements": []}
        except Exception:
            return {"elements": []}

    def _elem_point(e: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        la = safe_float(e.get("lat"))
        lo = safe_float(e.get("lon"))
        if la is not None and lo is not None:
            return la, lo
        c = e.get("center")
        if isinstance(c, dict):
            la2 = safe_float(c.get("lat"))
            lo2 = safe_float(c.get("lon"))
            if la2 is not None and lo2 is not None:
                return la2, lo2
        return None

    def _dedupe_keep_order(items: List[str], limit: int) -> List[str]:
        seen = set()
        out: List[str] = []
        for s in items:
            t = (s or "").strip()
            if not t:
                continue
            k = t.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(t)
            if len(out) >= limit:
                break
        return out

    # Query radii tuned for “property report usefulness”
    bus_radius_m = DEFAULT_OSM_RADIUS              # density proxy
    rail_radius_m = max(DEFAULT_OSM_RADIUS, 6000)  # station catchment
    road_radius_m = 10000                          # major roads often further away

    try:
        # BUS + RAIL (nodes)
        selectors_nodes = f"""
node["railway"="station"](around:{rail_radius_m},{lat},{lng});
node["highway"="bus_stop"](around:{bus_radius_m},{lat},{lng});
"""
        payload_nodes = overpass_query(lat, lng, selectors_nodes)
        elems_nodes = payload_nodes.get("elements", []) if isinstance(payload_nodes, dict) else []
        if not isinstance(elems_nodes, list):
            elems_nodes = []

        bus_stop_count = 0
        rail_stations: List[Dict[str, Any]] = []

        for e in elems_nodes:
            if not isinstance(e, dict):
                continue
            tags = e.get("tags") or {}
            if not isinstance(tags, dict):
                continue

            if tags.get("highway") == "bus_stop":
                bus_stop_count += 1

            if tags.get("railway") == "station":
                pt = _elem_point(e)
                miles = _haversine_miles(lat, lng, pt[0], pt[1]) if pt else None
                rail_stations.append(
                    {
                        "name": (tags.get("name") or "").strip() or "Rail station",
                        "operator": (tags.get("operator") or "").strip() or None,
                        "network": (tags.get("network") or "").strip() or None,
                        "miles": miles,
                    }
                )

        rail_stations = sorted(
            rail_stations,
            key=lambda r: (r["miles"] if isinstance(r.get("miles"), (int, float)) else 10**9),
        )
        rail_top = rail_stations[:3]
        rail_nearest = rail_top[0]["miles"] if rail_top and isinstance(rail_top[0].get("miles"), (int, float)) else None

        # ROADS (ways + out center)
        selectors_roads = f"""
way["highway"~"motorway|trunk|primary"](around:{road_radius_m},{lat},{lng});
"""
        payload_roads = _overpass_center(selectors_roads, timeout_sec=45)
        elems_roads = payload_roads.get("elements", []) if isinstance(payload_roads, dict) else []
        if not isinstance(elems_roads, list):
            elems_roads = []

        motorway_refs: List[str] = []
        trunk_refs: List[str] = []
        primary_refs: List[str] = []
        road_candidates: List[Dict[str, Any]] = []

        for e in elems_roads:
            if not isinstance(e, dict):
                continue
            tags = e.get("tags") or {}
            if not isinstance(tags, dict):
                continue

            hw = (tags.get("highway") or "").strip()
            ref = (tags.get("ref") or "").strip()
            name = (tags.get("name") or "").strip()
            label = ref or name

            if label:
                if hw == "motorway":
                    motorway_refs.append(label)
                elif hw == "trunk":
                    trunk_refs.append(label)
                elif hw == "primary":
                    primary_refs.append(label)

            pt = _elem_point(e)
            miles = _haversine_miles(lat, lng, pt[0], pt[1]) if pt else None
            if label and isinstance(miles, (int, float)):
                road_candidates.append({"label": label, "highway": hw, "miles": miles})

        road_candidates = sorted(road_candidates, key=lambda r: r["miles"])
        nearest_major_road_miles = road_candidates[0]["miles"] if road_candidates else None

        motorway_top = _dedupe_keep_order(motorway_refs, 3)
        trunk_top = _dedupe_keep_order(trunk_refs, 3)
        primary_top = _dedupe_keep_order(primary_refs, 3)

        # bus density heuristic (readable, not “map junk”)
        if bus_stop_count >= 40:
            bus_density = "high"
        elif bus_stop_count >= 15:
            bus_density = "medium"
        elif bus_stop_count > 0:
            bus_density = "low"
        else:
            bus_density = "none"

        summary_bullets: List[str] = []

        if rail_nearest is not None:
            summary_bullets.append(
                f"Rail: nearest station ~{rail_nearest:.2f} mi; top: {', '.join([r['name'] for r in rail_top])}"
            )
        else:
            summary_bullets.append("Rail: no stations detected within catchment radius (OSM).")

        roads_bits: List[str] = []
        if motorway_top:
            roads_bits.append("Motorways: " + ", ".join(motorway_top))
        if trunk_top:
            roads_bits.append("Trunk: " + ", ".join(trunk_top))
        if primary_top:
            roads_bits.append("Primary: " + ", ".join(primary_top))

        if roads_bits:
            suffix = f" (nearest major road ~{nearest_major_road_miles:.2f} mi)" if isinstance(nearest_major_road_miles, (int, float)) else ""
            summary_bullets.append("Road: " + " • ".join(roads_bits) + suffix)
        else:
            summary_bullets.append("Road: no major roads detected within ~10km (OSM).")

        summary_bullets.append(f"Bus: {bus_stop_count} stops within ~{int(bus_radius_m)}m (density: {bus_density}).")

        value_obj = {
            "rail": {
                "radiusMeters": int(rail_radius_m),
                "nearestMiles": rail_nearest,
                "topStations": rail_top,
                "stationCount": len(rail_stations),
            },
            "road": {
                "radiusMeters": int(road_radius_m),
                "nearestMiles": nearest_major_road_miles,
                "motorways": motorway_top,
                "trunkRoads": trunk_top,
                "primaryRoads": primary_top,
            },
            "bus": {
                "radiusMeters": int(bus_radius_m),
                "stopCount": bus_stop_count,
                "density": bus_density,
            },
            "summaryBullets": summary_bullets,
        }

        summary_text = "\n".join([f"• {b}" for b in summary_bullets])

        any_signal = (bus_stop_count > 0) or (len(rail_stations) > 0) or (len(elems_roads) > 0)
        conf = 0.90 if any_signal else 0.0

        out = metric_ok(
            summary_text if any_signal else "No transport features returned from OSM for this area.",
            value_obj,
            base_sources,
            retrieved,
            conf,
        )
        out["metrics"] = {
            "busRadiusMeters": int(bus_radius_m),
            "railRadiusMeters": int(rail_radius_m),
            "roadRadiusMeters": int(road_radius_m),
            "busStopCount": bus_stop_count,
            "railStationCount": len(rail_stations),
            "roadElements": len(elems_roads),
        }
        return out

    except Exception as e:
        return metric_unavailable(
            f"Transport data fetch failed: {str(e)}",
            base_sources,
            retrieved,
        )


# ======================================================
# AMENDED: Amenities (high-end buckets + examples, not street list)
# ======================================================
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

    def _dedupe_keep_order(items: List[str], limit: int) -> List[str]:
        seen = set()
        out: List[str] = []
        for s in items:
            t = (s or "").strip()
            if not t:
                continue
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
            if len(out) >= limit:
                break
        return out

    radius = DEFAULT_OSM_RADIUS

    # Simple, property-report buckets
    SHOPPING_SHOPS = {
        "supermarket", "convenience", "bakery", "butcher", "greengrocer",
        "clothes", "shoes", "department_store"
    }
    HEALTH_AMENITIES = {"pharmacy", "doctors", "dentist", "hospital", "clinic", "veterinary"}
    FOOD_AMENITIES = {"restaurant", "cafe", "fast_food", "pub", "bar"}
    SERVICES_AMENITIES = {"bank", "post_office", "atm", "police", "fire_station", "library"}
    LEISURE_LEISURE = {"park", "fitness_centre", "sports_centre", "pitch", "playground"}

    try:
        selectors = f"""
node["amenity"](around:{radius},{lat},{lng});
node["shop"](around:{radius},{lat},{lng});
node["leisure"](around:{radius},{lat},{lng});
"""
        payload = overpass_query(lat, lng, selectors)
        elements = payload.get("elements", []) if isinstance(payload, dict) else []
        if not isinstance(elements, list):
            elements = []

        buckets = {
            "shopping": {"count": 0, "top": []},
            "foodDrink": {"count": 0, "top": []},
            "healthcare": {"count": 0, "top": []},
            "leisure": {"count": 0, "top": []},
            "services": {"count": 0, "top": []},
            "other": {"count": 0, "top": []},
        }
        names_by_bucket: Dict[str, List[str]] = {k: [] for k in buckets.keys()}

        for e in elements:
            if not isinstance(e, dict):
                continue
            tags = e.get("tags") or {}
            if not isinstance(tags, dict):
                continue

            amenity = (tags.get("amenity") or "").strip()
            shop = (tags.get("shop") or "").strip()
            leisure = (tags.get("leisure") or "").strip()
            name = (tags.get("name") or "").strip()

            bucket_key = "other"

            if shop:
                bucket_key = "shopping"
            elif amenity:
                if amenity in HEALTH_AMENITIES:
                    bucket_key = "healthcare"
                elif amenity in FOOD_AMENITIES:
                    bucket_key = "foodDrink"
                elif amenity in SERVICES_AMENITIES:
                    bucket_key = "services"
                else:
                    bucket_key = "other"
            elif leisure:
                bucket_key = "leisure"

            if bucket_key not in buckets:
                bucket_key = "other"

            buckets[bucket_key]["count"] += 1
            if name:
                names_by_bucket[bucket_key].append(name)

        for k in buckets.keys():
            buckets[k]["top"] = _dedupe_keep_order(names_by_bucket.get(k, []), 5)

        total = sum(v["count"] for v in buckets.values())

        label_map = {
            "shopping": "Shopping",
            "foodDrink": "Food & drink",
            "healthcare": "Healthcare",
            "leisure": "Leisure",
            "services": "Services",
            "other": "Other",
        }
        ordered = ["shopping", "foodDrink", "healthcare", "leisure", "services", "other"]

        summary_bullets: List[str] = []
        for k in ordered:
            c = buckets[k]["count"]
            if c <= 0:
                continue
            top = buckets[k]["top"]
            label = label_map.get(k, k)
            if top:
                summary_bullets.append(f"{label}: {c} (e.g., {', '.join(top[:3])}).")
            else:
                summary_bullets.append(f"{label}: {c}.")

        value_obj = {
            "radiusMeters": int(radius),
            "total": total,
            "buckets": buckets,
            "summaryBullets": summary_bullets,
        }

        summary_text = "\n".join([f"• {b}" for b in summary_bullets]) if summary_bullets else "No amenities returned from OSM for this area."

        conf = 0.90 if total > 0 else 0.0
        out = metric_ok(
            summary_text,
            value_obj,
            base_sources,
            retrieved,
            conf,
        )
        out["metrics"] = {
            "radiusMeters": int(radius),
            "totalElements": len(elements),
            "totalAmenities": total,
            "bucketCounts": {k: v["count"] for k, v in buckets.items()},
        }
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
