# app.py

from flask import Flask, request, jsonify
import requests
import os
import time
import json
from flask_cors import CORS
from supabase import create_client, Client
from typing import Dict, Any, Optional, Tuple, List

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

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

# Provider toggles
BROADBAND_PROVIDER = os.getenv("BROADBAND_PROVIDER", "").strip().lower()  # e.g. "thinkbroadband"
SCHOOLS_PROVIDER = os.getenv("SCHOOLS_PROVIDER", "").strip().lower()      # e.g. "dfe"
HOUSING_PROVIDER = os.getenv("HOUSING_PROVIDER", "").strip().lower()      # e.g. "landregistry_only"

# Generic provider config (no assumptions)
BROADBAND_API_URL = os.getenv("BROADBAND_API_URL", "").strip()            # you supply
BROADBAND_API_KEY = os.getenv("BROADBAND_API_KEY", "").strip()            # you supply
SCHOOLS_API_URL = os.getenv("SCHOOLS_API_URL", "").strip()                # you supply
SCHOOLS_API_KEY = os.getenv("SCHOOLS_API_KEY", "").strip()                # you supply

# Optional: stricter confidence
MIN_VERIFIED = float(os.getenv("MIN_VERIFIED_CONFIDENCE", "0.95"))

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("🟢 Supabase enabled. Key prefix:", SUPABASE_KEY[:20])
else:
    print("🔴 Supabase env vars not set. Supabase features (save-analysis, logs) are DISABLED locally.")


# ----------------------------
# Helpers
# ----------------------------
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


def _first_source_url(sources: Any) -> str:
    """Compatibility helper for frontend that expects sourceUrl."""
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
# Geocoding (Nominatim)
# ----------------------------
def geocode_postcode(postcode: str) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    """
    Postcode -> lat/lng via Nominatim (OpenStreetMap).
    We DO NOT invent coords. If geocode fails, lat/lng remain None.
    """
    pc = normalize_postcode(postcode)
    meta = {
        "retrievedAtISO": now_iso(),
        "sources": [{"label": "OpenStreetMap Nominatim", "url": "https://nominatim.openstreetmap.org/"}],
        "referenceLinks": [],
        "notes": "",
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
            "Crime data not available: postcode could not be geocoded to coordinates.",
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
            "Transport data not available: postcode could not be geocoded to coordinates.",
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
            0.90 if elements else 0.0,  # honest: OSM completeness varies
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
            "Amenities data not available: postcode could not be geocoded to coordinates.",
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
# Schools (provider-ready)
# ----------------------------
def get_schools_data(postcode: str) -> Dict[str, Any]:
    retrieved = now_iso()
    pc = normalize_postcode(postcode)

    # Provider-ready pattern:
    # - if configured: fetch real data, return metric_ok with ≥0.95 only when objective dataset is used
    # - if not configured: return missing_provider (no lies)

    if SCHOOLS_PROVIDER and SCHOOLS_API_URL and SCHOOLS_API_KEY and pc:
        try:
            # You define your provider contract; we fail-closed.
            # Expected to return something like:
            # {
            #   "schools": [...],
            #   "summary": "...",
            #   "metrics": {...}
            # }
            status, payload = _http_get_json(
                SCHOOLS_API_URL,
                params={"postcode": pc},
                headers={"Authorization": f"Bearer {SCHOOLS_API_KEY}"},
                timeout=20,
            )
            if status == 200 and isinstance(payload, dict):
                summary = str(payload.get("summary") or "").strip()
                schools = payload.get("schools")
                metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
                sources = payload.get("sources") if isinstance(payload.get("sources"), list) else [
                    {"label": "Schools provider", "url": SCHOOLS_API_URL}
                ]

                out = metric_ok(
                    summary or "Schools data returned by provider.",
                    schools if schools is not None else [],
                    sources,
                    retrieved,
                    MIN_VERIFIED,
                )
                out["metrics"] = metrics
                return out
        except Exception as e:
            return metric_unavailable(
                f"Schools provider call failed: {str(e)}",
                [{"label": "DfE Find and Compare Schools", "url": "https://www.compare-school-performance.service.gov.uk/"}],
                retrieved,
                extra_metrics={"postcode": pc} if pc else {},
            )

    # Default: missing provider
    return metric_missing_provider(
        "Schools data provider not configured. Add an integration (DfE/Ofsted dataset or a commercial provider) to populate nearby schools.",
        [
            {"label": "Ofsted reports", "url": "https://reports.ofsted.gov.uk/"},
            {"label": "DfE Find and Compare Schools", "url": "https://www.compare-school-performance.service.gov.uk/"},
        ],
        retrieved,
        extra_metrics={"postcode": pc} if pc else {},
    )


# ----------------------------
# Broadband (provider-ready)
# ----------------------------
def get_broadband_data(postcode: str) -> Dict[str, Any]:
    retrieved = now_iso()
    pc = normalize_postcode(postcode)

    # Provider-ready pattern (same as schools).
    # You plug in a provider endpoint that returns objective measures.

    if BROADBAND_PROVIDER and BROADBAND_API_URL and BROADBAND_API_KEY and pc:
        try:
            status, payload = _http_get_json(
                BROADBAND_API_URL,
                params={"postcode": pc},
                headers={"Authorization": f"Bearer {BROADBAND_API_KEY}"},
                timeout=20,
            )
            if status == 200 and isinstance(payload, dict):
                # Recommended provider contract:
                # payload = {
                #   "summary": "Average download 74 Mbps, upload 18 Mbps. Full fibre availability 62%.",
                #   "broadband": { "downloadMbps": 74, "uploadMbps": 18, "fullFibrePct": 62, "providers": [...] },
                #   "metrics": {...},
                #   "sources": [{label,url},...]
                # }
                summary = str(payload.get("summary") or "").strip()
                broadband = payload.get("broadband")
                metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
                sources = payload.get("sources") if isinstance(payload.get("sources"), list) else [
                    {"label": "Broadband provider", "url": BROADBAND_API_URL}
                ]

                out = metric_ok(
                    summary or "Broadband data returned by provider.",
                    broadband if broadband is not None else payload,
                    sources,
                    retrieved,
                    MIN_VERIFIED,
                )
                out["metrics"] = metrics
                return out
        except Exception as e:
            return metric_unavailable(
                f"Broadband provider call failed: {str(e)}",
                [{"label": "Ofcom", "url": "https://www.ofcom.org.uk/"}],
                retrieved,
                extra_metrics={"postcode": pc} if pc else {},
            )

    return metric_missing_provider(
        "Broadband data provider not configured. Add an integration (Ofcom/ThinkBroadband dataset or a commercial checker) to populate speeds and availability.",
        [
            {"label": "Ofcom", "url": "https://www.ofcom.org.uk/"},
            {"label": "ThinkBroadband", "url": "https://www.thinkbroadband.com/"},
        ],
        retrieved,
        extra_metrics={"postcode": pc} if pc else {},
    )


# ----------------------------
# Housing (baseline stub)
# ----------------------------
def get_housing_data(postcode: str) -> Dict[str, Any]:
    retrieved = now_iso()
    pc = normalize_postcode(postcode)
    return metric_missing_provider(
        "Housing market provider not configured. For investor-grade 'current market' signals, integrate a sales/rents/listings API and return evidence-linked comps.",
        [
            {"label": "UK House Price Index (GOV.UK)", "url": "https://www.gov.uk/government/collections/uk-house-price-index-reports"},
            {"label": "HM Land Registry (open data)", "url": "https://landregistry.data.gov.uk/"},
        ],
        retrieved,
        extra_metrics={"postcode": pc} if pc else {},
    )


# ----------------------------
# Land Registry SOLD COMPS (open data)
# ----------------------------
def _sparql_query(endpoint: str, query: str, timeout: int = 25) -> dict:
    r = requests.get(
        endpoint,
        params={"query": query, "format": "application/sparql-results+json"},
        headers={"User-Agent": HTTP_USER_AGENT},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def _format_gbp(n: Optional[float]) -> str:
    try:
        if n is None:
            return ""
        return f"£{int(round(float(n))):,}"
    except Exception:
        return ""


def get_land_registry_sold_comps(postcode: str, limit: int = 8) -> Dict[str, Any]:
    """
    Pull recent SOLD transactions from HM Land Registry (open data, Price Paid).
    Returns compatible object where frontend expects comparableProperties.forSale (array).
    NOTE: Uses postcode district filter (fast + robust), NOT exact postcode (LR data varies).
    """
    retrieved = now_iso()
    pc = normalize_postcode(postcode)
    sources = [{"label": "HM Land Registry (Linked Data) — Price Paid", "url": "https://landregistry.data.gov.uk/"}]

    if not pc:
        return {
            "forSale": [],
            "sourceUrl": _first_source_url(sources),
            "sources": sources,
            "retrievedAtISO": retrieved,
            "confidenceValue": 0.0,
            "status": "unavailable",
            "summary": "No postcode provided for Land Registry comps.",
        }

    district = pc.split(" ")[0] if " " in pc else pc
    endpoint = "https://landregistry.data.gov.uk/landregistry/query"

    query = f"""
PREFIX lrppi: <http://landregistry.data.gov.uk/def/ppi/>

SELECT ?price ?date ?paon ?saon ?street ?town ?postcode ?propertyType
WHERE {{
  ?tx a lrppi:TransactionRecord .
  ?tx lrppi:pricePaid ?price .
  ?tx lrppi:transactionDate ?date .
  OPTIONAL {{ ?tx lrppi:paon ?paon . }}
  OPTIONAL {{ ?tx lrppi:saon ?saon . }}
  OPTIONAL {{ ?tx lrppi:street ?street . }}
  OPTIONAL {{ ?tx lrppi:town ?town . }}
  OPTIONAL {{ ?tx lrppi:postcode ?postcode . }}
  OPTIONAL {{ ?tx lrppi:propertyType ?propertyType . }}

  FILTER(BOUND(?postcode))
  FILTER(STRSTARTS(UCASE(STR(?postcode)), UCASE("{district}")))
}}
ORDER BY DESC(?date)
LIMIT {int(limit)}
""".strip()

    try:
        payload = _sparql_query(endpoint, query, timeout=25)
        rows = payload.get("results", {}).get("bindings", [])
        if not isinstance(rows, list):
            rows = []

        comps = []
        for r in rows:
            def _g(k: str) -> str:
                v = (r.get(k) or {}).get("value")
                return v.strip() if isinstance(v, str) else ""

            price_raw = _g("price")
            date_raw = _g("date")
            paon = _g("paon")
            saon = _g("saon")
            street = _g("street")
            town = _g("town")
            pc_row = _g("postcode")
            ptype = _g("propertyType")

            price_num = safe_float(price_raw)
            price_gbp = _format_gbp(price_num)

            address_parts = [saon, paon, street, town, pc_row]
            address = ", ".join([p for p in address_parts if isinstance(p, str) and p.strip()])

            desc_bits = []
            if date_raw:
                desc_bits.append(f"Sold {date_raw[:10]}")
            if price_gbp:
                desc_bits.append(price_gbp)
            if ptype:
                desc_bits.append(ptype.split("/")[-1])
            description = " • ".join(desc_bits) if desc_bits else "Sold transaction (Land Registry)."

            comps.append({
                "address": address or f"{district} (exact address not provided)",
                "description": description,
                "distance": "",  # do not invent
                "source": "HM Land Registry (sold prices)",
                "estimatedValue": price_gbp or "",
                "sourceUrl": "https://landregistry.data.gov.uk/",
                "confidenceValue": MIN_VERIFIED,
            })

        summary = (
            f"Recent sold transactions found for {district} (sample: {len(comps)}). Source: HM Land Registry Price Paid."
            if comps else
            f"No recent sold transactions returned for {district} from Land Registry."
        )

        return {
            "forSale": comps,
            "sourceUrl": _first_source_url(sources),
            "sources": sources,
            "retrievedAtISO": retrieved,
            "confidenceValue": MIN_VERIFIED if comps else 0.0,
            "status": "ok" if comps else "unavailable",
            "summary": summary,
            "postcodeDistrict": district,
            "postcode": pc,
        }

    except Exception as e:
        return {
            "forSale": [],
            "sourceUrl": _first_source_url(sources),
            "sources": sources,
            "retrievedAtISO": retrieved,
            "confidenceValue": 0.0,
            "status": "unavailable",
            "summary": f"Land Registry comps fetch failed: {str(e)}",
            "postcodeDistrict": district,
            "postcode": pc,
        }


# Backwards-compatible stub (kept, but unused once LR comps are used)
def get_zoopla_comps(postcode: str) -> Dict[str, Any]:
    return {
        "forSale": [],
        "sourceUrl": "",
        "sources": [],
        "retrievedAtISO": now_iso(),
        "confidenceValue": 0.0,
        "status": "missing_provider",
        "summary": "Comparable sales/listings provider not configured.",
    }


# ----------------------------
# Route: /market-insights
# ----------------------------
@app.route("/market-insights", methods=["POST"])
def market_insights():
    data = request.get_json(silent=True) or {}
    postcode = normalize_postcode(data.get("postcode", "") or "")

    lat = safe_float(data.get("lat"))
    lng = safe_float(data.get("lng"))

    # cache key respects explicit lat/lng (if provided)
    cache_key = f"market-insights::{postcode}::{lat or ''}::{lng or ''}" if postcode else "market-insights::no-postcode"
    cached = cache_get(cache_key)
    if cached:
        return jsonify({**cached, "_cache": {"hit": True, "ttlSeconds": CACHE_TTL_SECONDS}})

    try:
        geo_meta = None
        if (lat is None or lng is None) and postcode:
            lat, lng, geo_meta = geocode_postcode(postcode)

        # SOLD comps (real evidence)
        sold_comps = get_land_registry_sold_comps(postcode)

        # Housing metric becomes REAL when sold comps exist (still not "market today", but factual sold evidence)
        if sold_comps.get("status") == "ok" and sold_comps.get("forSale"):
            housing_metric = metric_ok(
                sold_comps.get("summary", ""),
                {
                    "soldCount": len(sold_comps.get("forSale", [])),
                    "postcodeDistrict": sold_comps.get("postcodeDistrict", ""),
                },
                sold_comps.get("sources", []),
                sold_comps.get("retrievedAtISO", now_iso()),
                MIN_VERIFIED,
            )
        else:
            housing_metric = get_housing_data(postcode)

        local_area = {
            "retrievedAtISO": now_iso(),
            "postcode": postcode,
            "schools": get_schools_data(postcode),
            "housing": housing_metric,
            "transport": get_transport_data(lat, lng),
            "amenities": get_amenities_data(lat, lng),
            "crime": get_crime_data(lat, lng),
            "broadband": get_broadband_data(postcode),
        }

        results = {
            "postcode": postcode,
            "location": {"lat": lat, "lng": lng, "geocodeMeta": geo_meta},
            "localAreaAnalysis": local_area,
            # Keep contract name unchanged for frontend
            "comparableProperties": sold_comps,
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
