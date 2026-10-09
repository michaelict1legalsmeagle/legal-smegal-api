"""ew_flood.py — FLOOD-EVID-1 (9 Oct 2026): England flood zone from the Environment
Agency's official Flood Map for Planning — Flood Zones (OGL v3.0).

Why this exists (verified live 9 Oct 2026): the old get_flood_risk asked the EA
*flood-monitoring* API for flood ALERT/WARNING areas within 0.5 km and tried to read a
zone number out of their names ("River Rea", "Upper Tame"…). Those names carry no zone,
so every successful call came out "Zone 1, below 0.1%". Checked against the official
Flood Map for Planning at each deal's point, 8 of 59 stored "Zone 1" deals sit in Flood
Zone 2 or 3 (HU9 3AQ, E6 2AU, DN31 2HX, DN32 7RT, LN5 7NQ = Zone 3).

This module is pure (no network, no DB): it turns the OGC API - Features response for a
tiny box around the deal point into the area_json.flood block. app.py does the fetch.

Definitions are the EA's own (Flood Zones product description):
  Flood Zone 3 — 1 in 100 or greater annual probability of river flooding, or 1 in 200
                 or greater of sea flooding.
  Flood Zone 2 — 1 in 100 to 1 in 1,000 (river), 1 in 200 to 1 in 1,000 (sea); also
                 accepted recorded flood outlines.
  Flood Zone 1 — all land outside Zones 2 and 3 (below 1 in 1,000).
The zones show present-day risk and IGNORE flood defences; they describe land, not
whether an individual property will flood.
"""
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA_VERSION = "flood-evid-1"
SOURCE_LABEL = "Environment Agency Flood Map for Planning (Flood Zones)"
DATASET_PAGE = ("https://www.data.gov.uk/dataset/104434b0-5263-4c90-9b1e-e43b1d57c750/"
                "flood-map-for-planning-flood-zones1")
ITEMS_URL = ("https://environment.data.gov.uk/spatialdata/flood-map-for-planning-flood-zones/"
             "ogc/features/v1/collections/Flood_Zones_2_3_Rivers_and_Sea/items")
CHECK_MAP_URL = "https://flood-map-for-planning.service.gov.uk/"
# Half-width of the query box in degrees (~2 m). The box only selects candidate polygons;
# the answer is decided by an exact point-in-polygon test below.
BOX_DEG = 0.00002
PAGE_LIMIT = 200

ZONE_TEXT = {
    3: "Flood Zone 3 (high probability)",
    2: "Flood Zone 2 (medium probability)",
    1: "Flood Zone 1 (low probability)",
}


def query_params(lat: float, lng: float) -> Dict[str, str]:
    d = BOX_DEG
    return {"bbox": f"{lng - d},{lat - d},{lng + d},{lat + d}", "f": "json", "limit": str(PAGE_LIMIT)}


# ── geometry (GeoJSON lon/lat order; verified on the live service 9 Oct 2026) ──────────
def _in_ring(x: float, y: float, ring: List[List[float]]) -> bool:
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _in_polygon(x: float, y: float, poly: List[List[List[float]]]) -> bool:
    if not poly or not _in_ring(x, y, poly[0]):
        return False
    return not any(_in_ring(x, y, hole) for hole in poly[1:])


def point_in_geometry(lng: float, lat: float, geom: Any) -> bool:
    if not isinstance(geom, dict):
        return False
    t, c = geom.get("type"), geom.get("coordinates")
    if t == "Polygon":
        return _in_polygon(lng, lat, c or [])
    if t == "MultiPolygon":
        return any(_in_polygon(lng, lat, p) for p in (c or []))
    return False


def _zone_num(fz: Any) -> Optional[int]:
    s = str(fz or "").upper().replace(" ", "")
    if s in ("FZ3", "3"):
        return 3
    if s in ("FZ2", "2"):
        return 2
    return None


def _sources(feats: Iterable[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for f in feats:
        for part in str((f.get("properties") or {}).get("flood_source") or "").replace(" and ", ",").split(","):
            p = part.strip().lower()
            if p and p not in out:
                out.append(p)
    return sorted(out)


def probability_text(zone: int, sources: List[str]) -> str:
    river, sea = "river" in sources, "sea" in sources
    if zone == 3:
        if river and sea:
            return "1 in 100 or more a year (river) / 1 in 200 or more (sea)"
        return "1 in 200 or more a year (sea)" if sea else "1 in 100 or more a year (river)"
    if zone == 2:
        if river and sea:
            return "1 in 1,000 to 1 in 100 a year (river) / 1 in 1,000 to 1 in 200 (sea)"
        return "1 in 1,000 to 1 in 200 a year (sea)" if sea else "1 in 1,000 to 1 in 100 a year (river)"
    return "below 1 in 1,000 a year"


def flood_block(http_status: Optional[int], payload: Any, lat: float, lng: float,
                retrieved: str, request_url: str = "") -> Dict[str, Any]:
    """OGC response -> area_json.flood. Never invents a zone: anything that is not a
    complete, valid answer is 'unavailable' (transient=True so it is never persisted)."""
    sources_meta = [{"label": SOURCE_LABEL, "url": DATASET_PAGE}]
    if http_status != 200 or not isinstance(payload, dict) or not isinstance(payload.get("features"), list):
        return _unavailable("The Environment Agency flood zone map did not return a valid answer.",
                            sources_meta, retrieved, transient=True)
    feats = [f for f in payload["features"] if isinstance(f, dict)]
    hits = [f for f in feats if point_in_geometry(lng, lat, f.get("geometry"))]
    zones = [z for z in (_zone_num((f.get("properties") or {}).get("flood_zone")) for f in hits) if z]
    zone = max(zones) if zones else 1
    matched = payload.get("numberMatched")
    returned = payload.get("numberReturned", len(feats))
    try:
        incomplete = matched is not None and int(matched) > int(returned)
    except (TypeError, ValueError):
        incomplete = False
    if incomplete and zone < 3:
        # Some candidate polygons were not returned; a higher zone could be among them.
        return _unavailable("The Environment Agency flood zone answer was incomplete for this point.",
                            sources_meta, retrieved, transient=True)
    srcs = _sources(f for f in hits if _zone_num((f.get("properties") or {}).get("flood_zone")) == zone) if zone > 1 else []
    origins = sorted({str((f.get("properties") or {}).get("origin") or "") for f in hits} - {""})
    prob = probability_text(zone, srcs)
    src_txt = (" from " + " and ".join(srcs)) if srcs else ""
    summary = (f"{ZONE_TEXT[zone]}{src_txt}: {prob}, at the postcode's point. "
               "Flood zones ignore flood defences and describe land, not whether this "
               f"property will flood ({SOURCE_LABEL}).")
    return {
        "status": "ok",
        "summary": summary,
        "value": [{"zone": zone, "flood_zone": f"FZ{zone}", "sources": srcs}],
        "metrics": {
            "schema": SCHEMA_VERSION,
            "zone": zone,
            "flood_zone": f"FZ{zone}",
            "flood_sources": srcs,
            "origin": origins,
            "probability": prob,
            "basis": "postcode point",
            "defences": "ignored",
            "check_url": CHECK_MAP_URL,
        },
        "sources": sources_meta,
        "sourceUrl": DATASET_PAGE,
        "retrievedAtISO": retrieved,
        "confidenceValue": 0.9,
        "needsEvidence": False,
        "requestUrl": request_url,
    }


def _unavailable(note: str, sources_meta: list, retrieved: str, transient: bool) -> Dict[str, Any]:
    return {
        "status": "unavailable",
        "summary": note + " Check the Environment Agency flood map for this property.",
        "value": None,
        "metrics": {"schema": SCHEMA_VERSION, "check_url": CHECK_MAP_URL, "transient": transient},
        "sources": sources_meta,
        "sourceUrl": DATASET_PAGE,
        "retrievedAtISO": retrieved,
        "confidenceValue": 0.0,
        "needsEvidence": True,
    }


def no_coordinates(retrieved: str) -> Dict[str, Any]:
    return _unavailable("Flood zone not available: the postcode could not be resolved to a point.",
                        [{"label": SOURCE_LABEL, "url": DATASET_PAGE}], retrieved, transient=False)


def pending_recheck(retrieved: str) -> Dict[str, Any]:
    """Served in place of a pre-FLOOD-EVID-1 block while its re-check is rate-limited:
    the old block's zone came from the wrong dataset and must not be shown."""
    return _unavailable("Flood zone is being re-checked against the Environment Agency flood map.",
                        [{"label": SOURCE_LABEL, "url": DATASET_PAGE}], retrieved, transient=True)


def is_current(flood: Any) -> bool:
    """A stored E&W flood block that can be shown as is: built by FLOOD-EVID-1 and ok."""
    if not isinstance(flood, dict):
        return False
    m = flood.get("metrics") or {}
    return m.get("schema") == SCHEMA_VERSION and flood.get("status") == "ok"
