"""
naptan_transport.py — transport enrichment from the locally-loaded naptan.stops
table on Hetzner, replacing the live OSM/Overpass call in app.get_transport_data().

WHY: public Overpass mirrors intermittently hang/429 (app.py's own comments say
so), which is why the Transport card blanks at random. NaPTAN is a static DfT
register: load once, query locally via the app's existing data_query() with the
same PostGIS pattern as nspl_postcodes. No external HTTP -> no hang, no false blank.

CONTRACT (unchanged, so no frontend edit is required for the counts/map). It
returns the same envelope as the old function and sets:
  metrics = {
    "radiusMeters": int,
    "counts":  {"stations": int, "tram_stops": int, "public_transport": int, "bus_stops": int},
    "sample":  {"stations": [str], "tram": [str], "bus": [str]},
    "points":  [{"lat": float, "lng": float, "kind": "rail"|"tram"|"bus", "name": str}],
    "nearest": {"rail": {"name","dist_m"}|None, "tram": ..|None, "bus": ..|None},
  }
  counts.stations   = rail WITHIN the card radius (drives the grade — stays honest
                      to what is genuinely near; sparse rail rarely sits in 1.2km)
  counts.tram_stops = metro/tram within radius
  counts.bus_stops  = bus/coach within radius
  (ferry/air/other are NOT folded into these three — honest, not mislabelled.)

NEAREST-STATION FIX: "counts" is bounded by the card radius, but the nearest RAIL
station is usually further out (sparse). So metrics["nearest"]["rail"] / ["tram"]
are computed with a SEPARATE, wider, distance-capped lookup (near_cap_m, default
16 km) that is independent of the card radius. That is what fills the frontend
"Nearest station" row with a real name + distance, wherever a station is in reach,
without inflating the grade. Beyond the cap -> None -> the row honestly shows "—".

All app helpers are injected so this module has zero import of app.py (no cycle)
and reuses app's exact metric_ok/metric_unavailable envelope.
"""
from typing import Any, Dict, List, Optional

_MODE_TO_KIND = {"rail": "rail", "metro": "tram", "bus": "bus"}  # FE map only styles these
_POINT_CAP = 250            # cap map pins by nearest; counts stay full
_SAMPLE_CAP = {"rail": 6, "metro": 6, "bus": 8}


def _nearest(data_query, table, lng, lat, mode, cap_m, safe_float) -> Optional[Dict[str, Any]]:
    """Nearest single stop of `mode` within cap_m metres, or None. Independent of
    the card radius — this is what lets the 'Nearest station' row reach a station
    that sits beyond the bus radius."""
    rows = data_query(
        f"SELECT stop_name, "
        f"       ST_Distance(geog, ST_SetSRID(ST_MakePoint(%s,%s),4326)::geography) AS d "
        f"FROM {table} "
        f"WHERE mode = %s "
        f"  AND ST_DWithin(geog, ST_SetSRID(ST_MakePoint(%s,%s),4326)::geography, %s) "
        f"ORDER BY d LIMIT 1",
        (lng, lat, mode, lng, lat, cap_m),
    ) or []
    if not rows:
        return None
    name = (rows[0].get("stop_name") or "").strip()
    d = safe_float(rows[0].get("d"))
    if not name or d is None:
        return None
    return {"name": name, "dist_m": int(round(d))}


def get_transport_data(
    lat: Optional[float],
    lng: Optional[float],
    *,
    data_query,
    now_iso,
    metric_ok,
    metric_unavailable,
    safe_float,
    radius_m: int = 1200,
    near_cap_m: int = 16000,
    table: str = "naptan.stops",
) -> Dict[str, Any]:
    retrieved = now_iso()
    sources = [
        {"label": "NaPTAN (DfT)", "url": "https://beta-naptan.dft.gov.uk/"},
        {"label": "Hetzner (naptan.stops)", "url": ""},
    ]

    if lat is None or lng is None:
        return metric_unavailable(
            "Transport data not available: postcode could not be resolved to coordinates.",
            sources, retrieved,
        )

    # Distinguish "table not loaded yet" from "genuinely no stops nearby".
    probe = data_query("SELECT to_regclass(%s) AS t;", (table,))
    if not probe or probe[0].get("t") is None:
        return metric_unavailable(
            f"Transport data unavailable: {table} is not loaded on the data box yet.",
            sources, retrieved,
        )

    sql = f"""
        SELECT stop_name, mode, lat, lng,
               ST_Distance(geog, ST_SetSRID(ST_MakePoint(%s,%s),4326)::geography) AS dist_m
        FROM {table}
        WHERE ST_DWithin(geog, ST_SetSRID(ST_MakePoint(%s,%s),4326)::geography, %s)
        ORDER BY dist_m
    """

    def _fetch(rad: int) -> List[Dict[str, Any]]:
        return data_query(sql, (lng, lat, lng, lat, rad)) or []

    rows = _fetch(radius_m)
    used_radius = radius_m
    if not rows:                       # widen once — local + cheap, no network
        rows = _fetch(2500)
        used_radius = 2500 if rows else radius_m

    counts = {"stations": 0, "tram_stops": 0, "public_transport": 0, "bus_stops": 0}
    sample: Dict[str, List[str]] = {"stations": [], "tram": [], "bus": []}
    points: List[Dict[str, Any]] = []
    nearest_bus: Optional[Dict[str, Any]] = None
    seen_name = {"stations": set(), "tram": set(), "bus": set()}

    for r in rows:
        mode = (r.get("mode") or "").lower()
        kind = _MODE_TO_KIND.get(mode)          # rail/tram/bus, else skip the 3 buckets
        if kind is None:
            continue
        name = (r.get("stop_name") or "").strip()
        dist = safe_float(r.get("dist_m"))

        if kind == "rail":
            counts["stations"] += 1
            skey = "stations"
        elif kind == "tram":
            counts["tram_stops"] += 1
            skey = "tram"
        else:
            counts["bus_stops"] += 1
            skey = "bus"
            if nearest_bus is None and name and dist is not None:
                nearest_bus = {"name": name, "dist_m": int(round(dist))}
        counts["public_transport"] += 1

        cap = _SAMPLE_CAP["metro" if mode == "metro" else mode]
        if name and name not in seen_name[skey] and len(sample[skey]) < cap:
            seen_name[skey].add(name)
            sample[skey].append(name)

        if len(points) < _POINT_CAP:
            plat, plng = safe_float(r.get("lat")), safe_float(r.get("lng"))
            if plat is not None and plng is not None:
                points.append({"lat": plat, "lng": plng, "kind": kind, "name": name})

    total = counts["stations"] + counts["tram_stops"] + counts["bus_stops"]

    # NEAREST — rail/metro searched wide (independent of the card radius) so the
    # "Nearest station" row fills wherever a station is realistically in reach.
    nearest = {
        "rail": _nearest(data_query, table, lng, lat, "rail", near_cap_m, safe_float),
        "tram": _nearest(data_query, table, lng, lat, "metro", near_cap_m, safe_float),
        "bus":  nearest_bus,
    }

    if total == 0 and not nearest["rail"] and not nearest["tram"]:
        out = metric_ok(
            f"No NaPTAN transport stops within ~{used_radius}m of this address.",
            [], sources, retrieved, 0.0,
        )
    else:
        bullets = []
        if nearest["rail"]:
            miles = nearest["rail"]["dist_m"] / 1609.34
            bullets.append(f"• Nearest rail: {nearest['rail']['name']} (~{miles:.1f} mi)")
        bullets.append(f"• Rail stations within ~{used_radius}m: {counts['stations']}")
        if counts["tram_stops"]:
            bullets.append(f"• Metro/tram within ~{used_radius}m: {counts['tram_stops']}")
        bullets.append(f"• Bus stops within ~{used_radius}m: {counts['bus_stops']}")
        out = metric_ok(
            "Transport (NaPTAN):\n" + "\n".join(bullets),
            bullets, sources, retrieved, 0.95,
        )

    out["metrics"] = {
        "radiusMeters": used_radius,
        "counts": counts,
        "sample": sample,
        "points": points,
        "nearest": nearest,
    }
    return out
