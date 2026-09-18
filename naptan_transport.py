"""
naptan_transport.py — transport enrichment from the locally-loaded naptan.stops
table on Hetzner, replacing the live OSM/Overpass call in app.get_transport_data().

WHY: public Overpass mirrors intermittently hang/429 (app.py's own comments say
so), which is why the Transport card blanks at random. NaPTAN is a static DfT
register: load once, query locally via the app's existing data_query() with the
same PostGIS pattern as nspl_postcodes. No external HTTP -> no hang, no false blank.

CONTRACT (unchanged, so no frontend edit is needed). It returns the same envelope
as the old function and sets:
  metrics = {
    "radiusMeters": int,
    "counts":  {"stations": int, "tram_stops": int, "public_transport": int, "bus_stops": int},
    "sample":  {"stations": [str], "tram": [str], "bus": [str]},
    "points":  [{"lat": float, "lng": float, "kind": "rail"|"tram"|"bus", "name": str}],
  }
  counts.stations   = rail
  counts.tram_stops = metro/tram
  counts.bus_stops  = bus/coach
  (ferry/air/other are NOT folded into these three — honest, not mislabelled.)

ENHANCEMENT (additive keys the old path never produced; existing FE ignores them):
  metrics["nearest"] = {"rail": {"name","dist_m"}|None, "tram": .., "bus": ..}
and sample["stations"][0] is annotated with distance, e.g. "Peterlee (0.4 mi)",
so the existing (previously always-blank) "Nearest station" row fills in with a
real distance and NO html change.

All app helpers are injected so this module has zero import of app.py (no cycle)
and reuses app's exact metric_ok/metric_unavailable envelope.
"""
from typing import Any, Dict, List, Optional

_MODE_TO_KIND = {"rail": "rail", "metro": "tram", "bus": "bus"}  # FE map only styles these
_POINT_CAP = 250        # cap map pins by nearest; counts stay full
_SAMPLE_CAP = {"rail": 6, "metro": 6, "bus": 8}


def _mi(dist_m: float) -> str:
    return f"{dist_m / 1609.34:.1f} mi"


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
    schema, _, tbl = table.partition(".")
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
    if not rows:                       # widen once, like the old OSM fallback — but local + cheap
        rows = _fetch(2500)
        used_radius = 2500 if rows else radius_m

    counts = {"stations": 0, "tram_stops": 0, "public_transport": 0, "bus_stops": 0}
    sample: Dict[str, List[str]] = {"stations": [], "tram": [], "bus": []}
    nearest: Dict[str, Optional[Dict[str, Any]]] = {"rail": None, "tram": None, "bus": None}
    points: List[Dict[str, Any]] = []
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
        counts["public_transport"] += 1

        # nearest per FE-kind (rows are distance-ordered, so first wins)
        nk = "rail" if kind == "rail" else ("tram" if kind == "tram" else "bus")
        if nearest[nk] is None and name and dist is not None:
            nearest[nk] = {"name": name, "dist_m": int(round(dist))}

        if name and name not in seen_name[skey] and len(sample[skey]) < _SAMPLE_CAP[mode if mode != "metro" else "metro"]:
            seen_name[skey].add(name)
            sample[skey].append(name)

        if len(points) < _POINT_CAP:
            plat, plng = safe_float(r.get("lat")), safe_float(r.get("lng"))
            if plat is not None and plng is not None:
                points.append({"lat": plat, "lng": plng, "kind": kind, "name": name})

    total = counts["stations"] + counts["tram_stops"] + counts["bus_stops"]

    # ENHANCEMENT: annotate the nearest station name with its distance so the
    # existing "Nearest station" row (sample.stations[0]) shows "<name> (x.x mi)".
    if nearest["rail"] and sample["stations"]:
        n = nearest["rail"]
        sample["stations"][0] = f"{n['name']} ({_mi(n['dist_m'])})"
    elif nearest["tram"] and sample["tram"]:
        n = nearest["tram"]
        sample["tram"][0] = f"{n['name']} ({_mi(n['dist_m'])})"

    if total == 0:
        out = metric_ok(
            f"No NaPTAN transport stops within ~{used_radius}m of this address.",
            [], sources, retrieved, 0.0,
        )
    else:
        bullets = [f"• Rail: {counts['stations']} station(s) within ~{used_radius}m"
                   + (f" (nearest: {sample['stations'][0]})" if sample["stations"] else "")]
        if counts["tram_stops"]:
            bullets.append(f"• Metro/tram: {counts['tram_stops']} stop(s)")
        bullets.append(f"• Bus: {counts['bus_stops']} stop(s) within ~{used_radius}m")
        out = metric_ok(
            "Transport (NaPTAN within ~%dm):\n%s" % (used_radius, "\n".join(bullets)),
            bullets, sources, retrieved, 0.95,
        )

    out["metrics"] = {
        "radiusMeters": used_radius,
        "counts": counts,
        "sample": sample,
        "points": points,
        "nearest": nearest,     # additive; future FE can render "Peterlee · 0.4 mi"
    }
    return out
