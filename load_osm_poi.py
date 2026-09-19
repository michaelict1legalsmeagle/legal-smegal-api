#!/usr/bin/env python3
"""
load_osm_poi.py - load OSM POIs into an ISOLATED schema on the Hetzner box so the
Area page's Amenities card reads LOCAL data (ST_DWithin) instead of calling live
Overpass, which fails intermittently and freezes blank amenities into area_json.

Same playbook as NaPTAN: load once, query locally, no live dependency.

INPUT: pois.geojsonseq produced by (verified 2026-09-19, 850,049 features):
    osmium tags-filter england-latest.osm.pbf \
        n/amenity w/amenity n/shop w/shop n/leisure w/leisure \
        n/tourism w/tourism n/healthcare w/healthcare -o pois.osm.pbf
    osmium export pois.osm.pbf -f geojsonseq --add-unique-id=type_id -o pois.geojsonseq

Each line is one GeoJSON Feature. Nodes export as Point; ways export as
LineString/Polygon (osmium builds area geometry for closed ways). We reduce any
non-point geometry to a coordinate-mean centroid so ways load as a single point.
coordinates are [lng, lat]. Relations are intentionally excluded (negligible slice).

Run ON THE BOX as postgres (peer auth):
    su - postgres -c "cd /srv/osm && python3 load_osm_poi.py \
        --dsn 'dbname=legalsmegal_data' --geojsonseq /srv/osm/pois.geojsonseq"

Driver: psycopg (v3), matching app.py's data_query().
"""
import argparse
import json
import sys

import psycopg


# ---- category bucketing -----------------------------------------------------
# Maps to the SIX buckets the Amenities card already renders (get_amenities_data
# / the frontend read metrics.buckets.{foodDrink,shopping,healthcare,education,
# leisure,services}). "other" is loaded but excluded from the headline buckets.
# VERIFY against `SELECT category, count(*) FROM osm_poi GROUP BY 1 ORDER BY 2 DESC`
# after load and extend where a high-count tag value falls into "other".
FOOD_AMENITY   = {"restaurant", "cafe", "pub", "bar", "fast_food", "food_court",
                  "ice_cream", "biergarten"}
HEALTH_AMENITY = {"pharmacy", "hospital", "clinic", "doctors", "dentist",
                  "veterinary", "nursing_home", "social_facility"}
EDU_AMENITY    = {"school", "college", "university", "kindergarten", "library",
                  "childcare", "language_school", "music_school"}
SERVICE_AMENITY = {"bank", "atm", "post_office", "police", "fire_station",
                   "townhall", "fuel", "car_wash", "car_rental", "bureau_de_change",
                   "courthouse", "post_depot"}


def classify(props: dict) -> str:
    """Return one of: foodDrink, shopping, healthcare, education, leisure,
    services, other - from the OSM tags on this feature."""
    shop      = props.get("shop")
    amenity   = props.get("amenity")
    leisure   = props.get("leisure")
    tourism   = props.get("tourism")
    healthcare = props.get("healthcare")

    if healthcare or (amenity in HEALTH_AMENITY):
        return "healthcare"
    if amenity in FOOD_AMENITY:
        return "foodDrink"
    if amenity in EDU_AMENITY:
        return "education"
    if amenity in SERVICE_AMENITY:
        return "services"
    if shop:
        return "shopping"
    if leisure or tourism:
        return "leisure"
    if amenity:
        return "services"   # remaining amenity=* are civic/services-ish
    return "other"


def _centroid(geom: dict):
    """Return (lng, lat) for any geometry: Point as-is, otherwise the mean of its
    coordinates. Good enough for amenity-proximity; ways/areas collapse to a point."""
    t = geom.get("type")
    c = geom.get("coordinates")
    if not c:
        return None
    if t == "Point":
        return c[0], c[1]
    if t == "LineString":
        pts = c
    elif t == "MultiLineString":
        pts = [pt for line in c for pt in line]
    elif t == "Polygon":
        pts = c[0]
    elif t == "MultiPolygon":
        pts = c[0][0]
    else:
        return None
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def rows_from_geojsonseq(path: str):
    """Yield (osm_id, category, name, subtype, lat, lng) tuples. Streams the
    file line-by-line - never loads it all into memory."""
    kept = skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lstrip("\x1e")   # tolerate RS byte if present
            if not line:
                continue
            try:
                feat = json.loads(line)
                geom = feat.get("geometry") or {}
                cen = _centroid(geom)
                if cen is None:
                    skipped += 1
                    continue
                lng, lat = float(cen[0]), float(cen[1])
                if not (-8.7 <= lng <= 1.9 and 49.8 <= lat <= 61.1):
                    skipped += 1
                    continue
                props = feat.get("properties") or {}
                cat = classify(props)
                subtype = (props.get("shop") or props.get("amenity")
                           or props.get("leisure") or props.get("tourism")
                           or props.get("healthcare") or "")
                name = (props.get("name") or "").strip()[:200]
                osm_id = str(feat.get("id") or "")[:32]
                kept += 1
                yield (osm_id, cat, name, subtype[:64], lat, lng)
            except (ValueError, KeyError, TypeError, IndexError):
                skipped += 1
                continue
    print(f"[parse] kept {kept:,} POIs (nodes+ways), skipped {skipped:,}", file=sys.stderr)


def load(dsn: str, path: str, schema: str = "osm") -> None:
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS {schema};
    DROP TABLE IF EXISTS {schema}.poi;
    CREATE TABLE {schema}.poi (
        osm_id    text,
        category  text,
        name      text,
        subtype   text,
        lat       double precision,
        lng       double precision,
        geog geography(Point,4326)
             GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(lng,lat),4326)::geography) STORED
    );
    """
    with psycopg.connect(dsn, options=f"-c search_path={schema},public") as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT postgis_full_version();")   # fail early if no PostGIS
            cur.execute(ddl)
            n = 0
            with cur.copy(
                f"COPY {schema}.poi (osm_id,category,name,subtype,lat,lng) FROM STDIN"
            ) as cp:
                for row in rows_from_geojsonseq(path):
                    cp.write_row(row)
                    n += 1
            cur.execute(f"CREATE INDEX ON {schema}.poi USING GIST (geog);")
            cur.execute(f"CREATE INDEX ON {schema}.poi (category);")

            # ---- self-verify (inside txn; raises -> full rollback) ----
            cur.execute(f"SELECT count(*) FROM {schema}.poi;")
            total = cur.fetchone()[0]
            cur.execute(f"SELECT category, count(*) FROM {schema}.poi GROUP BY 1 ORDER BY 2 DESC;")
            by_cat = cur.fetchall()
            cur.execute(f"SELECT min(lat),max(lat),min(lng),max(lng) FROM {schema}.poi;")
            bbox = cur.fetchone()

            print(f"[OK] loaded {total:,} POIs")
            print("[OK] category split:", ", ".join(f"{c}={n:,}" for c, n in by_cat))
            print(f"[OK] bbox lat[{bbox[0]:.3f},{bbox[1]:.3f}] lng[{bbox[2]:.3f},{bbox[3]:.3f}]")

            cats = dict(by_cat)
            assert total > 300000, f"suspiciously few POIs: {total}"
            assert cats.get("shopping", 0) > 0 and cats.get("foodDrink", 0) > 0, \
                "core buckets empty - classify() or tags wrong"
            other = cats.get("other", 0)
            if other > total * 0.25:
                print(f"[WARN] {other:,} POIs fell to 'other' (>25%) - extend classify() "
                      f"before trusting bucket counts.")
        conn.commit()
    print("[OK] committed. Rollback (if needed): DROP SCHEMA osm CASCADE;")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True, help='e.g. "dbname=legalsmegal_data"')
    ap.add_argument("--geojsonseq", required=True, help="path to pois.geojsonseq")
    ap.add_argument("--schema", default="osm")
    a = ap.parse_args()
    load(a.dsn, a.geojsonseq, a.schema)
