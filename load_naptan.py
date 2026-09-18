#!/usr/bin/env python3
"""
load_naptan.py — load the DfT NaPTAN stops CSV into an ISOLATED schema on the
Hetzner box (159.69.27.104, db legalsmegal_data), for local transport queries
that replace the live OSM/Overpass call.

Follows the schema-staging doctrine:
  - all objects live in a dedicated `naptan` schema (never public)
  - additive + atomic: everything in ONE transaction, rolls back to nothing
  - self-verifying: prints counts, mode split, coord bbox, and a sanity guard
  - the app's existing role gets SELECT via a separate psql GRANT step (see the
    commands in the chat), NOT from this script.

Run ON THE BOX as the superuser OS user (peer auth), after the CREATE SCHEMA +
GRANT step:

    sudo -u postgres python3 load_naptan.py --dsn "dbname=legalsmegal_data" \
        --csv /path/to/Stops.csv

Driver: psycopg (v3), to match app.py's data_query().
NaPTAN CSV: the DfT "Stops.csv" download (national). Standard columns used below
are ATCOCode, CommonName, Longitude, Latitude, StopType, Status. If your download
has Easting/Northing instead of Longitude/Latitude, this loader stops with a clear
message rather than loading wrong coordinates.
"""
import argparse
import csv
import sys

import psycopg


# ── StopType -> mode map ──────────────────────────────────────────────────────
# VERIFY THIS against your actual download. After loading, run:
#     SELECT stop_type, count(*) FROM naptan.stops GROUP BY 1 ORDER BY 2 DESC;
# and against the NaPTAN StopType schema. Counts on the transport card depend on
# this map being right. Anything not listed here is loaded as mode 'other' and
# excluded from the card's three buckets (honest — not silently counted as bus).
STOPTYPE_MODE = {
    # National Rail
    "RLY": "rail", "RSE": "rail", "RPL": "rail",
    # Metro / underground / tram (NaPTAN folds tram in here; no distinct tram code)
    "MET": "metro", "PLT": "metro", "TMU": "metro",
    # Bus / coach
    "BCT": "bus", "BCS": "bus", "BCQ": "bus", "BCE": "bus", "BST": "bus",
    "MKD": "bus", "CUS": "bus", "HAR": "bus", "FLX": "bus",
    # Ferry
    "FER": "ferry", "FBT": "ferry", "FTD": "ferry",
    # Air
    "AIR": "air", "GAT": "air",
}
ACTIVE_STATUSES = {"active", "act", "1", "true"}


def _norm_headers(fieldnames):
    """Map lower-cased header -> actual header, so we tolerate case variation."""
    return {h.lower().strip().lstrip("\ufeff"): h for h in (fieldnames or [])}


def load(dsn: str, csv_path: str, schema: str = "naptan") -> None:
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        hmap = _norm_headers(reader.fieldnames)

        need = ["commonname", "stoptype", "status"]
        for col in need:
            if col not in hmap:
                sys.exit(f"[FATAL] NaPTAN CSV missing required column '{col}'. "
                         f"Found: {list(hmap.keys())[:20]}")
        if "longitude" not in hmap or "latitude" not in hmap:
            sys.exit("[FATAL] NaPTAN CSV has no Longitude/Latitude columns. Your "
                     "download is likely Easting/Northing (OSGB36); convert to "
                     "WGS84 (pyproj EPSG:27700->4326) before loading. Refusing to "
                     "load without real lat/lng rather than guess coordinates.")

        c_name, c_type, c_stat = hmap["commonname"], hmap["stoptype"], hmap["status"]
        c_lng, c_lat = hmap["longitude"], hmap["latitude"]
        c_atco = hmap.get("atcocode")

        rows = []
        skipped_coord = skipped_inactive = 0
        for r in reader:
            if (r.get(c_stat) or "").strip().lower() not in ACTIVE_STATUSES:
                skipped_inactive += 1
                continue
            try:
                lng = float(r[c_lng]); lat = float(r[c_lat])
            except (TypeError, ValueError):
                skipped_coord += 1
                continue
            # UK bbox sanity — anything outside is bad data, not a stop.
            if not (-8.7 <= lng <= 1.9 and 49.8 <= lat <= 61.1):
                skipped_coord += 1
                continue
            stype = (r.get(c_type) or "").strip().upper()
            rows.append((
                (r.get(c_atco) or "").strip() if c_atco else "",
                (r.get(c_name) or "").strip(),
                stype,
                STOPTYPE_MODE.get(stype, "other"),
                lat, lng,
            ))

    if not rows:
        sys.exit("[FATAL] No active, well-coordinated NaPTAN rows parsed — check "
                 "the Status values and coordinate columns in your download.")

    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS {schema};
    DROP TABLE IF EXISTS {schema}.stops;
    CREATE TABLE {schema}.stops (
        atco       text,
        stop_name  text,
        stop_type  text,
        mode       text,
        lat        double precision,
        lng        double precision,
        geog geography(Point,4326)
             GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(lng,lat),4326)::geography) STORED
    );
    """

    with psycopg.connect(dsn, options=f"-c search_path={schema}") as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT postgis_full_version();")  # fail early if no PostGIS
            cur.execute(ddl)
            with cur.copy(
                f"COPY {schema}.stops (atco,stop_name,stop_type,mode,lat,lng) FROM STDIN"
            ) as cp:
                for row in rows:
                    cp.write_row(row)
            cur.execute(f"CREATE INDEX ON {schema}.stops USING GIST (geog);")
            cur.execute(f"CREATE INDEX ON {schema}.stops (mode);")

            # ── self-verify (still inside the txn; raises -> full rollback) ──
            cur.execute(f"SELECT count(*) FROM {schema}.stops;")
            total = cur.fetchone()[0]
            cur.execute(f"SELECT mode, count(*) FROM {schema}.stops GROUP BY 1 ORDER BY 2 DESC;")
            by_mode = cur.fetchall()
            cur.execute(f"SELECT min(lat),max(lat),min(lng),max(lng) FROM {schema}.stops;")
            bbox = cur.fetchone()

            print(f"[OK] loaded {total:,} stops "
                  f"(skipped {skipped_inactive:,} inactive, {skipped_coord:,} bad-coord)")
            print("[OK] mode split:", ", ".join(f"{m}={n:,}" for m, n in by_mode))
            print(f"[OK] bbox lat[{bbox[0]:.3f},{bbox[1]:.3f}] lng[{bbox[2]:.3f},{bbox[3]:.3f}]")

            modes = dict(by_mode)
            assert total > 100000, f"suspiciously few stops: {total}"
            assert modes.get("bus", 0) == max(modes.values()), \
                "bus is not the largest mode — StopType map is probably wrong"
            other = modes.get("other", 0)
            if other > total * 0.10:
                print(f"[WARN] {other:,} rows fell to mode 'other' (>10%). Eyeball "
                      f"DISTINCT stop_type and extend STOPTYPE_MODE before trusting counts.")
        conn.commit()
    print("[OK] committed. Rollback (if needed): DROP SCHEMA naptan CASCADE;")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True, help='e.g. "dbname=legalsmegal_data" (peer auth)')
    ap.add_argument("--csv", required=True, help="path to NaPTAN Stops.csv")
    ap.add_argument("--schema", default="naptan")
    a = ap.parse_args()
    load(a.dsn, a.csv, a.schema)
