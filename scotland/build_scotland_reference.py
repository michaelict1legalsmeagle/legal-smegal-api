#!/usr/bin/env python3
"""
build_scotland_reference.py  —  ONE-TIME ETL

Loads the three verified Scottish reference files into Postgres so the web service
never parses a 100 MB CSV or an xlsx at runtime. Idempotent: drops & rebuilds the
three tables. Run once per data refresh (SPD ~quarterly, price annual, RoS monthly).

    python3 build_scotland_reference.py \
        --dsn "postgresql://user:pass@host:5432/legalsmegal" \
        --ref-dir /data/scotland          # holds the 3 files below

Reuses the SAME verified parsers as the engine (encoding, quoted-comma handling,
live-only filter, 2011 vintage, RoS column offset) — no divergent logic.
"""
import argparse, csv
from pathlib import Path
import psycopg
import scotland_band_engine as e   # reuse verified loaders

DDL = """
DROP TABLE IF EXISTS scotland_postcode;
DROP TABLE IF EXISTS scotland_area_price;
DROP TABLE IF EXISTS scotland_la_median;

CREATE TABLE scotland_postcode (
    postcode       text PRIMARY KEY,          -- normalised: no spaces, upper-case
    dz2011         text NOT NULL,
    iz2011         text NOT NULL,
    la_code        text NOT NULL,
    simd2020_rank  integer,
    ur6            text,
    lat            double precision,
    lng            double precision
);
CREATE TABLE scotland_area_price (
    area_code      text PRIMARY KEY,          -- S01 DZ / S02 IZ / S12 LA / S92 Scotland
    median_2023    integer NOT NULL
);
CREATE TABLE scotland_la_median (
    la_code         text PRIMARY KEY,
    current_median  integer,
    current_period  text,
    median_2023     integer
);
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--ref-dir", required=True)
    args = ap.parse_args()
    ref_dir = Path(args.ref_dir)

    # --- parse with the engine's verified loaders --------------------------------
    pc    = e.load_postcode_index(ref_dir / "SmallUser.csv")
    price = e.load_price_median_2023(ref_dir / "residential-properties-sales-and-price.csv")
    cur, period, y2023 = e.load_ros_la_medians(ref_dir / "ros_all_stats_June_2026.xlsx")

    with psycopg.connect(args.dsn) as conn, conn.cursor() as cur_db:
        cur_db.execute(DDL)

        with cur_db.copy("COPY scotland_postcode "
                         "(postcode,dz2011,iz2011,la_code,simd2020_rank,ur6,lat,lng) "
                         "FROM STDIN") as cp:
            for code, g in pc.items():
                cp.write_row((code, g["dz2011"], g["iz2011"], g["la_code"],
                              g["simd2020_rank"], g["ur6"], g["lat"], g["long"]))

        with cur_db.copy("COPY scotland_area_price (area_code,median_2023) FROM STDIN") as cp:
            for code, med in price.items():
                cp.write_row((code, med))

        with cur_db.copy("COPY scotland_la_median "
                         "(la_code,current_median,current_period,median_2023) FROM STDIN") as cp:
            for la in set(cur) | set(y2023):
                cp.write_row((la, cur.get(la), period, y2023.get(la)))

        conn.commit()

        # --- built-in verification (fail loud) -----------------------------------
        cur_db.execute("SELECT count(*) FROM scotland_postcode");   n_pc = cur_db.fetchone()[0]
        cur_db.execute("SELECT count(*) FROM scotland_area_price"); n_pr = cur_db.fetchone()[0]
        cur_db.execute("SELECT count(*) FROM scotland_la_median");  n_la = cur_db.fetchone()[0]
        cur_db.execute("SELECT count(*) FROM scotland_area_price "
                       "WHERE area_code LIKE %s AND area_code > %s", ("S01%", e.DZ2011_MAX))
        leaked = cur_db.fetchone()[0]
        assert leaked == 0, f"{leaked} non-2011 data-zone codes leaked into price table"
        print(f"loaded: scotland_postcode={n_pc:,}  scotland_area_price={n_pr:,}  "
              f"scotland_la_median={n_la}  | RoS period={period}  | vintage guard OK")

if __name__ == "__main__":
    main()
