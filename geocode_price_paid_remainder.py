"""
geocode_price_paid_remainder.py
================================
Catches all remaining lat IS NULL rows that are matchable via nspl_lookup.
Batches by postcode district (full outward code e.g. 'SW1A', 'B44', 'NG1')
rather than 2-char prefix to avoid missing multi-char districts.
Safe to re-run — only touches rows where lat IS NULL.
"""

import os
import sys
import time
import logging
import psycopg
from psycopg.rows import dict_row

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

DB_URL = os.environ.get("SUPABASE_DB_URL") or os.environ.get("DATABASE_URL")
if not DB_URL:
    log.error("SUPABASE_DB_URL not set")
    sys.exit(1)

def run():
    log.info("Connecting to Supabase Postgres...")
    conn = psycopg.connect(DB_URL, row_factory=dict_row)
    conn.autocommit = False

    # Get all distinct postcode districts that still have null rows
    log.info("Fetching distinct postcode districts with null lat...")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 
                REGEXP_REPLACE(postcode_nospace, '[0-9][A-Z]{2}$', '') as district,
                COUNT(*) as cnt
            FROM price_paid_raw_2025
            WHERE lat IS NULL
              AND postcode_nospace IS NOT NULL
              AND postcode_nospace != ''
            GROUP BY 1
            ORDER BY 2 DESC
        """)
        districts = cur.fetchall()

    log.info(f"Found {len(districts)} distinct districts with null lat rows")

    # Starting null count
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) as n FROM price_paid_raw_2025 WHERE lat IS NULL")
        start_null = cur.fetchone()["n"]
    log.info(f"Starting null count: {start_null:,}")

    total_updated = 0
    errors = 0

    for i, row in enumerate(districts):
        district = row["district"]
        if not district:
            continue

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE price_paid_raw_2025 p
                    SET lat = n.lat, lng = n.lng
                    FROM nspl_lookup n
                    WHERE n.pcd_nospace = p.postcode_nospace
                      AND p.lat IS NULL
                      AND REGEXP_REPLACE(p.postcode_nospace, '[0-9][A-Z]{2}$', '') = %s
                """, (district,))
                rows = cur.rowcount
            conn.commit()
            total_updated += rows

            if i % 100 == 0 or rows > 1000:
                log.info(f"  [{i+1}/{len(districts)}] {district}: {rows:,} rows  (total: {total_updated:,})")

        except Exception as e:
            conn.rollback()
            log.error(f"  {district}: FAILED — {e}")
            errors += 1
            continue

    # Final count
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(lat) as geocoded,
                COUNT(*) - COUNT(lat) as still_null,
                ROUND(COUNT(lat) * 100.0 / COUNT(*), 1) as pct_geocoded
            FROM price_paid_raw_2025
        """)
        result = cur.fetchone()

    log.info("=" * 50)
    log.info("COMPLETE")
    log.info(f"  Total rows:    {result['total']:,}")
    log.info(f"  Geocoded:      {result['geocoded']:,}  ({result['pct_geocoded']}%)")
    log.info(f"  Still null:    {result['still_null']:,}")
    log.info(f"  Updated now:   {total_updated:,}")
    log.info(f"  Errors:        {errors}")
    log.info("=" * 50)

    conn.close()

if __name__ == "__main__":
    run()
