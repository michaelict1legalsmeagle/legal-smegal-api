"""
geocode_price_paid.py
=====================
Backfills lat/lng on price_paid_raw_2025 from nspl_lookup via postcode join.
Run as a Render One-Off Job:
  Command: python geocode_price_paid.py
  Environment: SUPABASE_DB_URL (already set on the API service)

Approach: batch by 2-char postcode area prefix, commit after each.
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

# All 2-char postcode area prefixes with ungeooded rows, ordered small→large
PREFIXES = [
    'KY','DG','TD','WC','EC','LD',
    'HX','HG','SM','HR','IG','AL','UB','WD','TF','SR','SP','DT',
    'HD','BR','EN','N1','HA','WV','WN','SL','LU','WR','CR','DH',
    'FY','TQ','DY','SY','LN','TR','BL','CA','NW','LA','TA','TW',
    'WS','OL','DA','CB','CW','SG','DL','RM','E1','NP','BA','HU',
    'HP','CO','SS','WF','LL','SN','KT','CT','BB','BD','RH','MK',
    'PL','WA','PR','BH','OX','ME','SO','TS','IP','CH','SK','ST',
    'EX','SA','YO','CM','GL','GU','TN','NN','LS','SE','DN','DE',
    'RG','SW','NR','PO','BN','CV','CF','LE','BS','PE','NE','NG',
    # London split prefixes
    'E2','E3','E4','E5','E6','E7','E8','E9',
    'N2','N3','N4','N5','N6','N7','N8','N9',
    'W1','W2','W3','W4','W5','W6','W7','W8','W9',
    'S1','S2','S3','S4','S5','S6','S7','S8','S9',
    'B1','B2','B3','B4','B5','B6','B7','B8','B9',
    'L1','L2','L3','L4','L5','L6','L7','L8','L9',
    'M1','M2','M3','M4','M5','M6','M7','M8','M9',
]

def run():
    log.info("Connecting to Supabase Postgres...")
    conn = psycopg.connect(DB_URL, row_factory=dict_row)
    conn.autocommit = False

    # Check starting state
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) as n FROM price_paid_raw_2025 WHERE lat IS NULL")
        start_null = cur.fetchone()["n"]
    log.info(f"Starting null count: {start_null:,}")

    total_updated = 0

    for prefix in PREFIXES:
        t0 = time.time()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE price_paid_raw_2025 p
                    SET lat = n.lat, lng = n.lng
                    FROM nspl_lookup n
                    WHERE n.pcd_nospace = p.postcode_nospace
                      AND p.lat IS NULL
                      AND LEFT(p.postcode_nospace, 2) = %s
                """, (prefix,))
                rows = cur.rowcount
            conn.commit()
            elapsed = time.time() - t0
            total_updated += rows
            log.info(f"  {prefix}: {rows:,} rows updated in {elapsed:.1f}s  (total so far: {total_updated:,})")
        except Exception as e:
            conn.rollback()
            log.error(f"  {prefix}: FAILED — {e}")
            # Continue with next prefix rather than aborting
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
    log.info(f"COMPLETE")
    log.info(f"  Total rows:    {result['total']:,}")
    log.info(f"  Geocoded:      {result['geocoded']:,}  ({result['pct_geocoded']}%)")
    log.info(f"  Still null:    {result['still_null']:,}")
    log.info(f"  Updated now:   {total_updated:,}")
    log.info("=" * 50)

    conn.close()

if __name__ == "__main__":
    run()
