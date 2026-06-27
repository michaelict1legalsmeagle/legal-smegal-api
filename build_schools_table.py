"""
LegalSmegal — One-time merge: build a single unified `schools` table on Hetzner.

WHY THIS EXISTS
----------------
Two separate school datasets currently exist with no overlap in fields:

  Hetzner   public.schools_ofsted   (9,781 rows)
            urn, school_name, postcode, ofsted_rating, ofsted_label, lat, lng
            -> Ofsted-INSPECTED schools only. Has rating + geocoding,
               missing phase/establishment_type/local_authority.

  Supabase  public.schools_clean_v2 (50,688 rows, full DfE register)
            urn, name, establishment_type, phase, local_authority, town,
            postcode, status, telephone, website
            -> Full open-school register (England + Wales + FE/HE/etc).
               Has phase/type/LA, missing ofsted_rating and lat/lng.

get_schools_data() in app.py was written assuming ONE table had all of
this — it never existed. This script builds it for real, once, verified
against live data on both sides (see chat history 2026-06-27 for the
full audit: confirmed via direct psql on Hetzner and Supabase:execute_sql
on Supabase — not guessed).

WHAT THIS SCRIPT DOES
----------------------
1. Pulls the full open, English, mainstream-adjacent register from
   Supabase schools_clean_v2 (~24,563 rows expected, confirmed by audit).
2. Pulls all 9,781 rows from Hetzner schools_ofsted (the Ofsted-rated
   subset, already geocoded).
3. LEFT JOINs by urn: every row from the full register gets an Ofsted
   rating attached wherever a matching urn exists in the smaller
   Ofsted-rated table. No match -> NULL rating (existing frontend
   behaviour already handles this: "Ofsted report not available").
4. Geocodes every row via Hetzner's nspl_postcodes (lat/lng), since most
   of the larger register doesn't have this and schools_ofsted only
   covers the smaller Ofsted-rated subset.
5. Writes the merged result to a NEW Hetzner table: public.schools
   Does NOT touch or drop schools_ofsted or schools_clean_v2 — both stay
   in place as historical record until the new table is verified in
   production.

HOW TO RUN
----------
This is a one-time job. Run it from a machine that can reach BOTH
Supabase (via SUPABASE_URL/SUPABASE_SERVICE_KEY) and Hetzner (via
DATA_DATABASE_URL) — e.g. locally with both env vars set, or as a
Render one-off job.

    pip install psycopg[binary] supabase --break-system-packages
    export SUPABASE_URL=...
    export SUPABASE_SERVICE_KEY=...
    export DATA_DATABASE_URL=postgresql://legalsmegal:<password>@159.69.27.104:5432/legalsmegal_data
    python3 build_schools_table.py

VERIFICATION (run manually after, before flipping app.py over)
----------------------------------------------------------------
    SELECT COUNT(*) FROM public.schools;                          -- expect ~24,000-25,000
    SELECT COUNT(*) FROM public.schools WHERE ofsted_rating IS NOT NULL;  -- expect <= 9781
    SELECT COUNT(*) FROM public.schools WHERE lat IS NULL;         -- check geocoding gaps
"""

import os
import sys
import psycopg
from psycopg.rows import dict_row

DATA_DATABASE_URL = os.environ.get("DATA_DATABASE_URL")
if not DATA_DATABASE_URL:
    raise RuntimeError("DATA_DATABASE_URL is required (Hetzner connection string).")

# Matches the exact env var names app.py already uses (confirmed 2026-06-27):
# SUPABASE_SERVICE_ROLE_KEY preferred, SUPABASE_KEY as fallback.
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_KEY) are required.")

from supabase import create_client  # noqa: E402

# Mainstream-adjacent, open-only filter — matches the audit's confirmed
# "open_england_schools_only" definition (24,563 rows at audit time).
EXCLUDED_TYPES = (
    "Welsh establishment", "Further education", "Higher education institutions",
    "Offshore schools", "Special post 16 institution", "Miscellaneous",
)


def fetch_supabase_register(supabase) -> list[dict]:
    """Pull the full open, English, mainstream-adjacent school register.

    H4-FIX (2026-06-27): first run of this script stopped after exactly 999
    rows because the original loop treated "got fewer rows than page_size"
    as "no more data" — but Supabase/PostgREST applies its own server-side
    row cap per request (independent of the .range() bounds requested),
    so a short page does NOT mean the table is exhausted. The only safe
    stop condition is a genuinely EMPTY page. Confirmed bug: schools_clean_v2
    has ~24,563 matching rows; first run produced only 999.
    """
    print("Fetching schools_clean_v2 from Supabase (paginated)...")
    all_rows: list[dict] = []
    page_size = 1000
    offset = 0
    while True:
        res = (
            supabase.table("schools_clean_v2")
            .select("urn,name,establishment_type,phase,local_authority,town,postcode,status")
            .eq("status", "Open")
            .not_.in_("establishment_type", list(EXCLUDED_TYPES))
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = res.data if hasattr(res, "data") else []
        if not rows:
            break  # genuinely empty page — this IS the end of the data
        all_rows.extend(rows)
        offset += len(rows)  # advance by ACTUAL rows returned, not page_size —
                              # a short page (server-side cap) is not the end
        print(f"  ...{len(all_rows)} rows so far")
    print(f"Fetched {len(all_rows)} rows from schools_clean_v2.")
    return all_rows


def fetch_hetzner_ofsted(conn) -> dict[str, dict]:
    """Pull all Ofsted-rated rows from Hetzner, keyed by urn."""
    print("Fetching schools_ofsted from Hetzner...")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT urn, ofsted_rating, ofsted_label, lat, lng FROM public.schools_ofsted"
        )
        rows = cur.fetchall()
    by_urn = {r["urn"]: r for r in rows}
    print(f"Fetched {len(by_urn)} rows from schools_ofsted.")
    return by_urn


def geocode_postcode(conn, postcode: str) -> tuple:
    """Look up lat/lng for a postcode via nspl_postcodes. Returns (lat, lng) or (None, None)."""
    if not postcode:
        return (None, None)
    pcd_nospace = postcode.replace(" ", "").upper()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT lat, lng FROM public.nspl_postcodes WHERE pcd_nospace = %s LIMIT 1",
            (pcd_nospace,),
        )
        row = cur.fetchone()
    if row:
        return (row["lat"], row["lng"])
    return (None, None)


def main():
    print("=" * 70)
    print("LegalSmegal — building unified public.schools table on Hetzner")
    print("=" * 70)

    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    with psycopg.connect(DATA_DATABASE_URL, row_factory=dict_row) as conn:
        register_rows = fetch_supabase_register(supabase)

        # H4-SANITY (2026-06-27): the audit confirmed ~24,563 rows match this
        # filter. A first run silently produced only 999 due to a pagination
        # bug (since fixed) and reported "DONE" without any warning. Refuse
        # to proceed if the count is far below the audited expectation —
        # better to stop loudly than build a table that's 96% incomplete.
        EXPECTED_MIN_ROWS = 20000
        if len(register_rows) < EXPECTED_MIN_ROWS:
            print("=" * 70)
            print(f"ABORTING: only fetched {len(register_rows)} rows from "
                  f"schools_clean_v2, expected >= {EXPECTED_MIN_ROWS} "
                  f"(audited count was ~24,563). This looks like a partial "
                  f"fetch, not the real dataset. public.schools was NOT "
                  f"created or modified.")
            print("=" * 70)
            sys.exit(1)
        ofsted_by_urn = fetch_hetzner_ofsted(conn)

        print("Geocoding register rows via nspl_postcodes (this may take a while)...")
        merged = []
        geocode_misses = 0
        ofsted_matches = 0

        for i, r in enumerate(register_rows, 1):
            urn = r.get("urn")
            ofsted = ofsted_by_urn.get(urn)

            if ofsted and ofsted.get("lat") is not None and ofsted.get("lng") is not None:
                # Already geocoded via schools_ofsted — reuse it, skip a lookup.
                lat, lng = ofsted["lat"], ofsted["lng"]
            else:
                lat, lng = geocode_postcode(conn, r.get("postcode") or "")
                if lat is None:
                    geocode_misses += 1

            if ofsted:
                ofsted_matches += 1

            merged.append({
                "urn": urn,
                "school_name": r.get("name"),
                "postcode": r.get("postcode"),
                "lat": lat,
                "lng": lng,
                "ofsted_rating": ofsted.get("ofsted_rating") if ofsted else None,
                "ofsted_label": ofsted.get("ofsted_label") if ofsted else None,
                "phase": r.get("phase"),
                "establishment_type": r.get("establishment_type"),
                "local_authority": r.get("local_authority"),
            })

            if i % 2000 == 0:
                print(f"  ...processed {i}/{len(register_rows)}")

        print(f"Done. {ofsted_matches} rows matched an Ofsted rating. "
              f"{geocode_misses} rows could not be geocoded.")

        print("Creating public.schools table (drop+recreate if it already exists)...")
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS public.schools;")
            cur.execute("""
                CREATE TABLE public.schools (
                    urn TEXT PRIMARY KEY,
                    school_name TEXT,
                    postcode TEXT,
                    lat DOUBLE PRECISION,
                    lng DOUBLE PRECISION,
                    ofsted_rating INTEGER,
                    ofsted_label TEXT,
                    phase TEXT,
                    establishment_type TEXT,
                    local_authority TEXT
                );
            """)
            cur.execute("CREATE INDEX idx_schools_postcode ON public.schools (postcode);")
            cur.execute("CREATE INDEX idx_schools_lat_lng ON public.schools (lat, lng);")
        conn.commit()

        print(f"Inserting {len(merged)} merged rows...")
        with conn.cursor() as cur:
            with cur.copy(
                "COPY public.schools (urn, school_name, postcode, lat, lng, "
                "ofsted_rating, ofsted_label, phase, establishment_type, local_authority) "
                "FROM STDIN"
            ) as copy:
                for row in merged:
                    copy.write_row((
                        row["urn"], row["school_name"], row["postcode"],
                        row["lat"], row["lng"], row["ofsted_rating"],
                        row["ofsted_label"], row["phase"],
                        row["establishment_type"], row["local_authority"],
                    ))
        conn.commit()

        print("Verifying...")
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS total FROM public.schools;")
            total = cur.fetchone()["total"]
            cur.execute("SELECT COUNT(*) AS n FROM public.schools WHERE ofsted_rating IS NOT NULL;")
            with_ofsted = cur.fetchone()["n"]
            cur.execute("SELECT COUNT(*) AS n FROM public.schools WHERE lat IS NULL;")
            no_geo = cur.fetchone()["n"]

        print("=" * 70)
        print(f"DONE. public.schools created with {total} rows.")
        print(f"  {with_ofsted} rows have an Ofsted rating.")
        print(f"  {no_geo} rows could not be geocoded (lat IS NULL).")
        print("schools_ofsted and schools_clean_v2 were NOT modified or dropped.")
        print("=" * 70)


if __name__ == "__main__":
    main()
