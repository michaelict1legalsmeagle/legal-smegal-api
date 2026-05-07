"""
LegalSmegal — Load EPC CSV into Hetzner Postgres
=================================================
Confirmed column headers from your CSV — no changes needed.

STEP 1: Install dependencies:
  pip3 install psycopg pandas

STEP 2: Run:
  python3 load_epc_hetzner.py

RUNTIME: 40-90 minutes for 5.71GB
EXPECTED: 14-17 million rows after deduplication
"""

import psycopg
import pandas as pd
import os
import glob
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────
DB_URL = "postgresql://legalsmegal:Thesixkids68@159.69.27.104:5432/legalsmegal_data"

# Your confirmed CSV path — update if different
CSV_PATH = os.path.expanduser("~/Downloads/domestic-E-all-domestic-certificates.csv")

BATCH_SIZE = 5000
CHUNK_SIZE = 50_000
# ─────────────────────────────────────────────────────────────────

# Confirmed column mapping from your CSV headers
KEEP_COLS = {
    "certificate_number":      "lmk_key",           # unique cert ID
    "address1":                "address1",
    "address2":                "address2",
    "postcode":                "postcode",
    "property_type":           "property_type",
    "built_form":              "built_form",
    "current_energy_rating":   "current_energy_rating",
    "number_habitable_rooms":  "number_habitable_rooms",
    "total_floor_area":        "total_floor_area",
    "lodgement_date":          "lodgement_date",
}


def create_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS public.epc_certificates (
            lmk_key                 TEXT PRIMARY KEY,
            address1                TEXT,
            address2                TEXT,
            postcode                TEXT,
            property_type           TEXT,
            built_form              TEXT,
            current_energy_rating   TEXT,
            number_habitable_rooms  SMALLINT,
            total_floor_area        NUMERIC(8,2),
            lodgement_date          DATE
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_epc_postcode
        ON public.epc_certificates(postcode)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_epc_address
        ON public.epc_certificates(address1)
    """)
    conn.commit()
    print("Table and indexes ready.")


def load_chunk(df_raw, seen_keys, conn):
    # Normalise column names
    df_raw.columns = [c.strip().lower().strip('"') for c in df_raw.columns]

    available = {k.lower(): v for k, v in KEEP_COLS.items()}
    df = pd.DataFrame()
    for src, dst in available.items():
        if src in df_raw.columns:
            df[dst] = df_raw[src]
        else:
            df[dst] = None

    df = df.dropna(subset=['lmk_key', 'postcode'])
    df['postcode'] = df['postcode'].astype(str).str.strip().str.upper()
    df['lmk_key']  = df['lmk_key'].astype(str).str.strip().str.strip('"')

    # Parse date
    if 'lodgement_date' in df.columns:
        df['lodgement_date'] = pd.to_datetime(
            df['lodgement_date'], errors='coerce'
        ).dt.date

    # Dedup: most recent cert per address+postcode
    df = df.sort_values('lodgement_date', ascending=False, na_position='last')
    df = df.drop_duplicates(subset=['address1', 'postcode'], keep='first')

    # Skip already seen
    df = df[~df['lmk_key'].isin(seen_keys)]
    seen_keys.update(df['lmk_key'].tolist())

    if df.empty:
        return 0

    # Coerce numerics
    if 'number_habitable_rooms' in df.columns:
        df['number_habitable_rooms'] = pd.to_numeric(
            df['number_habitable_rooms'], errors='coerce'
        ).astype('Int64')
    if 'total_floor_area' in df.columns:
        df['total_floor_area'] = pd.to_numeric(
            df['total_floor_area'], errors='coerce'
        )

    col_order = list(KEEP_COLS.values())
    records = []
    for _, row in df.iterrows():
        val = row.get
        rec = tuple(
            None if pd.isna(row.get(c)) else row.get(c)
            for c in col_order
        )
        records.append(rec)

    update_cols = [c for c in col_order if c != 'lmk_key']
    update_str  = ', '.join(f"{c}=EXCLUDED.{c}" for c in update_cols)
    sql = (
        f"INSERT INTO public.epc_certificates ({','.join(col_order)}) "
        f"VALUES ({','.join(['%s'] * len(col_order))}) "
        f"ON CONFLICT (lmk_key) DO UPDATE SET {update_str}"
    )

    for i in range(0, len(records), BATCH_SIZE):
        conn.executemany(sql, records[i:i + BATCH_SIZE])
    conn.commit()
    return len(records)


def main():
    print("=" * 60)
    print("LegalSmegal EPC Loader → Hetzner")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

    if not os.path.isfile(CSV_PATH):
        # Try to find it
        candidates = (
            glob.glob(os.path.expanduser("~/Downloads/*domestic*cert*.csv")) +
            glob.glob(os.path.expanduser("~/Downloads/*epc*.csv")) +
            glob.glob(os.path.expanduser("~/Downloads/*EPC*.csv")) +
            glob.glob(os.path.expanduser("~/Downloads/*.csv"))
        )
        epc_candidates = [c for c in candidates if os.path.getsize(c) > 100_000_000]
        if epc_candidates:
            print(f"CSV_PATH not found. Large CSVs in Downloads:")
            for c in epc_candidates:
                print(f"  {c} ({os.path.getsize(c)/(1024**3):.1f} GB)")
            print(f"\nUpdate CSV_PATH at top of script and re-run.")
            return
        print(f"ERROR: {CSV_PATH} not found.")
        return

    size_gb = os.path.getsize(CSV_PATH) / (1024**3)
    print(f"File: {CSV_PATH}")
    print(f"Size: {size_gb:.2f} GB")
    print(f"Estimated runtime: {int(size_gb * 12)}-{int(size_gb * 18)} minutes")
    print()

    print("Connecting to Hetzner...")
    with psycopg.connect(DB_URL) as conn:
        conn.autocommit = False
        print("Connected.")
        create_table(conn)

        # Check current state
        cur = conn.execute("SELECT COUNT(*) FROM public.epc_certificates")
        existing = cur.fetchone()[0]
        if existing > 0:
            print(f"Table already has {existing:,} rows — will upsert (safe to re-run)")

        seen_keys    = set()
        total_loaded = 0
        chunk_num    = 0

        for enc in ['utf-8', 'latin-1']:
            try:
                reader = pd.read_csv(
                    CSV_PATH,
                    chunksize=CHUNK_SIZE,
                    low_memory=False,
                    encoding=enc,
                    on_bad_lines='skip',
                    quotechar='"'
                )
                for chunk in reader:
                    chunk_num += 1
                    n = load_chunk(chunk, seen_keys, conn)
                    total_loaded += n

                    if chunk_num % 10 == 0:
                        cur = conn.execute(
                            "SELECT COUNT(*) FROM public.epc_certificates"
                        )
                        db_count = cur.fetchone()[0]
                        elapsed  = (datetime.now() - start).seconds // 60
                        pct      = min(100, int((chunk_num * CHUNK_SIZE / (size_gb * 1_400_000)) * 100))
                        print(f"  Chunk {chunk_num:4d} | +{total_loaded:,} loaded | "
                              f"DB: {db_count:,} | {elapsed}m | ~{pct}%")
                break
            except UnicodeDecodeError:
                if enc == 'utf-8':
                    print("UTF-8 failed, retrying with latin-1...")
                    chunk_num = 0
                    total_loaded = 0
                    seen_keys = set()
                    continue
                raise

    print(f"\n{'='*60}")
    with psycopg.connect(DB_URL) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM public.epc_certificates")
        final = cur.fetchone()[0]
        cur2  = conn.execute("""
            SELECT current_energy_rating, COUNT(*) AS n
            FROM public.epc_certificates
            WHERE current_energy_rating IS NOT NULL
            GROUP BY 1 ORDER BY 1
        """)
        print(f"COMPLETE: {final:,} EPC certificates on Hetzner")
        print(f"Total time: {(datetime.now() - start).seconds // 60} minutes")
        print("\nEPC rating distribution:")
        for row in cur2.fetchall():
            print(f"  {row[0]}: {row[1]:,}")


start = datetime.now()
if __name__ == '__main__':
    main()
