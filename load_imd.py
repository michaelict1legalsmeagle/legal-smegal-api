"""
LegalSmegal — Load IMD 2019 data into Supabase lsoa_imd table

STEP 1: Download the file from:
  https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019
  -> File 7: All ranks, deciles and scores
  Filename: File_7_-_All_IoD2019_Scores__Ranks__Deciles_and_Population_Denominators_3.csv

STEP 2: Run this script from Terminal:
  python3 load_imd.py

STEP 3: Verify in Supabase SQL editor:
  SELECT imd_decile, COUNT(*) FROM public.lsoa_imd GROUP BY 1 ORDER BY 1;
"""

import psycopg2
import psycopg2.extras
import pandas as pd
import os

DB_URL  = "postgresql://postgres:Thesixkids68@db.qdgxmwdvmfrcicgpukhs.supabase.co:5432/postgres"
CSV_PATH = os.path.expanduser("~/Downloads/File_7_-_All_IoD2019_Scores__Ranks__Deciles_and_Population_Denominators_3.csv")

print("Loading IMD CSV...")
df = pd.read_csv(CSV_PATH, encoding="latin-1")
print(f"Columns: {list(df.columns)}")
print(f"Shape: {df.shape}")

# Column mapping — File 7 standard column names
# Adjust if your file uses different headers
col_map = {
    "LSOA code (2011)":                        "lsoa_code",
    "LSOA name (2011)":                         "lsoa_name",
    "Local Authority District code (2019)":     "lad_code",
    "Local Authority District name (2019)":     "lad_name",
    "Index of Multiple Deprivation (IMD) Score": "imd_score",
    "Index of Multiple Deprivation (IMD) Rank (where 1 is most deprived)": "imd_rank",
    "Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)": "imd_decile",
}

# Try to map columns
df_out = pd.DataFrame()
for src, dst in col_map.items():
    if src in df.columns:
        df_out[dst] = df[src]
    else:
        # Try partial match
        matches = [c for c in df.columns if dst.replace("_", " ") in c.lower() or src[:20].lower() in c.lower()]
        if matches:
            df_out[dst] = df[matches[0]]
            print(f"  Mapped {matches[0]} -> {dst}")
        else:
            print(f"  WARNING: Column not found: {src}")

print(f"\nMapped shape: {df_out.shape}")
print(df_out.head(3).to_string())

# Connect and load
print("\nConnecting to Supabase...")
conn = psycopg2.connect(DB_URL)
conn.autocommit = False
cur = conn.cursor()

print("Creating table...")
cur.execute("""
    CREATE TABLE IF NOT EXISTS public.lsoa_imd (
        lsoa_code   TEXT PRIMARY KEY,
        lsoa_name   TEXT,
        lad_code    TEXT,
        lad_name    TEXT,
        imd_score   NUMERIC,
        imd_rank    INTEGER,
        imd_decile  INTEGER
    );
""")
cur.execute("TRUNCATE public.lsoa_imd;")
conn.commit()

records = []
for _, row in df_out.iterrows():
    records.append((
        str(row.get("lsoa_code", "") or "").strip(),
        str(row.get("lsoa_name", "") or "").strip(),
        str(row.get("lad_code", "") or "").strip(),
        str(row.get("lad_name", "") or "").strip(),
        float(row["imd_score"]) if pd.notna(row.get("imd_score")) else None,
        int(row["imd_rank"])    if pd.notna(row.get("imd_rank"))  else None,
        int(row["imd_decile"])  if pd.notna(row.get("imd_decile")) else None,
    ))

print(f"Inserting {len(records)} rows...")
for i in range(0, len(records), 1000):
    chunk = records[i:i+1000]
    psycopg2.extras.execute_values(
        cur,
        "INSERT INTO public.lsoa_imd (lsoa_code,lsoa_name,lad_code,lad_name,imd_score,imd_rank,imd_decile) VALUES %s ON CONFLICT (lsoa_code) DO UPDATE SET imd_score=EXCLUDED.imd_score, imd_rank=EXCLUDED.imd_rank, imd_decile=EXCLUDED.imd_decile",
        chunk
    )
    conn.commit()
    print(f"  {min(i+1000, len(records))}/{len(records)}")

cur.execute("CREATE INDEX IF NOT EXISTS idx_lsoa_imd_code ON public.lsoa_imd(lsoa_code);")
cur.execute("CREATE INDEX IF NOT EXISTS idx_lsoa_imd_decile ON public.lsoa_imd(imd_decile);")
conn.commit()

cur.execute("SELECT imd_decile, COUNT(*) FROM public.lsoa_imd GROUP BY 1 ORDER BY 1;")
print("\nVerification:")
for row in cur.fetchall():
    print(f"  Decile {row[0]}: {row[1]:,} LSOAs")

cur.execute("SELECT COUNT(*) FROM public.lsoa_imd;")
print(f"Total: {cur.fetchone()[0]:,} LSOAs loaded")

cur.close()
conn.close()
print("\n=== IMD load complete ===")
