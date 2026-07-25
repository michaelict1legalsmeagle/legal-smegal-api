#!/usr/bin/env python3
"""
Item C fix for app.py — two edits, both guarded and reversible:

  1. Repoint _get_imd_for_lsoa from Hetzner (data_query) to Supabase
     (supabase_data_query), where the lsoa_imd table actually lives.
  2. Update the source label from "MHCLG IMD 2019" to "MHCLG IMD 2025",
     since the data being loaded is IoD2025 (matches the app's 2021 LSOA
     codes). Leaving it at 2019 would mislabel the provenance.

Makes a timestamped backup, asserts each target appears exactly once,
refuses to double-apply, and prints the changes.

Usage:
    python3 apply_imd_fix.py               # patches ./app.py
    python3 apply_imd_fix.py /path/to/app.py
"""
import sys, os, time

path = sys.argv[1] if len(sys.argv) > 1 else "app.py"

EDITS = [
    # (label, old, new)
    (
        "repoint IMD read to Supabase",
        '        rows = data_query(\n'
        '            "SELECT imd_rank, imd_decile FROM public.lsoa_imd WHERE lsoa_code = %s LIMIT 1",\n',
        '        rows = supabase_data_query(   # FIX Item C: lsoa_imd lives on Supabase, not Hetzner\n'
        '            "SELECT imd_rank, imd_decile FROM public.lsoa_imd WHERE lsoa_code = %s LIMIT 1",\n',
    ),
    (
        "update source label to 2025",
        '                "source": "MHCLG IMD 2019",\n',
        '                "source": "MHCLG IMD 2025",\n',
    ),
]

if not os.path.isfile(path):
    sys.exit(f"ERROR: {path} not found. Pass the path to app.py as an argument.")

src = open(path, encoding="utf-8").read()
orig = src
applied, skipped = [], []

for label, old, new in EDITS:
    if new in src:
        skipped.append(f"{label} (already applied)")
        continue
    n = src.count(old)
    if n != 1:
        sys.exit(f"ABORT [{label}]: target found {n} times (expected 1). "
                 f"File may differ from the reviewed version — no changes written.")
    src = src.replace(old, new)
    applied.append(label)

if src == orig:
    sys.exit("Nothing to do — both edits already present.")

backup = f"{path}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
open(backup, "w", encoding="utf-8").write(orig)
open(path, "w", encoding="utf-8").write(src)

print(f"Patched {path}")
print(f"Backup  {backup}")
for a in applied:
    print(f"  applied: {a}")
for s in skipped:
    print(f"  skipped: {s}")
print("\nReview, then commit + deploy the backend however you normally ship it.")
