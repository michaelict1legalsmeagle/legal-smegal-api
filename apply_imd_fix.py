#!/usr/bin/env python3
"""
Item C fix — repoint _get_imd_for_lsoa from Hetzner (data_query) to
Supabase (supabase_data_query), where the lsoa_imd table actually lives.

Safe to run once. It:
  - makes a timestamped backup of app.py first
  - asserts the target line appears EXACTLY once (uniqueness guard)
  - refuses to run if the fix is already applied
  - prints the changed lines so you can eyeball before committing

Usage:
    python3 apply_imd_fix.py               # patches ./app.py
    python3 apply_imd_fix.py /path/to/app.py
"""
import sys, os, time, difflib

path = sys.argv[1] if len(sys.argv) > 1 else "app.py"

OLD = (
    '        rows = data_query(\n'
    '            "SELECT imd_rank, imd_decile FROM public.lsoa_imd WHERE lsoa_code = %s LIMIT 1",\n'
)
NEW = (
    '        rows = supabase_data_query(   # FIX Item C: lsoa_imd lives on Supabase, not Hetzner\n'
    '            "SELECT imd_rank, imd_decile FROM public.lsoa_imd WHERE lsoa_code = %s LIMIT 1",\n'
)

if not os.path.isfile(path):
    sys.exit(f"ERROR: {path} not found. Pass the path to app.py as an argument.")

src = open(path, encoding="utf-8").read()

# Already applied?
if NEW.split("\n")[0] in src or 'supabase_data_query(   # FIX Item C' in src:
    sys.exit("Already patched — nothing to do.")

# Uniqueness guard — the IMD SQL string makes this 2-line block unique
# even though data_query( appears many times in app.py.
n = src.count(OLD)
if n != 1:
    sys.exit(f"ABORT: target block found {n} times (expected exactly 1). "
             f"File may differ from the reviewed version — not touching it.")

new_src = src.replace(OLD, NEW)
assert new_src != src and new_src.count(NEW) == 1, "internal check failed"

backup = f"{path}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
with open(backup, "w", encoding="utf-8") as f:
    f.write(src)

with open(path, "w", encoding="utf-8") as f:
    f.write(new_src)

print(f"Patched {path}")
print(f"Backup  {backup}")
print("\n--- change ---")
diff = difflib.unified_diff(
    OLD.splitlines(), NEW.splitlines(),
    lineterm="", n=0
)
for line in diff:
    if line and line[0] in "+-" and not line.startswith(("+++", "---")):
        print(line)
print("\nReview, then commit + deploy however you normally ship the backend.")
