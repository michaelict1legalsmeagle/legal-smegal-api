import os
import csv
import re
from datetime import datetime, date
from typing import Dict, Any, Optional, Tuple, List

from supabase import create_client

SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip()
SUPABASE_KEY = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY") or "").strip()

if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY/SUPABASE_KEY")

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

def to_month(v: str) -> Optional[date]:
    s = (v or "").strip()
    if not s:
        return None

    # Handles "2025-10" or "2025 OCT" or "Oct-25" style cases
    # First try ISO-ish
    try:
        if re.match(r"^\d{4}-\d{2}$", s):
            return date(int(s[:4]), int(s[5:7]), 1)
        if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
            d = datetime.strptime(s[:10], "%Y-%m-%d").date()
            return date(d.year, d.month, 1)
    except Exception:
        pass

    # Try "MMM-YY" / "MMM YYYY"
    for fmt in ("%b-%y", "%b %Y", "%B %Y"):
        try:
            d = datetime.strptime(s, fmt).date()
            return date(d.year, d.month, 1)
        except Exception:
            continue

    return None

def to_num(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None

def guess_cols(header: List[str]) -> Dict[str, str]:
    # We do "best effort" mapping because UK HPI CSV headers vary.
    h = [c.strip() for c in header]

    def pick(*cands: str) -> str:
        for c in cands:
            for col in h:
                if col.lower() == c.lower():
                    return col
        # contains match
        for c in cands:
            for col in h:
                if c.lower() in col.lower():
                    return col
        return ""

    return {
        "month": pick("month", "date"),
        "area_code": pick("area code", "areacode", "geography code", "code"),
        "area_name": pick("area name", "areaname", "geography", "region name", "name"),
        "avg_price": pick("average price", "avg price", "price"),
        "annual": pick("annual change", "annual change (%)", "annual % change", "12-month % change"),
        "monthly": pick("monthly change", "monthly change (%)", "monthly % change", "1-month % change"),
        "property_type": pick("property type", "type"),
    }

def upsert_rows(table: str, rows: List[Dict[str, Any]], chunk: int = 1000) -> None:
    for i in range(0, len(rows), chunk):
        batch = rows[i:i+chunk]
        sb.table(table).upsert(batch).execute()

def ingest_overall(csv_path: str) -> Tuple[int, int]:
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        cols = guess_cols(rdr.fieldnames or [])
        required = ("month", "area_code", "area_name")
        for r in required:
            if not cols[r]:
                raise SystemExit(f"Could not map required column: {r}. Header={rdr.fieldnames}")

        out: List[Dict[str, Any]] = []
        skipped = 0

        for row in rdr:
            m = to_month(row.get(cols["month"], ""))
            code = (row.get(cols["area_code"], "") or "").strip()
            name = (row.get(cols["area_name"], "") or "").strip()
            if not m or not code or not name:
                skipped += 1
                continue

            out.append({
                "month": m.isoformat(),
                "area_code": code,
                "area_name": name,
                "avg_price": to_num(row.get(cols["avg_price"])) if cols["avg_price"] else None,
                "annual_change_pct": to_num(row.get(cols["annual"])) if cols["annual"] else None,
                "monthly_change_pct": to_num(row.get(cols["monthly"])) if cols["monthly"] else None,
                "source": "uk_hpi",
            })

        upsert_rows("uk_hpi_monthly", out)
        return len(out), skipped

def ingest_by_type(csv_path: str) -> Tuple[int, int]:
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        cols = guess_cols(rdr.fieldnames or [])
        required = ("month", "area_code", "area_name", "property_type")
        for r in required:
            if not cols[r]:
                raise SystemExit(f"Could not map required column: {r}. Header={rdr.fieldnames}")

        out: List[Dict[str, Any]] = []
        skipped = 0

        for row in rdr:
            m = to_month(row.get(cols["month"], ""))
            code = (row.get(cols["area_code"], "") or "").strip()
            name = (row.get(cols["area_name"], "") or "").strip()
            ptype = (row.get(cols["property_type"], "") or "").strip()
            if not m or not code or not name or not ptype:
                skipped += 1
                continue

            out.append({
                "month": m.isoformat(),
                "area_code": code,
                "area_name": name,
                "property_type": ptype,
                "avg_price": to_num(row.get(cols["avg_price"])) if cols["avg_price"] else None,
                "annual_change_pct": to_num(row.get(cols["annual"])) if cols["annual"] else None,
                "monthly_change_pct": to_num(row.get(cols["monthly"])) if cols["monthly"] else None,
                "source": "uk_hpi",
            })

        upsert_rows("uk_hpi_monthly_by_property_type", out)
        return len(out), skipped

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--overall", help="Path to Average price CSV")
    ap.add_argument("--by-type", help="Path to Average price by property type CSV")
    args = ap.parse_args()

    if not args.overall and not args.by_type:
        raise SystemExit("Provide --overall and/or --by-type")

    if args.overall:
        n, s = ingest_overall(args.overall)
        print(f"✅ overall: upserted={n} skipped={s}")

    if args.by_type:
        n, s = ingest_by_type(args.by_type)
        print(f"✅ by-type: upserted={n} skipped={s}")
