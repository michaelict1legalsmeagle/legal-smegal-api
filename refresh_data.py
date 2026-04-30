"""
LegalSmegal — Monthly Data Refresh
===================================
Runs on the 21st of each month via Render Cron Job.
Downloads latest Land Registry HPI and Price Paid data and upserts into Supabase.

Render Cron Job config:
  Command: python refresh_data.py
  Schedule: 0 6 21 * *   (6am on the 21st of every month)
  Environment: same as API service (needs SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY)

Backfill (run once manually to catch up missing years):
  python refresh_data.py --backfill            # backfills last 2 years (e.g. 2025 + 2026)
  python refresh_data.py --backfill --all      # backfills 2018 through 2026

Data sources (all free, no API key required):
  HPI:        https://www.gov.uk/government/statistical-data-sets/uk-house-price-index-data-downloads-november-2024
  Price Paid: https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads
"""
import os
import io
import csv
import sys
import logging
import requests
from datetime import datetime
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
log = logging.getLogger(__name__)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BATCH_SIZE = 500  # rows per upsert batch — keeps memory low on Render free tier

# ── LAND REGISTRY HPI ────────────────────────────────────────────────────────
HPI_URL           = "https://www.gov.uk/government/uploads/system/uploads/attachment_data/file/UK-HPI-full-file-{year}-{month:02d}.csv"
HPI_TABLE_MONTHLY = "uk_hpi_monthly"
HPI_TABLE_BY_TYPE = "uk_hpi_monthly_by_property_type"


def get_latest_hpi_url() -> str:
    now = datetime.utcnow()
    # LR publishes ~2 months behind — try up to 4 months back
    for m_offset in [0, 1, 2, 3, 4]:
        month = now.month - m_offset
        year  = now.year
        if month <= 0:
            month += 12
            year -= 1
        url = HPI_URL.format(year=year, month=month)
        try:
            r = requests.head(url, timeout=10)
            if r.status_code == 200:
                log.info(f"HPI URL found: {url}")
                return url
        except Exception:
            pass
    fallback = HPI_URL.format(year=2026, month=2)
    log.warning(f"HPI: no live URL found in range, using fallback {fallback}")
    return fallback


def refresh_hpi():
    log.info("Starting HPI refresh...")
    url = get_latest_hpi_url()
    try:
        r = requests.get(url, timeout=120, stream=True)
        r.raise_for_status()
    except Exception as e:
        log.error(f"HPI download failed: {e}")
        return

    content       = r.content.decode("utf-8", errors="replace")
    reader        = csv.DictReader(io.StringIO(content))
    monthly_batch = []
    by_type_batch = []
    monthly_count = 0
    by_type_count = 0

    for row in reader:
        try:
            area_code  = (row.get("RegionName") or row.get("AreaCode") or "").strip()
            date_str   = (row.get("Date") or "").strip()
            avg_price  = row.get("AveragePrice") or row.get("Average price") or None
            pct_change = row.get("AnnualChange") or row.get("12m % change") or None
            if not area_code or not date_str:
                continue
            if "/" in date_str:
                parts = date_str.split("/")
                if len(parts) == 3:
                    date_str = f"{parts[2]}-{parts[1]}-{parts[0]}"
            monthly_batch.append({
                "area_code":     area_code,
                "date":          date_str,
                "avg_price":     float(avg_price) if avg_price else None,
                "annual_change": float(pct_change) if pct_change else None,
                "region_name":   row.get("RegionName") or area_code,
            })
            for prop_type, col_avg, col_chg in [
                ("detached",  "DetachedPrice",     "DetachedAnnualChange"),
                ("semi",      "SemiDetachedPrice",  "SemiDetachedAnnualChange"),
                ("terraced",  "TerracedPrice",     "TerracedAnnualChange"),
                ("flat",      "FlatPrice",         "FlatAnnualChange"),
            ]:
                avg = row.get(col_avg) or row.get(f"{prop_type.capitalize()} Average Price")
                chg = row.get(col_chg)
                if avg:
                    by_type_batch.append({
                        "area_code":     area_code,
                        "date":          date_str,
                        "property_type": prop_type,
                        "avg_price":     float(avg),
                        "annual_change": float(chg) if chg else None,
                    })
            if len(monthly_batch) >= BATCH_SIZE:
                supabase.table(HPI_TABLE_MONTHLY).upsert(monthly_batch, on_conflict="area_code,date").execute()
                monthly_count += len(monthly_batch)
                monthly_batch = []
                log.info(f"HPI monthly: {monthly_count} rows upserted")
            if len(by_type_batch) >= BATCH_SIZE:
                supabase.table(HPI_TABLE_BY_TYPE).upsert(by_type_batch, on_conflict="area_code,date,property_type").execute()
                by_type_count += len(by_type_batch)
                by_type_batch = []
        except Exception as e:
            log.warning(f"HPI row error: {e} — row: {dict(list(row.items())[:4])}")
            continue

    if monthly_batch:
        supabase.table(HPI_TABLE_MONTHLY).upsert(monthly_batch, on_conflict="area_code,date").execute()
        monthly_count += len(monthly_batch)
    if by_type_batch:
        supabase.table(HPI_TABLE_BY_TYPE).upsert(by_type_batch, on_conflict="area_code,date,property_type").execute()
        by_type_count += len(by_type_batch)
    log.info(f"HPI refresh complete: {monthly_count} monthly rows, {by_type_count} by-type rows")


# ── LAND REGISTRY PRICE PAID ──────────────────────────────────────────────────
# Upsert target is the base table — price_paid_geo is a materialized view built from this
PP_GEO_TABLE = "price_paid_raw_2025"

# Confirmed live URLs as of April 2026 — LR moved to dedicated subdomain
PP_MONTHLY_URL = "https://price-paid-data.publicdata.landregistry.gov.uk/pp-monthly-update-new-version.csv"
PP_ANNUAL_URL  = "https://price-paid-data.publicdata.landregistry.gov.uk/pp-{year}.csv"

# All confirmed annual files
PP_ANNUAL_YEARS = list(range(2018, 2027))  # 2018–2026


def get_latest_pp_url() -> str:
    now = datetime.utcnow()
    candidates = [
        # Monthly update — confirmed live April 2026, contains current month only
        PP_MONTHLY_URL,
        # Annual file for current year as fallback if monthly ever fails
        PP_ANNUAL_URL.format(year=now.year),
        PP_ANNUAL_URL.format(year=now.year - 1),
    ]
    for url in candidates:
        try:
            r = requests.head(url, timeout=10, allow_redirects=True)
            if r.status_code == 200:
                log.info(f"PP URL found: {url}")
                return url
            else:
                log.debug(f"PP URL {r.status_code}: {url}")
        except Exception as e:
            log.debug(f"PP URL check failed ({url}): {e}")
    log.error(
        "PP: no working URL found — skipping. "
        "Check https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads"
    )
    return ""


_nspl_cache: dict = {}


def _get_nspl_lat_lng(pcd_nospace: str) -> tuple:
    if pcd_nospace in _nspl_cache:
        return _nspl_cache[pcd_nospace]
    try:
        res = supabase.table("nspl_lookup").select("lat,lng").eq("pcd_nospace", pcd_nospace).limit(1).execute()
        if res.data:
            lat = res.data[0].get("lat")
            lng = res.data[0].get("lng")
            _nspl_cache[pcd_nospace] = (lat, lng)
            return lat, lng
    except Exception:
        pass
    _nspl_cache[pcd_nospace] = (None, None)
    return None, None


def _upsert_pp_rows_from_csv(content_str: str, label: str):
    """Parse PP CSV content and upsert into price_paid_geo. Returns (count, skipped)."""
    reader  = csv.reader(io.StringIO(content_str))
    batch   = []
    count   = 0
    skipped = 0
    for row in reader:
        if len(row) < 15:
            continue
        try:
            raw_pc = row[3].strip().upper()
            pcd_ns = raw_pc.replace(" ", "")
            if not pcd_ns:
                continue
            record = {
                "transaction_unique_identifier": row[0].strip("{}"),
                "price":             int(row[1]) if row[1].strip().isdigit() else None,
                "date_of_transfer":  row[2][:10] if row[2] else None,
                "postcode":          raw_pc,
                "property_type":     row[4].strip(),
                "old_new":           row[5].strip(),
                "duration":          row[6].strip(),
                "paon":              row[7].strip() or None,
                "saon":              row[8].strip() or None,
                "street":            row[9].strip() or None,
                "locality":          row[10].strip() or None,
                "town_city":         row[11].strip().title() if row[11] else None,
                "district":          row[12].strip().title() if row[12] else None,
                "county":            row[13].strip().title() if row[13] else None,
                "ppd_category_type": row[14].strip() if len(row) > 14 else None,
                "record_status":     row[15].strip() if len(row) > 15 else None,
            }
            if not record["transaction_unique_identifier"] or not record["price"]:
                skipped += 1
                continue
            batch.append(record)
            if len(batch) >= BATCH_SIZE:
                supabase.table(PP_GEO_TABLE).upsert(
                    batch, on_conflict="transaction_unique_identifier"
                ).execute()
                count += len(batch)
                batch = []
                log.info(f"price_paid_geo [{label}]: {count} rows upserted")
        except Exception as e:
            log.warning(f"PP geo row error [{label}]: {e}")
            continue
    if batch:
        supabase.table(PP_GEO_TABLE).upsert(
            batch, on_conflict="transaction_unique_identifier"
        ).execute()
        count += len(batch)
    return count, skipped


def refresh_price_paid():
    """Download and upsert the current monthly PP update file."""
    log.info("Starting price_paid_geo refresh from monthly update...")
    url = get_latest_pp_url()
    if not url:
        log.warning("Price paid refresh skipped — no valid URL found")
        return
    try:
        r = requests.get(url, timeout=300, stream=True)
        if r.status_code != 200:
            log.warning(f"PP download returned {r.status_code} for {url} — skipping")
            return
    except Exception as e:
        log.error(f"Price paid download failed: {e}")
        return
    count, skipped = _upsert_pp_rows_from_csv(
        r.content.decode("utf-8", errors="replace"), label="monthly"
    )
    log.info(f"price_paid_geo monthly refresh complete: {count} rows upserted, {skipped} skipped")


def backfill_price_paid(years: list):
    """
    Download full annual PP files and upsert into price_paid_geo.
    Safe to re-run — upserts on transaction_unique_identifier so no duplicates.

    Annual files confirmed at:
      https://price-paid-data.publicdata.landregistry.gov.uk/pp-{year}.csv
      Available: 2018–2026

    Usage:
      python refresh_data.py --backfill            # last 2 years
      python refresh_data.py --backfill --all      # 2018–2026
    """
    log.info(f"Starting price paid backfill for years: {years}")
    for year in years:
        url = PP_ANNUAL_URL.format(year=year)
        log.info(f"Backfill {year}: downloading {url}")
        try:
            r = requests.get(url, timeout=600, stream=True)
            if r.status_code != 200:
                log.warning(f"Backfill {year}: returned {r.status_code} — skipping")
                continue
        except Exception as e:
            log.error(f"Backfill {year}: download failed: {e}")
            continue
        count, skipped = _upsert_pp_rows_from_csv(
            r.content.decode("utf-8", errors="replace"), label=str(year)
        )
        log.info(f"Backfill {year} complete: {count} rows upserted, {skipped} skipped")
        # Clear NSPL cache between years to avoid unbounded memory growth on large runs
        _nspl_cache.clear()
    log.info("Backfill complete — running materialized view refresh")
    refresh_materialized_view()


# ── MATERIALIZED VIEW ─────────────────────────────────────────────────────────
def refresh_materialized_view():
    """
    Refresh price_paid_geo materialized view so nspl_lat/nspl_lng populate via JOIN.
    This is what makes housing_comps_v1 return results (filters WHERE nspl_lat IS NOT NULL).
    Uses exec_refresh_price_paid_geo RPC — confirmed working April 2026.
    psycopg2 direct-SQL fallback removed — not installed on Render, RPC is the correct path.
    """
    log.info("Refreshing materialized view price_paid_geo...")
    try:
        supabase.rpc("exec_refresh_price_paid_geo", {}).execute()
        log.info("Materialized view refresh complete")
    except Exception as e:
        log.error(
            f"Materialized view RPC refresh failed: {e}. "
            "Run manually in Supabase SQL editor: REFRESH MATERIALIZED VIEW public.price_paid_geo;"
        )


# ── SCHOOLS (OFSTED) ──────────────────────────────────────────────────────────
SCHOOLS_URL   = "https://www.compare-school-performance.service.gov.uk/api/upload/national/england_schoolinformation.csv"
SCHOOLS_TABLE = "schools_clean_v2"


def refresh_schools():
    log.info("Starting schools refresh...")
    try:
        r = requests.get(SCHOOLS_URL, timeout=120)
        r.raise_for_status()
    except Exception as e:
        log.warning(f"Schools download failed (non-fatal): {e}")
        return
    content = r.content.decode("utf-8", errors="replace")
    reader  = csv.DictReader(io.StringIO(content))
    batch   = []
    count   = 0
    for row in reader:
        try:
            urn = row.get("URN") or row.get("urn")
            if not urn:
                continue
            batch.append({
                "urn":           str(urn),
                "school_name":   row.get("EstablishmentName") or row.get("school_name") or "",
                "postcode":      (row.get("Postcode") or row.get("postcode") or "").strip().upper(),
                "ofsted_rating": row.get("OfstedRating") or row.get("ofsted_rating") or "",
                "school_type":   row.get("TypeOfEstablishment (name)") or row.get("school_type") or "",
                "phase":         row.get("PhaseOfEducation (name)") or "",
                "la_name":       row.get("LA (name)") or "",
            })
            if len(batch) >= BATCH_SIZE:
                supabase.table(SCHOOLS_TABLE).upsert(batch, on_conflict="urn").execute()
                count += len(batch)
                batch = []
        except Exception as e:
            log.warning(f"Schools row error: {e}")
    if batch:
        supabase.table(SCHOOLS_TABLE).upsert(batch, on_conflict="urn").execute()
        count += len(batch)
    log.info(f"Schools refresh complete: {count} rows upserted")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = sys.argv[1:]

    if "--backfill" in args:
        # Manual backfill mode — not run by cron
        if "--all" in args:
            years = PP_ANNUAL_YEARS  # 2018–2026
        else:
            # Default: backfill last 2 years to catch gaps
            now   = datetime.utcnow()
            years = [now.year - 1, now.year]
        log.info(f"Backfill mode — years: {years}")
        backfill_price_paid(years)
    else:
        # Normal monthly cron run
        log.info(f"LegalSmegal data refresh started — {datetime.utcnow().isoformat()}")
        refresh_hpi()
        refresh_price_paid()
        refresh_materialized_view()
        month = datetime.utcnow().month
        if month in (1, 4, 9):
            refresh_schools()
        else:
            log.info(f"Skipping schools refresh (month={month}, runs in Jan/Apr/Sep)")
        log.info(f"Data refresh complete — {datetime.utcnow().isoformat()}")
