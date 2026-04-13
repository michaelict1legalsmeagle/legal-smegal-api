"""
LegalSmegal — Monthly Data Refresh
===================================
Runs on the 21st of each month via Render Cron Job.
Downloads latest Land Registry HPI and Price Paid data and upserts into Supabase.

Render Cron Job config:
  Command: python refresh_data.py
  Schedule: 0 6 21 * *   (6am on the 21st of every month)
  Environment: same as API service (needs SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY)

Data sources (all free, no API key required):
  HPI:        https://www.gov.uk/government/statistical-data-sets/uk-house-price-index-data-downloads-november-2024
  Price Paid: https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads
"""

import os
import io
import csv
import gzip
import logging
import requests
from datetime import datetime, date
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
log = logging.getLogger(__name__)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]  # needs service role for bulk upsert

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BATCH_SIZE = 500  # rows per upsert batch — keeps memory low on Render free tier


# ── LAND REGISTRY HPI ───────────────────────────────────────────────────────
# Published: third Wednesday of each month
# URL pattern: https://www.gov.uk/government/uploads/system/uploads/attachment_data/file/...
# Stable latest-version URL:
HPI_URL = "https://www.gov.uk/government/uploads/system/uploads/attachment_data/file/UK-HPI-full-file-{year}-{month:02d}.csv"
HPI_TABLE_MONTHLY     = "uk_hpi_monthly"
HPI_TABLE_BY_TYPE     = "uk_hpi_monthly_by_property_type"


def get_latest_hpi_url() -> str:
    """Land Registry publishes HPI on the third Wednesday. Try current month, fall back to previous."""
    now = datetime.utcnow()
    for m_offset in [0, 1, 2]:
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
    # Fallback to known stable URL
    return "https://www.gov.uk/government/uploads/system/uploads/attachment_data/file/UK-HPI-full-file-2025-12.csv"


def refresh_hpi():
    """Download and upsert UK HPI data into Supabase."""
    log.info("Starting HPI refresh...")
    url = get_latest_hpi_url()

    try:
        r = requests.get(url, timeout=120, stream=True)
        r.raise_for_status()
    except Exception as e:
        log.error(f"HPI download failed: {e}")
        return

    content = r.content.decode("utf-8", errors="replace")
    reader  = csv.DictReader(io.StringIO(content))

    monthly_batch     = []
    by_type_batch     = []
    monthly_count     = 0
    by_type_count     = 0

    for row in reader:
        try:
            # Area-level monthly record
            area_code = (row.get("RegionName") or row.get("AreaCode") or "").strip()
            date_str  = (row.get("Date") or "").strip()  # format: YYYY-MM-DD or DD/MM/YYYY
            avg_price = row.get("AveragePrice") or row.get("Average price") or None
            pct_change = row.get("AnnualChange") or row.get("12m % change") or None

            if not area_code or not date_str:
                continue

            # Normalise date
            if "/" in date_str:
                parts = date_str.split("/")
                if len(parts) == 3:
                    date_str = f"{parts[2]}-{parts[1]}-{parts[0]}"

            monthly_row = {
                "area_code":    area_code,
                "date":         date_str,
                "avg_price":    float(avg_price) if avg_price else None,
                "annual_change": float(pct_change) if pct_change else None,
                "region_name":  row.get("RegionName") or area_code,
            }
            monthly_batch.append(monthly_row)

            # Property type breakdown if columns present
            for prop_type, col_avg, col_chg in [
                ("detached",   "DetachedPrice",   "DetachedAnnualChange"),
                ("semi",       "SemiDetachedPrice","SemiDetachedAnnualChange"),
                ("terraced",   "TerracedPrice",   "TerracedAnnualChange"),
                ("flat",       "FlatPrice",       "FlatAnnualChange"),
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

            # Flush batches
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
            log.warning(f"Row error: {e} — row: {dict(list(row.items())[:4])}")
            continue

    # Flush remaining
    if monthly_batch:
        supabase.table(HPI_TABLE_MONTHLY).upsert(monthly_batch, on_conflict="area_code,date").execute()
        monthly_count += len(monthly_batch)
    if by_type_batch:
        supabase.table(HPI_TABLE_BY_TYPE).upsert(by_type_batch, on_conflict="area_code,date,property_type").execute()
        by_type_count += len(by_type_batch)

    log.info(f"HPI refresh complete: {monthly_count} monthly rows, {by_type_count} by-type rows")


# ── LAND REGISTRY PRICE PAID ─────────────────────────────────────────────────
# Published: 20th of each month (previous month's transactions)
# Full dataset: https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads
# Monthly update file (current year): smaller, faster to ingest
PP_URL_MONTHLY = "https://www.gov.uk/government/uploads/system/uploads/attachment_data/file/pp-{year}-part{part}.csv"
PP_TABLE       = "price_paid_raw_2025"

PP_COLUMNS = [
    "transaction_id", "price", "date_of_transfer", "postcode",
    "property_type", "new_build", "estate_type",
    "saon", "paon", "street", "locality", "town", "district", "county",
    "ppd_category", "record_status",
]

def refresh_price_paid():
    """Download current year price paid data and upsert into Supabase."""
    log.info("Starting price paid refresh...")
    year = datetime.utcnow().year

    # LR publishes two parts per year, merged as year progresses
    # Use the monthly update endpoint for incremental refresh
    monthly_url = f"https://www.gov.uk/government/uploads/system/uploads/attachment_data/file/pp-monthly-update-new-version.csv"

    try:
        r = requests.get(monthly_url, timeout=180, stream=True)
        if r.status_code != 200:
            log.warning(f"Monthly PP update returned {r.status_code} — skipping price paid refresh")
            return
    except Exception as e:
        log.error(f"Price paid download failed: {e}")
        return

    content = r.content.decode("utf-8", errors="replace")
    reader  = csv.reader(io.StringIO(content))
    batch   = []
    count   = 0

    for row in reader:
        if len(row) < 15:
            continue
        try:
            record = {
                "transaction_id":    row[0].strip('{}'),
                "price":             int(row[1]) if row[1].strip().isdigit() else None,
                "date_of_transfer":  row[2][:10] if row[2] else None,
                "postcode":          row[3].strip().upper().replace(" ", ""),
                "property_type":     row[4].strip(),
                "new_build":         row[5].strip() == "Y",
                "estate_type":       row[6].strip(),
                "town":              row[11].strip().title() if row[11] else None,
                "district":          row[12].strip().title() if row[12] else None,
                "county":            row[13].strip().title() if row[13] else None,
            }
            if not record["transaction_id"] or not record["price"]:
                continue
            batch.append(record)

            if len(batch) >= BATCH_SIZE:
                supabase.table(PP_TABLE).upsert(batch, on_conflict="transaction_id").execute()
                count += len(batch)
                batch = []
                log.info(f"Price paid: {count} rows upserted")

        except Exception as e:
            log.warning(f"PP row error: {e}")
            continue

    if batch:
        supabase.table(PP_TABLE).upsert(batch, on_conflict="transaction_id").execute()
        count += len(batch)

    log.info(f"Price paid refresh complete: {count} rows upserted")


# ── SCHOOLS (OFSTED) ──────────────────────────────────────────────────────────
# Published: termly — September, January, April
# Source: https://www.compare-school-performance.service.gov.uk/download-data
SCHOOLS_URL   = "https://www.compare-school-performance.service.gov.uk/api/upload/national/england_schoolinformation.csv"
SCHOOLS_TABLE = "schools_clean_v2"

def refresh_schools():
    """Download Ofsted school data and upsert into Supabase."""
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
                "urn":            str(urn),
                "school_name":    row.get("EstablishmentName") or row.get("school_name") or "",
                "postcode":       (row.get("Postcode") or row.get("postcode") or "").strip().upper(),
                "ofsted_rating":  row.get("OfstedRating") or row.get("ofsted_rating") or "",
                "school_type":    row.get("TypeOfEstablishment (name)") or row.get("school_type") or "",
                "phase":          row.get("PhaseOfEducation (name)") or "",
                "la_name":        row.get("LA (name)") or "",
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


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info(f"LegalSmegal data refresh started — {datetime.utcnow().isoformat()}")

    # Monthly refreshes (run every month)
    refresh_hpi()
    refresh_price_paid()

    # Schools — only refresh in September, January, April (Ofsted term releases)
    month = datetime.utcnow().month
    if month in (1, 4, 9):
        refresh_schools()
    else:
        log.info(f"Skipping schools refresh (month={month}, runs in Jan/Apr/Sep)")

    log.info(f"Data refresh complete — {datetime.utcnow().isoformat()}")
