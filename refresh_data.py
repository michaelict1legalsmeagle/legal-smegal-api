"""
LegalSmegal — Monthly Data Refresh
===================================
Runs on the 21st of each month via Render Cron Job.
Downloads latest Land Registry HPI and Price Paid data.
HPI + Schools → Supabase. Price Paid → Hetzner.

Render Cron Job config:
  Command: python refresh_data.py
  Schedule: 0 6 21 * *   (6am on the 21st of every month)
  Environment: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, DATA_DATABASE_URL

Backfill (run once manually to catch up missing years):
  python refresh_data.py --backfill            # backfills last 2 years
  python refresh_data.py --backfill --all      # backfills 2018 through 2026
"""
import os
import io
import csv
import sys
import logging
import requests
import psycopg
from psycopg.rows import dict_row
from datetime import datetime
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
log = logging.getLogger(__name__)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Hetzner — price_paid_raw_2025 lives here, not on Supabase
DATA_DATABASE_URL = os.environ.get("DATA_DATABASE_URL")
if not DATA_DATABASE_URL:
    raise RuntimeError("DATA_DATABASE_URL is required")

BATCH_SIZE = 500  # rows per batch — keeps memory low on Render free tier


def _get_hetzner_conn():
    return psycopg.connect(DATA_DATABASE_URL, row_factory=dict_row)


# ── LAND REGISTRY HPI ────────────────────────────────────────────────────────
# Updated URL: Land Registry moved to publicdata.landregistry.gov.uk in 2025
HPI_URL           = "https://publicdata.landregistry.gov.uk/market-trend-data/house-price-index-data/UK-HPI-full-file-{year}-{month:02d}.csv"
HPI_TABLE_MONTHLY = "uk_hpi_monthly"
HPI_TABLE_BY_TYPE = "uk_hpi_monthly_by_property_type"


def get_latest_hpi_url() -> str:
    now = datetime.utcnow()
    # HPI data lags ~2 months, so try current month back 5 months
    for m_offset in range(0, 6):
        month = now.month - m_offset
        year  = now.year
        if month <= 0:
            month += 12
            year -= 1
        url = HPI_URL.format(year=year, month=month)
        try:
            r = requests.head(url, timeout=10, allow_redirects=True)
            if r.status_code == 200:
                log.info(f"HPI URL found: {url}")
                return url
            else:
                log.debug(f"HPI {r.status_code}: {url}")
        except Exception as e:
            log.debug(f"HPI head check failed ({url}): {e}")

    # Dynamic fallback: last December as safe anchor (always published by March)
    fallback_year  = now.year if now.month > 3 else now.year - 1
    fallback_month = 12
    fallback = HPI_URL.format(year=fallback_year, month=fallback_month)
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

    # Stream line-by-line — never loads full file into memory.
    # Full HPI file is 200-400MB uncompressed; r.content would OOM at 512Mi.
    reader        = csv.DictReader(r.iter_lines(decode_unicode=True))
    monthly_batch = []
    by_type_batch = []
    monthly_count = 0
    by_type_count = 0

    for row in reader:
        try:
            # FIXED: prefer AreaCode (E09000005 ONS format) over RegionName ("Brent")
            # Queries use lad_code from postcodes.io which returns E-code format
            # Storing RegionName as primary caused all WHERE area_code = lad_code to return 0 rows
            area_code  = (row.get("AreaCode") or row.get("RegionName") or "").strip()
            date_str   = (row.get("Date") or "").strip()
            avg_price  = row.get("AveragePrice") or row.get("Average price") or None
            # Land Registry HPI CSV column name varies by year:
            # 2024+ format: "Annual % Change"
            # Older format:  "AnnualChange" or "12m % change"
            pct_change = (row.get("Annual % Change")
                          or row.get("AnnualChange")
                          or row.get("12m % change")
                          or row.get("Annual Percentage Change")
                          or None)
            if not area_code or not date_str:
                continue
            if "/" in date_str:
                parts = date_str.split("/")
                if len(parts) == 3:
                    date_str = f"{parts[2]}-{parts[1]}-{parts[0]}"
            monthly_batch.append({
                "area_code":     area_code,
                "date":          date_str,
                "average_price": float(avg_price) if avg_price else None,
                "annual_change": float(pct_change) if pct_change else None,
                "region_name":   row.get("RegionName") or area_code,
            })
            for prop_type, col_avg, col_chg in [
                ("detached",  "DetachedPrice",      "DetachedAnnualChange"),
                ("semi",      "SemiDetachedPrice",   "SemiDetachedAnnualChange"),
                ("terraced",  "TerracedPrice",      "TerracedAnnualChange"),
                ("flat",      "FlatPrice",           "FlatAnnualChange"),
            ]:
                avg = row.get(col_avg) or row.get(f"{prop_type.capitalize()} Average Price")
                chg = row.get(col_chg)
                if avg:
                    by_type_batch.append({
                        "area_code":     area_code,
                        "date":          date_str,
                        "property_type": prop_type,
                        "average_price": float(avg),
                        "annual_change": float(chg) if chg else None,
                        "region_name":   row.get("RegionName") or area_code,
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


# ── ONS PRMS (Private Rental Market Statistics) ──────────────────────────────
# Writes to Supabase uk_prms_monthly: area_code, date, region_name, rent_price_gbp,
# rent_yoy_pct. Source: ONS private rental index CSV (England, LAD level).
PRMS_URL = (
    "https://www.ons.gov.uk/generator?format=csv"
    "&uri=/economy/inflationandpriceindices/bulletins"
    "/privaterentalmarketsummarystatisticsinengland/latest"
)
# Stable direct CSV: ONS publish median rent by local authority
PRMS_LA_URL = (
    "https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices"
    "/datasets/privaterentalmarketsummarystatisticsinengland"
    "/current/prt1a.csv"
)
PRMS_TABLE = "uk_prms_monthly"


def refresh_prms():
    """
    Download ONS Private Rental Market Statistics (England, local authority level).
    Upserts into Supabase uk_prms_monthly.
    ONS publish annually (typically March) covering the previous 12 months.
    Table columns: area_code, date, region_name, rent_price_gbp, rent_yoy_pct.
    """
    log.info("Starting ONS PRMS refresh...")
    urls_to_try = [PRMS_LA_URL, PRMS_URL]
    r = None
    used_url = None
    for url in urls_to_try:
        try:
            resp = requests.get(url, timeout=60, allow_redirects=True)
            if resp.status_code == 200 and len(resp.content) > 500:
                r = resp
                used_url = url
                break
            else:
                log.warning(f"PRMS URL {resp.status_code}: {url}")
        except Exception as e:
            log.warning(f"PRMS URL failed ({url}): {e}")

    if not r:
        log.error("PRMS: no working URL found — skipping.")
        return

    log.info(f"PRMS: downloaded from {used_url}")
    # Stream line-by-line — avoids loading full CSV into memory
    reader = csv.DictReader(r.iter_lines(decode_unicode=True))

    batch = []
    count = 0
    skipped = 0

    # Build case-insensitive column lookup from first row
    _col_norm = {}  # lowercase_no_special -> actual header

    for row in reader:
        # Build normalised column map once from first row
        if not _col_norm:
            for k in row.keys():
                _col_norm[k.lower().replace(" ", "_").replace("-", "_")] = k

        def _get(*keys):
            for k in keys:
                # Try exact key first
                v = row.get(k)
                if v is not None and str(v).strip():
                    return str(v).strip()
                # Try normalised lookup
                norm = k.lower().replace(" ", "_").replace("-", "_")
                actual = _col_norm.get(norm)
                if actual:
                    v = row.get(actual)
                    if v is not None and str(v).strip():
                        return str(v).strip()
            return ""

        try:
            # ONS prt1a.csv columns: Area_Code, Area_Name, Year_ending_Month, Median
            # Also handles older variants and alternative column names
            area_code   = _get("area_code", "Area_Code", "LACODE", "Code", "geography_code")
            region_name = _get("area_name", "Area_Name", "LANAME", "Area", "Name", "geography")
            date_str    = _get("date", "Date", "Period", "Year_ending_Month", "year_ending_month", "Time")
            rent_raw    = _get("median_rent", "Median", "median", "rent_price_gbp", "Value", "value") or None
            yoy_raw     = _get("annual_change", "Annual_Change", "YoY", "rent_yoy_pct") or None

            if not area_code or not date_str:
                skipped += 1
                continue

            # Normalise date
            if "/" in date_str:
                parts = date_str.split("/")
                if len(parts) == 3:
                    date_str = f"{parts[2]}-{parts[1]}-{parts[0]}"
            elif len(date_str) == 7 and "-" in date_str:
                date_str = date_str + "-01"

            rent = float(str(rent_raw).replace(",", "")) if rent_raw else None
            yoy  = float(str(yoy_raw).replace("%", "").replace(",", "")) if yoy_raw else None

            batch.append({
                "area_code":    area_code,
                "period":       date_str,
                "region_name":  region_name or area_code,
                "rent_price_gbp": rent,
                "rent_yoy_pct": yoy,
            })

            if len(batch) >= BATCH_SIZE:
                supabase.table(PRMS_TABLE).upsert(batch, on_conflict="area_code,period").execute()
                count += len(batch)
                batch = []
                log.info(f"PRMS: {count} rows upserted")
        except Exception as e:
            log.warning(f"PRMS row error: {e} — row: {dict(list(row.items())[:4])}")
            skipped += 1
            continue

    if batch:
        supabase.table(PRMS_TABLE).upsert(batch, on_conflict="area_code,period").execute()
        count += len(batch)

    log.info(f"PRMS refresh complete: {count} rows upserted, {skipped} skipped")


# ── LAND REGISTRY PRICE PAID ──────────────────────────────────────────────────
# Price Paid writes to Hetzner (price_paid_raw_2025), NOT Supabase
PP_MONTHLY_URL  = "https://price-paid-data.publicdata.landregistry.gov.uk/pp-monthly-update-new-version.csv"
PP_ANNUAL_URL   = "https://price-paid-data.publicdata.landregistry.gov.uk/pp-{year}.csv"
PP_ANNUAL_YEARS = [2021]

PP_UPSERT_SQL = """
    INSERT INTO public.price_paid_raw_2025 (
        transaction_unique_identifier, price, date_of_transfer, postcode,
        property_type, old_new, duration, paon, saon, street, locality,
        town_city, district, county, ppd_category_type, record_status
    ) VALUES (
        %(transaction_unique_identifier)s, %(price)s, %(date_of_transfer)s, %(postcode)s,
        %(property_type)s, %(old_new)s, %(duration)s, %(paon)s, %(saon)s, %(street)s, %(locality)s,
        %(town_city)s, %(district)s, %(county)s, %(ppd_category_type)s, %(record_status)s
    )
    ON CONFLICT (transaction_unique_identifier) DO UPDATE SET
        price             = EXCLUDED.price,
        date_of_transfer  = EXCLUDED.date_of_transfer,
        postcode          = EXCLUDED.postcode,
        property_type     = EXCLUDED.property_type,
        old_new           = EXCLUDED.old_new,
        duration          = EXCLUDED.duration,
        paon              = EXCLUDED.paon,
        saon              = EXCLUDED.saon,
        street            = EXCLUDED.street,
        locality          = EXCLUDED.locality,
        town_city         = EXCLUDED.town_city,
        district          = EXCLUDED.district,
        county            = EXCLUDED.county,
        ppd_category_type = EXCLUDED.ppd_category_type,
        record_status     = EXCLUDED.record_status
"""


def get_latest_pp_url() -> str:
    now = datetime.utcnow()
    candidates = [
        PP_MONTHLY_URL,
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
    log.error("PP: no working URL found — skipping.")
    return ""


PP_DELETE_SQL = """
    DELETE FROM public.price_paid_raw_2025
    WHERE transaction_unique_identifier = ANY(%s)
"""


def _flush_pp_upserts(conn, batch: list, label: str, count_before: int) -> int:
    """Upsert one batch of A/C rows. Returns rows sent."""
    with conn.cursor() as cur:
        cur.executemany(PP_UPSERT_SQL, batch)
    conn.commit()
    log.info(f"price_paid_raw_2025 [{label}]: {count_before + len(batch)} rows upserted")
    return len(batch)


def _flush_pp_deletes(conn, ids: list) -> int:
    """Delete withdrawn (D) transactions. Matches the bare ID and the braced
    form, because rows loaded before 2 Oct 2026 may still carry braces
    (3,129 braced rows remained after the duplicate clean-up). Returns rows deleted."""
    keys = []
    for i in ids:
        keys.append(i)
        keys.append("{" + i + "}")
    with conn.cursor() as cur:
        cur.execute(PP_DELETE_SQL, (keys,))
        n = cur.rowcount or 0
    conn.commit()
    return n


def _upsert_pp_rows_from_csv(lines_iter, label: str):
    """
    Stream PP CSV lines and upsert into price_paid_raw_2025 on Hetzner.
    Accepts any line iterator — never loads full CSV into memory.
    Uses a single Hetzner connection for the entire run (not one per batch).
    Returns (count, skipped).
    """
    reader  = csv.reader(lines_iter)
    batch   = []
    count   = 0
    skipped = 0
    delete_ids: list = []
    deleted = 0

    try:
        conn = _get_hetzner_conn()
    except Exception as e:
        log.error(f"PP [{label}]: Hetzner connection failed: {e}")
        return 0, 0

    try:
        for row in reader:
            if len(row) < 15:
                continue
            try:
                raw_pc = row[3].strip().upper()
                if not raw_pc.replace(" ", ""):
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
                # PP-DELETE (2026-10-02): a record_status "D" row is Land Registry
                # withdrawing a transaction. Remove it; never store it as a row
                # (7,241 stored "D" rows were deleted by hand on 2 Oct 2026).
                if record["record_status"] == "D":
                    if not record["transaction_unique_identifier"]:
                        skipped += 1
                        continue
                    if batch:   # keep file order: pending upserts land first
                        count += _flush_pp_upserts(conn, batch, label, count)
                        batch = []
                    delete_ids.append(record["transaction_unique_identifier"])
                    if len(delete_ids) >= BATCH_SIZE:
                        deleted += _flush_pp_deletes(conn, delete_ids)
                        delete_ids = []
                    continue
                if not record["transaction_unique_identifier"] or not record["price"]:
                    skipped += 1
                    continue
                if delete_ids:   # keep file order: pending deletes land first
                    deleted += _flush_pp_deletes(conn, delete_ids)
                    delete_ids = []
                batch.append(record)
                if len(batch) >= BATCH_SIZE:
                    count += _flush_pp_upserts(conn, batch, label, count)
                    batch = []
            except Exception as e:
                log.warning(f"PP row error [{label}]: {e}")
                skipped += 1
                continue

        # Flush remaining batches (upserts before deletes keeps file order:
        # whichever kind is still pending is the most recent run of rows)
        if batch:
            count += _flush_pp_upserts(conn, batch, label, count)
        if delete_ids:
            deleted += _flush_pp_deletes(conn, delete_ids)

    finally:
        conn.close()

    log.info(f"price_paid_raw_2025 [{label}]: {deleted} withdrawn (record_status D) rows deleted")
    return count, skipped


def refresh_price_paid():
    """Download and upsert the current monthly PP update file into Hetzner."""
    log.info("Starting price_paid_raw_2025 refresh from monthly update...")
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
    # Stream directly — never decode full response into memory
    count, skipped = _upsert_pp_rows_from_csv(
        r.iter_lines(decode_unicode=True), label="monthly"
    )
    log.info(f"price_paid_raw_2025 monthly refresh complete: {count} rows upserted, {skipped} skipped")


def backfill_price_paid(years: list):
    """
    Download full annual PP files and upsert into price_paid_raw_2025 on Hetzner.
    Re-runnable: IDs are stored without braces and upserted on
    transaction_unique_identifier. (A braced load done another way bypassed this
    and duplicated 591,189 rows in 2025 — removed 2 Oct 2026.)
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
        # Stream directly — annual PP files can be 300MB+
        count, skipped = _upsert_pp_rows_from_csv(
            r.iter_lines(decode_unicode=True), label=str(year)
        )
        log.info(f"Backfill {year} complete: {count} rows upserted, {skipped} skipped")
    log.info("Backfill complete — running materialized view refresh")
    refresh_materialized_view()


# ── MATERIALIZED VIEW ─────────────────────────────────────────────────────────
def refresh_crime_csp():
    """CRIME-EVID-1: reload ew.crime_by_csp (Hetzner) when gov.uk has published a newer
    Home Office CSP file than the one loaded. Cheap no-op otherwise. Isolated: a failure
    here is logged and never stops the other refreshes."""
    try:
        import ew_crime
        import load_crime_by_csp
        page = requests.get(ew_crime.DATASET_PAGE, timeout=60)
        page.raise_for_status()
        found = ew_crime.latest_csp_file_url(page.text)
        if not found:
            log.warning("[crime-csp] no CSP .ods link found on the gov.uk dataset page — skipped")
            return
        url, name = found
        with _get_hetzner_conn() as conn:
            row = conn.execute("SELECT 1 AS hit FROM ew.crime_by_csp WHERE source_file = %s LIMIT 1",
                               (name,)).fetchone()
        if row:
            log.info(f"[crime-csp] already loaded: {name}")
            return
        path, name = load_crime_by_csp.download_latest(log.info)
        try:
            parsed = ew_crime.parse_csp_ods(path)
            with _get_hetzner_conn() as conn:
                load_crime_by_csp.load_parsed(conn, parsed, name, log.info)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    except Exception as e:
        log.exception(f"[crime-csp] refresh failed (other refreshes unaffected): {e}")


def refresh_materialized_view():
    """No-op — price_paid_geo matview not used by API. Kept for backfill compatibility."""
    log.info("Materialized view refresh skipped — API queries price_paid_raw_2025 directly.")


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
        if "--all" in args:
            years = PP_ANNUAL_YEARS
        else:
            now   = datetime.utcnow()
            years = [now.year - 1, now.year]
        log.info(f"Backfill mode — years: {years}")
        backfill_price_paid(years)
    elif "--schools" in args:
        # Manual schools-only refresh: python refresh_data.py --schools
        log.info("Schools-only refresh triggered manually")
        refresh_schools()
        log.info("Schools refresh complete")
    elif "--crime" in args:
        # Manual crime-only refresh: python refresh_data.py --crime
        log.info("Crime (Home Office CSP) refresh triggered manually")
        refresh_crime_csp()
        log.info("Crime refresh complete")
    elif "--prms" in args:
        # Manual PRMS-only refresh: python refresh_data.py --prms
        log.info("PRMS-only refresh triggered manually")
        refresh_prms()
        log.info("PRMS refresh complete")
    else:
        log.info(f"LegalSmegal data refresh started — {datetime.utcnow().isoformat()}")
        refresh_hpi()
        refresh_prms()
        refresh_price_paid()
        refresh_materialized_view()
        refresh_crime_csp()   # CRIME-EVID-1: loads only when gov.uk publishes a newer file
        month = datetime.utcnow().month
        if month in (1, 4, 9):
            refresh_schools()
        else:
            log.info(f"Skipping schools refresh (month={month}, runs in Jan/Apr/Sep)")
        log.info(f"Data refresh complete — {datetime.utcnow().isoformat()}")
