#!/usr/bin/env python3
"""
boe_rates_sync.py — LegalSmegal benchmark-rate cron (Table 1: bench_rates)

Pulls the Bank of England's published *quoted household interest rates* from the
free IADB CSV export and upserts them into bench_rates. Run monthly (BoE
publishes on the 5th working day).

WHY BoE and not a scrape: the IADB is a stable, official, key-free government
data endpoint. A cron against it is clean and reliable in a way scraping 20
lender sites never is. It gives representative market rates by term/LTV — NOT
per-named-lender (that is Table 2's job, fed separately).

FAILURE CONTRACT (matches LegalSmegal doctrine):
  - On ANY fetch/parse error, or a series returning no rows, we DO NOT touch that
    row. Last good value + its as_of stay in place. We never write NULL over a
    real value, and we never fabricate.
  - Exit code 1 on failure so the scheduler surfaces it.

BUILD-TIME CHECK (do this once): confirm the CSV actually pulls from YOUR server.
  A local test:  python3 boe_rates_sync.py --dry-run
  It prints the parsed latest value per series. If BoE blocks the request, set a
  browser-like User-Agent (already set below) or run from the VPS (server IPs are
  not robots-blocked the way a headless fetcher is; ONS runs this exact pull).

BTL series: BoE publishes a small set of buy-to-let quoted rates. Their exact
IADB series codes are NOT hard-verified in this build — add them to SERIES below
once confirmed from the BoE series list, then re-run. Do NOT guess codes.
"""

import os
import sys
import csv
import io
import datetime as dt
import urllib.request
import urllib.error

# --- Verified representative series (owner-occupier quoted rates) -------------
# label must match the label seeded in bench_rates.
SERIES = {
    "IUMABEDR": "Bank Rate",
    "IUMB2GH":  "2yr fixed 75% LTV",
    "IUMB5GH":  "5yr fixed 75% LTV",
    "IUMB2IH":  "2yr fixed 90% LTV",
    "IUMB5IH":  "5yr fixed 90% LTV",
    # TODO(confirm): add BoE buy-to-let quoted-rate series codes here once
    # verified from the BoE series list. Leave out until confirmed — never guess.
}

BOE_BASE = "https://www.bankofengland.co.uk/boeapps/iadb/fromshowcolumns.asp"
USER_AGENT = "LegalSmegal-rates-sync/1.0 (+admin contact)"
FRESH_DAYS = 45  # bench considered stale beyond this (BoE is monthly)


def build_url() -> str:
    today = dt.date.today()
    datefrom = (today - dt.timedelta(days=400)).strftime("%d/%b/%Y")
    dateto = today.strftime("%d/%b/%Y")
    codes = ",".join(SERIES.keys())
    return (
        f"{BOE_BASE}?csv.x=yes"
        f"&Datefrom={datefrom}&Dateto={dateto}"
        f"&SeriesCodes={codes}"
        f"&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N"
    )


def fetch_csv() -> str:
    req = urllib.request.Request(build_url(), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_latest(csv_text: str) -> dict:
    """Return {series_code: (rate_pct, as_of_date)} for the most recent non-empty
    observation of each series. Missing/blank cells are skipped (never coerced to 0)."""
    reader = csv.reader(io.StringIO(csv_text))
    rows = [r for r in reader if r]
    if len(rows) < 2:
        raise ValueError("BoE CSV returned no data rows")

    header = rows[0]  # ['Series code'/'DATE', <code>, <code>, ...] — first col is the date label
    code_cols = {}
    for idx, col in enumerate(header[1:], start=1):
        c = col.strip()
        if c in SERIES:
            code_cols[c] = idx

    latest: dict = {}
    for r in rows[1:]:
        if not r or not r[0].strip():
            continue
        try:
            d = dt.datetime.strptime(r[0].strip(), "%d %b %Y").date()
        except ValueError:
            continue
        for code, col in code_cols.items():
            if col >= len(r):
                continue
            val = r[col].strip()
            if val == "":
                continue
            try:
                rate = float(val)
            except ValueError:
                continue
            prev = latest.get(code)
            if prev is None or d > prev[1]:
                latest[code] = (rate, d)
    return latest


def upsert(latest: dict) -> int:
    """Upsert into bench_rates via supabase-py. Only writes rows we actually got."""
    from supabase import create_client  # same client app.py uses
    url = (os.getenv("SUPABASE_URL") or "").strip()
    key = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY") or "").strip()
    if not url or not key:
        raise RuntimeError("SUPABASE_URL / key not set")
    sb = create_client(url, key)
    written = 0
    for code, (rate, as_of) in latest.items():
        sb.table("bench_rates").update({
            "rate_pct": rate,
            "as_of": as_of.isoformat(),
            "label": SERIES[code],
            "source": "Bank of England IADB",
            "fetched_at": dt.datetime.utcnow().isoformat(),
        }).eq("series_code", code).execute()
        written += 1
    return written


def main():
    dry = "--dry-run" in sys.argv
    try:
        csv_text = fetch_csv()
        latest = parse_latest(csv_text)
    except (urllib.error.URLError, ValueError, TimeoutError) as e:
        # Failure contract: leave last-good rows untouched, surface the error.
        print(f"[boe-sync] FAILED — {e}. bench_rates left unchanged (last-good kept).", file=sys.stderr)
        sys.exit(1)

    if not latest:
        print("[boe-sync] FAILED — no parseable series. bench_rates left unchanged.", file=sys.stderr)
        sys.exit(1)

    for code, (rate, as_of) in sorted(latest.items()):
        print(f"[boe-sync] {code} ({SERIES[code]}): {rate:.3f}%  as of {as_of.isoformat()}")

    if dry:
        print("[boe-sync] --dry-run: nothing written.")
        return

    n = upsert(latest)
    print(f"[boe-sync] OK — {n} benchmark series updated.")


if __name__ == "__main__":
    main()
