#!/usr/bin/env python3
"""load_crime_by_csp.py — CRIME-EVID-1 loader for ew.crime_by_csp (Hetzner).

Loads the LATEST financial year from the Home Office "Police recorded crime
Community Safety Partnership open data" .ods into ew.crime_by_csp, in ONE
transaction (delete that year, insert, verify) — rolls back to nothing on error.

Run on the box (after sql/20261008_ew_crime_by_csp.sql):
    sudo -u postgres python3 load_crime_by_csp.py --dsn "dbname=legalsmegal_data" \
        --ods /root/prc-csp-mar2021-mar2026-tables-230726.ods
or let it download the newest file itself:
    sudo -u postgres python3 load_crime_by_csp.py --dsn "dbname=legalsmegal_data" --latest

The same load_parsed() is used by refresh_data.py (monthly cron) so manual and
scheduled loads are identical.
"""
import argparse
import os
import sys
import tempfile

import psycopg

import ew_crime

INSERT_SQL = (
    "INSERT INTO ew.crime_by_csp (financial_year, police_force, csp_name, offence_group, "
    "offences, quarters, usable, quality_note, source_file, published) "
    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)")


def load_parsed(conn, parsed, source_file, log=print):
    rows = ew_crime.load_rows(parsed, source_file)
    if not rows:
        raise RuntimeError("No rows parsed from the file — nothing loaded.")
    fy = parsed["financial_year"]
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("DELETE FROM ew.crime_by_csp WHERE financial_year = %s", (fy,))
            cur.executemany(INSERT_SQL, rows)
            cur.execute("SELECT count(*) AS n, count(DISTINCT csp_name) AS ncsp, sum(offences) AS total "
                        "FROM ew.crime_by_csp WHERE financial_year = %s", (fy,))
            row = cur.fetchone()
            n, ncsp, total = list(row.values()) if isinstance(row, dict) else row
            if n != len(rows):
                raise RuntimeError(f"Row count mismatch: inserted {len(rows)}, table has {n}")
    unusable = sorted({r[2] for r in rows if not r[6] and not r[2].lower().startswith("unassigned")})
    log(f"[crime-csp] loaded {fy} from {source_file}: {n} rows, {ncsp} CSPs, "
        f"{total:,} offences, published {parsed.get('published')}; "
        f"flagged by Home Office note: {unusable or 'none'}")
    return n


def download_latest(log=print):
    import requests
    page = requests.get(ew_crime.DATASET_PAGE, timeout=60)
    page.raise_for_status()
    found = ew_crime.latest_csp_file_url(page.text)
    if not found:
        raise RuntimeError("No CSP .ods link found on the gov.uk dataset page.")
    url, name = found
    log(f"[crime-csp] newest file on gov.uk: {name}")
    tmp = tempfile.NamedTemporaryFile(suffix=".ods", delete=False)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        for chunk in r.iter_content(1 << 20):
            tmp.write(chunk)
    tmp.close()
    return tmp.name, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ods")
    g.add_argument("--latest", action="store_true")
    a = ap.parse_args()
    if a.latest:
        path, name = download_latest()
    else:
        path, name = a.ods, os.path.basename(a.ods)
    parsed = ew_crime.parse_csp_ods(path)
    print(f"[crime-csp] parsed sheet {parsed['sheet']} ({parsed['financial_year']}), "
          f"quarters {parsed['quarters']}, {len(parsed['rows'])} rows")
    with psycopg.connect(a.dsn) as conn:
        load_parsed(conn, parsed, name)
        with conn.cursor() as cur:
            cur.execute("SELECT police_force, csp_name, sum(offences) FROM ew.crime_by_csp "
                        "WHERE financial_year=%s AND csp_name IN ('Manchester','Birmingham','Medway') "
                        "GROUP BY 1,2 ORDER BY 2", (parsed["financial_year"],))
            for r in cur.fetchall():
                print("[crime-csp] check:", r)


if __name__ == "__main__":
    sys.exit(main())
