"""pipr_rents.py — RENT-PIPR-1 (10 Oct 2026): ONS Price Index of Private Rents (PIPR),
UK monthly price statistics -> rows for Supabase public.uk_prms_monthly.

Why this exists (verified live 10 Oct 2026): refresh_data.refresh_prms() downloaded the
discontinued "Private Rental Market Summary Statistics in England" (ONS URLs now 404/502),
so uk_prms_monthly was never refreshed after the one-off load in May 2026 (latest period
March 2026). That load came from PIPR; this module reads PIPR's own published workbook.

Source:  https://www.ons.gov.uk/economy/inflationandpriceindices/datasets/
         priceindexofprivaterentsukmonthlypricestatistics   (OGL v3.0, monthly)
Sheet:   "Table 1" — one row per (Time period, Area code). Columns used, by header name:
         Time period (Excel serial date) · Area code · Area name · Region or country name ·
         Index · Annual change · Rental price.  "[x]" = not available, "[z]" = not applicable.

Table columns written (unchanged meaning, two added):
  period          first of month
  area_code       ONS GSS code (UK, GB, countries, English regions, LADs, Scottish BRMAs)
  rent_index      ONS index rebased so January 2015 = 100 for that area (the table's
                  established convention; growth ratios are identical to ONS's own index)
  rent_yoy_pct    ONS annual change, % (1 dp, as ONS presents it)
  rent_price_gbp  ONS average monthly private rent, £
  area_name       ONS area name                       (added by RENT-PIPR-1)
  region_name     ONS "Region or country name" (NULL where "[z]")   (added by RENT-PIPR-1)

Pure: no network, no database. Streams the 90 MB sheet XML with the standard library
(about 20 MB peak), so the cron needs no spreadsheet dependency.
"""
import re
import zipfile
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from typing import Any, Dict, Iterator, List, Optional, Tuple

VERSION = "rent-pipr-1"
SOURCE_LABEL = "ONS Price Index of Private Rents"
DATASET_PAGE = ("https://www.ons.gov.uk/economy/inflationandpriceindices/datasets/"
                "priceindexofprivaterentsukmonthlypricestatistics")
SHEET_NAME = "Table 1"
BASE_PERIOD = date(2015, 1, 1)

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_RNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PKG = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_EXCEL_EPOCH = date(1899, 12, 30)
_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
_LINK = re.compile(
    r"(?:https?://www\.ons\.gov\.uk)?/file\?uri=(/economy/inflationandpriceindices/datasets/"
    r"priceindexofprivaterentsukmonthlypricestatistics/([0-9]{1,2})([a-z]+)([0-9]{4})/[^\"'<>\s]+?\.xlsx)",
    re.I)

HEADERS = {
    "period": "Time period", "area_code": "Area code", "area_name": "Area name",
    "region": "Region or country name", "index": "Index",
    "annual_change": "Annual change", "price": "Rental price",
}


# ── link discovery ─────────────────────────────────────────────────────────────
def latest_xlsx_url(page_html: str) -> Optional[Tuple[str, date]]:
    """Newest edition's .xlsx on the dataset page -> (absolute url, edition date).
    Editions are dated folders (e.g. 16september2026); file names vary per edition."""
    best: Optional[Tuple[date, str]] = None
    for m in _LINK.finditer((page_html or "").replace("&amp;", "&")):
        mon = _MONTHS.get(m.group(3).lower())
        if not mon:
            continue
        try:
            d = date(int(m.group(4)), mon, int(m.group(2)))
        except ValueError:
            continue
        url = "https://www.ons.gov.uk/file?uri=" + m.group(1)
        if best is None or d > best[0]:
            best = (d, url)
    return (best[1], best[0]) if best else None


# ── streaming xlsx reader (stdlib) ─────────────────────────────────────────────
def _col(ref: str) -> int:
    n = 0
    for ch in ref:
        if not ch.isalpha():
            break
        n = n * 26 + (ord(ch.upper()) - 64)
    return n - 1


def _sheet_path(z: zipfile.ZipFile, name: str) -> str:
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rid = None
    for s in wb.iter(_NS + "sheet"):
        if s.get("name") == name:
            rid = s.get(_RNS + "id")
            break
    if rid is None:
        raise ValueError(f"sheet {name!r} not found in workbook")
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    for r in rels.iter(_PKG + "Relationship"):
        if r.get("Id") == rid:
            t = r.get("Target").lstrip("/")
            return t if t.startswith("xl/") else "xl/" + t
    raise ValueError(f"no relationship for sheet {name!r}")


def _shared_strings(z: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    out: List[str] = []
    for _, el in ET.iterparse(z.open("xl/sharedStrings.xml")):
        if el.tag == _NS + "si":
            out.append("".join(t.text or "" for t in el.iter(_NS + "t")))
            el.clear()
    return out


def iter_sheet_rows(path: str, sheet: str = SHEET_NAME) -> Iterator[List[Optional[str]]]:
    with zipfile.ZipFile(path) as z:
        ss = _shared_strings(z)
        for _, el in ET.iterparse(z.open(_sheet_path(z, sheet))):
            if el.tag != _NS + "row":
                continue
            vals: Dict[int, Optional[str]] = {}
            for c in el.iter(_NS + "c"):
                t, v = c.get("t"), c.find(_NS + "v")
                if t == "s" and v is not None:
                    val: Optional[str] = ss[int(v.text)]
                elif t == "inlineStr":
                    val = "".join(x.text or "" for x in c.iter(_NS + "t"))
                else:
                    val = v.text if v is not None else None
                vals[_col(c.get("r") or "")] = val
            yield [vals.get(i) for i in range(max(vals) + 1)] if vals else []
            el.clear()


# ── parsing ────────────────────────────────────────────────────────────────────
def _num(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.startswith("["):          # [x] not available, [z] not applicable
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def _period(x: Any) -> Optional[date]:
    v = _num(x)
    if v is not None:
        d = _EXCEL_EPOCH + timedelta(days=int(v))
        return date(d.year, d.month, 1)
    s = str(x or "").strip()
    m = re.match(r"^(\d{4})-(\d{2})", s)
    return date(int(m.group(1)), int(m.group(2)), 1) if m else None


def _is_code(s: Optional[str]) -> bool:
    return bool(s) and bool(re.match(r"^[EWSNK]\d{8}$", s.strip()))


def parse_rows(rows: Iterator[List[Optional[str]]]) -> Dict[str, Any]:
    """Sheet rows -> {'rows': [...table rows...], 'latest_period', 'areas', 'skipped'}.
    Locates the header row by its column names (not by position), so a layout shift in
    a future edition fails loudly instead of loading the wrong columns."""
    idx: Optional[Dict[str, int]] = None
    raw: List[Dict[str, Any]] = []
    skipped = 0
    for r in rows:
        if idx is None:
            cells = [str(c or "").strip() for c in r]
            if HEADERS["period"] in cells and HEADERS["area_code"] in cells:
                missing = [h for h in HEADERS.values() if h not in cells]
                if missing:
                    raise ValueError(f"PIPR header row lacks columns: {missing}")
                idx = {k: cells.index(h) for k, h in HEADERS.items()}
            continue

        def g(k):
            i = idx[k]
            return r[i] if i < len(r) else None

        code = str(g("area_code") or "").strip()
        per = _period(g("period"))
        if not _is_code(code) or per is None:
            skipped += 1                    # NI BRMA rows carry area code "[z]"
            continue
        region = str(g("region") or "").strip()
        raw.append({
            "period": per, "area_code": code,
            "area_name": str(g("area_name") or "").strip() or None,
            "region_name": None if (not region or region.startswith("[")) else region,
            "ons_index": _num(g("index")),
            "rent_yoy_pct": _num(g("annual_change")),
            "rent_price_gbp": _num(g("price")),
        })
    if idx is None:
        raise ValueError("PIPR header row not found")

    base = {r["area_code"]: r["ons_index"] for r in raw
            if r["period"] == BASE_PERIOD and r["ons_index"]}
    out: List[Dict[str, Any]] = []
    for r in raw:
        b = base.get(r["area_code"])
        if not b or r["ons_index"] is None:
            skipped += 1                    # rent_index is NOT NULL; no Jan-2015 base -> no row
            continue
        yoy = r["rent_yoy_pct"]
        price = r["rent_price_gbp"]
        out.append({
            "period": r["period"].isoformat(),
            "area_code": r["area_code"],
            "rent_index": round(r["ons_index"] / b * 100.0, 4),
            "rent_yoy_pct": round(yoy, 1) if yoy is not None else None,
            "rent_price_gbp": round(price) if price is not None else None,
            "area_name": r["area_name"],
            "region_name": r["region_name"],
        })
    latest = max((r["period"] for r in out), default=None)
    return {"rows": out, "latest_period": latest,
            "areas": len({r["area_code"] for r in out}), "skipped": skipped}


def parse_workbook(path: str) -> Dict[str, Any]:
    return parse_rows(iter_sheet_rows(path, SHEET_NAME))
