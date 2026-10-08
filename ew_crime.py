"""CRIME-EVID-1 (8 Oct 2026) — England & Wales crime from two official sources.

1. STREET  — police.uk street-level crimes around the property, latest published
             month. Only records with location_type "Force" count as area crime
             (British Transport Police station records are reported separately).
             When police.uk returns no force records the street line says so —
             it is NEVER turned into a "0" (police.uk publishes no data at all
             for Greater Manchester Police, and individual forces miss months;
             see https://data.police.uk/changelog/).
2. DISTRICT — Home Office police recorded crime by Community Safety Partnership
             (CSP), latest financial year, loaded on Hetzner (ew.crime_by_csp)
             from the official .ods. Covers all 43 forces incl. GMP. A CSP the
             Home Office's own Notes flag as recorded under an "unassigned" group
             is marked unusable and shown as unavailable with that note quoted.

No rates, no indices, no national comparisons, no mixing of the two sources.
This module is pure (no network, no DB) so it is unit-testable; app.py and
load_crime_by_csp.py / refresh_data.py do the I/O.
"""
import collections
import re
import zipfile
from typing import Any, Dict, Iterable, List, Optional, Tuple
from xml.etree.ElementTree import iterparse

SCHEMA_VERSION = "crime-evid-1"
DISTRICT_SOURCE = "Home Office police recorded crime (Community Safety Partnership)"
DATASET_PAGE = ("https://www.gov.uk/government/statistical-data-sets/"
                "police-recorded-crime-and-outcomes-open-data-tables")
STREET_SOURCE = "police.uk"
POLICEUK_CHANGELOG = "https://data.police.uk/changelog/"

# e.g. https://assets.publishing.service.gov.uk/media/<hex>/prc-csp-mar2021-mar2026-tables-230726.ods
CSP_FILE_RE = re.compile(
    r"https://assets\.publishing\.service\.gov\.uk/media/[0-9a-f]+/"
    r"prc-csp-mar(\d{4})-mar(\d{4})-tables-(\d{6})\.ods")

_T = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
_TX = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
_O = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
_YEAR_SHEET = re.compile(r"^(\d{4})_(\d{2})$")
_Q_MONTHS = {"1": ("Apr", "Jun"), "2": ("Jul", "Sep"), "3": ("Oct", "Dec"), "4": ("Jan", "Mar")}


# ── Home Office .ods parsing ────────────────────────────────────────────────
def _row_cells(row_el) -> List[str]:
    cells: List[str] = []
    for c in row_el:
        if c.tag not in (f"{{{_T}}}table-cell", f"{{{_T}}}covered-table-cell"):
            continue
        rep = int(c.get(f"{{{_T}}}number-columns-repeated", "1"))
        v = c.get(f"{{{_O}}}value")
        if v is None:
            v = "".join("".join(p.itertext()) for p in c.findall(f"{{{_TX}}}p"))
        cells.extend([v] * min(rep, 50))
    while cells and cells[-1] == "":
        cells.pop()
    return cells


def parse_csp_ods(path: str) -> Dict[str, Any]:
    """Stream the Home Office CSP .ods (1.4 GB of XML; constant memory).

    Returns {financial_year, quarters, rows: {(force, csp, group): offences},
             notes_lines: [str], published: str|None}
    for the LATEST financial-year sheet only.
    """
    z = zipfile.ZipFile(path)
    best_sheet: Optional[str] = None
    agg: Dict[Tuple[str, str, str], float] = collections.defaultdict(float)
    quarters: set = set()
    fy: Optional[str] = None
    notes: List[str] = []
    sheet: Optional[str] = None
    header: Optional[List[str]] = None
    collecting = False
    with z.open("content.xml") as f:
        for ev, el in iterparse(f, events=("start", "end")):
            if ev == "start":
                if el.tag == f"{{{_T}}}table":
                    sheet = el.get(f"{{{_T}}}name")
                    header = None
                    collecting = False
                    if sheet and _YEAR_SHEET.match(sheet) and (best_sheet is None or sheet > best_sheet):
                        best_sheet, collecting = sheet, True
                        agg = collections.defaultdict(float)
                        quarters, fy = set(), None
                continue
            if el.tag != f"{{{_T}}}table-row":
                if el.tag == f"{{{_T}}}table":
                    el.clear()
                continue
            cells = _row_cells(el)
            el.clear()
            if not cells:
                continue
            if sheet == "Notes":
                notes.append(" ".join(c for c in cells if c))
                continue
            if not collecting:
                continue
            if header is None:
                header = cells
                continue
            r = dict(zip(header, cells))
            try:
                n = float(r.get("Offence Count") or 0)
            except ValueError:
                continue
            key = ((r.get("Police Force") or "").strip(), (r.get("CSP Name") or "").strip(),
                   (r.get("Offence Group") or "").strip())
            if not all(key):
                continue
            agg[key] += n
            quarters.add(str(r.get("Financial Quarter") or "").strip())
            fy = fy or (r.get("Financial Year") or "").strip()
    published = None
    for line in notes:
        m = re.match(r"^Updated:\s*(.+)$", line.strip())
        if m:
            published = m.group(1).strip()
            break
    return {"financial_year": fy, "quarters": sorted(q for q in quarters if q),
            "rows": {k: int(round(v)) for k, v in agg.items()},
            "notes_lines": notes, "published": published, "sheet": best_sheet}


def period_label(financial_year: Optional[str], quarters: Iterable[str]) -> Optional[str]:
    """'2025/26' + quarters 1-4 -> 'Apr 2025 – Mar 2026'."""
    m = re.match(r"^(\d{4})/(\d{2})$", financial_year or "")
    qs = sorted(q for q in quarters if q in _Q_MONTHS)
    if not m or not qs:
        return None
    y0 = int(m.group(1))
    def _yr(q):  # Q4 (Jan-Mar) falls in the following calendar year
        return y0 + 1 if q == "4" else y0
    first, last = qs[0], qs[-1]
    return f"{_Q_MONTHS[first][0]} {_yr(first)} – {_Q_MONTHS[last][1]} {_yr(last)}"


def quality_flags(notes_lines: List[str], csp_names: Iterable[str],
                  financial_year: Optional[str]) -> Dict[str, str]:
    """CSPs the Home Office's OWN Notes say are recorded under an 'unassigned'
    group for this year -> {csp_name: verbatim note}. Nothing is inferred from
    the numbers; only the published note counts."""
    end_year = None
    m = re.match(r"^(\d{4})/(\d{2})$", financial_year or "")
    if m:
        end_year = str(int(m.group(1)) + 1)
    out: Dict[str, str] = {}
    in_force = False
    names = sorted(set(csp_names), key=len, reverse=True)
    for line in notes_lines:
        s = line.strip()
        if s.startswith("Force Notes"):
            in_force = True
            continue
        if s.startswith("Additional Notes"):
            in_force = False
            continue
        if not in_force or "unassigned" not in s.lower():
            continue
        years = re.findall(r"year ending March (\d{4})", s)
        if years and end_year and end_year not in years:
            continue  # note applies to a different year
        for n in names:
            if n and not n.lower().startswith("unassigned") and re.search(r"\b" + re.escape(n) + r"\b", s):
                out[n] = s
    return out


def load_rows(parsed: Dict[str, Any], source_file: str) -> List[Tuple]:
    """Rows for ew.crime_by_csp: (financial_year, police_force, csp_name,
    offence_group, offences, quarters, usable, quality_note, source_file, published)."""
    fy = parsed["financial_year"]
    q = ",".join(parsed["quarters"])
    flags = quality_flags(parsed["notes_lines"], {k[1] for k in parsed["rows"]}, fy)
    out = []
    for (force, csp, group), n in sorted(parsed["rows"].items()):
        unassigned = csp.lower().startswith("unassigned")
        note = flags.get(csp)
        usable = not unassigned and note is None
        if unassigned:
            note = "Offences the force could not allocate to a district (Home Office 'Unassigned' group)."
        out.append((fy, force, csp, group, int(n), q, usable, note, source_file, parsed.get("published")))
    return out


def latest_csp_file_url(html: str) -> Optional[Tuple[str, str]]:
    """Newest CSP .ods link on the gov.uk dataset page -> (url, file_name)."""
    best = None
    for m in CSP_FILE_RE.finditer(html or ""):
        key = (int(m.group(2)), m.group(3)[4:6] + m.group(3)[2:4] + m.group(3)[0:2])  # end year, yymmdd
        if best is None or key > best[0]:
            best = (key, m.group(0))
    if not best:
        return None
    url = best[1]
    return url, url.rsplit("/", 1)[-1]


# ── payload assembly (used by app.get_crime_data) ───────────────────────────
def month_label(m: Optional[str]) -> Optional[str]:
    mm = re.match(r"^(\d{4})-(\d{2})$", m or "")
    if not mm:
        return None
    names = ["January", "February", "March", "April", "May", "June", "July",
             "August", "September", "October", "November", "December"]
    return f"{names[int(mm.group(2)) - 1]} {mm.group(1)}"


def street_block(http_status: Optional[int], crimes: Any, month: Optional[str],
                 url: str) -> Dict[str, Any]:
    """police.uk result -> street block. Never reports 0 for 'no records'."""
    if http_status != 200 or not isinstance(crimes, list):
        return {"status": "unavailable", "month": month, "total": None, "categories": {},
                "btp_excluded": 0, "records": [], "url": url,
                "note": "police.uk did not return a valid response for this location."}
    force = [c for c in crimes if isinstance(c, dict) and (c.get("location_type") or "Force") == "Force"]
    btp = len(crimes) - len(force)
    # The month comes from the records themselves when there are any (evidence over request);
    # only an empty answer is labelled with the month that was asked for.
    rec_month = next((c.get("month") for c in force if c.get("month")), None) or month
    if not force:
        ml = month_label(rec_month)
        return {"status": "no_records", "month": rec_month, "total": None, "categories": {},
                "btp_excluded": btp, "records": [], "url": url,
                "note": ("police.uk returned no street-level records for this location"
                         + (f" for {ml}" if ml else "") + ". Some forces do not publish to "
                         "police.uk for some months (Greater Manchester Police publishes none).")}
    cats: Dict[str, int] = {}
    for c in force:
        k = c.get("category") or "unknown"
        cats[k] = cats.get(k, 0) + 1
    return {"status": "ok", "month": rec_month, "total": len(force), "categories": cats,
            "btp_excluded": btp, "records": force, "url": url, "note": None}


# ── council (LAD) -> Home Office CSP, via the official ONS lookup ───────────
LOOKUP_FILE = "ew_lad_csp_lookup.json"   # ONS LAD24->CSP24->PFA24 (Dec 2024), committed
# The ONS lookup uses CSP24NM; the Home Office file uses its own CSP names. These three
# differ in more than punctuation. Each is the ONLY unclaimed Home Office CSP in the same
# force and covers exactly the councils the ONS row lists (verified 8 Oct 2026):
HO_NAME_ALIASES = {
    "Shepway": "Folkestone and Hythe",       # Kent; ONS LAD24NM for this CSP is "Folkestone and Hythe"
    "City of London": "London, City of",     # City of London Police
    "North Devon": "Northern Devon",         # Devon & Cornwall; covers North Devon + Torridge
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower().replace("&", "and"))


def load_lookup(path: str) -> Dict[str, List[Dict[str, str]]]:
    import json
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    by_lad: Dict[str, List[Dict[str, str]]] = collections.defaultdict(list)
    for lad_cd, lad_nm, csp_cd, csp_nm, pfa in doc["rows"]:
        by_lad[lad_cd].append({"lad_name": lad_nm, "csp_code": csp_cd, "csp_name": csp_nm, "pfa": pfa})
    return dict(by_lad)


def resolve_district(area_code: Optional[str], lookup: Dict[str, List[Dict[str, str]]],
                     ho_csp_names: Iterable[str]) -> Dict[str, Any]:
    """Council code -> which Home Office CSP(s) to read, and how to label them.

    modes: 'exact'       one CSP covering exactly this council
           'partnership' one CSP shared with other councils (figure is the partnership's)
           'combined'    several CSPs that together cover exactly this council (summed)
           'unmatched'   no honest mapping (e.g. a council split across shared CSPs)
    """
    code = (area_code or "").strip()
    rows = lookup.get(code) or []
    if not rows:
        return {"mode": "unmatched", "lad_name": None, "csps": [],
                "note": "The property's council could not be matched to a Home Office crime district."}
    lad_name = rows[0]["lad_name"]
    ho_by_norm = {_norm(n): n for n in ho_csp_names}

    def _ho(csp_nm: str) -> Optional[str]:
        return ho_by_norm.get(_norm(HO_NAME_ALIASES.get(csp_nm, csp_nm)))

    sharers: Dict[str, List[str]] = collections.defaultdict(list)
    for lad_cd, rs in lookup.items():
        for r in rs:
            sharers[r["csp_code"]].append(r["lad_name"])
    csps = []
    for r in rows:
        ho = _ho(r["csp_name"])
        if not ho:
            return {"mode": "unmatched", "lad_name": lad_name, "csps": [],
                    "note": f"No Home Office crime district matches the {r['csp_name']} partnership."}
        others = [n for n in sharers[r["csp_code"]] if n != lad_name]
        csps.append({"ho_name": ho, "shared_with": others})
    if len(csps) == 1:
        c = csps[0]
        return {"mode": "partnership" if c["shared_with"] else "exact", "lad_name": lad_name,
                "csps": [c["ho_name"]], "shared_with": c["shared_with"]}
    if any(c["shared_with"] for c in csps):
        names = ", ".join(c["ho_name"] for c in csps)
        return {"mode": "unmatched", "lad_name": lad_name, "csps": [],
                "note": (f"The Home Office reports {lad_name} across partnerships shared with other "
                         f"councils ({names}), so no published figure matches this council.")}
    return {"mode": "combined", "lad_name": lad_name, "csps": [c["ho_name"] for c in csps], "shared_with": []}


def district_block(resolution: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """resolve_district() + ew.crime_by_csp rows (latest year, those CSPs) -> district block."""
    base = {"source": DISTRICT_SOURCE, "source_url": DATASET_PAGE}
    lad = resolution.get("lad_name")
    if resolution.get("mode") == "unmatched":
        return {**base, "status": "unavailable", "name": lad, "note": resolution.get("note")}
    want = resolution["csps"]
    have = {r["csp_name"] for r in rows}
    missing = [c for c in want if c not in have]
    if missing:
        return {**base, "status": "unavailable", "name": lad,
                "note": "No Home Office district crime data loaded for " + ", ".join(missing) + "."}
    rows = [r for r in rows if r["csp_name"] in want]
    bad = [r for r in rows if not r.get("usable")]
    r0 = rows[0]
    if bad:
        note = next((r.get("quality_note") for r in bad if r.get("quality_note")), None)
        return {**base, "status": "unavailable", "name": lad,
                "police_force": r0.get("police_force"), "financial_year": r0.get("financial_year"),
                "note": ("Home Office note: " + note) if note else
                        "Flagged unusable by the Home Office for this year."}
    groups: Dict[str, int] = collections.defaultdict(int)
    for r in rows:
        groups[r["offence_group"]] += int(r["offences"])
    groups = dict(groups)
    total = sum(groups.values())
    top = max(groups.items(), key=lambda kv: kv[1]) if groups else None
    mode = resolution["mode"]
    if mode == "exact":
        name = want[0]
    elif mode == "partnership":
        name = f"{want[0]} community safety partnership (shared with {', '.join(resolution['shared_with'])})"
    else:
        name = f"{lad} (Home Office districts: {', '.join(want)})"
    return {**base, "status": "ok", "name": name, "council": lad, "mode": mode, "csps": want,
            "police_force": r0.get("police_force"),
            "financial_year": r0.get("financial_year"),
            "period": period_label(r0.get("financial_year"), (r0.get("quarters") or "").split(",")),
            "total": total, "groups": groups,
            "top_group": top[0] if top else None, "top_count": top[1] if top else None,
            "published": r0.get("published"), "source_file": r0.get("source_file"), "note": None}


def assemble(street: Dict[str, Any], district: Dict[str, Any], retrieved: str,
             max_records: int = 200) -> Dict[str, Any]:
    """Final area_json.crime (England & Wales). Backward-compatible fields:
    metrics.total / categories / month refer to the STREET line only and
    total is None (not 0) whenever police.uk had no force records."""
    sources = [{"label": "UK Police Data API (crimes-street)", "url": street.get("url") or ""},
               {"label": DISTRICT_SOURCE, "url": DATASET_PAGE}]
    parts = []
    if street["status"] == "ok":
        parts.append(f"{street['total']} street-level crimes recorded in {month_label(street['month']) or street['month']} (police.uk).")
    else:
        parts.append(street["note"])
    if district["status"] == "ok":
        parts.append(f"{district['name']}: {district['total']:,} recorded offences, {district['period']} "
                     f"({DISTRICT_SOURCE}).")
    else:
        parts.append(district["note"])
    ok = street["status"] == "ok" or district["status"] == "ok"
    return {
        "status": "ok" if ok else "unavailable",
        "summary": " ".join(p for p in parts if p),
        "value": street["records"][:max_records],
        "metrics": {
            "schema": SCHEMA_VERSION,
            "total": street["total"],
            "categories": street["categories"],
            "month": street["month"],
            "window": "latest month",
            "street_status": street["status"],
            "street_note": street["note"],
            "btp_excluded": street["btp_excluded"],
            "district": district,
        },
        "sources": sources,
        "sourceUrl": street.get("url") or DATASET_PAGE,
        "retrievedAtISO": retrieved,
        "confidenceValue": 0.95 if ok else 0.0,
        "needsEvidence": not ok,
    }


def is_stale(crime: Any, latest_month: Optional[str]) -> bool:
    """True when a stored E&W crime block must be re-fetched: built before
    CRIME-EVID-1, or police.uk has published a newer month than the one stored."""
    if not isinstance(crime, dict):
        return True
    m = crime.get("metrics") or {}
    if (m.get("jurisdiction") or "").lower() == "scotland":
        return False
    if m.get("schema") != SCHEMA_VERSION:
        return True
    if latest_month and (m.get("month") or "") < latest_month:
        return True
    return False
