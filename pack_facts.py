"""
pack_facts.py — subject facts read from the legal pack's OWN documents
(V-PACK, 2026-09-24). Pure: no Flask, no DB, no I/O.

Why (live, deals ef885edd / 7835cfa7 — 2C Talbot Road North, same pack uploaded twice):
  * The same EPC PDF was stored with its text in two different orders:
      23 Sep: "Property type ⏎ End-terrace house ⏎ Total floor area ⏎ 53 square metres"
      24 Sep: "Property type ⏎ Total floor area ⏎ Energy rating ⏎ … ⏎ End-terrace house ⏎ 53 square metres"
    The label-adjacent readers (_extract_epc_floor_area_from_text,
    _extract_epc_property_type_from_text) found nothing on the second, so the
    subject was typed from EPC-register NEIGHBOURS ("Semi", medium) with no floor
    area, and a 53 m² end-terrace was valued off semi-detached houses.
  * Tenure came from the LLM, which read "Leasehold" off service-charge papers in
    a pack with no title register; the register (when present) says Freehold.
  * Rent actually paid (£850/month, agent statements) was never read into any field.

Rules:
  * Order-robust: values are matched by their fixed wording, not by position.
  * An EPC certificate is used only if its OWN address block matches the subject
    (postcode, and house number when the subject has one).
  * Every fact carries the file it came from and the line it was read from.
  * Nothing is inferred: a fact that cannot be read is None.
"""
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

_WS = re.compile(r"\s+")


def _norm_pc(pc: Optional[str]) -> str:
    return re.sub(r"\s+", "", (pc or "")).upper()


def _house_token(address: Optional[str]) -> Optional[str]:
    """First house-number token of an address, e.g. '2c' from '2c Talbot Road North'."""
    m = re.match(r"\s*(?:flat\s+\w+,?\s*)?(\d+[a-z]?)\b", (address or ""), re.I)
    return m.group(1).lower() if m else None


def _page1(text: str) -> str:
    """Text of the first page (DocAI / extractor '=== PAGE n ===' markers)."""
    parts = re.split(r"===\s*PAGE\s+\S+\s*===", text or "")
    body = [p for p in parts if p.strip()]
    return body[0] if body else (text or "")


# ── EPC certificate (GOV.UK format) ─────────────────────────────────────────
_EPC_TYPE = re.compile(
    r"\b(enclosed\s+end-terrace|enclosed\s+mid-terrace|end-terrace|mid-terrace|"
    r"semi-detached|detached)\s+(house|bungalow)\b"
    r"|\b(ground-floor|mid-floor|top-floor|basement)\s+(flat|maisonette)\b"
    r"|\b(flat|maisonette)\b(?=\s*(?:\n|total floor area|$))",
    re.I,
)
# Label-adjacent first (England/Wales "Total floor area: 81 square metres",
# Scotland / energy reports "Total floor area: 117 m2"), then the England/Wales
# layout where labels and values are stored in separate runs ("… 53 square metres").
# A bare "m2" is never read without its label (it also appears as kWh/m2).
_EPC_AREA_LABELLED = re.compile(
    r"total\s+floor\s+area\s*:?\s*(\d{2,4}(?:\.\d+)?)\s*(?:square\s*met(?:re|er)s?|m2|m²)\b", re.I)
_EPC_AREA = re.compile(r"\b(\d{2,4}(?:\.\d+)?)\s*square\s*met(?:re|er)s?\b", re.I)
_EPC_DATE = re.compile(r"date\s+of\s+assessment\s*[:\n]?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})", re.I)
_EPC_CERT = re.compile(r"\b(\d{4}-\d{4}-\d{4}-\d{4}-\d{4})\b")


def _type_code(label: str) -> Optional[str]:
    l = label.upper()
    if "FLAT" in l or "MAISONETTE" in l:
        return "F"
    if "SEMI" in l:
        return "S"
    if "TERRACE" in l:
        return "T"
    if "DETACH" in l:
        return "D"
    return None


def read_epc(text: str, subject_postcode: Optional[str], subject_address: Optional[str]) -> Optional[Dict[str, Any]]:
    """Read one GOV.UK EPC certificate. Returns None if the text is not an EPC or
    its address block does not match the subject."""
    if not text or not re.search(r"energy\s+performance\s+certificate", text, re.I):
        return None
    p1 = _page1(text)
    head = p1[:600]
    spc = _norm_pc(subject_postcode)
    if not spc or spc not in _norm_pc(head):
        return None
    tok = _house_token(subject_address)
    addr_ok = True
    if tok:
        addr_ok = re.search(r"(?<![0-9a-z])" + re.escape(tok) + r"(?![0-9a-z])", head, re.I) is not None
    if not addr_ok:
        return None
    t = _EPC_TYPE.search(p1)
    a = _EPC_AREA_LABELLED.search(p1) or _EPC_AREA.search(p1)
    area = None
    if a:
        try:
            v = float(a.group(1))
            area = v if 10.0 <= v <= 2000.0 else None
        except ValueError:
            area = None
    label = _WS.sub(" ", t.group(0)).strip() if t else None
    d = _EPC_DATE.search(text)
    date_iso = None
    if d:
        try:
            date_iso = datetime.strptime(d.group(1).strip(), "%d %B %Y").date().isoformat()
        except ValueError:
            date_iso = None
    c = _EPC_CERT.search(p1)
    return {
        "property_type_label": label,
        "type_code": _type_code(label) if label else None,
        "floor_area_m2": area,
        "assessment_date": date_iso,
        "certificate_number": c.group(1) if c else None,
        "address_match": "postcode+house_number" if tok else "postcode_only",
        "evidence": {
            "type_line": label,
            "area_line": _WS.sub(" ", a.group(0)).strip() if a else None,
        },
    }


# ── Title register tenure ───────────────────────────────────────────────────
_REG_TENURE = re.compile(r"\bthe\s+(freehold|leasehold)\s+land\b", re.I)


def read_register_tenures(text: str) -> List[str]:
    return sorted({m.group(1).capitalize() for m in _REG_TENURE.finditer(text or "")})


# ── Rent actually paid (managing-agent statements) ──────────────────────────
_STMT_PERIOD = re.compile(
    r"rents?\s+received\s+for\s+the\s+period\s*:?\s*"
    r"(\d{2}/\d{2}/\d{4})\s*-\s*(\d{2}/\d{2}/\d{4})\s*£\s*([\d,]+\.\d{2})",
    re.I,
)
_STMT_COMMISSION = re.compile(
    r"commission\s+on\s+collection\s*(?:vat\s*)?£\s*([\d,]+\.\d{2})\s*£\s*([\d,]+\.\d{2})",
    re.I,
)


def _money(s: str) -> float:
    return float(s.replace(",", ""))


def read_rent_statements(text: str, subject_address: Optional[str]) -> List[Dict[str, Any]]:
    """Each 'Rents received for the Period: dd/mm/yyyy-dd/mm/yyyy £N' block,
    only when the statement names the subject (house number + a street word)."""
    t = text or ""
    tok = _house_token(subject_address)
    street = None
    m = re.match(r"\s*\d+[a-z]?\s+([a-z]+)", subject_address or "", re.I)
    if m:
        street = m.group(1)
    if tok and street:
        if not re.search(re.escape(tok) + r"\s+" + re.escape(street), t, re.I):
            return []
    flat = _WS.sub(" ", t)
    out = []
    for pm in _STMT_PERIOD.finditer(flat):
        rec = {"period_start": pm.group(1), "period_end": pm.group(2), "rent_gbp": _money(pm.group(3))}
        cm = _STMT_COMMISSION.search(flat, pm.end(), pm.end() + 200)
        if cm:
            rec["commission_gbp"] = _money(cm.group(1))
            rec["commission_vat_gbp"] = _money(cm.group(2))
        out.append(rec)
    return out


def _iso(dmy: str) -> Optional[str]:
    try:
        return datetime.strptime(dmy, "%d/%m/%Y").date().isoformat()
    except ValueError:
        return None


# ── Resolver ────────────────────────────────────────────────────────────────
def _is_flat_address(address: Optional[str]) -> bool:
    return bool(re.search(r"\b(flat|apartment|maisonette)\b", address or "", re.I))


def resolve_pack_facts(docs: List[Dict[str, Any]], subject_address: Optional[str],
                       subject_postcode: Optional[str], subject_is_flat: bool = False) -> Dict[str, Any]:
    """docs: [{file_name, doc_type, extracted_text, extraction_status}]"""
    out: Dict[str, Any] = {"epc": None, "tenure": None, "rent": None, "unread_documents": []}

    for d in docs or []:
        if (d.get("extraction_status") or "") == "empty" or not (d.get("extracted_text") or "").strip():
            if d.get("file_name"):
                out["unread_documents"].append(d["file_name"])

    # EPC — the first address-matched certificate; conflicting certificates → none
    epcs = []
    for d in docs or []:
        r = read_epc(d.get("extracted_text") or "", subject_postcode, subject_address)
        if r:
            r["source_file"] = d.get("file_name")
            epcs.append(r)
    if epcs:
        codes = {e["type_code"] for e in epcs if e["type_code"]}
        areas = {e["floor_area_m2"] for e in epcs if e["floor_area_m2"]}
        if len(codes) <= 1 and len(areas) <= 1:
            out["epc"] = epcs[0]
        else:
            out["epc_conflict"] = [{"file": e["source_file"], "type": e["property_type_label"],
                                    "area": e["floor_area_m2"]} for e in epcs]

    # Tenure — title registers only, and only registers that NAME the subject
    # (house number). Freehold AND leasehold registers together → ambiguous.
    # AUDIT 2026-09-24 (live): 2 leasehold-flat deals (95b Woodside SW19; Flat B,
    # 18 Grosvenor Avenue) carry only the BUILDING's freehold register. A freehold
    # register for a flat is the landlord's title, not the subject's interest, so
    # it must never override the subject's tenure.
    flat = subject_is_flat or _is_flat_address(subject_address) or \
        bool(out.get("epc") and out["epc"].get("type_code") == "F")
    tok = _house_token(subject_address)
    tenures, reg_files = set(), []
    for d in docs or []:
        if (d.get("doc_type") or "") == "title_register":
            txt = d.get("extracted_text") or ""
            ts = read_register_tenures(txt)
            if not ts:
                continue
            if tok and not re.search(r"(?<![0-9a-z])" + re.escape(tok) + r"(?![0-9a-z])", txt, re.I):
                continue
            tenures.update(ts)
            reg_files.append(d.get("file_name"))
    if len(tenures) == 1:
        val = tenures.pop()
        if flat and val == "Freehold":
            out["tenure_ambiguous"] = {"values": ["Freehold"], "source_files": reg_files,
                                       "reason": "freehold register in a flat's pack — the building's title, not the flat's"}
        else:
            out["tenure"] = {"value": val, "source_files": reg_files}
    elif len(tenures) > 1:
        out["tenure_ambiguous"] = {"values": sorted(tenures), "source_files": reg_files}

    # Rent — statement periods, de-duplicated by period
    periods: Dict[str, Dict[str, Any]] = {}
    files = set()
    for d in docs or []:
        for rec in read_rent_statements(d.get("extracted_text") or "", subject_address):
            periods.setdefault(rec["period_start"], rec)
            files.add(d.get("file_name"))
    if periods:
        recs = sorted(periods.values(), key=lambda r: _iso(r["period_start"]) or "")
        mode_rent, _n = Counter(r["rent_gbp"] for r in recs).most_common(1)[0]
        latest = recs[-1]
        comm = [r for r in recs if "commission_gbp" in r and r["rent_gbp"] > 0]
        out["rent"] = {
            "monthly_rent_gbp": latest["rent_gbp"],
            "most_common_rent_gbp": mode_rent,
            "months_evidenced": len(recs),
            "first_period_start": _iso(recs[0]["period_start"]),
            "last_period_end": _iso(latest["period_end"]),
            "agent_commission_pct_incl_vat": (
                round(100 * (comm[-1]["commission_gbp"] + comm[-1]["commission_vat_gbp"]) / comm[-1]["rent_gbp"], 1)
                if comm else None),
            "source": "managing-agent rent statements in the legal pack",
            "source_files": sorted(f for f in files if f),
        }
    return out


# ── OCR routing: a text layer that is only HM Land Registry stamps is NOT text ──
# AUDIT 2026-09-24 (live): 50 documents on 38 of 111 deals (1,437 pages) were
# stored "complete" although their only text was the HMLR official-copy notes
# and the per-page stamp "This official copy is incomplete without the
# preceding notes page." — the scanned body (lease terms, transfer covenants,
# TA6 answers) was never OCR'd, because OCR ran only when extraction returned
# NO text at all. Live distribution of genuine documents: p05 = 130
# non-whitespace chars/page; every stamp-only document was < 40 after the
# boilerplate below is removed.
_HMLR_BOILERPLATE = re.compile(
    r"this official copy is incomplete without the preceding notes page\.?"
    r"|these are the notes referred to on the following official copy"
    r"|the electronic official copy of the (?:document|register|title plan) follows this\s*message\.?"
    r"|this copy may not be the same size as the\s*original\.?"
    r"|please note that this is the only official copy we will issue\.?\s*we will not issue\s*a paper official copy\.?"
    r"|title number\s+\S+"
    r"|===\s*page\s+\S+\s*===",
    re.I,
)
MIN_REAL_CHARS_PER_PAGE = 40


def real_chars_per_page(text: Optional[str], pages: Optional[int]) -> float:
    body = _HMLR_BOILERPLATE.sub("", text or "")
    return len(re.sub(r"\s", "", body)) / max(int(pages or 0), 1)


# Font-garbled text layers (broken encodings: text present but unreadable).
# Live: 2 of 1,111 'complete' documents exceed this share (e.g. Lot 132 EPC).
_READABLE = re.compile(r"[ -~\s£€²³–—’‘“”•…·]")
MAX_UNREADABLE_SHARE = 0.20


def unreadable_share(text: Optional[str]) -> float:
    t = text or ""
    return (len(t) - len(_READABLE.findall(t))) / len(t) if t else 0.0


def text_layer_is_unusable(text: Optional[str], pages: Optional[int]) -> bool:
    """True when extracted text is empty, only HMLR boilerplate/stamps, or
    font-garbled → the document needs OCR."""
    if not (text or "").strip():
        return True
    if real_chars_per_page(text, pages) < MIN_REAL_CHARS_PER_PAGE:
        return True
    return len(text) > 200 and unreadable_share(text) > MAX_UNREADABLE_SHARE
