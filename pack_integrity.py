"""
pack_integrity.py — PACK-INTEG-1 (2026-10-04)

Finds documents in an upload that are about a DIFFERENT property, so they are
not analysed as part of this lot. Pure functions, no I/O.

WHY (evidence, deal 92f1d4c4, 4 Oct 2026):
  The Lot 73 upload (Highfields Care Home, Newent GL18 1JA) also contained
  `Lot_6_Special_conditions.docx`, whose own text says "The property is known
  as 4 White Lion Yard, Gainsborough, DN21 2DD". Nothing compared the two.
  21 of the deal's 50 flags — all 4 criticals — came from that document, and
  the deal showed two contradictory completion periods (14 days from Lot 6,
  20 working days from Lot 73).

RULE (content only, never the file name; fails OPEN — when unsure, nothing is
excluded):
  1. Property postcode of a document = a postcode that follows a phrase naming
     the property itself, within ANCHOR_SPAN characters:
       "property (is) known as", "property/premises/site/search address",
       "address of (the) property/premises", "the property:",
       "subject property", "land … shown/edged … being".
     Party addresses ("… of 10 Fenchurch Avenue, London EC3M 5AG"), council,
     solicitor and search-provider addresses are not property postcodes.
  2. The lot postcode = the property postcode named by the MOST documents.
     A tie, or no property postcode anywhere -> undetermined, nothing excluded.
  3. A document is about another property only if ALL hold:
       a. it names a property postcode;
       b. none of its property postcodes is in the lot's postcode district;
       c. the lot's district appears nowhere in it;
       d. its property district appears in no document that also mentions the
          lot's district (so a portfolio lot whose special conditions list
          every address keeps all its documents).
  Excluded documents are named with the quote that identifies them.

Corpus check (4 Oct 2026, all 1,260 stored documents across 122 deals, rule
run as SQL): exactly 1 document met the rule — Lot_6_Special_conditions.docx
in deal 92f1d4c4. A postcode-only rule (no anchors) caught 66, 65 of them
the lot's own documents (lender, council, solicitor addresses), which is why
the anchors are required.

Also reported, never excluded: files whose "Lot_<n>_" name prefix differs
from the prefix most files share (a name is not evidence of content).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional

VERSION = "pack-integ-1"
ANCHOR_SPAN = 140

_PC = r"([A-Z]{1,2}[0-9][A-Z0-9]?) ?([0-9][A-Z]{2})(?![A-Z0-9])"
_PC_ANY = re.compile(r"(?:^|[^A-Z0-9])" + _PC)
_ANCHOR = re.compile(
    r"(PROPERTY\s+(?:IS\s+)?KNOWN\s+AS"
    r"|(?:PROPERTY|PREMISES|SITE|SEARCH)\s+ADDRESS"
    r"|ADDRESS\s+OF\s+(?:THE\s+)?(?:PROPERTY|PREMISES)"
    r"|THE\s+PROPERTY\s*:"
    r"|SUBJECT\s+PROPERTY"
    r"|LAND\s+(?:AND\s+BUILDINGS\s+)?(?:SHOWN|EDGED)[^.]{0,120}?BEING)"
    r"[^0-9A-Z]{0,3}?[\s\S]{0,%d}?[^A-Z0-9]" % ANCHOR_SPAN + _PC, re.I)
_LOT_PREFIX = re.compile(r"^lot_([0-9]+[a-z]?(?:\.[0-9]+)?)_", re.I)


def _norm(outward: str, inward: str) -> str:
    return f"{outward} {inward}"


def _district(pc: str) -> str:
    return pc.split(" ")[0]


def property_postcodes(text: str) -> List[Dict[str, str]]:
    """Postcodes the document gives as the property's own address, with the quote."""
    # Case-insensitive on the original text (not .upper(): "ß" -> "SS" would
    # shift the quote offsets). Same match set as the SQL corpus check.
    out, seen = [], set()
    for m in _ANCHOR.finditer(text or ""):
        pc = _norm(m.group(2).upper(), m.group(3).upper())
        if pc in seen:
            continue
        seen.add(pc)
        quote = re.sub(r"\s+", " ", (text or "")[m.start():m.end()]).strip()
        out.append({"postcode": pc, "quote": quote})
    return out


def all_districts(text: str) -> set:
    return {m.group(1) for m in _PC_ANY.finditer((text or "").upper())}


def check(documents: List[Dict]) -> Dict:
    """Return the integrity result. `excluded` holds indexes into `documents`."""
    rows = []
    for i, d in enumerate(documents or []):
        txt = d.get("extracted_text") or ""
        if not txt.strip():
            continue
        rows.append({"i": i, "name": d.get("file_name") or "(unnamed)",
                     "props": property_postcodes(txt), "districts": all_districts(txt)})

    named = Counter()
    for r in rows:
        for p in {x["postcode"] for x in r["props"]}:
            named[p] += 1

    result: Dict = {"version": VERSION, "lot_postcode": None, "lot_postcode_documents": 0,
                    "status": "no_property_address", "excluded": [], "excluded_files": [],
                    "named_for_other_lot": _other_lot_names(documents)}
    if not named:
        return result
    top = named.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        result["status"] = "undetermined"
        result["candidates"] = [{"postcode": p, "documents": n} for p, n in top if n == top[0][1]]
        return result

    lot_pc, lot_n = top[0]
    lot_d = _district(lot_pc)
    result.update({"lot_postcode": lot_pc, "lot_postcode_documents": lot_n, "status": "checked"})

    linking = [r["districts"] for r in rows if lot_d in r["districts"]]
    for r in rows:
        if not r["props"]:
            continue
        prop_ds = {_district(x["postcode"]) for x in r["props"]}
        if lot_d in prop_ds or lot_d in r["districts"]:
            continue
        if any(pd in ds for pd in prop_ds for ds in linking):
            continue
        first = r["props"][0]
        result["excluded"].append(r["i"])
        result["excluded_files"].append({
            "file_name": r["name"], "postcode": first["postcode"], "quote": first["quote"],
            "reason": f"names {first['postcode']} as its property; this lot is {lot_pc}",
        })
    return result


def _other_lot_names(documents: List[Dict]) -> List[str]:
    pref = []
    for d in documents or []:
        m = _LOT_PREFIX.match(d.get("file_name") or "")
        pref.append(m.group(1).lower() if m else None)
    counted = Counter(p for p in pref if p)
    if len(counted) < 2:
        return []
    main, n = counted.most_common(1)[0]
    if n < 2 or list(counted.values()).count(n) > 1:
        return []
    return [d.get("file_name") for d, p in zip(documents, pref) if p and p != main]


def split(documents: List[Dict]) -> (List[Dict], Dict):
    """(documents to analyse, integrity result)."""
    res = check(documents)
    drop = set(res["excluded"])
    return [d for i, d in enumerate(documents or []) if i not in drop], res
