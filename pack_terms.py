"""
pack_terms.py — COMM-1 (2026-10-05)

Reads, without a model, two things a buyer needs from the special conditions:

  1. BUYER COSTS: every sum the special conditions make the buyer pay on top
     of the price, each with the clause it comes from.
  2. WHAT YOU'RE BUYING: the property as the pack defines it, title
     guarantee, possession, completion period.

Every item carries `quote`, the pack's own wording, plus the file it came
from. Nothing is estimated:
  * an amount written in words is converted only when every word is a
    number word; otherwise the item shows the quote with amount None;
  * an obligation with no stated sum ("the cost of any local search") is
    listed with amount None ("not stated in the pack");
  * a percentage of the price is never turned into pounds here (the price
    is the user's input);
  * VAT is never calculated, only marked ("+ VAT") as the clause says.
Costs that arise only on a condition (notice to complete, late completion,
change of solicitor) are listed separately as `contingent`.

Documents read: special conditions and addenda only. Standard auction
conditions (Common Auction Conditions / RICS CAC) are skipped: their clauses
are generic ("the buyer must pay the deposit") and are not this lot's terms.
Live store, 5 Oct 2026: 39 of 141 stored special-conditions documents open as
Common Auction Conditions.

Display only. Nothing here feeds a valuation.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

VERSION = "pack-terms-1"

# ── number words ─────────────────────────────────────────────────────────────
_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_NUMWORD = r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)"
_WORDS_RUN = r"(?:%s(?:[\s,\-]+(?:and[\s,]+)?(?:a\s+half|point|%s))*)" % (_NUMWORD, _NUMWORD)


def words_to_number(phrase: str) -> Optional[float]:
    """'one thousand, four hundred and forty' -> 1440; 'two point seventy five'
    -> 2.75; 'two and a half' -> 2.5; 'fifteen hundred' -> 1500. None if any
    token is not a number word."""
    toks = [t for t in re.split(r"[\s,\-]+", phrase.lower().strip()) if t]
    if not toks:
        return None
    if "point" in toks:
        i = toks.index("point")
        whole, frac = words_to_number(" ".join(toks[:i])), words_to_number(" ".join(toks[i + 1:]))
        if whole is None or frac is None or frac != int(frac):
            return None
        return whole + float("0." + str(int(frac)))
    total, cur, half, seen = 0, 0, False, False
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "and":
            i += 1
            continue
        if t == "a" and i + 1 < len(toks) and toks[i + 1] == "half":
            half = True
            i += 2
            continue
        if t in _UNITS:
            cur += _UNITS[t]
        elif t in _TENS:
            cur += _TENS[t]
        elif t == "hundred":
            cur = (cur or 1) * 100
        elif t == "thousand":
            total += (cur or 1) * 1000
            cur = 0
        else:
            return None
        seen = True
        i += 1
    if not seen:
        return None
    return total + cur + (0.5 if half else 0)


# ── document selection ───────────────────────────────────────────────────────
def is_lot_conditions(doc: Dict) -> bool:
    """Special conditions / addendum of THIS lot (not standard auction conditions)."""
    name = (doc.get("file_name") or "").lower()
    dt = (doc.get("doc_type") or "").lower()
    text = doc.get("extracted_text") or doc.get("text") or ""
    if not text.strip():
        return False
    if not (dt in ("special_conditions", "addendum") or "special condition" in name.replace("_", " ")
            or "addendum" in name):
        return False
    head = text[:800].lower()
    if "common auction conditions" in head and "special condition" not in head:
        return False
    return True


# ── clause splitting ─────────────────────────────────────────────────────────
_PAGE = re.compile(r"=== PAGE \S+ ===")


def _clauses(text: str) -> List[str]:
    """Paragraph-level clauses; numbered clauses split even without a blank line."""
    t = _PAGE.sub("\n", text or "")
    t = re.sub(r"[ \t]*\n[ \t]*\n\s*", "\n\n", t)
    t = re.sub(r"\n(?=\s*\d{1,2}(?:\.\d{1,2})?\.?\s+[A-Z(])", "\n\n", t)
    out = []
    for p in t.split("\n\n"):
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            out.append(p)
    return out


def _sentences(clause: str) -> List[str]:
    parts = re.split(r"(?<=[.;])\s+(?=[A-Z(])", clause)
    return [p.strip() for p in parts if p.strip()]


# ── buyer costs ──────────────────────────────────────────────────────────────
_BUYER_PAY = re.compile(
    r"\bbuyer\b[^.;]{0,120}?\b(?:shall|will|must|is\s+to|is\s+liable\s+to|shall\s+become\s+liable\s+to|"
    r"shall\s+be\s+responsible\s+to|will\s+be\s+charged|be\s+charged|is\s+responsible\s+for)\b"
    r"|\bcharge\s+the\s+buyer\b|\bpayable\s+by\s+the\s+buyer\b|\bat\s+the\s+cost\s+of\s+the\s+buyer\b|\bbuyer['’]s\s+(?:cost|expense)\b",
    re.I)
_PAYISH = re.compile(r"\b(?:pay|paid|payable|charged|cost|costs|contribution|reimburse|indemnify|fee|sum)\b", re.I)
_CONTINGENT = re.compile(
    r"\b(?:if|should|in\s+the\s+event|where\s+notice|when\s+notice|notice\s+to\s+complete|fails?\s+to|"
    r"default|delay(?:ed)?|change\s+their\s+legal|represent\s+themselves|any\s+such\s+enquir)", re.I)
_NOT_EXTRA = re.compile(r"\bdeposit\b", re.I)
_UNSTATED_NOUN = re.compile(
    r"\b(?:local\s+search|drainage\s+search|search(?:es)?|arrears|service\s+charges?|estate\s+charges?|"
    r"ground\s+rent|insurance|indemnity\s+polic(?:y|ies)|deed\s+of\s+covenant|licence\s+to\s+assign|disbursements?)\b",
    re.I)
_VAT_AFTER = re.compile(
    r"(?:plus|\+|exclusive\s+of)\s*(?:a\s+sum\s+equivalent\s+to\s+)?(?:vat|value\s+added\s+tax)", re.I)
_CONTINUES = re.compile(r"^(?:plus|in\s+addition|additionally|together\s+with)\b", re.I)
_COST_OF_ANY = re.compile(r"\bpay\b(?P<mid>[^.;£]{0,40}?)\bthe\s+cost\s+of\s+any\b[^.;£]{0,120}?(?=\s+and\s+a\b|[.;]|$)", re.I)
_UNIT_AFTER = re.compile(r"^\s*(?:plus\s+vat\s+)?per\s+(day|hour|transfer|calendar\s+month|month|annum)\b", re.I)
_GBP_DIGITS = re.compile(r"£\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?")
_GBP_WORDS = re.compile(r"\b(" + _WORDS_RUN + r")\s+pounds\b", re.I)
_PCT = re.compile(r"\b(\d+(?:\.\d+)?|" + _WORDS_RUN + r")\s*(?:%|per\s*cent|percent)\s+of\s+the\s+(?:purchase\s+)?price", re.I)
_MINIMUM = re.compile(r"minimum\s+(?:sum\s+)?of\s+(?:£\s?(\d{1,3}(?:,\d{3})+|\d+)|(" + _WORDS_RUN + r")\s+pounds)", re.I)
_WHEN = [(re.compile(r"\bexchange\b", re.I), "exchange"), (re.compile(r"\bcompletion\b", re.I), "completion")]


def _amounts(sentence: str) -> List[Dict]:
    found = []
    for m in _GBP_DIGITS.finditer(sentence):
        val = float(m.group(1).replace(",", "") + ("." + m.group(2) if m.group(2) else ""))
        found.append((m.start(), m.end(), {"basis": "fixed", "amount_gbp": val, "as_written": m.group(0)}))
    for m in _GBP_WORDS.finditer(sentence):
        val = words_to_number(m.group(1))
        found.append((m.start(), m.end(), {"basis": "fixed", "amount_gbp": val, "as_written": m.group(0)}))
    for m in _PCT.finditer(sentence):
        raw = m.group(1)
        val = float(raw) if re.fullmatch(r"\d+(?:\.\d+)?", raw) else words_to_number(raw)
        found.append((m.start(), m.end(), {"basis": "percent_of_price", "percent": val, "as_written": m.group(0)}))
    # a "minimum of £X" belongs to a percentage in the same sentence, not a
    # separate charge; with no percentage it IS the charge (a stated minimum)
    has_pct = any(it["basis"] == "percent_of_price" for _, _, it in found)
    mins = [(mm.start(), mm.end()) for mm in _MINIMUM.finditer(sentence)] if has_pct else []
    out = []
    found = sorted(found, key=lambda x: x[0])
    for k, (a, b, item) in enumerate(found):
        if any(ma <= a < mb for ma, mb in mins):
            continue
        # VAT words belong to this amount if they come before the next amount
        nxt = next((fa for fa, _, _ in found[k + 1:] if not any(ma <= fa < mb for ma, mb in mins)), len(sentence))
        tail = sentence[b:nxt]
        item["plus_vat"] = bool(_VAT_AFTER.search(tail)) and not re.search(r"inclusive\s+of\s+(?:vat|value)", tail, re.I)
        um = _UNIT_AFTER.search(tail)
        item["per"] = um.group(1).lower() if um else None
        if not has_pct and re.search(r"minimum\s+(?:sum\s+)?of\s*$", sentence[max(0, a - 25):a], re.I):
            item["is_minimum"] = True
        out.append(item)
    for mm in (_MINIMUM.finditer(sentence) if has_pct else []):
        mv = (float(mm.group(1).replace(",", "")) if mm.group(1) else words_to_number(mm.group(2)))
        for it in out:
            if it["basis"] == "percent_of_price" and it.get("minimum_gbp") is None:
                it["minimum_gbp"] = mv
                break
    return out


def extract_buyer_costs(documents: List[Dict]) -> Dict:
    items: List[Dict] = []
    read = []
    for d in documents or []:
        if not is_lot_conditions(d):
            continue
        name = d.get("file_name") or "(unnamed)"
        read.append(name)
        text = d.get("extracted_text") or d.get("text") or ""
        for clause in _clauses(text):
            prev_pay = False
            for s in _sentences(clause):
                pays = bool(_BUYER_PAY.search(s) and _PAYISH.search(s)) or (prev_pay and bool(_CONTINUES.search(s)))
                prev_pay = pays
                if not pays:
                    continue
                if _NOT_EXTRA.search(s):
                    continue   # the deposit is part of the price, not a cost on top
                contingent = bool(_CONTINGENT.search(s))
                hits = [(m.start(), w) for rx, w in _WHEN for m in [rx.search(s)] if m]
                when = min(hits)[1] if hits else None
                amts = _amounts(s)
                co = _COST_OF_ANY.search(s)
                if co and not re.search(r"towards|contribution", co.group("mid"), re.I):
                    items.append({"basis": "not_stated", "amount_gbp": None, "plus_vat": False, "per": None,
                                  "contingent": contingent, "when": when, "quote": s, "file_name": name,
                                  "as_written": co.group(0)[co.group(0).lower().index("the cost of"):].strip()})
                if amts:
                    for a in amts:
                        items.append({**a, "contingent": contingent, "when": when,
                                      "quote": s, "file_name": name})
                elif _UNSTATED_NOUN.search(s):
                    items.append({"basis": "not_stated", "amount_gbp": None, "plus_vat": False, "per": None,
                                  "contingent": contingent, "when": when, "quote": s, "file_name": name,
                                  "as_written": None})
    certain = [i for i in items if not i["contingent"]]
    fixed = [i for i in certain if i["basis"] == "fixed" and i.get("amount_gbp") is not None and not i.get("per")]
    return {
        "version": VERSION,
        "documents_read": read,
        "items": certain,
        "contingent": [i for i in items if i["contingent"]],
        "fixed_total_gbp_ex_vat": round(sum(i["amount_gbp"] for i in fixed), 2) if fixed else None,
        "fixed_total_counts": len(fixed),
        "percent_items": [i for i in certain if i["basis"] == "percent_of_price"],
        "unstated_items": [i for i in certain if i["basis"] == "not_stated"
                           or (i["basis"] == "fixed" and i.get("amount_gbp") is None)],
    }


# ── what you're buying ───────────────────────────────────────────────────────
_TITLE_GUARANTEE = re.compile(r"\b(full|limited)\s+title\s+guarantee\b", re.I)
_COMPLETION = re.compile(
    r"\bcompletion\b[^.;]{0,60}?\b(\d{1,3}|" + _WORDS_RUN + r")\s+(working\s+days|days|weeks)\b[^.;]{0,60}"
    r"|\bcompletion\s+date\s+(?:shall\s+be|is)\s+(\d{1,3}|" + _WORDS_RUN + r")\s+(working\s+days|days|weeks)\b[^.;]{0,60}"
    r"|agreed\s+completion\s+date\s*:?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})", re.I)
_VP = re.compile(r"vacant\s+possession", re.I)
_PROPERTY_DEF = re.compile(
    r"(?:[\"“]the\s+property[\"”]\s+is|the\s+property\s+is\s+known\s+as|brief\s+description\s+of\s+the\s+lot)\s*:?\s*"
    r"([^.]{5,200}?\b[A-Z]{1,2}[0-9][A-Z0-9]?\s?[0-9][A-Z]{2}\b)", re.I)
_TENURE_SC = re.compile(r"\b(?:title\s*:?\s*|sold\s+)(freehold|leasehold)\b", re.I)
_TENURE_REG = re.compile(r"\bthe\s+(freehold|leasehold)\s+land\b", re.I)
_TITLE_NO = re.compile(r"\btitle\s+(?:number|no\.?)\s*:?\s*([A-Z]{1,3}\d{2,7})\b", re.I)


def extract_buying(documents: List[Dict]) -> Dict:
    """Facts from the lot's special conditions, each with its quote. Missing
    facts are listed in `not_stated`, never filled."""
    out: Dict = {"version": VERSION, "facts": {}, "not_stated": []}
    sc = [d for d in documents or [] if is_lot_conditions(d)]
    for d in sc:
        name = d.get("file_name") or "(unnamed)"
        flat = re.sub(r"\s+", " ", _PAGE.sub(" ", d.get("extracted_text") or d.get("text") or ""))
        f = out["facts"]
        if "property" not in f:
            m = _PROPERTY_DEF.search(flat)
            if m:
                f["property"] = {"value": m.group(1).strip(" ,;:"), "quote": m.group(0).strip(), "file_name": name}
        if "title_number" not in f:
            m = _TITLE_NO.search(flat)
            if m:
                f["title_number"] = {"value": m.group(1).upper(), "quote": m.group(0), "file_name": name}
        if "title_guarantee" not in f:
            m = _TITLE_GUARANTEE.search(flat)
            if m:
                f["title_guarantee"] = {"value": m.group(1).capitalize() + " title guarantee",
                                        "quote": _around(flat, m, 30, 30), "file_name": name}
        if "completion" not in f:
            m = _COMPLETION.search(flat)
            if m:
                f["completion"] = {"value": m.group(0).strip(), "quote": m.group(0).strip(), "file_name": name}
        if "tenure" not in f:
            m = _TENURE_SC.search(flat)
            if m:
                f["tenure"] = {"value": m.group(1).capitalize(), "quote": _around(flat, m, 30, 30), "file_name": name}
        if "possession" not in f:
            for m in _VP.finditer(flat):
                state = possession_statement(flat, m.start(), m.end())
                if state:
                    f["possession"] = {"value": state, "quote": _around(flat, m, 200, 60), "file_name": name}
                    break
    if "tenure" not in out["facts"]:
        regs = [d for d in documents or [] if (d.get("doc_type") or "") == "title_register"]
        for d in regs:
            flat = re.sub(r"\s+", " ", _PAGE.sub(" ", d.get("extracted_text") or d.get("text") or ""))
            m = _TENURE_REG.search(flat)
            if m:
                vals = {x.group(1).lower() for x in _TENURE_REG.finditer(flat)}
                if len(vals) == 1:
                    out["facts"]["tenure"] = {"value": m.group(1).capitalize(), "quote": _around(flat, m, 200, 0),
                                              "file_name": d.get("file_name")}
                break
    for k in ("property", "title_number", "tenure", "title_guarantee", "possession", "completion"):
        if k not in out["facts"]:
            out["not_stated"].append(k)
    out["documents_read"] = [d.get("file_name") for d in sc]
    return out


def _around(text: str, m, span: int = 160, before: int = 40) -> str:
    """The sentence around a match, cut at the next sentence end or clause
    number ("… Guarantee 9. The contract …" stops before "9."), never mid-word."""
    a = max(0, max(text.rfind(ch, 0, m.start()) for ch in ".?") + 1, m.start() - before)
    nxt = re.compile(r"\.(?:\s|$)|\s\d{1,2}(?:\.\d{1,2})?\.\s")
    em = nxt.search(text, m.end())
    b = em.start() + (1 if text[em.start()] == "." else 0) if em else len(text)
    if b > m.end() + span:
        b = m.end() + span
        sp = text.rfind(" ", m.end(), b)
        b = sp if sp > m.end() else b
    if a > 0 and a == m.start() - before:
        sp = text.find(" ", a, m.start())
        a = sp + 1 if sp != -1 else a
    return text[a:b].strip()


# ── vacant possession: shared with commercial_extraction ─────────────────────
_VP_NOT_SALE = re.compile(
    r"yield\s+up|deliver\s+up|give\s+up|return\s+the\s+(?:property|premises)|surrender|determination|"
    r"expiry|end\s+of\s+the\s+term|hypothetical|assum|willing\s+(?:landlord|tenant)|review|"
    r"let\s+as\s+a\s+whole|otherwise", re.I)
_VP_SALE = re.compile(r"\bsold\b|\bsale\b|vacant\s+or\s+let", re.I)
_VP_DELETED = re.compile(
    r"^[^.]{0,80}?(?:\b(?:are|is|shall\s+be|were)\s+deleted\b|\bdoes\s+not\s+apply\b|\bnot\s+applicable\b|"
    r"\bremoved\b|\bexcluded\b|\bstruck\s+out\b)", re.I)
_VP_SUBJECT_TO = re.compile(
    r"^[^.]{0,40}?\bsubject\s+to\b[^.]{0,120}?\b(?:lease|underlease|tenanc(?:y|ies)|occupation|licence)", re.I)


def possession_statement(text: str, start: int, end: int) -> Optional[str]:
    """Classify one 'vacant possession' occurrence as a statement ABOUT THE SALE.

    Returns "vacant possession", "vacant possession subject to a lease/tenancy",
    or None when the words belong to a lease covenant (yield-up, return the
    property), a rent-review assumption, a deleted standard condition, or are
    not tied to the sale. Evidence (4 Oct 2026): the old pattern matched
    60B's underlease clause 14.1 (return the Property with vacant possession)
    and 59A's rent-review assumption, marking both let lots vacant."""
    before = text[max(0, start - 160):start]
    after = text[end:end + 200]
    near_before = before[-90:]
    if _VP_NOT_SALE.search(near_before):
        return None
    if _VP_DELETED.search(after):
        return None
    stated_now = text[start:end].lower().startswith("currently")   # "the property is currently vacant"
    if not (stated_now or _VP_SALE.search(near_before) or re.match(r"^\s*(?:on|upon)\s+completion", after, re.I)):
        return None
    if _VP_SUBJECT_TO.search(after):
        return "vacant possession subject to a lease/tenancy"
    return "vacant possession"
