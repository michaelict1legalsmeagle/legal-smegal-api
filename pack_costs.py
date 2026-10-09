"""pack_costs.py — PACK-COSTS-1 (9 Oct 2026): buyer costs STATED in the legal pack,
found deterministically and quoted verbatim.

Why (verified on live data 9 Oct 2026): the model-based reader left stated buyer costs
out of the analysis. Examples from real packs: "the Buyer is liable to pay the Seller
£5,995 ... by way of a buyer's premium" (6 deals, not extracted); "a contribution of
the sum of One thousand pounds (£1,000) in respect of their search fees" (6 deals);
HU9 3AQ clause 36 "The Buyer shall pay the Seller's Solicitors fees of £121.00 plus VAT"
(recorded as no seller's legal costs). A clause that charges the buyer must not depend
on the model noticing it, so this module reads the clauses directly (same principle as
the overage backstop).

Rules — evidence only, nothing inferred:
  * a cost item is a sentence that names the BUYER/PURCHASER, a cost word (premium,
    administration fee/charge, legal costs/fees, search fees, disbursements,
    contribution, reimburse) and an amount (£ figure, written pounds, or a % of the
    price). Each item carries the verbatim sentence, its clause number and document.
  * amounts are reported exactly as stated (with "plus VAT" / "inc VAT" as stated);
    nothing is added up, converted or estimated.
  * a cost that only applies if something happens (notice to complete, enquiries
    answered at an hourly rate, change of buyer name, late completion) is marked
    conditional, never mixed with what every buyer pays.
  * a clause that says search fees are reimbursed without an amount is kept with
    amount null ("amount not stated in the pack").
  * deposit terms are reported separately, as stated.
Pure module: no network, no DB. Unit-tested on real clause text (tests/test_pack_costs.py).
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

VERSION = "pack-costs-1"
COST_DOC_TYPES = {"special_conditions", "addendum", "legal_pack", "auction_tcs", "unknown"}
MAX_SENTENCE = 700

_MONEY = re.compile(r"£\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?")
_PCT = re.compile(r"(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s?(?:%|per\s?cent\b|percent\b)", re.I)
_PCT_DIGITS = re.compile(r"(\d+(?:\.\d+)?)\s?(?:%|per\s?cent\b|percent\b)", re.I)
_SMALL = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _pct_value(tok: str) -> float:
    t = tok.lower()
    return float(_SMALL[t]) if t in _SMALL else float(t)
_BUYER = re.compile(r"\b(buyer|buyers|purchaser|purchasers)\b|buyer[’']s|purchaser[’']s", re.I)
_PLUS_VAT = re.compile(r"(plus|\+|and)\s*v\.?a\.?t\.?\b|together with v\.?a\.?t|with value added tax|plus value added tax", re.I)
_INC_VAT = re.compile(r"\binc(?:l(?:\.|usive|uding)?)?\.?\s*(?:of\s+)?v\.?a\.?t\b", re.I)
_MIN = re.compile(r"minimum(?:\s+(?:fee|sum|amount|charge))?\s+of\s+£\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?", re.I)
_BLANK = re.compile(r"£\s?_{2,}|£\s*\.{3,}")
_COND = re.compile(
    r"^\s*(?:\(?[a-z0-9]{1,3}[.)]\s*)?(?:if|in the event|should|where|in case)\b"
    r"|notice to complete|per hour|hourly|requisitions? (?:raised|after)|enquir(?:y|ies) (?:are )?raised"
    r"|change (?:the )?(?:name|buyer)|nominee|late completion|delay in completion|default",
    re.I)
_EXCLUDE = re.compile(
    r"per claim|ombudsman|indemnity (?:policy|insurance)|insurance premium|remediation contribution"
    r"|eligible for|professional indemnity|ground rent|service charge|council tax|rent of", re.I)
CATEGORIES = [
    ("buyers_premium", re.compile(r"buyer[’'`]?s?\s+premium|buyers?[’']?\s+fee\b", re.I)),
    ("admin_fee", re.compile(r"administration\s+(?:fee|charge)|admin\.?\s+fee|signing fee", re.I)),
    ("auctioneer_payment", re.compile(r"(?:pay|payable)\s+(?:to\s+)?the\s+auctioneers?\b(?!\s+as\s+agents?)", re.I)),
    ("search_fees", re.compile(r"search(?:es)?\s+(?:fees?|costs?)|disbursements?|cost of the (?:local )?search|in respect of (?:the )?searches", re.I)),
    ("seller_legal_costs", re.compile(
        r"legal\s+(?:and\s+agents?\s+)?(?:costs|fees)|solicitor[’']?s?[’']?\s+(?:legal\s+)?(?:costs|fees|charges)|seller[’']?s?\s+(?:auction\s+)?(?:costs|fees)"
        r"|auction costs|auctioneers? (?:and legal )?costs|engrossment fee", re.I)),
    ("other", re.compile(r"\bcontribution\b|\breimburse", re.I)),
]
LABEL = {
    "buyers_premium": "Buyer's premium / buyer's fee",
    "admin_fee": "Administration fee",
    "search_fees": "Search fees / disbursements",
    "seller_legal_costs": "Seller's legal costs",
    "auctioneer_payment": "Payment to the auctioneers",
    "other": "Other stated payment",
}

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_WRITTEN = re.compile(
    r"\b((?:(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|"
    r"ninety|hundred|thousand|and)[\s-]+)+)pounds\b", re.I)


def _words_to_int(s: str) -> Optional[int]:
    total, cur, seen = 0, 0, False
    for w in re.split(r"[\s-]+", s.lower().strip()):
        if not w or w == "and":
            continue
        if w in _UNITS:
            cur += _UNITS[w]; seen = True
        elif w in _TENS:
            cur += _TENS[w]; seen = True
        elif w == "hundred":
            cur = (cur or 1) * 100; seen = True
        elif w == "thousand":
            total += (cur or 1) * 1000; cur = 0; seen = True
        else:
            return None
    return (total + cur) if seen else None


def _money(m: re.Match) -> float:
    whole = int(m.group(1).replace(",", ""))
    pence = m.group(2)
    return round(whole + (int(pence.ljust(2, "0")) / 100.0 if pence else 0.0), 2)


# ── clause segmentation ─────────────────────────────────────────────────────
_CLAUSE_START = re.compile(r"(?m)^\s*(\(?\d{1,3}(?:\.\d{1,2})?[a-z]?[.)]?|\([a-z]{1,3}\)|[A-Z]\d{1,2}(?:\.\d{1,2})*)\s+(?=\S)")
_PAGE_MARK = re.compile(r"=== PAGE \d+ ===")
_SENT_SPLIT = re.compile(r"(?<=[.;:])\s+(?=[A-Z(\d])|\n[ \t]*\n(?=\s*(?:\(?\d{1,3}[a-z]?[.)]|\([a-z]{1,3}\)))|\n(?=\s*(?:\(?\d{1,3}[a-z]?[.)]|\([a-z]{1,3}\))\s)")


def _clauses_with_numbers(text: str):
    """Yield (clause_no, sentence) pairs. Clause numbers come from the nearest
    numbered line at or before the sentence."""
    # page markers sit inside sentences when a clause runs over a page: remove them
    # (same length, so positions are unchanged) rather than splitting there.
    text = _PAGE_MARK.sub(lambda m: " " * len(m.group(0)), text)
    starts = [(m.start(), m.group(1).strip("().")) for m in _CLAUSE_START.finditer(text)]
    pos = 0
    for piece in _SENT_SPLIT.split(text):
        if piece is None:
            continue
        idx = text.find(piece, pos)
        if idx < 0:
            idx = pos
        pos = idx + len(piece)
        sent = re.sub(r"\s+", " ", piece).strip()
        if not sent:
            continue
        num = None
        for s, n in starts:
            if s <= idx + 3:
                num = n
            else:
                break
        yield num, sent


def _vat_basis(s: str) -> Optional[str]:
    if _INC_VAT.search(s):
        return "inc VAT"
    if _PLUS_VAT.search(s):
        return "plus VAT"
    return None


def _amount_in(seg: str):
    """First amount in a text segment: (gbp or None, pct or None, minimum or None)."""
    m = _MONEY.search(seg)
    w = _WRITTEN.search(seg)
    gbp = None
    if m and (not w or m.start() <= w.start() + 200):
        gbp = _money(m)
    elif w:
        v = _words_to_int(w.group(1))
        gbp = float(v) if v else None
    pct = None
    pm = _PCT_DIGITS.search(seg) or _PCT.search(seg)
    if pm and (re.search(r"(?:of|on)\s+the\s+(?:final\s+|agreed\s+|contractual\s+)?(?:purchase\s+|contract\s+)?(?:price|hammer)", seg, re.I)
               or re.search(r"rate of\s*$", seg[:pm.start()], re.I)):
        pct = _pct_value(pm.group(1))
    mn = _MIN.search(seg)
    minimum = _money(mn) if mn else None
    if minimum is None:
        wm = re.search(r"minimum(?:\s+(?:fee|sum|amount|charge))?\s+of\s+" + _WRITTEN.pattern[2:], seg, re.I)
        if wm:
            v = _words_to_int(wm.group(1))
            minimum = float(v) if v else None
    if minimum is not None and gbp == minimum:
        gbp = None      # the only figure is the minimum of a percentage fee
    return gbp, pct, minimum


def _items_from_sentence(sent: str) -> List[Dict[str, Any]]:
    """Every stated buyer payment in one sentence. A sentence such as "an administration
    fee of 2.75% ... plus VAT, a contribution to the Seller's legal fees of one thousand
    one hundred and eighty pounds plus VAT and an engrossment fee ... of four hundred and
    fifty pounds" yields three items; each amount is read from the text that follows its
    own cost words (or, failing that, the text just before them)."""
    if len(sent) > MAX_SENTENCE or _BLANK.search(sent) or _EXCLUDE.search(sent):
        return []
    if not _BUYER.search(sent):
        return []
    cues = []
    for name, rx in CATEGORIES:
        for m in rx.finditer(sent):
            cues.append((m.start(), m.end(), name))
    if not cues:
        return []
    cues.sort()
    # drop a cue nested inside / immediately after a stronger one (e.g. "contribution
    # to the Seller's legal fees" -> one legal-costs item, not "other" + "legal")
    # A "contribution"/"reimburse" cue followed (within 120 chars) by specific cost words
    # becomes ONE item of that specific kind, its amount read from the text after the
    # contribution/reimburse words ("a contribution of £1,000 in respect of their search
    # fees, and legal fees" -> one search-fees item of £1,000).
    merged = []        # (start, amount_from, name)
    i = 0
    while i < len(cues):
        a, b, name = cues[i]
        if name == "other":
            j = i + 1
            spec = None
            while j < len(cues) and cues[j][0] - b < 120:
                if cues[j][2] != "other":
                    spec = cues[j][2]
                    j += 1
                    break
                j += 1
            if spec:
                merged.append((a, b, cues[j - 1][1], spec))
                i = j
                continue
        if merged and a - merged[-1][2] < 6 and name == merged[-1][3]:
            i += 1
            continue
        merged.append((a, b, b, name))
        i += 1
    out = []
    cond = bool(_COND.search(sent))
    vat_sentence = _vat_basis(sent)
    used_until = 0     # an amount already given to one item is never given to another
    for i, (a, b, span_end, name) in enumerate(merged):
        nxt = merged[i + 1][0] if i + 1 < len(merged) else len(sent)
        after = sent[b:nxt]
        gbp, pct, minimum = _amount_in(after)
        seg = after
        if gbp is not None or pct is not None:
            used_until = nxt
        else:
            prev_end = merged[i - 1][2] if i > 0 else 0
            # "... search fees, and legal fees" -> the same payment covers both
            if out and i > 0 and re.fullmatch(r"[\s,]*(?:and|&)?[\s,]*", sent[prev_end:a] or ""):
                out[-1]["label"] = out[-1]["label"] + " and " + LABEL[name].lower()
                continue
            before = sent[max(prev_end, used_until, a - 160):a]
            gbp, pct, minimum = _amount_in(before)
            seg = before
            if gbp is not None or pct is not None:
                used_until = a
        if gbp is None and pct is None and out and out[-1]["category"] == "other":
            out[-1]["category"], out[-1]["label"] = name, LABEL[name]   # "... £500 in respect of searches"
            continue
        if gbp is None and pct is None and out and a - prev_end < 80:
            continue    # more words describing the payment just recorded
        reimburse_search = (name == "search_fees" and re.search(r"reimburse|contribution|\bpay\b", sent, re.I)
                            and not re.search(r"may (?:become )?(?:be )?payable|if applicable|will be detailed", sent, re.I))
        if gbp is None and pct is None and not reimburse_search:
            continue
        out.append({
            "category": name, "label": LABEL[name],
            "amount_gbp": gbp, "pct_of_price": pct, "minimum_gbp": minimum,
            "vat": _vat_basis(seg) or (vat_sentence if len(merged) == 1 else None),
            "conditional": cond, "quote": sent,
        })
    return out


def _is_cost_doc(d: Dict[str, Any]) -> bool:
    fn = (d.get("file_name") or "").lower()
    return (d.get("doc_type") or "unknown") in COST_DOC_TYPES or "special" in fn or "addendum" in fn or "condition" in fn


def find_costs(documents: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """documents: [{file_name, doc_type, extracted_text}] -> stated costs block."""
    items: List[Dict[str, Any]] = []
    deposits: List[Dict[str, Any]] = []
    seen = set()
    for d in documents or []:
        txt = d.get("extracted_text") or ""
        if not txt.strip() or not _is_cost_doc(d):
            continue
        name = d.get("file_name") or "(unnamed)"
        for num, sent in _clauses_with_numbers(txt):
            for it in _items_from_sentence(sent):
                key = (it["category"], it["amount_gbp"], it["pct_of_price"], it["conditional"])
                if key not in seen:
                    seen.add(key)
                    it.update({"clause": num, "document": name})
                    items.append(it)
            dep = _deposit_from_sentence(sent)
            if dep:
                k = ("deposit", dep["pct_of_price"], dep["minimum_gbp"], dep["amount_gbp"])
                if k not in seen:
                    seen.add(k)
                    dep.update({"clause": num, "document": name})
                    deposits.append(dep)
    return {"version": VERSION, "items": items, "deposit_terms": deposits}


_DEP = re.compile(r"\bdeposit\b", re.I)
_DEP_GENERIC = re.compile(r"if the buyer paid (?:no|a) deposit|deposit of less than|balance of that|further deposit", re.I)


def _deposit_from_sentence(sent: str) -> Optional[Dict[str, Any]]:
    if len(sent) > MAX_SENTENCE or not _DEP.search(sent) or _DEP_GENERIC.search(sent) or _BLANK.search(sent):
        return None
    if not re.search(r"\b(pay|paid|payable|accept|shall be|is to be|required)\b", sent, re.I):
        return None
    # the figure must belong to the deposit: "deposit of 10%", "a 5% deposit",
    # "minimum deposit of £5,000", "deposit payable is £8,000.00 or 10%"
    pm = (re.search(r"deposit[^.;£%]{0,40}?(\d+(?:\.\d+)?)\s?(?:%|per\s?cent\b|percent\b)", sent, re.I)
          or re.search(r"(\d+(?:\.\d+)?)\s?(?:%|per\s?cent\b|percent\b)\s+deposit", sent, re.I))
    mn = (re.search(r"minimum (?:deposit\s+)?(?:of\s+|we accept is\s+|is\s+)?£\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?", sent, re.I)
          if re.search(r"minimum", sent, re.I) else None)
    fixed = None
    if not pm and not mn:
        m = re.search(r"deposit[^.;£]{0,30}£\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?", sent, re.I)
        fixed = _money(m) if m else None
    if not pm and not mn and fixed is None:
        return None
    return {"pct_of_price": float(pm.group(1)) if pm else None,
            "minimum_gbp": _money(mn) if mn else None,
            "amount_gbp": fixed, "quote": sent}


def backfill_fields(sc: Dict[str, Any], ct: Dict[str, Any], costs: Dict[str, Any]) -> List[str]:
    """Fill model fields that are EMPTY from unambiguous stated items (never overwrite).
    Returns the field names filled."""
    filled: List[str] = []
    items = [i for i in (costs or {}).get("items") or [] if not i.get("conditional")]

    def only(cat, key):
        vals = {i[key] for i in items if i["category"] == cat and i.get(key) is not None}
        return vals.pop() if len(vals) == 1 else None

    def fill(d, k, v):
        if v is not None and d.get(k) in (None, "", 0):
            d[k] = v
            filled.append(k)

    fill(sc, "buyers_premium_gbp", only("buyers_premium", "amount_gbp"))
    fill(sc, "buyers_premium_pct", only("buyers_premium", "pct_of_price"))
    fill(sc, "admin_fee_gbp", only("admin_fee", "amount_gbp"))
    fill(sc, "seller_legal_costs_gbp", only("seller_legal_costs", "amount_gbp"))
    if not sc.get("search_fee_reimbursement") and any(i["category"] == "search_fees" for i in items):
        sc["search_fee_reimbursement"] = True
        filled.append("search_fee_reimbursement")
    deps = (costs or {}).get("deposit_terms") or []
    pcts = {d["pct_of_price"] for d in deps if d.get("pct_of_price") is not None}
    if len(pcts) == 1:
        fill(ct, "deposit_pct", pcts.pop())
    return filled
