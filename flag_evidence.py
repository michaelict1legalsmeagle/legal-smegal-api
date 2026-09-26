"""
flag_evidence.py — V-FLAGS (2026-09-26)

Every flag shown to a user must be evidenced by the pack itself.

  * Issue flags (critical / high / note): kept only if the evidence quote is
    located in the pack text — verbatim, elided ("a ... b"), or a contiguous run
    of at least PARTIAL_MIN_WORDS words (or the whole quote if shorter). Flags
    whose quote cannot be located are removed and kept in
    `flags_removed_unevidenced` for audit; they are never shown.
  * Missing flags: removed if a document of that kind is in the pack inventory
    (read OR unreadable). A missing flag whose kind cannot be identified is kept.
  * deal_score: computed tariff from the verified flags only
    (100 - 12*critical - 6*high - 4*missing - 1*note, floor 0). Never model-set.

No thresholds or weights here are invented: the tariff is the one already
stated in the analysis prompt since April; the matching rules are mechanical.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Tuple

SCORE_START = 100
SCORE_TARIFF = {"critical": 12, "high": 6, "missing": 4, "note": 1}
PARTIAL_MIN_WORDS = 8          # contiguous words that must match for a partial locate
ELIDE_MIN_WORDS = 3            # each elided fragment must have at least this many words

FLAG_RULES = """FLAG RULES — FOLLOW EXACTLY:
1. Flag only what the text in front of you shows. If there is no issue, return an empty flags array. There is no minimum and no target number of flags.
2. Every critical, high or note flag MUST quote the pack verbatim in "evidence" (max 30 words, copied exactly, no paraphrase). If you cannot quote it, do not raise it.
3. Do not raise positive or reassurance flags (e.g. "title verified", "no issues found").
4. Do not escalate severity beyond what the quoted text supports.
5. A "missing" flag is only for a document type that does NOT appear in the DOCUMENT INVENTORY supplied with the pack. Documents listed as unreadable ARE present — never flag them as missing. For missing flags set evidence to "Not in document inventory".
6. Never state a figure, date, name or period that is not in the quoted text."""

# ── text normalisation ───────────────────────────────────────────────────────
_QUOTES = {"\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
           "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2032": "'",
           "\u2013": "-", "\u2014": "-", "\u2012": "-", "\u2212": "-",
           "\u00a0": " "}


def normalise(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    for k, v in _QUOTES.items():
        t = t.replace(k, v)
    t = t.lower()
    t = re.sub(r"[^\w£%'\-\.\s]", " ", t)       # drop punctuation that OCR varies on
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _words(text: str) -> List[str]:
    return [w for w in normalise(text).replace(".", " ").split() if w]


def _contains_run(hay_words_joined: str, words: List[str], n: int) -> bool:
    if len(words) < n:
        return False
    for i in range(0, len(words) - n + 1):
        if " " + " ".join(words[i:i + n]) + " " in hay_words_joined:
            return True
    return False


def locate_quote(quote: str, pack_norm_words: str) -> str:
    """Return 'verbatim' | 'elided' | 'partial' | '' (not located).
    pack_norm_words: ' ' + ' '.join(_words(pack_text)) + ' '"""
    if not quote or not quote.strip():
        return ""
    qw = _words(quote)
    if not qw:
        return ""
    joined = " " + " ".join(qw) + " "
    if joined in pack_norm_words:
        return "verbatim"
    # elided quote: fragments separated by ... or …
    frags = [f for f in re.split(r"\.\.\.|\u2026|\[\s*\.\.\.\s*\]", quote) if f.strip()]
    if len(frags) > 1:
        pos = 0
        ok = True
        for f in frags:
            fw = _words(f)
            if len(fw) < ELIDE_MIN_WORDS:
                ok = False
                break
            j = pack_norm_words.find(" " + " ".join(fw) + " ", pos)
            if j < 0:
                ok = False
                break
            pos = j + 1
        if ok:
            return "elided"
    n = min(PARTIAL_MIN_WORDS, len(qw))
    if len(qw) >= PARTIAL_MIN_WORDS and _contains_run(pack_norm_words, qw, n):
        return "partial"
    return ""


# ── document kinds for "missing" flags ───────────────────────────────────────
# kind -> (doc_types that count, filename/heading patterns that count)
DOC_KINDS: Dict[str, Tuple[set, str]] = {
    "special_conditions": ({"special_conditions"}, r"special\s*conditions?"),
    "addendum":           ({"addendum"}, r"addend|amendment|day\s*of\s*sale\s*notice"),
    "title_register":     ({"title_register", "freehold"}, r"title\s*register|official\s*cop(y|ies)|register\s*of\s*title|\boc1\b"),
    "title_plan":         ({"title_plan"}, r"title\s*plan|filed\s*plan"),
    "local_search":       ({"local_auth_search"}, r"local\s*(authority\s*)?search|\bllc1\b|\bcon\s*29"),
    "environmental":      (set(), r"environment(al)?\s*(search|report|screen)|envirosearch|groundsure|landmark"),
    "drainage":           (set(), r"drainage|water\s*(and|&)\s*drainage|\bcon\s*29\s*dw\b|\bdw\s*search"),
    "epc":                ({"epc"}, r"\bepc\b|energy\s*performance"),
    "lease":              ({"lease"}, r"\blease\b"),
    "tenancy":            ({"tenancy_ast"}, r"tenancy|\bast\b|rent\s*statement"),
    "auction_tcs":        ({"auction_tcs"}, r"common\s*auction\s*conditions|general\s*conditions|auction\s*(terms|conditions)"),
    "ta6":                (set(), r"\bta\s*6\b|property\s*information\s*form"),
    "ta7":                (set(), r"\bta\s*7\b|leasehold\s*information\s*form"),
    "ta10":               (set(), r"\bta\s*10\b|fittings\s*(and|&)\s*contents"),
    "mining":             (set(), r"coal\s*(mining|authority)|mining\s*(search|report)"),
}

# how a missing flag's title/evidence names a kind (checked in this order —
# drainage before environmental so "drainage search" never maps to environmental)
MISSING_NAME_PATTERNS: List[Tuple[str, str]] = [
    ("drainage", r"drainage"),
    ("environmental", r"environment"),
    ("special_conditions", r"special\s*condition"),
    ("addendum", r"addend"),
    ("title_plan", r"title\s*plan|filed\s*plan"),
    ("title_register", r"title\s*register|register\s*of\s*title|official\s*cop|title\s*(doc|deed)s?\b|\btitle\b"),
    ("local_search", r"local\s*(authority\s*)?search|\bllc1\b|con\s*29"),
    ("epc", r"\bepc\b|energy\s*performance"),
    ("lease", r"\blease\b"),
    ("tenancy", r"tenancy|\bast\b"),
    ("auction_tcs", r"auction\s*(terms|conditions)|general\s*conditions"),
    ("ta6", r"\bta\s*6\b|property\s*information"),
    ("ta7", r"\bta\s*7\b|leasehold\s*information"),
    ("ta10", r"\bta\s*10\b|fittings"),
    ("mining", r"mining"),
]


def doc_kinds(doc: Dict) -> set:
    """Kinds a single document satisfies, from doc_type + filename + first 3000 chars."""
    dt = (doc.get("doc_type") or "").lower()
    probe = ((doc.get("file_name") or "") + "\n" + (doc.get("extracted_text") or "")[:3000]).lower()
    kinds = set()
    for kind, (types, pat) in DOC_KINDS.items():
        if dt in types or re.search(pat, probe, re.IGNORECASE):
            kinds.add(kind)
    return kinds


def inventory_kinds(documents: List[Dict]) -> set:
    out = set()
    for d in documents or []:
        out |= doc_kinds(d)
    return out


def missing_flag_kind(flag: Dict) -> str:
    probe = " ".join(str(flag.get(k) or "") for k in ("title", "summation")).lower()
    for kind, pat in MISSING_NAME_PATTERNS:
        if re.search(pat, probe, re.IGNORECASE):
            return kind
    return ""


# ── main entry points ────────────────────────────────────────────────────────
def verify_flags(flags: List[Dict], documents: List[Dict]) -> Dict:
    """Return {'flags': kept, 'removed': [...], 'stats': {...}}.
    documents: every document in the pack (read and unreadable), each with
    file_name, doc_type, extracted_text."""
    pack_text = "\n".join((d.get("extracted_text") or "") for d in (documents or []))
    pack_words = " " + " ".join(_words(pack_text)) + " "
    present = inventory_kinds(documents)

    kept, removed = [], []
    for f in flags or []:
        if not isinstance(f, dict):
            continue
        sev = (f.get("severity") or "").strip().lower()
        if sev not in SCORE_TARIFF:
            removed.append({**f, "_removed_reason": f"unknown severity '{sev}'"})
            continue
        f = {**f, "severity": sev}
        if sev == "missing":
            kind = missing_flag_kind(f)
            if kind and kind in present:
                removed.append({**f, "_removed_reason": f"document of kind '{kind}' is in the pack"})
                continue
            f["evidence_check"] = "missing:not_in_inventory" if kind else "missing:kind_unidentified"
            kept.append(f)
            continue
        how = locate_quote(f.get("evidence") or "", pack_words)
        if not how:
            removed.append({**f, "_removed_reason": "evidence quote not found in pack text"})
            continue
        f["evidence_check"] = how
        kept.append(f)

    stats = {
        "input": len(flags or []),
        "kept": len(kept),
        "removed": len(removed),
        "removed_unlocated_quote": sum(1 for r in removed if "not found" in r.get("_removed_reason", "")),
        "removed_missing_present": sum(1 for r in removed if "is in the pack" in r.get("_removed_reason", "")),
    }
    return {"flags": kept, "removed": removed, "stats": stats}


def flag_counts(flags: List[Dict]) -> Dict[str, int]:
    c = {k: 0 for k in SCORE_TARIFF}
    for f in flags or []:
        s = (f.get("severity") or "").lower()
        if s in c:
            c[s] += 1
    return c


def compute_deal_score(flags: List[Dict]) -> int:
    c = flag_counts(flags)
    score = SCORE_START - sum(SCORE_TARIFF[k] * c[k] for k in SCORE_TARIFF)
    return max(0, score)


def dedupe_flags(flags: List[Dict]) -> List[Dict]:
    """Merge exact duplicates raised by overlapping sections: same evidence
    (first 12 normalised words) AND same title start (first 3 words); missing
    flags merge by document kind. Different issues quoting the same clause are
    kept (losing a genuine flag is worse than showing a near-duplicate)."""
    seen, out = set(), []
    for f in flags or []:
        if not isinstance(f, dict):
            continue
        sev = (f.get("severity") or "").lower()
        if sev == "missing":
            key = ("missing", missing_flag_kind(f) or normalise(f.get("title") or ""))
        else:
            ew = _words(f.get("evidence") or "")[:12]
            tw = _words(f.get("title") or "")[:3]
            key = ("ev", " ".join(ew), " ".join(tw))
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out
