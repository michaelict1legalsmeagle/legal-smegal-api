"""
doc_classifier.py — deterministic legal-pack document typing.

Replaces app.py's DOCUMENT_PATTERNS / first-substring-match classifier.

WHY (D-CLASSIFIER, 2026-09-23 — traced on live deal 3ca6f024…, 15 Wolsingham
Terrace, Lot 12):
  The old classifier ran ONE pass over (filename + first 3000 chars of text),
  returned the FIRST type in dict order with ANY raw substring hit, and had very
  broad title_register triggers ("land registry", "official copy") checked early.
  Live result on that pack: local search, drainage search, probate, TA6 and the
  OC1 *title plan* all typed 'title_register'; the death certificate typed
  'title_plan' ("administrative area"). 436 of 1,071 live documents were
  'title_register'.
  Downstream, /summarise gives 'title_register' a 12k legal budget and high
  priority under a 40k hard cap — the mistyped bulky searches filled the cap,
  the EPC (typed correctly) was never sent to the LLM, and the report said
  "No EPC in Pack" while an EPC was in the pack.

HOW:
  1. Filename first — auction/solicitor filenames are the highest-signal input.
  2. Text heading zone (first 800 normalised chars) — document titles live here.
  3. Text body zone (first 3000 chars).
  Within each stage, types are checked in a deliberate specificity order.
  Every pattern is WORD-BOUNDED (no "please" -> lease, "released" -> lease).
  Whitespace is collapsed so "Special Conditions\nof Sale" still matches.
  No hit -> 'unknown' (honest; never guessed).

Adds two types the old taxonomy lacked, both common in executor sales:
  'probate', 'death_certificate'. Unlisted types sort last in the existing
  prompt-priority lists and are not in the pack-completeness checklist, so
  completeness scoring is unchanged for them.

Pure module: no Flask, no DB, no I/O. PATTERN TABLES ARE THE SINGLE SOURCE —
to_postgres_case() renders the same rules as SQL for live-data audits.
"""
import re
from typing import List, Tuple

HEAD_CHARS = 800
BODY_CHARS = 3000

# ── Stage 1: filename rules (ordered: specific -> broad) ─────────────────────
FILENAME_RULES: List[Tuple[str, List[str]]] = [
    ("special_conditions", ["special condition", "special conditions", "special cond",
                            "sale agreement", "auction contract", "contract for sale",
                            "contract of sale", "memorandum of sale"]),
    ("addendum",           ["addendum", "amendment", "amendments"]),
    ("probate",            ["probate", "grant of representation", "letters of administration"]),
    ("death_certificate",  ["death cert", "death certificate"]),
    ("title_plan",         ["title plan", "filed plan"]),            # before register: "OC1 Title Plan"
    ("lease",              ["lease", "underlease"]),                 # before register: "Official copy of lease"
    # bare "register" deliberately absent: live packs carry "REGISTER_TO_BID" and "EPC_Register"
    ("title_register",     ["title register", "register of title", "oc1", "official copy", "official copies"]),
    ("epc",                ["epc", "energy performance"]),
    ("local_auth_search",  ["local search", "local authority", "con29", "con29r", "llc1"]),
    ("environmental",      ["environmental", "drainage", "water search", "groundsure", "flood",
                            "con29dw", "con29m", "dws", "coal", "mining"]),
    ("tenancy_ast",        ["tenancy", "ast"]),
    ("survey",             ["survey", "homebuyer"]),
    ("auction_tcs",        ["auction terms", "auctioneer terms", "bidder terms", "bidding terms"]),
    ("deed",               ["tr1", "tr2", "tp1", "ta6", "ta7", "ta10", "pif", "deed", "transfer",
                            "transfer deed", "conveyance", "assent", "epitome", "abstract of title",
                            "property information", "fittings and contents"]),
    ("legal_pack",         ["legal pack", "auction pack"]),
]

# ── Stages 2+3: text rules (ordered: specific -> broad; legal_pack MUST stay last) ──
TEXT_RULES: List[Tuple[str, List[str]]] = [
    ("special_conditions", ["special conditions", "special condition of sale", "conditions of sale",
                            "common auction conditions", "auction sale agreement",
                            "the seller will sell and the buyer will buy"]),
    ("addendum",           ["addendum", "day of sale", "lot amendment", "amendment notice",
                            "late amendment", "revised conditions", "updated conditions",
                            "pre auction notice", "vendor notice"]),
    ("probate",            ["grant of probate", "letters of administration",
                            "grant of representation", "hmcts probate", "probate registry"]),
    ("death_certificate",  ["deaths registration act", "entry of death",
                            "certified copy of an entry of death"]),
    # Water-company names ("thames water", "severn trent", "anglian water") deliberately
    # NOT used: now that environmental precedes title_register, an easement to a water
    # company in a register would be mistyped. Water searches are caught by their own
    # headings ("drainage and water search") and filenames (CON29DW, DWS, drainage).
    # searches BEFORE title_register: a local search reads "Land Charges Register"
    ("local_auth_search",  ["local authority search", "regulated local authority",
                            "local search enquiries", "search of local land charges",
                            "local land charges", "enquiries of the local authority",
                            "con29", "llc1", "city council", "district council", "borough council"]),
    ("environmental",      ["groundsure", "homebuyer environmental", "environmental search",
                            "drainage and water search", "water and drainage search",
                            "drainage and water enquiry", "con29dw", "coal mining", "coal authority",
                            "mining report",
                            "drainage search", "water search", "regulated drainage",
                            "combined drainage", "utilities search", "flood risk",
                            "ground risk", "chancel"]),
    # plan BEFORE register; "administrative area" (death certs) and "title number"
    # (every register) removed — both caused false title_plan hits.
    ("title_plan",         ["title plan", "filed plan", "ordnance survey map reference",
                            "ordnance survey national grid reference"]),
    # "land registry" / "hm land registry" / "official copy" removed — they appear on
    # plans, searches (Crown-copyright notices) and probate grants.
    ("title_register",     ["official copy of the register", "official copy of register",
                            "property register", "proprietorship register", "charges register",
                            "register of title", "title absolute"]),
    ("epc",                ["energy performance certificate", "energy performance", "epc",
                            "energy certificate", "domestic energy", "energy rating"]),
    ("lease",              ["lease", "underlease", "sublease", "leasehold land",
                            "lease dated", "term of years"]),
    ("survey",             ["structural survey", "building survey", "rics survey",
                            "condition report", "level 2", "level 3", "homebuyer report"]),
    ("auction_tcs",        ["auction terms", "auctioneer terms", "conditions of auction"]),
    ("deed",               ["transfer deed", "conveyance", "tr1", "deed of", "ta6", "ta10",
                            "transfer of whole", "transfer of part", "transferor",
                            "seller property", "fittings and contents", "property information form"]),
    ("tenancy_ast",        ["assured shorthold", "tenancy agreement", "rental agreement"]),
    ("freehold",           ["freehold", "absolute freehold", "possessory freehold"]),
    ("legal_pack",         ["legal pack", "auction pack", "lot information", "information pack",
                            "document archive", "pack archive"]),
]


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[_\-.]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _rx(patterns: List[str]) -> "re.Pattern":
    return re.compile(r"\b(?:" + "|".join(re.escape(p) for p in patterns) + r")\b")


_FN_COMPILED   = [(t, _rx(p)) for t, p in FILENAME_RULES]
_TEXT_COMPILED = [(t, _rx(p)) for t, p in TEXT_RULES]


def classify_document(filename: str, text: str) -> str:
    """Return a doc_type string. Never raises; 'unknown' when nothing matches."""
    try:
        fn = _norm(filename)
        for doc_type, rx in _FN_COMPILED:
            if rx.search(fn):
                return doc_type
        body = _norm((text or "")[:BODY_CHARS])
        if body:
            head = body[:HEAD_CHARS]
            for zone in (head, body):
                for doc_type, rx in _TEXT_COMPILED:
                    if rx.search(zone):
                        return doc_type
    except Exception:
        pass
    return "unknown"


# ── Audit helper: same rules as a Postgres CASE (for read-only live cross-tabs) ──
def _pg_rx(patterns: List[str]) -> str:
    alts = "|".join(re.sub(r"([.^$*+?()\[\]{}|\\])", r"\\\1", p) for p in patterns)
    return "\\y(" + alts + ")\\y"


def to_postgres_case(fn_expr: str, head_expr: str, body_expr: str) -> str:
    """Render classify_document as a SQL CASE over pre-normalised expressions."""
    lines = ["CASE"]
    for t, p in FILENAME_RULES:
        lines.append(f"  WHEN {fn_expr} ~ '{_pg_rx(p)}' THEN '{t}'")
    for zone in (head_expr, body_expr):
        for t, p in TEXT_RULES:
            lines.append(f"  WHEN {zone} ~ '{_pg_rx(p)}' THEN '{t}'")
    lines.append("  ELSE 'unknown' END")
    return "\n".join(lines)
