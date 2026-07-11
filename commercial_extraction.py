"""
commercial_extraction.py — P1: legal-pack extraction into commercial fields.

WHAT THIS DOES
Reads a deal's uploaded legal-pack documents (as already stored in Supabase
`documents`: doc_type, file_name, extracted_text, page_count) and produces
a subset of the fields commercial_routes.py accepts
(_ALLOWED_COMMERCIAL_FIELDS in that file), each tagged with provenance:
  {field: {"source": "extracted", "citation": "<file, page N>", "at": iso}}
matching the v2.4 provenance contract in commercial_valuation_engine.py's
_evidence_tier(). This module does NOT write to the database and does NOT
call calculate_commercial_ceiling — it only produces the fields+provenance
dict; the calling route is responsible for merging and persisting, exactly
as commercial_routes.py's save_commercial_inputs() already does for
user-entered fields (see the new /extract route added to that file).

WHY TWO EXTRACTION LAYERS (deterministic regex, then LLM)
Tested against three real commercial legal packs during this build (13 &
9 Harborne Park Road, Jubilee House Codsall) — see build notes below.
Deterministic regex extraction handles the fields that appear in a
near-fixed sentence template on the Rent Statement / Special Conditions
document (tenure, passing rent, review basis, vacant possession) and was
verified correct on all three real packs, including both trap cases
(Lot 2's confusable leasehold-register-vs-freehold-sale, Lot 6's vacant
property). It never guesses: if the pattern isn't found, the field is
simply absent — no LLM inference involved for these fields, so there is
nothing for a model to hallucinate.

The LLM layer only runs for facts that don't follow a fixed template and
therefore can't be regexed reliably: tenant name, lease start date, lease
term length. Even here, the LLM's own arithmetic is NEVER trusted for
anything date-sensitive — see compute_unexpired_term_years() below, which
recomputes and validates server-side against today's date before anything
gets written.

DOCTRINE RULES BAKED INTO THIS MODULE (all found by testing real packs,
not theoretical):

  1. TENURE SOURCE RULE — tenure must come from the seller's own "sold
     freehold subject to..." / "Title: Freehold Title..." wording (Rent
     Statement, Special Conditions, Contract), NEVER from the mere
     presence of a registered leasehold title in the pack. A tenant's own
     leasehold registration (an "LH register" document) is an encumbrance
     on the freehold being sold, not the tenure of the sale itself. Tested
     against Lot 2 (9 Harborne Park Road): the pack contains a registered
     leasehold title (WM817058) for the occupying tenant, but the property
     itself is being sold freehold — extracting "leasehold" from the
     presence of that register would be wrong.

  2. NO-FABRICATION-ON-VACANCY RULE — if a pack states vacant possession
     and contains no rent statement, passing_rent_pa is left ABSENT, never
     written as 0 or guessed from a market assumption. Tested against
     Lot 6 (Jubilee House): a genuinely vacant former police station with
     no tenancy document in the pack at all.

  3. EXPIRED-TERM RULE — a lease's stated term (e.g. "10 years from
     25.12.03") is a historical fact about when the lease was GRANTED, not
     evidence of a currently unexpired term. compute_unexpired_term_years()
     computes the actual end date and returns None + an evidence_gap if
     that date has already passed, rather than ever writing a stale
     unexpired_term_years / wault_years. Tested against Lot 2: a 10-year
     lease from 25 Dec 2003 expired 25 Dec 2013 — the tenant is presumably
     holding over on a statutory/periodic basis, which the deal needs a
     human to look at, not a confident number.

  4. PAGE-CITATION RULE — every accepted field must cite an exact document
     + page number, read from the "=== PAGE N ===" markers that
     extraction_service_hetzner.py / docai_ocr.py now embed in
     extracted_text (S-COMM-P1 fix, same session). If extracted_text
     predates that fix and has no page markers, citations degrade to
     "<file>, page unknown" rather than a fabricated page number.

  5. CROSS-DOCUMENT CONFLICT RULE (S-COMM-P1-CONFLICT, added same session,
     proven live on a real deal) — a legal pack is not one document, it's
     several, and they don't always agree. Once the OCR page-boundary fix
     let the Tenancy Agreement's own scanned cover page actually be read
     for 13 Harborne Park Road, it stated a materially different passing
     rent from the pack's separate Rent Statement for the same property.
     extract_deterministic() now scans every document for every field
     before deciding anything, and if two documents disagree on the same
     fact, NEITHER value is written — the field is left out and both
     values (with citations) go into evidence_gaps instead. The same
     applies to a pack that states vacant possession in one document while
     another states a passing rent: that contradiction is surfaced rather
     than quietly picking the rent figure. Resolving which of two real
     documents is correct is a human decision, not an automated one.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
  - Does not extract yield, market_rent_pa, or any figure requiring market
    judgement — legal packs don't contain those; they must stay
    user-entered. Evidence tier correctly stays C regardless (see
    _evidence_tier() in commercial_valuation_engine.py).
  - Does not handle multi-tenancy WAULT weighting. All three real packs
    tested were single-tenancy; wault_years is only populated when exactly
    one clean lease was identified, using the same figure as
    unexpired_term_years. A multi-let asset needs a person to weight it.
  - Does not touch doc_type-based document prioritisation. Tested finding:
    DOCUMENT_PATTERNS (app.py) has no entry for "Rent Statement",
    "Notice of Assignment", or "Certificate of Title" — the single most
    important document in every pack tested classified as doc_type
    "unknown" or an unrelated bucket. Relying on doc_type to prioritise
    which documents to send would have SKIPPED the Rent Statement in every
    test pack. This module sends every document with extracted_text,
    labelled by filename only.

STATUS: the deterministic layer (extract_deterministic, below) and the
server-side term guard (compute_unexpired_term_years) were run against
three real legal packs during this build and produced correct output
including both trap cases, cited to the correct page in every case. The
conflict-detection logic (rule 5) was added after a live deploy surfaced a
genuine rent conflict on 13 Harborne Park Road, and was verified with two
targeted tests: a synthetic second rent figure appended to the real Lot 4
pack (correctly withheld passing_rent_pa, correctly cited both values),
and a synthetic rent statement appended to the real vacant Lot 6 pack
(correctly withheld passing_rent_pa, correctly flagged the vacant/let
contradiction). The LLM layer (extract_via_llm) has since been run once
live, against a real deal (73b21ec9-7064-432b-a772-7e3653bb1a01), and its
own date-arithmetic guard fired correctly on that run — but its new
conflicts-reporting addition (rule 6 in the system prompt) has not yet
been proven live and should be treated as unverified until it has.
"""

import re
from datetime import date, datetime, timezone

# Kept in sync manually with commercial_routes.py's _ALLOWED_COMMERCIAL_FIELDS.
# Only the subset a legal pack can actually contain — never yield/market
# figures, which require judgement a document can't supply.
EXTRACTABLE_FIELDS = {
    "tenure", "passing_rent_pa", "rent_review_basis", "tenant_name",
    "unexpired_term_years", "wault_years",
}

_WHITESPACE_RE = re.compile(r"\s+")
_PAGE_MARKER_RE = re.compile(r"=== PAGE (\S+) ===")

_TENURE_PATTERNS = [
    (r"sold\s+freehold", "Freehold"),
    (r"being\s+sold\s+freehold", "Freehold"),
    (r"sold\s+leasehold", "Leasehold"),
    (r"title\s*\n?\s*freehold\s+title", "Freehold"),
    (r"title\s*\n?\s*leasehold\s+title", "Leasehold"),
]
_RENT_PATTERN = re.compile(
    r"rent\s*(?:of|:)\s*£\s?([\d,]+(?:\.\d{2})?)\s+per\s+annum", re.IGNORECASE
)
_REVIEW_PATTERN = re.compile(r"\(([^()]*review[^()]*)\)", re.IGNORECASE)
_VACANT_PATTERN = re.compile(r"vacant\s+possession", re.IGNORECASE)


def _find_page(text_with_markers: str, match_start: int):
    """Nearest preceding '=== PAGE N ===' marker before match_start, or
    None if the text has no markers (older extraction, pre S-COMM-P1)."""
    page_no = None
    for m in _PAGE_MARKER_RE.finditer(text_with_markers):
        if m.start() <= match_start:
            page_no = m.group(1)
        else:
            break
    return page_no


def _citation(file_name: str, page) -> str:
    return f"{file_name}, page {page}" if page is not None else f"{file_name}, page unknown"


def extract_deterministic(documents: list) -> tuple:
    """Regex-based extraction — see module docstring for what's covered
    and why. `documents` is a list of {"file_name": str, "text": str}
    (text should carry '=== PAGE N ===' markers from the fixed extraction
    service, but degrades gracefully if it doesn't).

    S-COMM-P1-CONFLICT (2026-07-11): scans EVERY document for every field,
    rather than stopping at the first match. Found by testing Lot 4 (13
    Harborne Park Road) once the Tenancy document's scanned pages could
    actually be OCR'd: the Rent Statement states one passing rent, and the
    Tenancy Agreement's own cover page — once genuinely readable — states
    a materially different figure for the same property. First-match-wins
    would have silently picked whichever document happened to be iterated
    first (an accident of list order, not a judgement about which document
    is more reliable) and never surfaced that the pack disagrees with
    itself. That is exactly the kind of silent wrong-number risk the
    no-fabrication doctrine exists to prevent — a confidently-written
    figure is worse than an absent one if it's the wrong one of two real
    candidates.

    Returns (fields: dict, citations: dict, conflicts: list[str]).
    A field is only written when every document that states it agrees on
    the same value. When two or more documents disagree, the field is
    left OUT of `fields` entirely and a description of the conflict
    (both values, both citations) is added to `conflicts` — never averaged,
    never resolved by picking one, since neither this function nor an LLM
    has the standing to decide which of two real documents is correct.
    That decision belongs to a person looking at the actual pages.
    """
    matches: dict = {}  # field -> list of (value, citation), in document order
    for doc in documents:
        fname = doc.get("file_name", "unknown")
        text = doc.get("text") or ""
        if not text.strip():
            continue
        low = text.lower()

        for pat, val in _TENURE_PATTERNS:
            m = re.search(pat, low)
            if m:
                matches.setdefault("tenure", []).append(
                    (val, _citation(fname, _find_page(text, m.start())))
                )
                break  # one tenure statement per document is enough

        m = _RENT_PATTERN.search(text)
        if m:
            val = float(m.group(1).replace(",", ""))
            page = _find_page(text, m.start())
            matches.setdefault("passing_rent_pa", []).append((val, _citation(fname, page)))
            window = text[m.end(): m.end() + 120]
            rm = _REVIEW_PATTERN.search(window)
            if rm:
                matches.setdefault("rent_review_basis", []).append(
                    (rm.group(1).strip(), _citation(fname, page))
                )

        m = _VACANT_PATTERN.search(text)
        if m:
            matches.setdefault("_vacant_possession", []).append(
                (True, _citation(fname, _find_page(text, m.start())))
            )

    fields: dict = {}
    citations: dict = {}
    conflicts: list = []

    for field, found in matches.items():
        distinct: list = []
        for val, cite in found:
            if val not in [v for v, _ in distinct]:
                distinct.append((val, cite))
        if len(distinct) == 1:
            fields[field] = distinct[0][0]
            citations[field] = distinct[0][1]
        else:
            # More than one distinct value found across documents for the
            # same fact. _vacant_possession is boolean-only (True or
            # nothing is ever recorded, so this branch can't fire for it)
            # -- this is real for tenure / passing_rent_pa / rent_review_basis.
            detail = "; ".join(f"{v!r} ({c})" for v, c in distinct)
            conflicts.append(
                f"Conflicting values found for {field} across documents: "
                f"{detail} — needs manual review; nothing written "
                f"automatically for this field."
            )

    return fields, citations, conflicts


def compute_unexpired_term_years(lease_start_iso: str, lease_term_years: float,
                                  today: date = None):
    """Server-side guard — see module docstring rule 3 (EXPIRED-TERM RULE).
    NEVER trust an LLM's own arithmetic for a persisted, date-sensitive
    valuation input. Returns (years: float|None, evidence_gap: str|None).
    Tested against a real 10-year lease from 25.12.2003 (Lot 2): correctly
    returns (None, "<explanation>") rather than a stale figure, because
    that term ended 25.12.2013.
    """
    if today is None:
        today = date.today()
    try:
        y, m, d = [int(x) for x in lease_start_iso.split("-")]
        start = date(y, m, d)
    except Exception:
        return None, f"unparseable lease start date: {lease_start_iso!r}"
    if lease_term_years is None:
        return None, "lease start date found but no term length found"
    end_year = start.year + int(lease_term_years)
    try:
        end = start.replace(year=end_year)
    except ValueError:
        end = start.replace(year=end_year, day=28)
    if end <= today:
        return None, (
            f"lease term expired {end.isoformat()} "
            f"({(today - end).days} days ago) — tenant status is "
            f"periodic/holdover, not a confirmed unexpired term; needs "
            f"legal review, not an assumed figure"
        )
    remaining_days = (end - today).days
    return round(remaining_days / 365.25, 2), None


_LLM_SYSTEM_PROMPT = """You extract facts from UK commercial auction legal-pack documents. You are reading documents already known to be legally readable text (not scanned images).

Extract ONLY these raw facts, each with an exact page citation using the "=== PAGE N ===" markers in the text:
- tenant_name: the individual or company named as tenant/lessee in a lease or tenancy document (not the seller/council/landlord)
- lease_start_date: the commencement date of the CURRENT/relevant lease or tenancy, normalised to YYYY-MM-DD
- lease_term_years: the length of that lease's term in years, as a plain number

CRITICAL RULES:
1. Only extract a fact you can point to an exact page for. If you cannot find a page reference, do not include the field.
2. Do NOT compute or state an "unexpired term" or "years remaining" yourself — only extract the raw start date and term length exactly as stated in the document. A separate process checks whether that term has already ended.
3. Do NOT infer a fact from a document title or file name alone — extract only from the body text.
4. If a document shows a registered leasehold title belonging to a TENANT (an "LH register" or similar), that is the tenant's own interest, not necessarily related to what is being sold. Only use it as a source for tenant_name / lease_start_date / lease_term_years, never for the tenure of the sale.
5. If nothing in the provided text supports a field, omit it entirely. Do not guess, estimate, or use general market knowledge.
6. If two documents state DIFFERENT values for the same fact (e.g. two different tenant names, or two different start dates for what appears to be the same tenancy), do NOT pick one — report it in "conflicts" instead, describing both values and which document/page each came from. Do not silently resolve a conflict yourself.

Return ONLY valid JSON, no prose, no markdown fences:
{
  "tenant_name": {"value": "...", "citation": "<file name>, page N"} or null,
  "lease_start_date": {"value": "YYYY-MM-DD", "citation": "<file name>, page N"} or null,
  "lease_term_years": {"value": number, "citation": "<file name>, page N"} or null,
  "conflicts": ["description of any contradictory facts noticed, with both citations"]
}"""


def extract_via_llm(documents: list, already_found: dict) -> tuple:
    """LLM fallback layer for facts the regex layer can't reliably get
    (tenant_name, lease_start_date, lease_term_years). See module docstring
    STATUS note — unverified in a live run as of this build (no API access
    in the build sandbox). Uses app.py's existing _llm_json_anthropic()
    helper via a lazy import, matching commercial_routes.py's own pattern
    for avoiding a circular import at module load time.

    `already_found` is the deterministic layer's output — passed in only
    so this function can skip calling the LLM at all if there's nothing
    left for it to usefully find (keeps cost down; the LLM never overrides
    a deterministic match, it only fills genuine gaps).

    Returns (raw_facts: dict, citations: dict, conflicts: list[str]) where
    raw_facts may contain tenant_name (final, allow-listed) and
    lease_start_date/lease_term_years (intermediate only — consumed by
    compute_unexpired_term_years, never written directly to
    financials_json since they aren't in the engine's allow-list).
    conflicts is the model's own report of contradictory facts it noticed
    across documents (see rule 6 in _LLM_SYSTEM_PROMPT) — folded into the
    caller's evidence_gaps alongside the deterministic layer's conflicts.
    """
    from app import _llm_json_anthropic  # lazy import — avoids circular import at load time

    _PER_DOC = 12000   # generous vs the residential 6000-char cap: lease
                        # review clauses often sit deep in a 15-20 page
                        # document, and P1 only targets a handful of fields
                        # rather than a full risk sweep, so the token cost
                        # of a larger per-doc cap is worth it here.
    _HARD_CAP = 60000

    parts = []
    total = 0
    for doc in documents:
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        label = f"=== DOCUMENT: {doc.get('file_name', 'unknown')} ===\n"
        capped = text[:_PER_DOC] + ("\n[...truncated...]" if len(text) > _PER_DOC else "")
        chunk = label + capped + "\n\n"
        if total + len(chunk) > _HARD_CAP:
            break
        parts.append(chunk)
        total += len(chunk)
    combined = "".join(parts)

    if not combined.strip():
        return {}, {}, []

    try:
        result = _llm_json_anthropic(
            system=_LLM_SYSTEM_PROMPT,
            prompt=f"Extract from these documents:\n\n{combined}",
            temperature=0.1,
        )
    except Exception:
        # Degrade gracefully — a failed LLM call must never block the
        # deterministic fields (tenure, passing rent) that already
        # succeeded. Caller proceeds with whatever it already has.
        return {}, {}, []

    fields: dict = {}
    citations: dict = {}
    for key in ("tenant_name", "lease_start_date", "lease_term_years"):
        entry = result.get(key) if isinstance(result, dict) else None
        if isinstance(entry, dict) and entry.get("value") is not None and entry.get("citation"):
            fields[key] = entry["value"]
            citations[key] = str(entry["citation"])
    conflicts = [str(c) for c in (result.get("conflicts") or []) if isinstance(result, dict)] \
        if isinstance(result, dict) else []
    return fields, citations, conflicts


def extract_commercial_fields(documents: list, today: date = None) -> dict:
    """Main entry point. `documents` — list of {"file_name": str,
    "text": str} for a deal (text = the documents table's extracted_text
    column, page-marked once the S-COMM-P1 extraction service fix is
    deployed). Returns:
      {
        "fields": {field: value, ...},          # allow-listed only
        "provenance": {field: {"source": "extracted", "citation": ..., "at": iso}},
        "evidence_gaps": [str, ...],
      }
    Ready for the caller to merge into financials_json.inputs.commercial /
    commercial_provenance exactly as commercial_routes.py's
    save_commercial_inputs() already merges user-entered fields — this
    function does not touch the database itself.
    """
    evidence_gaps: list = []

    det_fields, det_citations, det_conflicts = extract_deterministic(documents)
    vacant = det_fields.pop("_vacant_possession", False)
    vacant_citation = det_citations.pop("_vacant_possession", None)
    evidence_gaps.extend(det_conflicts)

    # S-COMM-P1-CONFLICT: a pack stating vacant possession while some
    # document also states a passing rent is itself a contradiction —
    # distinct from two documents disagreeing on the rent AMOUNT (handled
    # above by the conflict-detection in extract_deterministic). This is
    # "is there a tenancy at all", and it needs a person's eyes just as
    # much as a numeric mismatch does.
    vacant_rent_conflict = vacant and "passing_rent_pa" in det_fields
    if vacant_rent_conflict:
        withheld_rent = det_fields.pop("passing_rent_pa")
        withheld_citation = det_citations.pop("passing_rent_pa", "unknown")
        det_fields.pop("rent_review_basis", None)
        det_citations.pop("rent_review_basis", None)
        evidence_gaps.append(
            f"Pack states vacant possession ({vacant_citation}) but a passing "
            f"rent of £{withheld_rent:,.2f} was also found ({withheld_citation}) "
            f"— these contradict each other, so passing_rent_pa was NOT "
            f"written; needs a person to confirm whether the property is "
            f"actually let or vacant before either figure is used."
        )

    llm_fields, llm_citations, llm_conflicts = extract_via_llm(documents, det_fields)
    evidence_gaps.extend(llm_conflicts)

    # Deterministic never gets overridden — only fill genuine gaps.
    tenant_name = llm_fields.get("tenant_name")
    lease_start_date = llm_fields.get("lease_start_date")
    lease_term_years = llm_fields.get("lease_term_years")

    fields: dict = {}
    citations: dict = {}

    if "tenure" in det_fields:
        fields["tenure"] = det_fields["tenure"]
        citations["tenure"] = det_citations["tenure"]

    if "passing_rent_pa" in det_fields:
        fields["passing_rent_pa"] = det_fields["passing_rent_pa"]
        citations["passing_rent_pa"] = det_citations["passing_rent_pa"]
    elif vacant and not vacant_rent_conflict:
        evidence_gaps.append(
            "Pack states vacant possession and no passing rent was found — "
            "passing_rent_pa left unset rather than assumed."
        )

    if "rent_review_basis" in det_fields:
        fields["rent_review_basis"] = det_fields["rent_review_basis"]
        citations["rent_review_basis"] = det_citations["rent_review_basis"]

    if tenant_name:
        fields["tenant_name"] = tenant_name
        citations["tenant_name"] = llm_citations.get("tenant_name", "unknown")

    if lease_start_date and lease_term_years:
        years, gap = compute_unexpired_term_years(lease_start_date, lease_term_years, today=today)
        if years is not None:
            fields["unexpired_term_years"] = years
            fields["wault_years"] = years  # single-tenancy only — see module docstring
            cite = (f"computed from lease start "
                    f"({llm_citations.get('lease_start_date', 'unknown')}) and term length "
                    f"({llm_citations.get('lease_term_years', 'unknown')})")
            citations["unexpired_term_years"] = cite
            citations["wault_years"] = cite
        else:
            evidence_gaps.append(gap)
    elif lease_start_date or lease_term_years:
        evidence_gaps.append(
            "Found either a lease start date or a term length but not both — "
            "cannot safely compute unexpired term from a partial fact."
        )

    now_iso = datetime.now(timezone.utc).isoformat()
    provenance = {
        field: {"source": "extracted", "citation": citations[field], "at": now_iso}
        for field in fields
        if field in EXTRACTABLE_FIELDS
    }
    fields = {k: v for k, v in fields.items() if k in EXTRACTABLE_FIELDS}

    return {"fields": fields, "provenance": provenance, "evidence_gaps": evidence_gaps}
