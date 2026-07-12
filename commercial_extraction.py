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

  6. MONTHLY-RENT RULE (S-COMM-P1-MONTHLY, added same session) — found by
     testing a real pack (28B Snow Hill) where the only recurring-payment
     figure in the whole document set was a Licence Fee stated as "£600.00
     per calendar month" — a figure the per-annum-only pattern could never
     see. Monthly figures are annualised (×12) and the citation says so
     explicitly. Because a monthly figure's label is often several words
     away from the amount (unlike the tightly-bound per-annum pattern),
     this required a second check: a monthly SERVICE CHARGE reads almost
     identically to monthly rent, so a figure is only accepted when the
     NEAREST preceding label is "rent" or "licence/license fee" — not
     "service charge", "insurance", "deposit", or "premium". If the
     source document is itself a Licence (not a formal Lease), an
     evidence_gap says so: a licence fee is not the same legal instrument
     as lease passing rent and shouldn't be treated as equivalent income
     security without that caveat.

  7. RESIDENTIAL-AST EXCLUSION RULE (S-COMM-P1-AST, added same session) —
     found by testing a real Mixed Use pack (68 & 68A Three Shires Oak
     Road) where the ONLY rent figure anywhere in the whole document set
     was a residential Assured Shorthold Tenancy's £700 PCM for the flat
     above, while the landlord's own CPSE.2 reply confirmed the commercial
     shop unit was genuinely VACANT ("SHOP VACANT... FLAT LET TO TENANT").
     A document that identifies itself as an AST (or otherwise invokes the
     Housing Act 1988) within its own opening ~2000 chars is excluded
     entirely from every field this module extracts — not just rent. An
     AST is a residential letting regime with none of the security-of-
     tenure or rent-review dynamics of a commercial lease; blending its
     rent into a RICS Investment Method valuation input would be a
     category error, not a quality-of-evidence issue, so it is excluded
     outright rather than merely caveated the way a Licence Fee is. The
     excluded figure is still named in an evidence_gap for transparency —
     it's just never written to passing_rent_pa.

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
contradiction). The monthly-rent logic (rule 6) was added after a real
pack (28B Snow Hill) turned out to have no per-annum figure anywhere at
all; verified against that real Licence document (correctly annualised
£600pcm to £7,200pa with the licence caveat) plus four synthetic
guard-rail tests: a monthly service charge is correctly NOT mistaken for
rent; an ordinary monthly tenancy rent is correctly annualised WITHOUT the
licence caveat; a document with both a monthly rent and a monthly service
charge correctly picks the rent; and — the trickiest case — a document
where "rent" is mentioned earlier in the text than a service-charge label
still correctly excludes the service-charge figure, because the nearest
preceding label governs, not the first-seen one. The residential-AST
exclusion (rule 7) was added after a real Mixed Use pack (68 & 68A Three
Shires Oak Road) showed the risk concretely: its only rent figure was a
residential AST's £700 PCM, safe under the code as it stood only because
"PCM" wasn't yet a matched abbreviation -- extending monthly-rent matching
to cover PCM without this safeguard first would have created a live
fabrication risk. Verified against the real pack's AST/CPSE2/Auction
Contract text (correctly excluded, correctly explained, £700 pcm named in
the evidence_gap) plus two further tests: two ASTs in one pack produce two
separate exclusion notes rather than a false same-field conflict, and a
genuine commercial lease that merely cites "Housing Act 1988" deep in an
unrelated clause (well outside the 2000-char opening window) is correctly
NOT excluded. The LLM layer
(extract_via_llm) has since been run once live, against a real deal
(73b21ec9-7064-432b-a772-7e3653bb1a01), and its own date-arithmetic guard
fired correctly on that run — but its conflicts-reporting addition (rule 6
in the system prompt, unrelated numbering to rule 6 above) has not yet
been proven live and should be treated as unverified until it has.

THREE FURTHER FIXES ADDED 2026-07-12, verified against real document text
pulled directly from Supabase for the three deals that had only ever been
analysed locally (28B Snow Hill, Unit 12 Premier Partnership Estate
Brierley Hill, 68 & 68A Three Shires Oak Road Smethwick) — not yet proven
through a live /extract call, since that requires app.py/the Render
route and wasn't reachable from the sandbox this fix was built in:

  8. S-COMM-P1-PCM — "pcm" added to the monthly-rent pattern (deferred
     until rule 7/AST landed; it now has, and the AST check runs before
     this pattern on every document, so a residential PCM figure is
     excluded before it's ever seen here). Verified: Snow Hill's real
     licence fee ("£600.00 per calendar month") still annualises
     correctly (unaffected — it already matched the spelled-out form);
     confirmed the Smethwick pack's AST clause never reaches this pattern
     regardless, since _is_residential_ast() already excludes the whole
     document first.

  9. S-COMM-P1-TENURE2 — new structured pattern for the Auction Contract
     particulars field "Freehold/Leasehold:\n<value>", checked ahead of
     the free-text _TENURE_PATTERNS. Verified against both real packs
     named in the OPEN ITEMS note: Brierley Hill's Auction Contract
     ("Freehold/Leasehold:\nLeasehold") and Smethwick's, a different
     solicitor's template ("Freehold/Leasehold \n: \nFreehold") — both
     previously extracted zero tenure, both now correctly extract it.

  10. S-COMM-P1-SUBJECT — extract_commercial_fields() and extract_via_llm()
      now accept a `subject_address` parameter forwarded into the LLM
      prompt, plus a new rule 7 in _LLM_SYSTEM_PROMPT instructing the
      model to ignore facts belonging to other units in a shared title's
      schedule of leases. Built after confirming the concrete risk on two
      real packs: Brierley Hill's title register schedule names 13 other
      leases (Units 1-17) alongside the subject Unit 12, including
      individual lease start dates and terms per unit; Snow Hill's
      register similarly schedules "22 and 24 Convent Close", "32 Snow
      Hill (Ground Floor)" etc. alongside the subject 28B. In both real
      packs tested, neither the affected schedule document nor any other
      document in the pack actually stated a £ rent figure for the
      wrong unit, so this fix could not be verified against a real
      wrong-value extraction — it is a documented, evidenced risk with a
      targeted prompt fix, not yet observed causing an actual bad value.

  11. S-COMM-P1-INTRADOC (closes the gap left open by #10 above) —
      extract_deterministic() now uses finditer() instead of search() for
      tenure and rent, so multiple distinct values found WITHIN a single
      document (not just across documents) go through the same
      conflict/no-write logic as S-COMM-P1-CONFLICT. This was flagged as
      an unfixed theoretical exposure in the first version of this note;
      closed the same session once it was clear it could be done by
      reusing the already-tested cross-document conflict machinery rather
      than inventing new subject-anchoring text-windowing logic — and
      doing so is more consistent with this module's own doctrine (a
      conflict is a person's decision, not this function's) than trying
      to algorithmically guess which of two matches is "closer" to the
      subject property. Verified: re-ran the full four-deal regression
      suite (Snow Hill, Brierley Hill, Smethwick, 13 Harborne Park Road)
      afterward with identical results — a document restating the same
      value twice still writes once, exactly as before.

  12. S-COMM-P1-VACANT2 — _VACANT_PATTERN broadened from the literal
      "vacant possession" to also match "currently vacant", the actual
      wording used in Brierley Hill's real CPSE.7 reply ("The Property is
      currently vacant as the Seller was occupying"), which the original
      pattern missed entirely. Deliberately narrow — added only the
      specific phrase confirmed in a real reply, not a broad "is vacant"
      match, which would also match the standard CPSE boilerplate
      QUESTION text ("If the Property is vacant, when and why...") on
      packs where the actual answer is "no" or "N/A", causing a false
      vacancy flag on a genuinely let property.

Also this session: cleared a batch of thirteen `user_entered` test values
(tenant_name "lolod", passing_rent_pa £35,000, etc., all timestamped
identically) that were sitting in 13 Harborne Park Road's live
financials_json.inputs.commercial, blocking that deal from ever being
usable as a genuine untouched test case for the write path per the
never-overwrite-user_entered rule. Cleared directly in Supabase, not via
this module (this module never touches the database itself).

TWO MORE FIXES, same session, found by running the updated module
against a brand new real deal (1 Oxford Street and 171 Stockton Road,
Hartlepool — a receiver's sale, RICS Common Auction Conditions 4th
Edition, Addleshaw Goddard template):

  13. S-COMM-P1-TENURE3 — added "title to the lot is freehold/leasehold"
      to _TENURE_PATTERNS. This exact RICS CAC template's Extra Special
      Conditions clause 4 ("Title to the lot is freehold and is
      registered at the Land Registry...") matched none of the existing
      patterns — a third distinct real-world tenure phrasing, after the
      "sold freehold" prose and the "Freehold/Leasehold:" particulars
      field. This specific Addleshaw Goddard / RICS CAC 4th Edition
      template is common enough that it will very likely recur.

  14. S-COMM-P1-VACANT3 — added a negation guard (_VACANT_NEGATION_RE) to
      vacant detection. Found live: the same Hartlepool pack's standard
      conditions include "(b) in condition G1.2 the words ', but
      otherwise with vacant possession on completion' are deleted" — a
      clause confirming the property IS tenanted (that's precisely why
      the standard vacant-possession term has to be struck from the
      contract), but which contains the literal substring "vacant
      possession" and would have been mismarked as an assertion of
      vacancy. Now checks a bounded window after each match for
      deletion/exclusion language ("are deleted", "does not apply",
      "removed", "excluded") before accepting it. Verified against both
      the real negated clause (correctly excluded) and a genuine
      assertion ("the property is sold with vacant possession on
      completion" — correctly still fires).

  Full four-deal regression suite (Snow Hill, Brierley Hill, Smethwick,
  13 Harborne Park Road) plus all PCM/AST/intra-document guard-rail tests
  re-run clean after both fixes. The Hartlepool pack itself: tenure
  correctly extracts "Freehold" (cited to the clause 4 sentence);
  passing_rent_pa correctly stays absent (the pack's own tenancy and
  arrears schedules are blank tables, and a full-text search confirms
  zero "per annum" or monthly-rent mentions anywhere in the 19-document
  pack) — consistent with the Receivers Note's own "occupied on terms
  unknown ... no further information" disclosure. No fabrication either
  way: a real, present tenancy with genuinely unknown terms correctly
  produces no rent figure and no false vacancy flag, rather than either
  guessing a number or wrongly implying the property is empty.
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
    (r"title\s+to\s+the\s+lot\s+is\s+freehold", "Freehold"),
    (r"title\s+to\s+the\s+lot\s+is\s+leasehold", "Leasehold"),
]

# S-COMM-P1-TENURE2 (2026-07-12): found by testing two real packs (Unit 12
# Premier Partnership Estate, Brierley Hill; 68 & 68A Three Shires Oak Road,
# Smethwick) -- different solicitors, same Auction Contract particulars
# template field: "Freehold/Leasehold:" followed by the value on the same
# or next line. Neither pack matched any pattern in _TENURE_PATTERNS above
# (that list expects a free-text sentence like "sold freehold" or "Title:
# Freehold Title", not a label/value particulars field), so both packs
# extracted zero tenure -- correctly safe (no fabrication), but zero
# coverage on a template two independent packs have now used. This is
# checked FIRST, ahead of the free-text patterns, because it is a more
# specific and reliable signal (a labelled particulars field naming the
# sale itself) than a prose sentence that might describe something else.
_TENURE_STRUCTURED_RE = re.compile(
    r"freehold\s*/\s*leasehold\s*:\s*(freehold|leasehold)\b", re.IGNORECASE
)
_RENT_PATTERN = re.compile(
    r"rent\s*(?:of|:)\s*£\s?([\d,]+(?:\.\d{2})?)\s+per\s+annum", re.IGNORECASE
)
_REVIEW_PATTERN = re.compile(r"\(([^()]*review[^()]*)\)", re.IGNORECASE)
_VACANT_PATTERN = re.compile(r"vacant\s+possession|currently\s+vacant\b", re.IGNORECASE)
# A "vacant possession"/"currently vacant" match doesn't always assert
# vacancy -- auction conditions routinely quote the STANDARD contract
# wording only to delete or exclude it, e.g. "...the words ', but
# otherwise with vacant possession on completion' are deleted" is a
# receiver's-sale clause confirming the property is TENANTED (that's
# precisely why the standard vacant-possession term has to be struck).
# Checked in a bounded window immediately after the match, matching the
# "nearest label" idiom already used for the monthly-rent check above.
_VACANT_NEGATION_RE = re.compile(
    r"\b(?:are|is|shall\s+be|were)\s+deleted\b|\bdoes\s+not\s+apply\b|"
    r"\bnot\s+applicable\b|\bremoved\b|\bexcluded\b|\bstruck\s+out\b",
    re.IGNORECASE,
)
_VACANT_NEGATION_WINDOW_CHARS = 80

# S-COMM-P1-MONTHLY (2026-07-12): found by testing a real pack (28B Snow
# Hill) where the only recurring-payment figure in the whole document set
# is a Licence Fee stated as "£600.00 per calendar month" -- the per-annum
# pattern above can never match this, so the field was silently absent
# even though a real, extractable figure was sitting right there. Unlike
# the per-annum pattern, a monthly figure's label is often several words
# or a line away from the amount (e.g. '"Licence Fee"\n£600.00 per
# calendar month'), so this can't require strict adjacency the way the
# per-annum pattern does -- it looks backward through a bounded window
# instead. That widened search window is exactly why a second, disjoint
# pattern for exclusions was needed: monthly SERVICE CHARGES are common in
# the same documents and read almost identically ("...£150 per calendar
# month...") -- mistaking one for rent would be a real fabrication risk,
# not a cosmetic one.
_RENT_MONTHLY_PATTERN = re.compile(
    r"£\s?([\d,]+(?:\.\d{2})?)\s*(?:per\s+(?:calendar\s+)?month|pcm)\b", re.IGNORECASE
)
_MONTHLY_LABEL_RE = re.compile(r"\b(rent|licen[cs]e fee)\b", re.IGNORECASE)
_MONTHLY_EXCLUDE_RE = re.compile(
    r"\b(service charge|insurance|deposit|premium)\b", re.IGNORECASE
)
_MONTHLY_LOOKBACK_CHARS = 150


def _find_monthly_rent(text: str) -> list:
    """Find monthly rent/licence-fee figures in `text` and return the
    regex match objects that pass the label/exclusion check (see
    S-COMM-P1-MONTHLY above). For each "£X per (calendar) month" hit,
    looks backward up to 150 chars for the nearest label word: if the
    nearest is 'rent' or 'licence/license fee', the match is accepted; if
    the nearest is 'service charge' / 'insurance' / 'deposit' / 'premium',
    or no label word appears at all in that window, the match is rejected.
    'Nearest' is deliberately by position, not by any keyword ranking --
    whichever label word sits closest to the amount is the one describing
    it, the same way a human reader would parse the sentence."""
    accepted = []
    for m in _RENT_MONTHLY_PATTERN.finditer(text):
        window = text[max(0, m.start() - _MONTHLY_LOOKBACK_CHARS): m.start()]
        labels = list(_MONTHLY_LABEL_RE.finditer(window))
        excludes = list(_MONTHLY_EXCLUDE_RE.finditer(window))
        if not labels:
            continue
        nearest_label_pos = labels[-1].start()
        nearest_exclude_pos = excludes[-1].start() if excludes else -1
        if nearest_exclude_pos > nearest_label_pos:
            continue  # an exclusion word sits closer to the amount than a rent/fee word
        accepted.append(m)
    return accepted


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


# S-COMM-P1-AST (2026-07-12): found by testing a real pack (68 & 68A Three
# Shires Oak Road, Smethwick) where the ONLY rent figure anywhere in the
# whole document set was a residential Assured Shorthold Tenancy's £700
# PCM for the flat above -- while the commercial shop unit was confirmed
# genuinely VACANT by the landlord's own CPSE.2 reply ("SHOP VACANT...
# FLAT LET TO TENANT"). The monthly-rent pattern didn't happen to match
# "£700.00 PCM" (it only matched spelled-out "per calendar month" at the
# time), so this specific pack was safe by accident, not by design --
# extending that pattern to also catch the "PCM" abbreviation (a very
# reasonable, likely next step) would have made this a live fabrication
# risk: residential letting income, blended into a RICS Investment Method
# commercial valuation, as if it were the same kind of thing. It isn't --
# an AST is governed by the Housing Act 1988, not commercial lease law,
# and has none of the same security-of-tenure or rent-review dynamics.
_AST_SIGNAL_RE = re.compile(
    r"assured\s+shorthold\s+tenancy|housing\s+act\s+1988\b", re.IGNORECASE
)
_AST_SIGNAL_WINDOW_CHARS = 2000
# Deliberately more permissive than the commercial-rent patterns (matches
# "pcm"/"pa" abbreviations too) -- there is no fabrication risk in merely
# NAMING a figure that is about to be excluded outright, and abbreviations
# are exactly what real residential tenancy templates use.
_RAW_RENT_ANY_RE = re.compile(
    r"£\s?([\d,]+(?:\.\d{2})?)\s*(per\s+annum|per\s+(?:calendar\s+)?month|pcm|pa)\b",
    re.IGNORECASE,
)


def _is_residential_ast(text: str) -> bool:
    """True if this document identifies itself as an Assured Shorthold
    Tenancy (or otherwise invokes the Housing Act 1988) within its own
    opening ~2000 chars -- i.e. where a document declares what kind of
    document it is, not a stray citation buried in an unrelated clause
    deep in a genuinely commercial lease. See S-COMM-P1-AST above."""
    return bool(_AST_SIGNAL_RE.search(text[:_AST_SIGNAL_WINDOW_CHARS]))


def _describe_excluded_ast_rent(text: str, fname: str) -> str:
    """Best-effort, transparency-only description of a rent figure found
    in a document already identified as a residential AST. This value is
    NEVER written to passing_rent_pa -- see S-COMM-P1-AST. There is
    nothing to reduce or dedupe here (a pack can legitimately contain
    several ASTs, e.g. flats above a parade of shops), so this returns a
    plain string for the caller to append directly, rather than going
    through the same-field distinct-value conflict logic used for real
    commercial fields."""
    m = _RAW_RENT_ANY_RE.search(text)
    if not m:
        return (
            f"{fname}: identified as a residential AST — excluded from "
            f"commercial rent extraction entirely (no parseable rent "
            f"figure to quote)."
        )
    amount, period = m.group(1), m.group(2).lower()
    page = _find_page(text, m.start())
    return (
        f"{_citation(fname, page)}: identified as a residential AST "
        f"stating £{amount} {period} — this is Housing Act 1988 "
        f"residential letting income, not commercial passing rent, and "
        f"was excluded entirely rather than used for passing_rent_pa."
    )


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

    S-COMM-P1-INTRADOC (2026-07-12): the same principle now also applies
    WITHIN a single document, not just across documents. Tenure and rent
    are matched with finditer() rather than search(), so a document that
    states the same field more than once — e.g. a title register's
    schedule of leases with a per-unit rent column — feeds every distinct
    value into the conflict check below, rather than the first one
    silently winning. A document restating the SAME value twice (the
    common, harmless case) still dedupes to a single field write.

    Returns (fields: dict, citations: dict, conflicts: list[str]).
    A field is only written when every document — and every match within
    each document — that states it agrees on the same value. When two or
    more distinct values are found, the field is left OUT of `fields`
    entirely and a description of the conflict (every value, every
    citation) is added to `conflicts` — never averaged, never resolved by
    picking one, since neither this function nor an LLM has the standing
    to decide which of several real candidates is correct. That decision
    belongs to a person looking at the actual pages.
    """
    matches: dict = {}  # field -> list of (value, citation), in document order
    conflicts: list = []
    for doc in documents:
        fname = doc.get("file_name", "unknown")
        text = doc.get("text") or ""
        if not text.strip():
            continue

        # S-COMM-P1-AST: an AST document contributes NOTHING to any field
        # this function extracts -- not just rent. See helper docstrings
        # above for why. This is checked before tenure/rent/vacant so a
        # residential document can never leak into any of them.
        if _is_residential_ast(text):
            conflicts.append(_describe_excluded_ast_rent(text, fname))
            continue

        low = text.lower()

        tenure_hits = list(_TENURE_STRUCTURED_RE.finditer(text))
        if tenure_hits:
            for tm in tenure_hits:
                matches.setdefault("tenure", []).append(
                    (tm.group(1).capitalize(), _citation(fname, _find_page(text, tm.start())))
                )
        else:
            for pat, val in _TENURE_PATTERNS:
                pat_hits = list(re.finditer(pat, low))
                if pat_hits:
                    for m in pat_hits:
                        matches.setdefault("tenure", []).append(
                            (val, _citation(fname, _find_page(text, m.start())))
                        )
                    break  # one tenure-pattern family per document is enough

        annum_hits = list(_RENT_PATTERN.finditer(text))
        if annum_hits:
            for m in annum_hits:
                val = float(m.group(1).replace(",", ""))
                page = _find_page(text, m.start())
                matches.setdefault("passing_rent_pa", []).append((val, _citation(fname, page)))
                window = text[m.end(): m.end() + 120]
                rm = _REVIEW_PATTERN.search(window)
                if rm:
                    matches.setdefault("rent_review_basis", []).append(
                        (rm.group(1).strip(), _citation(fname, page))
                    )
        else:
            # No per-annum figure in this document -- try the monthly
            # fallback before giving up on it entirely (S-COMM-P1-MONTHLY).
            # Uses ALL accepted hits, not just the first, for the same
            # S-COMM-P1-INTRADOC reason as the annum branch above.
            monthly_hits = _find_monthly_rent(text)
            for mm in monthly_hits:
                monthly_val = float(mm.group(1).replace(",", ""))
                annual_val = round(monthly_val * 12, 2)
                page = _find_page(text, mm.start())
                cite = (
                    f"{_citation(fname, page)} "
                    f"(annualised from £{monthly_val:,.2f} per calendar month)"
                )
                matches.setdefault("passing_rent_pa", []).append((annual_val, cite))
                # A Licence Fee is not the same legal instrument as lease
                # passing rent -- a licence is a personal permission to
                # occupy, generally easier to terminate and not binding on
                # successors the way a lease is. Treating it as equivalent
                # security of income without saying so would overstate
                # what the figure actually represents. Checked against the
                # start of THIS document only (not the whole pack) so the
                # caveat is tied to the document the figure actually came
                # from, not to a licence appearing elsewhere in the pack.
                if re.search(r"\blicen[cs]e\b", text[:500], re.IGNORECASE):
                    matches.setdefault("_licence_not_lease_gap", []).append(
                        (True, _citation(fname, page))
                    )

        for vm in _VACANT_PATTERN.finditer(text):
            lookahead = text[vm.end(): vm.end() + _VACANT_NEGATION_WINDOW_CHARS]
            if _VACANT_NEGATION_RE.search(lookahead):
                continue  # this occurrence describes the term being deleted/excluded, not asserted
            matches.setdefault("_vacant_possession", []).append(
                (True, _citation(fname, _find_page(text, vm.start())))
            )
            break  # boolean field -- one confirmed (non-negated) hit is enough

    fields: dict = {}
    citations: dict = {}

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
7. A "SUBJECT PROPERTY" is named below, if known. Documents in a legal pack — especially a freehold or leasehold title register's "schedule of leases" or "schedule of notices of leases" — very often list OTHER units, flats, or addresses within the same shared title, alongside the subject property. Only extract a fact that clearly belongs to the SUBJECT PROPERTY. Never extract a tenant name, lease start date, or lease term from a table row, schedule entry, or clause that names a different unit/flat/address, even if it is formatted identically to a row that does belong to the subject property. If a fact's ownership is ambiguous — you cannot tell which unit it belongs to — omit it entirely rather than guessing. If no SUBJECT PROPERTY is given below, apply this same caution using only the property address stated in the documents themselves.

Return ONLY valid JSON, no prose, no markdown fences:
{
  "tenant_name": {"value": "...", "citation": "<file name>, page N"} or null,
  "lease_start_date": {"value": "YYYY-MM-DD", "citation": "<file name>, page N"} or null,
  "lease_term_years": {"value": number, "citation": "<file name>, page N"} or null,
  "conflicts": ["description of any contradictory facts noticed, with both citations"]
}"""


def extract_via_llm(documents: list, already_found: dict, subject_address: str = None) -> tuple:
    """LLM fallback layer for facts the regex layer can't reliably get
    (tenant_name, lease_start_date, lease_term_years). See module docstring
    STATUS note — proven live once (13 Harborne Park Road), unverified on a
    genuinely untouched deal as of this build. Uses app.py's existing
    _llm_json_anthropic() helper via a lazy import, matching
    commercial_routes.py's own pattern for avoiding a circular import at
    module load time.

    `already_found` is the deterministic layer's output — passed in only
    so this function can skip calling the LLM at all if there's nothing
    left for it to usefully find (keeps cost down; the LLM never overrides
    a deterministic match, it only fills genuine gaps).

    `subject_address` — S-COMM-P1-SUBJECT (2026-07-12): the deal's own
    address (e.g. deals.address), passed by the caller so the prompt can
    tell the model which unit is actually being valued when a document
    lists several (a shared title's schedule of leases). Optional —
    degrades to a general caution instruction (system prompt rule 7) if
    the caller doesn't have it, but the real protection needs the caller
    (commercial_routes.py's /extract route) to pass the live deal address
    in here.

    Returns (raw_facts: dict, citations: dict, conflicts: list[str]) where
    raw_facts may contain tenant_name (final, allow-listed) and
    lease_start_date/lease_term_years (intermediate only — consumed by
    compute_unexpired_term_years, never written directly to
    financials_json since they aren't in the engine's allow-list).
    conflicts is the model's own report of contradictory facts it noticed
    across documents (see rule 6 in _LLM_SYSTEM_PROMPT) — folded into the
    caller's evidence_gaps alongside the deterministic layer's conflicts.
    """
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

    subject_line = (
        f"SUBJECT PROPERTY: {subject_address}\n\n" if subject_address else ""
    )
    try:
        from app import _llm_json_anthropic  # lazy import — avoids circular import at load time
        result = _llm_json_anthropic(
            system=_LLM_SYSTEM_PROMPT,
            prompt=f"{subject_line}Extract from these documents:\n\n{combined}",
            temperature=0.1,
        )
    except Exception:
        # Degrade gracefully — a failed LLM call (or a failed import of
        # the helper itself; see S-COMM-P1-IMPORT-GUARD above) must never
        # block the deterministic fields (tenure, passing rent) that
        # already succeeded. Caller proceeds with whatever it already has.
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


def extract_commercial_fields(documents: list, today: date = None,
                               subject_address: str = None) -> dict:
    """Main entry point. `documents` — list of {"file_name": str,
    "text": str} for a deal (text = the documents table's extracted_text
    column, page-marked once the S-COMM-P1 extraction service fix is
    deployed). `subject_address` — S-COMM-P1-SUBJECT: the deal's own
    address (deals.address), forwarded to the LLM layer so it can tell
    the subject property apart from other units named in a shared title's
    schedule of leases. Optional but should be passed by the caller
    whenever available — see extract_via_llm's docstring. Returns:
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
    licence_not_lease = det_fields.pop("_licence_not_lease_gap", False)
    licence_citation = det_citations.pop("_licence_not_lease_gap", None)
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

    llm_fields, llm_citations, llm_conflicts = extract_via_llm(documents, det_fields, subject_address)
    evidence_gaps.extend(llm_conflicts)

    # S-COMM-P1-MONTHLY: a passing rent sourced from a Licence (not a
    # formal Lease) needs saying so -- a licence is a personal permission
    # to occupy, generally easier to terminate than a lease and not
    # binding on successors in title the same way. Only relevant if the
    # figure is actually going to be written (it won't be if the vacant/
    # rent contradiction above already withheld it).
    if licence_not_lease and "passing_rent_pa" in det_fields:
        evidence_gaps.append(
            f"passing_rent_pa was sourced from a Licence, not a formal Lease "
            f"({licence_citation}) — a licence fee is not the same legal "
            f"instrument as lease passing rent (generally easier to "
            f"terminate, not binding on successors the same way); treat "
            f"the income security accordingly, not as equivalent to a "
            f"standard FRI lease."
        )

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
