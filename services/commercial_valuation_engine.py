"""
services/commercial_valuation_engine.py
========================================
LegalSmegal Commercial Valuation Engine — Phase 1.

SCOPE
-----
This engine is deliberately SEPARATE from services/ceiling_engine.py and is
NEVER called by it. ceiling_engine.py values residential property via
comparable sold prices; it gates out ("manual_review_required") the moment
a deal is identified as Commercial (see S-COMM-GATE in ceiling_engine.py).
This file is where Commercial deals are valued instead.

METHODOLOGY — RICS Red Book Global Standards (Dec 2024 ed., eff. 31 Jan 2025)
-----------------------------------------------------------------------------
Implements the Investment Method (VPS 3), the standard RICS approach for
income-producing commercial property:
  - Rack-rented (passing rent == market rent): straight income capitalisation
    in perpetuity at a single yield.
  - Under-rented / reversionary (passing < market): term & reversion
    (vertical slicing) — term income capitalised to the reversion date at a
    term yield, plus the market rent capitalised in perpetuity, deferred,
    at a reversion yield.
  - Over-rented (passing > market): hardcore/layer (horizontal slicing) —
    the market-rent "core" capitalised in perpetuity, plus the rent
    "top-slice" above market capitalised only for the remaining secure term.

PHASE 1 SCOPE AND LIMITS (see LegalSmegal_Commercial_Scotland_Briefing)
------------------------------------------------------------------------
- This engine implements ONLY the Investment Method, for property let to a
  tenant on an income-producing basis (asset_class="income_producing_let",
  the default). RICS values other commercial asset classes by a genuinely
  different method: trade-related property (pubs, hotels, care homes,
  petrol stations) by the Profits Method; development/redevelopment sites
  by the Residual Method; specialised owner-occupied property by
  Depreciated Replacement Cost. Running Investment Method math on those
  asset classes would be the wrong method, not just an approximation — so
  this engine does not attempt it. Callers state asset_class explicitly
  (this engine does not infer it from pack text, which would be guessing);
  anything other than income_producing_let gates out with
  status="manual_review_required" and names the correct RICS method.
- Yield is a MANUAL USER INPUT. This engine does not source or fabricate a
  yield from any market database — no CoStar/MSCI integration exists yet
  (that is Phase 2). Any deal missing a yield returns insufficient_evidence.
- This computes the NOMINAL equivalent yield convention (annually in
  arrears), not the "true equivalent yield" (quarterly-in-advance adjusted).
  That refinement is deferred — flagged here AND in every response's
  "yield_convention"/"yield_convention_note" fields, not silently assumed.
- Where a single yield is supplied and no separate term/reversion/top-slice
  yields are given, the same yield is applied to all slices. This is stated
  as an explicit assumption, not presented as a differentiated market view.

CRITICAL PROHIBITIONS (mirrors services/ceiling_engine.py doctrine)
--------------------------------------------------------------------
- Do NOT fabricate a yield, market rent, or lease term that was not supplied.
- Do NOT present output as a RICS Red Book valuation — it is not one unless
  a Registered Valuer applies professional judgement to it (RICS Red Book
  2025 ed., PS 1 / VPS 5 — automated/model output is not itself a written
  valuation).
- Do NOT reuse or call anything from services/ceiling_engine.py — the two
  engines must stay structurally independent (see LegalSmegal's documented
  dual-scoring-system lesson: do not create a third ambiguous truth source).

Decision-support only. Not financial advice. Not a RICS valuation.
LegalSmegal Technologies Ltd is not FCA-regulated.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)

VERSION = "commercial_multi_method_v2_4_provenance_contract"

# Yield basis — v2.3. The nominal (annually in arrears) convention is the
# Argus/market default; the TRUE equivalent yield basis (rent received
# quarterly in advance, the actual UK payment convention) is now also
# implemented — the person chooses which basis their entered yield is on.
# The two are different interpretations of the same number: at the same
# yield, quarterly-in-advance capitalisation produces a HIGHER value
# (income arrives earlier).
YIELD_BASIS_NOMINAL        = "nominal_annually_in_arrears"
YIELD_BASIS_TRUE_QUARTERLY = "true_quarterly_in_advance"

YIELD_CONVENTION_QUARTERLY = "true_equivalent_yield_quarterly_in_advance"
YIELD_CONVENTION_NOTE_QUARTERLY = (
    "Yield is capitalised on the TRUE equivalent yield basis — rent treated "
    "as received quarterly in advance, the actual UK payment convention. At "
    "the same yield figure this produces a higher value than the nominal "
    "(annually in arrears) basis, because income arrives earlier. Make sure "
    "the yield entered was analysed on this basis; a nominal yield entered "
    "here will overstate value."
)

# Input-plausibility sanity bounds — v2.2. These are ENGINEERING-JUDGEMENT
# thresholds for catching fat-finger input errors (a misplaced zero), NOT
# market data and NOT figures RICS specifies. They only ever append a
# warning; they never block, refuse, or alter the computed number — the
# valuation is always computed honestly from the inputs as given.
RENT_RATIO_SANITY_MULTIPLE = 3.0   # market vs passing rent ratio beyond which an input-check warning fires
NIY_SANITY_FLOOR_PCT   = 1.0       # implied initial yield below this is far outside observed UK commercial trading
NIY_SANITY_CEILING_PCT = 20.0      # implied initial yield above this likewise

# Yield sensitivity display steps — v2.2. ARGUS-style sensitivity output:
# the same formula recomputed with every slice yield shifted by these basis
# points. Display steps only, not a forecast.
SENSITIVITY_SHIFTS_BPS = (-50, -25, 25, 50)

RACK_RENT_TOLERANCE_PCT = 0.01  # within 1% treated as rack-rented (avoids float-equality edge cases) — an engineering judgement call, not a figure RICS specifies numerically.

YIELD_CONVENTION = "nominal_equivalent_yield_annually_in_arrears"
YIELD_CONVENTION_NOTE = (
    "Yield is capitalised on a NOMINAL equivalent yield basis (income treated "
    "as received annually in arrears) — the Argus/market default. This is NOT "
    "the true equivalent yield (adjusted for the UK convention of rent "
    "actually being received quarterly in advance), which would produce a "
    "slightly higher yield / slightly lower capital value. That refinement "
    "is not implemented in Phase 1."
)

# Purchaser's costs — v2.1. Deducting purchaser's costs to move from a
# capital value gross of costs to a net price is standard UK institutional
# practice (the London-market convention totals ~6.78% at large lot sizes:
# ~1.78% fees incl. VAT + SDLT at the 5% top band). The FEES component is a
# STATED, EDITABLE ASSUMPTION (default below); the SDLT component is COMPUTED
# on the stepped England & NI non-residential freehold bands — never a flat
# percentage, because the bands step at £150k/£250k and a flat 6.78% is only
# right for large lots. Wales (LTT) and Scotland (LBTT) have different
# non-residential bands NOT implemented here — those nations gate the
# net-of-costs figure rather than computing it on the wrong nation's tax.
DEFAULT_PURCHASER_FEES_PCT = 1.8  # agent ~1% + legal ~0.5% + VAT on fees — assumption, editable per deal

# SDLT non-residential freehold consideration bands, England & Northern
# Ireland: 0% to £150,000; 2% on £150,001–£250,000; 5% above £250,000.
# Top rate capped at 5%; no 3% additional-dwelling surcharge (that is
# residential-only). Stepped/marginal, not slab.
_SDLT_NONRES_BAND_1_UPPER = 150_000
_SDLT_NONRES_BAND_2_UPPER = 250_000
_SDLT_NONRES_BAND_2_RATE = 0.02
_SDLT_NONRES_BAND_3_RATE = 0.05

# Upward-only rent review (UORR) legislative risk — England & Wales.
# Facts as verified 2026-07: English Devolution and Community Empowerment
# Act 2026, Royal Assent 29 April 2026, bans upward-only rent reviews in
# new commercial leases in England and Wales. NOT yet in force — secondary
# legislation expected 2027 (consultation on caps/collars pending) — but a
# late retrospective amendment catches tenancy RENEWAL arrangements entered
# into on or after 17 March 2026. Phrased below as legislative risk, never
# as law currently in force.
UORR_LEGISLATIVE_RISK_WARNING = (
    "Rent review basis is upward-only: the English Devolution and Community "
    "Empowerment Act 2026 (Royal Assent 29 April 2026) bans upward-only rent "
    "reviews in new commercial leases in England and Wales. The ban is not "
    "yet in force — secondary legislation is expected in 2027 — but renewal "
    "arrangements entered into on or after 17 March 2026 may be caught by a "
    "retrospective amendment. An upward-only review pattern supporting this "
    "yield may not be replicable on re-letting or renewal — treat as a "
    "legislative risk to reversionary value, not current law in force."
)

# Asset classes RICS values by a DIFFERENT method to the Investment Method
# this engine implements. Rather than guessing an asset's subtype from
# whatever text happens to be in the pack (which would be fabrication),
# the caller states it explicitly via financial_inputs["asset_class"].
# Four classes route to a genuinely built RICS method — see the router
# calculate_commercial_ceiling() below and its four _calculate_* functions.
# A fifth, mixed_use, does NOT route to any single method — see its comment below.
ASSET_CLASS_INCOME_PRODUCING_LET      = "income_producing_let"       # -> Investment Method
ASSET_CLASS_TRADE_RELATED             = "trade_related"              # -> RICS Profits Method (pubs, hotels, care homes, petrol stations)
ASSET_CLASS_DEVELOPMENT_SITE          = "development_site"           # -> RICS Residual Method
ASSET_CLASS_SPECIALISED_OWNER_OCCUPIED = "specialised_owner_occupied" # -> RICS Depreciated Replacement Cost / Contractor's Method
# Mixed use (e.g. ground-floor retail let to a tenant + residential flats
# above, in one title) is NOT a single-method case: RICS practice apportions
# value between the commercial element (Investment Method, on its own rent/
# yield) and the residential element (comparable sold prices), typically
# needing a floor-area-by-use split this engine does not collect in Phase 1.
# Running any ONE of the four methods on the whole asset would misvalue it —
# so like trade_related/development_site/specialised_owner_occupied used to
# gate, mixed_use gates too, but for a different reason: not "wrong method",
# but "needs apportionment this engine doesn't do yet."
ASSET_CLASS_MIXED_USE                 = "mixed_use"                  # -> gates: apportioned valuation not yet supported


def _yp_years(n: float, i: float, quarterly: bool = False) -> Optional[float]:
    """Years' Purchase for n years at yield i (decimal). Capitalises a
    finite income stream — used for the term slice and the over-rented
    top-slice. quarterly=True uses the quarterly-in-advance convention:
    YP = (1 − (1+i)^−n) / (4 × (1 − (1+i)^−0.25))."""
    if i is None or i <= 0 or n is None or n <= 0:
        return None
    if quarterly:
        return (1 - (1 + i) ** -n) / (4 * (1 - (1 + i) ** -0.25))
    return (1 - (1 + i) ** -n) / i


def _yp_perpetuity(i: float, quarterly: bool = False) -> Optional[float]:
    """Years' Purchase in perpetuity at yield i (decimal). Capitalises an
    income stream assumed to continue indefinitely — used for rack-rented
    and hardcore-layer core valuations. quarterly=True uses the
    quarterly-in-advance convention: YP = 1 / (4 × (1 − (1+i)^−0.25))."""
    if i is None or i <= 0:
        return None
    if quarterly:
        return 1 / (4 * (1 - (1 + i) ** -0.25))
    return 1 / i


def _yp_perpetuity_deferred(n: float, i: float, quarterly: bool = False) -> Optional[float]:
    """Years' Purchase in perpetuity, deferred n years, at yield i
    (decimal). Used for the reversion slice in term & reversion — the
    market rent is only received from year n onward. The deferment factor
    (1+i)^−n discounts at the effective annual rate on both bases."""
    if i is None or i <= 0 or n is None or n < 0:
        return None
    base = _yp_perpetuity(i, quarterly)
    return base * ((1 + i) ** -n) if base else None


def _pct_to_decimal(pct: Optional[float]) -> Optional[float]:
    try:
        v = float(pct)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    # Accept either "6.5" (percent) or "0.065" (decimal) — anything >1 is treated as percent.
    return v / 100 if v > 1 else v


def _sdlt_non_residential_england_ni(price: float) -> float:
    """Stepped SDLT on non-residential/mixed freehold consideration,
    England & Northern Ireland only: 0% to £150,000; 2% on the portion
    £150,001–£250,000; 5% on the portion above £250,000."""
    if price is None or price <= 0:
        return 0.0
    tax = 0.0
    if price > _SDLT_NONRES_BAND_1_UPPER:
        tax += (min(price, _SDLT_NONRES_BAND_2_UPPER) - _SDLT_NONRES_BAND_1_UPPER) * _SDLT_NONRES_BAND_2_RATE
    if price > _SDLT_NONRES_BAND_2_UPPER:
        tax += (price - _SDLT_NONRES_BAND_2_UPPER) * _SDLT_NONRES_BAND_3_RATE
    return tax


def _net_of_purchasers_costs(gross_value: float, fees_pct: Optional[float], nation: str) -> dict:
    """Solve the net price P such that P + SDLT(P) + fees%×P = gross capital
    value, by bisection (SDLT is stepped on the net consideration, so the
    relationship is circular and has no closed form). Convention: yields are
    analysed against prices gross of costs, so the YP capitalisation output
    is the GROSS value and the buyer-pays price is the NET value.

    Wales/Scotland gate: LTT/LBTT non-residential bands are not implemented —
    the net figure is withheld rather than computed on the wrong nation's tax."""
    if nation in ("wales", "scotland"):
        tax_name = "LTT (Wales)" if nation == "wales" else "LBTT (Scotland)"
        return {
            "status": "unavailable",
            "nation": nation,
            "reason": (
                f"{tax_name} non-residential bands are not implemented in this "
                "phase — the net-of-costs figure is not computed rather than "
                "computed on the wrong nation's tax. The capital value shown "
                "is gross of purchaser's costs."
            ),
        }
    fees_rate = (fees_pct if fees_pct is not None and fees_pct >= 0 else DEFAULT_PURCHASER_FEES_PCT) / 100.0
    lo, hi = 0.0, float(gross_value)
    mid = gross_value
    for _ in range(200):
        mid = (lo + hi) / 2
        f = mid + _sdlt_non_residential_england_ni(mid) + fees_rate * mid - gross_value
        if abs(f) < 0.5:
            break
        if f > 0:
            hi = mid
        else:
            lo = mid
    net = round(mid, 2)
    sdlt = round(_sdlt_non_residential_england_ni(net), 2)
    fees = round(fees_rate * net, 2)
    return {
        "status": "ok",
        "nation": "england_ni",
        "net_value_gbp": net,
        "sdlt_gbp": sdlt,
        "fees_pct": round(fees_rate * 100, 2),
        "fees_gbp": fees,
        "total_costs_gbp": round(sdlt + fees, 2),
        "basis": (
            "SDLT England & NI non-residential freehold bands (0% to £150,000; "
            "2% on £150,001–£250,000; 5% above £250,000) computed on the net "
            "price by bisection, plus purchaser's fees at the stated percentage "
            "of net price. Net price + SDLT + fees = capital value gross of costs."
        ),
    }


def _solve_yield_bisection(target_cv: float, value_at) -> Optional[float]:
    """Find the single yield e (decimal) at which value_at(e) == target_cv.
    value_at must be monotonically decreasing in e (true of every YP
    capitalisation). Used for the equivalent yield — the single weighted
    yield that, applied to all slices of the same cashflow structure,
    reproduces the capital value. Returns None if no solution brackets
    within (0.0001, 1.0) — never guesses."""
    lo, hi = 1e-4, 1.0
    v_lo, v_hi = value_at(lo), value_at(hi)
    if v_lo is None or v_hi is None or not (v_lo >= target_cv >= v_hi):
        return None
    mid = (lo + hi) / 2
    for _ in range(200):
        mid = (lo + hi) / 2
        v = value_at(mid)
        if v is None:
            return None
        if abs(v - target_cv) < 0.5:
            return mid
        if v > target_cv:
            lo = mid
        else:
            hi = mid
    return mid


def _evidence_tier(inputs_used: dict, provenance: Optional[dict] = None) -> dict:
    """Evidence tier — v2.4 two-layer model.

    Layer 1 (the TIER, A/B/C by analogy to the RICS hierarchy of evidence):
    keyed to the VALUATION-OPINION inputs — above all the yield. The tier
    stays C while the yield is user-supplied rather than market-derived,
    regardless of how well the factual inputs are documented, because the
    hierarchy of evidence is about comparable evidence for the opinion,
    not about subject-property facts.

    Layer 2 (per-field VERIFICATION): factual inputs (rent, term, review
    basis, tenure) can be verified against the legal pack by the
    server-side extraction pipeline, which writes
    {field: {"source": "extracted", "citation": "..."}} into the
    provenance map. Anything else — including any value arriving from the
    browser form, which the routes stamp as user_entered, and any
    unrecognised source string — is treated as user-entered and
    unverified. The client can never assert extraction.

    This is a provenance grade, not a statistical confidence score; no
    outcome data exists to derive one, and none is faked."""
    prov = provenance if isinstance(provenance, dict) else {}
    input_sources: dict = {}
    citations: dict = {}
    verified_fields: list[str] = []
    for k, v in (inputs_used or {}).items():
        if v is None or k == "asset_class":
            continue
        entry = prov.get(k)
        src = entry.get("source") if isinstance(entry, dict) else None
        if src == "extracted":
            input_sources[k] = "extracted"
            verified_fields.append(k)
            cit = entry.get("citation") if isinstance(entry, dict) else None
            if cit:
                citations[k] = str(cit)
        else:
            input_sources[k] = "user_entered"

    n_ver = len(verified_fields)
    if n_ver:
        tier_label = (
            f"Evidence tier C — yield unverified; {n_ver} input"
            f"{'s' if n_ver != 1 else ''} verified against documents"
        )
        basis = (
            "The tier is keyed to the valuation-opinion inputs — above all "
            "the yield, which is user-supplied, not market-derived, so the "
            "tier remains C (by analogy to the RICS hierarchy of evidence: "
            "Category A: direct comparable transactions; B: published market "
            "data; C: other sources). Separately, the factual inputs listed "
            "as 'extracted' were read from the uploaded legal pack by the "
            "extraction pipeline, with citations — facts verified against "
            "documents, distinct from evidence for the yield. This is a "
            "provenance grade, not a statistical confidence score."
        )
    else:
        tier_label = "Evidence tier C — user-entered inputs, unverified"
        basis = (
            "All supplied inputs are user-entered — none were verified "
            "against the legal pack by the extraction pipeline, and no "
            "licensed market-data path into commercial fields exists. By "
            "analogy to the RICS hierarchy of evidence (Category A: direct "
            "comparable transactions; B: published market data; C: other "
            "sources), unverified user inputs sit at Category C. This is a "
            "provenance grade, not a statistical confidence score."
        )
    out = {
        "tier": "C",
        "tier_label": tier_label,
        "basis": basis,
        "input_sources": input_sources,
    }
    if verified_fields:
        out["verified_fields"] = verified_fields
    if citations:
        out["citations"] = citations
    return out


def _ok_result(
    *, valuation_type: str, method: str, inputs_used: dict, valuation_components: dict,
    capital_value: float, assumptions: list[str], evidence_gaps: list[str],
    warnings: list[str], formula_trace: list[str], extra: Optional[dict] = None,
    provenance: Optional[dict] = None,
) -> dict:
    """Shared success-path schema builder. All four RICS method functions
    below call this — a single place the top-level shape is defined means
    the four methods can never drift into inconsistent schemas."""
    out = {
        "valuation_type":     valuation_type,
        "not_rics_valuation": True,
        "status":             "ok",
        "method":              method,
        "inputs_used":         inputs_used,
        "valuation_components": valuation_components,
        "comparable_valuation": round(capital_value, 2),
        "risk_adjusted_value":  round(capital_value, 2),  # no legal-pack risk model in Phase 1
        "valuation_range": {
            "low": None, "midpoint": round(capital_value, 2), "high": None, "uncertainty_band": None,
        },
        "confidence": {
            "raw": None, "caps": [], "final": None,
            "label": "indicative — see assumptions and evidence_gaps",
        },
        # Date this computation ran — VPS-style minimum reporting matter.
        # Explicitly NOT a valuation "as at" an inspected date: inputs and
        # market yields move, and nothing here was inspected.
        "valuation_date": date.today().isoformat(),
        "valuation_date_note": (
            "Date this computation ran — not a valuation 'as at' an inspection "
            "date; inputs and market yields move."
        ),
        "evidence_tier": _evidence_tier(inputs_used, provenance),
        "audit": {
            "comparable_method":      method,
            "not_rics_valuation":     True,
            "formal_valuation":       False,
            "decision_support_only":  True,
            "source_decision":        f"computed_from_{valuation_type}_phase1",
            "fallback_used":          False,
            "assumptions":            assumptions,
            "evidence_gaps":          evidence_gaps,
            "warnings":               warnings,
            "formula_trace":          formula_trace,
            "version":                VERSION,
        },
    }
    if extra:
        out.update(extra)
    return out


def calculate_commercial_ceiling(financial_inputs: dict, provenance: Optional[dict] = None) -> dict:
    """
    Router — dispatches to the RICS valuation method appropriate for the
    deal's asset_class. Each class is valued by a genuinely different RICS
    method (see PHASE 1 SCOPE AND LIMITS above); running the wrong method's
    maths on an asset would be incorrect, not just approximate, so this
    router — not a guess from pack text — decides which function runs.

    financial_inputs["asset_class"] one of:
        income_producing_let        (default) -> Investment Method
        trade_related                          -> Profits Method
        development_site                       -> Residual Method
        specialised_owner_occupied             -> Depreciated Replacement Cost
        mixed_use                              -> gates (see _gate_mixed_use)
    Any other/unrecognised value falls through to Investment Method (the
    default), rather than silently gating a deal on a typo.
    """
    fi = financial_inputs if isinstance(financial_inputs, dict) else {}
    # v2.4: provenance is server-supplied ONLY (routes stamp user_entered on
    # every form save; the extraction pipeline writes extracted+citation).
    # Reserved key is overwritten unconditionally so a value smuggled into
    # stored inputs can never masquerade as verified provenance.
    fi = dict(fi)
    fi["_provenance"] = provenance if isinstance(provenance, dict) else {}
    asset_class = str(fi.get("asset_class") or ASSET_CLASS_INCOME_PRODUCING_LET).strip().lower()

    if asset_class == ASSET_CLASS_TRADE_RELATED:
        return _attach_cross_check(_calculate_profits_method(fi, asset_class), fi, asset_class)
    if asset_class == ASSET_CLASS_DEVELOPMENT_SITE:
        return _calculate_residual_method(fi, asset_class)
    if asset_class == ASSET_CLASS_SPECIALISED_OWNER_OCCUPIED:
        return _calculate_drc_method(fi, asset_class)
    if asset_class == ASSET_CLASS_MIXED_USE:
        return _gate_mixed_use(fi, asset_class)
    return _attach_cross_check(_calculate_investment_method(fi, asset_class), fi, asset_class)


def _attach_cross_check(primary: dict, fi: dict, asset_class: str) -> dict:
    """
    Cross-check — v2.2. RICS practice: a second method 'may be used to
    cross check or sense check the final valuation output', and the two
    figures are RECONCILED, never averaged or blended. This attaches a
    labelled secondary figure ONLY when the person has genuinely supplied
    the second method's required inputs — it never fabricates them:
      - income_producing_let primary + fmop_pa & profit_multiplier supplied
        -> Profits Method cross-check (realistic for a let pub/hotel where
        trading data is also known);
      - trade_related primary + rent & yield supplied -> Investment Method
        cross-check.
    The primary figure is never altered by the presence of a cross-check.
    """
    if not isinstance(primary, dict) or primary.get("status") != "ok":
        return primary

    secondary = None
    label = None
    if asset_class == ASSET_CLASS_INCOME_PRODUCING_LET and fi.get("fmop_pa") and fi.get("profit_multiplier"):
        secondary = _calculate_profits_method(fi, ASSET_CLASS_TRADE_RELATED)
        label = "Profits Method (FMOP × multiplier)"
    elif asset_class == ASSET_CLASS_TRADE_RELATED and fi.get("passing_rent_pa") and (
        fi.get("yield_pct") or (fi.get("term_yield_pct") and fi.get("reversion_yield_pct"))
    ):
        secondary = _calculate_investment_method(fi, ASSET_CLASS_INCOME_PRODUCING_LET)
        label = "Investment Method"

    if not secondary or secondary.get("status") != "ok":
        return primary

    p = primary.get("comparable_valuation")
    s = secondary.get("comparable_valuation")
    diff_pct = round((s - p) / p * 100, 1) if p else None
    primary["cross_check"] = {
        "method_label": label,
        "capital_value_gbp": s,
        "difference_pct": diff_pct,
        "note": (
            "Sense-check only — RICS practice reconciles a second method's "
            "figure against the primary; it never averages or blends them. "
            "Figures compared before any purchaser's-costs deduction. The "
            "primary method follows the stated asset class; a material gap "
            "usually means the two methods are pricing different things "
            "(trading potential vs lease income) — investigate which "
            "assumption drives it."
        ),
    }
    return primary


def _gate_mixed_use(fi: dict, asset_class: str) -> dict:
    """
    Mixed use gates rather than computes — see ASSET_CLASS_MIXED_USE comment
    above for why. Unlike the manual_review_required path this engine used
    for ALL commercial deals before the four methods were built, this is not
    "we haven't built the method yet" — it is "no single method applies to
    a mixed asset without apportioning value by use first," which is a
    genuinely different, harder problem RICS practice solves case-by-case.
    """
    evidence_gaps = [
        "Asset class 'mixed_use' has both a commercial and a residential "
        "element — RICS practice apportions value between the commercial "
        "part (Investment Method, its own rent/yield) and the residential "
        "part (comparable sold prices), typically using a floor-area-by-use "
        "split. This engine does not collect that split in Phase 1 and will "
        "not force the whole asset through one method — manual RICS "
        "valuation, or a per-element split entered as two separate deals, "
        "is required."
    ]
    return _insufficient(
        fi, evidence_gaps, [], [], [],
        status="manual_review_required",
        valuation_type="commercial_mixed_use",
        inputs_used={"asset_class": asset_class},
    )


def _calculate_investment_method(fi: dict, asset_class: str) -> dict:
    """
    RICS Investment Method — for property let to a tenant on an
    income-producing basis (asset_class == income_producing_let).

    Expected keys in fi (all optional except where noted):
        passing_rent_pa      : float — current annual rent passing (£/yr). REQUIRED.
        market_rent_pa       : float — current open-market rent (£/yr). If
                                absent, assumed equal to passing_rent_pa
                                (rack-rented) with an explicit assumption logged.
        yield_pct            : float — single equivalent yield, e.g. 6.5 or
                                0.065. REQUIRED unless term_yield_pct AND
                                reversion_yield_pct (or top_slice_yield_pct)
                                are both supplied instead.
        term_yield_pct        : float — optional override for the term slice.
        reversion_yield_pct   : float — optional override for the reversion slice.
        top_slice_yield_pct   : float — optional override for the over-rented top slice.
        unexpired_term_years  : float — years to next rent review or lease
                                expiry (whichever governs the reversion/
                                over-rent event). REQUIRED unless passing
                                rent is rack-rented (== market rent).
        wault_years           : float — optional, informational only, not
                                used in the valuation maths.
        wault_to_break_years  : float — optional, informational only. WAULT
                                to first break, alongside WAULT to expiry —
                                break clauses cluster the income risk.
        tenant_name           : str  — optional, informational only.
        rent_review_basis     : str  — optional: upward_only | open_market |
                                fixed_stepped | index_linked. upward_only
                                triggers the UORR legislative-risk warning.
        nation                : str  — optional: england_ni (default, with
                                assumption logged if absent) | wales |
                                scotland. Governs the SDLT leg of purchaser's
                                costs; wales/scotland gate the net figure.
        purchaser_fees_pct    : float — optional, % of net price for agent/
                                legal/VAT. Defaults to
                                DEFAULT_PURCHASER_FEES_PCT with the default
                                stated as an assumption.
        void_months           : float — optional. Void period at reversion
                                before market rent commences (term &
                                reversion branch only).
        rent_free_months      : float — optional. Rent-free incentive at
                                reversion (term & reversion branch only).
    """
    assumptions: list[str] = []
    evidence_gaps: list[str] = []
    warnings: list[str] = []
    formula_trace: list[str] = []

    # ── v2.3: tenure gate — perpetuity capitalisation is a FREEHOLD
    #    technique. A leasehold interest is valued as profit rent over the
    #    unexpired head-lease term, a genuinely different calculation this
    #    engine does not implement — so leasehold gates (same pattern as
    #    mixed_use) rather than getting a freehold answer that is wrong,
    #    not approximate, for a wasting asset. ──────────────────────────
    tenure = str(fi.get("tenure") or "").strip().lower() or None
    if tenure == "leasehold":
        evidence_gaps.append(
            "Tenure stated as leasehold — the Investment Method as "
            "implemented here capitalises in perpetuity, which values a "
            "FREEHOLD interest. A leasehold interest is valued as the profit "
            "rent (rent received less head rent payable) over the unexpired "
            "head-lease term only — a wasting asset, not a perpetuity. That "
            "calculation is not built in this phase; this engine refuses to "
            "produce a freehold number for a leasehold asset. Manual "
            "valuation of the leasehold interest is required."
        )
        return _insufficient(
            fi, evidence_gaps, warnings, assumptions, formula_trace,
            status="manual_review_required",
        )
    if tenure is None:
        assumptions.append(
            "tenure not stated — FREEHOLD assumed (perpetuity capitalisation). "
            "If the interest is leasehold this valuation basis is wrong, not "
            "approximate: state the tenure."
        )
    elif tenure != "freehold":
        warnings.append(f"Unrecognised tenure '{tenure}' — freehold assumed.")

    # ── v2.3: yield basis — which convention the entered yield is on ────
    yield_basis = str(fi.get("yield_basis") or "").strip().lower() or YIELD_BASIS_NOMINAL
    if yield_basis not in (YIELD_BASIS_NOMINAL, YIELD_BASIS_TRUE_QUARTERLY):
        warnings.append(
            f"Unrecognised yield_basis '{yield_basis}' — nominal "
            "(annually in arrears) assumed."
        )
        yield_basis = YIELD_BASIS_NOMINAL
    q = yield_basis == YIELD_BASIS_TRUE_QUARTERLY
    formula_trace.append(f"yield_basis: {yield_basis}")

    passing_rent = fi.get("passing_rent_pa")
    try:
        passing_rent = float(passing_rent) if passing_rent is not None else None
    except (TypeError, ValueError):
        passing_rent = None

    market_rent = fi.get("market_rent_pa")
    try:
        market_rent = float(market_rent) if market_rent is not None else None
    except (TypeError, ValueError):
        market_rent = None

    if passing_rent is None or passing_rent <= 0:
        evidence_gaps.append(
            "No passing rent supplied — commercial valuation requires the "
            "current annual rent, which residential comparable data cannot provide."
        )
        return _insufficient(fi, evidence_gaps, warnings, assumptions, formula_trace)

    if market_rent is None or market_rent <= 0:
        market_rent = passing_rent
        assumptions.append(
            "market_rent_pa not supplied — assumed equal to passing rent "
            "(rack-rented assumption). Provide a market rent for a more accurate result."
        )
        formula_trace.append("market_rent_pa defaulted to passing_rent_pa (assumption)")

    yield_pct     = _pct_to_decimal(fi.get("yield_pct"))
    term_yield    = _pct_to_decimal(fi.get("term_yield_pct")) or yield_pct
    reversion_yld = _pct_to_decimal(fi.get("reversion_yield_pct")) or yield_pct
    top_yield     = _pct_to_decimal(fi.get("top_slice_yield_pct")) or yield_pct

    n_years = fi.get("unexpired_term_years")
    try:
        n_years = float(n_years) if n_years is not None else None
    except (TypeError, ValueError):
        n_years = None

    is_rack_rented = abs(passing_rent - market_rent) <= (market_rent * RACK_RENT_TOLERANCE_PCT)
    formula_trace.append(
        f"rack_rented_check: passing={passing_rent} vs market={market_rent}, "
        f"tolerance={RACK_RENT_TOLERANCE_PCT*100:.0f}% (engineering judgement, not a RICS-specified figure) "
        f"-> is_rack_rented={is_rack_rented}"
    )

    if not is_rack_rented and (n_years is None or n_years <= 0):
        evidence_gaps.append(
            "Passing rent differs from market rent but no unexpired_term_years "
            "was supplied — cannot split term/reversion or core/top-slice "
            "without knowing when the rent changes."
        )
        return _insufficient(fi, evidence_gaps, warnings, assumptions, formula_trace)

    if yield_pct is None and (term_yield is None or reversion_yld is None):
        evidence_gaps.append(
            "No yield supplied. This engine does not fabricate or source a "
            "yield automatically — provide a market-derived yield (Phase 2 "
            "will support licensed benchmark data; for now this must be a "
            "user input)."
        )
        return _insufficient(fi, evidence_gaps, warnings, assumptions, formula_trace)

    if fi.get("term_yield_pct") is None or fi.get("reversion_yield_pct") is None:
        assumptions.append(
            "single equivalent yield applied uniformly to all slices — "
            "differentiated term/reversion/top-slice yields were not supplied"
        )

    # ── v2.1 inputs: review basis, nation, purchaser's fees, void/rent-free ──
    def _opt_num(key):
        v = fi.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    review_basis = str(fi.get("rent_review_basis") or "").strip().lower() or None
    if review_basis == "upward_only":
        warnings.append(UORR_LEGISLATIVE_RISK_WARNING)

    nation = str(fi.get("nation") or "").strip().lower() or None
    if nation not in ("england_ni", "wales", "scotland"):
        if nation is not None:
            warnings.append(
                f"Unrecognised nation '{nation}' — England & NI SDLT bands assumed."
            )
        nation = "england_ni"
        if fi.get("nation") is None:
            assumptions.append(
                "nation not supplied — England & Northern Ireland SDLT "
                "non-residential bands assumed for purchaser's costs. Select "
                "Wales or Scotland if applicable (different tax: LTT/LBTT)."
            )

    purchaser_fees_pct = _opt_num("purchaser_fees_pct")
    if purchaser_fees_pct is None:
        assumptions.append(
            f"purchaser's fees defaulted to {DEFAULT_PURCHASER_FEES_PCT}% of net "
            "price (agent ~1% + legal ~0.5% + VAT — London-market convention "
            "component). A stated assumption, editable per deal."
        )
    elif purchaser_fees_pct < 0:
        warnings.append("Negative purchaser_fees_pct ignored — default applied.")
        purchaser_fees_pct = None

    void_months = _opt_num("void_months")
    rent_free_months = _opt_num("rent_free_months")
    if void_months is not None and void_months < 0:
        warnings.append("Negative void_months ignored.")
        void_months = None
    if rent_free_months is not None and rent_free_months < 0:
        warnings.append("Negative rent_free_months ignored.")
        rent_free_months = None
    extra_defer_years = ((void_months or 0.0) + (rent_free_months or 0.0)) / 12.0

    wault_to_break = _opt_num("wault_to_break_years")

    # ── Investment Method ─────────────────────────────────────────────────
    components: dict = {}
    waterfall: list[dict] = []
    method_reasoning = ""
    equivalent_yield: Optional[float] = None
    defer_years: Optional[float] = None  # set in the term & reversion branch; used by the sensitivity recompute
    if is_rack_rented:
        method = "investment_method_rack_rented_perpetuity"
        rack_yield = yield_pct or term_yield
        yp = _yp_perpetuity(rack_yield, q)
        if yp is None:
            evidence_gaps.append("Yield invalid (<=0) — cannot capitalise rack-rented income.")
            return _insufficient(fi, evidence_gaps, warnings, assumptions, formula_trace)
        capital_value = passing_rent * yp
        formula_trace.append(
            f"rack_rented: capital_value = passing_rent({passing_rent}) × YP_perp({yp:.4f})"
        )
        # No separate components to show — the headline figure IS the whole calculation.
        method_reasoning = (
            f"Passing rent (£{passing_rent:,.0f}/yr) is within "
            f"{RACK_RENT_TOLERANCE_PCT * 100:.0f}% of market rent "
            f"(£{market_rent:,.0f}/yr) — treated as rack-rented and capitalised "
            f"in perpetuity at {rack_yield * 100:.2f}%."
        )
        equivalent_yield = rack_yield  # single slice — the yield IS the equivalent yield
        waterfall.append({
            "label": (
                f"Passing rent £{passing_rent:,.0f}/yr × YP in perpetuity "
                f"@ {rack_yield * 100:.2f}% ({yp:.4f})"
            ),
            "amount": round(capital_value, 2),
        })
        if extra_defer_years > 0:
            warnings.append(
                "void_months / rent_free_months supplied but not applied — "
                "only modelled for the under-rented term & reversion case in "
                "this phase."
            )

    elif passing_rent < market_rent:
        method = "investment_method_term_and_reversion"
        defer_years = n_years + extra_defer_years
        yp_term = _yp_years(n_years, term_yield, q)
        yp_rev  = _yp_perpetuity_deferred(defer_years, reversion_yld, q)
        if yp_term is None or yp_rev is None:
            evidence_gaps.append("Term or reversion yield invalid — cannot compute term & reversion.")
            return _insufficient(fi, evidence_gaps, warnings, assumptions, formula_trace)
        term_value      = passing_rent * yp_term
        reversion_value = market_rent * yp_rev
        capital_value   = term_value + reversion_value
        formula_trace.append(
            f"term_and_reversion: term={passing_rent}×YP_years({n_years}y,{term_yield:.4f})="
            f"{term_value:.2f}; reversion={market_rent}×YP_perp_deferred({defer_years}y,{reversion_yld:.4f})="
            f"{reversion_value:.2f}"
        )
        if extra_defer_years > 0:
            formula_trace.append(
                f"reversion deferment extended by void/rent-free: "
                f"{n_years}y + {extra_defer_years:.4f}y "
                f"(void {void_months or 0:g}m + rent-free {rent_free_months or 0:g}m) "
                f"= {defer_years:.4f}y"
            )
            assumptions.append(
                f"Reversion income deferred a further "
                f"{(void_months or 0):g} void + {(rent_free_months or 0):g} "
                f"rent-free months beyond the review/expiry date, per supplied inputs."
            )
        else:
            assumptions.append(
                "No void or rent-free period modelled on reversion (none "
                "supplied) — reversion income assumed to commence immediately "
                "at review/expiry. This can overstate value where re-letting "
                "is required."
            )
        components = {
            "Term value (secure income to reversion)": round(term_value, 2),
            "Reversion value (market rent, deferred)":  round(reversion_value, 2),
        }
        method_reasoning = (
            f"Passing rent (£{passing_rent:,.0f}/yr) is below market rent "
            f"(£{market_rent:,.0f}/yr) with {n_years:g} years to the next "
            f"review/expiry — term & reversion applied: the secure term income "
            f"is capitalised to the reversion at {term_yield * 100:.2f}%, and "
            f"the market rent is capitalised in perpetuity, deferred "
            f"{defer_years:g} years, at {reversion_yld * 100:.2f}%."
        )
        equivalent_yield = _solve_yield_bisection(
            capital_value,
            lambda e: (
                (passing_rent * (_yp_years(n_years, e, q) or 0))
                + (market_rent * (_yp_perpetuity_deferred(defer_years, e, q) or 0))
            ),
        )
        waterfall += [
            {
                "label": (
                    f"Term: passing rent £{passing_rent:,.0f}/yr × YP "
                    f"{n_years:g} yrs @ {term_yield * 100:.2f}% ({yp_term:.4f})"
                ),
                "amount": round(term_value, 2),
            },
            {
                "label": (
                    f"Reversion: market rent £{market_rent:,.0f}/yr × YP perp "
                    f"deferred {defer_years:g} yrs @ {reversion_yld * 100:.2f}% ({yp_rev:.4f})"
                ),
                "amount": round(reversion_value, 2),
            },
        ]

    else:  # passing_rent > market_rent — over-rented
        method = "investment_method_hardcore_layer"
        yp_core = _yp_perpetuity(reversion_yld, q)
        yp_top  = _yp_years(n_years, top_yield, q)
        if yp_core is None or yp_top is None:
            evidence_gaps.append("Core or top-slice yield invalid — cannot compute hardcore/layer.")
            return _insufficient(fi, evidence_gaps, warnings, assumptions, formula_trace)
        core_value      = market_rent * yp_core
        top_slice_value = (passing_rent - market_rent) * yp_top
        capital_value   = core_value + top_slice_value
        formula_trace.append(
            f"hardcore_layer: core={market_rent}×YP_perp({reversion_yld:.4f})={core_value:.2f}; "
            f"top_slice=({passing_rent}-{market_rent})×YP_years({n_years}y,{top_yield:.4f})="
            f"{top_slice_value:.2f}"
        )
        warnings.append(
            "Over-rented property: top-slice income is not secure beyond the "
            "next review/expiry — treat with more caution than a rack-rented equivalent."
        )
        components = {
            "Core value (market rent, perpetuity)":  round(core_value, 2),
            "Top-slice value (rent above market)":    round(top_slice_value, 2),
        }
        method_reasoning = (
            f"Passing rent (£{passing_rent:,.0f}/yr) is above market rent "
            f"(£{market_rent:,.0f}/yr) with {n_years:g} years of the over-rent "
            f"remaining — hardcore/layer applied: the market-rent core is "
            f"capitalised in perpetuity at {reversion_yld * 100:.2f}%, and the "
            f"top slice above market for {n_years:g} years at "
            f"{top_yield * 100:.2f}%."
        )
        equivalent_yield = _solve_yield_bisection(
            capital_value,
            lambda e: (
                (market_rent * (_yp_perpetuity(e, q) or 0))
                + ((passing_rent - market_rent) * (_yp_years(n_years, e, q) or 0))
            ),
        )
        waterfall += [
            {
                "label": (
                    f"Core: market rent £{market_rent:,.0f}/yr × YP perpetuity "
                    f"@ {reversion_yld * 100:.2f}% ({yp_core:.4f})"
                ),
                "amount": round(core_value, 2),
            },
            {
                "label": (
                    f"Top slice: £{passing_rent - market_rent:,.0f}/yr above market "
                    f"× YP {n_years:g} yrs @ {top_yield * 100:.2f}% ({yp_top:.4f})"
                ),
                "amount": round(top_slice_value, 2),
            },
        ]
        if extra_defer_years > 0:
            warnings.append(
                "void_months / rent_free_months supplied but not applied — "
                "only modelled for the under-rented term & reversion case in "
                "this phase."
            )

    wault = fi.get("wault_years")
    try:
        wault = float(wault) if wault is not None else None
    except (TypeError, ValueError):
        wault = None

    if wault is not None and wault_to_break is not None and wault_to_break > wault:
        warnings.append(
            "WAULT to break exceeds WAULT to expiry — a break cannot fall "
            "after expiry; check the inputs."
        )

    # ── Gross capital value row, then purchaser's costs → net price ──────
    waterfall.append({
        "label": "Capital value (gross of purchaser's costs)",
        "amount": round(capital_value, 2),
        "emphasis": True,
    })

    purchasers_costs = _net_of_purchasers_costs(capital_value, purchaser_fees_pct, nation)
    net_value = None
    if purchasers_costs.get("status") == "ok":
        net_value = purchasers_costs["net_value_gbp"]
        formula_trace.append(
            f"purchasers_costs: net({net_value}) + sdlt({purchasers_costs['sdlt_gbp']}) "
            f"+ fees({purchasers_costs['fees_gbp']} @ {purchasers_costs['fees_pct']}%) "
            f"= gross({capital_value:.2f}) — bisection on England & NI "
            f"non-residential SDLT bands"
        )
        waterfall += [
            {
                "label": "SDLT (England & NI non-residential bands, on net price)",
                "amount": -purchasers_costs["sdlt_gbp"],
            },
            {
                "label": (
                    f"Purchaser's fees ({purchasers_costs['fees_pct']:g}% of net "
                    f"price — stated assumption)"
                ),
                "amount": -purchasers_costs["fees_gbp"],
            },
            {
                "label": "Indicative value (net of purchaser's costs)",
                "amount": net_value,
                "emphasis": True,
            },
        ]

    # ── Yields: NIY (rent ÷ gross-of-costs value), GIY (rent ÷ net price),
    #    equivalent yield (single yield reproducing the capital value) ────
    niy = round(passing_rent / capital_value * 100, 2) if capital_value > 0 else None
    giy = round(passing_rent / net_value * 100, 2) if net_value else None
    ry  = round(market_rent / capital_value * 100, 2) if capital_value > 0 else None
    yields = {
        "net_initial_yield_pct":   niy,
        "reversionary_yield_pct":  ry,
        "gross_initial_yield_pct": giy,
        "equivalent_yield_pct":    round(equivalent_yield * 100, 2) if equivalent_yield else None,
        "note": (
            "NIY = passing rent ÷ capital value gross of purchaser's costs; "
            "reversionary yield = market rent ÷ the same gross value (the "
            "institutional NIY/reversionary/equivalent trio); GIY = passing "
            "rent ÷ net price — the NIY–GIY gap is the purchaser's costs. "
            "Equivalent yield is the single yield that, applied to every "
            "slice of this cashflow, reproduces the capital value."
        ),
    }

    # ── v2.2: input-plausibility sanity warnings ─────────────────────────
    # Warnings only — never block, never alter the number. Thresholds are
    # engineering judgement for catching fat-finger errors, not market data.
    if passing_rent > 0 and market_rent > 0:
        rent_ratio = market_rent / passing_rent
        if rent_ratio >= RENT_RATIO_SANITY_MULTIPLE or rent_ratio <= 1 / RENT_RATIO_SANITY_MULTIPLE:
            warnings.append(
                f"Market rent is {rent_ratio:.1f}× passing rent — check for an "
                "input error (e.g. a misplaced zero). The valuation is computed "
                "honestly from the inputs as given; if the ratio is genuine, "
                "expect the reversion (or top slice) to dominate the result."
            )
    if niy is not None and (niy < NIY_SANITY_FLOOR_PCT or niy > NIY_SANITY_CEILING_PCT):
        warnings.append(
            f"Implied initial yield of {niy}% is far outside typically "
            "observed UK commercial trading ranges — check the rent and yield "
            f"inputs. (Sanity bounds {NIY_SANITY_FLOOR_PCT:g}–"
            f"{NIY_SANITY_CEILING_PCT:g}% are an engineering judgement for "
            "catching input errors, not market data.)"
        )

    # ── v2.2: yield sensitivity — ARGUS-style, same formula at shifted
    #    yields, showing how the output moves with the least-evidenced
    #    input (the user-supplied yield). Recomputation only, no new data. ─
    def _cv_at_shift(delta: float) -> Optional[float]:
        if method == "investment_method_rack_rented_perpetuity":
            yp_s = _yp_perpetuity(((yield_pct or term_yield) or 0) + delta, q)
            return passing_rent * yp_s if yp_s else None
        if method == "investment_method_term_and_reversion":
            t_s = _yp_years(n_years, term_yield + delta, q)
            r_s = _yp_perpetuity_deferred(defer_years, reversion_yld + delta, q)
            return passing_rent * t_s + market_rent * r_s if (t_s and r_s) else None
        c_s = _yp_perpetuity(reversion_yld + delta, q)
        p_s = _yp_years(n_years, top_yield + delta, q)
        return market_rent * c_s + (passing_rent - market_rent) * p_s if (c_s and p_s) else None

    sensitivity: list[dict] = []
    for bps in SENSITIVITY_SHIFTS_BPS:
        cv_s = _cv_at_shift(bps / 10000.0)
        if cv_s is None or cv_s <= 0:
            continue
        row = {"yield_shift_bps": bps, "gross_value_gbp": round(cv_s, 2), "net_value_gbp": None}
        if purchasers_costs.get("status") == "ok":
            pc_s = _net_of_purchasers_costs(cv_s, purchaser_fees_pct, nation)
            if pc_s.get("status") == "ok":
                row["net_value_gbp"] = pc_s["net_value_gbp"]
        sensitivity.append(row)
    sensitivity_note = (
        "Same formula recomputed with every slice yield shifted by the stated "
        "basis points — showing how sensitive the output is to the "
        "least-evidenced input (the user-supplied yield). ±25/50 bps are "
        "display steps, not a forecast."
    )

    # ── v2.2: uncertainty & limits statement — built from the ACTUAL state
    #    of this computation, never boilerplate: each line appears only when
    #    its condition is true on this deal. ──────────────────────────────
    unc_parts = [
        "Yield is user-supplied, not market-derived — no licensed benchmark "
        "integration exists in this phase, and the yield is the single input "
        "the output is most sensitive to (see sensitivity table).",
    ]
    if any("market_rent_pa not supplied" in a for a in assumptions):
        unc_parts.append("Market rent was assumed equal to passing rent, not evidenced.")
    if any("purchaser's fees defaulted" in a for a in assumptions):
        unc_parts.append("Purchaser's fees are a stated default assumption, not a quoted figure.")
    if any("No void or rent-free" in a for a in assumptions):
        unc_parts.append(
            "No void or rent-free period is modelled on the reversion, which "
            "can overstate value where re-letting is required."
        )
    unc_parts.append(
        "Income is capitalised on the nominal (annually in arrears) "
        "convention, not the true equivalent yield."
        if not q else
        "Income is capitalised on the true (quarterly in advance) "
        "convention — the entered yield must have been analysed on the same "
        "basis or the value will be overstated."
    )
    _prov = fi.get("_provenance") or {}
    _n_extracted = sum(
        1 for _k, _e in _prov.items()
        if isinstance(_e, dict) and _e.get("source") == "extracted"
    )
    if _n_extracted:
        unc_parts.append(
            f"{_n_extracted} factual input"
            f"{'s were' if _n_extracted != 1 else ' was'} read from the "
            "uploaded legal pack with citations; no inspection or covenant "
            "assessment has occurred, and the yield remains unverified "
            "(Evidence tier C)."
        )
    else:
        unc_parts.append(
            "No inspection, lease reading, or covenant assessment has occurred — "
            "all inputs are unverified (Evidence tier C)."
        )
    unc_parts.append(
        "Treat the figure as an indicative anchor, not a point of precision."
    )
    uncertainty_statement = " ".join(unc_parts)

    return _ok_result(
        valuation_type="commercial_investment_method",
        provenance=fi.get("_provenance"),
        method=method,
        inputs_used={
            "passing_rent_pa":      passing_rent,
            "market_rent_pa":       market_rent,
            "yield_pct":            round(yield_pct * 100, 3) if yield_pct else None,
            "term_yield_pct":       round(term_yield * 100, 3) if term_yield else None,
            "reversion_yield_pct":  round(reversion_yld * 100, 3) if reversion_yld else None,
            "top_slice_yield_pct":  round(top_yield * 100, 3) if top_yield else None,
            "unexpired_term_years": n_years,
            "wault_years":          wault,
            "wault_to_break_years": wault_to_break,
            "tenant_name":          fi.get("tenant_name"),
            "rent_review_basis":    review_basis,
            "nation":               nation,
            "purchaser_fees_pct":   purchaser_fees_pct,
            "void_months":          void_months,
            "rent_free_months":     rent_free_months,
            "tenure":               tenure or "freehold",
            "yield_basis":          yield_basis,
            "asset_class":          asset_class,
        },
        valuation_components=components,
        capital_value=capital_value,
        assumptions=assumptions, evidence_gaps=evidence_gaps, warnings=warnings, formula_trace=formula_trace,
        extra={
            "yield_convention":      YIELD_CONVENTION_QUARTERLY if q else YIELD_CONVENTION,
            "yield_convention_note": YIELD_CONVENTION_NOTE_QUARTERLY if q else YIELD_CONVENTION_NOTE,
            "method_reasoning":      method_reasoning,
            "waterfall":             waterfall,
            "purchasers_costs":      purchasers_costs,
            "yields":                yields,
            "sensitivity":           sensitivity,
            "sensitivity_note":      sensitivity_note,
            "uncertainty_statement": uncertainty_statement,
        },
    )


def _calculate_profits_method(fi: dict, asset_class: str) -> dict:
    """
    RICS Profits Method (VPGA 4) — for trade-related property: pubs,
    hotels, care homes, petrol stations, and similar assets where value
    derives from the business the property hosts, not from rent.

    Capital value = Fair Maintainable Operating Profit (FMOP) × a market
    multiplier (a Years'-Purchase-equivalent applied to maintainable
    profit for this trade sector). Both are REQUIRED user inputs — this
    engine does not fabricate a multiplier from sector rules-of-thumb; the
    person must supply one derived from comparable trading transactions.

    Expected keys in fi:
        fmop_pa           : float — Fair Maintainable Operating Profit (£/yr,
                            i.e. the "reasonably efficient operator" adjusted
                            net profit before rent/finance/depreciation). REQUIRED.
        profit_multiplier : float — capitalisation multiple applied to FMOP,
                            e.g. 6.5. REQUIRED.
        fmt_pa            : float — Fair Maintainable Turnover, optional,
                            informational only (not used in the maths).
        tenant_name       : str — optional, informational only.
    """
    assumptions: list[str] = []
    evidence_gaps: list[str] = []
    warnings: list[str] = []
    formula_trace: list[str] = []

    fmop = fi.get("fmop_pa")
    try:
        fmop = float(fmop) if fmop is not None else None
    except (TypeError, ValueError):
        fmop = None

    multiplier = fi.get("profit_multiplier")
    try:
        multiplier = float(multiplier) if multiplier is not None else None
    except (TypeError, ValueError):
        multiplier = None

    if fmop is None or fmop <= 0:
        evidence_gaps.append(
            "No Fair Maintainable Operating Profit (fmop_pa) supplied — the "
            "Profits Method requires the reasonably-efficient-operator adjusted "
            "net profit, which cannot be derived from rent or sold-price data."
        )
        return _insufficient(
            fi, evidence_gaps, warnings, assumptions, formula_trace,
            valuation_type="commercial_profits_method",
            inputs_used={"fmop_pa": fmop, "profit_multiplier": multiplier, "fmt_pa": fi.get("fmt_pa"), "tenant_name": fi.get("tenant_name"), "asset_class": asset_class},
        )

    if multiplier is None or multiplier <= 0:
        evidence_gaps.append(
            "No profit_multiplier supplied. This engine does not fabricate a "
            "sector rule-of-thumb multiplier — provide one derived from "
            "comparable trading transactions for this trade sector."
        )
        return _insufficient(
            fi, evidence_gaps, warnings, assumptions, formula_trace,
            valuation_type="commercial_profits_method",
            inputs_used={"fmop_pa": fmop, "profit_multiplier": multiplier, "fmt_pa": fi.get("fmt_pa"), "tenant_name": fi.get("tenant_name"), "asset_class": asset_class},
        )

    capital_value = fmop * multiplier
    formula_trace.append(
        f"profits_method: capital_value = FMOP({fmop}) × multiplier({multiplier})"
    )
    warnings.append(
        "Profits Method valuations assume the business is a going concern "
        "with adequate profitability — if trade has been materially "
        "disrupted or the business is not viable as a going concern, this "
        "figure may not be reliable (RICS UK VPGA 1.5)."
    )

    fmt = fi.get("fmt_pa")
    try:
        fmt = float(fmt) if fmt is not None else None
    except (TypeError, ValueError):
        fmt = None

    return _ok_result(
        valuation_type="commercial_profits_method",
        provenance=fi.get("_provenance"),
        method="profits_method_fmop_multiplier",
        inputs_used={
            "fmop_pa":           fmop,
            "profit_multiplier": multiplier,
            "fmt_pa":            fmt,
            "tenant_name":       fi.get("tenant_name"),
            "asset_class":       asset_class,
        },
        valuation_components={"Fair Maintainable Operating Profit (FMOP, £/yr)": round(fmop, 2)},
        capital_value=capital_value,
        assumptions=assumptions, evidence_gaps=evidence_gaps, warnings=warnings, formula_trace=formula_trace,
        extra={
            "method_reasoning": (
                "Asset class stated as trade-related — valued by the Profits "
                "Method: value derives from the maintainable profit of the "
                "business the property hosts, not from rent."
            ),
            "uncertainty_statement": (
                "FMOP and the profit multiplier are user-supplied and "
                "unverified — no trading accounts have been inspected and the "
                "multiplier has not been evidenced against comparable trading "
                "transactions. The figure assumes a viable going concern (see "
                "warnings). All inputs are unverified (Evidence tier C). Treat "
                "the figure as an indicative anchor, not a point of precision."
            ),
        },
    )


def _calculate_residual_method(fi: dict, asset_class: str) -> dict:
    """
    RICS Residual Method — for development / redevelopment sites.

    Residual site value = GDV − (build costs + professional fees +
    finance costs + contingency + developer's profit).

    Expected keys in fi:
        gdv                          : float — Gross Development Value, the
                                       completed scheme's value on the special
                                       assumption it is finished. REQUIRED.
        build_costs_gbp              : float — total construction cost. REQUIRED.
        professional_fees_gbp        : float — optional; OR
        professional_fees_pct_of_build: float — optional, percent of build_costs_gbp.
                                       If NEITHER is supplied, defaults to £0
                                       with an explicit assumption logged (this
                                       UNDERSTATES true cost — RICS practice
                                       normally includes fees; provide a figure).
        finance_cost_gbp             : float — optional direct override; OR
        interest_rate_pct + build_period_years: used to approximate finance
                                       cost via the standard average-drawdown
                                       convention: build_costs × rate × (period/2).
                                       If none of these are supplied, finance
                                       cost defaults to £0 with an assumption logged.
        contingency_gbp              : float — optional; OR
        contingency_pct_of_build     : float — optional. Defaults to £0 with
                                       an assumption logged if neither supplied.
        developer_profit_gbp         : float — optional; OR
        developer_profit_pct_of_gdv  : float — optional. REQUIRED (one of the
                                       two) — a developer's required return
                                       varies by deal and is never assumed.
    """
    assumptions: list[str] = []
    evidence_gaps: list[str] = []
    warnings: list[str] = []
    formula_trace: list[str] = []

    def _num(key):
        v = fi.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    gdv         = _num("gdv")
    build_costs = _num("build_costs_gbp")

    if gdv is None or gdv <= 0:
        evidence_gaps.append(
            "No gdv (Gross Development Value) supplied — the Residual Method "
            "requires the completed scheme's value, which is not derivable "
            "from residential sold-price comparables for a development site."
        )
        return _insufficient(
            fi, evidence_gaps, warnings, assumptions, formula_trace,
            valuation_type="commercial_residual_method",
            inputs_used={"gdv": gdv, "build_costs_gbp": build_costs, "asset_class": asset_class},
        )

    if build_costs is None or build_costs <= 0:
        evidence_gaps.append(
            "No build_costs_gbp supplied — the Residual Method requires "
            "total construction cost; this engine does not fabricate a "
            "cost from a licensed cost database (e.g. BCIS)."
        )
        return _insufficient(
            fi, evidence_gaps, warnings, assumptions, formula_trace,
            valuation_type="commercial_residual_method",
            inputs_used={"gdv": gdv, "build_costs_gbp": build_costs, "asset_class": asset_class},
        )

    # Professional fees
    prof_fees = _num("professional_fees_gbp")
    if prof_fees is None:
        prof_fees_pct = _num("professional_fees_pct_of_build")
        if prof_fees_pct is not None:
            prof_fees = build_costs * (prof_fees_pct / 100)
            formula_trace.append(f"professional_fees = build_costs × {prof_fees_pct}% = {prof_fees:.2f}")
        else:
            prof_fees = 0.0
            assumptions.append(
                "professional_fees not supplied — assumed £0. This understates "
                "true cost; RICS practice normally includes survey/legal/planning "
                "fees as a proportion of build cost — provide your own figure."
            )

    # Finance costs
    finance_cost = _num("finance_cost_gbp")
    if finance_cost is None:
        rate = _pct_to_decimal(fi.get("interest_rate_pct"))
        period = _num("build_period_years")
        if rate is not None and period is not None and period > 0:
            finance_cost = build_costs * rate * (period / 2)
            formula_trace.append(
                f"finance_cost ≈ build_costs({build_costs}) × rate({rate:.4f}) × "
                f"(period({period})/2) — standard average-drawdown approximation, "
                f"not an exact drawdown schedule = {finance_cost:.2f}"
            )
            assumptions.append(
                "finance_cost computed via the standard average-drawdown "
                "approximation (half the build period at the full facility "
                "balance) — not an exact drawdown schedule."
            )
        else:
            finance_cost = 0.0
            assumptions.append(
                "finance_cost not supplied (no finance_cost_gbp or "
                "interest_rate_pct+build_period_years) — assumed £0. This "
                "understates true cost for any funded scheme."
            )

    # Contingency
    contingency = _num("contingency_gbp")
    if contingency is None:
        contingency_pct = _num("contingency_pct_of_build")
        if contingency_pct is not None:
            contingency = build_costs * (contingency_pct / 100)
            formula_trace.append(f"contingency = build_costs × {contingency_pct}% = {contingency:.2f}")
        else:
            contingency = 0.0
            assumptions.append("contingency not supplied — assumed £0.")

    # Developer's profit — REQUIRED, never assumed
    dev_profit = _num("developer_profit_gbp")
    if dev_profit is None:
        dev_profit_pct = _num("developer_profit_pct_of_gdv")
        if dev_profit_pct is not None:
            dev_profit = gdv * (dev_profit_pct / 100)
            formula_trace.append(f"developer_profit = gdv × {dev_profit_pct}% = {dev_profit:.2f}")
        else:
            evidence_gaps.append(
                "No developer_profit_gbp or developer_profit_pct_of_gdv "
                "supplied. This engine does not assume a standard developer's "
                "profit margin — required return varies by deal; provide one."
            )
            return _insufficient(
                fi, evidence_gaps, warnings, assumptions, formula_trace,
                valuation_type="commercial_residual_method",
            inputs_used={"gdv": gdv, "build_costs_gbp": build_costs, "developer_profit_gbp": dev_profit, "asset_class": asset_class},
            )

    total_costs = build_costs + prof_fees + finance_cost + contingency + dev_profit
    residual_value = gdv - total_costs
    formula_trace.append(
        f"residual_method: residual_value = gdv({gdv}) − "
        f"(build_costs({build_costs}) + fees({prof_fees:.2f}) + "
        f"finance({finance_cost:.2f}) + contingency({contingency:.2f}) + "
        f"developer_profit({dev_profit:.2f})) = {residual_value:.2f}"
    )
    if residual_value < 0:
        warnings.append(
            "Residual value is negative — on these inputs the scheme does "
            "not cover its costs and developer's profit at this GDV."
        )

    return _ok_result(
        valuation_type="commercial_residual_method",
        provenance=fi.get("_provenance"),
        method="residual_method_development_appraisal",
        inputs_used={
            "gdv": gdv, "build_costs_gbp": build_costs,
            "professional_fees_gbp": round(prof_fees, 2),
            "finance_cost_gbp": round(finance_cost, 2),
            "contingency_gbp": round(contingency, 2),
            "developer_profit_gbp": round(dev_profit, 2),
            "asset_class": asset_class,
        },
        valuation_components={
            "Build costs":         round(build_costs, 2),
            "Professional fees":   round(prof_fees, 2),
            "Finance costs":       round(finance_cost, 2),
            "Contingency":         round(contingency, 2),
            "Developer's profit":  round(dev_profit, 2),
            "Total costs":         round(total_costs, 2),
        },
        capital_value=residual_value,
        extra={
            "method_reasoning": (
                "Asset class stated as a development/redevelopment site — "
                "valued by the Residual Method: completed value (GDV) less all "
                "development costs and developer's profit leaves the residual "
                "site value."
            ),
            "uncertainty_statement": (
                "GDV and cost inputs are user-supplied and unverified — no "
                "cost-database (e.g. BCIS) or planning verification has "
                "occurred, and residual outputs are highly sensitive to small "
                "changes in GDV and build cost (see the assumptions list for "
                "any cost element defaulted to £0, which understates true "
                "cost). All inputs are unverified (Evidence tier C). Treat the "
                "figure as an indicative anchor, not a point of precision."
            ),
        },
        assumptions=assumptions, evidence_gaps=evidence_gaps, warnings=warnings, formula_trace=formula_trace,
    )


def _calculate_drc_method(fi: dict, asset_class: str) -> dict:
    """
    RICS Depreciated Replacement Cost / Contractor's Method — for
    specialised owner-occupied property rarely sold on the open market
    (schools, hospitals, refineries, specialist industrial).

    Capital value = land_value + gross_replacement_cost × (1 − depreciation_pct/100)

    Expected keys in fi:
        land_value_gbp           : float — REQUIRED.
        gross_replacement_cost_gbp: float — cost of reinstating the building. REQUIRED.
        depreciation_pct          : float — percent deducted for age/condition/
                                    obsolescence, 0–100. REQUIRED (0 is a valid
                                    explicit value but must be stated, never assumed).
    """
    assumptions: list[str] = []
    evidence_gaps: list[str] = []
    warnings: list[str] = []
    formula_trace: list[str] = []

    def _num(key):
        v = fi.get(key)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    land_value = _num("land_value_gbp")
    grc        = _num("gross_replacement_cost_gbp")
    dep_pct    = fi.get("depreciation_pct")
    try:
        dep_pct = float(dep_pct) if dep_pct is not None else None
    except (TypeError, ValueError):
        dep_pct = None

    if land_value is None or land_value < 0:
        evidence_gaps.append(
            "No land_value_gbp supplied — DRC requires a land value, which "
            "is not derivable from residential sold-price comparables for a "
            "specialised, rarely-transacted asset."
        )
        return _insufficient(
            fi, evidence_gaps, warnings, assumptions, formula_trace,
            valuation_type="commercial_drc_method",
            inputs_used={"land_value_gbp": land_value, "gross_replacement_cost_gbp": grc, "depreciation_pct": dep_pct, "asset_class": asset_class},
        )

    if grc is None or grc <= 0:
        evidence_gaps.append(
            "No gross_replacement_cost_gbp supplied — this engine does not "
            "fabricate a rebuild cost from a licensed cost database (e.g. BCIS)."
        )
        return _insufficient(
            fi, evidence_gaps, warnings, assumptions, formula_trace,
            valuation_type="commercial_drc_method",
            inputs_used={"land_value_gbp": land_value, "gross_replacement_cost_gbp": grc, "depreciation_pct": dep_pct, "asset_class": asset_class},
        )

    if dep_pct is None:
        evidence_gaps.append(
            "No depreciation_pct supplied. This engine never assumes 0% "
            "depreciation by default — every building has some age/condition/ "
            "obsolescence deduction; state the figure explicitly (0 is a "
            "valid value if genuinely justified, but must be supplied)."
        )
        return _insufficient(
            fi, evidence_gaps, warnings, assumptions, formula_trace,
            valuation_type="commercial_drc_method",
            inputs_used={"land_value_gbp": land_value, "gross_replacement_cost_gbp": grc, "depreciation_pct": dep_pct, "asset_class": asset_class},
        )

    depreciated_grc = grc * (1 - dep_pct / 100)
    capital_value = land_value + depreciated_grc
    formula_trace.append(
        f"drc_method: capital_value = land_value({land_value}) + "
        f"grc({grc}) × (1 − {dep_pct}%) = {land_value} + {depreciated_grc:.2f} = {capital_value:.2f}"
    )
    warnings.append(
        "DRC valuations for an operational/trading entity are reported "
        "subject to the adequate profitability of the business (RICS UK "
        "VPGA 1.5) — this figure assumes the business is a viable going "
        "concern; if not, this may materially overstate value."
    )

    return _ok_result(
        valuation_type="commercial_drc_method",
        provenance=fi.get("_provenance"),
        method="drc_contractors_method",
        inputs_used={
            "land_value_gbp": land_value, "gross_replacement_cost_gbp": grc,
            "depreciation_pct": dep_pct, "asset_class": asset_class,
        },
        valuation_components={
            "Land value":                    round(land_value, 2),
            "Gross replacement cost":        round(grc, 2),
            "Depreciated replacement cost":  round(depreciated_grc, 2),
        },
        capital_value=capital_value,
        assumptions=assumptions, evidence_gaps=evidence_gaps, warnings=warnings, formula_trace=formula_trace,
        extra={
            "method_reasoning": (
                "Asset class stated as specialised owner-occupied — valued by "
                "Depreciated Replacement Cost: land value plus reinstatement "
                "cost less depreciation, used where no rental or sales market "
                "exists for the asset."
            ),
            "uncertainty_statement": (
                "Land value, replacement cost, and depreciation are "
                "user-supplied and unverified — no cost-database (e.g. BCIS) or "
                "land-comparable verification has occurred, and the figure is "
                "subject to the adequate-profitability condition in the "
                "warnings. All inputs are unverified (Evidence tier C). Treat "
                "the figure as an indicative anchor, not a point of precision."
            ),
        },
    )


def _insufficient(
    fi: dict,
    evidence_gaps: list[str],
    warnings: list[str],
    assumptions: list[str],
    formula_trace: list[str],
    status: str = "insufficient_evidence",
    valuation_type: str = "commercial_investment_method",
    inputs_used: Optional[dict] = None,
) -> dict:
    """Shared insufficient-evidence return — same top-level shape as the
    success path so callers never need to special-case a short/long schema.
    status defaults to "insufficient_evidence" (missing data) but can be
    overridden to "manual_review_required" (asset-class scope gate — wrong
    method, not missing data).

    inputs_used defaults to the Investment Method's field set for backward
    compatibility with existing calls from _calculate_investment_method.
    Profits/Residual/DRC pass their own inputs_used dict so a person who
    entered gdv/build_costs (say) sees THOSE fields echoed back, not
    passing_rent_pa/yield_pct which they never touched."""
    if inputs_used is None:
        inputs_used = {
            "passing_rent_pa":      fi.get("passing_rent_pa"),
            "market_rent_pa":       fi.get("market_rent_pa"),
            "yield_pct":            fi.get("yield_pct"),
            "term_yield_pct":       fi.get("term_yield_pct"),
            "reversion_yield_pct":  fi.get("reversion_yield_pct"),
            "top_slice_yield_pct":  fi.get("top_slice_yield_pct"),
            "unexpired_term_years": fi.get("unexpired_term_years"),
            "wault_years":          fi.get("wault_years"),
            "wault_to_break_years": fi.get("wault_to_break_years"),
            "tenant_name":          fi.get("tenant_name"),
            "rent_review_basis":    fi.get("rent_review_basis"),
            "nation":               fi.get("nation"),
            "purchaser_fees_pct":   fi.get("purchaser_fees_pct"),
            "void_months":          fi.get("void_months"),
            "rent_free_months":     fi.get("rent_free_months"),
            "tenure":               fi.get("tenure"),
            "yield_basis":          fi.get("yield_basis"),
            "asset_class":          str(fi.get("asset_class") or ASSET_CLASS_INCOME_PRODUCING_LET).strip().lower(),
        }
    return {
        "valuation_type":     valuation_type,
        "not_rics_valuation": True,
        "status":             status,
        "method":              None,
        "inputs_used":        inputs_used,
        "valuation_components": {},
        "comparable_valuation": None,
        "risk_adjusted_value":  None,
        "valuation_range": {"low": None, "midpoint": None, "high": None, "uncertainty_band": None},
        "confidence": {"raw": None, "caps": [], "final": None, "label": None},
        "audit": {
            "comparable_method":      None,
            "not_rics_valuation":     True,
            "formal_valuation":       False,
            "decision_support_only":  True,
            "source_decision":        status,
            "fallback_used":          True,
            "assumptions":            assumptions,
            "evidence_gaps":          evidence_gaps,
            "warnings":               warnings,
            "formula_trace":          formula_trace,
            "version":                VERSION,
        },
    }
