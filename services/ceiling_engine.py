"""
ceiling_engine.py — LegalSmegal Ceiling Engine v2.0
====================================================
LegalSmegal comparable valuation ceiling engine.


NOT a RICS Red Book valuation.
NOT legal advice.
NOT financial advice.
NOT acquisition underwriting.

Formula
-------
base_value =
    weighted_median(adjusted_value_i, weight_i)
    from sold comparables within 0.5 miles only.

improved_ceiling_midpoint =
    base_value × legal_pack_value_risk_adjustment_factor

legal_pack_value_risk_adjustment_factor =
    product(1 − value_adjustment_i)   [for each included legal-pack value risk]

ceiling_low  = improved_ceiling_midpoint × (1 − uncertainty_band)
ceiling_high = improved_ceiling_midpoint × (1 + uncertainty_band)

Acquisition costs (SDLT, buyer premium, legal fees, bridging) are
EXCLUDED from the ceiling valuation formula entirely. They are surfaced
separately in the acquisition_costs informational block only.

Architecture
------------
- Backend owns canonical ceiling object.
- Frontend must not mutate ceiling_range values.
- Arithmetic mean is NOT the primary base.
- Primary comparable universe: distance_miles <= 0.5.
"""

from __future__ import annotations
import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)

VERSION = "ceiling_relational_paper_valuation_v1"

# =============================================================================
# DISTANCE RULE
# =============================================================================
# S33-STEP1 (2026-06-21): PRIMARY_RADIUS_MILES retained as the band used for
# the "primary" comp population (drives confidence labelling, NOT inclusion).
# Inclusion is no longer a hard cutoff at 0.5mi — see EXTENDED_DISTANCE_BANDS
# and _distance_score() below. This was changed because a live audit (Hey
# Street, NG10 3HA, 2026-06-20) found type/room/age-band-matched comps at
# 0.82-1.27 miles being excluded outright by the old hard cutoff, leaving a
# single comp to anchor the entire valuation. Per RICS "Comparable Evidence
# in Real Estate Valuation" (2019/2023), comparables should be "comprehensive
# — several rather than a single transaction" and thin evidence should be
# addressed by looking further afield, not accepted as n=1.
PRIMARY_RADIUS_MILES = 0.5       # still used for confidence labelling only
MAX_RADIUS_MILES     = 3.0       # absolute outer bound — never search beyond this
MIN_REQUIRED_COMPS   = 3
PREFERRED_COMPS      = 5

# =============================================================================
# EVIDENCE TIER (S33-STEP3, 2026-06-21)
# =============================================================================
# WHY: the subject property has not yet transacted — every deal on this
# platform is a forthcoming auction lot, not a completed sale. Land Registry
# Price Paid Data contains NO field identifying which comparable SALES were
# themselves auction sales — PPD's only channel signal is ppd_category_type
# (A = standard market sale; B = repossession/power-of-sale/buy-to-let/
# corporate transfer — a blended, non-auction-specific category per HMLR's
# own documentation). Per RICS "Comparable Evidence in Real Estate
# Valuation," sale price evidence from a forced/distressed disposal is a
# different basis of value to open-market evidence and must not be blended
# into one pool.
#
# This platform will obtain real EIG (Essential Information Group) auction
# results data in production — genuine hammer prices at known addresses,
# the closest available analogue to "what will THIS property actually
# fetch at auction." EIG_ENABLED is False until that feed is connected;
# the tier exists now so connecting it later requires no further engine
# changes — only setting EIG_ENABLED=True and supplying real records via
# get_eig_comps_for_postcode().
EIG_ENABLED = False  # flip to True once the EIG feed is live

EVIDENCE_TIER_EIG_AUCTION   = "eig_auction_hammer_price"     # Tier 1 — not yet connected
EVIDENCE_TIER_PPD_CATEGORY_B = "ppd_category_b_distressed"    # Tier 2 — repossession/power-of-sale proxy
EVIDENCE_TIER_PPD_CATEGORY_A = "ppd_category_a_open_market"   # Tier 3 — open-market reference only

def get_eig_comps_for_postcode(postcode: str, radius_miles: float) -> list[dict]:
    """
    Placeholder for the EIG auction-results feed (S33-STEP3). Returns an
    empty list until EIG_ENABLED is True and a real data source is wired in.
    Expected real-record shape, once connected (per EIG's documented public
    fields): address, postcode, hammer_price, guide_price, sale_date,
    auction_house, lot_status ('sold'/'withdrawn'/'unsold'), property_type.
    DO NOT populate this with estimated/inferred values — only real EIG
    records once the feed exists. An empty list here is the correct,
    honest state until that integration is built.
    """
    if not EIG_ENABLED:
        return []
    # TODO(S33-STEP3-EIG): real EIG API/data integration goes here.
    raise NotImplementedError(
        "EIG_ENABLED is True but get_eig_comps_for_postcode() has no real "
        "data source wired in yet. Do not stub this with fake data."
    )

# =============================================================================
# WEIGHT CALIBRATION (deterministic)
# =============================================================================
# type_match_score
TYPE_SAME        = 1.00
TYPE_NEAR_EQUIV  = 0.75
TYPE_MISMATCH    = None   # exclude

# tenure_match_score
TENURE_SAME      = 1.00
TENURE_UNKNOWN   = 0.50   # + confidence cap
TENURE_DIFFERENT = None   # exclude unless audited degraded

# lease_match_score
LEASE_SAME_BAND  = 1.00
LEASE_ADJ_BAND   = 0.70
LEASE_UNKNOWN    = 0.40   # + confidence cap
LEASE_NON_ADJ    = None   # exclude

# distance_score (miles)
# S33-STEP1: extended beyond the old hard 0.5mi cutoff. The 0-0.5mi shape is
# UNCHANGED from the original DISTANCE_BANDS (1.00 / 0.90 / 0.80) — only the
# tail beyond 0.5mi is new, continuing the same decay pattern outward to
# MAX_RADIUS_MILES rather than excluding outright. A comp at 0.51mi is not
# meaningfully different from one at 0.50mi; a hard cutoff there was always
# arbitrary precision, not a real distinction.
EXTENDED_DISTANCE_BANDS = [
    (0.00, 0.10, 1.00),
    (0.10, 0.25, 0.90),
    (0.25, 0.50, 0.80),
    (0.50, 1.00, 0.65),
    (1.00, 1.50, 0.50),
    (1.50, 2.00, 0.35),
    (2.00, 3.00, 0.20),
]
# Backward-compat alias — some call sites may still reference DISTANCE_BANDS
# directly for the primary-band confidence check (0-0.5mi only).
DISTANCE_BANDS = [b for b in EXTENDED_DISTANCE_BANDS if b[1] <= 0.50]

# recency_score (months)
RECENCY_BANDS = [
    (0,   3,  1.00),
    (3,   6,  0.90),
    (6,  12,  0.80),
    (12, 24,  0.60),
    (24, 999, 0.40),
]

# size_score (area ratio subject/comp)
SIZE_BANDS = [
    (0.90, 1.10, 1.00),
    (0.80, 0.90, 0.80),
    (1.10, 1.25, 0.80),
    (0.75, 0.80, 0.60),
    (1.25, 1.33, 0.60),
]
SIZE_OUTER_SCORE = 0.40
SIZE_ADJ_CAP_LO  = 0.80
SIZE_ADJ_CAP_HI  = 1.25

# evidence_quality_score
EVIDENCE_OFFICIAL         = 1.00
EVIDENCE_PARTIAL          = 0.75
EVIDENCE_WEAK             = 0.50
EVIDENCE_UNCORROBORATED   = 0.25

# =============================================================================
# ADJUSTMENT CALIBRATION (deterministic)
# =============================================================================
# condition_adjustment
CONDITION_SAME_OR_UNKNOWN = 1.00
CONDITION_COMP_BETTER     = 0.95
CONDITION_COMP_WORSE      = 1.05

# tenure_adjustment (applied to adjusted_value if degraded and audited)
TENURE_ADJ_DIFFERENT      = None  # excluded by default

# lease_adjustment
LEASE_ADJ_SAME_BAND   = 1.00
LEASE_ADJ_ADJ_BAND    = 0.95
LEASE_ADJ_NON_ADJ     = None  # excluded

# =============================================================================
# LEASE BANDS (deterministic)
# =============================================================================
LEASE_BANDS = [(0, 10), (10, 20), (20, 40), (40, 60), (60, 80), (80, 99), (99, 9999)]

def _lease_band(years: Optional[float]) -> Optional[int]:
    """Return band index (0-based) for a lease length, or None."""
    if years is None:
        return None
    for i, (lo, hi) in enumerate(LEASE_BANDS):
        if lo <= years < hi:
            return i
    return len(LEASE_BANDS) - 1  # 99+

def _bands_adjacent(a: Optional[int], b: Optional[int]) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) == 1

# =============================================================================
# LEGAL-PACK VALUE RISK CALIBRATION (deterministic)
# =============================================================================
# Only property-value risks — NO acquisition costs.
VALUE_RISK_SEVERITY_ADJ: dict[str, float] = {
    "critical":         0.10,
    "high":             0.06,
    "medium":           0.035,
    "low":              0.015,
    "note":             0.00,
    "missing":          0.025,  # missing but material
}
MAX_TOTAL_VALUE_RISK_ADJ = 0.35

VALUE_RISK_CATEGORIES = {
    "short_lease", "defective_lease", "defective_title", "missing_title",
    "missing_lease", "missing_management_pack", "restrictive_covenant",
    "rights_of_way", "easement_defect", "planning_issue",
    "building_control_issue", "service_charge_burden", "ground_rent_burden",
    "forfeiture_risk", "lenderability_defect", "resale_impairment",
}

# Excluded from ceiling valuation
EXCLUDED_FROM_CEILING = [
    "sdlt", "buyer_premium", "auction_admin_fee", "buyer_solicitor_fee",
    "finance_cost", "bridging_cost", "investor_margin", "execution_buffer",
    "generic_acquisition_costs",
]

EXCLUDE_CATEGORY_KEYWORDS = {
    "financial": ["buyer's premium", "buyers premium", "administration fee",
                  "sdlt", "stamp duty", "legal fee", "solicitor fee", "bridging",
                  "finance cost", "investor margin"],
}

def _is_acquisition_cost_only(flag: dict) -> bool:
    """Return True if flag is purely an acquisition cost — excluded from ceiling."""
    text = " ".join(filter(None, [
        flag.get("title", ""), flag.get("summation", ""),
        flag.get("implication", ""), flag.get("category", ""),
    ])).lower()
    acq_terms = ["buyer's premium", "buyers premium", "auction admin",
                  "sdlt", "stamp duty", "legal fee", "solicitor fee",
                  "bridging cost", "finance cost", "administration fee",
                  "non-refundable deposit"]
    return any(t in text for t in acq_terms)

def _value_risk_adjustment(flag: dict) -> float:
    """
    DEPRECATED — kept for backward compat with calculate_ceiling() (used by
    calculate_verdict_ceiling which passes no flags, so this is never called
    with real flags in the Verdict path).
    For the Workbench path use _flag_to_segments() instead.
    """
    sev = (flag.get("severity") or "note").lower().strip()
    return VALUE_RISK_SEVERITY_ADJ.get(sev, 0.00)


# =============================================================================
# MARKET-CONSEQUENCE SEGMENT ROUTING
# =============================================================================
# Five segments — each flag is routed to one or more segments.
# Segment amounts are cash amounts derived from comparable_valuation × factor.
# Fractions below are the adjustment proportions applied to comparable_valuation.
#
# Signal keywords used to route a flag (order-sensitive; first match wins for
# the dominant segment; non-dominant segments may also fire).
#
# KEY DESIGN RULE:
#   severity is kept as a DESCRIPTIVE LABEL only — it informs confidence, not pricing.
#   Pricing comes from the nature of the defect (cure / delay / insurance / lender / resale).

# Routing table: list of (signal_keywords, segment_fractions_dict)
# signal_keywords: substring match against flag text (title + summation + implication + category)
# segment_fractions_dict: {segment_name: fraction_of_comparable_valuation}
#   fractions must total ≤ MAX_TOTAL_VALUE_RISK_ADJ across ALL active flags (enforced in aggregate)

_SEGMENT_RULES: list[tuple[list[str], dict]] = [
    # ── Document retrieval only — low cure cost, minimal residual ────────────
    (["hmlr", "land registry", "office copy", "title copy", "title register copy",
      "title plan copy", "filed copy"],
     {"direct_cure_cost": 0.005, "delay_finance_drag": 0.005}),

    # ── Missing searches — cure cost + timing/delay ──────────────────────────
    (["local authority search", "local search", "drainage search", "environmental search",
      "water search", "chancel search", "coal search", "mining search",
      "missing search", "no search"],
     {"direct_cure_cost": 0.008, "delay_finance_drag": 0.012}),

    # ── Management pack / service charge info missing ────────────────────────
    (["management pack", "service charge", "managing agent", "maintenance charge",
      "ground rent information", "missing management"],
     {"direct_cure_cost": 0.006, "delay_finance_drag": 0.010, "lender_certifiability_risk": 0.008}),

    # ── Short or escalating ground rent — lender + resale ───────────────────
    (["ground rent", "escalating ground rent", "doubling ground rent", "onerous ground rent"],
     {"lender_certifiability_risk": 0.040, "residual_marketability_risk": 0.030}),

    # ── Short lease ──────────────────────────────────────────────────────────
    (["short lease", "lease extension", "lease below", "unexpired lease",
      "fewer than 80 years", "less than 80 years"],
     {"lender_certifiability_risk": 0.060, "residual_marketability_risk": 0.040}),

    # ── Forfeiture / breach of covenant ─────────────────────────────────────
    (["forfeiture", "breach of covenant", "breach of lease"],
     {"indemnity_insurance_cost": 0.010, "lender_certifiability_risk": 0.025,
      "residual_marketability_risk": 0.015}),

    # ── Defective / missing original lease ──────────────────────────────────
    (["defective lease", "missing lease", "missing original lease", "no lease", "lease not available",
      "lease missing"],
     {"lender_certifiability_risk": 0.050, "residual_marketability_risk": 0.035}),

    # ── Defective / missing title — unregistered / possessory ───────────────
    (["defective title", "missing title", "unregistered title", "possessory title",
      "title defect", "title not registered", "good root of title"],
     {"lender_certifiability_risk": 0.055, "residual_marketability_risk": 0.045}),

    # ── Indemnity available — reduces but does not eliminate resale risk ──────
    (["indemnity insurance", "indemnity policy", "title indemnity", "insurance policy available",
      "indemnity available"],
     {"indemnity_insurance_cost": 0.012, "residual_marketability_risk": 0.008}),

    # ── Restrictive covenant ─────────────────────────────────────────────────
    (["restrictive covenant", "restrictive covenants"],
     {"indemnity_insurance_cost": 0.010, "lender_certifiability_risk": 0.020,
      "residual_marketability_risk": 0.020}),

    # ── Rights of way / easement defects ─────────────────────────────────────
    (["right of way", "rights of way", "easement", "right to light", "access rights",
      "ransom strip"],
     {"lender_certifiability_risk": 0.020, "residual_marketability_risk": 0.025}),

    # ── Planning / building control ───────────────────────────────────────────
    (["planning", "building regulations", "building control", "planning permission",
      "listed building", "conservation area", "enforcement notice"],
     {"indemnity_insurance_cost": 0.012, "lender_certifiability_risk": 0.018,
      "residual_marketability_risk": 0.020}),

    # ── Section 20 / major works notice ──────────────────────────────────────
    (["section 20", "s20", "major works", "major work notice"],
     {"delay_finance_drag": 0.015, "residual_marketability_risk": 0.020}),

    # ── Arrears / rent / service charge debt ─────────────────────────────────
    (["arrears", "rent arrears", "service charge arrears", "maintenance arrears",
      "outstanding service charge"],
     {"direct_cure_cost": 0.015, "delay_finance_drag": 0.010}),

    # ── Flying freehold ───────────────────────────────────────────────────────
    (["flying freehold"],
     {"lender_certifiability_risk": 0.030, "residual_marketability_risk": 0.025}),

    # ── Contamination / environmental ────────────────────────────────────────
    (["contamination", "environmental risk", "flood risk", "subsidence"],
     {"lender_certifiability_risk": 0.025, "residual_marketability_risk": 0.035}),
]

# Severity zero-gate: "note" flags carry no market consequence.
# Severity does NOT scale matched route fractions — route type is the pricing truth.
_SEVERITY_NOTE_ONLY = {"note"}

# Unmatched-fallback amounts: used only when no _SEGMENT_RULES entry matches.
# These are the ONLY place severity influences a cash amount.
_FALLBACK_RESIDUAL_FRACTION: dict[str, float] = {
    "critical": 0.025,
    "high":     0.015,
    "medium":   0.008,
    "low":      0.004,
    "missing":  0.010,
}

def _flag_to_segments(flag: dict) -> dict[str, float]:
    """
    Route a single flag to one or more market-consequence segments.
    Returns {segment_name: fraction_of_comparable_valuation} for that flag.

    Pricing rules:
      - Route type is the primary pricing input (matched from _SEGMENT_RULES).
      - Severity is NOT a multiplier on matched route fractions.
      - Severity gates "note" flags to zero (notes have no market consequence).
      - Severity sets the fallback fraction ONLY when no route matches
        (there is no route-specific calibration to use in that case).

    Logic:
      1. Gate: note severity → {} (no adjustment).
      2. Build search text; find first matching rule in _SEGMENT_RULES.
      3. If matched: return rule fractions unchanged. Severity is not applied.
      4. If unmatched: return conservative residual_marketability_risk
         from _FALLBACK_RESIDUAL_FRACTION[severity].
    """
    sev = (flag.get("severity") or "note").lower().strip()

    # Gate: notes carry no market-consequence adjustment
    if sev in _SEVERITY_NOTE_ONLY:
        return {}

    text = " ".join(filter(None, [
        flag.get("title", ""),
        flag.get("summation", ""),
        flag.get("implication", ""),
        flag.get("category", ""),
    ])).lower()

    # Match first applicable rule — route type is truth, severity not applied
    for keywords, fractions in _SEGMENT_RULES:
        if any(kw in text for kw in keywords):
            # Return rule fractions as-is. Do not multiply by severity.
            return {seg: round(frac, 6) for seg, frac in fractions.items()}

    # No route matched — severity-calibrated fallback (only allowed use of severity in pricing)
    fallback_frac = _FALLBACK_RESIDUAL_FRACTION.get(sev, 0.010)
    return {"residual_marketability_risk": round(fallback_frac, 6)}


def _build_market_consequence_adjustments(
    risks: list[dict],
    comparable_valuation: float,
    total_adjustment: Optional[float] = None,
) -> dict:
    """
    Aggregate per-flag segment fractions into cash amounts.
    Returns market_consequence_adjustments dict with five segments.

    total_adjustment is the canonical figure (comparable_valuation - risk_adjusted_value)
    from the product formula.  Segment amounts are scaled so they sum exactly to
    total_adjustment; the last non-zero segment absorbs any integer rounding residual.

    If total_adjustment is None or zero (all flags resolved), all amounts are zero.

    Each segment entry:
      {amount: int, items: [{flag_title, segment, fraction, amount, confidence}]}
    """
    SEGMENTS = [
        "direct_cure_cost",
        "delay_finance_drag",
        "indemnity_insurance_cost",
        "lender_certifiability_risk",
        "residual_marketability_risk",
    ]

    # Build raw proportional amounts from segment fractions × comparable_valuation
    raw_buckets: dict[str, list[dict]] = {s: [] for s in SEGMENTS}

    for r in risks:
        segs = r.get("segments", {})
        for seg, frac in segs.items():
            if seg not in raw_buckets:
                continue
            raw_amt = comparable_valuation * frac
            raw_buckets[seg].append({
                "flag_title":  r.get("title", ""),
                "segment":     seg,
                "fraction":    round(frac, 6),
                "_raw_amount": raw_amt,
                "confidence":  r.get("confidence"),
                "reason":      r.get("reason", ""),
            })

    # Sum raw amounts per segment
    seg_raw_totals: dict[str, float] = {
        seg: sum(i["_raw_amount"] for i in raw_buckets[seg])
        for seg in SEGMENTS
    }
    grand_raw = sum(seg_raw_totals.values())

    # Scale so segments sum exactly to total_adjustment
    canonical_total = round(total_adjustment or 0.0)
    active_segs = [s for s in SEGMENTS if seg_raw_totals[s] > 0]

    scaled_amounts: dict[str, int] = {}
    if canonical_total > 0 and grand_raw > 0:
        allocated = 0
        for idx, seg in enumerate(active_segs):
            if idx == len(active_segs) - 1:
                # Last active segment absorbs rounding residual
                scaled_amounts[seg] = canonical_total - allocated
            else:
                amt = round(canonical_total * seg_raw_totals[seg] / grand_raw)
                scaled_amounts[seg] = amt
                allocated += amt
    else:
        for seg in active_segs:
            scaled_amounts[seg] = 0

    # Build final result — scale individual item amounts proportionally within each segment
    result: dict[str, dict] = {}
    for seg in SEGMENTS:
        seg_total_scaled = scaled_amounts.get(seg, 0)
        items_raw = raw_buckets[seg]
        seg_raw = seg_raw_totals[seg]
        items_out = []
        for item in items_raw:
            item_scaled = (
                round(seg_total_scaled * item["_raw_amount"] / seg_raw)
                if seg_raw > 0 else 0
            )
            items_out.append({
                "flag_title":  item["flag_title"],
                "segment":     seg,
                "fraction":    item["fraction"],
                "amount":      item_scaled,
                "confidence":  item["confidence"],
                "reason":      item["reason"],
            })
        result[seg] = {"amount": seg_total_scaled, "items": items_out}

    # Inactive segments (raw == 0) → amount: 0, items: []
    for seg in SEGMENTS:
        if seg not in scaled_amounts:
            result[seg] = {"amount": 0, "items": []}

    return result

# =============================================================================
# CONFIDENCE CALIBRATION (deterministic)
# =============================================================================
# Caps
CAP_COMPS_LT_3         = 0.50
CAP_NO_VALID_COMPS     = 0.45
CAP_TENURE_UNRESOLVED  = 0.45
CAP_LEASE_MISSING      = 0.40
CAP_SHORT_LEASE_NO_BAND= 0.35
CAP_UNQUANTIFIED_RISKS = 0.55
# S33-STEP4a (2026-06-21): confidence cap when the legal pack contains
# language suggesting the subject property's actual condition may not be
# comparable to the comp evidence used — e.g. a clause stating the seller
# will not answer buyer enquiries, an unusually extended death/probate
# completion contingency, or explicit reference to squatters/unauthorised
# occupiers. These are real, document-traceable signals (see the flag
# extraction prompt rule added in app.py under the same tag) but do NOT
# translate into any defensible numeric price adjustment — a confidence
# penalty, never a fabricated discount.
CAP_CONDITION_RISK_SIGNALS = 0.55
# S33-FIX (2026-06-21): cap for EVIDENCE_TIER_PPD_CATEGORY_A (open-market
# only, no distressed-sale or auction comparables available — see
# EVIDENCE_TIER_* near the DISTANCE RULE section). Found live: a Hey Street
# (NG10 3HA) run with 10 type/age-matched comps, all Category A, scored
# 0.60-0.79 ("Moderate confidence") purely from comp count and other
# factors, with NOTHING in the confidence formula penalising the fact that
# every comp was the weakest available evidence tier. The Section 1 card
# text for this exact case explicitly says "Treat this figure as an
# UPPER-BOUND reference, not an expected auction outcome" — labelling that
# same valuation "Moderate confidence" directly contradicts the disclosure
# sitting right next to it. 0.59 keeps Category-A-only results strictly
# below the 0.60 "Moderate" threshold regardless of comp count, so the
# confidence label can never contradict the upper-bound caveat text.
CAP_CATEGORY_A_ONLY = 0.59

# S33-WORKBENCH-FIX (2026-06-21): hoisted from inside _calculate_confidence
# so calculate_workbench_ceiling can apply the identical condition-risk match
# to the REAL active_legal_flags it receives, rather than a duplicated tuple
# that could silently drift out of sync with the one Verdict's path uses.
# Root cause this fixes: calculate_workbench_ceiling previously copied
# verdict_ceiling.confidence verbatim and never looked at active_legal_flags
# for confidence purposes — confirmed twice on real, independently-extracted
# legal packs (Hey Street NG10 3HA, two separate fresh runs) where a real
# "Seller Will Not Answer Buyer Enquiries" flag existed and matched this
# exact tuple, yet no condition-risk cap ever appeared in the persisted
# confidence object, because _calculate_confidence was never called with
# real flags anywhere in the live request path.
_CONDITION_RISK_TITLE_MARKERS = (
    "no enquiries", "will not answer", "seller will not respond",
    "death of seller", "death of the seller", "probate", "grant of administration",
    "squatter", "unauthorised occupier", "unknown occupier", "unauthorized occupier",
)

def _condition_risk_flags(flags: list[dict]) -> list[dict]:
    """Real flags whose title matches a condition/distress risk marker.
    Shared by _calculate_confidence (Verdict path, called with flags=[] in
    the live system) and calculate_workbench_ceiling (Workbench path, the
    only place in production that ever receives the real flags array)."""
    if not isinstance(flags, list):
        return []
    return [
        f for f in flags
        if isinstance(f, dict) and any(m in (f.get("title") or "").lower() for m in _CONDITION_RISK_TITLE_MARKERS)
    ]

def _category_a_only_active(evidence_tier_used: str) -> bool:
    """True when the comparable valuation was built from PPD Category A
    (open-market) evidence only — the weakest available tier. Shared
    helper so workbench and verdict apply the identical test."""
    return evidence_tier_used == "ppd_category_a_open_market"

def _confidence_label(v: float) -> str:
    if v >= 0.80: return "High confidence"
    if v >= 0.60: return "Moderate confidence"
    if v >= 0.40: return "Low confidence"
    return "Insufficient evidence"

# =============================================================================
# UNCERTAINTY BAND (deterministic)
# =============================================================================
BASE_UNCERTAINTY = 0.05
UNCERTAINTY_CAP_LO = 0.05
UNCERTAINTY_CAP_HI = 0.20

def _uncertainty_band(valid_count: int, caps: list[dict]) -> float:
    u = BASE_UNCERTAINTY
    if valid_count <= 2:
        u += 0.08
    elif valid_count <= 4:
        u += 0.03
    for cap in caps:
        r = cap.get("reason", "")
        if "tenure" in r:    u += 0.04
        if "lease" in r:     u += 0.05
        if "legal_pack" in r:u += 0.04
        if "evidence" in r:  u += 0.04
    return max(UNCERTAINTY_CAP_LO, min(UNCERTAINTY_CAP_HI, round(u, 4)))

# =============================================================================
# WEIGHTED MEDIAN (deterministic)
# =============================================================================
def _weighted_median(pairs: list[tuple[float, float]]) -> Optional[float]:
    """
    Deterministic weighted median.
    pairs = [(adjusted_value, weight), ...]
    Rules:
      - sort by adjusted_value ascending
      - ignore zero or negative weights
      - sum positive weights
      - return first adjusted_value where cumulative_weight >= total_weight / 2
      - if no positive weights exist, return None
      - arithmetic mean must not be primary base
    """
    valid = [(v, w) for v, w in pairs if w > 0 and v > 0]
    if not valid:
        return None
    valid.sort(key=lambda x: x[0])
    total_w = sum(w for _, w in valid)
    if total_w <= 0:
        return None
    half = total_w / 2.0
    cumulative = 0.0
    for value, weight in valid:
        cumulative += weight
        if cumulative >= half:
            return value
    return valid[-1][0]

# =============================================================================
# SCORE HELPERS
# =============================================================================
def _distance_score(miles: Optional[float]) -> Optional[float]:
    # S33-STEP1: only excludes beyond MAX_RADIUS_MILES (3mi) now, not 0.5mi.
    # Comps between PRIMARY_RADIUS_MILES and MAX_RADIUS_MILES are included
    # with reduced weight via EXTENDED_DISTANCE_BANDS, not excluded outright.
    if miles is None or miles > MAX_RADIUS_MILES:
        return None  # outside absolute outer bound — exclude
    for lo, hi, score in EXTENDED_DISTANCE_BANDS:
        if lo <= miles <= hi:
            return score
    return EXTENDED_DISTANCE_BANDS[-1][2]  # beyond last band but within MAX_RADIUS_MILES

def _recency_score(months: Optional[float]) -> float:
    if months is None:
        return 0.60  # unknown age — treat as >12 months
    for lo, hi, score in RECENCY_BANDS:
        if lo <= months < hi:
            return score
    return 0.40

def _size_score(ratio: Optional[float]) -> float:
    if ratio is None:
        return 0.80  # unknown size — partial penalty
    for lo, hi, score in SIZE_BANDS:
        if lo <= ratio <= hi:
            return score
    return SIZE_OUTER_SCORE

def _size_adjustment(subject_area: Optional[float], comp_area: Optional[float]) -> float:
    if not subject_area or not comp_area or comp_area <= 0:
        return 1.00
    ratio = subject_area / comp_area
    capped = max(SIZE_ADJ_CAP_LO, min(SIZE_ADJ_CAP_HI, ratio))
    return round(capped, 4)

def _time_adjustment(hpi_factor: Optional[float]) -> float:
    """Use HPI-derived time adjustment if provided; else 1.00 with audit warning."""
    if hpi_factor and hpi_factor > 0:
        return float(hpi_factor)
    return 1.00

def _lease_adjustment(subj_band: Optional[int], comp_band: Optional[int]) -> Optional[float]:
    if comp_band is None:
        return 1.00  # unknown — use 1.00, cap confidence
    if subj_band is None:
        return 1.00
    if subj_band == comp_band:
        return LEASE_ADJ_SAME_BAND
    if _bands_adjacent(subj_band, comp_band):
        return LEASE_ADJ_ADJ_BAND
    return None  # non-adjacent → exclude

def _tenure_adjustment_match(subj_tenure: Optional[str], comp_tenure: Optional[str]) -> Optional[float]:
    """Returns adjustment factor or None (exclude)."""
    if not subj_tenure or not comp_tenure:
        return 1.00  # unknown — use 1.00, cap confidence
    if subj_tenure.lower() == comp_tenure.lower():
        return 1.00
    return None  # mismatch → exclude

# =============================================================================
# COMPARABLE ASSESSMENT
# =============================================================================
def _assess_comp(
    comp: dict,
    subject: dict,
    comp_idx: int,
) -> tuple[Optional[dict], Optional[dict]]:
    """
    Assess one sold comparable against the subject property.
    Returns (valid_comp_dict, None) or (None, excluded_comp_dict).
    """
    reasons_excluded = []
    audit_warnings = []

    # ── Normalise field names from housing_comps_v1 RPC ──────────────────────
    # RPC returns:  miles, price, duration (F/L), age_months, floor_area,
    #               hpi_multiplier, property_type, postcode, paon, saon, street
    # Engine expects: distance_miles, price, tenure, months_ago, internal_area,
    #               hpi_adjustment, property_type
    # Build a normalised view without mutating the original comp dict.
    price = comp.get("price") or comp.get("sale_price") or comp.get("price_paid")
    try:
        price = float(price)
    except (TypeError, ValueError):
        price = None

    if not price or price <= 0:
        return None, {"comp_idx": comp_idx, "reason": "invalid_price", "comp": comp}

    dist = comp.get("distance_miles") or comp.get("distance") or comp.get("miles")
    try:
        dist = float(dist)
    except (TypeError, ValueError):
        dist = None

    dist_score = _distance_score(dist)
    if dist_score is None:
        return None, {"comp_idx": comp_idx, "reason": f"outside_{MAX_RADIUS_MILES}_mile_outer_bound dist={dist}", "comp": comp}

    # Duplicate detection by address
    comp_addr = (comp.get("address") or "").strip().lower()
    # (dedup is done at the call site across the comp universe)

    # Type match
    # Normalise both sides: LR single-char codes (F/S/T/D) from the RPC must
    # map to their full-word equivalents before comparison with LLM-supplied labels.
    subj_type = _normalise_property_type(subject.get("property_type") or "")
    comp_type = _normalise_property_type(comp.get("property_type") or comp.get("type") or "")
    if subj_type and comp_type:
        if subj_type == comp_type:
            type_score = TYPE_SAME
        elif _types_near_equiv(subj_type, comp_type):
            type_score = TYPE_NEAR_EQUIV
        else:
            return None, {"comp_idx": comp_idx, "reason": f"type_mismatch subj={subj_type} comp={comp_type}", "comp": comp}
    else:
        type_score = TYPE_NEAR_EQUIV  # unknown — partial
        audit_warnings.append("property_type unknown for comp or subject")

    # Tenure: engine expects "freehold"/"leasehold" strings.
    # RPC returns duration="F" or "L". Normalise both.
    _raw_tenure = comp.get("tenure") or comp.get("duration") or ""
    _dur = str(_raw_tenure).strip().upper()
    if _dur == "F":
        comp_tenure_norm = "freehold"
    elif _dur == "L":
        comp_tenure_norm = "leasehold"
    elif _dur:
        comp_tenure_norm = _raw_tenure.lower()
    else:
        comp_tenure_norm = None

    subj_tenure = (subject.get("tenure") or "").lower() or None
    # Normalise subject tenure from RPC duration codes if needed
    if subj_tenure == "f":
        subj_tenure = "freehold"
    elif subj_tenure == "l":
        subj_tenure = "leasehold"
    comp_tenure  = comp_tenure_norm
    tenure_adj = _tenure_adjustment_match(subj_tenure, comp_tenure)
    if tenure_adj is None:
        return None, {"comp_idx": comp_idx, "reason": f"tenure_mismatch subj={subj_tenure} comp={comp_tenure}", "comp": comp}
    tenure_unknown = not subj_tenure or not comp_tenure
    tenure_score = TENURE_SAME if not tenure_unknown else TENURE_UNKNOWN

    # Lease match (only material for leasehold)
    subj_lease_len = subject.get("lease_length")
    comp_lease_len = comp.get("lease_length")
    try:
        subj_lease_len = float(subj_lease_len) if subj_lease_len is not None else None
    except (TypeError, ValueError):
        subj_lease_len = None
    try:
        comp_lease_len = float(comp_lease_len) if comp_lease_len is not None else None
    except (TypeError, ValueError):
        comp_lease_len = None

    leasehold_material = (subj_tenure and "leasehold" in subj_tenure) or (comp_tenure and "leasehold" in comp_tenure)
    subj_band = _lease_band(subj_lease_len) if leasehold_material else None
    comp_band = _lease_band(comp_lease_len) if leasehold_material else None

    lease_adj = 1.00
    lease_score = LEASE_SAME_BAND
    if leasehold_material:
        la = _lease_adjustment(subj_band, comp_band)
        if la is None:
            return None, {"comp_idx": comp_idx, "reason": f"non_adjacent_lease_band subj_band={subj_band} comp_band={comp_band}", "comp": comp}
        lease_adj = la
        if comp_band is None:
            lease_score = LEASE_UNKNOWN
            audit_warnings.append("comp lease_length unknown for leasehold comp")
        elif _bands_adjacent(subj_band, comp_band):
            lease_score = LEASE_ADJ_BAND
        else:
            lease_score = LEASE_SAME_BAND

    # Size
    subj_area = subject.get("internal_area") or subject.get("floor_area")
    comp_area  = comp.get("internal_area") or comp.get("floor_area") or comp.get("total_floor_area")
    try:
        subj_area = float(subj_area) if subj_area else None
        comp_area  = float(comp_area)  if comp_area  else None
    except (TypeError, ValueError):
        subj_area = comp_area = None
    size_adj  = _size_adjustment(subj_area, comp_area)
    size_ratio = (subj_area / comp_area) if (subj_area and comp_area and comp_area > 0) else None
    sz_score  = _size_score(size_ratio)
    if subj_area is None or comp_area is None:
        audit_warnings.append("floor_area missing for size adjustment")

    # Recency
    months = comp.get("months_ago") or comp.get("age_months") or comp.get("age")
    try:
        months = float(months) if months is not None else None
    except (TypeError, ValueError):
        months = None
    rec_score = _recency_score(months)

    # Time adjustment
    hpi_factor = comp.get("hpi_adjustment") or comp.get("time_adjustment") or comp.get("hpi_multiplier")
    try:
        hpi_factor = float(hpi_factor) if hpi_factor else None
    except (TypeError, ValueError):
        hpi_factor = None
    time_adj = _time_adjustment(hpi_factor)
    if hpi_factor is None:
        audit_warnings.append("hpi_adjustment missing — time_adjustment=1.00")

    # Condition
    subj_cond = (subject.get("condition") or "unknown").lower()
    comp_cond = (comp.get("condition") or "unknown").lower()
    if subj_cond == "unknown" or comp_cond == "unknown":
        cond_adj = CONDITION_SAME_OR_UNKNOWN
    elif _condition_better(comp_cond, subj_cond):
        cond_adj = CONDITION_COMP_BETTER
    elif _condition_worse(comp_cond, subj_cond):
        cond_adj = CONDITION_COMP_WORSE
    else:
        cond_adj = CONDITION_SAME_OR_UNKNOWN

    # Evidence quality
    ev_qual = comp.get("evidence_quality") or "partial"
    eq_map = {
        "official": EVIDENCE_OFFICIAL, "full": EVIDENCE_OFFICIAL,
        "partial": EVIDENCE_PARTIAL,
        "weak": EVIDENCE_WEAK,
        "uncorroborated": EVIDENCE_UNCORROBORATED,
    }
    ev_score = eq_map.get(str(ev_qual).lower(), EVIDENCE_PARTIAL)

    # Adjusted value
    adjusted_value = round(
        price * time_adj * size_adj * tenure_adj * lease_adj * cond_adj, 2
    )

    # Weight
    weight = round(
        type_score * tenure_score * lease_score * dist_score * rec_score * sz_score * ev_score, 6
    )

    if weight <= 0:
        return None, {"comp_idx": comp_idx, "reason": "zero_weight", "comp": comp}

    return {
        "comp_idx":        comp_idx,
        "address":         comp.get("address", ""),
        "sale_price":      price,
        "adjusted_value":  adjusted_value,
        "weight":          weight,
        "distance_miles":  dist,
        "months_ago":      months,
        "property_type":   comp_type,
        "tenure":          comp_tenure,
        "lease_length":    comp_lease_len,
        "lease_band":      comp_band,
        "internal_area":   comp_area,
        # S33-STEP3: PPD's only real transaction-channel signal. A = standard
        # market sale; B = repossession/power-of-sale/buy-to-let/corporate
        # transfer (HMLR's own documented definition — a blended distressed-
        # adjacent category, NOT a clean "auction" flag, since PPD has no
        # such field). Carried through here so the tiered evidence-selection
        # logic in the calling function can split the pool rather than blend
        # categories with materially different bases of value.
        "ppd_category_type": (comp.get("ppd_category_type") or "A").strip().upper() or "A",
        "evidence_tier":     EVIDENCE_TIER_PPD_CATEGORY_B if str(comp.get("ppd_category_type") or "").strip().upper() == "B" else EVIDENCE_TIER_PPD_CATEGORY_A,
        "adjustments": {
            "time":      round(time_adj, 4),
            "size":      round(size_adj, 4),
            "tenure":    round(tenure_adj, 4),
            "lease":     round(lease_adj, 4),
            "condition": round(cond_adj, 4),
        },
        "scores": {
            "type":              round(type_score, 4),
            "tenure":            round(tenure_score, 4),
            "lease":             round(lease_score, 4),
            "distance":          round(dist_score, 4),
            "recency":           round(rec_score, 4),
            "size":              round(sz_score, 4),
            "evidence_quality":  round(ev_score, 4),
        },
        "audit_warnings": audit_warnings,
    }, None


def _normalise_property_type(t: str) -> str:
    """
    Normalise property type strings so Land Registry single-character codes
    (used by the housing_comps_v1 RPC / price_paid_raw_2025) match the
    human-readable labels that the LLM returns for the subject property.

    Land Registry codes (price_paid_raw_2025.property_type):
      F → flat/maisonette, D → detached, S → semi-detached, T → terraced, O → other

    After normalisation all comparisons are against the full-word forms used by
    FLAT_TERMS and HOUSE_TERMS so no other call-site needs to change.
    """
    _LR_MAP = {
        "f": "flat",
        "d": "detached",
        "s": "semi-detached",
        "t": "terraced",
        "o": "other",
    }
    s = t.strip().lower()
    # Single-char LR code
    if s in _LR_MAP:
        return _LR_MAP[s]
    # Compound LLM labels → canonical form
    if "maisonette" in s:
        return "maisonette"
    if "flat" in s:
        return "flat"
    if "apartment" in s:
        return "apartment"
    if "semi" in s:
        return "semi-detached"
    if "terraced" in s or "terrace" in s or "end-terrace" in s:
        return "terraced"
    if "detached" in s:
        return "detached"
    return s


def _types_near_equiv(a: str, b: str) -> bool:
    # Normalise both sides so LR single-char codes match full-word LLM labels.
    a = _normalise_property_type(a)
    b = _normalise_property_type(b)
    FLAT_TERMS  = {"flat", "apartment", "maisonette"}
    HOUSE_TERMS = {"terraced", "semi-detached", "detached", "end-terrace", "terrace"}
    if a in FLAT_TERMS and b in FLAT_TERMS:
        return True
    if a in HOUSE_TERMS and b in HOUSE_TERMS:
        return True
    return False


def _condition_better(comp: str, subj: str) -> bool:
    ORDER = ["poor", "fair", "average", "good", "excellent"]
    try:
        return ORDER.index(comp) > ORDER.index(subj)
    except ValueError:
        return False


def _condition_worse(comp: str, subj: str) -> bool:
    ORDER = ["poor", "fair", "average", "good", "excellent"]
    try:
        return ORDER.index(comp) < ORDER.index(subj)
    except ValueError:
        return False

# =============================================================================
# CONFIDENCE CALCULATION
# =============================================================================
def _calculate_confidence(
    valid_comps: list[dict],
    subject: dict,
    legal_flags: list[dict],
    included_risks: list[dict],
    tenure_unknown: bool,
    lease_unknown: bool,
    short_lease_no_band: bool,
    evidence_tier_used: str = "",
) -> tuple[float, list[dict], str]:
    """
    raw_confidence =
        0.30 × comp_quality
      + 0.20 × tenure_certainty
      + 0.15 × lease_certainty
      + 0.15 × legal_pack_completeness
      + 0.10 × market_depth
      + 0.10 × audit_cleanliness

    Returns (final_confidence, caps_applied, label).
    """
    n = len(valid_comps)

    # comp_quality: mean weight of valid comps, normalised
    if valid_comps:
        avg_w = sum(c["weight"] for c in valid_comps) / n
        comp_quality = min(1.00, avg_w / 0.60)  # 0.60 treated as full certainty threshold
    else:
        comp_quality = 0.00

    # tenure_certainty
    tenure_certainty = 0.25 if tenure_unknown else 1.00

    # lease_certainty
    subj_tenure = (subject.get("tenure") or "").lower()
    if subj_tenure == "f": subj_tenure = "freehold"
    if subj_tenure == "l": subj_tenure = "leasehold"
    leasehold_material = "leasehold" in subj_tenure
    if not leasehold_material:
        lease_certainty = 1.00
    elif lease_unknown:
        lease_certainty = 0.25
    else:
        lease_certainty = 0.75

    # legal_pack_completeness: fraction of flags that are not "missing"
    all_flags = legal_flags if isinstance(legal_flags, list) else []
    missing_count = sum(1 for f in all_flags if (f.get("severity") or "").lower() == "missing")
    if not all_flags:
        legal_pack_completeness = 1.00
    else:
        legal_pack_completeness = max(0.00, 1.00 - missing_count / len(all_flags))

    # market_depth
    if n >= 5:  market_depth = 1.00
    elif n >= 3: market_depth = 0.75
    elif n >= 1: market_depth = 0.50
    else:        market_depth = 0.00

    # audit_cleanliness: penalise if many comps have warnings
    warn_count = sum(len(c.get("audit_warnings", [])) for c in valid_comps)
    if warn_count == 0:   audit_cleanliness = 1.00
    elif warn_count <= 3: audit_cleanliness = 0.75
    elif warn_count <= 8: audit_cleanliness = 0.50
    else:                 audit_cleanliness = 0.25

    raw = round(
        0.30 * comp_quality
      + 0.20 * tenure_certainty
      + 0.15 * lease_certainty
      + 0.15 * legal_pack_completeness
      + 0.10 * market_depth
      + 0.10 * audit_cleanliness,
        4
    )

    caps = []
    conf = raw

    if n < 3:
        if conf > CAP_COMPS_LT_3:
            caps.append({"cap": CAP_COMPS_LT_3, "reason": "valid_comparable_count < 3"})
            conf = min(conf, CAP_COMPS_LT_3)

    if n == 0:
        if conf > CAP_NO_VALID_COMPS:
            caps.append({"cap": CAP_NO_VALID_COMPS, "reason": "no valid 0.5-mile comps"})
            conf = min(conf, CAP_NO_VALID_COMPS)

    if tenure_unknown:
        if conf > CAP_TENURE_UNRESOLVED:
            caps.append({"cap": CAP_TENURE_UNRESOLVED, "reason": "tenure unresolved and material"})
            conf = min(conf, CAP_TENURE_UNRESOLVED)

    if leasehold_material and lease_unknown:
        if conf > CAP_LEASE_MISSING:
            caps.append({"cap": CAP_LEASE_MISSING, "reason": "lease length missing for leasehold"})
            conf = min(conf, CAP_LEASE_MISSING)

    if short_lease_no_band:
        if conf > CAP_SHORT_LEASE_NO_BAND:
            caps.append({"cap": CAP_SHORT_LEASE_NO_BAND, "reason": "subject lease < 80 and no same-band lease comps"})
            conf = min(conf, CAP_SHORT_LEASE_NO_BAND)

    # S33-STEP4a: scan the already-extracted flags array for condition/
    # distress risk language. Matches on flag TITLE since the extraction
    # prompt (app.py) was instructed to title these specific patterns
    # consistently — matching on title rather than free-text evidence
    # avoids false positives from incidental word overlap elsewhere in a
    # flag's summation/implication text.
    # S33-WORKBENCH-FIX: now calls the shared module-level helper (see
    # _condition_risk_flags near CAP_CATEGORY_A_ONLY) instead of an inline
    # tuple, so this and calculate_workbench_ceiling can never drift apart.
    condition_risk_flags = _condition_risk_flags(all_flags)

    unquantified = any(r.get("value_adjustment", 0) == 0 and r.get("severity", "note") not in ("note", "low")
                       for r in included_risks)
    if unquantified:
        if conf > CAP_UNQUANTIFIED_RISKS:
            caps.append({"cap": CAP_UNQUANTIFIED_RISKS, "reason": "material unquantified legal-pack value risks"})
            conf = min(conf, CAP_UNQUANTIFIED_RISKS)

    if condition_risk_flags:
        if conf > CAP_CONDITION_RISK_SIGNALS:
            caps.append({
                "cap": CAP_CONDITION_RISK_SIGNALS,
                "reason": (
                    f"legal pack contains {len(condition_risk_flags)} condition/distress "
                    f"risk signal(s) (e.g. no-enquiries clause, extended probate "
                    f"contingency, or occupier risk language) that may make this "
                    f"property's actual condition non-comparable to nearby sold "
                    f"comparables — no numeric adjustment applied, confidence reduced"
                ),
            })
            conf = min(conf, CAP_CONDITION_RISK_SIGNALS)

    # S33-FIX: Category-A-only evidence (no distressed/auction comps found)
    # must never score "Moderate" or higher — that label would contradict
    # the upper-bound disclosure shown alongside it on the Verdict page.
    if _category_a_only_active(evidence_tier_used):
        if conf > CAP_CATEGORY_A_ONLY:
            caps.append({
                "cap": CAP_CATEGORY_A_ONLY,
                "reason": (
                    "evidence_tier=ppd_category_a_open_market — no distressed-sale "
                    "or auction comparables found; this is the weakest available "
                    "evidence tier and cannot be rated above Low confidence"
                ),
            })
            conf = min(conf, CAP_CATEGORY_A_ONLY)

    final = round(max(0.00, min(1.00, conf)), 2)
    return final, caps, _confidence_label(final)

# =============================================================================
# LEGAL-PACK VALUE RISK PROCESSING
# =============================================================================
def _process_legal_risks(legal_flags: list[dict]) -> list[dict]:
    """
    Filter and map legal flags to property-value risks only.
    Excludes acquisition costs.

    Each risk now carries a segments dict of {segment_name: fraction} from
    _flag_to_segments(). The legacy value_adjustment field is retained as the
    SUM of all segment fractions for backward compatibility with
    _legal_pack_adjustment_factor() and the workbench risk_factor product formula.
    """
    risks = []
    for i, flag in enumerate(legal_flags):
        if _is_acquisition_cost_only(flag):
            continue
        sev  = (flag.get("severity") or "note").lower().strip()
        segs = _flag_to_segments(flag)
        total_adj = round(sum(segs.values()), 6)
        risks.append({
            "risk_id":         f"r{i:03d}",
            "title":           flag.get("title", ""),
            "source_evidence": flag.get("summation", "") or flag.get("implication", ""),
            "category":        flag.get("risk_category") or flag.get("category") or "general",
            "severity":        sev,
            "value_adjustment": total_adj,
            "segments":         segs,
            "included":         True,
            "reason": (
                f"severity={sev}; market_consequence_segments={list(segs.keys())}; "
                f"total_value_adjustment={total_adj}"
            ),
        })
    return risks

def _legal_pack_adjustment_factor(risks: list[dict]) -> float:
    """
    legal_pack_value_risk_adjustment_factor = product(1 - value_adjustment_i)
    Total reduction capped at MAX_TOTAL_VALUE_RISK_ADJ.
    """
    factor = 1.0
    total_reduction = 0.0
    for r in risks:
        adj = r.get("value_adjustment", 0.0)
        if adj <= 0:
            continue
        total_reduction += adj
        if total_reduction > MAX_TOTAL_VALUE_RISK_ADJ:
            # Hard cap: do not apply further
            break
        factor *= (1.0 - adj)
    return round(max(1.0 - MAX_TOTAL_VALUE_RISK_ADJ, factor), 6)


# =============================================================================
# VERDICT CEILING — comparable base only, NO legal-pack flag risks applied
# =============================================================================
def calculate_verdict_ceiling(
    sold_comps: Optional[list[dict]] = None,
    subject: Optional[dict] = None,
    base_valuation: Optional[float] = None,
    strategy: str = "BTL",
    fallback_allowed: bool = True,
) -> dict:
    """
    Verdict ceiling: weighted median of 0.5-mile relational comparable evidence.
    Legal-pack flag risks are NOT applied.
    This is the ceiling the investor sees before legal-pack adjustment.
    """
    result = calculate_ceiling(
        legal_flags=[],   # ← no flag risks: verdict is pure comps
        financial_inputs={},
        base_valuation=base_valuation,
        strategy=strategy,
        sold_comps=sold_comps,
        subject=subject,
        fallback_allowed=fallback_allowed,
    )
    result["_ceiling_type"] = "verdict"
    return result


# =============================================================================
# WORKBENCH CEILING — verdict ceiling × active legal-pack flag risk product
# =============================================================================
def calculate_workbench_ceiling(
    verdict_ceiling: dict,
    active_legal_flags: list[dict],
) -> dict:
    """
    Workbench ceiling: verdict_ceiling × active_flag_risk_factor.

    active_flag_risk_factor = product(1 - value_adjustment_i)
        for each active (unresolved) legal-pack value risk.

    Workbench ceiling is clamped so it cannot exceed verdict ceiling.
    If active_legal_flags is empty, workbench_ceiling equals verdict_ceiling.
    """
    active_legal_flags = active_legal_flags if isinstance(active_legal_flags, list) else []

    verdict_vr   = verdict_ceiling.get("valuation_range") or {}
    # Prefer explicit comparable_valuation (unambiguous). Fall back to midpoint for
    # backward compat with objects that pre-date this field.
    verdict_comparable_valuation = (
        verdict_ceiling.get("comparable_valuation")
        or (verdict_ceiling.get("base") or {}).get("value")
        or verdict_vr.get("midpoint")
    )
    verdict_mid  = verdict_comparable_valuation   # canonical base for workbench derivation
    verdict_low  = verdict_vr.get("low")
    verdict_high = verdict_vr.get("high")
    u_band       = verdict_vr.get("uncertainty_band", BASE_UNCERTAINTY)

    # If verdict has no valid comparable_valuation, workbench is also insufficient
    if not verdict_mid or verdict_mid <= 0:
        return {
            "_ceiling_type": "workbench",
            "status": "insufficient_evidence",
            "valuation_range": {"low": None, "midpoint": None, "high": None, "uncertainty_band": None},
            "ceiling_range":   {"low": None, "high": None},
            "legal_pack_value_risks": {"method": "property_value_risk_adjustment_only",
                                       "adjustment_factor": 1.0, "adjusted_value": None, "risks": []},
            "active_flag_count": len(active_legal_flags),
            "confidence": verdict_ceiling.get("confidence"),
            "audit": {"warnings": ["verdict_ceiling has no valid midpoint — workbench cannot be computed"]},
        }

    # Compute active flag risk adjustment
    active_risks = _process_legal_risks(active_legal_flags)
    risk_factor  = _legal_pack_adjustment_factor(active_risks)

    # risk_discount_pct: percentage reduction applied by active flags.
    # If active_legal_flags is empty, risk_factor = 1.0 → risk_discount_pct = 0.
    risk_discount_pct = round((1.0 - risk_factor) * 100, 1)

    wb_mid  = round(verdict_mid  * risk_factor, 2)
    wb_low  = round(verdict_low  * risk_factor, 2) if verdict_low  is not None else round(wb_mid * (1 - u_band), 2)
    wb_high = round(verdict_high * risk_factor, 2) if verdict_high is not None else round(wb_mid * (1 + u_band), 2)

    # Clamp: workbench must never exceed verdict
    wb_mid  = min(wb_mid,  verdict_mid)
    wb_low  = min(wb_low,  verdict_low  if verdict_low  is not None else wb_low)
    wb_high = min(wb_high, verdict_high if verdict_high is not None else wb_high)

    # All-flags-resolved path: range and discount must exactly equal verdict
    all_resolved = len(active_legal_flags) == 0
    if all_resolved:
        wb_mid  = verdict_mid
        wb_low  = verdict_low  if verdict_low  is not None else wb_low
        wb_high = verdict_high if verdict_high is not None else wb_high
        risk_discount_pct = 0.0

    # wb_mid = risk_adjusted_value for this workbench state.
    wb_risk_adjusted_value = wb_mid

    # Compute total_adjustment and adjustment_pct from the engine (not from app.py).
    _total_adj = round(verdict_comparable_valuation - wb_risk_adjusted_value, 2)
    _adj_pct   = round(_total_adj / verdict_comparable_valuation * 100, 1) if verdict_comparable_valuation else 0.0

    # Build market-consequence adjustments from active risks.
    # Segment amounts are scaled so they sum exactly to total_adjustment.
    _mca = _build_market_consequence_adjustments(
        risks=active_risks,
        comparable_valuation=verdict_comparable_valuation,
        total_adjustment=_total_adj,
    )

    # ── S33-WORKBENCH-FIX (2026-06-21): workbench-specific confidence ──────
    # ROOT CAUSE FIXED: this function previously copied verdict_ceiling's
    # confidence object verbatim (see "confidence": verdict_ceiling.get(...)
    # below, prior to this fix) and never looked at active_legal_flags for
    # confidence purposes. Confirmed twice on real, independently-extracted
    # legal packs (Hey Street NG10 3HA) that a genuine "Seller Will Not
    # Answer Buyer Enquiries" flag — correctly detected by
    # _condition_risk_flags() — never produced a confidence cap, because
    # Verdict is computed with legal_flags=[] by design ("Verdict =
    # comparable base only") and Workbench, the only place that ever
    # receives the real flags, never used them for this purpose.
    #
    # EXPLICIT STACKING RULE (decided here, not left to code-order accident):
    #   1. Start from verdict_ceiling's own confidence + its own caps
    #      (e.g. comp-count, category-A-only) — never discarded.
    #   2. If the REAL active_legal_flags contain a condition/distress risk
    #      signal, apply CAP_CONDITION_RISK_SIGNALS on top.
    #   3. Final confidence = min(verdict_confidence, condition_cap) when
    #      the condition cap applies; otherwise unchanged from Verdict.
    #   Confidence can only move DOWN from additional real flag evidence,
    #   never up — Workbench is reviewing real risk, not discovering reasons
    #   to be more confident than Verdict already was.
    #
    # SCOPE BOUNDARY: this does NOT change what Verdict sees or computes.
    # Verdict's own legal_flags=[] call is untouched. This only makes
    # Workbench's *own* confidence honest about the flags Workbench itself
    # already receives — it does not reopen "should Verdict see flags."
    _verdict_conf_obj = verdict_ceiling.get("confidence") or {}
    _verdict_conf_final = _verdict_conf_obj.get("final")
    try:
        _verdict_conf_final = float(_verdict_conf_final) if _verdict_conf_final is not None else None
    except (TypeError, ValueError):
        _verdict_conf_final = None
    _verdict_caps = list(_verdict_conf_obj.get("caps") or [])

    _wb_condition_flags = _condition_risk_flags(active_legal_flags)
    if _verdict_conf_final is not None and _wb_condition_flags and _verdict_conf_final > CAP_CONDITION_RISK_SIGNALS:
        _wb_conf_final = CAP_CONDITION_RISK_SIGNALS
        _wb_caps = _verdict_caps + [{
            "cap": CAP_CONDITION_RISK_SIGNALS,
            "reason": (
                f"legal pack contains {len(_wb_condition_flags)} condition/distress "
                f"risk signal(s) (e.g. no-enquiries clause, extended probate "
                f"contingency, or occupier risk language) among the active "
                f"legal-pack flags reviewed in this workbench — no numeric "
                f"adjustment applied, confidence reduced"
            ),
        }]
        _wb_confidence = {
            "raw":   _verdict_conf_obj.get("raw"),
            "caps":  _wb_caps,
            "final": round(_wb_conf_final, 2),
            "label": _confidence_label(_wb_conf_final),
        }
    else:
        # No real condition-risk flags active, or Verdict's confidence was
        # already at/below the condition cap — Workbench confidence is
        # identical to Verdict's, unchanged from pre-fix behaviour.
        _wb_confidence = verdict_ceiling.get("confidence")

    return {
        "_ceiling_type": "workbench",
        "status": "all_flags_resolved" if all_resolved else verdict_ceiling.get("status", "ok"),
        # Explicit semantic fields (unambiguous).
        "comparable_valuation": verdict_comparable_valuation,  # from Verdict, unchanged
        "risk_adjusted_value":  wb_risk_adjusted_value,        # comparable_valuation × risk_factor
        "valuation_range": {
            "low":              wb_low,
            # midpoint = risk_adjusted_value (backward-compat alias).
            "midpoint":         wb_mid,
            "high":             wb_high,
            "uncertainty_band": u_band,
        },
        "ceiling_range": {
            "low":  int(round(wb_low))  if wb_low  is not None else None,
            "high": int(round(wb_high)) if wb_high is not None else None,
        },
        "legal_pack_value_risks": {
            "method":            "property_value_risk_adjustment_only",
            "adjustment_factor": risk_factor,
            "adjusted_value":    wb_risk_adjusted_value,  # = comparable_valuation × risk_factor
            "risks":             active_risks,
        },
        "risk_discount_pct":   risk_discount_pct,
        # Market-consequence segment breakdown — canonical from engine, used by Verdict waterfall.
        "comparable_valuation_for_adjustment": verdict_comparable_valuation,
        "total_adjustment":              _total_adj,
        "adjustment_pct":               _adj_pct,
        "market_consequence_adjustments": _mca,
        "active_flag_count":   len(active_legal_flags),
        "all_flags_resolved":  all_resolved,
        "verdict_midpoint":    verdict_mid,      # backward compat
        "verdict_range": {
            "low":      verdict_low,
            "midpoint": verdict_mid,             # backward compat
            "high":     verdict_high,
        },
        "confidence":        _wb_confidence,
        "base":              verdict_ceiling.get("base"),
        "base_valuation":    verdict_ceiling.get("base_valuation"),
        "base_method":       verdict_ceiling.get("base_method"),
        "strategy_used":     verdict_ceiling.get("strategy_used"),
        "audit": {
            "comparable_valuation":        verdict_comparable_valuation,
            "risk_adjusted_value":         wb_risk_adjusted_value,
            "verdict_ceiling_midpoint":    verdict_mid,   # backward compat
            "active_flag_count":           len(active_legal_flags),
            "all_flags_resolved":          all_resolved,
            "risk_adjustment_factor":      risk_factor,
            "risk_discount_pct":           risk_discount_pct,
            "formula": "risk_adjusted_value = comparable_valuation × active_flag_risk_factor",
            "resolved_flags_excluded":     True,
        },
        # acquisition costs excluded
        "acquisition_costs":   None,
        "excluded_from_ceiling": EXCLUDED_FROM_CEILING,
    }


# =============================================================================
# FINANCIAL CURRENT STANDING — read workbench ceiling vs current bid
# =============================================================================
def calculate_financial_standing(
    workbench_ceiling: dict,
    current_bid: Optional[float] = None,
) -> dict:
    """
    Financial current standing: comparison of current_bid vs workbench ceiling.
    Does NOT calculate valuation. Does NOT apply legal risk.
    MY BID changes current_standing only — never alters verdict or workbench ceiling.
    """
    wb_vr  = workbench_ceiling.get("valuation_range") or {}
    wb_mid = wb_vr.get("midpoint")
    wb_low = wb_vr.get("low")
    wb_high= wb_vr.get("high")

    if current_bid and current_bid > 0 and wb_mid and wb_mid > 0:
        gap_to_ceiling = round(wb_mid - current_bid, 2)
        pct_of_ceiling = round((current_bid / wb_mid) * 100, 1)
        if current_bid < wb_low if wb_low else False:
            position = "below_range"
        elif current_bid > wb_high if wb_high else False:
            position = "above_ceiling"
        elif current_bid > wb_mid:
            position = "above_midpoint"
        else:
            position = "within_range"
    else:
        gap_to_ceiling = None
        pct_of_ceiling = None
        position = "no_bid"

    return {
        "workbench_ceiling_range": {
            "low":      wb_low,
            "midpoint": wb_mid,
            "high":     wb_high,
        },
        "current_bid":     current_bid,
        "gap_to_ceiling":  gap_to_ceiling,
        "pct_of_ceiling":  pct_of_ceiling,
        "position":        position,
        "_note": "MY BID changes current_standing only. Verdict and Workbench ceilings are unchanged.",
    }


def calculate_ceiling(
    legal_flags: list[dict],
    financial_inputs: dict,
    base_valuation: Optional[float] = None,
    strategy: str = "BTL",
    sold_comps: Optional[list[dict]] = None,
    subject: Optional[dict] = None,
    fallback_allowed: bool = True,
) -> dict:
    """
    Red-book-style paper valuation similarity ceiling engine.

    Parameters
    ----------
    legal_flags     : list of flag dicts from LLM analysis
    financial_inputs: dict of financial user inputs (NOT used in ceiling formula;
                      retained for downstream consumers)
    base_valuation  : optional external override — overrides comparable base
                      if provided and > 5000
    strategy        : investment strategy label (informational only)
    sold_comps      : list of sold comparable dicts (primary input)
    subject         : subject property dict (type, tenure, lease_length, area…)
    fallback_allowed: if True and comp count < 3, return degraded ceiling

    Returns
    -------
    Canonical ceiling dict. Backend owns this. Frontend must not mutate values.
    """
    legal_flags      = legal_flags if isinstance(legal_flags, list) else []
    financial_inputs = financial_inputs if isinstance(financial_inputs, dict) else {}
    sold_comps       = sold_comps if isinstance(sold_comps, list) else []
    subject          = subject if isinstance(subject, dict) else {}

    assumptions: list[str] = []
    evidence_gaps: list[str] = []
    warnings: list[str] = []
    formula_trace: list[str] = []

    # ── STEP 1: Type/tenure matching + HPI normalisation ─────────────────────
    # Doctrine: type + tenure match BEFORE IQR trimming.
    # HPI normalisation occurs inside _assess_comp via hpi_multiplier field.
    # All comps returned by _assess_comp have already been:
    #   (a) matched for type and tenure (excluded if mismatch),
    #   (b) HPI-normalised (adjusted_value = price × hpi_multiplier × size_adj × …).
    valid_comps:    list[dict] = []
    excluded_comps: list[dict] = []
    seen_addresses: set[str]   = set()

    formula_trace.append("step_1: type+tenure match + HPI normalisation within 0.5 miles")

    _hpi_used_count    = 0
    _hpi_missing_count = 0

    for idx, comp in enumerate(sold_comps):
        addr = (comp.get("address") or "").strip().lower()
        if addr and addr in seen_addresses:
            excluded_comps.append({"comp_idx": idx, "reason": "duplicate_address", "comp": comp})
            continue
        if addr:
            seen_addresses.add(addr)

        valid, excl = _assess_comp(comp, subject, idx)
        if valid:
            # Track HPI coverage
            _hpi_raw = comp.get("hpi_adjustment") or comp.get("time_adjustment") or comp.get("hpi_multiplier")
            if _hpi_raw and float(_hpi_raw) != 1.0:
                _hpi_used_count += 1
            else:
                _hpi_missing_count += 1
            valid_comps.append(valid)
        else:
            excluded_comps.append(excl)

    _pre_trim_count = len(valid_comps)
    formula_trace.append(
        f"step_1_result: matched={_pre_trim_count} excluded={len(excluded_comps)} "
        f"hpi_used={_hpi_used_count} hpi_missing={_hpi_missing_count}"
    )

    # ── STEP 1b: IQR outlier trimming on HPI-adjusted values ─────────────────
    # Doctrine: IQR must run AFTER type/tenure matching AND after HPI normalisation.
    # We trim on adjusted_value (which already incorporates hpi_multiplier), not on
    # nominal sale_price. This is the authoritative IQR step. The IQR in
    # get_housing_data (area fetch pipeline) uses nominal prices as a pre-filter
    # only; this step is the canonical post-HPI trim inside the valuation engine.
    _iqr_lower: Optional[float] = None
    _iqr_upper: Optional[float] = None
    _iqr_removed_count = 0
    _iqr_removed_reasons: list[str] = []

    if len(valid_comps) >= 6:
        _adj_values = sorted(c["adjusted_value"] for c in valid_comps)
        _n = len(_adj_values)
        _q1 = _adj_values[_n // 4]
        _q3 = _adj_values[(3 * _n) // 4]
        _iqr_val = _q3 - _q1
        _iqr_lower = _q1 - 1.5 * _iqr_val
        _iqr_upper = _q3 + 1.5 * _iqr_val
        _iqr_survivors = [
            c for c in valid_comps
            if _iqr_lower <= c["adjusted_value"] <= _iqr_upper
        ]
        if len(_iqr_survivors) >= MIN_REQUIRED_COMPS:
            _removed = [c for c in valid_comps if c not in _iqr_survivors]
            for _rc in _removed:
                _reason = (
                    f"iqr_outlier_below_fence adj={_rc['adjusted_value']:.0f} "
                    f"fence_lo={_iqr_lower:.0f}"
                    if _rc["adjusted_value"] < _iqr_lower
                    else f"iqr_outlier_above_fence adj={_rc['adjusted_value']:.0f} "
                         f"fence_hi={_iqr_upper:.0f}"
                )
                excluded_comps.append({
                    "comp_idx":  _rc["comp_idx"],
                    "reason":    _reason,
                    "comp":      {},  # comp already assessed — no raw dict needed
                })
                _iqr_removed_reasons.append(_reason)
            _iqr_removed_count = len(_removed)
            valid_comps = _iqr_survivors
            formula_trace.append(
                f"step_1b: iqr_trim applied "
                f"fence=[{_iqr_lower:.0f},{_iqr_upper:.0f}] "
                f"removed={_iqr_removed_count} survivors={len(valid_comps)}"
            )
        else:
            warnings.append(
                f"IQR trim skipped: only {len(_iqr_survivors)} comps survive "
                f"fence [{_iqr_lower:.0f},{_iqr_upper:.0f}] — keeping all {_pre_trim_count}"
            )
            formula_trace.append(
                f"step_1b: iqr_trim skipped — "
                f"would leave only {len(_iqr_survivors)} comps < MIN_REQUIRED_COMPS={MIN_REQUIRED_COMPS}"
            )
            _iqr_lower = _iqr_upper = None  # mark as not applied
    else:
        formula_trace.append(
            f"step_1b: iqr_trim skipped — only {len(valid_comps)} comps "
            f"(need ≥6 for IQR to be meaningful)"
        )

    n_valid = len(valid_comps)
    formula_trace.append(f"step_1b_result: post_trim_comps={n_valid}")

    # ── STEP 1c: Tiered evidence-source selection (S33-STEP3) ──────────────
    # Do not blend PPD Category A (standard open-market sale) and Category B
    # (repossession/power-of-sale/buy-to-let/corporate transfer) comps into
    # one pool — RICS treats forced/distressed-sale evidence as a different
    # basis of value to open-market evidence. Try the best available tier
    # first; fall through only when a tier has no usable evidence.
    #
    # Tier 1 — EIG auction hammer prices (not yet connected; see EIG_ENABLED
    #          near the top of this module). Real comparable hammer prices
    #          at known addresses — the closest available analogue to "what
    #          will THIS property fetch at auction," once this feed exists.
    # Tier 2 — PPD Category B. A real, present-day signal correlated with
    #          distressed/below-open-market conditions, though HMLR's own
    #          documentation is explicit this is a blended category, not a
    #          clean auction flag.
    # Tier 3 — PPD Category A. Open-market reference only. Always labelled
    #          as such — never presented as the central auction-outcome
    #          estimate.
    evidence_tier_used = EVIDENCE_TIER_PPD_CATEGORY_A
    _subject_postcode = (subject.get("postcode") or "").strip()
    eig_comps = (
        get_eig_comps_for_postcode(_subject_postcode, MAX_RADIUS_MILES)
        if EIG_ENABLED and _subject_postcode
        else []
    )

    cat_b_comps = [c for c in valid_comps if c.get("ppd_category_type") == "B"]
    cat_a_comps = [c for c in valid_comps if c.get("ppd_category_type") != "B"]

    if eig_comps:
        # Tier 1 not yet populated by any real call site — defensive only.
        valid_comps = eig_comps
        evidence_tier_used = EVIDENCE_TIER_EIG_AUCTION
        formula_trace.append(f"step_1c: evidence_tier=EIG n={len(eig_comps)}")
    elif len(cat_b_comps) >= 1:
        valid_comps = cat_b_comps
        evidence_tier_used = EVIDENCE_TIER_PPD_CATEGORY_B
        formula_trace.append(
            f"step_1c: evidence_tier=PPD_CATEGORY_B n={len(cat_b_comps)} "
            f"(repossession/power-of-sale proxy — {len(cat_a_comps)} "
            f"category-A comps available as open-market reference only, not used in central estimate)"
        )
        warnings.append(
            "Central estimate uses PPD Category B (repossession/power-of-sale) "
            "comparables as the closer analogue to a forced/auction-adjacent "
            "sale. This is a blended HMLR category, not a confirmed auction "
            "flag — treat with appropriate caution."
        )
    else:
        valid_comps = cat_a_comps
        evidence_tier_used = EVIDENCE_TIER_PPD_CATEGORY_A
        formula_trace.append(
            f"step_1c: evidence_tier=PPD_CATEGORY_A n={len(cat_a_comps)} "
            f"(no Category B or EIG evidence available — open-market reference only)"
        )
        warnings.append(
            "No distressed-sale (PPD Category B) or auction (EIG) comparables "
            "found nearby. This valuation is based on standard open-market "
            "sales only and should be treated as an UPPER-BOUND reference, "
            "not an expected auction outcome — open-market sale prices and "
            "auction hammer prices are different bases of value and "
            "commonly diverge."
        )

    n_valid = len(valid_comps)
    formula_trace.append(f"step_1c_result: evidence_tier={evidence_tier_used} comps={n_valid}")

    # ── STEP 2: Compute base_value via weighted median ────────────────────────
    insufficient_evidence = False
    base_value: Optional[float] = None
    # base_method reflects actual pipeline path
    base_method = (
        "weighted_median_hpi_normalised_like_for_like_iqr_trimmed_comps"
        if _iqr_removed_count > 0 or len(valid_comps) >= 6
        else "weighted_median_hpi_normalised_like_for_like_comps"
    )

    if base_valuation and float(base_valuation) > 5_000:
        base_value  = float(base_valuation)
        base_method = "external_override"
        formula_trace.append(f"step_2: base_value={base_value} method=external_override")

    elif n_valid == 0:
        insufficient_evidence = True
        formula_trace.append(f"step_2: insufficient_evidence — no valid comps within {MAX_RADIUS_MILES} miles after matching+trim")
        evidence_gaps.append(f"No valid sold comparables within {MAX_RADIUS_MILES} miles after type/tenure matching and IQR trim")

    elif n_valid < MIN_REQUIRED_COMPS and not fallback_allowed:
        insufficient_evidence = True
        formula_trace.append(f"step_2: insufficient_evidence — {n_valid} comps < min {MIN_REQUIRED_COMPS} and fallback_allowed=False")
        evidence_gaps.append(f"Only {n_valid} valid comps; minimum required = {MIN_REQUIRED_COMPS}")

    else:
        pairs = [(c["adjusted_value"], c["weight"]) for c in valid_comps]
        wm    = _weighted_median(pairs)
        if wm is None or wm <= 0:
            insufficient_evidence = True
            formula_trace.append("step_2: weighted_median returned None — insufficient evidence")
            evidence_gaps.append("Weighted median could not be computed — check comp weights")
        else:
            base_value = round(wm, 2)
            formula_trace.append(f"step_2: base_value={base_value} method={base_method} n_valid={n_valid}")
            if n_valid < MIN_REQUIRED_COMPS:
                warnings.append(f"Only {n_valid} valid comp(s) — below minimum {MIN_REQUIRED_COMPS}; ceiling is indicative with low confidence")
            if n_valid < PREFERRED_COMPS:
                warnings.append(f"Fewer than preferred {PREFERRED_COMPS} comps ({n_valid}) — confidence reduced")

    # ── STEP 3: Legal-pack value risk adjustment ─────────────────────────────
    formula_trace.append("step_3: legal-pack property-value risk adjustment")
    included_risks  = _process_legal_risks(legal_flags)
    risk_adj_factor = _legal_pack_adjustment_factor(included_risks)
    formula_trace.append(f"step_3_result: risk_adj_factor={risk_adj_factor} risks_included={len(included_risks)}")

    # ── STEP 4: Compute ceiling values ───────────────────────────────────────
    formula_trace.append("step_4: risk_adjusted_value = base_value × risk_adj_factor")

    ceiling_midpoint: Optional[float] = None   # backward-compat alias for risk_adjusted_value
    risk_adjusted_value: Optional[float] = None
    ceiling_low:      Optional[float] = None
    ceiling_high:     Optional[float] = None

    if not insufficient_evidence and base_value and base_value > 0:
        risk_adjusted_value = round(base_value * risk_adj_factor, 2)
        ceiling_midpoint    = risk_adjusted_value  # backward-compat alias
        formula_trace.append(f"step_4_result: risk_adjusted_value={risk_adjusted_value}")
    else:
        formula_trace.append("step_4: skipped — insufficient_evidence")

    # ── STEP 5: Confidence ───────────────────────────────────────────────────
    formula_trace.append("step_5: confidence calculation")

    _subj_tenure_raw = (subject.get("tenure") or "").strip()
    if _subj_tenure_raw.upper() == "F": _subj_tenure_raw = "freehold"
    if _subj_tenure_raw.upper() == "L": _subj_tenure_raw = "leasehold"
    tenure_unknown = not _subj_tenure_raw
    lease_unknown  = (
        "leasehold" in _subj_tenure_raw.lower()
        and subject.get("lease_length") is None
    )

    subj_lease_len = subject.get("lease_length")
    try:
        subj_lease_f = float(subj_lease_len) if subj_lease_len is not None else None
    except (TypeError, ValueError):
        subj_lease_f = None

    subj_band = _lease_band(subj_lease_f)
    short_lease = subj_lease_f is not None and subj_lease_f < 80
    same_band_lease_comps = [
        c for c in valid_comps if c.get("lease_band") is not None and c["lease_band"] == subj_band
    ] if short_lease else []
    short_lease_no_band = short_lease and len(same_band_lease_comps) == 0

    if insufficient_evidence:
        final_conf = 0.00
        conf_caps  = [{"cap": 0.00, "reason": "insufficient_evidence"}]
        conf_label = "Insufficient evidence"
    else:
        final_conf, conf_caps, conf_label = _calculate_confidence(
            valid_comps, subject, legal_flags, included_risks,
            tenure_unknown, lease_unknown, short_lease_no_band,
            evidence_tier_used,
        )

    formula_trace.append(f"step_5_result: confidence={final_conf} label={conf_label}")

    # ── STEP 6: Uncertainty band ─────────────────────────────────────────────
    formula_trace.append("step_6: uncertainty_band")
    u_band = _uncertainty_band(n_valid, conf_caps)
    formula_trace.append(f"step_6_result: uncertainty_band={u_band}")

    if risk_adjusted_value:
        ceiling_low  = round(risk_adjusted_value * (1 - u_band), 2)
        ceiling_high = round(risk_adjusted_value * (1 + u_band), 2)
        formula_trace.append(f"step_6: ceiling_low={ceiling_low} ceiling_high={ceiling_high} (from risk_adjusted_value={risk_adjusted_value})")

    # ── STEP 7: Warnings and audit ───────────────────────────────────────────
    if not valid_comps:
        evidence_gaps.append("No valid sold comparables available for base valuation")
    if tenure_unknown:
        evidence_gaps.append("Subject tenure unknown — confidence capped")
    if lease_unknown:
        evidence_gaps.append("Subject lease length unknown for leasehold — confidence capped")
    if short_lease_no_band:
        evidence_gaps.append("Subject lease < 80 years and no comparable comps in same lease band")

    any_hpi_missing = any(
        "hpi_adjustment missing" in w
        for c in valid_comps
        for w in c.get("audit_warnings", [])
    )
    if any_hpi_missing:
        warnings.append("HPI time adjustment missing for one or more comps — time_adjustment=1.00 assumed")
        assumptions.append("time_adjustment=1.00 where HPI data absent")

    if base_method == "external_override":
        assumptions.append("base_value from external_override parameter — comparables not primary base")

    # ── STEP 8: Status ───────────────────────────────────────────────────────
    if insufficient_evidence:
        status = "insufficient_evidence"
    elif n_valid < MIN_REQUIRED_COMPS:
        status = "degraded_low_comps"
    else:
        status = "ok"

    return {
        "valuation_type":    "red_book_style_paper_valuation_similarity",
        "not_rics_valuation": True,
        "status":            status,
        "base": {
            "method":                  base_method,
            "value":                   base_value,
            "minimum_required_comps":  MIN_REQUIRED_COMPS,
            "preferred_required_comps": PREFERRED_COMPS,
            "valid_comparable_count":  n_valid,
            "excluded_comparable_count": len(excluded_comps),
            "pre_iqr_trim_count":      _pre_trim_count,
            "post_iqr_trim_count":     n_valid,
            # S33-STEP3: which evidence tier actually produced base_value.
            # See EVIDENCE_TIER_* constants near top of module. Frontend
            # should surface this plainly — "open-market reference only" vs
            # "distressed-sale proxy" carry materially different confidence.
            "evidence_tier":           evidence_tier_used,
        },
        # Explicit semantic fields — unambiguous across Verdict and Workbench.
        # comparable_valuation: weighted median of HPI-adjusted like-for-like comps.
        # risk_adjusted_value:  comparable_valuation × legal_pack_risk_factor.
        #   Verdict:   risk_factor = 1.0  → risk_adjusted_value == comparable_valuation
        #   Workbench: risk_factor < 1.0  → risk_adjusted_value < comparable_valuation
        # ceiling_range.low/high are derived from risk_adjusted_value only.
        "comparable_valuation": base_value,
        "risk_adjusted_value":  risk_adjusted_value,
        "comparables": {
            "radius_miles": PRIMARY_RADIUS_MILES,
            "valid":        valid_comps,
            "excluded":     excluded_comps,
        },
        "legal_pack_value_risks": {
            "method":            "property_value_risk_adjustment_only",
            "adjustment_factor": risk_adj_factor,
            "adjusted_value":    risk_adjusted_value,  # = comparable_valuation × risk_factor
            "risks":             included_risks,
        },
        "valuation_range": {
            "low":              ceiling_low,
            # midpoint = risk_adjusted_value (backward-compat alias).
            # Verdict: equals base_value (risk_factor=1). Workbench: equals base_value × risk_factor.
            # Prefer comparable_valuation / risk_adjusted_value for new code.
            "midpoint":         ceiling_midpoint,
            "high":             ceiling_high,
            "uncertainty_band": u_band,
        },
        "confidence": {
            "raw":   None,   # not surfaced separately — final is authoritative
            "caps":  conf_caps,
            "final": final_conf,
            "label": conf_label,
        },
        "audit": {
            # Phase 2 doctrine fields
            "comparable_method":   "red_book_style_comparable_evidence",
            "not_rics_valuation":  True,
            "formal_valuation":    False,
            "decision_support_only": True,
            "comps_source_path":   "area_json.housing.soldComps",
            "sold_comps_count":    len(sold_comps),
            "type_tenure_matched_count": _pre_trim_count,
            "pre_trim_comp_count": _pre_trim_count,
            "post_trim_comp_count": n_valid,
            "valid_comparable_count":    n_valid,
            "excluded_comparable_count": len(excluded_comps),
            "excluded_reasons_summary": {
                r: sum(1 for e in excluded_comps if r in (e.get("reason") or ""))
                for r in {(e.get("reason") or "unknown").split(" ")[0]
                          for e in excluded_comps}
            },
            "hpi_normalisation_summary": {
                "hpi_used_count":    _hpi_used_count,
                "hpi_missing_count": _hpi_missing_count,
                "hpi_source_fields": ["hpi_multiplier", "hpi_adjustment", "time_adjustment"],
                "time_adjustment_assumptions": (
                    "time_adjustment=1.00 where hpi_multiplier absent or ==1.0"
                    if _hpi_missing_count > 0 else "all comps HPI-adjusted"
                ),
            },
            "iqr_trim_summary": {
                "applied":             _iqr_removed_count > 0,
                "iqr_lower_bound":     round(_iqr_lower, 2) if _iqr_lower is not None else None,
                "iqr_upper_bound":     round(_iqr_upper, 2) if _iqr_upper is not None else None,
                "pre_trim_count":      _pre_trim_count,
                "post_trim_count":     n_valid,
                "removed_outliers_count": _iqr_removed_count,
                "removed_outlier_reasons": _iqr_removed_reasons,
            },
            "base_method":          base_method,
            "base_value":           base_value,
            "comparable_valuation": base_value,        # explicit alias: comparable evidence result
            "risk_adjusted_value":  risk_adjusted_value,  # base_value × risk_factor
            "source_decision": (
                "computed_from_sold_comps" if base_value and not insufficient_evidence
                else "external_override" if base_method == "external_override"
                else "insufficient_evidence"
            ),
            "fallback_used":   insufficient_evidence,
            # Existing audit fields
            "assumptions":     assumptions,
            "evidence_gaps":   evidence_gaps,
            "warnings":        warnings,
            "formula_trace":   formula_trace,
            "version":         VERSION,
        },
        # Legacy compatibility fields — mapped from new structure for
        # existing app.py consumers that read these keys.
        # ceiling_range mirrors valuation_range for backward compat.
        "ceiling_range": {
            "low":  int(round(ceiling_low))  if ceiling_low  else None,
            "high": int(round(ceiling_high)) if ceiling_high else None,
        },
        "base_valuation": int(round(base_value)) if base_value else None,
        "base_method":    base_method,
        "confidence_final": final_conf,
        "strategy_used":  strategy,
        "investment_value_note": (
            "Red-book-style paper valuation similarity — not a RICS valuation, "
            "not financial advice. Decision-support only."
        ),
        # Acquisition costs: informational only — NOT in ceiling formula
        "acquisition_costs": None,
        "excluded_from_ceiling": EXCLUDED_FROM_CEILING,
    }


# =============================================================================
# BACKFILL / NORMALISATION HELPER
# ensure_ceiling_owned_objects(summary_json, area_json, financials_json, flags, current_bid)
#
# Called before any page payload or API response is returned.
# Rules:
#  - preserve existing verdict_ceiling if valid (has valuation_range.midpoint > 0)
#  - preserve existing workbench_ceiling if valid AND <= verdict ceiling
#  - compute verdict_ceiling if missing and enough comp data exists
#  - compute workbench_ceiling from verdict_ceiling and active flags
#  - compute financial_current_standing from current_bid and workbench_ceiling
#  - clamp workbench_ceiling to verdict_ceiling (hard invariant)
#  - use legacy summary_json.ceiling only as a safety cap if no owned objects
#  - mark missing-data state explicitly if not enough source data
#  - write audit notes: computed | preserved | backfilled | capped | missing_data
#  - acquisition costs do not enter verdict or workbench ceiling
#  - bid_ceiling is not canonical valuation
# =============================================================================

def ensure_ceiling_owned_objects(
    summary_json:     dict,
    area_json:        Optional[dict] = None,
    financials_json:  Optional[dict] = None,
    legal_flags:      Optional[list[dict]] = None,
    current_bid:      Optional[float] = None,
    strategy:         str = "BTL",
    subject:          Optional[dict] = None,
) -> dict:
    """
    Normalise summary_json to contain verdict_ceiling, workbench_ceiling,
    and financial_current_standing as separate owned objects.

    Returns the mutated summary_json dict.
    Does NOT write to the database — caller is responsible for persistence.
    """
    summary_json  = summary_json  if isinstance(summary_json,  dict) else {}
    area_json     = area_json     if isinstance(area_json,     dict) else {}
    financials_json = financials_json if isinstance(financials_json, dict) else {}
    legal_flags   = legal_flags   if isinstance(legal_flags,   list) else []
    subject       = subject       if isinstance(subject,        dict) else {}

    audit_notes: list[str] = []

    # ── 0. Extract sold comps from area_json ──────────────────────────────
    _housing   = area_json.get("housing") or {}
    _sold_comps = _housing.get("soldComps") or _housing.get("value") or []
    _sold_comps = _sold_comps if isinstance(_sold_comps, list) else []

    # ── 1. Validate existing verdict_ceiling ─────────────────────────────
    _existing_vc = summary_json.get("verdict_ceiling")
    _vc_valid = (
        isinstance(_existing_vc, dict)
        and (
            # Accept via explicit comparable_valuation field (new objects)
            (_existing_vc.get("comparable_valuation") or 0) > 0
            or (
                # Accept via valuation_range.midpoint (backward compat for older objects)
                isinstance(_existing_vc.get("valuation_range"), dict)
                and (_existing_vc["valuation_range"].get("midpoint") or 0) > 0
            )
        )
    )
    # A verdict_ceiling is only trusted if it was computed from sold comps
    # (no _legacy_source flag). Legacy-sourced verdicts must be upgraded
    # when comps are now available — they were built from a pre-area v1
    # ceiling that may have used a yield-based or otherwise incorrect base.
    _vc_is_legacy = _vc_valid and bool(_existing_vc.get("_legacy_source"))

    if _vc_valid and not _vc_is_legacy:
        # Non-legacy computed verdict — preserve as-is.
        verdict = _existing_vc
        audit_notes.append("verdict_ceiling: preserved (existing non-legacy computed object)")
    else:
        # Either no valid verdict_ceiling yet, or the existing one is _legacy_source=True
        # (built from a pre-area v1 base, not from sold comps).
        #
        # Source order:
        #   A. Attempt relational comp recompute if area_json.housing.soldComps available.
        #   B. If recompute produces midpoint > 0 → use it, clear _legacy_source.
        #   C. If recompute fails or no comps → fall back to legacy summary_json.ceiling.
        #   D. If neither → missing_data.
        #
        # This means legacy ceilings are automatically upgraded as soon as comps exist
        # and the engine can produce a valid midpoint (e.g. after Fix A normalises LR codes).

        # ── A. Attempt relational comp recompute ──────────────────────────
        _comp_verdict: Optional[dict] = None
        _comp_excluded_reasons: dict  = {}
        if _sold_comps:
            _comp_verdict = calculate_verdict_ceiling(
                sold_comps=_sold_comps,
                subject=subject,
                strategy=strategy,
                fallback_allowed=True,
            )

        _comp_mid = (
            (_comp_verdict.get("valuation_range") or {}).get("midpoint") or 0
            if _comp_verdict else 0
        )

        if _comp_mid and _comp_mid > 0:
            # ── B. Comp recompute succeeded ───────────────────────────────
            verdict = _comp_verdict
            # Ensure no _legacy_source leaks in
            verdict.pop("_legacy_source", None)
            _vc_audit = verdict.setdefault("audit", {})
            _vc_audit["source_decision"]          = "computed_from_sold_comps"
            _vc_audit["sold_comps_count"]         = len(_sold_comps)
            _vc_audit["valid_comparable_count"]   = len(
                (verdict.get("comparables") or {}).get("valid") or []
            )
            _vc_audit["excluded_comparable_count"] = len(
                (verdict.get("comparables") or {}).get("excluded") or []
            )
            _vc_audit["fallback_used"] = False
            audit_notes.append(
                f"verdict_ceiling: computed_from_sold_comps "
                f"midpoint={_comp_mid} comps={len(_sold_comps)} "
                f"valid={_vc_audit['valid_comparable_count']} "
                f"excluded={_vc_audit['excluded_comparable_count']}"
            )

        else:
            # ── C. Comp recompute failed or no comps — fall back to legacy ─
            # Collect excluded reasons for the audit trail.
            if _comp_verdict:
                for _ex in ((_comp_verdict.get("comparables") or {}).get("excluded") or []):
                    _r = (_ex or {}).get("reason", "unknown")
                    _comp_excluded_reasons[_r] = _comp_excluded_reasons.get(_r, 0) + 1

            _legacy   = summary_json.get("ceiling") or {}
            _leg_base: Optional[float] = None
            _leg_lo:   Optional[float] = None
            _leg_hi:   Optional[float] = None
            try:
                _lb = _legacy.get("base_valuation")
                if _lb and float(_lb) > 5000:
                    _leg_base = float(_lb)
                _lcr  = _legacy.get("ceiling_range") or _legacy.get("valuation_range") or {}
                _lclo = _lcr.get("low")
                _lchi = _lcr.get("high")
                if _lclo and float(_lclo) > 5000:
                    _leg_lo = float(_lclo)
                if _lchi and float(_lchi) > 5000:
                    _leg_hi = float(_lchi)
            except (TypeError, ValueError):
                pass

            _src_decision = (
                "legacy_fallback_comp_recompute_failed"
                if (_sold_comps and _comp_verdict is not None)
                else "legacy_fallback_no_comps"
            )

            if _leg_base and _leg_base > 5000:
                _ub   = 0.05
                _v_lo = _leg_lo if _leg_lo else round(_leg_base * (1 - _ub), 2)
                _v_hi = _leg_hi if _leg_hi else round(_leg_base * (1 + _ub), 2)
                verdict = {
                    "_ceiling_type":  "verdict",
                    "_legacy_source": True,
                    "status":         "ok",
                    "base": {
                        "value":  _leg_base,
                        "method": _legacy.get("base_method", "legacy_ceiling"),
                    },
                    "base_valuation": int(round(_leg_base)),
                    "base_method":    _legacy.get("base_method", "legacy_ceiling"),
                    "valuation_range": {
                        "low":              round(_v_lo, 2),
                        "midpoint":         round(_leg_base, 2),
                        "high":             round(_v_hi, 2),
                        "uncertainty_band": _ub,
                    },
                    "ceiling_range": {
                        "low":  int(round(_v_lo)),
                        "high": int(round(_v_hi)),
                    },
                    "comparables": {"radius_miles": PRIMARY_RADIUS_MILES, "valid": [], "excluded": []},
                    "legal_pack_value_risks": {
                        "method":           "property_value_risk_adjustment_only",
                        "adjustment_factor": 1.0, "adjusted_value": None, "risks": [],
                    },
                    "confidence": _legacy.get("confidence") or {"final": 0.45, "label": "Low confidence"},
                    "audit": {
                        "source_decision":            _src_decision,
                        "sold_comps_count":           len(_sold_comps),
                        "excluded_reasons_summary":   _comp_excluded_reasons,
                        "fallback_used":              True,
                        "assumptions":   ["base_value from legacy summary_json.ceiling"],
                        "evidence_gaps": [],
                        "warnings": [
                            f"verdict_ceiling built from legacy ceiling ({_src_decision}). "
                            "Comps: " + (
                                f"{len(_sold_comps)} supplied, all excluded — see excluded_reasons_summary."
                                if _comp_excluded_reasons
                                else "none available."
                            ) + " Re-fetch area or re-analyse to recompute from sold comps."
                        ],
                        "formula_trace": [
                            f"legacy_source: base_valuation={_leg_base} from summary_json.ceiling"
                        ],
                        "version": VERSION,
                    },
                    "acquisition_costs":     None,
                    "excluded_from_ceiling": EXCLUDED_FROM_CEILING,
                }
                audit_notes.append(
                    f"verdict_ceiling: {_src_decision} "
                    f"base={_leg_base} lo={_v_lo} hi={_v_hi} "
                    f"excluded_reasons={_comp_excluded_reasons}"
                )

            else:
                # ── D. No legacy base and no usable comps ─────────────────
                verdict = {
                    "_ceiling_type": "verdict",
                    "status":        "missing_data",
                    "valuation_range": {
                        "low": None, "midpoint": None, "high": None, "uncertainty_band": None
                    },
                    "ceiling_range":   {"low": None, "high": None},
                    "base":            {"value": None, "method": "none"},
                    "confidence": {
                        "final": 0.0,
                        "caps":  [{"cap": 0.0, "reason": "no_data"}],
                        "label": "Insufficient evidence",
                    },
                    "audit": {
                        "source_decision":          "missing_data",
                        "sold_comps_count":         len(_sold_comps),
                        "excluded_reasons_summary": _comp_excluded_reasons,
                        "fallback_used":            False,
                        "warnings": [
                            "no valid sold comps and no legacy ceiling — missing_data. "
                            + (
                                f"Comps supplied: {len(_sold_comps)}, "
                                f"excluded reasons: {_comp_excluded_reasons}."
                                if _sold_comps else "No comps available."
                            )
                        ],
                        "version": VERSION,
                    },
                    "acquisition_costs":     None,
                    "excluded_from_ceiling": EXCLUDED_FROM_CEILING,
                }
                audit_notes.append(
                    f"verdict_ceiling: missing_data — "
                    f"sold_comps={len(_sold_comps)} excluded={_comp_excluded_reasons}"
                )

        summary_json["verdict_ceiling"] = verdict

    # ── 2. Validate / compute workbench_ceiling ───────────────────────────
    _existing_wb = summary_json.get("workbench_ceiling")
    # Use comparable_valuation as the authoritative verdict base for workbench guard.
    # Falls back to valuation_range.midpoint for pre-fix objects.
    _v_comparable = (
        verdict.get("comparable_valuation")
        or (verdict.get("base") or {}).get("value")
        or (verdict.get("valuation_range") or {}).get("midpoint")
        or 0
    )
    _v_mid = _v_comparable  # backward-compat alias used in clamp below
    # Workbench is valid if its risk_adjusted_value (or midpoint) <= verdict comparable_valuation + £1
    _wb_rav = _existing_wb.get("risk_adjusted_value") if isinstance(_existing_wb, dict) else None
    _wb_check_val = (
        _wb_rav
        if _wb_rav is not None
        else ((_existing_wb or {}).get("valuation_range") or {}).get("midpoint")
    ) or 0
    # A workbench_ceiling is valid only if:
    #   (a) it is a dict with a valid valuation_range,
    #   (b) its risk_adjusted_value is <= verdict comparable_valuation + £1 tolerance,
    #   (c) it carries market_consequence_adjustments — absent means it was persisted
    #       before the r6 segment engine and must be recomputed to show correct waterfall.
    _wb_has_segments = (
        isinstance(_existing_wb, dict)
        and isinstance(_existing_wb.get("market_consequence_adjustments"), dict)
    )
    _wb_valid = (
        isinstance(_existing_wb, dict)
        and isinstance(_existing_wb.get("valuation_range"), dict)
        and _wb_check_val <= _v_comparable + 1  # clamp tolerance £1
        and _wb_check_val > 0
        and _wb_has_segments  # stale pre-r6 objects must be recomputed
    )

    if _wb_valid:
        workbench = _existing_wb
        audit_notes.append("workbench_ceiling: preserved (existing valid and <= verdict)")
    else:
        workbench = calculate_workbench_ceiling(
            verdict_ceiling=verdict,
            active_legal_flags=legal_flags,
        )
        audit_notes.append(
            f"workbench_ceiling: computed from verdict × "
            f"active_flag_risk_factor={workbench.get('legal_pack_value_risks', {}).get('adjustment_factor')}"
        )
        summary_json["workbench_ceiling"] = workbench

    # Hard clamp: workbench must never exceed verdict (re-enforce after any path)
    _v_vr = verdict.get("valuation_range")   or {}
    _w_vr = workbench.get("valuation_range") or {}
    _v_lo = _v_vr.get("low")  or 0
    _v_md = _v_vr.get("midpoint") or 0
    _v_hi = _v_vr.get("high") or 0
    _w_lo = _w_vr.get("low")  or 0
    _w_md = _w_vr.get("midpoint") or 0
    _w_hi = _w_vr.get("high") or 0

    _clamped = False
    if _v_md > 0 and _w_md > _v_md:
        workbench["valuation_range"]["midpoint"] = _v_md
        workbench["valuation_range"]["low"]  = _v_lo
        workbench["valuation_range"]["high"] = _v_hi
        if workbench.get("ceiling_range"):
            workbench["ceiling_range"]["low"]  = int(round(_v_lo)) if _v_lo else None
            workbench["ceiling_range"]["high"] = int(round(_v_hi)) if _v_hi else None
        workbench["_hard_clamped_to_verdict"] = True
        _clamped = True
        audit_notes.append(f"workbench_ceiling: hard-clamped to verdict (wb_mid={_w_md} > v_mid={_v_md})")

    summary_json["workbench_ceiling"] = workbench

    # ── 3. financial_current_standing ────────────────────────────────────
    # Always recompute from current workbench (current_bid may change)
    standing = calculate_financial_standing(workbench, current_bid=current_bid)
    standing["_audit_note"] = f"backfill_helper: {'; '.join(audit_notes)}"
    summary_json["financial_current_standing"] = standing

    # ── 4. Legacy alias ───────────────────────────────────────────────────
    # summary_json.ceiling kept as alias = workbench_ceiling for backward compat.
    # It does NOT override owned objects.
    if "ceiling" not in summary_json or _clamped:
        summary_json["ceiling"] = workbench   # alias only

    return summary_json
