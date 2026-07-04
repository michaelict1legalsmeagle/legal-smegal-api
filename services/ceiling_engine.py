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
    from sold comparables within MAX_RADIUS_MILES (3.0 miles).
    Comps beyond PRIMARY_RADIUS_MILES (0.5mi) are NOT excluded — they are
    included with reduced weight via EXTENDED_DISTANCE_BANDS (S33-STEP1,
    2026-06-21). PRIMARY_RADIUS_MILES now drives confidence labelling only.

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
- Primary comparable universe: distance_miles <= MAX_RADIUS_MILES (3.0),
  weighted by distance via EXTENDED_DISTANCE_BANDS (see DISTANCE RULE below).
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

# S37-SEGMENT-CAPS (2026-07-03): per-segment sub-caps, added because a real
# 32-deal audit showed residual_marketability_risk and lender_certifiability_
# risk — the two catch-all segments almost every _SEGMENT_RULES entry routes
# into — accumulate near-unbounded across 20-30+ flags per deal (observed:
# residual median 31.5%, max 40%; lender median 19.8%, max 34.9% — EITHER ONE
# ALONE already exceeds the old 35% global cap on a typical real deal, before
# any other segment is even added). direct_cure_cost and indemnity_insurance_
# cost stayed bounded by comparison (median 5.6% and 3.6%), because far fewer
# rules route into them.
#
# These specific cap VALUES are a calibration proposal, not a derived fact —
# data shows what the current system produces, not what the correct ceiling
# is. Grounded near the observed median/p75 for the two overloaded segments
# (constraining them well below their runaway max) and left close to the
# already-bounded range for the other three. Flagged for review before this
# is treated as final, not asserted as settled.
_SEGMENT_CAPS: dict[str, float] = {
    "direct_cure_cost":           0.10,
    "delay_finance_drag":         0.15,
    "indemnity_insurance_cost":   0.08,
    "lender_certifiability_risk": 0.20,
    "residual_marketability_risk": 0.20,
}

# ═════════════════════════════════════════════════════════════════════════════
# S40-CALIBRATION-METADATA (2026-07-04) — Phase A of the credibility-theory
# calibration governance plan.
#
# Discipline: actuarial credibility theory (Bühlmann/Bühlmann-Straub) as the
# governing methodology — converged on independently by two separate research
# passes (2026-07-01 session; 2026-07-03 deep research). Under that
# methodology, calibration values start as structured expert priors and are
# formally blended toward real observed outcomes as the outcome dataset grows
# (Phase E), with the blend weight Z = n/(n+k) always disclosed. The
# non-negotiable prerequisite is honest labelling of what each number
# currently IS — expert prior vs empirically credible — which is what this
# registry provides. Modelled on RICS "material valuation uncertainty"
# disclosure (label the interim methodology, never present it with false
# precision) and SR 11-7 model-governance metadata.
#
# calibration_status values:
#   "expert_prior"          — grounded reasoning/observed-distribution shape,
#                             zero outcome backtesting. Honest default today.
#   "partially_credible"    — blended with real outcomes, Z below full
#                             credibility (set by Phase E, never by hand).
#   "empirically_credible"  — Z at/near 1.0 against the outcome dataset.
# source_type values mirror the _SEGMENT_RULES doctrine:
#   "observed_distribution" | "structured_elicitation" | "external_source"
#   | "backtested"
# review_trigger: the explicit condition on which this value MUST be
#   re-examined — no silent permanence.
# ═════════════════════════════════════════════════════════════════════════════
CALIBRATION_METADATA: dict[str, dict] = {
    "segment_caps": {
        "values": dict(_SEGMENT_CAPS),
        "calibration_status": "expert_prior",
        "source_type": "observed_distribution",
        "basis": (
            "Grounded in the per-segment cumulative-fraction distribution "
            "observed across 32 real analysed deals (2026-07-03 audit): caps "
            "set near observed median/p75 for the two overloaded catch-all "
            "segments (residual median 31.5% max 40%; lender median 19.8% "
            "max 34.9%), near observed range for the three bounded segments. "
            "NOT derived from completed-transaction outcomes."
        ),
        "sample_size_at_calibration": 32,
        "calibrated_at": "2026-07-04",
        "review_trigger": (
            "Recalibrate via credibility blend once >=50 deals have recorded "
            "auction outcomes in deal_outcomes; review immediately if floored "
            "share of live deals exceeds 20% or falls below 1%."
        ),
    },
    "marginal_decay_rate": {
        "values": {"decay_per_rank": 0.5},
        "calibration_status": "expert_prior",
        "source_type": "observed_distribution",
        "basis": (
            "Geometric decay (0.5**rank) across descending capped segment "
            "totals. Chosen so the theoretical worst case (all 5 segments "
            "simultaneously at cap) lands at 35.5%, marginally above the 35% "
            "global backstop — i.e. the backstop binds only at the true "
            "extreme. Verified against the same 32-deal corpus: floored share "
            "fell from 91% to 6% and factor spread tracks severity mix. A "
            "correlation-matrix aggregation (Solvency II style) is the "
            "identified successor candidate, deliberately DEFERRED until "
            "deal_outcomes data exists to champion-challenger the two — "
            "replacing a freshly-verified mechanism on theoretical preference "
            "alone would be uncontrolled change."
        ),
        "sample_size_at_calibration": 32,
        "calibrated_at": "2026-07-04",
        "review_trigger": (
            "Champion-challenger against correlation-matrix aggregation once "
            ">=50 deals have recorded outcomes in deal_outcomes."
        ),
    },
    "global_backstop": {
        "values": {"max_total_value_risk_adj": MAX_TOTAL_VALUE_RISK_ADJ},
        "calibration_status": "expert_prior",
        "source_type": "observed_distribution",
        "basis": (
            "Pre-existing 35% ceiling on total legal-pack value reduction. "
            "Predates the outcome dataset; retained as defense-in-depth "
            "behind the segment caps and marginal decay, which now bind "
            "first in all but extreme cases."
        ),
        "sample_size_at_calibration": 0,
        "calibrated_at": "pre-2026-06",
        "review_trigger": (
            "Re-examine against observed ceiling-vs-hammer deltas once "
            ">=50 deal outcomes recorded."
        ),
    },
}


def get_calibration_disclosure() -> dict:
    """
    Machine-readable calibration disclosure for downstream surfaces (Verdict /
    Workbench / Deal Report). Returns the full metadata registry plus a
    single plain-English summary line suitable for direct UI display.

    S40 (2026-07-04): exists so no risk-adjusted figure ships with false
    precision — the RICS material-uncertainty principle applied to model
    calibration. A prior audit pattern in this codebase (confidence labels
    computed then silently discarded, 2026-06-25; _resolved_flags persisted
    then never read, 2026-07-03) showed that metadata which nothing consumes
    is metadata that silently dies — hence a dedicated accessor rather than
    a bare module-level dict, and a pre-built summary line so surfacing it
    costs the frontend one field read.
    """
    statuses = {k: v["calibration_status"] for k, v in CALIBRATION_METADATA.items()}
    all_expert_prior = all(s == "expert_prior" for s in statuses.values())
    if all_expert_prior:
        summary = (
            "Risk-adjustment calibration is currently expert-prior: grounded "
            "in the observed distribution of 32 real analysed deals, not yet "
            "validated against completed auction outcomes. Calibration "
            "shifts automatically toward real outcomes as they are recorded."
        )
    else:
        summary = (
            "Risk-adjustment calibration is partially outcome-validated; "
            "see per-parameter status for detail."
        )
    return {
        "summary": summary,
        "parameters": CALIBRATION_METADATA,
        "methodology": "buhlmann_straub_credibility_pending_outcomes",
    }


# ═════════════════════════════════════════════════════════════════════════════
# S40-CREDIBILITY-BLEND (2026-07-04) — Phase E.
#
# Bühlmann-style credibility weighting: blended = Z*observed + (1-Z)*prior,
# Z = n / (n + K). Standard actuarial form (Bühlmann 1967; Bühlmann-Straub
# 1970) for exactly this situation: a risk class with real stakes but thin
# experience data, where the class's own data earns weight as it accumulates
# rather than being trusted fully from day one or ignored until an arbitrary
# threshold.
#
# K (the credibility constant) controls how fast real data earns weight:
# at n = K observations, Z = 0.5 (evidence and prior weighted equally).
# K = 50 here, matching the review_trigger threshold in CALIBRATION_METADATA:
# at the 50-outcome review point the blend is exactly half-and-half, which is
# when human review of the shifted values is mandated. K itself is labelled
# expert-prior — it is a governance choice (how cautious to be with early
# data), not an empirical estimate, and standard practice derives it
# empirically (variance ratio) only once far more data exists.
#
# HONEST CURRENT STATE: deal_outcomes has 0 rows at implementation time, so
# Z = 0 and blended == prior exactly. This function changes nothing today —
# by design. It exists so the shift toward real outcomes is automatic,
# formulaic, and disclosed (Z is returned, not hidden) rather than requiring
# a future manual recalibration that history says would be skipped.
# ═════════════════════════════════════════════════════════════════════════════
CREDIBILITY_K: float = 50.0


def credibility_blend(prior: float, observed: Optional[float], n_observations: int,
                      k: float = CREDIBILITY_K) -> dict:
    """
    Blend an expert-prior calibration value with an observed empirical value
    using Bühlmann credibility weighting.

    Returns {value, z, n, prior, observed, status} — Z always disclosed.
    If observed is None or n_observations == 0, returns the prior with Z=0
    and status 'expert_prior' (never fabricates an observation).
    """
    if observed is None or n_observations <= 0:
        return {"value": prior, "z": 0.0, "n": max(0, n_observations),
                "prior": prior, "observed": None, "status": "expert_prior"}
    z = n_observations / (n_observations + k)
    blended = z * observed + (1.0 - z) * prior
    if z >= 0.9:
        status = "empirically_credible"
    elif z > 0.0:
        status = "partially_credible"
    else:
        status = "expert_prior"
    return {"value": round(blended, 6), "z": round(z, 4), "n": n_observations,
            "prior": prior, "observed": observed, "status": status}

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
      "water search", "chancel search", "chancel repair", "chancel liability",
      "coal search", "mining search", "missing search", "no search"],
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
    (["forfeiture", "breach of covenant", "breach of lease", "prohibition on dealings"],
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

    # ── Restrictive covenant — UNKNOWN CONTENT or UNUSUAL OBLIGATION ─────────
    # (S37-COVENANT-SPLIT, 2026-07-03): the single most-firing rule in the
    # engine (58/660 real flags, 8.8%) was matching one flat 5% fraction to
    # both genuinely uncertain/onerous covenants AND fully-disclosed standard
    # estate-restriction wording. Split on real observed vocabulary: content
    # that is unknown, undisclosed, or incomplete carries genuine title/
    # insurability uncertainty; a named obligation with real financial/value
    # consequence (clawback, minimum resale value, land surrender, ongoing
    # cost-share) is not standard regardless of disclosure. This branch is
    # checked FIRST — an ambiguous covenant defaults to elevated scrutiny,
    # not to the standard tier. Fractions unchanged from the original rule.
    (["content unknown", "content not visible", "not disclosed", "not provided",
      "not yet provided", "unknown scope", "unknown content", "extent unknown",
      "full effect unknown", "schedule d unknown", "redacted",
      "document incomplete", "pages missing", "copy incomplete", "missing copy",
      "clawback", "minimum house value", "minimum value restriction",
      "land surrender", "structural complexity", "maintenance costs",
      "boundary wall maintenance", "personal obligation",
      "undisclosed encumbrances", "transfer of part", "retained land",
      "covenants not disclosed", "covenants binding", "historic covenants",
      "multiple restrictive covenants"],
     {"indemnity_insurance_cost": 0.010, "lender_certifiability_risk": 0.020,
      "residual_marketability_risk": 0.020}),

    # ── Restrictive covenant — STANDARD, NAMED, FULLY DISCLOSED ──────────────
    # Ordinary estate-restriction covenants with fully known, named content
    # (no trade, no nuisance, residential-only, detached-only) are present on
    # the large majority of English residential titles and do not
    # differentiate this property from any other. Fraction reused verbatim
    # from Rule 8 (indemnity insurance available) — same segments, same
    # "known, low-cost, routinely insurable" profile. Not invented.
    (["no trade or business", "no business use", "no nuisance", "no annoyance",
      "residential use only", "private dwellinghouse", "detached or semi-detached",
      "no right to light", "no alcohol licence", "building line", "setback",
      "restrictive covenant", "restrictive covenants", "covenant"],
     {"indemnity_insurance_cost": 0.012, "residual_marketability_risk": 0.008}),

    # ── Rights of way / easement defects ─────────────────────────────────────
    (["right of way", "rights of way", "easement", "right to light", "access rights",
      "ransom strip", "third-party right", "right to use", "party wall"],
     {"lender_certifiability_risk": 0.020, "residual_marketability_risk": 0.025}),

    # ── Planning / building control ───────────────────────────────────────────
    (["planning", "building regulations", "building control", "planning permission",
      "listed building", "conservation area", "enforcement notice",
      "article 4", "permitted development"],
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
    (["contamination", "environmental risk", "environmental liability",
      "environmental indemnity", "flood risk", "subsidence"],
     {"lender_certifiability_risk": 0.025, "residual_marketability_risk": 0.035}),

    # ── S35-RISK-COVERAGE (2026-06-25): three new entries closing the fallback
    # gap found this session (16 Penmanor: 12/17 substantive flags fell to the
    # generic severity table). Each entry's fractions are COPIED VERBATIM from
    # an existing, already-trusted rule for a structurally analogous defect —
    # no new fraction values invented, per explicit instruction not to fabricate
    # calibration data that doesn't exist.

    # Unlettable-property defects (EPC G / no heating system) — ongoing,
    # known, quantifiable defects requiring works before the property can be
    # let, not just missing paperwork. Closest analogue: "Section 20 / major
    # works notice" — a known, costed remediation requirement that delays
    # completion confidence and dents resale appeal. Fractions copied exactly.
    (["epc rating g", "epc rating f", "cannot be let", "no heating system",
      "no central heating", "electric heaters only"],
     {"delay_finance_drag": 0.015, "residual_marketability_risk": 0.020}),

    # Missing standard disclosure/contract documents (TA6, draft contract,
    # transfer deed, special conditions of sale) — unknown contractual terms
    # or seller disclosures, same shape as a missing search: needs to be
    # obtained before completion confidence, modest direct cost, real delay
    # risk. Closest analogue: "Missing searches" rule. Fractions copied exactly.
    (["ta6", "ta10", "draft contract", "transfer deed", "special conditions of sale",
      "property information form"],
     {"direct_cure_cost": 0.008, "delay_finance_drag": 0.012}),

    # Title/ownership administrative defects not yet rising to a registered
    # title DEFECT (e.g. multiple titles to transfer, estate/probate sale
    # without an established title problem) — administrative complexity that
    # could complicate completion but isn't itself a defect in the title.
    # Closest analogue: "Management pack / service charge info missing" —
    # administrative/process risk with modest cure cost, delay, and a small
    # lender-certifiability component. Fractions copied exactly.
    (["separate title", "dual title", "two titles", "probate", "estate sale",
      "deceased proprietor"],
     {"direct_cure_cost": 0.006, "delay_finance_drag": 0.010, "lender_certifiability_risk": 0.008}),

    # ─────────────────────────────────────────────────────────────────────────
    # S36-RISK-COVERAGE (2026-07-01): ten new rules closing the 59.6% fallback
    # gap confirmed against 5 live deals (138 substantive flags measured).
    # Each rule's fractions are COPIED VERBATIM from an existing, already-
    # trusted rule for the structurally closest analogous defect — no new
    # fraction values invented. Fraction sources documented per rule.
    # ─────────────────────────────────────────────────────────────────────────

    # ── Occupier / vacant possession risk ────────────────────────────────────
    # Unknown occupiers, tenancy uncertainty, receiver has no tenancy info.
    # Market consequence: legal process to achieve vacant possession (delay +
    # direct cost) and residual risk that property cannot be cleared.
    # Fractions: arrears rule (direct_cure 0.015), section_20 (delay 0.015),
    # contamination rule (residual 0.035).
    (["vacant possession", "unknown occupier", "tenancy agreement",
      "tenancy schedule", "receiver has none", "no tenancy",
      "occupiers or tenancies", "occupancy status", "missing tenancy"],
     {"direct_cure_cost": 0.015, "delay_finance_drag": 0.015,
      "residual_marketability_risk": 0.035}),

    # ── Distress-sale title risk — receiver / attorney / unverified authority ─
    # (S37-COVENANT-SPLIT, 2026-07-03): genuine distress/verification signals,
    # not standard auction practice. Checked FIRST. Fractions unchanged.
    (["lpa receiver", "lpa receivership", "attorney sale",
      "power of attorney validity unverified", "seller lacks direct ownership",
      "tupe liability", "not original owner"],
     {"lender_certifiability_risk": 0.030, "residual_marketability_risk": 0.025}),

    # ── Standard limited/no title guarantee — routine auction practice ────────
    # The large majority of auction sellers give limited title guarantee; on
    # its own this is not a distress signal. Fraction reused verbatim from
    # Rule 8 (same segments, same "known, low-cost, routinely insurable"
    # profile) rather than inventing a new figure.
    (["no title guarantee", "limited title guarantee", "good leasehold title",
      "seller gives no warranty", "title guarantee", "minimal warranty",
      "not absolute", "structural complexity"],
     {"indemnity_insurance_cost": 0.012, "residual_marketability_risk": 0.008}),

    # ── Registered mortgage / lender's charge outstanding ────────────────────
    # Registered charges that require lender consent or discharge on completion.
    # Market consequence: delay (coordination with selling lender required) and
    # lender certifiability risk for the buyer's lender until charge removed.
    # Fractions: missing_search (delay 0.012), flying_freehold (lender 0.030).
    (["registered charge", "mortgage outstanding", "lender consent",
      "further advances", "lender entitled", "charge outstanding",
      "mortgage charge", "lender's charge", "bank mortgage",
      "separate mortgage charge", "restriction on title", "consent of"],
     {"delay_finance_drag": 0.012, "lender_certifiability_risk": 0.030}),

    # ── Coal mining confirmed / ground instability ────────────────────────────
    # Confirmed coal mining, ground instability, mines and minerals excepted.
    # Distinct from missing coal search (handled by missing_search rule) —
    # this is where a confirmed risk has been identified.
    # Fractions: contamination rule (lender 0.025, residual 0.035) — same
    # risk profile as physical contamination.
    (["coal mining", "underground coal", "coal mine", "mining confirmed",
      "mining ground instability", "mines and minerals", "past mining",
      "ground stability", "mine entry"],
     {"lender_certifiability_risk": 0.025, "residual_marketability_risk": 0.035}),

    # ── Seller will not answer buyer enquiries — STANDARD, solicitor-handled ──
    # (S37-COVENANT-SPLIT, 2026-07-03): reclassified per direct product
    # decision — a routine solicitor due-diligence matter at auction, not a
    # value-differentiating defect. Already captured separately via
    # CAP_SUBJECT... confidence reduction (line ~812) when this clause is
    # present, so the uncertainty signal is not lost, only removed from the
    # price. Fraction reused verbatim from Rule 8/17's admin tier.
    (["will not answer buyer enquiries", "seller will not answer",
      "replies qualified", "enquiries not answered",
      "will not answer enquiries"],
     {"indemnity_insurance_cost": 0.012, "residual_marketability_risk": 0.008}),

    # ── Genuinely compressed completion — below the standard auction window ──
    # (S37-COVENANT-SPLIT, 2026-07-03): 20-28 days is the standard UK auction
    # completion window (Common Auction Conditions default); buyer-insures-
    # from-exchange is the CAC default position, not a distress signal on its
    # own. What's genuinely compressed/unusual: sub-14-day completion,
    # shortened notice-to-complete periods, and blank/uncertain dates.
    # Checked FIRST. Fractions unchanged.
    (["14-day completion", "seven-day completion", "5 business days",
      "five business days", "shortened notice to complete",
      "completion date blank", "extremely short completion"],
     {"delay_finance_drag": 0.015, "residual_marketability_risk": 0.020}),

    # ── Standard completion terms / notice-to-complete fee ────────────────────
    # Standard 20-28 day completion, buyer-insures-from-exchange (CAC
    # default), and routine notice-to-complete fee clauses are present on the
    # large majority of auction lots. Fraction reused verbatim from Rule 17
    # (missing standard docs) — same admin/timing profile, not invented.
    (["buyer insures from exchange", "buyer bears risk from exchange",
      "seller insurance excluded", "28-day completion", "non-refundable deposit",
      "notice to complete", "short completion"],
     {"direct_cure_cost": 0.008, "delay_finance_drag": 0.012}),

    # ── Building Safety Act / fire safety certification ───────────────────────
    # Missing EWS1 / BSA leaseholder certificates render affected flats
    # unmortgageable with most mainstream lenders. Equivalent severity to a
    # defective lease in lender certifiability terms.
    # Fractions: defective_lease rule (lender 0.050, residual 0.035).
    (["building safety act", "leaseholder certificate", "ews1", "ews 1",
      "cladding", "fire safety certificate", "building safety",
      "remediation contribution"],
     {"lender_certifiability_risk": 0.050, "residual_marketability_risk": 0.035}),

    # ── EPC E / MEES compliance risk ─────────────────────────────────────────
    # EPC E / MEES non-compliance creates a letting risk (properties must
    # be EPC E or better to let legally from 2025). Extends the existing
    # EPC G/F rule to catch E-rated and MEES-flagged properties, plus
    # structural defects (solid brick, electric storage heating) that imply
    # near-term mandatory improvement works.
    # Fractions: existing EPC G/F rule (delay 0.015, residual 0.020).
    (["mees compliance", "mees", "epc rating e", "minimum energy efficiency",
      "electric storage heating", "solid brick walls", "very poor rating",
      "no epc in", "epc not present", "epc — not present",
      "near expiry", "epc near expiry"],
     {"delay_finance_drag": 0.015, "residual_marketability_risk": 0.020}),

    # ── Missing / incomplete leasehold management documents ───────────────────
    # LPE1, TA7, stale title registers, unreadable lease documents, rent
    # apportionment issues, conveyance plan gaps, and similar leasehold
    # administrative deficiencies short of a full lease defect.
    # Fractions: management_pack rule (direct_cure 0.006, delay 0.010,
    # lender 0.008) — same administrative-complexity profile.
    (["lpe1", "ta7", "leaseholder information",
      "leasehold management information", "sums payable",
      "leasehold form incomplete", "lease documents unreadable",
      "lease copy incomplete", "conveyance plan", "stale title register",
      "very stale", "leasehold information form",
      "full lease document", "missing copy", "rent apportionment",
      "tupe"],
     {"direct_cure_cost": 0.006, "delay_finance_drag": 0.010,
      "lender_certifiability_risk": 0.008}),

    # ── Absent landlord / corporate freeholder ────────────────────────────────
    # Freehold held by an absent, corporate, or uncontactable landlord creates
    # difficulty obtaining consent for works, subletting, or resale; some
    # lenders require evidence of a responsive freeholder.
    # Fractions: management_pack (delay 0.010, lender 0.008) +
    # rights_of_way (residual 0.025) — reflects consent/delay risk and
    # moderate ongoing resale friction.
    (["absent landlord", "freeholder is corporate", "corporate freeholder",
      "freeholder held by", "freehold held by third", "absentee freeholder",
      "corporate entity — absent"],
     {"delay_finance_drag": 0.010, "lender_certifiability_risk": 0.020,
      "residual_marketability_risk": 0.025}),
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

def _flag_routing_source(flag: dict) -> str:
    """Report whether _flag_to_segments would match a _SEGMENT_RULES entry or
    fall through to the generic severity-keyed fallback, WITHOUT duplicating
    the pricing logic — re-runs the same match loop _flag_to_segments uses,
    returns only which branch fired.

    Added 2026-06-25 to make the fallback rate measurable. Found by direct
    trace against a real deal's 25 legal-pack flags: 12 of 17 substantive
    flags (71%) fell through to the 5-value severity table rather than a
    defect-specific rule — e.g. "EPC Rating G — Property Cannot Be Let" and
    "No Heating System" both reduce to the same generic number as "Missing
    Chancel Repair Search", despite being different categories of risk.
    This function turns that into something every deal reports on its own,
    instead of something that has to be re-discovered by hand-testing flag
    text, the way it was found this session.

    Returns: "note" | "matched_rule" | "fallback"
    """
    sev = (flag.get("severity") or "note").lower().strip()
    if sev in _SEVERITY_NOTE_ONLY:
        return "note"
    text = " ".join(filter(None, [
        flag.get("title", ""),
        flag.get("summation", ""),
        flag.get("implication", ""),
        flag.get("category", ""),
    ])).lower()
    for keywords, _fractions in _SEGMENT_RULES:
        if any(kw in text for kw in keywords):
            return "matched_rule"
    return "fallback"


def filter_active_flags(all_flags: list[dict], resolved_map: dict) -> list[dict]:
    """
    Return only the flags NOT marked resolved in resolved_map.

    S38-RESOLVED-FLAG-FILTER (2026-07-03): calculate_workbench_ceiling's own
    docstring says it expects "each active (unresolved) legal-pack value
    risk" — but nothing in the codebase ever actually filtered by resolved
    status before calling it. /api/deals/<id>/flags-resolved persists
    resolved state to summary_json._resolved_flags as an index -> bool map
    (see app.py save_flags_resolved), but that map was never read anywhere
    except by its own GET endpoint. Verified by grep: zero other references
    before this fix. Extracted into its own function specifically so it has
    a direct unit test — the earlier gap was untestable inline logic in a
    12,000-line route function, not a wrong implementation once written.

    resolved_map keys are strings (JSON object keys), matching the format
    the frontend already sends: {"0": true, "3": true}. A flag is treated
    as resolved only if its index key maps to a truthy value — missing keys,
    false, and non-boolean-truthy values all mean "still active", which is
    the safe default (never silently drops a flag's risk without an
    explicit resolved=true).
    """
    resolved_map = resolved_map or {}
    return [
        f for i, f in enumerate(all_flags or [])
        if not resolved_map.get(str(i))
    ]


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
# S35-TYPE-CONF (2026-06-30): confidence cap when the subject's property type
# was resolved with low confidence — specifically the 50/50 EPC neighbour
# tiebreak case (1 Lid Lane=Semi vs 5 Lid Lane=End-Terrace, no clear majority,
# tie-broke by nearest-house distance). Confirmed live against 10 Lid Lane
# (DE6 2EG) on 2026-06-25 — flagged as blocking go-live in two separate audit
# sessions (2026-06-25 and 2026-06-28) without being actioned. The valuation
# TYPE FILTER (Step 1) runs on this resolved type: if we valued a semi off
# terrace comps (or vice versa) because of a wrong 50/50 tiebreak, the ceiling
# number is structurally uncertain. The comp selection was correct IF the type
# was right, but we cannot be confident it was. A confidence cap is the
# correct signal — not an adjustment to the number, a flag on its reliability.
# Also fires when the LLM listing-text crosscheck overrides the EPC tiebreak
# (llm_crosscheck_override) — the override improves accuracy but the
# underlying evidence was split, so low confidence is still appropriate.
# 0.55 = same as CAP_UNQUANTIFIED_RISKS (same principle: real evidence exists,
# degraded quality). Strictly less severe than CAP_TENURE_UNRESOLVED (0.45)
# because we HAVE a type, just low confidence in it.
CAP_SUBJECT_TYPE_LOW_CONFIDENCE        = 0.55
# S35-AREA-CONF (2026-06-30): same principle for subject floor area resolved
# with low confidence — specifically when _compute_gia_from_text built a GIA
# estimate from only 1-2 room dimensions from a partial room schedule
# (returns _gia_conf="low"). Rare in practice: 80% of the deal book has EPC
# cert text (returns "high"), 20% has no area data at all (returns None, no
# cap fires). Only fires when the engine has a number but the number is shaky.
CAP_SUBJECT_FLOOR_AREA_LOW_CONFIDENCE  = 0.55
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
    "refuses to answer",
    # ^ S33-STEP4a-FIX (2026-06-22): confirmed live on Hey Street (NG10 3HA)
    # that the LLM extraction produced "Seller Refuses to Answer Buyer
    # Enquiries" — a real condition-risk signal that matched NONE of the
    # markers above ("no enquiries" / "will not answer" / "seller will not
    # respond" are none of them substrings of "refuses to answer"). This is
    # why CAP_CONDITION_RISK_SIGNALS never fired despite the flag existing
    # and being correctly extracted: title-substring matching is brittle
    # against LLM phrasing variance. Title-substring matching itself is NOT
    # being redesigned here — that's a separate, larger piece of work
    # (matching on flag category/risk_id semantics instead of title text).
    # This is a surgical addition of one confirmed real phrasing gap, not a
    # rewrite of the matching mechanism.
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

# S33-TIME-COST-LOOKUP (2026-06-22): deterministic time/cost lookup, NOT LLM
# estimation. Per explicit product instruction: "no estimates, we have the
# ranges used by other models, no guessing, trust [built] by the user." The
# LLM extraction step identifies flags and severity; it is never asked to
# invent a cost or time figure, because that would recreate the exact
# unfounded-number problem already found and removed from legal_risk_weight
# (an ungoverned 1-10 score with no stated methodology anywhere in this
# file's prompts). Every entry below traces to a specific source from one of
# two research passes run this session:
#   - "LegalSmegal Risk-Pricing Engine" report (council search turnaround
#     data, defect cost/time benchmarks across 17 categories)
#   - "Monetising Time-to-Resolve" report (RICS CAC interest rate, bridging
#     finance, development appraisal rates) — NOT used for per-flag figures
#     below, since the user explicitly ruled out auction-window/day-rate
#     normalisation; kept only as background context.
# Matching is by title substring, same brittleness-aware approach as
# _CONDITION_RISK_TITLE_MARKERS — a flag type not listed here returns
# UNRESEARCHED, never a guess. This table is expected to grow; it is NOT
# exhaustive across the ~17+ defect categories identified in research.

TIME_COST_UNDISCLOSED = "UNDISCLOSED"   # the legal pack itself withholds the figure
TIME_COST_UNRESEARCHED = "UNRESEARCHED"  # not yet covered by researched data
TIME_COST_NO_RESOLUTION = "NO_RESOLUTION"  # not a clearable defect — a disclosure/
#   risk-awareness flag with no standard "commission a fix and get a number back"
#   path (e.g. NSIP proximity). Distinct from UNRESEARCHED: the absence of a figure
#   is the RESEARCHED answer here, not a gap. Carries a fixed disclosure message in
#   `methodology`; the frontend renders that instead of Time/Cost rows.
TIME_COST_CONFIRMED = "CONFIRMED"  # S33-CONFIRMED (2026-06-27): a POSITIVE or
#   neutral finding — the flag confirms something is fine / clean / already in
#   order (e.g. "Freehold Title Absolute — No Mortgage Registered", "No Disputes
#   Declared", "Vacant Possession Confirmed"). Cost 0 / time 0 — there is
#   genuinely nothing to do. Distinct from NO_RESOLUTION (which is a CAUTION the
#   buyer must accept and which renders amber): CONFIRMED is reassurance and must
#   render neutral/positive, never as a warning. Per the product principle that a
#   positive finding should read as common-sense reassurance, not an unresearched
#   gap or a risk. Carries its (brief) confirming reason in `methodology`.


# S33-TIME-COST-INDICATIVE (2026-06-22): two SRA-bar gaps closed here.
# (1) Every entry now carries an explicit `methodology` string — the
#     citation a user or auditor can be shown, not just a code comment only
#     visible to a developer. This is the "named, explained, indicative not
#     definitive" requirement from the SRA risk-scoring warning notice.
# (2) `outlier_note` is populated ONLY where the source report itself
#     flagged genuine skew/tail risk — currently just local authority
#     search (the report's own words: "same-day to 180 working days").
#     Adding this note to every entry generically would be a second kind of
#     dishonesty — implying uncertainty that isn't in the source data for
#     entries like chancel repair or drainage, which the report describes
#     as tight, low-variance ranges. Most entries correctly have
#     outlier_note=None.
_TIME_COST_LOOKUP = (
    # (markers, time_days, cost_gbp, methodology, outlier_note)
    (("local authority search",), (7, 21), (100, 300),
     "LegalSmegal Risk-Pricing Engine report, 2026: national median ~8wd, "
     "two-thirds of councils within 10wd; £100-300 typical official-search fee.",
     "Source range is a national TYPICAL band, not a guarantee. The same "
     "report found genuine outliers from same-day (e.g. East Suffolk) to "
     "180 working days (e.g. Hackney) depending which council the property "
     "falls under. This figure does not know which council applies here."),
    (("environmental search",), (1, 3), (35, 60),
     "LegalSmegal Risk-Pricing Engine report, 2026: standard search-provider turnaround.",
     None),
    # S39-TIME-COST-GAP-CLOSE (2026-07-03): three entries closing named gaps
    # found via live screenshot audit — flags that were rendering "Unknown —
    # not yet researched" despite the 71-entry table. Sourced via targeted
    # research, not generic web search, per standing instruction. Time is
    # left as TIME_COST_UNRESEARCHED on all three: a real, named cost source
    # was found for each, but no source meeting the same evidentiary bar
    # covered turnaround time — left honestly unresearched rather than
    # estimated, per explicit "no fabrication" standard. Cost and time are
    # never forced to resolve together (see lookup_time_cost docstring).
    (("former landfill", "contaminated land", "landfill site",
      "historic landfill", "waste site"),
     TIME_COST_UNRESEARCHED, (550, 800),
     "Phase 1 contaminated-land desk study, £550-£800 +VAT (Earth "
     "Environmental, published 2025/2026 rates) — the standard follow-up "
     "investigation where an environmental search flags historic landfill "
     "or waste-site proximity. This is a SEPARATE, ADDITIONAL cost to the "
     "environmental search fee itself, not a replacement for it.",
     "If findings from the Phase 1 desk study warrant a Phase 2 intrusive "
     "site investigation, cost and time both increase substantially beyond "
     "this figure — not covered here. Turnaround time for the Phase 1 study "
     "itself was not found from a source meeting the same evidentiary bar "
     "as the cost figure and is left unresearched rather than estimated."),
    (("structural issues disclaimer", "structural risk disclaimer",
      "accepts all structural risk", "buyer accepts all structural"),
     TIME_COST_UNRESEARCHED, (400, 1500),
     "Independent RICS survey typically commissioned by the buyer where a "
     "contract disclaims seller responsibility for structural condition: "
     "RICS Level 2 (HomeBuyer) Report from ~£400, RICS Level 3 (Building "
     "Survey) up to ~£1,500 depending on property age/size/value "
     "(HomeOwners Alliance, published 2025/2026 guidance). Range reflects "
     "survey-tier choice, not one fixed fee.",
     "Older, larger, or non-standard-construction properties typically "
     "warrant the higher Level 3 tier, not the Level 2 minimum. Turnaround "
     "time depends on local surveyor availability and was not sourced to "
     "the same evidentiary bar as the cost figure — left unresearched."),
    (("will not answer buyer enquiries", "seller will not answer",
      "will not answer enquiries", "enquiries not answered",
      "replies qualified"),
     TIME_COST_UNRESEARCHED, (429, 429),
     "Where a seller declines to answer pre-contract enquiries, the "
     "buyer's solicitor must independently verify matters a responsive "
     "seller would normally confirm directly — benchmarked against a "
     "published UK auction legal-pack review fee of ~£429 +VAT (Property "
     "Solvers, published 2025/2026 pricing), used here as an indicative "
     "floor for the additional diligence burden this clause creates.",
     "This is a benchmark for independent review effort, not a quote from "
     "any specific solicitor firm, and not a guarantee that this is what "
     "the buyer's actual solicitor will charge for this specific item. "
     "Turnaround time depends on solicitor workload and was not sourced to "
     "the same evidentiary bar as the cost figure — left unresearched."),
    # S33-RESEARCH-COVERAGE-v2 (2026-06-27): upgraded from a generic
    # internal-citation stub to named-source data. Markers cover coal
    # mining, CON29M, and generic ground-stability — coal-specific phrasings
    # ("coal mining", "coal authority", "con29m", "mining report") resolve
    # here rather than to a separate duplicate entry.
    (("mining", "coal mining", "coal authority", "con29m", "mining report",
      "ground stability"), (0, 1), (30, 150),
     "Coal mining search (CON29M) £30-66 inc VAT, returned ~2 working hours; "
     "interpretive report £147.83 +VAT (Mining Remediation Authority "
     "[formerly Coal Authority]; SAM Conveyancing; Severn Trent Searches; "
     "Local Conveyancing Direct). HIGH confidence on search cost.",
     "Critically: the Mining Remediation Authority FUNDS remedial work for "
     "properties damaged by past coal mining — the homeowner does not pay "
     "for the structural fix, and insurance covering emergency repairs is "
     "reimbursed. The search cost prices DETECTION; detection here does not "
     "imply a large remediation bill falls on the buyer. Non-coal ground "
     "stability (clay/subsidence) is a separate report (£100s) with "
     "property-specific remediation that is not in this range."),
    (("chancel repair",), (0, 1), (20, 30),
     "LegalSmegal Risk-Pricing Engine report, 2026: indemnity route, typically issued within 24h, often skipping the search entirely.",
     None),
    # ── Specific covenant / adoption rows — MUST precede the generic
    # "restrictive covenant" row below so specific markers win first
    # (same specific-before-generic discipline as the EPC reorder). ──
    (("road adoption", "hmpe"), (7, 10), (30, 90),
     "Standalone CON29 highways / road-adoption search from the county "
     "highway authority. £30-90 +VAT typical (Kent CC £30 +VAT; Hampshire "
     "CC £85 +VAT standard; general band per Local Conveyancing Direct). "
     "Turnaround ~10 working days (Kent CC, Via East Midlands targets). "
     "HIGH confidence.",
     "This figure confirms adoption STATUS only. Where a historic covenant "
     "appears to conflict with HMPE status, interpreting whether the covenant "
     "survives adoption is a solicitor-review item (case-by-case), not covered "
     "by the search fee."),
    # S33-RESEARCH-COVERAGE (2026-06-27): markers widened from "common drains"
    # only — confirmed via real flag-title audit (100 flags, 4 deals) that the
    # LLM extraction produces phrasings like "Boundary Wall Maintenance
    # Covenant — Personal Obligation" that describe the SAME underlying
    # problem (a positive/personal covenant to maintain a shared physical
    # feature) but matched none of the prior markers — same brittleness
    # pattern as the documented "refuses to answer" gap. Not new research:
    # personal/positive maintenance covenants (drains, boundary walls, shared
    # fences, access ways) are legally identical in the relevant respect —
    # the burden does not run with freehold land at common law (Law
    # Commission research briefing CBP-8560, 5 May 2026; LexisNexis
    # "Positive covenant" glossary) — so indemnity insurance does not fit
    # any of them, and solicitor-review of enforceability is the correct
    # route for all.
    (("common drains", "personal obligation", "positive covenant",
      "boundary wall maintenance", "maintenance covenant"), (2, 5), (500, 1500),
     "Solicitor review of scope/enforceability of a positive maintenance "
     "covenant (drains, boundary wall, shared fence, or similar physical "
     "feature). £500-1,500 +VAT for a discrete written enforceability "
     "opinion (Go Legal; Lawhive). Routine flagging is usually absorbed in "
     "the standard conveyancing fee. MEDIUM confidence.",
     "Indemnity insurance does NOT fit here: a positive covenant is a "
     "personal/ongoing obligation to actively do something (pay, maintain, "
     "repair), not a one-off breach risk a policy can underwrite — the "
     "burden does not run with freehold land at common law (Law Commission "
     "research briefing CBP-8560, 5 May 2026), so a policy would not cover "
     "the actual works (HomeOwners Alliance; Timms Solicitors). Treated as "
     "solicitor-review only."),
    (("tree and shrub", "tree/shrub"), (1, 2), (200, 2000),
     "Restrictive covenant indemnity insurance route. £200-2,000 typical "
     "premium (SAM Conveyancing; from £60 incl IPT floor per GCS), issued "
     "24-48h, often within minutes (HomeOwners Alliance; Prosperity Insurance). "
     "Solicitor-review alternative is £500-1,500 +VAT. MEDIUM confidence.",
     "Applicability caveat: insurers usually decline cover where the benefiting "
     "party is known/identifiable (Fladgate). A boundary-tree covenant tied to "
     "a specific named neighbour's amenity is exactly that scenario — the "
     "indemnity route may be unavailable, pushing this to the solicitor-review "
     "cost instead."),
    (("road making", "road-making"), (0, 1), (57, 180),
     "Road Adoption legal indemnity policy (a distinct named product, not "
     "generic covenant cover). From £57 incl IPT capped / £84 in perpetuity "
     "(GCS published rates); real-world example £180 for a flat with no S38 "
     "agreement (MoneySavingExpert). Issued minutes-to-24h. HIGH confidence "
     "on premium.",
     "Covers the buyer against a future highway-authority demand to contribute "
     "to making-up / adoption costs (Highways Act 1980 private street works). "
     "Distinct from recurring annual 'private road insurance' (~£175/yr public "
     "liability) — do not conflate. Solicitor-review alternative £500-1,500 "
     "+VAT."),
    (("restrictive covenant", "building line"), (0.5, 5), (200, 2000),
     "LegalSmegal Risk-Pricing Engine report, 2026: indemnity insurance route. "
     "Deed-of-release route is weeks-to-months with cost not researched — "
     "this figure covers the indemnity path only.",
     None),
    (("indemnity insurance",), (0, 1), (20, 500),
     "LegalSmegal Risk-Pricing Engine report, 2026: general indemnity premium range, same-day to 24h issue.",
     None),
    (("no epc",), (1, 7), (65, 125),
     "Commissioning a fresh EPC assessment, 2026: studio/1-bed £65-80, "
     "2-3 bed £75-95, 4-bed+ £90-125+ (commercial £150-500+ excluded — "
     "auction residential stock only); typical booking-to-issue 1-7 working "
     "days. Range shown is the full studio-to-4-bed+ span, not adjusted to "
     "this property's actual size.",
     "This figure is the cost of producing a missing EPC, not the cost of "
     "upgrading an existing low rating — see the separate EPC-rating row "
     "for D-to-C upgrade costs. The two are different problems."),
    (("epc rating", "no insulation", "solid brick"), (7, 30), (1500, 5000),
     "LegalSmegal Risk-Pricing Engine report, 2026: D-to-C uplift cost range. "
     "Works-dependent; MEES 2030 secondary legislation not yet passed — "
     "treat as a forward liability, not settled law.",
     None),
    (("drainage", "water search", "con29dw"), (5, 10), (40, 100),
     "LegalSmegal Risk-Pricing Engine report, 2026: CON29DW standard turnaround.",
     None),
    # S33-RESEARCH-COVERAGE (2026-06-27): three new categories researched to
    # the same standard as the entries above, sourced from practitioner
    # material (conveyancing solicitors, indemnity brokers, mortgage-panel
    # guidance) per explicit product instruction to check how solicitors/
    # brokers/accountants actually handle each problem before pricing it —
    # not generic web search. All three traced to the real flag-title audit
    # (100 flags, 4 deals, 2026-06-27): "Freeholder Identity — Ashcorn
    # Estates Limited, Absent Landlord Risk"; no live HMO-licence title in
    # this batch but named explicitly in the 2026-06-26 checkpoint; "Ground
    # Rent Escalation — Tripling Over Lease Term".
    (("absent landlord", "absentee landlord", "absent freeholder",
      "untraceable freeholder", "missing freeholder", "missing landlord"),
     (0, 1), (100, 300),
     "Absent/untraceable-freeholder indemnity insurance — a distinct named "
     "product, not generic restrictive-covenant cover. £100-300 typical "
     "premium (Homeward Legal published indemnity-cost guide, 2026), "
     "issued same-day to 24h where a lender accepts the policy. MEDIUM "
     "confidence — premium band is well-attested; lender acceptance is "
     "not universal (see outlier note).",
     "Indemnity is a workaround, not a fix, and several lenders decline it "
     "outright for an absent freeholder (LenderMonitor UK Finance Lenders' "
     "Handbook guidance, 2026). Where indemnity is declined or the buyer "
     "needs to deal with the freeholder directly (consent, lease extension, "
     "freehold purchase), the only route is a court-granted Vesting Order — "
     "a three-stage County Court / First-tier Tribunal process reported at "
     "2-3.5 years (Zero Down Lease, 2026), not weeks. This figure covers "
     "the indemnity route ONLY; it does not apply if indemnity is refused."),
    (("hmo licen",), (42, 84), (500, 2000),
     "HMO licence application/renewal: council fee £500-1,500 typical for a "
     "5-year term per most 2026 industry guides, though some sources report "
     "£800-2,000+ and London boroughs commonly charge above this band "
     "(Landlord Resource; AgentHMO; Foot Forward Property Investments, "
     "2026 guides). Processing time 6-12 weeks is the stated council target "
     "in most published service standards (Barnet, Cherwell, Lambeth, "
     "Westminster council guidance, 2026). MEDIUM confidence — fee and "
     "target-turnaround bands are well-attested across multiple councils, "
     "but neither is a single national figure.",
     "Both figures vary materially by local authority and are TARGETS, not "
     "guarantees. Genuine outliers exist on the time side: one practitioner "
     "account (Slater & Brandley, Nottingham) reports a real case taking "
     "two years to reach a final decision, and operating an HMO while an "
     "application is pending is generally lawful under the Housing Act "
     "2004 (so a missing/expired licence does not necessarily mean the "
     "property cannot be let during the wait — but it must already have a "
     "valid application in). Civil penalty exposure for an unlicensed HMO "
     "is up to £40,000 per breach since 1 May 2026 (Renters' Rights Act "
     "2025) — this is a liability-exposure figure, not a resolution cost, "
     "and is not included in the range above."),
    (("ground rent escalation", "ground rent doubling", "doubling ground rent",
      "escalating ground rent", "ground rent tripling"), (0, 1), (500, 7000),
     "USER VERIFICATION NEEDED — wide, low-confidence range spanning two "
     "genuinely different resolution routes, not a single product: (1) "
     "indemnity insurance, narrow scope, where accepted; (2) Deed of "
     "Variation, solicitor-negotiated with the freeholder, who is under no "
     "obligation to agree — legal fees alone typically quoted £500-1,500 "
     "+VAT (Lawhive, Stephensons), but real transactions reported £5,000 "
     "(MoneySavingExpert) and £7,000 (MoneySavingExpert) once freeholder "
     "compensation and both sides' costs are included. LOW confidence — "
     "this range is too wide to size a specific deal without the actual "
     "lease terms and a freeholder response, which a title-text match "
     "cannot determine.",
     "A material legal change limits how often this defect now needs "
     "resolving at all: from December 2025, the change removing leases "
     "from the 'AST trap' (assured shorthold tenancy risk under the "
     "Housing Act 1988 for ground rents above £250/£1,000 London) means "
     "the core mortgage-refusal driver behind a Deed of Variation or "
     "indemnity policy may no longer apply to many cases (Peppercorn Law, "
     "2026) — pre-Dec-2025 advice on this point may now be outdated. "
     "Separately, a draft Commonhold and Leasehold Reform Bill (Jan 2026) "
     "proposes capping existing ground rents at £250/year, but this is not "
     "in force and is not expected before 2028 (Starck Uberoi Solicitors). "
     "Do not treat this figure as reliable without confirming current "
     "lender requirements and lease-specific terms."),
    # ════════════════════════════════════════════════════════════════════
    # S33-RESEARCH-COVERAGE-FULL (2026-06-27): full verified-source set from
    # the cross-category research pass. Every figure traces to the NAMED
    # source in its methodology string; confidence (HIGH/MEDIUM) is stated
    # verbatim per source quality — no inflation. NO_RESOLUTION is used only
    # where the research found no fixable cost (notice-to-complete = an
    # exposure, not a clearable fee). Four entries carry prominent 2025-26
    # legal-change flags (planning enforcement 10-yr limit; lease extension /
    # marriage value; — ground rent already flagged above). Markers checked
    # for substring collisions against existing entries before insertion;
    # coal-mining merged UPWARD into the mining entry to avoid shadowing.
    # ════════════════════════════════════════════════════════════════════
    # ════════════════════════════════════════════════════════════════════
    # S34-COVERAGE-GAP-2026-07-03: five flags found live-unresearched on
    # 57 York Road (SY1 3QU) — two were near-misses on already-built
    # categories (fixed above by widening markers on the incomplete-
    # document entry), three are genuinely new categories added here at
    # the same named-source standard as the rest of the table. Where no
    # published figure exists for the exact reservation type, the entry
    # says so explicitly and maps to the nearest verified task-type
    # benchmark instead of inventing one.
    # ════════════════════════════════════════════════════════════════════
    (("alter access", "alter the route", "alter roads", "vary the route of",
      "reroute access", "reroute the access"), (2, 5), (500, 1500),
     "No published figure exists for pricing a transferor's reserved right "
     "to alter access routes specifically — this is a title condition, not "
     "a marketed product. The underlying professional task is the same "
     "class as the shared-access cost-sharing entry above: a solicitor's "
     "discrete opinion on the scope and enforceability of the reservation, "
     "£500-1,500 +VAT (Go Legal; Lawhive), 2-5 working days. MEDIUM "
     "confidence — mapped from a verified task-type benchmark, not a "
     "category-specific published source.",
     "This is a right reserved IN FAVOUR of the transferor/estate, not a "
     "right the buyer can remove. The £500-1,500 figure buys an opinion on "
     "how the clause could be exercised, not a way to extinguish it — "
     "removal would need the transferor's agreement (a deed variation, "
     "priced at whatever they will accept) and has no fixed figure."),
    (("multi-property title", "multi property title", "shared obligations",
      "estate-wide covenants", "large estate title", "large multi-property"),
     (3, 7), (500, 1500),
     "Same discrete-opinion task class as the shared-access cost-sharing "
     "entry: solicitor review of the scope and enforceability of the "
     "covenants affecting a plot within a large multi-property title, "
     "£500-1,500 +VAT (Go Legal; Lawhive). Time widened to 3-7 working "
     "days versus the single-clause cost-sharing entry, since reading an "
     "estate-wide scheme of covenants across dozens of plots is a larger "
     "review. MEDIUM confidence — task-type benchmark, not a category-"
     "specific published figure.",
     "Does not price any specific shared maintenance contribution — those "
     "depend on the estate's actual scheme of covenants and cannot be "
     "sized without reading it. Treat this as the cost of finding out what "
     "you're bound by, not the cost of the obligations themselves."),
    (("groundwater flooding", "groundwater flood"), (1, 5), (0, 0),
     "At MODERATE (not high) rating, the real remaining action is checking "
     "insurance availability, not buying a further report — the buyer "
     "already holds a Groundsure Homebuyers assessment in the pack. UK "
     "insurance brokers do not charge a fee for a flood-risk quote; they "
     "are paid by commission from the insurer (MoneyHelper/GOV.UK). HIGH "
     "confidence on the £0 broker-quote cost; 1-5 working days for a "
     "quote turnaround.",
     "Flood Re — the government scheme capping premiums for high-flood-"
     "risk homes — excludes buy-to-let and commercial property (Admiral; "
     "Go.Compare), so a BTL purchase cannot fall back on it if a standard "
     "insurer declines or loads the premium. A specialist broker (BIBA "
     "directory) may be needed, which can extend the timeline but remains "
     "fee-free at quote stage. Many standard policies do cover groundwater "
     "flooding at moderate ratings; confirm with the specific insurer "
     "rather than assuming exclusion (MoneySuperMarket)."),
    (("surface water flooding", "surface water flood"), (1, 5), (0, 0),
     "Same basis as groundwater flooding above: at MODERATE building-level "
     "risk (not the overall site's low rating), the action is an insurance "
     "quote check, not a further report purchase — the Groundsure "
     "Homebuyers assessment already covers this. Broker quotes are free, "
     "paid by insurer commission (MoneyHelper/GOV.UK). HIGH confidence on "
     "£0 cost; 1-5 working days for a quote.",
     "Flood Re excludes buy-to-let property (Admiral; Go.Compare) — same "
     "caveat as groundwater flooding. A moderate surface-water rating at "
     "building level (as opposed to the site-wide rating) most often "
     "reflects drainage/run-off risk rather than a fluvial flood zone, and "
     "is generally the more commonly insurable of the flood-risk types."),
    (("flood risk", "high flood", "flood zone", "flooding identified"), (0, 3), (20, 75),
     "Flood risk report £20-75 (Always Conveyancing £20-25 +VAT; Homeward "
     "Legal £50-75), returned ~2h to a few days. HIGH confidence on report "
     "cost. This prices CHARACTERISING the risk, not eliminating it.",
     "Report cost only — physical mitigation is separate and variable. "
     "Property flood resilience grants exist: Flood Re 'Build Back Better' "
     "up to £10,000 after a claim; Homeowner Flood Protection Grant covers "
     "90% up to £10,000 (CIWEM). High flood risk affects insurability and "
     "value; properties built after 1 Jan 2009 are excluded from Flood Re."),
    (("radon",), (10, 90), (53, 2500),
     "Radon test £52.80 (UKradon/UKHSA); active sump remediation ~£800 "
     "typical, range £500-2,500 (GOV.UK radon remediation analysis; UKradon; "
     "Checkatrade). Test ideally runs 3 months (10-day screening minimum); "
     "sump installs in 1-2 days. HIGH confidence.",
     "Range spans test-only (~£53) to full active-sump remediation (~£2,500). "
     "An active sump cuts levels ~6-fold and is the most effective measure; "
     "passive measures far less so. A radon bond/retention ~£2,500 is "
     "commonly used at sale pending results."),
    (("japanese knotweed", "knotweed"), (4, 30), (1200, 8000),
     "PCA-member survey £200-299 +VAT; herbicide treatment plan £1,200-2,500 "
     "+VAT (3-5 year programme); excavation/removal from £8,000 +VAT "
     "(Property Care Association; PBA Solutions; Total Weed Control). HIGH "
     "confidence. Time shown is survey-to-plan-start, not the multi-year "
     "treatment duration.",
     "Mortgage lenders require a PCA/INNSA member's plan WITH an insurance-"
     "backed guarantee (~£75 add-on, or from £2,995 incl for larger plans). "
     "Treatment programmes run 3-5 years to completion. CABI (2023) ranks "
     "knotweed the second most expensive invasive species in the UK."),
    (("building regulations completion", "no building regs", "building regs certificate",
      "regularisation"), (0, 7), (35, 300),
     "Building-regs indemnity insurance £35-300 (£180 for a £500k property "
     "per SAM Conveyancing; HomeOwners Alliance), issued instantly; OR a "
     "council regularisation application ~£450 + any remedial works. HIGH "
     "confidence on indemnity premium.",
     "Indemnity covers ENFORCEMENT only, NOT build quality or safety, and is "
     "voided the moment the council is contacted. Pre-11 Nov 1985 works "
     "cannot be regularised. A CLS case study saw an ~£11,000 regularisation "
     "claim met on a £35 policy — the cheap premium is the point. If "
     "remedial works are needed because the build is non-compliant, those "
     "are extra and can run to £10,000s."),
    (("planning enforcement", "enforcement notice", "planning breach"), (140, 260), (3000, 20000),
     "Enforcement-notice appeal: lodge within 28 days; Planning Inspectorate "
     "target ~24-26 weeks (often longer). Cost £3,000-20,000 — planning "
     "consultant £1,500-5,000, solicitor/barrister £150-350/hr, surveys "
     "£500-2,000 each (GOV.UK Planning Inspectorate; Checkatrade; Evans "
     "Jones). MEDIUM confidence — case-dependent.",
     "MAJOR RULE CHANGE: from 25 April 2024 a single 10-year enforcement "
     "limit applies to all breaches in England (was 4 years for operational "
     "development / change of use to a dwelling); works substantially "
     "completed before 25 April 2024 keep the old 4-year rule (SI 2024/452 "
     "reg.3(b); Levelling-up and Regeneration Act 2023 s.115). Whether the "
     "breach is now immune depends on this date."),
    (("lawful development certificate", "retrospective planning", "unpermitted works",
      "no planning permission"), (0, 56), (528, 700),
     "Lawful Development Certificate or retrospective householder application "
     "~£548 (2026 GOV.UK/Planning Portal fee) + Planning Portal £91.02 inc "
     "VAT, ~8-week determination; OR planning indemnity £20-300 instant where "
     "works are old enough (Planning Geek; Today's Conveyancer; Dorset "
     "Council). HIGH confidence on fees.",
     "If retrospective permission is REFUSED, enforcement risk crystallises — "
     "indemnity is the fallback only where the works predate the enforcement "
     "window. The 2026 fee (~£548) is up sharply from ~£258 historically "
     "after CPI uplifts — verify against the 1 April 2026 GOV.UK fee table."),
    (("eicr", "electrical certificate", "electrical installation condition",
      "no electrical"), (1, 7), (80, 350),
     "EICR £80-350 (most homes £100-200; London £120-400+), booked within "
     "days, valid 5 years for lettings (MyJobQuote; HomeSafety UK; Empire "
     "Chase). HIGH confidence. Minor remedial work £50-200; full rewire runs "
     "into the thousands and is separate.",
     "Required under the Electrical Safety Standards (Private Rented Sector) "
     "Regulations 2020 for lettings. The figure prices the inspection plus "
     "minor remedials, not a rewire."),
    (("gas safety", "gas certificate", "cp12", "no gas safety"), (0, 3), (40, 120),
     "Gas Safety Certificate (CP12) £40-120, same-/next-day booking, annual "
     "renewal (Gas Safe Register; Landlord-Certificates from £40; "
     "LetCompliance £60-120). HIGH confidence. Remedial work £50-500 extra.",
     "Annual renewal is mandatory for lettings — unlike most flags this is a "
     "recurring obligation, not a one-off clear. Remedial work on failed "
     "appliances is separate."),
    (("possessory title", "possessory class"), (0, 90), (40, 500),
     "Upgrade to absolute title via Land Registry form UT1 after 12 years' "
     "registered possession (s.62 Land Registration Act 2002): LR fee "
     "(Schedule 3) + statutory-declaration solicitor fee, typically a few "
     "hundred pounds; LR processing days-to-weeks once eligible (GOV.UK "
     "Practice Guide 42; Timms Solicitors). Interim handling = title "
     "indemnity. MEDIUM confidence — confirm LR fee against current Fee "
     "Order.",
     "Until 12 years of registered possession elapse, the upgrade is NOT "
     "available and the only handling is title indemnity insurance. Time "
     "shown is the post-eligibility processing, not the 12-year wait."),
    (("flying freehold",), (0, 1), (50, 300),
     "Flying-freehold indemnity insurance £50-100 (Free Conveyancing Advice) "
     "up to a few hundred pounds (Martin & Co; SAM Conveyancing; Countrywide "
     "Legal Indemnities; MoneySavingExpert ~£100), issued instantly. HIGH "
     "confidence on premium.",
     "Indemnity covers loss of value / legal costs, NOT the underlying lack "
     "of access or support rights. Most lenders cap the flying element at "
     "15-25% of floor area; above that, mortgageability — not premium — is "
     "the real constraint, which a deed of covenant (needing neighbour "
     "cooperation) addresses but indemnity does not."),
    (("short lease", "lease extension", "lease expires", "years remaining",
      "under 80 years", "under 85 years"), (90, 365), (2000, 15000),
     "Statutory lease extension (s.42 route): premium (largest, property-"
     "specific) + solicitor £1,000-2,500 + valuation £450-1,500 + freeholder's "
     "reasonable costs + LR registration + FTT fees if disputed (£110+£220) "
     "(LEASE/lease-advice.org; SAM Conveyancing; Olden Property). 3-12 "
     "months. MEDIUM confidence — the premium itself is property-specific "
     "and not in this range.",
     "MAJOR PENDING LAW CHANGE: the Leasehold and Freehold Reform Act 2024 "
     "will abolish marriage value (the 50% uplift added below 80 years) and "
     "extend terms to 990 years, but is NOT yet in force. In ARC Time v "
     "SSHCLG [2025] EWHC 2751 the High Court upheld the reforms (24 Oct "
     "2025); the Court of Appeal granted freeholders permission to appeal. "
     "Government expects to consult on valuation rates in 2026, so "
     "implementation is late-2026 at the earliest (Homehold). This changes "
     "whether to extend now or wait — flag, do not advise blind."),
    (("service charge", "management accounts", "management information pack",
      "lpe1", "section 20", "major works"), (10, 30), (200, 500),
     "Obtaining the management/LPE1 pack: managing-agent fee £200-500+, "
     "10-30 working days (Law Society LPE1 process; LEASE). MEDIUM "
     "confidence on the pack fee. This prices OBTAINING the information.",
     "This figure is the cost of getting the documents, NOT the service "
     "charge or major-works liability itself — those are forward financial "
     "obligations whose quantum is property-specific (a Section 20 major-"
     "works bill can be £1,000s-£10,000s) and is a risk-acceptance / "
     "valuation item, not a fixable cost. Reasonableness can be challenged "
     "at the FTT but that is a dispute route, not a standard cost."),
    (("notice to complete", "completion default", "completion penalty",
      "default interest", "interest penalty"), (0, 10), TIME_COST_NO_RESOLUTION,
     "Notice to complete = 10 working days under Standard Conditions of Sale "
     "5th ed. condition 6.8.2 (time of the essence). Default interest = the "
     "'contract rate' = Law Society interest rate, currently 7.75% effective "
     "18 Dec 2025 (The Law Society). Deposit at risk = 10% (SCS 2.2.1). HIGH "
     "confidence on the SCS terms.",
     "This is an EXPOSURE figure, not a resolution cost — there is no fee to "
     "'clear' it; the risk is forfeiting the 10% deposit plus default "
     "interest plus resale damages if you cannot complete on time. Auction "
     "Special Conditions / RICS Common Auction Conditions FREQUENTLY vary "
     "the period, rate and deposit — parse the actual pack, do not assume "
     "the SCS default."),
    # S33-RESEARCH-COVERAGE-FINAL (2026-06-27): the genuinely-new, non-overlapping
    # remainder after reconciling the full research pass against the table's
    # existing 30 entries. 8 entries: 2 with named-source time/cost (forfeiture,
    # tenant-in-occupation) and 6 correctly NO_RESOLUTION (auction premium,
    # cost-shifting, nearby development, overage, VAT, missing TA-forms) — these
    # are acquisition-cost/disclosure items with no clearable fix, marked as
    # such rather than given a fabricated figure. Categories already covered
    # earlier this session (flood, radon, knotweed, building-regs, planning
    # enforcement, LDC, EICR, gas, possessory, flying freehold, short lease,
    # service charge incl. section 20, notice-to-complete) were NOT duplicated.
    (('forfeiture', 'breach of lease'), (1, 30), (500, 2000),
     'Forfeiture/breach handling: remedy the breach (cost = the specific remedy) and/or apply for relief from forfeiture (County Court). Solicitor cost for relief typically low thousands. LOW confidence — quantum is breach-specific.',
     'MAJOR PENDING LEGAL CHANGE — the draft Commonhold and Leasehold Reform Bill (27 Jan 2026) proposes to ABOLISH forfeiture, replacing it with a proportionate enforcement scheme (GOV.UK; CMS; Hogan Lovells). Not yet in force, but the current forfeiture threat may diminish.'),
    (('sitting tenant', 'tenant in occupation', 'occupier remains', 'occupier in possession', 'tenant remains in', 'regulated tenancy', 'section 8 ground', 'section 21 notice', 'ast expiry', 'assured tenancy'), (90, 182), (404, 2000),
     "Possession with a tenant in occupation: from 1 May 2026 a Section 8 ground is required (Section 21 abolished). Notice-court-bailiff typically 3-6 months; court/warrant fees (£404 historic accelerated; £148 warrant, GOV.UK) + solicitor fees + lost rent. Source: GOV.UK Renters' Rights Act 2025 guidance; MHCLG Policy Paper (13 Nov 2025). HIGH on process, MEDIUM on total cost.",
     "MAJOR CHANGE — the Renters' Rights Bill received Royal Assent 27 October 2025; tenancy reforms commenced 1 May 2026, abolishing Section 21, last date to issue on a pre-commencement s.21 notice 31 July 2026. Section 8 sale/move-in grounds need 4 months' notice and cannot be used in the first 12 months. Almost every eviction now needs a hearing — court-backlog risk can push the timeline well beyond 6 months. A regulated (pre-1989) tenancy is a permanent value discount, not a clearable cost."),
    (("buyer's premium", 'buyers premium', 'buyer premium', "buyer's fee", 'reservation fee', 'non-refundable deposit', "buyer's administration"), TIME_COST_NO_RESOLUTION, TIME_COST_NO_RESOLUTION,
     "ACQUISITION COST, not a clearable defect — nothing is being 'fixed', so no resolution cost exists. A buyer's premium/reservation fee is added to the price: Savills £250 (below £30k) / £2,100 (£30k+); Allsop reservation ~£1,750; Modern Method of Auction 2.5-5% (typically £6,000+ inc VAT) (Savills Property Auctions; Allsop; Property Rescue). Handling = factor it into the bid ceiling.",
     None),
    (("seller's costs", "seller's legal costs", 'pays seller', 'search fees', 'reimburse', "vendor's costs", 'buyer pays seller'), TIME_COST_NO_RESOLUTION, TIME_COST_NO_RESOLUTION,
     "ACQUISITION COST, not a defect — where Special Conditions require the buyer to pay the seller's legal/search costs (often the £250-450 search pack plus a legal-fee contribution), that sum is added to the price, not a cost to clear (RICS Common Auction Conditions; HouseCheckup search-pack pricing). Handling = price it into the bid.",
     None),
    (('nearby planning', 'planning applications in', 'development risk', 'planning applications near', 'applications in search area', 'planning applications identified', 'applications near property', 'planning applications near property'), TIME_COST_NO_RESOLUTION, TIME_COST_NO_RESOLUTION,
     'Pure disclosure / risk-acceptance — NO resolution cost. Nearby planning applications are a contextual signal, not a defect on this title; handling = review the local authority planning register (free) and price the development risk into the valuation. Nothing to buy, insure, or remediate.',
     None),
    (('overage', 'clawback'), TIME_COST_NO_RESOLUTION, TIME_COST_NO_RESOLUTION,
     'No standard resolution cost — an overage/clawback clause is a contractual burden releasable only by negotiated buy-out with the beneficiary, who sets the price (general conveyancing practice). Handling = negotiate a release (price = whatever the beneficiary accepts) or accept and price the burden in. No researchable fixed figure exists.',
     None),
    (('vat uncertain', 'vat on the price', 'vat liability', 'plus vat on purchase', 'vat status'), TIME_COST_NO_RESOLUTION, TIME_COST_NO_RESOLUTION,
     "Acquisition-cost / disclosure item — no clearable defect. If VAT applies it is 20% on the price (a major acquisition cost), not a fee to 'resolve'. Handling = obtain a VAT opinion from a solicitor/accountant (low hundreds to low thousands) to confirm whether it applies; the VAT itself, if due, is part of the price (general conveyancing/tax practice). Opinion cost is case-specific and not given a fixed range.",
     None),
    (('ta6', 'property information form', 'ta10', 'fittings and contents', 'ta7', 'leasehold information form'), TIME_COST_NO_RESOLUTION, TIME_COST_NO_RESOLUTION,
     "Disclosure / risk-acceptance — generally no resolution cost. Missing TA6/TA10/TA7 forms are requested at no charge; at auction the buyer often proceeds without them (Law Society TA6/TA10/TA7). Where a leasehold management pack (LPE1/TA7) must be sourced from the managing agent, the agent's pack fee (~£200-500+, 10-30 working days) applies — but the missing FORM itself is a 'request it' gap, not a defect to price.",
     None),
    # S33-RESEARCH-COVERAGE-CLUSTER (2026-06-27): the boundary/easement/
    # planning-designation cluster — the last genuinely-researchable group.
    # Real named-source figures where a fix exists (party wall, absence-of-easement
    # indemnity, listed-building consent), and NO_RESOLUTION where the item is a
    # constraint or existing burden with no clearable fix (right-of-way burden,
    # adjacent listed building, conservation area, smoke control, AQMA). Markers
    # tested against the 100-flag audit; two greedy-substring mis-fires ('rooms
    # over passageway' grabbing the cost-sharing entry; 'adjacent' listed routing
    # to the subject-property figure) were caught and fixed before splicing.
    (('party wall',), (14, 60), (100, 2000),
     "Party-wall handling in a CONVEYANCING context (a missing historic award on past works, not new building works): a retrospective party wall agreement to satisfy the buyer's solicitor £1,000-2,000, OR indemnity insurance covering the absence of a party wall award £100-500 (Pine, 2026). Governed by the Party Wall etc. Act 1996 (England & Wales only). HIGH confidence on the conveyancing figure.",
     'Do NOT confuse with the cost of a party wall award for FUTURE building works (loft/extension), which is £900-3,500+ and paid by the building owner doing the works (HomeOwners Alliance; Alstruct; Pine). For an auction buyer the relevant risk is usually a past, undocumented alteration — the retrospective-agreement / indemnity figure above is the correct one. A genuine unresolved dispute with a neighbour is separate and can cost more.'),
    (('absence of easement', 'no right of access', 'missing easement', 'no easement', 'inadequate rights of access', 'landlocked', 'no right of way'), (0, 90), (100, 500),
     "Missing/defective access right: absence-of-easement indemnity insurance is the standard route, premium value-banded (CLS; Severn Trent Searches; Moore Barlow) — broadly aligned with the road-adoption / restrictive-covenant indemnity bands (low hundreds). Issued days. Alternative: a deed of easement (needs the servient owner's agreement) or a prescriptive-easement application to Land Registry — the latter has a ~3-month wait and costs that rise with objections (Michelmores; Sherrards). MEDIUM confidence — premium is value-banded, not a single published figure.",
     "Indemnity covers a CHALLENGE to using an existing route, NOT the ongoing cost of maintaining it — every absence-of-easement policy explicitly excludes the insured failing to pay their share of access-way / shared-path maintenance (Today's Conveyancer; Moore Barlow). A genuinely landlocked property with an uncooperative neighbour may have no affordable fix — see the cost-sharing and right-of-way-burden entries, which are handled differently."),
    (('right of way burden', 'easement burden', 'right of way over', 'third party right of way', 'burden over property'), TIME_COST_NO_RESOLUTION, TIME_COST_NO_RESOLUTION,
     "Existing easement/right-of-way BURDEN (someone else has a right over this property) — generally no resolution cost: it is a permanent feature of the title to be accepted and priced, not a clearable defect (Michelmores; Napthens). An easement runs with the land and binds successors. Removal requires the beneficiary's agreement (a negotiated release at their price) or proof of abandonment — neither has a researchable fixed figure. Handling = factor the burden into value.",
     None),
    (('shared easement', 'cost sharing', 'cost-sharing', 'passageway cost', 'easements over passageway', 'shared access maintenance', 'shared drive'), (2, 5), (500, 1500),
     'Shared-access / passageway COST-SHARING obligation: this is a positive maintenance obligation, handled like the positive-covenant entry — solicitor review of the scope/enforceability of the contribution obligation, £500-1,500 +VAT for a discrete opinion (Go Legal; Lawhive). Routine flagging is usually absorbed in the conveyancing fee. MEDIUM confidence.',
     "Indemnity insurance does NOT fit a cost-sharing obligation: every absence-of-easement policy explicitly excludes the insured failing or refusing to contribute to the cost of maintaining the access way (Today's Conveyancer; Moore Barlow). The actual future maintenance contributions are an ongoing liability whose quantum depends on the works — not a one-off cost to clear, and not insurable."),
    (('listed building on adjacent', 'adjacent listed', 'neighbouring listed', 'listed building nearby', 'consent on adjacent', 'on adjacent property', 'adjacent property — heritage', 'listed building consent on adjacent'), TIME_COST_NO_RESOLUTION, TIME_COST_NO_RESOLUTION,
     'A listed building on a NEIGHBOURING/adjacent property is a planning CONSTRAINT, not a defect on this title — no resolution cost. It can restrict what this property may do near the boundary (works affecting the setting of the adjacent listed building may need consent), but there is nothing to buy, insure, or remediate on this property. Handling = note the constraint for any future works.',
     None),
    (('listed building consent', 'lack of listed building', 'unauthorised works to listed', 'no listed building consent'), (0, 56), (105, 500),
     'Lack of Listed Building Consent on works to THIS property: indemnity insurance from £105 incl. IPT (GCS), issued instantly; OR a retrospective Listed Building Consent application (~8 weeks, no application fee) negotiated with the conservation officer. LOW-MEDIUM confidence — the premium is real but the cover is weak (see note).',
     'CRITICAL CAVEAT — listed-building indemnity is widely regarded as a weak fix: it covers the cost of local-authority ENFORCEMENT action only, NOT criminal prosecution, and unauthorised works to a listed building are a CRIMINAL offence with NO time limit on enforcement (Planning (Listed Buildings and Conservation Areas) Act 1990; Thomson Snell & Passmore; Kew Law). Practitioners commonly advise resolving via the conservation officer with a retrospective application and a vendor retention rather than relying on indemnity. The policy is also voided if the local authority has already been contacted.'),
    (('conservation area',), TIME_COST_NO_RESOLUTION, TIME_COST_NO_RESOLUTION,
     'Conservation-area status is an ongoing planning CONSTRAINT, not a clearable defect — no standard resolution cost. It removes some permitted-development rights (so future works need a full application) but does not require any spend to own the property (Planning (Listed Buildings and Conservation Areas) Act 1990). If there are UNCONSENTED past works in the conservation area, that is the lack-of-consent problem — conservation-area indemnity from low hundreds (CLI; CLS) covers enforcement on those specific works only. Absent unconsented works, this is a disclosure/constraint item.',
     None),
    (('smoke control',), TIME_COST_NO_RESOLUTION, TIME_COST_NO_RESOLUTION,
     'Smoke control area is a usage CONSTRAINT, not a property defect — no resolution cost to own. A non-Defra-exempt appliance can still be used with authorised (smokeless) fuel; the only spend arises if the buyer specifically wants to burn wood, requiring a Defra-exempt appliance (Stove Industry Association; Wandsworth BC, 2026). Penalties apply only to breach (up to £300 for chimney smoke; up to £1,000 for unauthorised fuel — Air Quality (Domestic Solid Fuels Standards) (England) Regulations 2020). Not a cost to clear at purchase.',
     None),
    (('air quality management', 'aqma', 'air quality management area'), TIME_COST_NO_RESOLUTION, TIME_COST_NO_RESOLUTION,
     'An Air Quality Management Area (AQMA) is a local-authority DESIGNATION (declared where national air-quality objectives are not met), not a defect on the property and not a cost to clear. It carries no homeowner remediation requirement; its relevance is to health, lettability and value perception (Environmental Protection Act 1995 Part IV / Local Air Quality Management regime). Handling = disclosure and risk-acceptance, factored into value — there is nothing to buy or remediate.',
     None),
    # S33-RESEARCH-COVERAGE-CLOSEOUT (2026-06-27): final close-out pass.
    # CONFIRMED (positive/neutral findings -> cost/time '-', reads as
    # reassurance, NOT amber caution); document-stated 21-day completion
    # (time=21, cost dash); constraint/acquisition-term/request-it items
    # (NO_RESOLUTION dash, each with its own reason); and the last
    # researchable figures (boundary survey, overhang/eaves indemnity,
    # rooms-over-passageway flying-freehold treatment, EPC-low, tenant-
    # covenant indemnity, buildings insurance). All reviewed and approved
    # against the real audit before splicing. Remaining UNRESEARCHED after
    # this are the probate/seller-silence confidence-cap items only, which
    # correctly carry no time/cost figure.
    (('absolute — no mortgage', 'no mortgage registered', 'absolute class confirmed', 'absolute — no charges', 'no charges registered', 'freehold title absolute', 'leasehold title absolute', 'title absolute'), TIME_COST_CONFIRMED, TIME_COST_CONFIRMED,
     'Confirmed clean: an absolute title class with no adverse entries is the best-quality title and a positive finding — nothing to resolve, no cost, no delay. Noted for completeness, not as a risk.',
     None),
    (('no disputes', 'no complaints declared', 'no disputes or complaints'), TIME_COST_CONFIRMED, TIME_COST_CONFIRMED,
     "Confirmed clean: the seller has declared no disputes or complaints — a positive finding. Nothing to resolve. (As with any seller declaration, its value rests on the seller's honesty, but on its face this is reassurance, not a cost.)",
     None),
    (('vacant possession — occupier letter', 'vacant possession confirmed', 'sold as vacant possession'), TIME_COST_CONFIRMED, TIME_COST_CONFIRMED,
     'Confirmed clean: vacant possession is confirmed — there is no tenant to remove and no possession cost. A positive finding, the opposite of a sitting-tenant problem.',
     None),
    (('epc valid until', 'no immediate renewal'), TIME_COST_CONFIRMED, TIME_COST_CONFIRMED,
     'Confirmed in order: a valid EPC is in place with no immediate renewal needed — nothing to do. (For lettings, check the rating meets the MEES minimum separately; validity alone is confirmed here.)',
     None),
    (('designated primarily residential', 'primarily residential', 'h14/h17', 'h14', 'h17'), TIME_COST_CONFIRMED, TIME_COST_CONFIRMED,
     'Confirmed/neutral: a residential planning designation is the expected, normal position for a home — not a defect or a cost. Noted for completeness.',
     None),
    (('furniture and effects may remain', 'furniture/effects may remain', 'furniture and effects', 'effects may remain', 'furniture/fittings may remain', 'furniture/fittings', 'fittings may remain', 'furniture may remain'), TIME_COST_CONFIRMED, TIME_COST_CONFIRMED,
     'Neutral: the seller noting that furniture/effects may remain is not grounds to delay completion and carries no buyer cost — at most the buyer arranges removal of any unwanted items (a personal/logistics cost, not a legal defect). Not a sitting-tenant or possession issue.',
     None),
    (('21-day completion', '21 day completion', 'non-standard 21-day', '21-day completion period'), (21, 21), TIME_COST_NO_RESOLUTION,
     "Stated in the pack: a 21-day completion period. The TIME is fixed by the contract (21 calendar days from exchange); there is no cost to 'clear' it — it is a deadline to meet, not a defect. The real consequence is FINANCING: a buyer who cannot complete in 21 days needs cash or bridging in place, and missing it risks deposit forfeiture and default interest (see notice-to-complete). Model the finance, don't price a fix.",
     None),
    (('limited title guarantee', 'title guarantee covenants excluded', 'title guarantee excluded'), TIME_COST_NO_RESOLUTION, TIME_COST_NO_RESOLUTION,
     'Risk-acceptance / disclosure — no resolution cost. A limited title guarantee (common on probate, repossession and some auction sales) means the seller gives reduced assurances about the title; it is accepted and priced, not fixed. Title indemnity insurance can sometimes cover a SPECIFIC identified gap, but the limited guarantee itself is a feature of the sale, not a clearable defect.',
     None),
    (('cannot assign contract', 'buyer cannot assign', 'no assignment', 'assign contract to third party'), TIME_COST_NO_RESOLUTION, TIME_COST_NO_RESOLUTION,
     'Acquisition term — no resolution cost. A no-assignment clause means the winning bidder cannot sell the contract on before completion; it is a standard auction term to accept, not a defect to fix. Relevant only if the buyer intended to flip the contract pre-completion.',
     None),
    (('sold as seen', 'no condition warranties', 'property sold as seen'), TIME_COST_NO_RESOLUTION, TIME_COST_NO_RESOLUTION,
     "Risk-acceptance — no resolution cost on the legal side. 'Sold as seen / no condition warranties' is the norm at auction: the buyer takes the property in its physical state with no recourse for defects. The 'handling' is a pre-bid survey to understand condition — a due-diligence step, not a cost to clear a legal defect. Any actual repairs are property-specific and outside this flag.",
     None),
    (('right-to-buy covenant', 'right to buy covenant', 'resale/subletting', 'resale restriction', 'subletting restriction'), TIME_COST_NO_RESOLUTION, TIME_COST_NO_RESOLUTION,
     "Constraint / risk-acceptance — no standard resolution cost. Right-to-Buy covenants can restrict resale or subletting for a period and may trigger a discount-repayment liability if the property is sold within the restricted window. The figure depends entirely on the original RTB discount and timing — read the specific covenant; there is no generic cost to 'clear' it.",
     None),
    (('buyer bears risk from exchange', 'risk from exchange', 'no seller insurance obligation', 'insurance from exchange'), TIME_COST_NO_RESOLUTION, TIME_COST_NO_RESOLUTION,
     "Acquisition/admin item — no resolution cost as a defect. Risk passing to the buyer on exchange (the standard position) means the buyer must have buildings insurance in force FROM exchange, not completion. The 'cost' is simply arranging a buildings policy (a few hundred £/yr) to start at exchange — an admin step, not a fix. The risk is being uninsured in the gap, which costs nothing to avoid if actioned.",
     None),
    (("discrepancy in seller's legal fees", 'legal fees between documents', 'discrepancy in seller'), TIME_COST_NO_RESOLUTION, TIME_COST_NO_RESOLUTION,
     "Query to raise — not a buyer cost. A discrepancy between documents in the seller's stated legal fees is a question for the buyer's solicitor to clarify before exchange, not a defect with a resolution price. Handling = raise the enquiry; cost is absorbed in standard conveyancing.",
     None),
    (('climateindex', 'climate index', 'climate moderate', 'climate risk'), TIME_COST_NO_RESOLUTION, TIME_COST_NO_RESOLUTION,
     'Disclosure / risk-acceptance — no resolution cost. A moderate climate-index score (a modelled long-horizon risk indicator) is contextual information for the valuation and insurability view, not a defect to remediate. Handling = factor into the long-term value/insurance outlook; there is nothing to buy or fix.',
     None),
    (('special conditions of sale not present', 'special conditions not present', 'special conditions missing'), TIME_COST_NO_RESOLUTION, TIME_COST_NO_RESOLUTION,
     "Request-it — no resolution cost, but a genuine gap to close BEFORE bidding. The Special Conditions of Sale govern the auction contract (deposit, completion period, costs, extra obligations) and must be obtained and read; bidding without them is bidding blind. Handling = request from the auctioneer/seller's solicitor at no charge.",
     None),
    (('conveyance document incomplete', 'lease document incomplete', 'transfer document incomplete', 'incomplete transfer document', 'incomplete conveyance', 'incomplete lease document', 'document incomplete', 'incomplete without preceding', 'transferee name blank', 'name blank in tr1', 'blank in tr1', 'consideration amount blank', 'tr1 form', 'blank in transfer', 'content unreadable', 'unreadable', 'illegible'), TIME_COST_NO_RESOLUTION, TIME_COST_NO_RESOLUTION,
     "Request-it — generally no resolution cost. An incomplete, blank, or unreadable conveyance, lease or transfer in the pack should be completed by obtaining Official Copy of a Document (Form OC2) from HM Land Registry — £11 per document by post, £7 per document via the portal/Business Gateway (GOV.UK, Land Registration Fee Order 2024, updated 9 Dec 2024) — or the missing pages from the seller's solicitor. HIGH confidence on the OC2 fee itself. Handling = request the complete document; only an unregistered lost deed needing reconstitution is more involved (and is a separate, rarer case).",
     "OC2 is a document-copy request, not a full title application — HM Land Registry's well-publicised multi-month backlogs apply to substantive applications (transfers, first registrations), not to ordering an existing document. Turnaround for a document-only request is materially faster, though GOV.UK does not publish a specific SLA for OC2."),
    (('overhanging structure', 'overhang', 'eavesdrop', 'trespass to air', 'encroachment over neighbour'), (0, 1), (50, 300),
     "Overhang / eavesdrop / air-space trespass: there is a named indemnity product (GCS 'Eavesdrop, Overhang and Trespass to Air-Space'), premium broadly £50-300 in line with standard title indemnities (GCS; Muve), issued quickly. Covers a neighbour seeking to prevent the overhang (guttering, balcony, flue, eaves). MEDIUM confidence — premium is value-banded.",
     'Indemnity covers a CHALLENGE to the encroachment, not its removal. An overhang can be a trespass even by a few inches, and if a neighbour pursues it and refuses to settle, the matter can escalate to a county court injunction (SAM Conveyancing; JustAnswer/UK property law). Where the overhang is part of a flying-freehold arrangement, see flying freehold (support/access rights and lender caps also apply).'),
    (('rooms over passageway', 'room over passage', 'over passageway', 'flying freehold passage', 'rooms over'), (0, 1), (50, 300),
     'Rooms over a passageway are a flying-freehold-type title arrangement (part of the property sits above land/airspace not in the same title). Handled like flying freehold: indemnity insurance £50-300 (Free Conveyancing Advice; Martin & Co; SAM Conveyancing), issued instantly. MEDIUM confidence on premium band.',
     'Indemnity covers loss of value / legal costs, NOT the underlying support, access and repair rights, which depend on the deeds (Girlings; Muve). Lenders treat flying freeholds cautiously and several cap the flying element or decline — mortgageability, not premium, is often the binding constraint. Confirm the deeds contain adequate support/access covenants.'),
    (('boundary moved', 'boundary alteration', 'boundary discrepancy', 'boundary disclosed', 'title plan mismatch', 'boundary altered'), (14, 21), (1500, 5000),
     'Boundary alteration/discrepancy: a measured boundary survey by a chartered surveyor £1,500-5,000, returned in ~2-3 weeks (RICS; Bennett Griffin; SAM Conveyancing). A determined-boundary application to HM Land Registry (£90 fee) can follow if the line needs formal fixing. MEDIUM-HIGH confidence on the survey cost.',
     'Land Registry title plans show only GENERAL boundaries, not the legal line (RICS; Clapham v Narga, CA 2024), so a disclosed alteration may or may not be a real problem. The survey PRICES CLARIFICATION; if it escalates to a contested boundary DISPUTE, costs rise steeply (mediation from ~£4,200; litigation £15,000-100,000+) — that is a separate, much larger risk, not included in the survey figure.'),
    (('epc present — rating appears low', 'rating appears low', 'epc rating low', 'low epc rating'), (1, 7), (45, 150),
     'Low EPC rating: a fresh EPC assessment is £45-150, booked and lodged within days (HomeSafety UK; standard assessor pricing). This prices CONFIRMING the rating; any energy-efficiency works to raise it (insulation, heating, glazing) are property-specific and vary widely (£100s-£1,000s+). MEDIUM confidence on the assessment cost.',
     'A low rating matters mainly for LETTINGS: the MEES minimum is currently EPC E to let, and a proposed future minimum of C (commonly cited ~2028) would raise upgrade costs (GOV.UK MEES guidance). For a buyer-occupier with no letting intent, a low rating is a running-cost/value factor, not a legal barrier. The works cost is NOT in the figure above.'),
    (('tenant covenants indemnity', 'leasehold tenant covenant', 'tenant covenant indemnity required'), (0, 1), (20, 300),
     "Leasehold/good-leasehold title indemnity where tenant-covenant or lease-validity assurance is missing: a 'good leasehold title' indemnity from ~£20 (Muve) up to a few hundred pounds depending on value, issued quickly. MEDIUM confidence — value-banded premium.",
     'This covers the RISK that the lease was not validly granted / the freehold title behind it is unproven — it does not cure the underlying title, and a lender may still require the freehold title to be deduced. Confirm the specific defect the indemnity is being asked to cover.'),
    (('no buildings insurance details', 'buildings insurance details', 'no buildings insurance'), (1, 7), (150, 600),
     'Missing buildings insurance details: for a freehold, the buyer simply arranges their own buildings policy (~£150-600/yr typical, property-dependent), in force from exchange. For a leasehold, the freeholder/managing agent insures and the details should be requested via the management pack. MEDIUM confidence — premium varies with property and rebuild cost.',
     "On a leasehold, 'no details provided' is a request-it gap (chase the managing agent), not a cost the buyer bears directly — the premium is recovered through the service charge. On a freehold, arranging cover from exchange is the action; being uninsured in the exchange-to-completion gap is the real risk (see 'risk from exchange')."),
    # S33-LIVE-DEAL (2026-06-28): two entries from live deal e541a62e.
    (('deemed to have full', 'deemed knowledge', 'deemed to have knowledge',
      'full planning knowledge', 'planning history unknown — buyer deemed',
      'buyer deemed', 'deemed full knowledge'),
     TIME_COST_NO_RESOLUTION, TIME_COST_NO_RESOLUTION,
     'No resolution cost — a deemed-knowledge clause shifts the onus onto the '
     'buyer (the contract treats you as knowing the planning/title position '
     'whether or not you actually checked). There is nothing to "clear": the '
     'handling is to do the due diligence yourself BEFORE bidding (planning '
     'register check, searches), since you cannot rely on the seller having '
     'disclosed it. The cost is your own pre-bid diligence time, not a fix to '
     'a defect. Same family as the seller-silence / no-enquiries flags.',
     None),
    (('title covers three properties', 'part-title', 'part title risk',
      'multiple properties one title', 'title covers multiple',
      'transfer of part', 'title split', 'split the title', 'part-title risk'),
     (20, 56), (200, 1000),
     'Part-title / title-split (transfer of part): where one registered title '
     'covers multiple properties and you may be buying only part, a transfer '
     'of part / title split is needed. Solicitor conveyancing £100-500 + Land '
     'Registry registration fee £20-125 + ~£8 ID check + £3 official copy; '
     'total commonly up to ~£1,000, simpler garden-style splits ~a couple of '
     'hundred pounds (SAM Conveyancing; Checkatrade; Propertyze, 2026). '
     'Land Registry turnaround ~4-8 weeks for electronic transfers (Property '
     'Passport, 2026). MEDIUM confidence — cost scales with complexity.',
     'This prices the PROCESS, not the tax: splitting a title can create new '
     'value and trigger Stamp Duty Land Tax, and possibly CGT on a transfer '
     'of equity — confirm with an accountant/solicitor (Propertyze; SAM '
     'Conveyancing). Also confirm whether a split is actually required: if '
     'the buyer intends to hold all parts, a single title with one mortgage '
     'may avoid an unnecessary SDLT charge. The figure assumes a genuine '
     'split is needed.'),
    (("nationally significant", "nsip"), TIME_COST_NO_RESOLUTION, TIME_COST_NO_RESOLUTION,
     "Risk-awareness flag — no standard resolution. NSIP proximity is not a "
     "clearable title/search defect: major infrastructure is consented via a "
     "Development Consent Order (Planning Act 2008), a multi-year process "
     "outside the conveyancing timeline (DLUHC NSIP Action Plan, 2023). "
     "Optional Energy & Infrastructure search ~£30-£79 can characterise the "
     "proximity (SAM Conveyancing; Local Conveyancing Direct) but detects/"
     "discloses only — it does not resolve. Impact is a valuation / risk-"
     "acceptance judgement, not a cost to clear.",
     None),
)

import re

# S33-DOCUMENT-STATED-FIGURE (2026-06-23): per explicit product correction —
# "10 day completion time = 10 days" was sitting in the flag's own evidence
# text the whole time, but the lookup only ever checked `title` against the
# research table, so it fell through to UNRESEARCHED even though the real
# answer was already extracted and present. Document-stated figures are MORE
# defensible than third-party research, not less — they're literally what
# the legal pack says about THIS property, not a general market range. This
# tier runs FIRST, before the research lookup. It also fixes the second
# point raised: time and cost do not have to both be known together — a
# flag can be time-only (e.g. "phone call to confirm, 1 day, no cost") or
# cost-only (e.g. "£1,500 plus VAT, due on completion, no wait"). Each field
# resolves independently; one being found in the document text does not
# require the other to also be found.

_TIME_PATTERNS = (
    # (regex, days extraction) — conservative: only matches explicit,
    # unambiguous day/week counts already stated in the text. Never infers.
    (re.compile(r'(\d+)\s*(?:business|working)\s*days?', re.I), lambda m: (int(m.group(1)), int(m.group(1)))),
    (re.compile(r'(\d+)\s*calendar\s*days?', re.I), lambda m: (int(m.group(1)), int(m.group(1)))),
    (re.compile(r'(\d+)\s*-\s*(\d+)\s*(?:business|working)\s*days?', re.I), lambda m: (int(m.group(1)), int(m.group(2)))),
    (re.compile(r'within\s*(\d+)\s*(?:hours|hrs)', re.I), lambda m: (0, 1)),  # same-day class
    (re.compile(r'(\d+)\s*weeks?', re.I), lambda m: (int(m.group(1)) * 5, int(m.group(1)) * 5)),  # 1 week = 5 working days
)

_COST_PATTERNS = (
    # Conservative: only matches an explicit £ figure already stated.
    # Does not attempt to parse "plus VAT" arithmetic — returns the stated
    # figure as-is; VAT-inclusive wording is preserved in the source text
    # the user already sees elsewhere on the card, not recalculated here.
    (re.compile(r'£\s*([\d,]+(?:\.\d{2})?)\s*-\s*£\s*([\d,]+(?:\.\d{2})?)'), lambda m: (
        float(m.group(1).replace(',', '')), float(m.group(2).replace(',', '')))),
    (re.compile(r'£\s*([\d,]+(?:\.\d{2})?)'), lambda m: (
        float(m.group(1).replace(',', '')), float(m.group(1).replace(',', '')))),
)

def _extract_stated_time(text: str):
    """Returns (low, high) days if an explicit time figure is found in the
    given text, else None. Conservative — only matches unambiguous,
    already-stated numbers; never infers or estimates."""
    if not text:
        return None
    for pattern, extractor in _TIME_PATTERNS:
        m = pattern.search(text)
        if m:
            return extractor(m)
    return None

def _extract_stated_cost(text: str):
    """Returns (low, high) £ if an explicit cost figure is found in the
    given text, else None. Conservative — only matches unambiguous,
    already-stated £ figures; never infers or estimates."""
    if not text:
        return None
    for pattern, extractor in _COST_PATTERNS:
        m = pattern.search(text)
        if m:
            return extractor(m)
    return None

def _flag_text_blob(flag: dict) -> str:
    """Concatenates the fields where a document-stated figure is most
    likely to already appear, in priority order. summation and implication
    are investor-facing summaries most likely to carry the key figure;
    evidence is the verbatim quote (max 30 words per the extraction prompt,
    so may be truncated); title is checked last as the least specific."""
    parts = [flag.get("summation"), flag.get("implication"), flag.get("evidence"), flag.get("title")]
    return " | ".join(p for p in parts if p)


def lookup_time_cost(flag) -> dict:
    """Resolves time_days and cost_gbp INDEPENDENTLY, each through its own
    three-tier order:
      1. Document-stated figure (regex-extracted from this flag's own
         summation/implication/evidence/title) — most defensible, since
         it's literally what the legal pack says about this property.
      2. Research lookup table (_TIME_COST_LOOKUP, matched on title)
         — used only if tier 1 found nothing for that field.
      3. TIME_COST_UNRESEARCHED — if neither tier found a real figure.
    Accepts either a flag dict (preferred — enables tier 1) or a bare title
    string (legacy call shape — tier 1 skipped, tier 2/3 only).
    A flag can resolve time at tier 1 and cost at tier 3 (or any other
    combination) — the two fields are never forced to match or to both be
    unknown together."""
    if isinstance(flag, dict):
        title = flag.get("title")
        text_blob = _flag_text_blob(flag)
    else:
        title = flag
        text_blob = title or ""

    title_l = (title or "").lower()

    research_time, research_cost, methodology, outlier_note = (
        TIME_COST_UNRESEARCHED, TIME_COST_UNRESEARCHED, None, None
    )
    for markers, t, c, m, o in _TIME_COST_LOOKUP:
        if any(mk in title_l for mk in markers):
            research_time, research_cost, methodology, outlier_note = t, c, m, o
            break

    stated_time = _extract_stated_time(text_blob)
    stated_cost = _extract_stated_cost(text_blob)

    if stated_time is not None:
        final_time = stated_time
        time_source = "document"
    else:
        final_time = research_time
        time_source = "research" if research_time != TIME_COST_UNRESEARCHED else "unresearched"

    if stated_cost is not None:
        final_cost = stated_cost
        cost_source = "document"
    else:
        final_cost = research_cost
        cost_source = "research" if research_cost != TIME_COST_UNRESEARCHED else "unresearched"

    show_methodology = (time_source == "research" or cost_source == "research")
    show_document_note = (time_source == "document" or cost_source == "document")
    # S33-UNRESEARCHED-REASON-v2 (2026-06-27): v1 of this fix (same session)
    # attached one static sentence to EVERY unresearched flag regardless of
    # what the flag actually was — confirmed live against the real audit:
    # "Intestate Death", "High Flood Risk", "118 Planning Applications",
    # "No TA6 Form", and "Radon Gas Affected Area" all rendered the
    # IDENTICAL sentence. Correctly called out as templated filler — worse
    # than the bare label it replaced, because it now LOOKS like effort was
    # spent saying nothing, which is the opposite of every domain reviewed
    # (WHO-UMC, ISA 705, GRADE): the reason must be specific to the actual
    # case, never a swapped-in template. This version differentiates using
    # only signal the system already has and can prove — severity (a real,
    # extracted field) and whether the flag's title hits a known
    # higher-level risk category — rather than inventing per-flag prose.
    # This is still a stopgap pending real per-category research (the
    # actual fix, as for ground rent/HMO/absent landlord/personal
    # covenant above) — it must read as visibly provisional, not as a
    # finished assessment.
    both_unresearched = (time_source == "unresearched" and cost_source == "unresearched")
    unresearched_reason = None
    if both_unresearched:
        sev = (flag.get("severity") or "").lower().strip() if isinstance(flag, dict) else ""
        if sev == "critical":
            unresearched_reason = (
                "Not yet costed. This is flagged CRITICAL severity with no "
                "researched benchmark behind it — the absence of a figure "
                "is a research gap, not a sign this is low-impact. Get a "
                "solicitor or specialist quote before relying on this deal's "
                "numbers."
            )
        elif sev == "high":
            unresearched_reason = (
                "Not yet costed. No researched benchmark or document-stated "
                "figure exists for this specific risk yet — treat the "
                "absence of a number as unresearched, not as low-cost."
            )
        elif sev == "missing":
            unresearched_reason = (
                "This flag records a missing document or disclosure, not a "
                "defect with a market cost to research — no figure is "
                "expected here. The action is to request the document, not "
                "to price it."
            )
        else:  # note / low / unclassified severity
            unresearched_reason = (
                "Not yet costed — no benchmark currently in the lookup "
                "table for this. Lower severity, but still genuinely "
                "unresearched rather than confirmed low-cost."
            )

    return {
        "time_days": final_time,
        "cost_gbp": final_cost,
        "time_source": time_source,
        "cost_source": cost_source,
        "methodology": methodology if show_methodology else (
            "Stated directly in the legal pack for this property." if show_document_note
            else (unresearched_reason if both_unresearched else None)
        ),
        "outlier_note": outlier_note if show_methodology else None,
    }

def attach_time_cost(flags: list[dict]) -> list[dict]:
    """Mutates each flag dict in place, adding time_days, cost_gbp,
    time_cost_methodology, and time_cost_outlier_note via lookup_time_cost.
    Safe to call multiple times (idempotent). Does not touch severity,
    value_adjustment, or any other existing field."""
    if not isinstance(flags, list):
        return flags
    for f in flags:
        if not isinstance(f, dict):
            continue
        result = lookup_time_cost(f)
        f["time_days"] = result["time_days"]
        f["cost_gbp"] = result["cost_gbp"]
        f["time_cost_methodology"] = result["methodology"]
        f["time_cost_outlier_note"] = result["outlier_note"]
        # SRA bar: indicative, not definitive. Always present, always true —
        # this is a statement about the nature of the figure, not a hedge
        # added only when uncertain.
        f["time_cost_indicative"] = True
    return flags

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
    # S-FIX (2026-06-28): category-keyed, not substring-matched against the
    # human-readable `reason` prose (was: "tenure"/"lease"/"legal_pack"/
    # "evidence" in r).
    #
    # 2026-06-30 DECISION — unquantified_risk and condition_risk increments
    # activated at +0.03 each (was 0.0 by accident of a substring typo in the
    # pre-S-FIX implementation — "legal_pack" underscore vs "legal-pack" / "legal pack"
    # hyphen/space — so they were inert since they were written, confirmed by
    # 2026-06-28 audit). Evidence for the decision:
    #   - unquantified_risk: CAP_UNQUANTIFIED_RISKS confirmed to never fire on
    #     any of the 19 live deals (unquantified_count=0 for all, because fallback
    #     pricing always assigns a non-zero value_adjustment). Zero current impact;
    #     activating establishes correct future behaviour.
    #   - condition_risk: CAN fire on real deals — 6 of 19 deals at 0.55 confidence
    #     (below Category-A cap of 0.59) are consistent with this cap having fired.
    #     When it fires (no-enquiries clause, squatters, extended probate), the property
    #     MAY not be comparable to the sold comps used, so the ceiling range SHOULD
    #     express that uncertainty. +0.03 widens a £300k deal's range from ±£15k to
    #     ±£24k, a £450k deal from ±£22.5k to ±£36k. Midpoint is UNCHANGED.
    #   - +0.03 calibrated below evidence_tier/tenure (+0.04): those cap for ABSENT
    #     data; these cap for DEGRADED quality. Strictly less severe.
    #   - subject_type/floor_area low confidence: +0.02 each — subject-resolution
    #     quality signals, less severe than legal-pack structural issues.
    # Golden test updated in test_ceiling_engine.py under the same tag.
    INCREMENTS = {
        "tenure":                          0.04,
        "lease":                           0.05,
        "evidence_tier":                   0.04,
        "comp_count":                      0.0,   # handled via valid_count path above
        "unquantified_risk":               0.03,  # 2026-06-30: activated (was 0.0 by typo)
        "condition_risk":                  0.03,  # 2026-06-30: activated (was 0.0 by typo)
        "subject_type_low_confidence":     0.02,  # 2026-06-30: S35-TYPE-CONF
        "subject_floor_area_low_confidence":0.02, # 2026-06-30: S35-AREA-CONF
    }
    for cap in caps:
        u += INCREMENTS.get(cap.get("category"), 0.0)
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
    # size_adj (the PRICE-rescaling multiplier feeding adjusted_value) must
    # ONLY ever use the subject's REAL area. Never the comp-set fallback —
    # rescaling a comp's price based on a comparison to a number that was
    # never the subject's true size would fabricate a size-adjustment with
    # no relationship to reality. If the real area is unknown, size_adj
    # stays exactly 1.00 (no-op), full stop, regardless of any fallback.
    size_adj  = _size_adjustment(subj_area, comp_area)
    # sz_score (relative WEIGHTING/confidence, never multiplies the price)
    # may use the comp-set median fallback when the subject's real area is
    # unknown — this only affects how much the weighted-median trusts this
    # comp relative to others in the set, not what price it contributes.
    _used_size_fallback = False
    _scoring_area = subj_area
    if subj_area is None and subject.get("_size_fallback_area"):
        _scoring_area = float(subject["_size_fallback_area"])
        _used_size_fallback = True
    size_ratio = (_scoring_area / comp_area) if (_scoring_area and comp_area and comp_area > 0) else None
    sz_score  = _size_score(size_ratio)
    # S35-COMP-SOURCE-DISCOUNT (2026-06-25): a comp's floor_area_source tag
    # (set by resolve_comp_size() in app.py) distinguishes the comp's OWN EPC
    # from a borrowed neighbour's ("comp_epc_postcode_any" — last-resort, any
    # EPC at the postcode). That tag previously existed nowhere and every comp
    # got equal size-trust regardless. This applies the SAME "unknown size"
    # penalty _size_score already uses (0.80) as a cap on a postcode-any
    # comp's score — not a new number, the existing scale's own floor applied
    # to a case it didn't previously see. A comp scoring 1.00 on a borrowed
    # neighbour's size is capped to 0.80; a comp already scoring below 0.80
    # (e.g. a genuine size outlier) is left at its own lower score, since the
    # discount should never make a bad match look better.
    _comp_size_source = comp.get("floor_area_source")
    if _comp_size_source == "comp_epc_postcode_any":
        sz_score = min(sz_score, 0.80)
        audit_warnings.append(
            "comp floor_area is from the nearest postcode EPC, not its own — "
            "size score capped at 0.80"
        )
    if subj_area is None or comp_area is None:
        if _used_size_fallback and comp_area is not None:
            audit_warnings.append(
                "subject floor_area unknown — weighted against comp-set median for "
                "ranking only; price NOT rescaled by size"
            )
        else:
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
            caps.append({"cap": CAP_COMPS_LT_3, "category": "comp_count", "reason": "valid_comparable_count < 3"})
            conf = min(conf, CAP_COMPS_LT_3)

    if n == 0:
        if conf > CAP_NO_VALID_COMPS:
            caps.append({"cap": CAP_NO_VALID_COMPS, "category": "comp_count", "reason": f"no valid comps within {MAX_RADIUS_MILES}mi"})
            conf = min(conf, CAP_NO_VALID_COMPS)

    if tenure_unknown:
        if conf > CAP_TENURE_UNRESOLVED:
            caps.append({"cap": CAP_TENURE_UNRESOLVED, "category": "tenure", "reason": "tenure unresolved and material"})
            conf = min(conf, CAP_TENURE_UNRESOLVED)

    if leasehold_material and lease_unknown:
        if conf > CAP_LEASE_MISSING:
            caps.append({"cap": CAP_LEASE_MISSING, "category": "lease", "reason": "lease length missing for leasehold"})
            conf = min(conf, CAP_LEASE_MISSING)

    if short_lease_no_band:
        if conf > CAP_SHORT_LEASE_NO_BAND:
            caps.append({"cap": CAP_SHORT_LEASE_NO_BAND, "category": "lease", "reason": "subject lease < 80 and no same-band lease comps"})
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
            caps.append({"cap": CAP_UNQUANTIFIED_RISKS, "category": "unquantified_risk", "reason": "material unquantified legal-pack value risks"})
            conf = min(conf, CAP_UNQUANTIFIED_RISKS)

    if condition_risk_flags:
        if conf > CAP_CONDITION_RISK_SIGNALS:
            caps.append({
                "cap": CAP_CONDITION_RISK_SIGNALS,
                "category": "condition_risk",
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
                "category": "evidence_tier",
                "reason": (
                    "evidence_tier=ppd_category_a_open_market — no distressed-sale "
                    "or auction comparables found; this is the weakest available "
                    "evidence tier and cannot be rated above Low confidence"
                ),
            })
            conf = min(conf, CAP_CATEGORY_A_ONLY)

    # S35-TYPE-CONF + S35-AREA-CONF (2026-06-30): subject-resolution quality caps.
    # Only fires when the value is specifically "low" — "high", "medium", and None
    # (old deals that pre-date this persist, or cases where resolution was clean)
    # all pass through with no cap. This closes the gap confirmed live against
    # 10 Lid Lane (DE6 2EG, 2026-06-25): the type was resolved on a 50/50 EPC
    # neighbour tiebreak (tagged "low") but the ceiling was labelled "High
    # confidence" — a direct contradiction. The comparable_valuation MIDPOINT is
    # not affected by this cap; only the label and the confidence.final score change.
    _type_conf_val        = (subject.get("type_confidence") or "").lower()
    _floor_area_conf_val  = (subject.get("floor_area_confidence") or "").lower()

    if _type_conf_val == "low":
        if conf > CAP_SUBJECT_TYPE_LOW_CONFIDENCE:
            caps.append({
                "cap":      CAP_SUBJECT_TYPE_LOW_CONFIDENCE,
                "category": "subject_type_low_confidence",
                "reason": (
                    "subject property type resolved with low confidence — "
                    "EPC neighbour evidence split (no clear majority) or "
                    "LLM listing-text crosscheck overrode a tiebreak; "
                    "comp type-matching and the valuation may be affected "
                    "if the resolved type is wrong"
                ),
            })
            conf = min(conf, CAP_SUBJECT_TYPE_LOW_CONFIDENCE)

    if _floor_area_conf_val == "low":
        if conf > CAP_SUBJECT_FLOOR_AREA_LOW_CONFIDENCE:
            caps.append({
                "cap":      CAP_SUBJECT_FLOOR_AREA_LOW_CONFIDENCE,
                "category": "subject_floor_area_low_confidence",
                "reason": (
                    "subject floor area estimated from partial room schedule "
                    "(low confidence) — only 1-2 room dimensions were "
                    "available; size normalisation of comp prices may be "
                    "less reliable than usual"
                ),
            })
            conf = min(conf, CAP_SUBJECT_FLOOR_AREA_LOW_CONFIDENCE)

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
            "_routing_source":  _flag_routing_source(flag),
            "reason": (
                f"severity={sev}; market_consequence_segments={list(segs.keys())}; "
                f"total_value_adjustment={total_adj}"
            ),
        })
    return risks


def risk_routing_coverage(risks: list[dict]) -> dict:
    """Summarise how many of a deal's risks matched a _SEGMENT_RULES entry
    vs fell through to the generic severity fallback. Reads the
    '_routing_source' tag _process_legal_risks already attaches to each risk
    — call this once per deal (e.g. when assembling legal_pack_value_risks)
    rather than re-deriving it by hand-testing flag text, which is how the
    71% fallback rate on a real deal was found this session.
    """
    tally = {"matched_rule": 0, "fallback": 0, "note": 0}
    for r in risks:
        tally[r.get("_routing_source", "fallback")] += 1
    substantive = tally["matched_rule"] + tally["fallback"]
    return {
        "matched_rule_count": tally["matched_rule"],
        "fallback_count":     tally["fallback"],
        "note_count":         tally["note"],
        "fallback_rate_pct": (
            round(100 * tally["fallback"] / substantive, 1) if substantive else None
        ),
    }

def _legal_pack_adjustment_factor(risks: list[dict]) -> float:
    """
    S37-SEGMENT-CAPS (2026-07-03): three-part restructure from the research
    recommendation — (1) boilerplate vs property-specific classification
    [see _SEGMENT_RULES covenant/title-guarantee/completion splits],
    (2) per-segment sub-caps [see _SEGMENT_CAPS], (3) diminishing-marginal
    combination across segments [this function]. An earlier version of this
    fix implemented only (1) and (2) and was verified NOT to work: even with
    every segment individually capped, straight multiplication of 5 bounded
    terms still produced a 55% worst-case combined reduction — because
    multiplicative combination was itself still the naive-stacking mechanism
    the research identified as the problem, regardless of what feeds into it.
    That gap is what part (3) closes.

    Method:
      1. Sum each risk's segment contributions into per-segment totals
         (unchanged from the two-part version).
      2. Clamp each segment total at _SEGMENT_CAPS[segment] (unchanged).
      3. Sort the capped segment totals descending, then combine with
         geometrically decaying weight (each subsequent segment contributes
         half the marginal weight of the one before it):
             total_reduction = sum(segment_i * 0.5**i) for i = 0, 1, 2, ...
         This is the same underlying principle as trauma medicine's Injury
         Severity Score, which sums only the top-ranked injury regions with
         deliberately diminishing weight rather than summing all injuries
         unconditionally — multiple simultaneous factors are real, but each
         additional one should move the outcome less than the last, not
         compound it. Verified worst case (all 5 segments simultaneously at
         their cap): 35.5% — versus 55.0% under straight multiplication of
         the same capped inputs, and versus the old flag-level method's
         effectively unbounded growth with flag count.
      4. MAX_TOTAL_VALUE_RISK_ADJ (35%) is retained as a final backstop —
         now a genuine edge-case safeguard rather than the typical outcome.
    """
    segment_totals: dict[str, float] = {}
    for r in risks:
        for seg, frac in r.get("segments", {}).items():
            if frac <= 0:
                continue
            segment_totals[seg] = segment_totals.get(seg, 0.0) + frac

    capped_totals = sorted(
        (min(total, _SEGMENT_CAPS.get(seg, MAX_TOTAL_VALUE_RISK_ADJ))
         for seg, total in segment_totals.items()),
        reverse=True,
    )

    total_reduction = sum(c * (0.5 ** i) for i, c in enumerate(capped_totals))
    total_reduction = min(total_reduction, MAX_TOTAL_VALUE_RISK_ADJ)

    return round(1.0 - total_reduction, 6)


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
    Verdict ceiling: weighted median of relational comparable evidence within
    MAX_RADIUS_MILES (3.0mi), distance-weighted via EXTENDED_DISTANCE_BANDS
    (S33-STEP1, 2026-06-21) — not a 0.5-mile hard cutoff.
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

    # S33-TIME-COST-LOOKUP (2026-06-22): attach time_days/cost_gbp to every
    # active flag via deterministic lookup (see lookup_time_cost above) —
    # never an LLM estimate. This is the one production code path that
    # receives the real flags array; attaching here means the frontend
    # (legalsmegal-workbench.html, renderTimeCostRows) finally has real data
    # to render instead of nothing, without touching the LLM extraction
    # prompts in app.py at all.
    attach_time_cost(active_legal_flags)

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
            "routing_coverage":  risk_routing_coverage(active_risks),
            # S40 (2026-07-04): calibration honesty layer — every risk-
            # adjusted figure carries its calibration provenance. See
            # CALIBRATION_METADATA / get_calibration_disclosure().
            "calibration":       get_calibration_disclosure(),
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

    formula_trace.append(f"step_1: type+tenure match + HPI normalisation within {MAX_RADIUS_MILES}mi")

    _hpi_used_count    = 0
    _hpi_missing_count = 0

    # ── Size-fallback for ranking only (NOT a claim about the subject) ───────
    # When subject.internal_area is genuinely unknown (resolve_subject_size()
    # returned None — no EPC, no room schedule), every comp previously got
    # size_adjustment=1.00 and size_score via _size_score(None)=0.80 flat,
    # meaning a 59 m² comp and a 119 m² comp were weighted IDENTICALLY on
    # size — the comp set's own size spread had zero influence on which
    # comps the weighted-median trusted more. This does not invent a subject
    # size: it uses the comp set's OWN median floor area purely so comps near
    # that median score better than comps at the tails — comps closer to
    # "what a typical comp in this set looks like" outrank outliers in either
    # direction, instead of all being scored as equally (un)informative.
    # subject["internal_area"] itself is left untouched — still None, still
    # flows through to the user-facing "size normalisation unavailable"
    # warning exactly as before. This only affects internal comp weighting.
    _subj_area_known = bool(subject.get("internal_area") or subject.get("floor_area"))
    _size_fallback_used = False
    if not _subj_area_known:
        _comp_areas = []
        for _c in sold_comps:
            _a = _c.get("internal_area") or _c.get("floor_area") or _c.get("total_floor_area")
            try:
                _a = float(_a) if _a else None
            except (TypeError, ValueError):
                _a = None
            if _a and _a > 0:
                _comp_areas.append(_a)
        if len(_comp_areas) >= 3:
            _comp_areas.sort()
            _mid = len(_comp_areas) // 2
            _comp_median_area = (
                _comp_areas[_mid] if len(_comp_areas) % 2 == 1
                else (_comp_areas[_mid - 1] + _comp_areas[_mid]) / 2
            )
            subject = dict(subject)  # don't mutate the caller's dict
            subject["_size_fallback_area"] = _comp_median_area  # internal use only, see _assess_comp
            _size_fallback_used = True
            formula_trace.append(
                f"step_1_size_fallback: subject area unknown — using comp-set "
                f"median {_comp_median_area:.0f} m² (n={len(_comp_areas)}) for "
                f"size SCORING ONLY, not as a claim about the subject"
            )
            warnings.append(
                "size_fallback_used: subject floor area unknown — comps are "
                "ranked against the comparable set's own median size as a "
                "working estimate; the comparable valuation has not been "
                "adjusted to the subject's actual size"
            )

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
            "radius_miles": MAX_RADIUS_MILES,  # engine's outer inclusion bound (S33-STEP1); PRIMARY_RADIUS_MILES drives confidence labelling only
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
                    "comparables": {"radius_miles": MAX_RADIUS_MILES, "valid": [], "excluded": []},
                    "legal_pack_value_risks": {
                        "method":           "property_value_risk_adjustment_only",
                        "adjustment_factor": 1.0, "adjusted_value": None, "risks": [],
                    },
                    "confidence": (
                        _legacy.get("confidence")
                        if isinstance(_legacy.get("confidence"), dict)
                        else {"final": _legacy.get("confidence") if isinstance(_legacy.get("confidence"), (int, float)) else 0.45,
                              "label": "Low confidence", "_legacy_scalar_normalised": True}
                    ),
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
