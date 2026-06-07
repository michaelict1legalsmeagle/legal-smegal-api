"""
ceiling_engine.py — LegalSmegal Ceiling Engine v2.0
====================================================
Red-book-style paper valuation similarity engine.

NOT a RICS Red Book valuation.
NOT institutional underwriting.
NOT an acquisition-cost model.

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
PRIMARY_RADIUS_MILES = 0.5
MIN_REQUIRED_COMPS   = 3
PREFERRED_COMPS      = 5

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
DISTANCE_BANDS = [
    (0.00, 0.10, 1.00),
    (0.10, 0.25, 0.90),
    (0.25, 0.50, 0.80),
]

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
    """Map flag severity to property-value risk adjustment."""
    sev = (flag.get("severity") or "note").lower().strip()
    return VALUE_RISK_SEVERITY_ADJ.get(sev, 0.00)

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
    if miles is None or miles > PRIMARY_RADIUS_MILES:
        return None  # outside primary radius — exclude
    for lo, hi, score in DISTANCE_BANDS:
        if lo <= miles <= hi:
            return score
    return 0.80  # exactly 0.5 miles

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
        return None, {"comp_idx": comp_idx, "reason": f"outside_0.5_miles dist={dist}", "comp": comp}

    # Duplicate detection by address
    comp_addr = (comp.get("address") or "").strip().lower()
    # (dedup is done at the call site across the comp universe)

    # Type match
    subj_type = (subject.get("property_type") or "").lower()
    comp_type = (comp.get("property_type") or comp.get("type") or "").lower()
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


def _types_near_equiv(a: str, b: str) -> bool:
    FLAT_TERMS = {"flat", "apartment", "maisonette"}
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

    unquantified = any(r.get("value_adjustment", 0) == 0 and r.get("severity", "note") not in ("note", "low")
                       for r in included_risks)
    if unquantified:
        if conf > CAP_UNQUANTIFIED_RISKS:
            caps.append({"cap": CAP_UNQUANTIFIED_RISKS, "reason": "material unquantified legal-pack value risks"})
            conf = min(conf, CAP_UNQUANTIFIED_RISKS)

    final = round(max(0.00, min(1.00, conf)), 2)
    return final, caps, _confidence_label(final)

# =============================================================================
# LEGAL-PACK VALUE RISK PROCESSING
# =============================================================================
def _process_legal_risks(legal_flags: list[dict]) -> list[dict]:
    """
    Filter and map legal flags to property-value risks only.
    Excludes acquisition costs.
    """
    risks = []
    for i, flag in enumerate(legal_flags):
        if _is_acquisition_cost_only(flag):
            continue
        sev = (flag.get("severity") or "note").lower().strip()
        adj = _value_risk_adjustment(flag)
        if adj == 0.0 and sev == "note":
            # notes with no value impact: include for completeness but mark adj=0
            pass
        risks.append({
            "risk_id":         f"r{i:03d}",
            "title":           flag.get("title", ""),
            "source_evidence": flag.get("summation", "") or flag.get("implication", ""),
            "category":        flag.get("risk_category") or flag.get("category") or "general",
            "severity":        sev,
            "value_adjustment":round(adj, 4),
            "included":        True,
            "reason":          f"severity={sev}; value_adjustment={adj}",
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
    verdict_mid  = verdict_vr.get("midpoint")
    verdict_low  = verdict_vr.get("low")
    verdict_high = verdict_vr.get("high")
    u_band       = verdict_vr.get("uncertainty_band", BASE_UNCERTAINTY)

    # If verdict has no valid midpoint, workbench is also insufficient
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

    return {
        "_ceiling_type": "workbench",
        "status": "all_flags_resolved" if all_resolved else verdict_ceiling.get("status", "ok"),
        "valuation_range": {
            "low":              wb_low,
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
            "adjusted_value":    wb_mid,
            "risks":             active_risks,
        },
        "risk_discount_pct":   risk_discount_pct,
        "active_flag_count":   len(active_legal_flags),
        "all_flags_resolved":  all_resolved,
        "verdict_midpoint":    verdict_mid,
        "verdict_range": {
            "low":      verdict_low,
            "midpoint": verdict_mid,
            "high":     verdict_high,
        },
        "confidence":        verdict_ceiling.get("confidence"),
        "base":              verdict_ceiling.get("base"),
        "base_valuation":    verdict_ceiling.get("base_valuation"),
        "base_method":       verdict_ceiling.get("base_method"),
        "strategy_used":     verdict_ceiling.get("strategy_used"),
        "audit": {
            "verdict_ceiling_midpoint": verdict_mid,
            "active_flag_count":        len(active_legal_flags),
            "all_flags_resolved":       all_resolved,
            "risk_adjustment_factor":   risk_factor,
            "risk_discount_pct":        risk_discount_pct,
            "formula": "workbench_midpoint = verdict_midpoint × active_flag_risk_factor",
            "resolved_flags_excluded":  True,
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

    # ── STEP 1: Process comparable universe ──────────────────────────────────
    valid_comps:    list[dict] = []
    excluded_comps: list[dict] = []
    seen_addresses: set[str]   = set()

    formula_trace.append("step_1: assess comparable universe within 0.5 miles")

    for idx, comp in enumerate(sold_comps):
        addr = (comp.get("address") or "").strip().lower()
        if addr and addr in seen_addresses:
            excluded_comps.append({"comp_idx": idx, "reason": "duplicate_address", "comp": comp})
            continue
        if addr:
            seen_addresses.add(addr)

        valid, excl = _assess_comp(comp, subject, idx)
        if valid:
            valid_comps.append(valid)
        else:
            excluded_comps.append(excl)

    n_valid = len(valid_comps)
    formula_trace.append(f"step_1_result: valid_comps={n_valid} excluded={len(excluded_comps)}")

    # ── STEP 2: Compute base_value ───────────────────────────────────────────
    insufficient_evidence = False
    base_value: Optional[float] = None
    base_method = "weighted_median_relational_comparables_0_5_mile"

    if base_valuation and float(base_valuation) > 5_000:
        base_value  = float(base_valuation)
        base_method = "external_override"
        formula_trace.append(f"step_2: base_value={base_value} method=external_override")

    elif n_valid == 0:
        insufficient_evidence = True
        formula_trace.append("step_2: insufficient_evidence — no valid comps within 0.5 miles")
        evidence_gaps.append("No valid sold comparables within 0.5 miles")

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
            formula_trace.append(f"step_2: base_value={base_value} method=weighted_median n_valid={n_valid}")
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
    formula_trace.append("step_4: ceiling_midpoint = base_value × risk_adj_factor")

    ceiling_midpoint: Optional[float] = None
    ceiling_low:      Optional[float] = None
    ceiling_high:     Optional[float] = None

    if not insufficient_evidence and base_value and base_value > 0:
        ceiling_midpoint = round(base_value * risk_adj_factor, 2)
        formula_trace.append(f"step_4_result: ceiling_midpoint={ceiling_midpoint}")
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
        )

    formula_trace.append(f"step_5_result: confidence={final_conf} label={conf_label}")

    # ── STEP 6: Uncertainty band ─────────────────────────────────────────────
    formula_trace.append("step_6: uncertainty_band")
    u_band = _uncertainty_band(n_valid, conf_caps)
    formula_trace.append(f"step_6_result: uncertainty_band={u_band}")

    if ceiling_midpoint:
        ceiling_low  = round(ceiling_midpoint * (1 - u_band), 2)
        ceiling_high = round(ceiling_midpoint * (1 + u_band), 2)
        formula_trace.append(f"step_6: ceiling_low={ceiling_low} ceiling_high={ceiling_high}")

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
        },
        "comparables": {
            "radius_miles": PRIMARY_RADIUS_MILES,
            "valid":        valid_comps,
            "excluded":     excluded_comps,
        },
        "legal_pack_value_risks": {
            "method":            "property_value_risk_adjustment_only",
            "adjustment_factor": risk_adj_factor,
            "adjusted_value":    ceiling_midpoint,
            "risks":             included_risks,
        },
        "valuation_range": {
            "low":              ceiling_low,
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
            "assumptions":  assumptions,
            "evidence_gaps": evidence_gaps,
            "warnings":     warnings,
            "formula_trace": formula_trace,
            "version":      VERSION,
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
        and isinstance(_existing_vc.get("valuation_range"), dict)
        and (_existing_vc["valuation_range"].get("midpoint") or 0) > 0
    )

    if _vc_valid:
        verdict = _existing_vc
        audit_notes.append("verdict_ceiling: preserved (existing valid object)")
    else:
        # For old deals without verdict_ceiling: use legacy summary_json.ceiling directly
        # as the verdict base — it is what the Verdict page displays and represents
        # the correct comparable ceiling for that deal.
        # Only call calculate_verdict_ceiling when no legacy ceiling exists.
        _legacy = summary_json.get("ceiling") or {}
        _leg_base = None
        _leg_lo   = None
        _leg_hi   = None
        try:
            _lb = _legacy.get("base_valuation")
            if _lb and float(_lb) > 5000:
                _leg_base = float(_lb)
            _lcr = _legacy.get("ceiling_range") or _legacy.get("valuation_range") or {}
            _lclo = _lcr.get("low")
            _lchi = _lcr.get("high")
            if _lclo and float(_lclo) > 5000:
                _leg_lo = float(_lclo)
            if _lchi and float(_lchi) > 5000:
                _leg_hi = float(_lchi)
        except (TypeError, ValueError):
            pass

        if _leg_base and _leg_base > 5000:
            # Build verdict_ceiling from the legacy ceiling — preserving the actual range
            _ub = 0.05
            _v_lo  = _leg_lo  if _leg_lo  else round(_leg_base * (1 - _ub), 2)
            _v_hi  = _leg_hi  if _leg_hi  else round(_leg_base * (1 + _ub), 2)
            verdict = {
                "_ceiling_type":   "verdict",
                "_legacy_source":  True,
                "status":          "ok",
                "base": {"value": _leg_base, "method": _legacy.get("base_method", "legacy_ceiling")},
                "base_valuation":  int(round(_leg_base)),
                "base_method":     _legacy.get("base_method", "legacy_ceiling"),
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
                    "method": "property_value_risk_adjustment_only",
                    "adjustment_factor": 1.0, "adjusted_value": None, "risks": [],
                },
                "confidence": _legacy.get("confidence") or {"final": 0.45, "label": "Low confidence"},
                "audit": {
                    "assumptions": ["base_value and range from legacy summary_json.ceiling; re-analyse for relational comparable base"],
                    "evidence_gaps": [],
                    "warnings": ["verdict_ceiling built from legacy ceiling — re-analyse to compute relational comparable base"],
                    "formula_trace": ["legacy_source: base_valuation from summary_json.ceiling.base_valuation"],
                    "version": VERSION,
                },
                "acquisition_costs": None,
                "excluded_from_ceiling": EXCLUDED_FROM_CEILING,
            }
            audit_notes.append(f"verdict_ceiling: built from legacy ceiling base={_leg_base} lo={_v_lo} hi={_v_hi}")

        elif _sold_comps:
            # No legacy base — try relational engine with sold comps
            verdict = calculate_verdict_ceiling(
                sold_comps=_sold_comps,
                subject=subject,
                strategy=strategy,
                fallback_allowed=True,
            )
            if (verdict.get("valuation_range") or {}).get("midpoint"):
                audit_notes.append(f"verdict_ceiling: computed from {len(_sold_comps)} sold comps")
            else:
                audit_notes.append("verdict_ceiling: insufficient comps and no legacy ceiling — missing_data")

        else:
            # No legacy base, no comps — explicit missing-data state
            verdict = {
                "_ceiling_type": "verdict",
                "status": "missing_data",
                "valuation_range": {"low": None, "midpoint": None, "high": None, "uncertainty_band": None},
                "ceiling_range":   {"low": None, "high": None},
                "base": {"value": None, "method": "none"},
                "confidence": {"final": 0.0, "caps": [{"cap": 0.0, "reason": "no_data"}], "label": "Insufficient evidence"},
                "audit": {"warnings": ["no sold comps and no legacy ceiling — missing_data"], "version": VERSION},
                "acquisition_costs": None,
                "excluded_from_ceiling": EXCLUDED_FROM_CEILING,
            }
            audit_notes.append("verdict_ceiling: missing_data — no comps and no legacy ceiling")

        summary_json["verdict_ceiling"] = verdict

    # ── 2. Validate / compute workbench_ceiling ───────────────────────────
    _existing_wb = summary_json.get("workbench_ceiling")
    _v_mid = (verdict.get("valuation_range") or {}).get("midpoint") or 0
    _wb_valid = (
        isinstance(_existing_wb, dict)
        and isinstance(_existing_wb.get("valuation_range"), dict)
        and (_existing_wb["valuation_range"].get("midpoint") or 0) <= _v_mid + 1  # clamp tolerance £1
        and (_existing_wb["valuation_range"].get("midpoint") or 0) > 0
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
