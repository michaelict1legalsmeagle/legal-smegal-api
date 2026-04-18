"""
ceiling_engine.py — LegalSmegal Bid Ceiling Engine v2.0
========================================================
RICS-aligned, comps-led investment value ceiling for UK residential auction.

ARCHITECTURE
------------
Part 1 — Valuation Adjustment Engine (RICS Market Approach)
Part 2 — Risk Engine (Structural + Asset, multiplicative, non-linear)
Part 3 — Ceiling Output (Investment Value + Dynamic Range)
Part 4 — Value Secured (UX Engine)
Part 5 — Confidence Model
Part 6 — JSON Output

SCOPE RESTRICTION
-----------------
Standard UK residential auction property ONLY.
Returns manual_review_required for commercial, mixed-use, land, portfolios,
development sites, complex leasehold structures.

CRITICAL PROHIBITIONS
---------------------
- Do NOT deduct SDLT, legal fees, or finance costs from ceiling
- Do NOT stack risk linearly
- Do NOT average comps blindly
- Do NOT output a single number (always a range)
- Do NOT double-count condition

Decision-support only. Not financial advice.
LegalSmegal Technologies Ltd is not FCA-regulated.
"""

from __future__ import annotations
import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# PART 1 — VALUATION ADJUSTMENT ENGINE
# =============================================================================

# Time index — simplified UK HPI proxy by region
# v2: replace with live UKHPI API call
UK_HPI_REGIONAL: dict[str, float] = {
    "london":       1.000,
    "south_east":   0.985,
    "east":         0.978,
    "south_west":   0.972,
    "east_midlands":0.965,
    "west_midlands":0.963,
    "yorkshire":    0.958,
    "north_west":   0.961,
    "north_east":   0.955,
    "wales":        0.952,
    "scotland":     0.960,
    "default":      0.965,
}

def _time_adjustment(sale_months_ago: float, region: str = "default") -> float:
    """
    T_adj = Index_current / Index_sale
    Approximates HPI growth: 3% annualised, region-weighted.
    v2: replace with UKHPI monthly index lookup.
    """
    if sale_months_ago <= 0:
        return 1.0
    annual_growth = 0.03
    regional_factor = UK_HPI_REGIONAL.get(
        (region or "default").lower().replace(" ", "_"), 0.965
    )
    months_growth = (1 + annual_growth) ** (sale_months_ago / 12)
    return round(regional_factor * months_growth + (1 - regional_factor), 4)


def _condition_adjustment(condition: str) -> float:
    """Condition relative adjustment — applied to per-sqm price."""
    mapping = {
        "good":      0.00,
        "average":   0.00,
        "poor":     -0.12,
        "very_poor": -0.25,
    }
    return mapping.get((condition or "average").lower().replace(" ", "_"), 0.00)


def _tenure_adjustment(lease_years: Optional[float], tenure: str) -> float:
    """
    Tenure adjustment applied to comparable value.
    Only applied when comp and subject share the same structural issue.
    """
    if tenure and tenure.lower() == "freehold":
        return 0.00
    if lease_years is None:
        return 0.00
    if lease_years > 80:
        return -0.05
    if lease_years >= 70:
        return -0.20
    return -0.35


def _location_adjustment(relationship: str) -> float:
    """Location quality adjustment between comp and subject."""
    mapping = {
        "same_street":      0.00,
        "same_micro":       0.00,
        "adjacent_better": +0.07,
        "adjacent_worse":  -0.07,
        "different_better":+0.12,
        "different_worse": -0.12,
    }
    return mapping.get(
        (relationship or "same_micro").lower().replace(" ", "_"), 0.00
    )


def _comp_weight(
    distance_miles: float,
    months_ago: float,
    similarity_score: float,  # 0–1
) -> float:
    """
    Composite weight: recency × proximity × similarity.
    Recency decays at 6-month half-life.
    Proximity decays at 1-mile half-life.
    """
    recency   = math.exp(-months_ago / 6)
    proximity = math.exp(-distance_miles / 1.0)
    return round(recency * proximity * similarity_score, 6)


def _adjust_comp(
    comp: dict,
    subject_area_sqm: float,
    region: str,
) -> Optional[dict]:
    """
    Apply RICS adjustments to a single comparable.
    Returns None if comp fails validity filter.
    """
    # Validity filter
    asset_class = (comp.get("asset_class") or "residential").lower()
    if asset_class not in ("residential", "flat", "house", "apartment"):
        return None  # wrong asset class — exclude

    price       = float(comp.get("price", 0) or 0)
    area_sqm    = float(comp.get("area_sqm", 0) or 0)
    months_ago  = float(comp.get("months_ago", 0) or 0)
    distance    = float(comp.get("distance_miles", 0.5) or 0.5)
    similarity  = float(comp.get("similarity_score", 0.8) or 0.8)
    condition   = comp.get("condition", "average") or "average"
    tenure      = comp.get("tenure", "") or ""
    lease_years = comp.get("lease_years")
    location_rel = comp.get("location_relationship", "same_micro")

    if price <= 0 or area_sqm <= 0:
        return None

    # Size normalisation
    price_sqm = price / area_sqm

    # Time adjustment
    t_adj = _time_adjustment(months_ago, region)

    # All adjustments applied to price_sqm
    cond_adj     = _condition_adjustment(condition)
    tenure_adj   = _tenure_adjustment(
        float(lease_years) if lease_years else None, tenure
    )
    location_adj = _location_adjustment(location_rel)

    # Adjusted price per sqm for subject property
    adj_price_sqm = price_sqm * t_adj * (1 + cond_adj + tenure_adj + location_adj)
    adjusted_value = adj_price_sqm * subject_area_sqm

    # Validity: outlier filter — exclude if adjusted value < £20k or > £5m
    if adjusted_value < 20_000 or adjusted_value > 5_000_000:
        return None

    weight = _comp_weight(distance, months_ago, similarity)

    return {
        "original_price":   price,
        "adjusted_value":   round(adjusted_value),
        "weight":           weight,
        "adjustments": {
            "time":      round(t_adj, 4),
            "condition": round(cond_adj, 4),
            "tenure":    round(tenure_adj, 4),
            "location":  round(location_adj, 4),
        },
    }


def _derive_market_value(
    comparables: list[dict],
    subject_area_sqm: float,
    region: str,
    fallback_avg: Optional[float],
) -> tuple[float, str, float]:
    """
    Weighted market value from adjusted comparables.
    Returns (market_value, method_label, variance_pct).
    """
    if not comparables and fallback_avg and fallback_avg > 5_000:
        return (float(fallback_avg), "comps_avg_fallback", 0.15)

    adjusted = [
        r for r in [
            _adjust_comp(c, subject_area_sqm, region)
            for c in (comparables or [])
        ]
        if r is not None
    ]

    if not adjusted and fallback_avg and fallback_avg > 5_000:
        return (float(fallback_avg), "comps_avg_fallback", 0.15)

    if not adjusted:
        return (0.0, "none", 0.0)

    total_weight = sum(r["weight"] for r in adjusted)
    if total_weight <= 0:
        mv = sum(r["adjusted_value"] for r in adjusted) / len(adjusted)
        return (round(mv), "comps_unweighted", 0.12)

    weighted_mv = sum(
        r["adjusted_value"] * r["weight"] for r in adjusted
    ) / total_weight

    # Variance: coefficient of variation of adjusted values
    values = [r["adjusted_value"] for r in adjusted]
    mean   = weighted_mv
    if len(values) > 1:
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        cv  = std / mean if mean > 0 else 0.15
    else:
        cv = 0.12  # single comp — assume moderate uncertainty

    return (round(weighted_mv), "comps_weighted_adjusted", round(cv, 4))


# =============================================================================
# PART 2 — RISK ENGINE
# =============================================================================

# D_max lookup table — severity-driven position within range
# Format: (low, mid, high) — critical=high end, high=mid, note=low
D_MAX_TABLE: dict[str, tuple[float, float, float]] = {
    "short_lease_80":       (0.20, 0.225, 0.25),   # 70–80 years
    "short_lease_70":       (0.30, 0.325, 0.35),   # <70 years
    "regulated_tenancy":    (0.25, 0.30,  0.35),
    "possessory_title":     (0.12, 0.15,  0.18),
    "restrictive_covenant": (0.10, 0.15,  0.20),
    "structural":           (0.10, 0.15,  0.20),
    "standard_critical":    (0.07, 0.09,  0.10),  # spec: 10% at critical severity
    "standard_high":        (0.03, 0.05,  0.07),
    "standard_missing":     (0.02, 0.035, 0.05),
    "standard_note":        (0.01, 0.015, 0.02),
}

# Keyword patterns → defect category
D_MAX_KEYWORDS: list[tuple[list[str], str]] = [
    (["short lease", "lease.*less than 80", "lease.*under 80",
      r"lease.{0,10}7\d year"], "short_lease_80"),
    (["lease.*less than 70", "lease.*under 70",
      r"lease.{0,10}6\d year", r"lease.{0,10}[1-5]\d year",
      "enfranchisement"], "short_lease_70"),
    (["regulated tenancy", "rent act", "protected tenant",
      "sitting tenant", "security of tenure",
      "long residential occupier"], "regulated_tenancy"),
    (["possessory title", "possessory freehold",
      "possessory leasehold", "unregistered title",
      "title not registered"], "possessory_title"),
    (["restrictive covenant", "covenant.*restrict",
      "no hmo", "single dwelling",
      "restriction on use"], "restrictive_covenant"),
    (["subsidence", "structural", "knotweed", "mining",
      "flood zone 3", "asbestos", "underpinning"], "structural"),
]

import re as _re


# =============================================================================
# FLAG BEHAVIOUR CONTROL — v2.1
# =============================================================================
# Priority order inside calculate_ceiling():
#   Step 1 — STOP check  (before any valuation)
#   Step 2 — Run existing model unchanged
#   Step 3 — CAP check   (after valuation, before range output)
#   Step 4 — Normal range output
#
# "stop"   -> return manual_review_required, ceiling null
# "cap"    -> value = min(value, base * CAP_CEILING_PCT)
# "normal" -> standard D_asset contribution (default)
# =============================================================================

CAP_CEILING_PCT = 0.45

STOP_KEYWORDS: list[list[str]] = [
    ["no legal access", "no access to property", "landlocked",
     "no right of way confirmed"],
    ["regulated tenancy", "rent act 1977", "protected tenant",
     "sitting tenant.*security of tenure",
     "long residential occupier.*cannot.*vacant"],
    ["active.*underpinning", "underpinning.*in progress",
     "structural movement.*active", "active.*structural movement"],
    ["possessory.*no indemnity", "title.*defective.*no insurance",
     "unregistered.*no deeds"],
]

CAP_KEYWORDS: list[list[str]] = [
    ["lease.*under 60", "lease.*less than 60",
     r"lease.{0,10}[1-5]\d year"],
    ["above.*shop", "above.*commercial", "above.*retail"],
    ["non.standard construction", "non-standard construction",
     "bisf", "airey", "cornish unit", "reema", "woolaway"],
]


def _resolve_flag_behaviour(flag: dict) -> str:
    """
    Return 'stop' | 'cap' | 'normal' for a single flag.
    Priority: explicit flag.behaviour field -> keyword match -> 'normal'.
    """
    explicit = (flag.get("behaviour") or "").lower()
    if explicit in ("stop", "cap", "normal"):
        return explicit

    text = " ".join(filter(None, [
        flag.get("title", ""),
        flag.get("summation", ""),
        flag.get("implication", ""),
    ])).lower()

    for kw_group in STOP_KEYWORDS:
        for kw in kw_group:
            try:
                if _re.search(kw, text):
                    return "stop"
            except _re.error:
                if kw in text:
                    return "stop"

    for kw_group in CAP_KEYWORDS:
        for kw in kw_group:
            try:
                if _re.search(kw, text):
                    return "cap"
            except _re.error:
                if kw in text:
                    return "cap"

    return "normal"


def _check_flag_behaviours(flags: list[dict]) -> tuple[str, list[str]]:
    """
    Scan all flags. Return (worst_behaviour, triggered_reasons).
    Precedence: stop > cap > normal.
    """
    stop_reasons: list[str] = []
    cap_reasons:  list[str] = []

    for f in flags:
        beh   = _resolve_flag_behaviour(f)
        title = f.get("title") or "Unknown flag"
        if beh == "stop":
            stop_reasons.append(title)
        elif beh == "cap":
            cap_reasons.append(title)

    if stop_reasons:
        return ("stop", stop_reasons)
    if cap_reasons:
        return ("cap", cap_reasons)
    return ("normal", [])


def _identify_defect_category(flag: dict) -> str:
    """Keyword-match flag to D_max category."""
    text = " ".join(filter(None, [
        flag.get("title", ""),
        flag.get("summation", ""),
        flag.get("implication", ""),
    ])).lower()

    for keywords, category in D_MAX_KEYWORDS:
        for kw in keywords:
            try:
                if _re.search(kw, text):
                    return category
            except _re.error:
                if kw in text:
                    return category

    # Fall back to severity-based default
    sev = (flag.get("severity") or "note").lower()
    return {
        "critical": "standard_critical",
        "high":     "standard_high",
        "missing":  "standard_missing",
        "note":     "standard_note",
    }.get(sev, "standard_note")


def _d_max_for_flag(flag: dict) -> float:
    """Return D_max for the dominant/worst flag."""
    category = _identify_defect_category(flag)
    low, mid, high = D_MAX_TABLE.get(category, (0.05, 0.075, 0.10))
    sev = (flag.get("severity") or "note").lower()
    if sev == "critical":
        return high
    if sev == "high":
        return mid
    return low


def _structural_discount(critical_count: int) -> float:
    """
    D_structural: auction-specific premium, always present.
    Driven by flag severity count (critical flag count as liquidity proxy).
    8%  — clean / high liquidity  (0–2 critical)
    12% — normal regional         (3–7 critical)
    16% — illiquid / high uncertainty (8+ critical)
    """
    if critical_count <= 2:
        return 0.08
    if critical_count <= 7:
        return 0.12
    return 0.16


def _asset_discount(
    flags: list[dict],
) -> tuple[float, float, float, dict]:
    """
    D_asset = D_max + 0.04 × (1 − e^(−0.8 × N_secondary))
    Capped at 0.40.
    Returns (d_asset, d_max, n_secondary, detail_dict).
    """
    if not flags:
        return (0.0, 0.0, 0, {})

    # Score every flag
    scored = sorted(
        [{"flag": f, "d_max": _d_max_for_flag(f),
          "category": _identify_defect_category(f)} for f in flags],
        key=lambda x: x["d_max"], reverse=True
    )

    dominant = scored[0]
    d_max    = dominant["d_max"]

    # Secondary: meaningful defects (critical/high/missing only)
    # Exclude duplicates of same root category — count each category once
    seen_cats = {dominant["category"]}
    secondary_count = 0
    for s in scored[1:]:
        sev = (s["flag"].get("severity") or "note").lower()
        if sev not in ("critical", "high", "missing"):
            continue
        if s["category"] in seen_cats:
            continue
        seen_cats.add(s["category"])
        secondary_count += 1
        if secondary_count >= 5:  # cap at 5
            break

    # Exponential decay — saturates quickly
    interaction = 0.04 * (1 - math.exp(-0.8 * secondary_count))
    d_asset = min(d_max + interaction, 0.40)

    detail = {
        "dominant_flag":    dominant["flag"].get("title", ""),
        "dominant_category":dominant["category"],
        "d_max":            round(d_max, 4),
        "n_secondary":      secondary_count,
        "interaction_pct":  round(interaction * 100, 2),
    }

    return (round(d_asset, 4), d_max, secondary_count, detail)


def _total_discount_multiplicative(
    d_structural: float,
    d_asset: float,
) -> float:
    """
    MULTIPLICATIVE combination — non-negotiable.
    D_total = 1 − ((1 − D_structural) × (1 − D_asset))
    Hard cap: 52%
    """
    d_total = 1 - ((1 - d_structural) * (1 - d_asset))
    return round(min(d_total, 0.52), 4)


# =============================================================================
# PART 3 — CEILING OUTPUT
# =============================================================================

def _dynamic_width(
    resolved_pct: float,
    critical_unresolved: int,
) -> float:
    """
    Width starts at ±12% (high uncertainty) and narrows to ±4%.
    resolved_pct: 0.0 → 1.0
    Blend: linear interpolation + critical flag adjustment.
    """
    base_width = 0.12 - (resolved_pct * 0.08)   # 12% → 4%
    # Extra tightening for low critical count
    critical_penalty = min(critical_unresolved * 0.005, 0.02)
    width = base_width + critical_penalty
    return round(max(0.04, min(width, 0.12)), 4)


# =============================================================================
# PART 5 — CONFIDENCE MODEL
# =============================================================================

def _confidence(
    variance_pct: float,
    comp_count: int,
    d_total: float,
    method: str,
) -> str:
    """
    High:   strong comps, low variance, low discount
    Medium: moderate variance or moderate discount
    Low:    weak comps or high discount
    """
    if method in ("none", "manual_review"):
        return "low"
    if comp_count >= 3 and variance_pct <= 0.08 and d_total <= 0.20:
        return "high"
    if comp_count >= 2 and variance_pct <= 0.15 and d_total <= 0.35:
        return "medium"
    return "low"


# =============================================================================
# SCOPE GUARD
# =============================================================================

OUT_OF_SCOPE_KEYWORDS = [
    "commercial", "mixed use", "mixed-use", "retail", "office",
    "industrial", "warehouse", "land only", "development site",
    "portfolio", "multi-let commercial", "hmo licence complex",
]

def _check_scope(financial_inputs: dict, legal_flags: list[dict]) -> bool:
    """Returns True if in scope (residential). False if out of scope."""
    strategy = (financial_inputs.get("strategy") or "BTL").lower()
    if any(kw in strategy for kw in ["commercial", "land", "portfolio"]):
        return False
    desc = (financial_inputs.get("property_description") or "").lower()
    if any(kw in desc for kw in OUT_OF_SCOPE_KEYWORDS):
        return False
    return True


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def calculate_ceiling(
    legal_flags: list[dict],
    financial_inputs: dict,
    base_valuation: Optional[float] = None,
    strategy: str = "BTL",
) -> dict:
    """
    RICS-aligned investment value ceiling for UK residential auction.

    Inputs:
        legal_flags:       list of flag dicts from LLM analysis
        financial_inputs:  dict with comps, area, rent, yield etc.
        base_valuation:    optional user override
        strategy:          BTL / BRRR / HMO / Flip / SA

    Output: see PART 6 JSON schema
    """
    legal_flags      = legal_flags      if isinstance(legal_flags, list)      else []
    financial_inputs = financial_inputs if isinstance(financial_inputs, dict)  else {}

    # ── Step 1: STOP check — must run FIRST, before any valuation ────────────
    worst_behaviour, behaviour_reasons = _check_flag_behaviours(legal_flags)

    if worst_behaviour == "stop":
        return {
            "status":       "manual_review_required",
            "ceiling":      None,
            "ceiling_range":{"low": None, "high": None},
            "cap_applied":  False,
            "reason":       behaviour_reasons[0] if behaviour_reasons else "Hard-stop flag identified",
            "stop_flags":   behaviour_reasons,
            "investment_value_note": (
                "This property contains one or more risks that cannot be reliably priced by this engine. "
                "A ceiling cannot be calculated. Independent surveyor review is required before bidding."
            ),
        }

    # ── Scope check ──────────────────────────────────────────────────────────
    if not _check_scope(financial_inputs, legal_flags):
        return {
            "error": "manual_review_required",
            "reason": "Property type outside engine scope (residential only)",
            "investment_value_note": (
                "This engine covers standard UK residential auction lots only. "
                "Commercial, mixed-use, land, and portfolio lots require manual RICS valuation."
            ),
        }

    # ── Part 1: Market Value ─────────────────────────────────────────────────
    region       = financial_inputs.get("region", "default") or "default"
    subject_area = float(financial_inputs.get("floor_area_sqm", 0) or 0)
    comparables  = financial_inputs.get("comparables", []) or []
    fallback_avg = (
        financial_inputs.get("comps_avg_value")
        or financial_inputs.get("avg_sold_price")
    )

    if base_valuation and float(base_valuation) > 5_000:
        market_value  = float(base_valuation)
        mv_method     = "external_valuation"
        mv_variance   = 0.05
        adjusted_comps = []
    elif subject_area > 0 and comparables:
        market_value, mv_method, mv_variance = _derive_market_value(
            comparables, subject_area, region, fallback_avg
        )
        adjusted_comps = [
            _adjust_comp(c, subject_area, region)
            for c in comparables
            if _adjust_comp(c, subject_area, region) is not None
        ]
    elif fallback_avg and float(fallback_avg) > 5_000:
        market_value  = float(fallback_avg)
        mv_method     = "comps_avg_fallback"
        mv_variance   = 0.12
        adjusted_comps = []
    else:
        return {
            "error":                  "no_base_valuation",
            "ceiling_range":          {"low": None, "high": None},
            "investment_value_note":  (
                "No comparable sales data available. "
                "Provide area_json with sold prices to calculate ceiling."
            ),
        }

    if market_value <= 0:
        return {
            "error":                 "no_base_valuation",
            "ceiling_range":         {"low": None, "high": None},
            "investment_value_note": "Insufficient comparable evidence.",
        }

    # ── Part 2: Risk Engine ──────────────────────────────────────────────────
    critical_count = sum(
        1 for f in legal_flags
        if (f.get("severity") or "").lower() == "critical"
    )

    d_structural = _structural_discount(critical_count)
    d_asset, d_max, n_secondary, asset_detail = _asset_discount(legal_flags)
    d_total = _total_discount_multiplicative(d_structural, d_asset)

    # ── Part 3: Ceiling Output ───────────────────────────────────────────────
    total_actionable = sum(
        1 for f in legal_flags
        if (f.get("severity") or "").lower() in ("critical", "high", "missing")
    )
    resolved_stored = financial_inputs.get("resolved_count", 0) or 0
    resolved_pct    = (
        resolved_stored / total_actionable
        if total_actionable > 0 else 0.0
    )

    critical_unresolved = max(0, critical_count - int(
        resolved_stored * (critical_count / max(total_actionable, 1))
    ))

    investment_value = round(market_value * (1 - d_total))

    # ── Step 3: CAP logic — applied after normal calculation ─────────────────
    cap_applied  = False
    cap_reasons_out: list[str] = []
    if worst_behaviour == "cap":
        cap_ceiling = round(market_value * CAP_CEILING_PCT)
        if investment_value > cap_ceiling:
            investment_value = cap_ceiling
            cap_applied      = True
            cap_reasons_out  = behaviour_reasons

    width = _dynamic_width(resolved_pct, critical_unresolved)
    # Widen range when cap is active — more uncertainty
    if cap_applied:
        width = min(width + 0.03, 0.12)

    range_low  = round(investment_value * (1 - width) / 500) * 500
    range_high = round(investment_value * (1 + width) / 500) * 500

    # ── Part 4: Value Secured ────────────────────────────────────────────────
    # Calculated against stored initial ceiling high if available
    initial_high = financial_inputs.get("initial_ceiling_high", 0) or 0
    value_secured = max(0, range_high - int(initial_high)) if initial_high > 0 else 0

    # ── Part 5: Confidence ───────────────────────────────────────────────────
    comp_count = len([c for c in comparables if c]) if comparables else 0
    confidence = _confidence(mv_variance, comp_count, d_total, mv_method)

    # ── Part 6: JSON Output ──────────────────────────────────────────────────
    return {
        # Status
        "status":                     "ok",
        "cap_applied":                cap_applied,
        "cap_reasons":                cap_reasons_out,

        # Primary output
        "ceiling_range":              {"low": int(range_low), "high": int(range_high)},
        "investment_value_midpoint":  investment_value,
        "width":                      width,
        "value_secured":              value_secured,
        "flags_resolved_pct":         round(resolved_pct, 3),
        "confidence":                 confidence,

        # Valuation layer
        "base_valuation":             int(market_value),
        "market_value":               int(market_value),
        "base_method":                mv_method,
        "market_value_variance":      mv_variance,

        # Risk layer
        "d_structural":               d_structural,
        "d_asset":                    round(d_asset, 4),
        "total_discount":             d_total,
        "risk_discount_pct":          round(d_total * 100, 1),
        "raw_d_asset_detail":         asset_detail,
        "cap_active":                 (d_total >= 0.52),

        # Preserved fields for frontend compatibility
        "risk_components": {
            "structural":  round(d_structural * 100, 1),
            "asset":       round(d_asset * 100, 1),
        },
        "midpoint":                   investment_value,
        "money_saved":                max(0, int(market_value) - investment_value),

        # Explanation (RICS-labelled)
        "explanation": {
            "valuation": (
                f"Market Value derived via {mv_method.replace('_', ' ')}. "
                f"RICS Market Approach (comparable sales). "
                f"Variance: {round(mv_variance * 100, 1)}%."
            ),
            "risk": (
                f"Structural auction discount: {round(d_structural * 100, 1)}% "
                f"({'clean' if d_structural == 0.08 else 'normal' if d_structural == 0.12 else 'illiquid'}). "
                f"Asset risk: {round(d_asset * 100, 1)}% "
                f"(dominant: {asset_detail.get('dominant_flag', 'n/a')[:60]}, "
                f"D_max {round(asset_detail.get('d_max', 0) * 100, 1)}%, "
                f"{n_secondary} secondary defects). "
                f"Combined (multiplicative): {round(d_total * 100, 1)}%."
            ),
            "range": (
                f"±{round(width * 100, 1)}% uncertainty band. "
                f"Narrows as flags resolved. "
                f"Currently {round(resolved_pct * 100)}% resolved."
            ),
        },

        # Strategy and scope
        "strategy_used":              strategy,
        "investment_value_note": (
            f"Investment Value ceiling for {strategy} strategy. "
            f"Base: RICS Market Value (adjusted comps). "
            f"Not a formal RICS valuation. Decision-support only. "
            f"LegalSmegal Technologies Ltd is not FCA-regulated."
        ),

        # Legacy compat
        "gross_ceiling_range":        {"low": int(range_low), "high": int(range_high)},
        "drivers":                    _build_drivers(legal_flags),
        "high_impact_flags":          [asset_detail.get("dominant_category", "")],
        "acquisition_costs":          None,  # NOT deducted — informational only
    }


def _build_drivers(flags: list[dict]) -> list[dict]:
    """Build driver list for frontend risk breakdown panel."""
    scored = sorted(
        [{
            "flag":       f.get("title", "Unknown"),
            "severity":   (f.get("severity") or "note").lower(),
            "category":   _identify_defect_category(f),
            "impact_pct": round(_d_max_for_flag(f) * 100, 1),
            "high_impact": _identify_defect_category(f) not in (
                "standard_critical", "standard_high",
                "standard_missing", "standard_note"
            ),
        } for f in flags],
        key=lambda x: x["impact_pct"], reverse=True
    )
    return scored
