"""
ceiling_engine.py — LegalSmegal Bid Ceiling Engine v1
======================================================
Turns legal risk flags + financial targets + market data into a
defendable bid ceiling RANGE for UK residential BTL/HMO auction property.

IMPORTANT: This is decision-support tooling only.
All outputs are analytical inferences — not financial advice.
LegalSmegal Technologies Ltd is not FCA-regulated.
Always engage a qualified solicitor and independent financial adviser
before bidding at auction.

Architecture
------------
Base Valuation  = strategy-appropriate market anchor
Risk Discount   = Σ weighted flag impacts (calibrated, explainable)
Ceiling Range   = Base × (1 − Total Discount) ± 5%
Confidence      = function of discount magnitude + data quality

v1: Deterministic and fully explainable.
v2 hooks: LLM suggested_discount_pct, outcome feedback retraining.
"""

from __future__ import annotations
import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION TABLE
# Edit these values as outcome data accumulates.
# v2: move to DB table with admin UI for A/B testing.
# ─────────────────────────────────────────────────────────────────────────────

# Base discount per severity (as fraction of base valuation, per flag)
# Capped later so multiple critical flags don't produce absurd totals.
DISCOUNT_CALIBRATION: dict[str, float] = {
    "critical": 0.09,   # £9k on a £100k property per critical flag
    "high":     0.05,
    "missing":  0.04,
    "note":     0.015,
}

# Hard cap on total discount — prevents >40% reduction on any deal
MAX_TOTAL_DISCOUNT = 0.38

# Missing document package penalty (applied when ≥ N missing doc flags)
MISSING_DOC_PENALTY = {
    0: 0.00,
    1: 0.01,
    2: 0.025,
    3: 0.04,   # 3+ missing docs: elevated pack quality concern
    5: 0.06,
}

# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY MULTIPLIERS
# Scale base discount per risk category depending on investment strategy.
# BTL   → lettability, MEES, lease risk weighted up
# HMO   → planning, licensing, covenants weighted up
# Flip  → title defects, mortgageability weighted up
# BRRR  → title + mortgageability (must refinance)
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_MULTIPLIERS: dict[str, dict[str, float]] = {
    "BTL": {
        "lease":        1.40,   # short lease = mortgage restriction = lower tenant pool
        "mees":         1.50,   # F/G EPC = cannot let legally
        "title":        1.00,
        "planning":     0.90,
        "covenant":     1.10,
        "financial":    1.20,   # service charge, ground rent escalation
        "structural":   1.00,
        "occupancy":    1.30,   # sitting tenant = immediate void/income risk
        "default":      1.00,
    },
    "HMO": {
        "lease":        1.00,
        "mees":         0.80,   # energy upgrade = capex cost, factored into refurb
        "title":        1.00,
        "planning":     1.80,   # Article 4, C4→SG permission critical for HMO
        "covenant":     1.60,   # restrictive covenants can kill HMO use entirely
        "financial":    1.10,
        "structural":   1.20,   # more rooms = more structural exposure
        "occupancy":    1.00,
        "default":      1.00,
    },
    "Flip": {
        "lease":        0.80,   # flips often cash-buyer, lease less critical short-term
        "mees":         0.50,   # upgrade cost = known capex, not ongoing risk
        "title":        1.60,   # title defects = unmortgageable on resale
        "planning":     1.30,
        "covenant":     1.20,
        "financial":    0.90,
        "structural":   1.40,   # structural risk = refurb overrun
        "occupancy":    0.70,
        "default":      1.00,
    },
    "BRRR": {
        "lease":        1.20,
        "mees":         0.70,
        "title":        1.50,   # must refinance — title must be clean for lender
        "planning":     1.30,
        "covenant":     1.30,
        "financial":    1.00,
        "structural":   1.30,
        "occupancy":    0.80,
        "default":      1.00,
    },
    "Serviced Accommodation": {
        "lease":        1.20,
        "mees":         0.80,
        "title":        1.00,
        "planning":     2.00,   # Change of use / short-let licensing critical
        "covenant":     1.80,   # HOA/leasehold covenants may prohibit SA
        "financial":    1.20,
        "structural":   1.00,
        "occupancy":    1.10,
        "default":      1.00,
    },
}

# Default strategy if unrecognised string passed
DEFAULT_STRATEGY = "BTL"

# ─────────────────────────────────────────────────────────────────────────────
# RISK CATEGORY MAPPING
# Maps flag titles/keywords → risk category for strategy weighting.
# v2: LLM classifier should return risk_category directly in flag dict.
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "lease": [
        "lease", "leasehold", "ground rent", "service charge", "lessee",
        "lessor", "enfranchisement", "extension", "forfeiture", "commonhold",
    ],
    "mees": [
        "epc", "energy performance", "mees", "minimum energy", "band f",
        "band g", "rating f", "rating g", "energy efficiency",
    ],
    "title": [
        "title", "possessory", "absolute title", "charges register",
        "mortgage charge", "registered charge", "restriction", "caution",
        "inhibition", "title guarantee", "limited title",
    ],
    "planning": [
        "planning", "permission", "consent", "building regulation",
        "article 4", "hmo licence", "change of use", "enforcement notice",
        "permitted development", "c4", "sui generis", "class e",
    ],
    "covenant": [
        "covenant", "restriction on use", "restrictive", "positive covenant",
        "freehold covenant", "deed of covenant",
    ],
    "financial": [
        "buyer's premium", "buyers premium", "administration fee",
        "vat", "non-refundable", "deposit", "service charge arrears",
        "ground rent arrears", "section 20",
    ],
    "structural": [
        "structural", "subsidence", "japanese knotweed", "knotweed",
        "drainage", "damp", "asbestos", "roof", "foundation",
        "mining", "chancel", "coal mining", "ground stability",
    ],
    "occupancy": [
        "sitting tenant", "occupier", "vacant possession", "tenancy",
        "ast", "assured shorthold", "regulated tenancy", "squatter",
        "unlawful occupant",
    ],
}


def _classify_flag(flag: dict) -> str:
    """
    Determine risk category for a flag using keyword matching.
    Falls back to 'default' if no match found.
    v2: replace with flag.get('risk_category') from LLM classifier.
    """
    # v2 hook: if LLM provides risk_category, use it directly
    if flag.get("risk_category"):
        return flag["risk_category"].lower()

    text = " ".join([
        (flag.get("title") or ""),
        (flag.get("summation") or ""),
        (flag.get("implication") or ""),
    ]).lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return category

    return "default"


# ─────────────────────────────────────────────────────────────────────────────
# TYPICAL RESOLUTION COSTS
# Used to generate downside scenario strings.
# Based on UK practitioner benchmarks (2024/25 data).
# ─────────────────────────────────────────────────────────────────────────────

RESOLUTION_COSTS: dict[str, tuple[int, int, str]] = {
    # (min_cost, max_cost, description)
    "lease":     (15_000, 45_000, "lease extension"),
    "mees":      (3_000,  18_000, "EPC upgrade works"),
    "title":     (500,    5_000,  "title indemnity insurance"),
    "planning":  (2_000,  15_000, "planning regularisation / indemnity"),
    "covenant":  (1_000,  8_000,  "covenant indemnity insurance"),
    "financial": (2_000,  25_000, "arrears / premium settlement"),
    "structural": (5_000, 40_000, "structural remediation"),
    "occupancy": (3_000,  15_000, "vacant possession / legal proceedings"),
}

# ─────────────────────────────────────────────────────────────────────────────
# ACQUISITION COSTS (BTL/HMO residential, England & Wales)
# These are DEDUCTED from the ceiling — they are guaranteed spend before
# the investor owns the property.
# ─────────────────────────────────────────────────────────────────────────────

SDLT_ADDITIONAL_DWELLING_BANDS = [
    # (threshold, rate) — additional dwelling surcharge (2025/26)
    (250_000, 0.05),
    (925_000, 0.10),
    (1_500_000, 0.12),
    (float("inf"), 0.14),
]
SDLT_ADDITIONAL_DWELLING_BASE = 0.05  # 5% on all up to £250k for additional dwellings

BUYERS_PREMIUM_DEFAULT = 2_340   # Bond Wolfe standard (£1,950+VAT)
LEGAL_FEES_DEFAULT      = 2_000  # conveyancing fees estimate
BRIDGING_ESTIMATE       = 2_500  # 1 month bridging at ~2% on £150k


def _calculate_sdlt(price: float) -> int:
    """
    Calculate SDLT for additional dwelling purchase (England).
    Flat 5% applies to the whole purchase price up to £250k
    for additional dwellings since 2024 Budget changes.
    """
    if price <= 0:
        return 0
    if price <= 250_000:
        return round(price * 0.05)
    elif price <= 925_000:
        return round(250_000 * 0.05 + (price - 250_000) * 0.10)
    elif price <= 1_500_000:
        return round(250_000 * 0.05 + 675_000 * 0.10 + (price - 925_000) * 0.12)
    else:
        return round(250_000 * 0.05 + 675_000 * 0.10 + 575_000 * 0.12 + (price - 1_500_000) * 0.14)


def _calculate_acquisition_costs(ceiling_mid: float, financial_inputs: dict) -> dict:
    """
    Calculate total acquisition costs at the estimated ceiling midpoint.
    Returns breakdown dict and total.
    """
    buyers_premium = financial_inputs.get("buyers_premium", BUYERS_PREMIUM_DEFAULT)
    legal_fees     = financial_inputs.get("legal_fees", LEGAL_FEES_DEFAULT)
    bridging       = financial_inputs.get("bridging_estimate", BRIDGING_ESTIMATE)
    sdlt           = _calculate_sdlt(ceiling_mid)

    total = buyers_premium + legal_fees + bridging + sdlt

    return {
        "sdlt":           sdlt,
        "buyers_premium": buyers_premium,
        "legal_fees":     legal_fees,
        "bridging":       bridging,
        "total":          total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BASE VALUATION CALCULATION
# Strategy-appropriate — avoids the max() error in the original spec.
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_yield_ceiling(financial_inputs: dict, strategy: str) -> Optional[float]:
    """
    Reverse-engineer maximum price from target yield and estimated rent.
    Formula: Annual Rent / Target Gross Yield = Max Price
    """
    monthly_rent = (
        financial_inputs.get("monthly_rent")
        or financial_inputs.get("estimated_monthly_rent")
    )
    target_yield = (
        financial_inputs.get("target_gross_yield")
        or financial_inputs.get("target_yield")
    )

    # Strategy-specific yield defaults if not provided
    if not target_yield:
        yield_defaults = {
            "BTL": 0.07,
            "HMO": 0.09,
            "Flip": None,      # Flip doesn't use yield ceiling
            "BRRR": 0.075,
            "Serviced Accommodation": 0.10,
        }
        target_yield = yield_defaults.get(strategy, 0.07)

    if not monthly_rent or not target_yield or target_yield <= 0:
        return None

    annual_rent = float(monthly_rent) * 12
    return round(annual_rent / float(target_yield))


def _calculate_base_valuation(
    financial_inputs: dict,
    strategy: str,
    base_valuation_override: Optional[float] = None,
) -> tuple[float, str]:
    """
    Determine strategy-appropriate base valuation.
    Returns (value, method_used).

    For BTL/BRRR:  min(yield_ceiling, comps_value) — never pay more than either
    For HMO:       min(yield_ceiling, max(comps_value, residual_gdv))
    For Flip:      comps_value (GDV thinking — no yield anchor)
    If only one source: use that source with lower confidence.
    """
    if base_valuation_override and base_valuation_override > 5_000:
        return (float(base_valuation_override), "provided")

    comps_value = financial_inputs.get("comps_avg_value") or financial_inputs.get("avg_sold_price")
    yield_ceil  = _calculate_yield_ceiling(financial_inputs, strategy)
    residual    = financial_inputs.get("residual_gdv")

    comps_value = float(comps_value) if comps_value and float(comps_value) > 5_000 else None
    residual    = float(residual)    if residual    and float(residual)    > 5_000 else None

    if strategy in ("Flip",):
        # Flip: use comps (GDV) as anchor
        if comps_value:
            return (comps_value, "comps")
        if yield_ceil:
            return (yield_ceil, "yield")

    elif strategy in ("HMO", "Serviced Accommodation"):
        candidates = [c for c in [yield_ceil, comps_value] if c]
        if residual:
            candidates.append(residual)
        if candidates:
            # Use minimum of yield and comps; include residual as uplift potential
            base = min(c for c in [yield_ceil, comps_value] if c) if (yield_ceil and comps_value) \
                   else (yield_ceil or comps_value)
            method = "yield+comps+residual" if (yield_ceil and comps_value and residual) \
                     else "yield+comps" if (yield_ceil and comps_value) \
                     else "yield" if yield_ceil else "comps"
            return (base, method)

    else:
        # BTL, BRRR: min(yield, comps) — never overpay on either dimension
        if yield_ceil and comps_value:
            base = min(yield_ceil, comps_value)
            return (base, "min(yield,comps)")
        if yield_ceil:
            return (yield_ceil, "yield")
        if comps_value:
            return (comps_value, "comps")

    # No data — cannot produce ceiling
    return (0.0, "none")


# ─────────────────────────────────────────────────────────────────────────────
# DOWNSIDE SCENARIOS
# Deterministic — based on flag categories present.
# Never LLM-generated for production reliability.
# ─────────────────────────────────────────────────────────────────────────────

def _generate_downside_scenarios(
    flags: list[dict],
    base_valuation: float,
    flag_categories: list[str],
) -> list[str]:
    """
    Generate 2–3 realistic what-if downside scenarios based on flags present.
    Uses RESOLUTION_COSTS table for cost estimates.
    Scenarios are conservative but realistic — not worst-case theatrical.
    """
    scenarios = []
    seen_categories: set[str] = set()

    for cat in flag_categories:
        if cat in seen_categories or cat not in RESOLUTION_COSTS:
            continue
        seen_categories.add(cat)

        lo, hi, label = RESOLUTION_COSTS[cat]
        impact_lo = round(lo / base_valuation * 100, 1) if base_valuation else 0
        impact_hi = round(hi / base_valuation * 100, 1) if base_valuation else 0

        if cat == "lease":
            scenarios.append(
                f"If lease extension costs £{lo:,}–£{hi:,} → effective ceiling "
                f"drops ~{impact_lo}–{impact_hi}% before bidding"
            )
        elif cat == "mees":
            scenarios.append(
                f"If EPC upgrade to Band E/D costs £{lo:,}–£{hi:,} → "
                f"add to acquisition budget before bidding"
            )
        elif cat == "title":
            scenarios.append(
                f"Title indemnity insurance typically £{lo:,}–£{hi:,} — "
                f"confirm lender will accept before bidding"
            )
        elif cat == "planning":
            scenarios.append(
                f"If planning regularisation costs £{lo:,}–£{hi:,} → "
                f"reduces net return by ~{impact_lo}–{impact_hi}%"
            )
        elif cat == "structural":
            scenarios.append(
                f"Structural remediation range £{lo:,}–£{hi:,} — "
                f"commission survey before bidding, not after"
            )
        elif cat == "occupancy":
            scenarios.append(
                f"Vacant possession proceedings £{lo:,}–£{hi:,} + potential "
                f"6–12 month delay to first rental income"
            )
        else:
            scenarios.append(
                f"{label.title()} costs estimated £{lo:,}–£{hi:,} — "
                f"verify and deduct from ceiling before bidding"
            )

        if len(scenarios) >= 3:
            break

    if not scenarios:
        scenarios.append(
            "No specific resolution costs identified — standard auction risk "
            "(28-day completion, non-refundable deposit) applies"
        )

    return scenarios


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def calculate_ceiling(
    legal_flags: list[dict],
    financial_inputs: dict,
    base_valuation: Optional[float] = None,
    strategy: str = "BTL",
) -> dict:
    """
    Calculate a bid ceiling range for a UK residential auction property.

    Parameters
    ----------
    legal_flags : list[dict]
        Flags from Stage 2 LLM classifier. Each flag must have:
        - severity: "critical" | "high" | "missing" | "note"
        - title: str
        Optional: risk_category (v2), suggested_discount_pct (v2)
    financial_inputs : dict
        Financial model inputs. Keys used:
        - monthly_rent / estimated_monthly_rent
        - target_gross_yield / target_yield
        - comps_avg_value / avg_sold_price
        - residual_gdv (HMO/dev only)
        - buyers_premium (default 2340)
        - legal_fees (default 2000)
        - bridging_estimate (default 2500)
    base_valuation : float, optional
        If provided, overrides internal base calculation.
        Use for when a RICS valuation or AVM figure is available.
    strategy : str
        "BTL" | "HMO" | "Flip" | "BRRR" | "Serviced Accommodation"

    Returns
    -------
    dict with keys: ceiling_range, confidence, base_valuation,
                    risk_discount_pct, drivers, downside_scenarios,
                    strategy_used, acquisition_costs
    """

    # ── Safety: normalise inputs ──────────────────────────────────────────
    if not isinstance(legal_flags, list):
        legal_flags = []
    if not isinstance(financial_inputs, dict):
        financial_inputs = {}
    strategy = strategy.strip() if strategy else DEFAULT_STRATEGY
    if strategy not in STRATEGY_MULTIPLIERS:
        logger.warning(f"[ceiling] Unknown strategy '{strategy}', defaulting to BTL")
        strategy = DEFAULT_STRATEGY

    multipliers = STRATEGY_MULTIPLIERS[strategy]

    # ── Step 1: Base valuation ─────────────────────────────────────────────
    base, base_method = _calculate_base_valuation(financial_inputs, strategy, base_valuation)

    if base <= 0:
        # Cannot produce a ceiling without any valuation anchor
        return {
            "ceiling_range":      {"low": None, "high": None},
            "confidence":         0.0,
            "base_valuation":     None,
            "base_method":        "none",
            "risk_discount_pct":  None,
            "drivers":            [],
            "downside_scenarios": ["Insufficient data to calculate ceiling — provide estimated rent or comparable sales value"],
            "strategy_used":      strategy,
            "acquisition_costs":  None,
            "error":              "no_base_valuation",
        }

    # ── Step 2: Flag impact calculation ───────────────────────────────────
    drivers: list[dict] = []
    flag_categories: list[str] = []
    total_discount = 0.0
    missing_doc_count = 0

    for flag in legal_flags:
        severity = (flag.get("severity") or "note").lower().strip()
        if severity not in DISCOUNT_CALIBRATION:
            severity = "note"

        category = _classify_flag(flag)
        flag_categories.append(category)

        # v2 hook: allow LLM to suggest a specific discount fraction
        # suggested = flag.get("suggested_discount_pct")
        # base_impact = (suggested / 100) if suggested else DISCOUNT_CALIBRATION[severity]

        base_impact = DISCOUNT_CALIBRATION[severity]
        strategy_mult = multipliers.get(category, multipliers.get("default", 1.0))
        effective_impact = round(base_impact * strategy_mult, 4)

        if severity == "missing":
            missing_doc_count += 1

        drivers.append({
            "flag":         flag.get("title", "Unspecified flag"),
            "severity":     severity,
            "category":     category,
            "impact_pct":   round(effective_impact * 100, 2),
        })
        total_discount += effective_impact

    # Missing document package penalty (escalating)
    missing_penalty = 0.0
    for threshold in sorted(MISSING_DOC_PENALTY.keys(), reverse=True):
        if missing_doc_count >= threshold:
            missing_penalty = MISSING_DOC_PENALTY[threshold]
            break
    total_discount += missing_penalty

    # Hard cap — no deal should be discounted into absurdity
    total_discount = min(total_discount, MAX_TOTAL_DISCOUNT)

    # ── Step 3: Ceiling range ─────────────────────────────────────────────
    ceiling_mid  = base * (1.0 - total_discount)
    band_pct     = 0.05   # ±5% band
    ceiling_low  = round(ceiling_mid * (1.0 - band_pct) / 1000) * 1000
    ceiling_high = round(ceiling_mid * (1.0 + band_pct) / 1000) * 1000

    # ── Step 4: Confidence score ──────────────────────────────────────────
    # Higher discount = higher uncertainty = lower confidence
    # Base confidence 0.65, rises as risk decreases
    data_quality_penalty = 0.0
    if base_method == "none":
        data_quality_penalty = 0.30
    elif base_method in ("yield", "comps"):
        data_quality_penalty = 0.08   # single anchor = less certain
    elif base_method == "provided":
        data_quality_penalty = 0.02   # external valuation = higher confidence

    raw_confidence = (
        0.65
        + (1.0 - min(total_discount, 0.40)) * 0.35
        - data_quality_penalty
    )
    confidence = round(max(0.20, min(0.95, raw_confidence)), 2)

    # ── Step 5: Acquisition costs ─────────────────────────────────────────
    acq_costs = _calculate_acquisition_costs(ceiling_mid, financial_inputs)

    # Subtract from ceiling range — investor's true max bid is ceiling minus costs
    net_low  = max(0, ceiling_low  - acq_costs["total"])
    net_high = max(0, ceiling_high - acq_costs["total"])

    # ── Step 6: Downside scenarios ────────────────────────────────────────
    # Use unique categories (preserve order, first occurrence)
    seen: set[str] = set()
    unique_cats = [c for c in flag_categories if not (c in seen or seen.add(c))]
    downside_scenarios = _generate_downside_scenarios(
        legal_flags, base, unique_cats
    )

    # ── Sort drivers by impact ────────────────────────────────────────────
    drivers.sort(key=lambda d: d["impact_pct"], reverse=True)

    # ── Build output ──────────────────────────────────────────────────────
    return {
        # Primary output — always a range, never a single number
        "ceiling_range": {
            "low":  int(net_low),
            "high": int(net_high),
        },
        # Gross ceiling before acquisition costs (for display/debug)
        "gross_ceiling_range": {
            "low":  int(ceiling_low),
            "high": int(ceiling_high),
        },
        "confidence":         confidence,
        "base_valuation":     int(base),
        "base_method":        base_method,
        "risk_discount_pct":  round(total_discount * 100, 1),
        "missing_doc_penalty_pct": round(missing_penalty * 100, 1),
        "drivers":            drivers,
        "downside_scenarios": downside_scenarios,
        "strategy_used":      strategy,
        "acquisition_costs":  acq_costs,
        # v2 hook: outcome feedback for calibration retraining
        # "outcome_feedback": {"bid": None, "hammer": None, "actual_costs": None}
    }
