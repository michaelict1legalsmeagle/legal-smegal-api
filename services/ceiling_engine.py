"""
ceiling_engine.py — LegalSmegal Bid Ceiling Engine v1.1
========================================================
Produces INVESTMENT VALUE ceiling range for UK residential BTL/HMO auction.

CRITICAL DISTINCTION
--------------------
Investment Value ≠ Market Value.

Market Value (RICS): price achievable in an arm's length open-market transaction.
Investment Value (RICS): value to a specific investor for their individual objectives.

This engine answers: "What is the maximum price at which this property still meets
my investment objectives, given the legal risks identified?"

IMPORTANT: Decision-support tooling only. Not financial advice.
LegalSmegal Technologies Ltd is not FCA-regulated.

Architecture
------------
BASE VALUATION ROUTING (comps primary — see decision table below):
  BTL / BRRR / SA  → comps_value primary; yield is cross-check only, not a cap
  HMO conversion   → max(comps_value, residual_gdv); yield cross-check only
  Flip             → comps_value only (no yield anchor)

RISK DISCOUNT:
  Each flag → base_discount × strategy_multiplier × high_impact_boost
  Total capped at 38% (standard) or 45% (15+ flags — heavy pack)

CEILING RANGE:
  gross_mid = base × (1 − total_discount)
  gross_range = gross_mid ± 5%
  net_range = gross_range − acquisition_costs (SDLT + premium + legal + bridging)

CONFIDENCE:
  f(discount_magnitude, data_quality, high_impact_flag_count)

v2 hooks (marked in code):
  - Calibration table → Supabase (update monthly from outcome data)
  - LLM provides suggested_discount_pct per flag
  - Outcome feedback: bid, hammer, actual_costs → retrain calibration
"""

from __future__ import annotations
import re as _re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# CALIBRATION TABLE v1 — HEURISTIC STARTING VALUES
# =============================================================================
# These are NOT empirically calibrated. They are practitioner benchmarks.
# DO NOT tune by intuition alone.
#
# v2 migration:
#   CREATE TABLE ceiling_calibration(severity TEXT, base_discount FLOAT,
#     updated_at TIMESTAMPTZ, outcome_count INT);
#   Load at startup; update monthly from user outcome feedback.
# =============================================================================

DISCOUNT_CALIBRATION: dict[str, float] = {
    "critical": 0.09,
    "high":     0.05,
    "missing":  0.04,
    "note":     0.015,
}

MAX_TOTAL_DISCOUNT      = 0.38  # standard cap
MAX_TOTAL_DISCOUNT_HEAVY = 0.45  # heavy pack cap (15+ unresolved flags)
HIGH_IMPACT_CONFIDENCE_PENALTY = 0.04
BAND_PCT = 0.05  # ±5% ceiling band


# =============================================================================
# MISSING DOCUMENT PACKAGE PENALTY (escalating)
# =============================================================================

MISSING_DOC_PENALTY: dict[int, float] = {
    0: 0.000,
    1: 0.010,
    2: 0.025,
    3: 0.040,
    5: 0.060,
}


# =============================================================================
# STRATEGY MULTIPLIERS
# =============================================================================
# Scale base discount per risk category per strategy.
# > 1.0 = this risk matters MORE for this strategy.
# < 1.0 = matters less but never zero.
#
# BASE VALUATION ROUTING DECISION TABLE:
#  Strategy | Formula                              | Rationale
#  ---------|--------------------------------------|------------------------
#  BTL      | comps primary, yield cross-check     | Comps = Land Registry reality
#  BRRR     | comps primary, yield cross-check     | Must refinance = comps matter
#  SA       | comps primary, yield cross-check     | SA yields operator-dependent
#  HMO      | max(comps, residual), yield check    | Conversion adds value
#  Flip     | comps only                           | GDV-based, no yield anchor
# =============================================================================

STRATEGY_MULTIPLIERS: dict[str, dict[str, float]] = {
    "BTL": {
        "lease": 1.40, "mees": 1.50, "title": 1.00, "planning": 0.80,
        "covenant": 1.10, "financial": 1.20, "structural": 1.00,
        "occupancy": 1.30, "default": 1.00,
    },
    "HMO": {
        "lease": 0.90, "mees": 0.70, "title": 1.00, "planning": 1.80,
        "covenant": 1.70, "financial": 1.00, "structural": 1.20,
        "occupancy": 0.90, "default": 1.00,
    },
    "Flip": {
        "lease": 0.70, "mees": 0.50, "title": 1.70, "planning": 1.30,
        "covenant": 1.10, "financial": 0.80, "structural": 1.50,
        "occupancy": 0.60, "default": 1.00,
    },
    "BRRR": {
        "lease": 1.20, "mees": 0.70, "title": 1.60, "planning": 1.20,
        "covenant": 1.20, "financial": 1.00, "structural": 1.30,
        "occupancy": 0.80, "default": 1.00,
    },
    "Serviced Accommodation": {
        "lease": 1.20, "mees": 0.70, "title": 1.00, "planning": 2.00,
        "covenant": 1.90, "financial": 1.10, "structural": 0.90,
        "occupancy": 1.00, "default": 1.00,
    },
}

DEFAULT_STRATEGY = "BTL"

STRATEGY_ALIASES = {
    "btl": "BTL", "hmo": "HMO", "flip": "Flip", "brrr": "BRRR",
    "sa": "Serviced Accommodation", "serviced": "Serviced Accommodation",
    "serviced accommodation": "Serviced Accommodation", "development": "HMO",
}


# =============================================================================
# HIGH-IMPACT FLAGS
# =============================================================================
# Deal-terminating or unusually high-magnitude conditions that warrant
# additional discount beyond the standard calibration table.
# (keywords, extra_discount, label, strategies_affected_or_None)
# =============================================================================

HIGH_IMPACT_FLAGS: list[tuple[list[str], float, str, Optional[list[str]]]] = [
    (["short lease", r"lease.{0,20}less than 80", r"lease.{0,20}under 80",
      r"lease.{0,10}7\d year", r"lease.{0,10}6\d year", "enfranchisement"],
     0.08, "short_lease", None),
    (["regulated tenancy", "security of tenure", "rent act", "protected tenant",
      "sitting tenant", "long residential occupier"],
     0.07, "security_of_tenure", None),
    (["possessory title", "possessory freehold", "possessory leasehold"],
     0.07, "possessory_title", None),
    (["article 4", "article 4 direction", r"sui.{0,5}generis", "full planning.*hmo"],
     0.10, "article4_hmo", ["HMO"]),
    ([r"covenant.{0,30}occupation", r"covenant.{0,30}multiple", "no hmo",
      "single dwelling", "houses in multiple"],
     0.10, "covenant_blocks_hmo", ["HMO", "Serviced Accommodation"]),
    (["flood zone 3", "high flood risk"],
     0.05, "flood_zone_3", None),
    (["chancel repair", "chancel liability"],
     0.03, "chancel_repair", None),
    (["japanese knotweed", "knotweed", "invasive plant"],
     0.06, "knotweed", None),
    ([r"mortgage charge.{0,30}(bank|lloyds|natwest|hsbc|barclays|nationwide|halifax)",
      "registered charge.*not discharged", "undischarged charge"],
     0.04, "undischarged_charge", None),
    (["possessory", "unregistered title", "title not registered"],
     0.05, "unregistered_title", None),
]


def _detect_high_impact(flag: dict, strategy: str) -> tuple[float, list[str]]:
    """Return (extra_discount, matched_labels) for high-impact patterns."""
    text = " ".join(filter(None, [
        flag.get("title", ""),
        flag.get("summation", ""),
        flag.get("implication", ""),
        flag.get("action", ""),
    ])).lower()

    extra = 0.0
    labels = []
    for keywords, discount, label, strategies in HIGH_IMPACT_FLAGS:
        if strategies and strategy not in strategies:
            continue
        for kw in keywords:
            try:
                if _re.search(kw, text):
                    extra += discount
                    labels.append(label)
                    break
            except _re.error:
                if kw in text:
                    extra += discount
                    labels.append(label)
                    break
    return round(extra, 4), labels


# =============================================================================
# RISK CATEGORY MAPPING
# =============================================================================

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "lease":     ["lease", "leasehold", "ground rent", "service charge", "lessee",
                  "enfranchisement", "extension", "forfeiture"],
    "mees":      ["epc", "energy performance", "mees", "minimum energy",
                  "band f", "band g", "rating f", "rating g"],
    "title":     ["title", "possessory", "charges register", "mortgage charge",
                  "registered charge", "restriction", "title guarantee", "unregistered"],
    "planning":  ["planning", "permission", "consent", "building regulation",
                  "article 4", "hmo licence", "change of use", "enforcement",
                  "permitted development", "sui generis"],
    "covenant":  ["covenant", "restriction on use", "restrictive",
                  "positive covenant", "no hmo", "single dwelling"],
    "financial": ["buyer's premium", "buyers premium", "administration fee",
                  "vat", "non-refundable", "deposit", "service charge arrears",
                  "ground rent arrears", "section 20"],
    "structural": ["structural", "subsidence", "knotweed", "drainage", "damp",
                   "asbestos", "mining", "chancel", "flood", "foundation"],
    "occupancy": ["sitting tenant", "occupier", "vacant possession", "tenancy",
                  "regulated tenancy", "squatter", "security of tenure"],
}


def _classify_flag(flag: dict) -> str:
    """Classify flag into risk category. v2: use LLM-provided risk_category."""
    if flag.get("risk_category"):
        return str(flag["risk_category"]).lower()
    text = " ".join(filter(None, [
        flag.get("title", ""), flag.get("summation", ""), flag.get("implication", "")
    ])).lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return cat
    return "default"


# =============================================================================
# RESOLUTION COSTS — downside scenario generation
# (min_cost, max_cost, description) — UK 2024/25 benchmarks
# =============================================================================

RESOLUTION_COSTS: dict[str, tuple[int, int, str]] = {
    "lease":      (15_000, 45_000, "lease extension"),
    "mees":       (3_000,  18_000, "EPC upgrade works"),
    "title":      (500,    8_000,  "title indemnity insurance"),
    "planning":   (2_000,  20_000, "planning regularisation / indemnity"),
    "covenant":   (1_000,  10_000, "covenant indemnity insurance"),
    "financial":  (2_000,  30_000, "arrears / premium / VAT settlement"),
    "structural": (5_000,  50_000, "structural remediation"),
    "occupancy":  (3_000,  20_000, "vacant possession / legal proceedings"),
}


# =============================================================================
# ACQUISITION COSTS — deducted from ceiling (certain costs, not risks)
# =============================================================================

BUYERS_PREMIUM_DEFAULT = 2_340
LEGAL_FEES_DEFAULT      = 2_000
BRIDGING_ESTIMATE       = 2_500


def _sdlt(price: float) -> int:
    """SDLT additional dwelling, England, post-Oct 2024 Budget."""
    if price <= 0:       return 0
    if price <= 250_000: return round(price * 0.05)
    if price <= 925_000: return round(250_000 * 0.05 + (price - 250_000) * 0.10)
    if price <= 1_500_000:
        return round(250_000 * 0.05 + 675_000 * 0.10 + (price - 925_000) * 0.12)
    return round(250_000 * 0.05 + 675_000 * 0.10 + 575_000 * 0.12
                 + (price - 1_500_000) * 0.14)


def _acq_costs(mid: float, fins: dict) -> dict:
    bp   = int(float(fins.get("buyers_premium", BUYERS_PREMIUM_DEFAULT)))
    lf   = int(float(fins.get("legal_fees",     LEGAL_FEES_DEFAULT)))
    br   = int(float(fins.get("bridging_estimate", BRIDGING_ESTIMATE)))
    sdlt = _sdlt(mid)
    return {"sdlt": sdlt, "buyers_premium": bp, "legal_fees": lf,
            "bridging": br, "total": sdlt + bp + lf + br}


# =============================================================================
# BASE VALUATION — strategy-explicit routing
# =============================================================================

def _yield_ceiling(fins: dict, strategy: str) -> Optional[float]:
    rent = fins.get("monthly_rent") or fins.get("estimated_monthly_rent")
    yld  = fins.get("target_gross_yield") or fins.get("target_yield")
    if not yld:
        yld = {"BTL": 0.07, "HMO": 0.09, "BRRR": 0.075,
               "Serviced Accommodation": 0.10, "Flip": None}.get(strategy)
    if not rent or not yld or float(yld) <= 0:
        return None
    return round(float(rent) * 12 / float(yld))


def _base_valuation(fins: dict, strategy: str,
                    override: Optional[float]) -> tuple[float, str]:
    """
    Strategy routing — COMPS PRIMARY (updated per product brief):

    Comps (Land Registry sold prices) are the non-negotiable primary anchor.
    Yield ceiling is a cross-check only — it warns if the rental maths don't
    support the comps value, but it does NOT cap or replace comps.

    Routing:
      External override → override value (user knows the area)
      Flip              → comps only (no yield anchor)
      HMO               → comps primary, residual as uplift check
      BTL / BRRR / SA   → comps primary, yield as secondary cross-check only

    Yield ceiling below comps does NOT reduce the base. It is surfaced as a
    warning in the confidence score and investment_value_note instead.

    Returns (value, method_label).
    """
    if override and float(override) > 5_000:
        return (float(override), "external_valuation")

    comps    = fins.get("comps_avg_value") or fins.get("avg_sold_price")
    comps    = float(comps) if comps and float(comps) > 5_000 else None
    yc       = _yield_ceiling(fins, strategy)
    residual = fins.get("residual_gdv")
    residual = float(residual) if residual and float(residual) > 5_000 else None

    if strategy == "Flip":
        # Flip: always comps-led, yield meaningless
        if comps: return (comps, "comps")
        if yc:    return (yc,    "yield_fallback")
        return (0.0, "none")

    if strategy == "HMO":
        # HMO: comps or residual (whichever higher), yield as cross-check
        if comps and residual: return (max(comps, residual), "comps_or_residual")
        if comps:              return (comps, "comps")
        if residual:           return (residual, "residual")
        if yc:                 return (yc, "yield_fallback")
        return (0.0, "none")

    # BTL / BRRR / SA — COMPS PRIMARY
    # If comps available, use them. Yield ceiling is a cross-check surfaced
    # in confidence/notes only — it does NOT reduce the base valuation.
    if comps: return (comps, "comps")
    if yc:    return (yc,    "yield")       # fallback when no comps
    return (0.0, "none")


# =============================================================================
# DOWNSIDE SCENARIOS — deterministic
# =============================================================================

PRIORITY_SCENARIO_MAP: dict[str, tuple[str, str]] = {
    "short_lease":        ("lease",      "Lease extension"),
    "security_of_tenure": ("occupancy",  "Vacant possession proceedings"),
    "possessory_title":   ("title",      "Title indemnity / re-registration"),
    "article4_hmo":       ("planning",   "Full planning application for HMO"),
    "covenant_blocks_hmo":("covenant",   "Covenant modification / insurance"),
    "knotweed":           ("structural", "Japanese knotweed remediation"),
    "flood_zone_3":       ("structural", "Flood insurance uplift"),
}


def _scenarios(cats: list[str], hi_labels: list[str], base: float) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    for lbl in hi_labels:
        if len(out) >= 3: break
        if lbl not in PRIORITY_SCENARIO_MAP: continue
        cat, desc = PRIORITY_SCENARIO_MAP[lbl]
        if cat in seen or cat not in RESOLUTION_COSTS: continue
        seen.add(cat)
        lo, hi, _ = RESOLUTION_COSTS[cat]
        pct_lo = round(lo / base * 100, 1) if base else 0
        pct_hi = round(hi / base * 100, 1) if base else 0
        out.append(f"{desc}: £{lo:,}–£{hi:,} (~{pct_lo}–{pct_hi}% of base) "
                   f"— deduct and verify before bidding")

    for cat in cats:
        if len(out) >= 3: break
        if cat in seen or cat not in RESOLUTION_COSTS: continue
        seen.add(cat)
        lo, hi, lbl = RESOLUTION_COSTS[cat]
        pct = round(lo / base * 100, 1) if base else 0
        out.append(f"If {lbl} costs £{lo:,}–£{hi:,} "
                   f"→ ceiling reduces ~{pct}%+ before bidding")

    if not out:
        out.append("Standard auction risks apply: 28-day completion, "
                   "non-refundable deposit.")
    return out


# =============================================================================
# INVESTMENT VALUE NOTE
# =============================================================================

def _iv_note(strategy: str, fins: dict) -> str:
    yld  = fins.get("target_gross_yield") or fins.get("target_yield")
    rent = fins.get("monthly_rent") or fins.get("estimated_monthly_rent")
    yld_s  = f"{float(yld)*100:.1f}%" if yld else "your target"
    rent_s = f"£{int(float(rent)):,}/mo" if rent else "estimated rent"

    note = (
        f"Investment value for {strategy} at {yld_s} gross yield / {rent_s}. "
        f"Base anchored to Land Registry comparable sales. "
        f"Not market value — decision-support only, not financial advice."
    )

    # Cross-check warning: if yield ceiling is materially below comps,
    # flag it as a yield warning rather than silently capping the base.
    comps = fins.get("comps_avg_value") or fins.get("avg_sold_price")
    if comps and rent and yld:
        try:
            yc = float(rent) * 12 / float(yld)
            comps_f = float(comps)
            if yc < comps_f * 0.75:
                gap_pct = round((1 - yc / comps_f) * 100)
                note += (
                    f" ⚠ Yield cross-check: at {yld_s} gross, rental maths support "
                    f"~£{int(yc):,} — {gap_pct}% below comps. "
                    f"Verify rent assumptions or adjust yield target."
                )
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    return note


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
    Calculate bid ceiling range (INVESTMENT VALUE) for UK residential
    BTL/HMO auction property.

    See module docstring for full architecture and output contract.
    """
    # Normalise
    legal_flags      = legal_flags if isinstance(legal_flags, list) else []
    financial_inputs = financial_inputs if isinstance(financial_inputs, dict) else {}
    strategy = STRATEGY_ALIASES.get((strategy or DEFAULT_STRATEGY).lower().strip(),
                                     strategy or DEFAULT_STRATEGY)
    if strategy not in STRATEGY_MULTIPLIERS:
        logger.warning(f"[ceiling] Unknown strategy '{strategy}' → BTL")
        strategy = DEFAULT_STRATEGY
    mults = STRATEGY_MULTIPLIERS[strategy]

    # Step 1 — base
    base, base_method = _base_valuation(financial_inputs, strategy, base_valuation)

    no_data = {
        "ceiling_range": {"low": None, "high": None},
        "gross_ceiling_range": {"low": None, "high": None},
        "confidence": 0.0, "base_valuation": None, "base_method": "none",
        "risk_discount_pct": None, "missing_doc_penalty_pct": None,
        "drivers": [], "high_impact_flags": [],
        "downside_scenarios": [
            "Insufficient data — provide estimated monthly rent or comparable "
            "sales value to calculate ceiling"
        ],
        "strategy_used": strategy, "acquisition_costs": None,
        "investment_value_note": _iv_note(strategy, financial_inputs),
        "error": "no_base_valuation",
    }
    if base <= 0:
        return no_data

    # ─────────────────────────────────────────────────────────────────────────
    # LOCKED FORMULA IMPLEMENTATION
    # Market Value  = base (comps-anchored, adjusted)              [Formula 1]
    # Total Risk%   = Legal + Condition + Liquidity + Finance      [Formula 2]
    # Investment IV = Market Value × (1 − Total Risk%)             [Formula 2]
    # Spread        = BaseSpread + RiskUncertaintyFactor           [Formula 3]
    # Lower         = IV × (1 − Spread)                           [Formula 3]
    # Upper         = IV × (1 + Spread)                           [Formula 3]
    # Midpoint      = (Lower + Upper) / 2                         [Formula 4]
    # Money Saved   = Market Value − Midpoint                     [Formula 5]
    # ─────────────────────────────────────────────────────────────────────────

    # ── Formula 1: Market Value (already derived above as `base`) ────────────
    market_value = base  # comps-anchored, adjusted per _base_valuation routing

    # ── Formula 2: Total Risk% = Legal + Condition + Liquidity + Finance ─────
    # Map engine risk categories to the four formula components:
    #   Legal      → title, covenant, planning, lease, occupancy
    #   Condition  → structural, mees
    #   Liquidity  → occupancy, missing docs
    #   Finance    → financial
    LEGAL_CATS      = {"title", "covenant", "planning", "lease"}
    CONDITION_CATS  = {"structural", "mees"}
    LIQUIDITY_CATS  = {"occupancy"}
    FINANCE_CATS    = {"financial"}

    drivers: list[dict] = []
    flag_cats: list[str] = []
    hi_labels_all: list[str] = []
    missing_count = 0

    # Accumulate by formula component
    legal_disc     = 0.0
    condition_disc = 0.0
    liquidity_disc = 0.0
    finance_disc   = 0.0

    for flag in legal_flags:
        sev = (flag.get("severity") or "note").lower().strip()
        if sev not in DISCOUNT_CALIBRATION:
            sev = "note"

        cat  = _classify_flag(flag)
        flag_cats.append(cat)

        base_d  = DISCOUNT_CALIBRATION[sev]
        strat_m = mults.get(cat, mults.get("default", 1.0))
        eff     = round(base_d * strat_m, 4)

        hi_extra, hi_lbls = _detect_high_impact(flag, strategy)
        eff += hi_extra
        hi_labels_all.extend(hi_lbls)

        if sev == "missing":
            missing_count += 1

        # Route to formula component
        if cat in LEGAL_CATS:
            legal_disc += eff
        elif cat in CONDITION_CATS:
            condition_disc += eff
        elif cat in LIQUIDITY_CATS:
            liquidity_disc += eff
        elif cat in FINANCE_CATS:
            finance_disc += eff
        else:
            legal_disc += eff  # default → legal bucket

        drivers.append({
            "flag":               flag.get("title", "Unspecified flag"),
            "severity":           sev,
            "category":           cat,
            "impact_pct":         round(eff * 100, 2),
            "high_impact":        bool(hi_lbls),
            "high_impact_labels": hi_lbls,
        })

    # Missing doc penalty → liquidity component
    miss_pen = 0.0
    for threshold in sorted(MISSING_DOC_PENALTY, reverse=True):
        if missing_count >= threshold:
            miss_pen = MISSING_DOC_PENALTY[threshold]
            break
    liquidity_disc += miss_pen

    # Total Risk% — raw (for transparency) and capped
    total_risk_raw = legal_disc + condition_disc + liquidity_disc + finance_disc
    # Dynamic cap — heavy packs (15+ flags) get 45% ceiling, standard 38%
    flag_count = len(legal_flags)
    active_cap = MAX_TOTAL_DISCOUNT_HEAVY if flag_count >= 15 else MAX_TOTAL_DISCOUNT
    total_disc = min(total_risk_raw, active_cap)

    # ── Formula 2: Investment Value = Market Value × (1 − Total Risk%) ───────
    investment_value = market_value * (1.0 - total_disc)

    # ── Formula 3: Spread = BaseSpread + RiskUncertaintyFactor ───────────────
    # BaseSpread: irreducible floor (auction day uncertainty)
    # RiskUncertaintyFactor: grows with unresolved flag count and discount magnitude
    BASE_SPREAD = 0.03  # 3% floor — irreducible auction day uncertainty
    hi_count    = len(set(hi_labels_all))
    # RiskUncertaintyFactor: scaled by discount magnitude and high-impact flags
    risk_uncertainty_factor = round(
        min(0.12, total_disc * 0.20 + hi_count * 0.01), 4
    )
    spread = BASE_SPREAD + risk_uncertainty_factor

    # Lower = IV × (1 − Spread),  Upper = IV × (1 + Spread)
    gross_low  = round(investment_value * (1 - spread) / 1_000) * 1_000
    gross_high = round(investment_value * (1 + spread) / 1_000) * 1_000

    # ── Formula 4: Midpoint = (Lower + Upper) / 2 ────────────────────────────
    gross_mid = (gross_low + gross_high) / 2

    # Acquisition costs deducted from net range
    acq      = _acq_costs(gross_mid, financial_inputs)
    net_low  = max(0, gross_low  - acq["total"])
    net_high = max(0, gross_high - acq["total"])
    midpoint = (net_low + net_high) / 2

    # ── Formula 5: Money Saved = Market Value − Midpoint ─────────────────────
    money_saved = max(0, int(market_value) - int(midpoint))

    # Confidence
    data_pen = {"none": 0.30, "yield": 0.08, "comps": 0.08,
                "yield_fallback": 0.15, "external_valuation": 0.02}.get(base_method, 0.05)
    conf = round(max(0.20, min(0.95,
        0.65 + (1.0 - min(total_disc, 0.40)) * 0.35
        - data_pen - hi_count * HIGH_IMPACT_CONFIDENCE_PENALTY
    )), 2)

    # Scenarios
    seen_c: set[str] = set()
    uniq_cats   = [c for c in flag_cats if not (c in seen_c or seen_c.add(c))]
    uniq_hi     = list(dict.fromkeys(hi_labels_all))
    scenarios   = _scenarios(uniq_cats, uniq_hi, market_value)

    drivers.sort(key=lambda d: d["impact_pct"], reverse=True)

    return {
        # Primary output
        "ceiling_range":           {"low": int(net_low),   "high": int(net_high)},
        "gross_ceiling_range":     {"low": int(gross_low), "high": int(gross_high)},
        "midpoint":                int(midpoint),
        "money_saved":             money_saved,
        # Formula components (for transparency)
        "market_value":            int(market_value),
        "investment_value":        int(investment_value),
        "spread_pct":              round(spread * 100, 1),
        "base_spread_pct":         round(BASE_SPREAD * 100, 1),
        "risk_uncertainty_pct":    round(risk_uncertainty_factor * 100, 1),
        "total_risk_pct":          round(total_disc * 100, 1),
        "raw_total_risk_pct":      round(total_risk_raw * 100, 1),
        "cap_active":              total_risk_raw > active_cap,
        "active_cap_pct":          round(active_cap * 100, 1),
        "risk_components": {
            "legal":     round(legal_disc * 100, 1),
            "condition": round(condition_disc * 100, 1),
            "liquidity": round(liquidity_disc + miss_pen * 100, 1),
            "finance":   round(finance_disc * 100, 1),
        },
        # Existing fields preserved
        "confidence":              conf,
        "base_valuation":          int(market_value),
        "base_method":             base_method,
        "risk_discount_pct":       round(total_disc * 100, 1),
        "missing_doc_penalty_pct": round(miss_pen * 100, 1),
        "drivers":                 drivers,
        "high_impact_flags":       uniq_hi,
        "downside_scenarios":      scenarios,
        "strategy_used":           strategy,
        "acquisition_costs":       acq,
        "investment_value_note":   _iv_note(strategy, financial_inputs),
    }
