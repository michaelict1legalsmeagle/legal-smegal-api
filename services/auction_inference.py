"""
services/auction_inference.py — Phase 9 LLM Inference Engine
============================================================
On-demand LLM inference for discovery listings.

Transforms structured investment_json into investor-facing inference:
  deal_label, headline, summary, market_interpretation,
  risk_flag, confidence_display

Uses existing llm_openrouter infrastructure.
On-demand only — never runs in batch enrichment.
Result cached in investment_json.llm_inference.
"""

from __future__ import annotations
import json
import logging
import os
import re
from typing import Optional

log = logging.getLogger(__name__)

# ── Phase 9 system prompt ─────────────────────────────────────────────────────

INFERENCE_SYSTEM_PROMPT = """You are the inference layer for LegalSmegal, an auction market intelligence system.

Your role is to compress structured auction evidence into investor-grade inference.

You are NOT:
- marketing listings
- persuading users
- generating sales copy
- speculating
- inventing facts or causes

You ARE:
- identifying probable pricing dynamics
- surfacing likely explanations for valuation gaps
- producing restrained commercial reasoning from structured data

CRITICAL CONSTRAINTS:
- Never invent facts, legal defects, structural issues, or planning potential unless planning data exists
- Never use emotional or estate-agent language
- Remain probabilistic at all times
- If evidence is weak, reduce confidence. If absent, say less.

OUTPUT: Return ONLY valid JSON with these exact keys:
{
  "deal_label": "",
  "headline": "",
  "summary": "",
  "market_interpretation": "",
  "risk_flag": "",
  "confidence_display": ""
}

FIELD RULES:

deal_label: Classify the dominant pattern. Use only:
POSSIBLE VALUE GAP | RENTAL YIELD ABOVE AREA | PRICING BELOW LOCAL EVIDENCE |
POSSIBLE REPOSITIONING ANGLE | MARKET PRICED | LIMITED EVIDENCE |
COMPRESSED AUCTION PRICING | POTENTIAL VALUE DISLOCATION

headline: One line, under 14 words. Evidence framing only. No sensationalism.
Good: "Guide sits materially below nearby sales evidence."
Bad: "Massive discount opportunity"

summary: 1-2 short sentences. Explain probable pricing dynamic. Remain probabilistic.
Good: "Nearby sales and rental evidence both suggest a pricing gap."
Bad: "This property is massively undervalued."

market_interpretation: Acquisitions reasoning. Concise. Evidence-linked.
Good: "Income-focused buyers may view pricing as attractive relative to local rents."
Bad: "This will fly at auction."

risk_flag: Anchor optimism with restraint. Mention verification requirements.
Good: "Condition, title, and legal pack still require verification."
Bad: "No major risks."

confidence_display: Map to evidence depth.
Format: "High confidence · 7 data sources" or "Medium confidence · 4 data sources"

TONE: Restrained. Intelligent. Commercially aware. Concise. Direct.
Use "pricing appears", "evidence suggests", "may reflect", "relative to nearby sales".
Never use "amazing", "huge", "incredible", "massive", "bargain", "opportunity"."""


# ── Context builder ───────────────────────────────────────────────────────────

def _build_context(listing: dict, inv: dict) -> str:
    """Compress listing + investment_json into structured context string for the LLM."""
    lines: list[str] = []

    gp = listing.get("guide_price")
    if gp:
        lines.append(f"Guide price: £{float(gp):,.0f}")

    if listing.get("address"):
        lines.append(f"Address: {listing['address']}")

    if listing.get("property_type"):
        lines.append(f"Property type: {listing['property_type']}")

    if listing.get("auction_date"):
        lines.append(f"Auction date: {listing['auction_date']}")

    # Comps
    comps = inv.get("comps") or {}
    if comps.get("avg_price"):
        tier = comps.get("tier") or "local"
        cnt  = comps.get("count") or "n"
        lines.append(f"Comparable sales average: £{float(comps['avg_price']):,.0f} ({cnt} sales, {tier} level)")

    # HPI
    hpi = inv.get("hpi") or {}
    if hpi.get("regional_avg_price"):
        lines.append(f"Regional HPI average: £{float(hpi['regional_avg_price']):,.0f}")
        if hpi.get("regional_yoy_pct") is not None:
            lines.append(f"HPI annual change: {hpi['regional_yoy_pct']}%")

    # Rental
    rental = inv.get("rental") or {}
    if rental.get("avg_rent_gbp"):
        lines.append(f"Regional average rent: £{float(rental['avg_rent_gbp']):,.0f}/month")
        if rental.get("latest_yoy_pct") is not None:
            lines.append(f"Rental growth: {rental['latest_yoy_pct']}% year-on-year")

    # Yield
    yld = inv.get("yield_estimate") or {}
    if yld.get("gross_yield_pct") is not None:
        lines.append(f"Estimated gross yield: {float(yld['gross_yield_pct']):.1f}%")

    # EPC
    epc = inv.get("epc") or {}
    if epc.get("rating"):
        lines.append(f"EPC rating: {epc['rating'].upper()}")

    # Ceiling
    ceiling = inv.get("ceiling") or {}
    if ceiling.get("ceiling_low") and ceiling.get("ceiling_high"):
        lines.append(
            f"Discovery bid ceiling (BTL, no legal flags): "
            f"£{int(ceiling['ceiling_low']):,}–£{int(ceiling['ceiling_high']):,}"
        )

    # Planning
    planning = inv.get("planning") or {}
    if planning.get("article_4") is True:
        ref = planning.get("article_4_ref") or "confirmed"
        lines.append(f"Article 4 direction: {ref}")
    if planning.get("conservation_area") is True:
        lines.append("Conservation area: yes")

    # Existing deterministic signals
    inference = inv.get("inference") or {}
    signals   = inference.get("signals") or []
    if signals:
        sig_ids = [s["id"] for s in signals if s.get("id")]
        if sig_ids:
            lines.append(f"Detected pricing signals: {', '.join(sig_ids)}")

    conf = inference.get("confidence")
    band = inference.get("confidence_band") or ""
    if conf is not None:
        lines.append(f"Deterministic confidence: {band} ({float(conf):.2f})")

    steps = inv.get("steps_completed") or []
    if steps:
        lines.append(f"Enrichment steps completed: {', '.join(steps)}")

    return "\n".join(lines)


# ── LLM call ─────────────────────────────────────────────────────────────────

REQUIRED_FIELDS = [
    "deal_label", "headline", "summary",
    "market_interpretation", "risk_flag", "confidence_display",
]


def run_inference(listing: dict, inv: dict) -> dict:
    """
    Call LLM with structured listing context.
    Returns parsed Phase 9 inference dict.
    Uses existing llm_openrouter infrastructure.
    """
    context = _build_context(listing, inv)

    user_msg = (
        "Structured listing data:\n\n"
        + context
        + "\n\nGenerate the inference JSON for this auction listing."
    )

    try:
        from services.llm_openrouter import _openrouter_chat, _extract_json

        messages = [
            {"role": "system", "content": INFERENCE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]

        raw = _openrouter_chat(messages, temperature=0.15)
        parsed = _extract_json(raw)

        if not isinstance(parsed, dict):
            log.error("[LLM_INFER] non-dict response: %s", str(raw)[:200])
            return {"error": "non_dict_response", "raw": str(raw)[:200]}

        # Ensure all required fields present
        for field in REQUIRED_FIELDS:
            if field not in parsed:
                parsed[field] = ""

        # Strip any extra keys beyond the contract
        return {k: parsed[k] for k in REQUIRED_FIELDS}

    except RuntimeError as e:
        # OPENROUTER_API_KEY not set — fall through to anthropic SDK
        log.warning("[LLM_INFER] OpenRouter unavailable: %s — trying Anthropic SDK", e)

    try:
        import anthropic as _anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            return {"error": "no_llm_api_key_configured"}

        client = _anthropic.Anthropic(api_key=api_key)
        resp   = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 512,
            system     = INFERENCE_SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": user_msg}],
        )
        raw = resp.content[0].text.strip()

        # Parse
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            parsed = json.loads(m.group(0)) if m else {}

        if not isinstance(parsed, dict):
            return {"error": "non_dict_response", "raw": raw[:200]}

        for field in REQUIRED_FIELDS:
            if field not in parsed:
                parsed[field] = ""

        return {k: parsed[k] for k in REQUIRED_FIELDS}

    except Exception as e:
        log.error("[LLM_INFER] LLM call failed: %s: %s", type(e).__name__, e)
        return {"error": f"{type(e).__name__}: {str(e)[:120]}"}
