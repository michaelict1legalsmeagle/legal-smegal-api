"""
residential_seed.py — COMM-3 (2026-10-06)

Dual pipeline: app.create_deal seeds every new deal's RESIDENTIAL Financial
Model before the pack has been read and the deal classified. When the pack
routes a deal to commercial, mixed use or unclassified, that seed does not
belong to it and is removed.

Seed values written by create_deal (GET /financials writes the same):
  * since V-NODEFAULT (24 Sep 2026): target_yield 6, ltv_pct 75;
  * before it: also finance_rate_pct 5.14, management_pct 12,
    maintenance_pct 1, legal_fees 1500, void_weeks 2, hold_years 10.
Live (5 Oct 2026): 50117369, 77042684, 4d83c8e0 carry all eight; 3ee5ee1b,
66217156 the two; 254367be had the two (removed by COMM-2 when opened).

Removed only while the model is untouched: seeded, never saved (no `ok`), no
purchase price, no fields the user has entered (`_user_fields`). Only keys
still holding the exact seed value are removed. Pure function — no I/O.
"""
from __future__ import annotations

from typing import Dict, Optional

NON_RESIDENTIAL = ("commercial", "mixed_use", "unclassified")

RESIDENTIAL_SEED: Dict[str, float] = {
    "target_yield": 6.0, "ltv_pct": 75.0, "finance_rate_pct": 5.14, "management_pct": 12.0,
    "maintenance_pct": 1.0, "legal_fees": 1500.0, "void_weeks": 2.0, "hold_years": 10.0,
}


def deal_class(deal: Dict) -> str:
    """asset_class from the analysis; deals analysed before ROUTE-1 carry it
    in deal_type only ("Commercial", "Mixed Use")."""
    cls = (((deal.get("summary_json") or {}).get("property") or {}).get("asset_class") or "").lower()
    if not cls:
        dt = (deal.get("deal_type") or "").strip().lower()
        cls = {"commercial": "commercial", "mixed use": "mixed_use", "mixed-use": "mixed_use"}.get(dt, "")
    return cls


def stripped(financials_json: Optional[Dict], asset_class: str) -> Optional[Dict]:
    """The financials_json with the residential seed removed, or None when
    nothing should change."""
    if asset_class not in NON_RESIDENTIAL:
        return None
    fins = financials_json or {}
    inp = fins.get("inputs") or {}
    if not (fins.get("_seeded") and not fins.get("ok") and not inp.get("purchase_price")
            and not inp.get("_user_fields")):
        return None
    drop = [k for k, v in RESIDENTIAL_SEED.items()
            if k in inp and isinstance(inp.get(k), (int, float)) and float(inp[k]) == v]
    if not drop:
        return None
    out = dict(fins)
    out["inputs"] = {k: v for k, v in inp.items() if k not in drop}
    return out
