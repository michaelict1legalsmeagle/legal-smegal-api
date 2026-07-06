"""S43 size-adjustment & aggregation back-test harness (2026-07-05).

Referenced by CALIBRATION_METADATA["size_adjustment_and_aggregation"]
.review_trigger. Re-run when the deal book reaches 75 deals, or before ANY
proposed change to the size adjustment or aggregation estimator — the
champion-challenger gate: a challenger ships only if it beats the champion
here, on real data, on MdAPE / PPE10 AND small-subject signed bias.

Method
------
1. INPUT: a JSON export of cached comps, one row per comp:
     {"deal": ..., "hp": hpi_adjusted_price, "a": floor_area,
      "w": _similarity_score}
   Produced from Supabase:
     SELECT d.id as deal, c->>'hpi_adjusted_price' as hp,
            c->>'floor_area' as a, c->>'_similarity_score' as w
     FROM public.deals d,
          jsonb_array_elements(d.area_json->'housing'->'soldComps') c
     WHERE d.area_json->'housing'->'soldComps' IS NOT NULL;

2. BETA CALIBRATION: within-deal fixed-effects regression
   ln(price) = alpha_deal + beta * ln(area). Within-deal demeaning controls
   for location; beta is the size elasticity (hedonic form, same family as
   the ONS UK HPI methodology).

3. LEAVE-ONE-OUT BACK-TEST: predict each sold comp from its deal's OTHER
   comps, exactly as the engine values a subject. Champion = production
   stack (linear size ratio capped [SIZE_ADJ_CAP_LO, SIZE_ADJ_CAP_HI] +
   weighted median). Challenger(s) supplied as (adjust_fn, aggregate_fn).
   Metrics: MdAPE, PPE10, PPE20 overall, plus median SIGNED bias on the
   small-subject subset (target area < 0.85x deal median area) — the class
   that looks anomalous on screen and where downward bias would live.

S43 baseline results (315 comps / 37 deals, 2026-07-05), for reference:
  beta = 0.47 (SE 0.053, CI95 [0.37, 0.58]) — a pure linear ratio
  overstates the size effect ~2x, BUT the production caps already dampen:
    champion  linear+caps+wmedian : MdAPE 13.81%  PPE10 39.7%
                                    small-subject signed bias -0.08%
    challenger pow(0.47)+wmean+br : MdAPE 14.72%  PPE10 36.8%
                                    small-subject signed bias +6.79%
  VERDICT: champion retained. A comparable valuation below every raw comp
  price when the subject is smaller than every comp is validated correct.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict

# Production cap constants — keep in sync with services/ceiling_engine.py.
SIZE_ADJ_CAP_LO = 0.80
SIZE_ADJ_CAP_HI = 1.25


def load(path: str) -> dict[str, list[dict]]:
    with open(path) as f:
        rows = json.load(f)
    by_deal: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if not r.get("hp"):
            continue
        by_deal[r["deal"]].append({
            "hp": float(r["hp"]),
            "a": float(r["a"]) if r.get("a") else None,
            "w": float(r["w"]) if r.get("w") else 1.0,
        })
    return by_deal


def calibrate_beta(by_deal) -> tuple[float, float, int]:
    xs, ys, n = [], [], 0
    for pts in by_deal.values():
        pairs = [(math.log(c["a"]), math.log(c["hp"]))
                 for c in pts if c["a"] and c["a"] > 10 and c["hp"] > 1000]
        if len(pairs) < 3:
            continue
        mx = sum(p[0] for p in pairs) / len(pairs)
        my = sum(p[1] for p in pairs) / len(pairs)
        for lx, ly in pairs:
            xs.append(lx - mx)
            ys.append(ly - my)
            n += 1
    sxx = sum(x * x for x in xs)
    beta = sum(x * y for x, y in zip(xs, ys)) / sxx
    resid = [y - beta * x for x, y in zip(xs, ys)]
    dof = n - len(by_deal) - 1
    se = math.sqrt(sum(e * e for e in resid) / dof / sxx)
    return beta, se, n


def adj_linear(sa, ca):
    if not sa or not ca:
        return 1.0
    return max(SIZE_ADJ_CAP_LO, min(SIZE_ADJ_CAP_HI, sa / ca))


def make_adj_power(beta):
    def adj(sa, ca):
        if not sa or not ca:
            return 1.0
        return max(SIZE_ADJ_CAP_LO, min(SIZE_ADJ_CAP_HI, (sa / ca) ** beta))
    return adj


def agg_wmedian(pairs):
    valid = sorted([(v, w) for v, w in pairs if w > 0 and v > 0])
    tw = sum(w for _, w in valid)
    c = 0.0
    for v, w in valid:
        c += w
        if c >= tw / 2:
            return v
    return valid[-1][0]


def agg_wmean_bracket(pairs):
    valid = [(v, w) for v, w in pairs if w > 0 and v > 0]
    tw = sum(w for _, w in valid)
    m = sum(v * w for v, w in valid) / tw
    return max(min(v for v, _ in valid), min(max(v for v, _ in valid), m))


def loo(by_deal, adj, agg, subset=None):
    signed = []
    for comps in by_deal.values():
        if len(comps) < 4:
            continue
        areas = [c["a"] for c in comps if c["a"]]
        if not areas:
            continue
        med_a = sorted(areas)[len(areas) // 2]
        for i, tgt in enumerate(comps):
            if not tgt["a"]:
                continue
            if subset and not subset(tgt["a"], med_a):
                continue
            others = comps[:i] + comps[i + 1:]
            pairs = [(o["hp"] * adj(tgt["a"], o["a"]), o["w"]) for o in others]
            signed.append((agg(pairs) - tgt["hp"]) / tgt["hp"])
    signed.sort()
    n = len(signed)
    ape = sorted(abs(e) for e in signed)
    mdape = ape[n // 2]
    med_bias = signed[n // 2]
    ppe10 = sum(1 for e in ape if e <= 0.10) / n
    ppe20 = sum(1 for e in ape if e <= 0.20) / n
    return n, mdape, ppe10, ppe20, med_bias


def main(path: str):
    by_deal = load(path)
    beta, se, n = calibrate_beta(by_deal)
    print(f"beta={beta:.4f} SE={se:.4f} n={n} "
          f"CI95=[{beta - 1.96 * se:.3f},{beta + 1.96 * se:.3f}]")
    small = lambda a, ma: a < 0.85 * ma
    variants = [
        ("champion linear+caps+wmedian", adj_linear, agg_wmedian),
        (f"challenger pow({beta:.2f})+wmean+bracket",
         make_adj_power(beta), agg_wmean_bracket),
    ]
    print(f"{'variant':<38}{'n':>5}{'MdAPE':>9}{'PPE10':>8}{'PPE20':>8}"
          f"{'small-bias':>12}")
    for name, adj, agg in variants:
        n_, md, p10, p20, _ = loo(by_deal, adj, agg)
        _, _, _, _, sb = loo(by_deal, adj, agg, subset=small)
        print(f"{name:<38}{n_:>5}{md * 100:>8.2f}%{p10 * 100:>7.1f}%"
              f"{p20 * 100:>7.1f}%{sb * 100:>+11.2f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "comps.json")
