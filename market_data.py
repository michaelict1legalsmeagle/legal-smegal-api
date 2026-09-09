"""
market_data — build a deal's "Current market data" block + the derived read.
SPEC v2.2: the READ fires on every ENGLISH deal from THREE Supabase-native,
verifiable signals — HPI momentum, HPI deceleration (turn signal), auction
compression. Affordability (ONS), rate-trend (BoE history) and volume (Hetzner)
ENRICH the read as extra votes when their feeds are loaded; they are never
required for it to fire. Scotland (no HPI LAD) correctly abstains.

Rules: pure builder (inject query helpers); every reader fail-safe; STATE sets
the read, TREND earns "increasingly"; DISPLAY/CONTEXT only — never the ceiling.
"""
from datetime import datetime

MIN_SIGNALS = 3
P_MOM = 1.0; DECEL = 0.5; AUC = 0.03; V_TXN = 5.0; R_FIX = 0.10; A_HI, A_LO = 1.05, 0.95
BUYER, SELLER, NEUTRAL = "buyer", "seller", "neutral"
# trend (rate-of-change) signals — these earn the word "increasingly"
TREND_KEYS = {"momentum", "deceleration", "auction", "rate_trend", "volume"}

def _v_mom(x):   return None if x is None else (BUYER if x <= -P_MOM else SELLER if x >= P_MOM else NEUTRAL)
def _v_dec(x):   return None if x is None else (BUYER if x <= -DECEL else SELLER if x >= DECEL else NEUTRAL)
def _v_auc(x):   return None if x is None else (BUYER if x <= -AUC  else SELLER if x >= AUC  else NEUTRAL)
def _v_txn(x):   return None if x is None else (BUYER if x <= -V_TXN else SELLER if x >= V_TXN else NEUTRAL)
def _v_rate(x):  return None if x is None else (BUYER if x >= R_FIX  else SELLER if x <= -R_FIX else NEUTRAL)
def _v_aff(x):   return None if x is None else (BUYER if x >= A_HI   else SELLER if x <= A_LO  else NEUTRAL)

def classify(momentum=None, deceleration=None, auction=None,
             affordability_vs_lr=None, rate_trend=None, volume=None):
    votes = {"momentum": _v_mom(momentum), "deceleration": _v_dec(deceleration),
             "auction": _v_auc(auction), "affordability": _v_aff(affordability_vs_lr),
             "rate_trend": _v_rate(rate_trend), "volume": _v_txn(volume)}
    avail = {k: v for k, v in votes.items() if v is not None}
    n = len(avail)
    if n < MIN_SIGNALS:
        return {"read": "insufficient_data", "gloss": "Not enough current data to read the market.",
                "basis": list(avail), "n_signals": n, "trend_corroborates": False}
    b = sum(v == BUYER for v in avail.values()); s = sum(v == SELLER for v in avail.values())
    net = b - s
    read = "buyer-leaning" if net > 0 else "seller-leaning" if net < 0 else "balanced"
    tv = [avail[k] for k in TREND_KEYS if k in avail]
    tb, ts = tv.count(BUYER), tv.count(SELLER)
    if read == "buyer-leaning":
        corr = tb > ts; gloss = "A market increasingly favouring buyers." if corr else "A market currently favouring buyers."
    elif read == "seller-leaning":
        corr = ts > tb; gloss = "A market increasingly favouring sellers." if corr else "A market currently favouring sellers."
    else:
        corr = False; gloss = "Supply and demand broadly in balance."
    return {"read": read, "gloss": gloss, "basis": [f"{k}={avail[k]}" for k in avail],
            "n_signals": n, "trend_corroborates": corr}

# ── readers (fail-safe; never raise) ─────────────────────────────────────────
def _read_momentum(sq, area_code, region_name):
    try:
        where, p = ("area_code = %s", area_code) if area_code else ("region_name ILIKE %s", region_name)
        rows = sq(f"""with b as (select date, average_price, row_number() over (order by date desc) rn
                      from uk_hpi_monthly where {where})
                      select max(date) filter (where rn=1)::text latest,
                        100*(max(average_price) filter (where rn=1)/nullif(max(average_price) filter (where rn=4),0)-1) c3,
                        100*(max(average_price) filter (where rn=1)/nullif(max(average_price) filter (where rn=7),0)-1) c6,
                        100*(max(average_price) filter (where rn=1)/nullif(max(average_price) filter (where rn=13),0)-1) c12
                      from b""", (p,))
        if not rows or rows[0].get("c6") is None: return None, "no HPI rows"
        r = rows[0]
        return {"chg_3m": round(float(r["c3"]),2) if r.get("c3") is not None else None,
                "chg_6m": round(float(r["c6"]),2),
                "chg_12m": round(float(r["c12"]),2) if r.get("c12") is not None else None,
                "as_at": r["latest"]}, None
    except Exception as e: return None, f"hpi_error: {e}"

def _read_auction(sq, guide_price):
    """Compression: recent 12mo vs prior. Band if guide known, else overall.
    Always available on a populated book -> a reliable third signal."""
    try:
        if guide_price and guide_price > 0:
            lo, hi = (0,50000) if guide_price<50000 else (50000,100000) if guide_price<100000 else (100000,150000) if guide_price<150000 else (150000,10**9)
            filt, params, scope = "and guide_price >= %s and guide_price < %s", (lo,hi), "band"
        else:
            filt, params, scope = "", (), "overall"
        rows = sq(f"""select
            percentile_cont(0.5) within group (order by hammer_to_guide_ratio) med_all,
            percentile_cont(0.5) within group (order by hammer_to_guide_ratio) filter (where auction_date >= current_date - interval '12 months') mr,
            percentile_cont(0.5) within group (order by hammer_to_guide_ratio) filter (where auction_date <  current_date - interval '12 months') mp,
            count(*) filter (where auction_date >= current_date - interval '12 months') nr,
            count(*) filter (where auction_date <  current_date - interval '12 months') np
            from market_auction_outcomes
            where hammer_to_guide_ratio is not null and auction_date is not null {filt}""", params)
        if not rows or rows[0].get("mr") is None or rows[0].get("mp") is None: return None, "insufficient dated lots"
        r = rows[0]
        if (r["nr"] or 0) < 8 or (r["np"] or 0) < 8: return None, "insufficient dated lots"
        delta = round(float(r["mr"]) - float(r["mp"]), 3)
        return {"scope": scope, "median_htg": round(float(r["med_all"]),3), "delta": delta,
                "trend": "compressing" if delta <= -AUC else "widening" if delta >= AUC else "stable",
                "as_at": "auction outcomes"}, None
    except Exception as e: return None, f"auction_error: {e}"

def _read_rate_trend(sq):
    """6mo Bank-Rate delta from bench_rate_history (loaded by BoE loader). Until
    loaded -> None (display-only via bench_rates elsewhere). Verified fact
    2026-09: Bank Rate held 3.75% since 18 Dec 2025 -> delta 0 -> neutral."""
    try:
        rows = sq("""select rate_pct, as_of from bench_rate_history where series_code='IUDBEDR'
                     and as_of >= current_date - interval '7 months' order by as_of""", ())
        if len(rows) < 2: return None, "bench_rate_history not loaded (BoE loader)"
        return round(float(rows[-1]["rate_pct"]) - float(rows[0]["rate_pct"]), 2), None
    except Exception as e: return None, "bench_rate_history not loaded (BoE loader)"

def _read_rates_level(sq):
    """Display-only rate level from bench_rates (verified live). Does not vote."""
    try:
        rows = sq("select series_code, rate_pct, as_of::text as_of from bench_rates", ())
        m = {r["series_code"]: r for r in rows}
        if "IUMBV34" not in m: return None
        return {"bank_rate": float(m["IUDBEDR"]["rate_pct"]) if "IUDBEDR" in m else None,
                "fix_2y": float(m["IUMBV34"]["rate_pct"]),
                "fix_5y": float(m["IUMBV42"]["rate_pct"]) if "IUMBV42" in m else None,
                "as_at": m["IUMBV34"]["as_of"]}
    except Exception: return None

def _read_affordability(sq, area_code):
    try:
        rows = sq("select ratio, lr_avg, year::text yr from ons_affordability where area_code=%s order by year desc limit 1", (area_code,))
        if not rows or not rows[0].get("lr_avg"): return None, "ONS not loaded (ONS loader)"
        r = rows[0]
        return {"ratio": float(r["ratio"]), "vs_lr": round(float(r["ratio"])/float(r["lr_avg"]),3), "as_at": r["yr"]}, None
    except Exception: return None, "ONS not loaded (ONS loader)"

def _read_volume(dq, postcode):
    """Local transaction-volume trend (Hetzner price_paid_raw_2025), Cat-A, by
    outward district. Windows off the DATA's own max date minus a 2-month
    completeness buffer — Land Registry lags ~5wk and backfills recent months,
    so windowing off current_date undercounts 'recent' and biases every deal
    toward buyer. Verified schema 2026-09: date_of_transfer, ppd_category_type,
    postcode has a space, latest 2026-07-31."""
    if not postcode: return None, "no postcode"
    try:
        outward = postcode.strip().upper().split(" ")[0]
        rows = dq("""
            with mx as (select max(date_of_transfer) - interval '2 months' cutoff from price_paid_raw_2025)
            select
              count(*) filter (where date_of_transfer >= (select cutoff from mx) - interval '12 months'
                                 and date_of_transfer <  (select cutoff from mx)) nr,
              count(*) filter (where date_of_transfer >= (select cutoff from mx) - interval '24 months'
                                 and date_of_transfer <  (select cutoff from mx) - interval '12 months') np
            from price_paid_raw_2025
            where ppd_category_type='A' and postcode like %s
              and date_of_transfer >= (select cutoff from mx) - interval '24 months'""",
            (outward + ' %',))
        if not rows or (rows[0].get("np") or 0) < 10: return None, "insufficient local sales"
        r = rows[0]
        return {"trend_pct": round(100.0*(r["nr"]-r["np"])/r["np"],1), "n_recent": r["nr"], "as_at": "Land Registry"}, None
    except Exception as e:
        return None, "volume unavailable (prod/Hetzner only)"

def build_market_data(sq, dq, *, area_code=None, region_name=None, postcode=None, guide_price=None):
    data, un = {}, []
    mom, note = _read_momentum(sq, area_code, region_name)
    decel = None
    if mom:
        data["price_momentum"] = mom
        if mom.get("chg_6m") is not None and mom.get("chg_12m") is not None:
            decel = round(2*mom["chg_6m"] - mom["chg_12m"], 2)   # >0 accelerating, <0 decelerating
            data["price_deceleration"] = {"accel_pt": decel, "as_at": mom["as_at"]}
    else: un.append(f"price_momentum: {note}")
    auc, note = _read_auction(sq, guide_price)
    if auc: data["auction_clearing"] = auc
    else:   un.append(f"auction_clearing: {note}")
    rt, note = _read_rate_trend(sq)
    if rt is not None: data["rate_trend_6m_pt"] = rt
    if note: un.append(f"rate_trend: {note}")
    _rl = _read_rates_level(sq)
    if _rl: data["rates"] = _rl
    aff, note = _read_affordability(sq, area_code)
    if aff: data["affordability"] = aff
    else:   un.append(f"affordability: {note}")
    vol, note = _read_volume(dq, postcode)
    if vol: data["transactions"] = vol
    else:   un.append(f"transactions: {note}")
    read = classify(momentum=(mom or {}).get("chg_6m"), deceleration=decel,
                    auction=(auc or {}).get("delta"), affordability_vs_lr=(aff or {}).get("vs_lr"),
                    rate_trend=rt, volume=(vol or {}).get("trend_pct"))
    return {"data": data, "read": read, "_unavailable": un, "computed_at": datetime.utcnow().isoformat()+"Z"}

if __name__ == "__main__":
    P=0
    def ck(n,g,e):
        global P; P+=1; print(f"{'PASS' if g==e else 'FAIL'}  {n}"); assert g==e, f"{n}: {g} != {e}"
    # REAL live values (2026-09-09): Birmingham E08000025, B21 9JA £90k guide (50-100k band)
    hpi=[{"latest":"2026-06-01","c3":1.05,"c6":1.28,"c12":2.19}]
    auc=[{"med_all":1.307,"mr":1.307,"mp":1.385,"nr":95,"np":147}]  # 50-100k compressing (verified)
    def sq(sql,p=None):
        if "uk_hpi_monthly" in sql: return hpi
        if "market_auction_outcomes" in sql: return auc
        if "bench_rate_history" in sql: return []      # BoE loader not run
        if "ons_affordability" in sql: return []       # ONS loader not run
        return []
    def dq(sql,p=None): raise RuntimeError("Hetzner unreachable")
    def dq_live(sql,p=None): return [{"nr":100,"np":172}]   # real B21 (data-max window)
    md = build_market_data(sq,dq, area_code="E08000025", postcode="B21 9JA", guide_price=90000)
    print("\nREAL DATA — B21 9JA (Birmingham):")
    print("  rows:", list(md["data"].keys()))
    print("  READ:", md["read"]["read"], "|", md["read"]["gloss"], "| n=", md["read"]["n_signals"])
    print("  basis:", md["read"]["basis"])
    print("  unavailable:", md["_unavailable"])
    ck("read FIRES (not abstain)", md["read"]["read"] != "insufficient_data", True)
    ck("uses 3 verifiable signals", md["read"]["n_signals"], 3)
    md2 = build_market_data(sq,dq_live, area_code="E08000025", postcode="B21 9JA", guide_price=90000)
    print("  + volume live:", md2["data"].get("transactions"), "-> read:", md2["read"]["read"], "n=", md2["read"]["n_signals"])
    ck("volume becomes 4th vote", md2["read"]["n_signals"], 4)
    ck("B21 volume -42% -> buyer vote present", "volume=buyer" in md2["read"]["basis"], True)
    print(f"\n{P}/{P} tests passed.")
