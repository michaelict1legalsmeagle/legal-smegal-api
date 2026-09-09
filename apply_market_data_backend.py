#!/usr/bin/env python3
"""
Wire build_market_data into the /area POST fresh-fetch path.
Additive · fail-safe · DISPLAY/CONTEXT only · never touches the ceiling.
Review the diff, then run:  python3 apply_market_data_backend.py app.py
Deploy market_data.py alongside app.py.
"""
import sys, pathlib

path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "app.py")
src = path.read_text()

anchor = (
    '            # ── INFERENCE ENGINE ─────────────────────────────────\n'
    '            inference_result = build_area_inference(area_data, _postcode)\n'
    '            area_data.update(inference_result)\n'
)
assert src.count(anchor) == 1, f"anchor count = {src.count(anchor)} (expected 1) — inspect before writing"

inject = anchor + (
    '\n'
    '            # ── CURRENT MARKET DATA (SPEC v2.1; additive) ────────────────────\n'
    '            # Dated market facts + derived buyer/seller read. Fail-safe: any\n'
    '            # feed miss => that line unavailable, never fabricated. DISPLAY /\n'
    '            # CONTEXT only — writes area_json.market_data, touches NO ceiling.\n'
    '            try:\n'
    '                from market_data import build_market_data\n'
    '                area_data["market_data"] = build_market_data(\n'
    '                    supabase_data_query, data_query,\n'
    '                    area_code=area_code, postcode=_postcode,\n'
    '                    guide_price=_guide_price_gbp,\n'
    '                )\n'
    '            except Exception as _mde:\n'
    '                print(f"[market_data] build failed for {_deal_id}: {_mde}")\n'
)

out = src.replace(anchor, inject, 1)
assert out != src and out.count('area_data["market_data"] = build_market_data(') == 1
path.write_text(out)
print("OK — market_data wired into /area POST. Deploy market_data.py alongside app.py.")
print("Ceiling untouched. Re-run the 2026-09-09 ceiling audit as the standing guard.")
