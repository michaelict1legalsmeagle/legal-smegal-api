"""RENT-PIPR-1 (10 Oct 2026) — ONS private rents.

Root causes (verified on live data 10 Oct 2026):
 1. refresh_prms() downloaded the discontinued ONS "Private Rental Market Summary
    Statistics" (URLs now 404/502): uk_prms_monthly stuck at March 2026.
 2. _get_rental_trend read "ORDER BY period ASC LIMIT 48" = Jan 2015 - Dec 2018, so the
    stored local rent growth and the Demand signal were 2018 figures (HU9: -0.8% shown,
    +7.1% actual Mar 2026; Manchester 3.5% vs 2.8%; Birmingham 2.9% vs 3.5%).
 3. "regional" was the local authority itself; "national" averaged 316 local authorities.

Fixture: tests/fixtures/pipr_excerpt_16sep2026.xlsx holds the header rows and data rows
VERBATIM from ONS's 16 September 2026 workbook (Table 1) for 9 areas x 4 months plus one
Northern Ireland BRMA row; only the selection of rows is ours. Sheet order is changed
(Table 1 is the 2nd sheet, its part is sheet4.xml) to prove the sheet is found by name.
"""
import ast
import os
import sys
from typing import Any, Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pipr_rents as P  # noqa: E402

FIX = os.path.join(ROOT, "tests", "fixtures", "pipr_excerpt_16sep2026.xlsx")
APP = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
REFRESH = open(os.path.join(ROOT, "refresh_data.py"), encoding="utf-8").read()


def _rows():
    return {(r["area_code"], r["period"]): r for r in P.parse_workbook(FIX)["rows"]}


# ── parsing real ONS rows ──────────────────────────────────────────────────────
def test_values_match_ons_workbook():
    r = _rows()
    hull = r[("E06000010", "2026-08-01")]
    assert hull["rent_price_gbp"] == 694 and hull["rent_yoy_pct"] == 7.4
    assert hull["area_name"] == "Kingston upon Hull, City of"
    assert hull["region_name"] == "Yorkshire and The Humber"
    bham = r[("E08000025", "2026-08-01")]
    assert (bham["rent_price_gbp"], bham["rent_yoy_pct"], bham["region_name"]) == (1099, 3.0, "West Midlands")
    assert r[("E92000001", "2026-08-01")]["rent_yoy_pct"] == 4.0
    assert r[("W06000015", "2026-08-01")]["region_name"] == "Wales"
    assert r[("E12000005", "2026-08-01")]["region_name"] is None        # "[z]"


def test_march_2026_matches_the_rows_already_in_the_table():
    """Same source as the May 2026 load: Mar 2026 price/annual change are unchanged."""
    r = _rows()
    assert (r[("E06000010", "2026-03-01")]["rent_price_gbp"], r[("E06000010", "2026-03-01")]["rent_yoy_pct"]) == (684, 7.1)
    assert (r[("E08000025", "2026-03-01")]["rent_price_gbp"], r[("E08000025", "2026-03-01")]["rent_yoy_pct"]) == (1086, 3.5)
    assert r[("W06000015", "2026-03-01")]["rent_price_gbp"] == 1157


def test_index_rebased_to_jan_2015():
    r = _rows()
    for code in ("E06000010", "E08000025", "E92000001", "K02000001"):
        assert r[(code, "2015-01-01")]["rent_index"] == 100.0
    # ONS index 129.348171 / 85.360234 (Jan 2015) * 100
    assert abs(r[("E06000010", "2026-08-01")]["rent_index"] - 151.5321) < 1e-4


def test_unavailable_and_uncoded_rows_are_not_loaded():
    parsed = P.parse_workbook(FIX)
    codes = {x["area_code"] for x in parsed["rows"]}
    assert "[z]" not in codes                                   # NI BRMA rows have no GSS code
    assert ("N92000002", "2026-08-01") not in _rows()           # NI Aug 2026 is "[x]"
    assert all(x["rent_yoy_pct"] is None for x in parsed["rows"] if x["period"] == "2015-01-01")
    assert parsed["latest_period"] == "2026-08-01"
    assert parsed["skipped"] == 2


def test_header_change_fails_loudly():
    import pytest
    bad = [["Title"], ["Time period", "Area code", "Area name"], ["46235", "E06000010", "Hull"]]
    with pytest.raises(ValueError):
        P.parse_rows(iter(bad))
    with pytest.raises(ValueError):
        P.parse_rows(iter([["no header here"]]))


# ── finding the newest edition on the dataset page ────────────────────────────
PAGE = """
<a href="/file?uri=/economy/inflationandpriceindices/datasets/priceindexofprivaterentsukmonthlypricestatistics/19august2026/priceindexofprivaterentsukmonthlypricestatistics.xlsx">xlsx (17.6 MB)</a>
<a href="https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/priceindexofprivaterentsukmonthlypricestatistics/16september2026/priceindexofprivaterentsukmonthlypricestatistics.xlsx">xlsx (17.8 MB)</a>
<a href="/file?uri=/economy/inflationandpriceindices/datasets/priceindexofprivaterentsukmonthlypricestatistics/22july2026/priceindexofprivaterentsukmonthlypricestatistics14.xlsx">xlsx (17.5 MB)</a>
<a href="/file?uri=/economy/inflationandpriceindices/datasets/privaterentalmarketsummarystatisticsinengland/current/prt1a.csv">old</a>
"""


def test_latest_edition_is_picked_by_date_not_position():
    url, d = P.latest_xlsx_url(PAGE)
    assert d.isoformat() == "2026-09-16"
    assert url == ("https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices/datasets/"
                   "priceindexofprivaterentsukmonthlypricestatistics/16september2026/"
                   "priceindexofprivaterentsukmonthlypricestatistics.xlsx")
    assert P.latest_xlsx_url("<html>nothing</html>") is None


# ── refresh wiring ────────────────────────────────────────────────────────────
def _func(src, name):
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.get_source_segment(src, n)
    raise AssertionError(name)


def test_refresh_uses_pipr_and_is_isolated():
    f = _func(REFRESH, "refresh_prms")
    assert "pipr_rents.latest_xlsx_url" in f and "pipr_rents.parse_workbook" in f
    assert 'on_conflict="period,area_code"' in f
    assert "except Exception" in f and "other refreshes unaffected" in f
    assert "privaterentalmarketsummarystatistics" not in REFRESH


# ── app: latest data, true region, England ────────────────────────────────────
def test_trend_reads_latest_months_with_index():
    f = _func(APP, "_get_rental_trend")
    assert ('"SELECT period, rent_index, rent_yoy_pct, rent_price_gbp FROM public.uk_prms_monthly '
            'WHERE area_code = %s ORDER BY period DESC LIMIT 48"') in f
    assert '"SELECT period, rent_yoy_pct, rent_price_gbp FROM public.uk_prms_monthly WHERE area_code = %s ORDER BY period ASC' not in f
    assert "reversed(" in f


def test_trend_series_is_chronological_and_latest_wins():
    src = _func(APP, "_get_rental_trend")
    desc = [{"period": f"2026-0{m}-01", "rent_index": 150 + m, "rent_yoy_pct": m, "rent_price_gbp": 600 + m}
            for m in range(8, 0, -1)]
    g = {"supabase_data_query": lambda *a, **k: desc, "safe_float": lambda v: None if v is None else float(v),
         "Dict": Dict, "Any": Any}
    exec(src, g)
    out = g["_get_rental_trend"]("E06000010")
    assert out["series"][0]["period"] == "2026-01-01" and out["series"][-1]["period"] == "2026-08-01"
    assert out["latest_yoy"] == 8.0 and out["series"][-1]["rent_index"] == 158.0


def test_regional_is_the_real_region():
    f = _func(APP, "_get_regional_rental_benchmark")
    assert "r.area_name = l.region_name" in f
    assert "E12%%" in f and "W92%%" in f


def test_benchmark_block_shape():
    seg = APP[APP.index('"rental": {\n                        # RENT-PIPR-1'):][:1500]
    for k in ("local_yoy", "local_rent_gbp", "local_name", "regional_yoy", "regional_rent_gbp",
              "regional_name", "national_yoy", "national_name", "as_of", '"schema":            RENT_SCHEMA'):
        assert k in seg, k
    assert '"rent_basis":     {"schema": RENT_SCHEMA' in APP
    assert '"rental":  demand_index' in APP
    assert "ONS PRMS" not in APP


# ── read-time heal ────────────────────────────────────────────────────────────
def test_heal_hooked_on_all_three_read_paths():
    assert APP.count("_maybe_refresh_rent_inference(deal_id,") == 3


def test_is_current_rules():
    g = {"RENT_SCHEMA": "rent-pipr-1", "Any": Any, "Optional": Optional}
    exec(_func(APP, "_rent_inference_is_current"), g)
    cur = g["_rent_inference_is_current"]
    assert cur(None, "2026-08-01")                                          # nothing to heal
    assert not cur({"benchmarks": {}}, "2026-08-01")                        # pre-RENT-PIPR-1
    assert not cur({"rent_basis": {"schema": "rent-pipr-1", "as_of": "2026-07-01"}}, "2026-08-01")
    assert cur({"rent_basis": {"schema": "rent-pipr-1", "as_of": "2026-08-01"}}, "2026-08-01")
    assert cur({"rent_basis": {"schema": "rent-pipr-1", "as_of": "2026-08-01"}}, None)


class _Res:
    def __init__(self, d): self.data = d


class _Q:
    def __init__(self, sb): self.sb = sb
    def select(self, *a): return self
    def eq(self, k, v): self.sb.log.append(("eq", k, v)); return self
    def limit(self, n): return self
    def update(self, payload): self.sb.log.append(("update", payload)); self.sb.updated = payload; return self
    def execute(self):
        if self.sb.updated is not None and self.sb.log[-1][0] == "eq" and self.sb.log[-1][1] == "updated_at":
            return _Res([{"id": "x"}])
        return _Res([{"updated_at": "T0", "area_json": {"inference": {"old": True}, "flood": {"keep": 1}}}])


class _SB:
    def __init__(self): self.log, self.updated = [], None
    def table(self, n): return _Q(self)


def _heal_env(build):
    import logging
    sb = _SB()
    cache = {}
    g = {"Dict": Dict, "Any": Any, "Optional": Optional, "supabase": sb,
         "_is_scotland_lsoa": lambda s: str(s).startswith("S"),
         "_rent_latest_period": lambda: "2026-08-01",
         "geo_cache_get": cache.get, "geo_cache_set": cache.__setitem__,
         "now_iso": lambda: "2026-10-10T00:00:00+00:00", "_json_sanitize": lambda x: x,
         "build_area_inference": build,
         "app": type("A", (), {"logger": logging.getLogger("t")})}
    g["RENT_SCHEMA"] = "rent-pipr-1"
    exec(_func(APP, "_rent_inference_is_current"), g)
    exec(_func(APP, "_maybe_refresh_rent_inference"), g)
    return g["_maybe_refresh_rent_inference"], sb


def test_heal_rebuilds_and_persists_only_inference():
    new_inf = {"rent_basis": {"schema": "rent-pipr-1", "as_of": "2026-08-01"},
               "benchmarks": {"rental": {"local_yoy": 7.4}}, "signals": {"demand": "Increasing"}}
    heal, sb = _heal_env(lambda area, pc: {"inference": new_inf})
    area = {"postcode": "HU9 3AQ", "lsoa_gss": "E01012345", "inference": {"benchmarks": {"rental": {"local_yoy": -0.8}}}}
    out = heal("dd813528", area)
    assert out["inference"] is new_inf and out["flood"] == {"keep": 1}     # other keys preserved
    assert ("eq", "updated_at", "T0") in sb.log                             # optimistic lock
    n = len(sb.log)
    heal("dd813528", {"postcode": "HU9 3AQ", "inference": {"benchmarks": {}}})   # rate-limited
    assert len(sb.log) == n


def test_heal_never_serves_a_failed_rebuild_or_touches_scotland():
    heal, sb = _heal_env(lambda area, pc: {"inference": {"error": "boom", "trajectory": "UNKNOWN"}})
    old = {"benchmarks": {"rental": {"local_yoy": -0.8}}}
    area = {"postcode": "HU9 3AQ", "lsoa_gss": "E01012345", "inference": old}
    assert heal("a", area)["inference"] is old and sb.updated is None
    heal2, sb2 = _heal_env(lambda area, pc: (_ for _ in ()).throw(AssertionError("must not run")))
    heal2("b", {"postcode": "EH1 1AA", "lsoa_gss": "S01008000", "inference": {}})
    assert sb2.log == []
