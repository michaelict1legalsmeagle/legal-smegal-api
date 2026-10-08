"""CRIME-EVID-1 (8 Oct 2026) — England & Wales crime: police.uk street + Home Office district.

Root cause (verified live 8 Oct 2026): police.uk holds NO Greater Manchester Police data
("Currently no crime, outcome or stop and search data is available" — data.police.uk/changelog)
and other forces miss months, so police.uk returns HTTP 200 with an EMPTY list. The old
get_crime_data treated that as a genuine "0 crimes": 8 Greater Manchester deals showed
"0 recorded crimes" (a false safety signal) and the Verdict read "0 recorded crimes in .".

These tests use the REAL ew_crime module, the REAL committed ONS lookup, rows parsed from
the REAL Home Office file and REAL police.uk responses (tests/fixtures), and lock the
app.py wiring with AST checks (app.py is not importable in CI).
"""
import ast
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ew_crime as ec  # noqa: E402

FX = json.load(open(os.path.join(ROOT, "tests", "fixtures", "crime_evid_fixtures.json"), encoding="utf-8"))
LOOKUP = ec.load_lookup(os.path.join(ROOT, ec.LOOKUP_FILE))
HO_NAMES = FX["ho_all_csp_names"]
APP = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()


def _rows_for(res):
    return [r for r in FX["ho_rows"] if r["csp_name"] in res["csps"]]


def _district(area_code):
    res = ec.resolve_district(area_code, LOOKUP, HO_NAMES)
    return ec.district_block(res, _rows_for(res))


# ── street (police.uk) ──────────────────────────────────────────────────────
def test_empty_policeuk_answer_is_never_zero():
    s = ec.street_block(200, FX["policeuk_cottonhill_2026_08"], "2026-08", "u")
    assert s["status"] == "no_records"
    assert s["total"] is None                      # NOT 0
    assert "no street-level records" in s["note"] and "August 2026" in s["note"]


def test_policeuk_records_counted_by_category():
    # requested 2026-08 but records are 2026-06: the label must follow the records
    s = ec.street_block(200, FX["policeuk_birmingham_b29_2026_06"], "2026-08", "u")
    assert s["status"] == "ok" and s["total"] == 5 and s["month"] == "2026-06"
    assert s["categories"] == {"anti-social-behaviour": 2, "bicycle-theft": 2, "burglary": 1}


def test_btp_station_records_are_not_area_crime():
    s = ec.street_block(200, [FX["policeuk_btp_cottonhill_2023_09"]], "2023-09", "u")
    assert s["status"] == "no_records" and s["total"] is None and s["btp_excluded"] == 1


def test_failed_fetch_is_unavailable_not_zero():
    for status, body in ((500, None), (200, {"error": "x"}), (None, None)):
        s = ec.street_block(status, body, "2026-08", "u")
        assert s["status"] == "unavailable" and s["total"] is None


# ── district (Home Office) via the official ONS lookup ─────────────────────
def test_manchester_gets_home_office_district_figure():
    d = _district("E08000003")
    assert d["status"] == "ok" and d["name"] == "Manchester" and d["mode"] == "exact"
    assert d["total"] == 84130 and d["period"] == "Apr 2025 – Mar 2026"
    assert d["top_group"] == "Violence against the person" and d["top_count"] == 30605


def test_shared_partnership_is_labelled_as_partnership():
    d = _district("E07000089")  # Hart -> North Hampshire CSP
    assert d["status"] == "ok" and d["mode"] == "partnership"
    assert d["name"].startswith("North Hampshire community safety partnership (shared with")


def test_new_unitary_sums_its_exclusive_legacy_districts():
    d = _district("E06000060")  # Buckinghamshire
    assert d["status"] == "ok" and d["mode"] == "combined"
    assert set(d["csps"]) == {"Aylesbury Vale", "Chiltern", "South Bucks", "Wycombe"}
    assert d["total"] == sum(r["offences"] for r in FX["ho_rows"] if r["csp_name"] in d["csps"])


def test_council_split_across_shared_partnerships_is_unavailable_not_guessed():
    d = _district("E06000058")  # BCP: Bournemouth + Poole + (shared) Dorset
    assert d["status"] == "unavailable" and "no published figure matches" in d["note"]


def test_home_office_flagged_district_is_unavailable_with_its_note():
    d = _district("E06000035")  # Medway — Home Office: recorded under 'Unassigned Kent'
    assert d["status"] == "unavailable"
    assert d["note"].startswith("Home Office note: Kent: Due to ongoing IT issues")


def test_quality_flags_come_only_from_the_published_note():
    flags = ec.quality_flags(FX["ho_notes"], HO_NAMES, "2025/26")
    assert set(flags) == {"Maidstone", "Medway", "Tonbridge and Malling",
                          "Dartford and Gravesham", "Tunbridge Wells"}
    # Derbyshire note applies to year ending March 2025 only and names no CSP
    assert not any("Derbyshire" in k for k in flags)


def test_every_council_in_the_lookup_resolves():
    modes = {ec.resolve_district(c, LOOKUP, HO_NAMES)["mode"] for c in LOOKUP}
    unmatched = [c for c in LOOKUP if ec.resolve_district(c, LOOKUP, HO_NAMES)["mode"] == "unmatched"]
    assert len(LOOKUP) == 318 and unmatched == ["E06000058"]   # only BCP (shared Dorset CSP)
    assert modes == {"exact", "partnership", "combined", "unmatched"}


# ── assembled block ─────────────────────────────────────────────────────────
def test_cotton_hill_shows_district_and_honest_street_line():
    street = ec.street_block(200, FX["policeuk_cottonhill_2026_08"], "2026-08", "u")
    out = ec.assemble(street, _district("E08000003"), "2026-10-08T00:00:00Z")
    m = out["metrics"]
    assert out["status"] == "ok" and m["schema"] == ec.SCHEMA_VERSION
    assert m["total"] is None and m["street_status"] == "no_records" and m["month"] == "2026-08"
    assert m["district"]["total"] == 84130
    assert "84,130" in out["summary"] and "no street-level records" in out["summary"]
    assert " 0 " not in out["summary"]


def test_both_sources_missing_is_unavailable():
    street = ec.street_block(503, None, "2026-08", "u")
    out = ec.assemble(street, _district("E06000058"), "t")
    assert out["status"] == "unavailable" and out["metrics"]["total"] is None


# ── freshness + refresh source ──────────────────────────────────────────────
def test_staleness_rules():
    assert ec.is_stale({"metrics": {"total": 0}}, "2026-08")                       # pre-CRIME-EVID-1
    assert ec.is_stale({"metrics": {"schema": ec.SCHEMA_VERSION, "month": "2026-07"}}, "2026-08")
    assert not ec.is_stale({"metrics": {"schema": ec.SCHEMA_VERSION, "month": "2026-08"}}, "2026-08")
    assert not ec.is_stale({"metrics": {"jurisdiction": "scotland"}}, "2026-08")   # Scotland untouched


def test_newest_csp_file_is_picked_from_govuk_page():
    url, name = ec.latest_csp_file_url(FX["govuk_page_snippet"])
    assert name == "prc-csp-mar2021-mar2026-tables-230726.ods" and url.endswith(name)


def test_period_label():
    assert ec.period_label("2025/26", ["1", "2", "3", "4"]) == "Apr 2025 – Mar 2026"
    assert ec.period_label("2026/27", ["1"]) == "Apr 2026 – Jun 2026"


# ── app.py wiring (AST/text locks; app.py is not importable in CI) ─────────
def _fn(name):
    tree = ast.parse(APP)
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.get_source_segment(APP, n)
    raise AssertionError(name + " missing")


def test_app_imports_module_and_lookup_at_load():
    assert "\nimport ew_crime as _ew_crime\n" in APP
    assert "_EW_LOOKUP = _ew_crime.load_lookup(" in APP


def test_get_crime_data_uses_both_sources():
    body = _fn("get_crime_data")
    assert "_ew_district_crime(area_code)" in body
    assert "_ew_crime.street_block(" in body and "_ew_crime.assemble(" in body
    assert "metric_ok(" not in body          # old single-source path gone


def test_both_call_sites_pass_area_code():
    assert "get_crime_data(lat, lng, area_code)" in APP
    assert "else [lat, lng, area_code]), {})," in APP
    assert "get_crime_data(lat, lng))" not in APP


def test_refresh_hooks_on_all_three_read_paths():
    assert "area = _maybe_refresh_crime(deal_id, area)" in _fn("get_area")
    assert "cached = _maybe_refresh_crime(deal_id, cached)" in _fn("save_area")
    assert 'deal["area_json"] = _maybe_refresh_crime(deal_id, deal["area_json"])' in _fn("get_deal")


def test_benchmark_local_total_not_zero_when_no_records():
    assert '"local_total":      int(crime_total),' not in APP


# ── _maybe_refresh_crime behaviour (function exec'd from app.py source with stubs) ──
class _Q:
    def __init__(self, log, rows):
        self.log, self.rows, self.kind = log, rows, None
    def select(self, *a): self.kind = "select"; return self
    def update(self, payload): self.kind = "update"; self.log.append(("update", payload)); return self
    def eq(self, k, v):
        if self.kind == "update": self.log.append(("eq", k, v))
        return self
    def limit(self, n): return self
    def execute(self):
        class R: pass
        r = R(); r.data = self.rows if self.kind == "select" else [{"id": "d1"}]
        return r


def _refresher(fresh, stored_rows):
    import logging
    import types
    log = []
    sb = types.SimpleNamespace(table=lambda name: _Q(log, stored_rows))
    ns = {"Optional": __import__("typing").Optional, "Dict": dict, "Any": object,
          "supabase": sb, "_ew_crime": ec, "_policeuk_latest_month": lambda: "2026-08",
          "geo_cache_get": lambda k: None, "geo_cache_set": lambda k, v: None,
          "now_iso": lambda: "2026-10-08T00:00:00Z", "safe_float": float,
          "get_crime_data": lambda lat, lng, code: fresh, "_json_sanitize": lambda x: x,
          "_is_scotland_lsoa": lambda s: s.startswith("S01"),
          "app": types.SimpleNamespace(logger=logging.getLogger("t"))}
    exec(compile(_fn("_maybe_refresh_crime"), "app.py", "exec"), ns)
    return ns["_maybe_refresh_crime"], log


def _legacy_area():
    return {"lat": 53.43, "lng": -2.23, "area_code": "E08000003", "lsoa_gss": "E01005307",
            "crime": {"metrics": {"total": 0, "categories": {}, "month": "2026-08"}}}


def test_refresh_persists_a_good_block_with_optimistic_lock():
    fresh = ec.assemble(ec.street_block(200, FX["policeuk_cottonhill_2026_08"], "2026-08", "u"),
                        _district("E08000003"), "t")
    fn, log = _refresher(fresh, [{"updated_at": "U0", "area_json": _legacy_area()}])
    out = fn("d1", _legacy_area())
    assert out["crime"]["metrics"]["district"]["total"] == 84130
    assert ("eq", "updated_at", "U0") in log and any(e[0] == "update" for e in log)


def test_refresh_never_persists_when_policeuk_is_down():
    fresh = ec.assemble(ec.street_block(503, None, "2026-08", "u"), _district("E08000003"), "t")
    fn, log = _refresher(fresh, [{"updated_at": "U0", "area_json": _legacy_area()}])
    out = fn("d1", _legacy_area())
    assert not log                                            # nothing written
    assert out["crime"]["metrics"]["street_status"] == "unavailable"   # served for this view only


def test_refresh_keeps_stored_block_when_both_sources_fail_and_skips_scotland():
    fresh = ec.assemble(ec.street_block(503, None, "2026-08", "u"), _district("E06000058"), "t")
    fn, log = _refresher(fresh, [])
    a = _legacy_area()
    assert fn("d1", a)["crime"]["metrics"]["total"] == 0 and not log
    scot = {"lsoa_gss": "S01000001", "crime": {"metrics": {"jurisdiction": "scotland", "total": 5}}}
    assert fn("d1", scot) is scot and not log


def test_refresh_never_persists_when_hetzner_district_read_fails():
    street = ec.street_block(200, FX["policeuk_cottonhill_2026_08"], "2026-08", "u")
    transient = {"source": ec.DISTRICT_SOURCE, "source_url": ec.DATASET_PAGE, "status": "unavailable",
                 "name": None, "transient": True, "note": "Home Office district crime data is not available right now."}
    fresh = ec.assemble(street, transient, "t")
    fn, log = _refresher(fresh, [{"updated_at": "U0", "area_json": _legacy_area()}])
    fn("d1", _legacy_area())
    assert not log
    assert 'return {"source": _ew_crime.DISTRICT_SOURCE' in _fn("_ew_district_crime")   # query failure is transient
    assert _fn("_ew_district_crime").count('"transient": True') == 2
