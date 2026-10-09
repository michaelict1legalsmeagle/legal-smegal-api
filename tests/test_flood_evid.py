"""FLOOD-EVID-1 (9 Oct 2026) — England flood zone from the EA Flood Map for Planning.

Root cause (verified live 9 Oct 2026): get_flood_risk asked the EA flood-monitoring API
for flood ALERT/WARNING areas (e.g. "River Rea", "Upper Tame") and read a zone digit out
of their names. The names carry none, so every successful call became "Zone 1, <0.1%".
Checked against the official Flood Map for Planning at each deal's point, 8 of 59 stored
"Zone 1" deals are in Flood Zone 2 or 3. Separately, a 6 s timeout (Cotton Hill,
7 Oct: "Read timed out") was stored as a permanent "unavailable".

Fixture note: the response ENVELOPE and feature PROPERTIES below are exactly as the live
service returned them on 9 Oct 2026 (ids, flood_zone, flood_source, origin,
numberMatched/numberReturned, crs). Real polygons run to hundreds or thousands of
vertices, so the GEOMETRY is replaced by small squares placed to reproduce the live
containment result for each point (point-in-polygon was checked live in the browser for
all 80 England & Wales deals). Tests therefore prove the logic, not the EA's shapes.
"""
import ast
import os
import sys
import types
import logging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ew_flood as ef  # noqa: E402

APP = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()


def _sq(lng, lat, d=0.001):
    return [[[lng - d, lat - d], [lng + d, lat - d], [lng + d, lat + d], [lng - d, lat + d], [lng - d, lat - d]]]


def _feat(fid, zone, source, origin, polys):
    return {"type": "Feature", "id": f"Flood_Zones_2_3_Rivers_and_Sea.{fid}",
            "geometry": {"type": "MultiPolygon", "coordinates": polys}, "geometry_name": "shape",
            "properties": {"origin": origin, "flood_zone": zone, "flood_source": source}}


def _env(features, matched=None):
    n = len(features)
    return {"type": "FeatureCollection", "features": features, "totalFeatures": n,
            "numberMatched": n if matched is None else matched, "numberReturned": n,
            "timeStamp": "2026-10-09T21:10:03.775Z",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}}}


# Live points (deal coordinates as stored on the deals)
TS24 = (54.680858, -1.208423)   # live: FZ2 river (feature 463497 contains it; 463498 does not)
DN32 = (53.570935, -0.060992)   # live: FZ3 sea   (615720 contains it; 615716 does not)
COTTON = (53.427835, -2.224123)  # live: no features -> Flood Zone 1


def test_zone_2_river_from_the_polygon_that_contains_the_point():
    la, lo = TS24
    payload = _env([
        _feat(463497, "FZ2", "river", "modelled", [_sq(lo, la)]),
        _feat(463498, "FZ2", "river", "modelled", [_sq(lo + 0.01, la)]),  # in the box list, not at the point
    ])
    b = ef.flood_block(200, payload, la, lo, "t")
    assert b["status"] == "ok" and b["metrics"]["zone"] == 2 and b["metrics"]["flood_sources"] == ["river"]
    assert b["value"][0]["zone"] == 2
    assert "Flood Zone 2" in b["summary"] and "1 in 1,000 to 1 in 100" in b["summary"]
    assert "ignore flood defences" in b["summary"]


def test_zone_3_sea_uses_sea_probability():
    la, lo = DN32
    payload = _env([
        _feat(615716, "FZ3", "sea", "modelled", [_sq(lo + 0.02, la)]),
        _feat(615720, "FZ3", "sea", "modelled", [_sq(lo, la)]),
    ])
    b = ef.flood_block(200, payload, la, lo, "t")
    assert b["metrics"]["zone"] == 3 and b["metrics"]["flood_sources"] == ["sea"]
    assert b["metrics"]["probability"] == "1 in 200 or more a year (sea)"


def test_highest_zone_wins_and_river_and_sea_split():
    la, lo = DN32
    payload = _env([_feat(1, "FZ2", "river", "modelled", [_sq(lo, la)]),
                    _feat(2, "FZ3", "river and sea", "modelled and recorded", [_sq(lo, la, 0.0005)])])
    b = ef.flood_block(200, payload, la, lo, "t")
    assert b["metrics"]["zone"] == 3 and b["metrics"]["flood_sources"] == ["river", "sea"]
    assert b["metrics"]["origin"] == ["modelled", "modelled and recorded"]


def test_point_in_a_hole_is_not_inside():
    la, lo = TS24
    outer = _sq(lo, la, 0.01)[0]
    hole = _sq(lo, la, 0.001)[0]
    b = ef.flood_block(200, _env([_feat(9, "FZ3", "river", "modelled", [[outer, hole]])]), la, lo, "t")
    assert b["metrics"]["zone"] == 1


def test_no_polygon_at_the_point_is_zone_1_by_the_ea_definition():
    la, lo = COTTON
    b = ef.flood_block(200, _env([]), la, lo, "t")
    assert b["status"] == "ok" and b["metrics"]["zone"] == 1 and b["metrics"]["flood_sources"] == []
    assert b["metrics"]["probability"] == "below 1 in 1,000 a year"
    assert b["metrics"]["schema"] == ef.SCHEMA_VERSION


def test_failures_are_unavailable_and_transient_never_a_zone():
    la, lo = COTTON
    for status, body in ((None, None), (500, None), (200, {"error": "x"}), (200, None)):
        b = ef.flood_block(status, body, la, lo, "t")
        assert b["status"] == "unavailable" and b["value"] is None
        assert "zone" not in b["metrics"] and b["metrics"]["transient"] is True


def test_incomplete_answer_below_zone_3_is_unavailable():
    la, lo = TS24
    payload = _env([_feat(1, "FZ2", "river", "modelled", [_sq(lo, la)])], matched=250)
    assert ef.flood_block(200, payload, la, lo, "t")["status"] == "unavailable"
    payload3 = _env([_feat(1, "FZ3", "river", "modelled", [_sq(lo, la)])], matched=250)
    assert ef.flood_block(200, payload3, la, lo, "t")["metrics"]["zone"] == 3   # cannot go higher


def test_old_flood_monitoring_blocks_are_not_current():
    old = {"status": "ok", "summary": "Zone 1 — no flood risk areas recorded at this location.",
           "value": [{"zone": 1, "areas": 0}], "metrics": {"zone": 1, "flood_areas": 0}}
    assert not ef.is_current(old)
    assert not ef.is_current({"status": "unavailable", "metrics": {}})
    assert ef.is_current(ef.flood_block(200, _env([]), *COTTON, "t"))
    assert not ef.is_current(ef.pending_recheck("t"))


def test_query_box_is_lon_lat_order():
    p = ef.query_params(53.0, -2.0)
    w, s, e, n = [float(x) for x in p["bbox"].split(",")]
    assert w < -2.0 < e and s < 53.0 < n and p["f"] == "json"


# ── app.py wiring (AST / exec with stubs; app.py is not importable in CI) ─────────────
def _fn(name):
    for n in ast.walk(ast.parse(APP)):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.get_source_segment(APP, n)
    raise AssertionError(name + " missing")


def test_get_flood_risk_uses_the_flood_map_not_flood_monitoring():
    body = _fn("get_flood_risk")
    assert "_ew_flood.ITEMS_URL" in body and "_ew_flood.flood_block(" in body
    assert "flood-monitoring/id/floodAreas" not in body and "dist=0.5" not in body


def test_refresh_hooks_on_all_three_read_paths():
    assert "area = _maybe_refresh_flood_amenities(deal_id, area)" in _fn("get_area")
    assert "cached = _maybe_refresh_flood_amenities(deal_id, cached)" in _fn("save_area")
    assert 'deal["area_json"] = _maybe_refresh_flood_amenities(deal_id, deal["area_json"])' in _fn("get_deal")


class _Q:
    def __init__(self, log, rows):
        self.log, self.rows, self.kind = log, rows, None
    def select(self, *a): self.kind = "select"; return self
    def update(self, payload): self.kind = "update"; self.log.append(("update", payload)); return self
    def eq(self, k, v):
        if self.kind == "update":
            self.log.append(("eq", k, v))
        return self
    def limit(self, n): return self
    def execute(self):
        r = types.SimpleNamespace()
        r.data = self.rows if self.kind == "select" else [{"id": "d1"}]
        return r


LOCAL_AMEN = {"status": "ok", "metrics": {"total": 721},
              "sources": [{"label": "OpenStreetMap POIs (local)", "url": ""}]}
OLD_AMEN = {"status": "ok", "value": [], "metrics": {},
            "sources": [{"label": "OpenStreetMap (Overpass API)", "url": "https://overpass-api.de/"}]}
OLD_FLOOD = {"status": "ok", "value": [{"zone": 1, "areas": 2}], "metrics": {"zone": 1, "flood_areas": 2}}


def _refresher(flood_out, amen_out, stored_rows, cached=False):
    log, calls = [], []
    sb = types.SimpleNamespace(table=lambda name: _Q(log, stored_rows))
    def _flood(lat, lng, pc):
        calls.append("flood")
        return flood_out
    def _amen(lat, lng):
        calls.append("amen")
        return amen_out
    ns = {"Optional": __import__("typing").Optional, "Dict": dict, "Any": object,
          "supabase": sb, "_ew_flood": ef, "geo_cache_get": lambda k: cached,
          "geo_cache_set": lambda k, v: None, "now_iso": lambda: "2026-10-09T00:00:00Z",
          "safe_float": lambda v: None if v is None else float(v),
          "get_flood_risk": _flood, "get_amenities_data": _amen, "_json_sanitize": lambda x: x,
          "_is_scotland_lsoa": lambda s: s.startswith("S01"),
          "app": types.SimpleNamespace(logger=logging.getLogger("t"))}
    exec(compile(_fn("_amenities_is_current"), "app.py", "exec"), ns)
    exec(compile(_fn("_maybe_refresh_flood_amenities"), "app.py", "exec"), ns)
    return ns["_maybe_refresh_flood_amenities"], log, calls


def _area(flood=OLD_FLOOD, amen=LOCAL_AMEN):
    return {"lat": 53.570935, "lng": -0.060992, "lsoa_gss": "E01013214", "flood": dict(flood), "amenities": dict(amen)}


def test_wrong_old_zone_1_is_replaced_and_persisted():
    fresh = ef.flood_block(200, _env([_feat(615720, "FZ3", "sea", "modelled", [_sq(-0.060992, 53.570935)])]),
                           53.570935, -0.060992, "t")
    fn, log, calls = _refresher(fresh, None, [{"updated_at": "U0", "area_json": _area()}])
    out = fn("d1", _area())
    assert calls == ["flood"] and out["flood"]["metrics"]["zone"] == 3
    assert ("eq", "updated_at", "U0") in log
    written = [e for e in log if e[0] == "update"][0][1]["area_json"]
    assert written["flood"]["metrics"]["zone"] == 3 and written["amenities"] == LOCAL_AMEN


def test_failed_recheck_is_served_unavailable_never_written_and_old_zone_never_shown():
    fresh = ef.flood_block(None, None, 53.57, -0.06, "t")
    fn, log, _ = _refresher(fresh, None, [{"updated_at": "U0", "area_json": _area()}])
    out = fn("d1", _area())
    assert not log and out["flood"]["status"] == "unavailable" and "zone" not in out["flood"]["metrics"]


def test_rate_limited_old_block_is_not_served_as_a_zone():
    fn, log, calls = _refresher(None, None, [], cached={"at": "x"})
    out = fn("d1", _area())
    assert calls == [] and not log
    assert out["flood"]["status"] == "unavailable" and "re-checked" in out["flood"]["summary"]


def test_old_overpass_amenities_rebuilt_from_local_table_and_persisted():
    cur_flood = ef.flood_block(200, _env([]), 53.57, -0.06, "t")
    fn, log, calls = _refresher(None, LOCAL_AMEN, [{"updated_at": "U0", "area_json": _area(cur_flood, OLD_AMEN)}])
    out = fn("d1", _area(cur_flood, OLD_AMEN))
    assert calls == ["amen"] and out["amenities"]["metrics"]["total"] == 721
    written = [e for e in log if e[0] == "update"][0][1]["area_json"]
    assert written["amenities"]["metrics"]["total"] == 721


def test_current_blocks_and_scotland_are_left_alone():
    cur_flood = ef.flood_block(200, _env([]), 53.57, -0.06, "t")
    fn, log, calls = _refresher(None, None, [])
    a = _area(cur_flood, LOCAL_AMEN)
    assert fn("d1", a) is a and calls == [] and not log
    scot = {"lsoa_gss": "S01006715", "flood": {"status": "ok", "metrics": {"risk": "Minimal"}}}
    assert fn("d1", scot) is scot and calls == []
