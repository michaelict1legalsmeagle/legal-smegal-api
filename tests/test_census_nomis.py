"""CENSUS-FIX-1 (8 Oct 2026) — Census 2021 people-profile tables on the Area page.

Root causes, verified against the live Nomis API on 8 Oct 2026:
  * TS021 ethnic: dataset NM_2041_1 has ONE category dimension, c2021_eth_20.
    c2021_eth_8 (Render env override) and c2021_eth_25 (old code default) both
    return Nomis error "Cannot create query" -> ethnic empty on all 89 deals.
  * TS007A age: c2021_age_19 codes are 1..18; 1001..1018 (Render env override)
    return "Query is incomplete" -> age empty on all 89 deals.
  * TS003 household: the old code list mixed group codes (1001..1007) with the
    leaf codes inside them, so values were counted 2-3 times and every pct was
    wrong (E01005307: rows summed to 1690 vs 716 households).

These tests run the REAL app.py config defaults and the REAL get_nomis_table /
parse_jsonstat_single_dimension / _normalize_census_items code (extracted with
ast, no network) against responses captured from Nomis for geography
E01005307 (tests/fixtures/nomis_census_E01005307.json). The fake fetch answers
ONLY queries that were captured as valid; any other dim/cats combination
raises exactly like Nomis did, so a wrong config cannot pass.
"""
import ast
import json
import os
import re
import time
from typing import Any, Dict, Optional

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
FX = json.load(open(os.path.join(ROOT, "tests", "fixtures", "nomis_census_E01005307.json"), encoding="utf-8"))
GEO = FX["geography"]
TOTALS = FX["nomis_totals_code0"]

TABLES = {  # key -> (dataset const, dim const, cats const, Nomis total key)
    "ethnic":    ("NOMIS_TS021_DATASET", "NOMIS_TS021_DIM", "NOMIS_TS021_CATS", "ethnic"),
    "religion":  ("NOMIS_TS030_DATASET", "NOMIS_TS030_DIM", "NOMIS_TS030_CATS", "religion"),
    "age":       ("NOMIS_TS007_DATASET", "NOMIS_TS007_DIM", "NOMIS_TS007_CATS", "age"),
    "household": ("NOMIS_TS003_DATASET", "NOMIS_TS003_DIM", "NOMIS_TS003_CATS", "household"),
}


class _NoEnvOS:
    """os stand-in so os.getenv returns the CODE default (env overrides on
    Render are a separate, documented operator step)."""
    @staticmethod
    def getenv(_key, default=None):
        return default


def _config() -> Dict[str, str]:
    """Evaluate app.py's module-level NOMIS_* assignments with no env set."""
    tree = ast.parse(APP)
    ns: Dict[str, Any] = {"os": _NoEnvOS}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id.startswith("NOMIS_"):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "app.py", "exec"), ns)
    return {k: v for k, v in ns.items() if k.startswith("NOMIS_")}


def _fake_fetch(dataset_id: str, params: dict) -> dict:
    dims = [k for k in params if k not in {"date", "geography", "freq", "measures"}]
    assert len(dims) == 1 and params["geography"] == GEO
    key = f"{dataset_id}|{dims[0]}={params[dims[0]]}"
    if key in FX["responses"]:
        return {"dataset": FX["responses"][key]}
    # Same message shape as app.fetch_nomis_jsonstat when Nomis rejects a query.
    msg = FX["nomis_errors"].get(key, "query not captured as valid against Nomis")
    raise ValueError(f"Nomis error for dataset {dataset_id}: {msg}")


def _app_ns() -> Dict[str, Any]:
    tree = ast.parse(APP)
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    cfg = _config()
    ns: Dict[str, Any] = {"re": re, "time": time, "Any": Any, "Dict": Dict,
                          "Optional": Optional, "List": list, **cfg}
    for name in ("safe_int", "_first_source_url", "now_iso", "is_digits_only",
                 "metric_ok", "metric_unavailable", "metric_missing_provider",
                 "parse_jsonstat_single_dimension", "_normalize_census_items",
                 "get_nomis_table"):
        assert name in fns, f"{name} missing from app.py"
        exec(compile(ast.get_source_segment(APP, fns[name]), "app.py", "exec"), ns)
    ns["fetch_nomis_jsonstat"] = _fake_fetch
    return ns


def _run(key: str, ns=None, cats_override: Optional[str] = None):
    ns = ns or _app_ns()
    ds_c, dim_c, cats_c, _ = TABLES[key]
    t = ns["get_nomis_table"](key, ns[dim_c], cats_override or ns[cats_c], GEO, ns[ds_c])
    return t, ns["_normalize_census_items"](t)


# ── config is a query Nomis actually answers ────────────────────────────────
@pytest.mark.parametrize("key", list(TABLES))
def test_config_default_is_a_valid_nomis_query(key):
    t, items = _run(key)
    assert t["status"] == "ok", t["summary"]
    assert items, f"{key} produced no rows"


def test_ethnic_uses_the_only_dimension_nomis_exposes():
    cfg = _config()
    assert cfg["NOMIS_TS021_DATASET"] == "NM_2041_1"
    assert cfg["NOMIS_TS021_DIM"] == "c2021_eth_20"
    assert cfg["NOMIS_TS021_DIM"] not in ("c2021_eth_8", "c2021_eth_25")


def test_age_uses_leaf_codes_not_1001_range():
    cats = _config()["NOMIS_TS007_CATS"]
    assert cats == ",".join(str(i) for i in range(1, 19))
    assert "1001" not in cats and ".." not in cats


# ── no double counting: rows must partition the Nomis total ─────────────────
@pytest.mark.parametrize("key", list(TABLES))
def test_rows_sum_to_nomis_total(key):
    t, items = _run(key)
    assert sum(i["value"] for i in items) == TOTALS[TABLES[key][3]]
    assert abs(sum(i["pct"] for i in items) - 100) <= 0.5


def test_household_percentages_are_true_shares():
    _, items = _run("household")
    by = {i["label"]: i for i in items}
    assert set(by) == {"One-person household", "Single family household", "Other household types"}
    # 330 / 716 households in E01005307 (Nomis code 0 total)
    assert by["Single family household"]["pct"] == round(330 / 716 * 100, 1) == 46.1
    assert items[0]["label"] == "Single family household"


def test_old_household_list_double_counted():
    """Documents the defect: the pre-fix code list inflates the denominator."""
    old = "1001,1,2,1002,1003,4,5,6,1004,7,8,9,1005,10,11,1006,12,1007,13,14"
    _, items = _run("household", cats_override=old)
    assert sum(i["value"] for i in items) == 1690 != TOTALS["household"]


def test_ethnic_broad_groups():
    _, items = _run("ethnic")
    labels = [i["label"] for i in items]
    assert labels[0] == "White"
    assert set(labels) == {"White", "Asian, Asian British or Asian Welsh",
                           "Black, Black British, Black Welsh, Caribbean or African",
                           "Mixed or Multiple ethnic groups", "Other ethnic group"}


def test_age_bands():
    _, items = _run("age")
    assert len(items) == 18
    assert items[0]["label"] == "Aged 20 to 24 years" and items[0]["value"] == 494


@pytest.mark.parametrize("dim,cats,dataset", [
    ("c2021_eth_8", "1,2,3,4,5,6,7,8", "NM_2041_1"),
    ("c2021_eth_25", ",".join(str(i) for i in range(1, 25)), "NM_2041_1"),
    ("c2021_age_19", ",".join(str(1000 + i) for i in range(1, 19)), "NM_2020_1"),
])
def test_old_configs_are_rejected_like_nomis_did(dim, cats, dataset):
    ns = _app_ns()
    t = ns["get_nomis_table"]("x", dim, cats, GEO, dataset)
    assert t["status"] == "unavailable"
    assert ns["_normalize_census_items"](t) == []
