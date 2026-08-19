"""
Regression tests for the Scotland band engine.

These lock in the verification audit as repeatable checks. Point the env vars at the
reference data + a sample Home Report dir to run the data-backed tests; they skip
cleanly when the data is absent (so the pure-logic tests still run in CI).

    export LS_SCOT_REF=/path/to/reference        # SmallUser.csv, residential-...csv, ros_...xlsx
    export LS_SCOT_HR=/path/to/home_reports       # *.pdf  (optional)
    python -m pytest test_scotland_band_engine.py -v
"""
import os, tempfile
from pathlib import Path
import pytest
import scotland_band_engine as e

REF_DIR = os.environ.get("LS_SCOT_REF")
HR_DIR  = os.environ.get("LS_SCOT_HR")
have_ref = REF_DIR and Path(REF_DIR, "SmallUser.csv").exists()
needs_ref = pytest.mark.skipif(not have_ref, reason="LS_SCOT_REF not set / data absent")


# ---- pure logic (no data required) ------------------------------------------

def test_simd_decile_bounds():
    assert e.simd_decile(1) == 1            # most deprived
    assert e.simd_decile(e.N_DZ2011) == 10  # least deprived
    assert e.simd_decile(None) is None

def test_bad_pdf_never_raises():
    a = e.extract_home_report("/no/such/file.pdf")
    assert a["error"] and a["market_value"] is None      # graceful, no guess
    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        a = e.extract_home_report(f.name)
        assert a["error"] and a["market_value"] is None

def test_attribution_present():
    assert len(e.ATTRIBUTION) >= 3


# ---- data-backed (verified facts locked as regression) ----------------------

@pytest.fixture(scope="module")
def ref():
    return e.load_reference(REF_DIR)

@needs_ref
def test_reference_shape(ref):
    assert len(ref["pc_index"]) > 150_000            # live postcodes
    assert ref["ros_period"] == "June 2026"          # latest RoS month parsed

@needs_ref
def test_price_datazones_are_2011_vintage(ref):
    dz = [c for c in ref["price_2023"] if c.startswith("S01")]
    assert dz and all(c <= e.DZ2011_MAX for c in dz)  # no 2022 codes leaked in

@needs_ref
def test_la_codes_consistent_across_sources(ref):
    """The Glasgow / N.Lanarkshire post-2019 code trap must not bite."""
    la_pc = {v["la_code"] for v in ref["pc_index"].values()}
    la_price = {c for c in ref["price_2023"] if c.startswith("S12")}
    assert la_pc - la_price == set()                 # every council joins to price
    assert la_pc - set(ref["ros_cur"]) == set()      # and to RoS
    assert "S12000049" in la_price and "S12000049" in ref["ros_cur"]  # Glasgow (new code)

@needs_ref
def test_unknown_postcode_is_unavailable_not_a_number(ref):
    b = e.build_band("ZZ99ZZ", ref)
    assert b["status"] == "unavailable"              # never fabricates a value

@needs_ref
def test_iz_fallback_fires_on_suppressed_datazone(ref):
    hit = next((pc for pc, g in ref["pc_index"].items()
                if g["dz2011"] not in ref["price_2023"] and g["iz2011"] in ref["price_2023"]), None)
    assert hit, "expected at least one suppressed-DZ postcode"
    b = e.build_band(hit, ref)
    assert b["geography_level"].startswith("Intermediate")
    assert b["tier_note"]                            # labelled, not silent

@needs_ref
def test_time_adjust_uses_same_source_factor(ref):
    # pick any LA present in both RoS current and 2023 -> factor is a positive ratio
    la = next(iter(set(ref["ros_cur"]) & set(ref["ros_2023"])))
    assert ref["ros_cur"][la] > 0 and ref["ros_2023"][la] > 0


@pytest.mark.skipif(not (have_ref and HR_DIR and any(Path(HR_DIR).glob("*.pdf"))),
                    reason="no sample Home Reports")
def test_end_to_end_on_sample_pack(ref):
    pdf = next(Path(HR_DIR).glob("*.pdf"))
    r = e.assess_lot(pdf, ref)
    assert "anchor" in r and "band" in r
    # every band figure that exists must carry a source label
    if r["band"]["status"] == "ok":
        assert r["band"]["band_source"]


# ---- Postgres-backed reference (gated on LS_SCOT_DSN) ------------------------

DSN = os.environ.get("LS_SCOT_DSN")
needs_pg = pytest.mark.skipif(not DSN, reason="LS_SCOT_DSN not set")

@needs_pg
def test_pg_reference_shape_and_guards():
    from scotland_reference_pg import load_reference_pg
    P = load_reference_pg(DSN)
    assert len(P["pc_index"]) > 150_000
    assert P["ros_period"] == "June 2026"
    assert e.build_band("ZZ99ZZ", P)["status"] == "unavailable"   # never a number

@needs_pg
@needs_ref
def test_pg_backed_equals_file_backed(ref):
    """DB path must return identical bands to the verified file path."""
    from scotland_reference_pg import load_reference_pg
    P = load_reference_pg(DSN)
    for pc in ("AB101AS", "G331FD", "KA244AA"):
        bf, bp = e.build_band(pc, ref), e.build_band(pc, P)
        assert bf.get("band_median_2023") == bp.get("band_median_2023")
        assert bf.get("geography_level") == bp.get("geography_level")
