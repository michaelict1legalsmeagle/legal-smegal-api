"""
Stale-processing detection (S-AUDIT-3, fixed by STALE-TS-1 on 7 Oct 2026).

get_deal flips a deal stuck in status='processing' to an honest error once
updated_at is older than STALE_PROCESSING_SECONDS. The previous version of
this file COPIED the parse logic and fed it only "Z" timestamps, so it could
not fail — while the real code raised on every Supabase "+00:00" whole-second
timestamp (104 of 106 live deals) and the check never fired.

These tests call the REAL helper (processing_staleness) with the formats
Supabase actually returns, and lock that app.py's get_deal uses it.
"""
import ast
import os
import sys
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import processing_staleness as ps  # noqa: E402

NOW = datetime(2026, 10, 7, 12, 0, 0, tzinfo=timezone.utc)


def _ago(seconds):
    return NOW - timedelta(seconds=seconds)


# ── formats Supabase / our writers actually produce ──────────────────────────
def test_supabase_whole_second_offset():
    # Exact live format (deals.updated_at, 7 Oct): "2026-10-06T19:14:08+00:00"
    ts = _ago(600).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    assert ps.seconds_since(ts, now=NOW) == 600


def test_supabase_microseconds_offset():
    ts = _ago(600).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
    assert ps.seconds_since(ts, now=NOW) == 600


def test_supabase_five_digit_fraction():
    # Postgres drops trailing zeros: .123450 -> .12345
    ts = "2026-10-07T11:50:00.12345+00:00"
    assert abs(ps.seconds_since(ts, now=NOW) - 599.87655) < 1e-6


def test_now_iso_z_format():
    # app.now_iso() writes "%Y-%m-%dT%H:%M:%SZ"
    ts = _ago(600).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert ps.seconds_since(ts, now=NOW) == 600


def test_space_separator_and_non_utc_offset():
    assert ps.seconds_since("2026-10-07 11:50:00+00:00", now=NOW) == 600
    # 12:50 BST == 11:50 UTC
    assert ps.seconds_since("2026-10-07T12:50:00+01:00", now=NOW) == 600


def test_naive_string_is_treated_as_utc():
    assert ps.seconds_since("2026-10-07T11:50:00", now=NOW) == 600


def test_unreadable_returns_none_not_a_guess():
    for bad in (None, "", "   ", "not a date", "07/10/2026 11:50"):
        assert ps.seconds_since(bad, now=NOW) is None


def test_parse_returns_aware_utc():
    dt = ps.parse_db_timestamp("2026-10-07T12:50:00+01:00")
    assert dt.tzinfo is not None and dt.utcoffset() == timedelta(0)
    assert dt == datetime(2026, 10, 7, 11, 50, tzinfo=timezone.utc)


# ── threshold ────────────────────────────────────────────────────────────────
def test_threshold_is_300_seconds():
    assert ps.STALE_PROCESSING_SECONDS == 300


def test_live_job_heartbeating_is_never_stale():
    # The analysis thread refreshes updated_at every 45s; real runs 60-180s.
    for age in (10, 45, 60, 90, 120, 180, 295):
        ts = _ago(age).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        assert ps.seconds_since(ts, now=NOW) < ps.STALE_PROCESSING_SECONDS


def test_dead_job_is_stale_in_every_format():
    for fmt in ("%Y-%m-%dT%H:%M:%S+00:00", "%Y-%m-%dT%H:%M:%S.%f+00:00",
                "%Y-%m-%dT%H:%M:%SZ"):
        ts = _ago(ps.STALE_PROCESSING_SECONDS + 1).strftime(fmt)
        assert ps.seconds_since(ts, now=NOW) > ps.STALE_PROCESSING_SECONDS, fmt


# ── lock: app.py's get_deal must use the real helper ─────────────────────────
def _get_deal_src():
    src = open(os.path.join(HERE, "app.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_deal":
            return src, ast.get_source_segment(src, node)
    raise AssertionError("get_deal not found in app.py")


def test_app_imports_helper_at_module_level():
    src, _ = _get_deal_src()
    assert "\nfrom processing_staleness import seconds_since as _ps_seconds_since" in src


def test_get_deal_uses_helper_not_strptime():
    _, body = _get_deal_src()
    assert "_ps_seconds_since(" in body
    assert "_STALE_PROCESSING_SECONDS" in body
    assert "strptime(" not in body
    assert "utcnow(" not in body
