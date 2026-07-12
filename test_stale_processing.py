"""
Regression test for stale-processing-job detection (S-AUDIT-3).

Background analysis/area/guest-report work runs in daemon threads with no
persistence or retry (see /areas notes on gunicorn max_requests recycling).
If the worker process recycles mid-run, a deal is left with
status='processing' forever and the frontend polls indefinitely.

get_deal() now detects a deal that has been 'processing' for longer than
_STALE_PROCESSING_SECONDS and flips it to an honest error state. This test
locks in the age-calculation and threshold logic in isolation, since the
full route requires heavy Supabase mocking to exercise end-to-end.
"""
from datetime import datetime, timedelta

STALE_PROCESSING_SECONDS = 300  # must match app.py's _STALE_PROCESSING_SECONDS


def _age_seconds(updated_at_iso: str) -> float:
    """Mirrors the exact parsing logic used in get_deal()."""
    updated_dt = datetime.strptime(
        updated_at_iso.split(".")[0].replace("Z", ""), "%Y-%m-%dT%H:%M:%S"
    )
    return (datetime.utcnow() - updated_dt).total_seconds()


def _now_iso_style(delta_seconds: float, with_micros: bool = False) -> str:
    """Mirrors now_iso()'s format (time.strftime with gmtime), optionally
    with microseconds to prove the parser handles both — Supabase can
    return either depending on the write path."""
    ts = datetime.utcnow() - timedelta(seconds=delta_seconds)
    fmt = "%Y-%m-%dT%H:%M:%S.%fZ" if with_micros else "%Y-%m-%dT%H:%M:%SZ"
    return ts.strftime(fmt)


class TestStaleProcessingDetection:
    def test_freshly_started_job_is_not_stale(self):
        ts = _now_iso_style(10)
        assert _age_seconds(ts) < STALE_PROCESSING_SECONDS

    def test_job_just_under_threshold_is_not_stale(self):
        ts = _now_iso_style(STALE_PROCESSING_SECONDS - 5)
        assert _age_seconds(ts) < STALE_PROCESSING_SECONDS

    def test_job_over_threshold_is_stale(self):
        ts = _now_iso_style(STALE_PROCESSING_SECONDS + 100)
        assert _age_seconds(ts) > STALE_PROCESSING_SECONDS

    def test_typical_analysis_duration_is_not_flagged(self):
        # Docs say real analysis takes ~60-120s — must never be falsely
        # flagged as stale mid-run.
        for real_duration in (60, 90, 120, 180):
            ts = _now_iso_style(real_duration)
            assert _age_seconds(ts) < STALE_PROCESSING_SECONDS, (
                f"{real_duration}s should not be flagged stale"
            )

    def test_handles_timestamp_with_microseconds(self):
        ts = _now_iso_style(STALE_PROCESSING_SECONDS + 50, with_micros=True)
        assert _age_seconds(ts) > STALE_PROCESSING_SECONDS

    def test_handles_timestamp_without_microseconds(self):
        ts = _now_iso_style(STALE_PROCESSING_SECONDS + 50, with_micros=False)
        assert _age_seconds(ts) > STALE_PROCESSING_SECONDS
