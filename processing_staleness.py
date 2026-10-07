"""Stale-processing detection for get_deal (STALE-TS-1, 7 Oct 2026).

A deal left in status='processing' by a background job that died (worker
restart, deploy, OOM) must be flipped to an honest error. get_deal does that
when updated_at is older than STALE_PROCESSING_SECONDS. The analysis thread
heartbeats updated_at every 45s, so a slow-but-alive job never reaches it.

Root cause fixed here: the old parse in get_deal used
strptime(ts.split(".")[0].replace("Z", ""), "%Y-%m-%dT%H:%M:%S"). Supabase
returns timestamptz as "2026-10-06T19:41:00+00:00"; for whole-second values
(104 of 106 live deals on 7 Oct) the "+00:00" survived and strptime raised
"unconverted data remains", which get_deal swallowed — so a dead job was
never flipped. This module parses every format Supabase returns, timezone-aware.
"""
from datetime import datetime, timezone
from typing import Optional

STALE_PROCESSING_SECONDS = 300  # real analysis ~60-120s; heartbeat every 45s


def parse_db_timestamp(value) -> Optional[datetime]:
    """Return a timezone-aware UTC datetime, or None if unreadable.

    Accepts: "...+00:00" (Supabase timestamptz), any fractional-second length,
    a trailing "Z" (now_iso()), a space separator, other offsets, a naive
    string (treated as UTC), or a datetime object.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip()
        if not s:
            return None
        if s.endswith(("Z", "z")):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def seconds_since(value, now: Optional[datetime] = None) -> Optional[float]:
    """Seconds between `value` and now (UTC). None if `value` is unreadable."""
    dt = parse_db_timestamp(value)
    if dt is None:
        return None
    ref = now if now is not None else datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    return (ref - dt).total_seconds()
