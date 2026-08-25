"""
scotland_crime_payload.py
=========================
Pure assembly of a Scotland crime payload from rows already fetched out of
`scotland.crime_by_area`. No network, no DB, no app imports — so it is unit
testable in isolation and cannot break app boot.

The output `metrics`/`value` shape is designed to slot into the SAME contract the
England/Wales `get_crime_data` produces (status/summary/value/metrics/...), so the
frontend and `build_area_inference` can read `crime.metrics.total` and
`crime.metrics.categories` unchanged — while ADDING Scottish-only fields
(period, geography, native categories, comparison grades, source, notice).

Governance:
  * Native categories preserved verbatim.
  * Missing data -> this returns None so the caller emits an honest "unavailable".
    We NEVER manufacture a zero-crime result. A count of 0 is only ever emitted
    when the source row genuinely carries 0.
  * `not_comparable` categories are flagged; the notice warns comparisons are
    approximate. We never fold `not_comparable` into a UK-wide index.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional


def _as_int(v: Any) -> Optional[int]:
    try:
        if v is None or v == "":
            return None
        return int(round(float(v)))
    except (TypeError, ValueError):
        return None


def _notice_for_source(source: str, geography_type: str) -> str:
    base = ("Scottish crime statistics use different classifications and geographic "
            "units from the Police.uk data used for England and Wales. Where indicated, "
            "comparisons are approximate; categories marked not comparable are shown in "
            "Scottish terms only.")
    s = (source or "").lower()
    if "police scotland" in s or geography_type == "multi_member_ward":
        base += (" Ward-level figures are Police Scotland management information "
                 "(provisional) and are not accredited official statistics.")
    elif geography_type == "local_authority":
        base += (" Local-authority figures are Scottish Government Accredited Official "
                 "Statistics.")
    return base


def assemble_scotland_crime(geography: Dict[str, Any],
                            rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build the Scotland crime payload, or return None if there is no real data.

    geography: {type, code, name, council, period, source}
    rows: each {native_group, native_category, display_category, comparison_quality,
                recorded_count, detected_count, rate_per_10000?}

    Returns a dict with keys: jurisdiction, summary, value, metrics
    (matching the app's crime-metric contract, plus Scottish extras), or None.
    """
    if not rows:
        return None

    # Guard: if EVERY recorded_count is unknown (None), we have no real signal.
    any_real = any(_as_int(r.get("recorded_count")) is not None for r in rows)
    if not any_real:
        return None

    gtype = str(geography.get("type") or "").strip() or "multi_member_ward"
    gname = str(geography.get("name") or "").strip() or "Unknown area"
    period = str(geography.get("period") or "").strip() or "unknown period"
    source = str(geography.get("source") or "").strip() or "Police Scotland"

    categories: Dict[str, int] = {}
    detected: Dict[str, int] = {}
    comparison: List[Dict[str, Any]] = []
    total = 0
    not_comparable_present = False

    for r in rows:
        native_cat = str(r.get("native_category") or r.get("native_group") or "").strip()
        if not native_cat:
            continue
        rec = _as_int(r.get("recorded_count"))
        det = _as_int(r.get("detected_count"))
        display = str(r.get("display_category") or "").strip() or native_cat
        quality = str(r.get("comparison_quality") or "not_comparable").strip() or "not_comparable"
        if quality == "not_comparable":
            not_comparable_present = True

        if rec is not None:
            categories[native_cat] = categories.get(native_cat, 0) + rec
            total += rec
        if det is not None:
            detected[native_cat] = detected.get(native_cat, 0) + det

        comparison.append({
            "native_group":       str(r.get("native_group") or "").strip() or None,
            "native_category":    native_cat,
            "display_category":   display,
            "comparison_quality": quality,
            "recorded":           rec,
            "detected":           det,
        })

    # Everything summed to nothing AND no categories -> no usable data.
    if not categories:
        return None

    geo_label = {
        "multi_member_ward": "Multi-Member Ward",
        "local_authority":   "Local Authority",
    }.get(gtype, gtype)

    summary = (f"{total} recorded crimes across {len(categories)} categories — "
               f"{gname} ({geo_label}), {period}, {source}.")

    metrics = {
        # --- shared contract keys the E&W path also provides ---
        "total":      total,
        "categories": categories,          # native categories -> recorded counts
        # --- Scotland-specific additions ---
        "detected":            detected,
        "period":              period,
        "geography_type":      gtype,
        "geography_name":      gname,
        "geography_code":      geography.get("code"),
        "council":             geography.get("council"),
        "source":              source,
        "comparison":          comparison,
        "not_comparable_present": not_comparable_present,
        "notice":              _notice_for_source(source, gtype),
        # explicit flag so build_area_inference / FE never apply the E&W crime_index
        "jurisdiction":        "scotland",
    }

    return {
        "jurisdiction": "scotland",
        "summary":      summary,
        "value":        comparison,   # the per-category rows are the "records"
        "metrics":      metrics,
    }
