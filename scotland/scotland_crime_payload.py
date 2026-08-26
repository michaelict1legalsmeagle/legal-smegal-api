"""
scotland_crime_payload.py
=========================
Pure assembly of a Scotland crime payload from rows already fetched out of
`scotland.crime_by_area`. No network, no DB, no app imports — so it is unit
testable in isolation and cannot break app boot.

The output `metrics`/`value` shape slots into the SAME contract the England/Wales
`get_crime_data` produces (status/summary/value/metrics/...), so the frontend and
`build_area_inference` read `crime.metrics.total` and `crime.metrics.categories`
unchanged — while ADDING Scottish-only fields (period, geography, native
categories, comparison grades, source, notice, and the crime/offence split).

GROUP SPLIT (the correctness rule this module enforces):
  Scottish recorded-crime statistics are published in 8 groups. Groups 1-5 are
  CRIMES; groups 6-8 (antisocial, miscellaneous, road-traffic) are OFFENCES.
  The provider serves ALL groups from the ward table. The HEADLINE crime total
  MUST count crimes only (1-5). Offences are surfaced SEPARATELY as
  `total_offences` / `offence_categories` and are NEVER folded into `total`.
  Verified against real data (Rutherglen South, 2025-26): crimes=483, offences=275.

Governance:
  * Native categories preserved verbatim.
  * Missing data -> return None so the caller emits an honest "unavailable".
    A count of 0 is only ever emitted when the source row genuinely carries 0;
    we NEVER manufacture a zero-crime result.
  * `not_comparable` categories are flagged; the notice warns comparisons are
    approximate. We never fold `not_comparable` into a UK-wide index.
  * A group name we cannot positively classify is treated as UNCLASSIFIED and is
    kept out of BOTH totals (never silently added to the crime headline), surfaced
    separately, and marked not_comparable.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple


def _as_int(v: Any) -> Optional[int]:
    try:
        if v is None or v == "":
            return None
        return int(round(float(v)))
    except (TypeError, ValueError):
        return None


# --- Scottish recorded-crime group classification -------------------------------
# Groups 1-5 = crimes, 6-8 = offences. Rules run in order; first match wins.
# The string is normalised (lowercased, hyphens->spaces, '&'->'and', collapsed)
# BEFORE matching, so "Non-sexual crimes of violence" is caught before "sexual".
_GROUP_RULES: List[Tuple[str, int, str]] = [
    ("non sexual",    1, "crime"),     # Non-sexual crimes of violence
    ("sexual",        2, "crime"),     # Sexual crimes
    ("dishonesty",    3, "crime"),     # Crimes of dishonesty
    ("damage",        4, "crime"),     # Damage and reckless behaviour
    ("reckless",      4, "crime"),
    ("society",       5, "crime"),     # Crimes against society
    ("antisocial",    6, "offence"),   # Antisocial offences
    ("anti social",   6, "offence"),
    ("miscellaneous", 7, "offence"),   # Miscellaneous offences
    ("traffic",       8, "offence"),   # Road traffic offences
]

# Canonical display label per group number (used for by_group ordering/labels).
_GROUP_LABEL = {
    1: "Non-sexual crimes of violence",
    2: "Sexual crimes",
    3: "Crimes of dishonesty",
    4: "Damage and reckless behaviour",
    5: "Crimes against society",
    6: "Antisocial offences",
    7: "Miscellaneous offences",
    8: "Road traffic offences",
}


def _normalise_group(name: str) -> str:
    s = (name or "").lower().replace("-", " ").replace("&", "and")
    return " ".join(s.split())


def classify_group(native_group: str) -> Tuple[Optional[int], str]:
    """Return (group_number 1-8 | None, kind) where kind in {crime, offence, unknown}.

    Accepts either a numeric group ("1".."8") or the descriptive group name.
    """
    raw = (native_group or "").strip()
    if raw.isdigit():
        n = int(raw)
        if 1 <= n <= 5:
            return n, "crime"
        if 6 <= n <= 8:
            return n, "offence"
        return None, "unknown"
    g = _normalise_group(raw)
    if not g:
        return None, "unknown"
    for token, num, kind in _GROUP_RULES:
        if token in g:
            return num, kind
    return None, "unknown"


def _notice_for_source(source: str, geography_type: str) -> str:
    base = ("Scottish crime statistics use different classifications and geographic "
            "units from the Police.uk data used for England and Wales. Where indicated, "
            "comparisons are approximate; categories marked not comparable are shown in "
            "Scottish terms only. Recorded crimes (groups 1-5) are counted in the crime "
            "total; offences (antisocial, miscellaneous and road-traffic) are reported "
            "separately and are not included in the crime total.")
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

    # CRIMES (groups 1-5) — the headline.
    categories: Dict[str, int] = {}          # native crime category -> recorded
    detected: Dict[str, int] = {}            # native crime category -> detected
    by_group: Dict[str, int] = {}            # crime group label -> recorded
    total = 0

    # OFFENCES (groups 6-8) — surfaced separately, never in `total`.
    offence_categories: Dict[str, int] = {}
    offence_by_group: Dict[str, int] = {}
    total_offences = 0

    # UNCLASSIFIED — kept out of both totals; surfaced honestly.
    unclassified: Dict[str, int] = {}

    comparison: List[Dict[str, Any]] = []
    not_comparable_present = False

    for r in rows:
        native_cat = str(r.get("native_category") or r.get("native_group") or "").strip()
        if not native_cat:
            continue
        native_grp = str(r.get("native_group") or "").strip()
        grp_num, kind = classify_group(native_grp or native_cat)
        grp_label = _GROUP_LABEL.get(grp_num) if grp_num else (native_grp or None)

        rec = _as_int(r.get("recorded_count"))
        det = _as_int(r.get("detected_count"))
        display = str(r.get("display_category") or "").strip() or native_cat
        quality = str(r.get("comparison_quality") or "not_comparable").strip() or "not_comparable"
        if quality == "not_comparable":
            not_comparable_present = True

        if kind == "crime":
            if rec is not None:
                categories[native_cat] = categories.get(native_cat, 0) + rec
                total += rec
                if grp_label:
                    by_group[grp_label] = by_group.get(grp_label, 0) + rec
            if det is not None:
                detected[native_cat] = detected.get(native_cat, 0) + det
        elif kind == "offence":
            if rec is not None:
                offence_categories[native_cat] = offence_categories.get(native_cat, 0) + rec
                total_offences += rec
                if grp_label:
                    offence_by_group[grp_label] = offence_by_group.get(grp_label, 0) + rec
        else:  # unknown — never counted in either total
            not_comparable_present = True
            if rec is not None:
                unclassified[native_cat] = unclassified.get(native_cat, 0) + rec

        comparison.append({
            "native_group":       native_grp or None,
            "group_number":       grp_num,
            "kind":               kind,           # crime | offence | unknown
            "is_offence":         kind == "offence",
            "native_category":    native_cat,
            "display_category":   display,
            "comparison_quality": quality,
            "recorded":           rec,
            "detected":           det,
        })

    # No usable CRIME data at all -> honest unavailable (do not let an offence-only
    # or unclassified-only ward masquerade as a crime figure).
    has_real_crime_row = any(
        c["kind"] == "crime" and c["recorded"] is not None for c in comparison)
    if not has_real_crime_row:
        return None

    geo_label = {
        "multi_member_ward": "Multi-Member Ward",
        "local_authority":   "Local Authority",
    }.get(gtype, gtype)

    off_note = ""
    if total_offences:
        off_note = (f" A further {total_offences} offences (antisocial, miscellaneous "
                    f"and road-traffic) are reported separately.")
    summary = (f"{total} recorded crimes across {len(categories)} categories — "
               f"{gname} ({geo_label}), {period}, {source}.{off_note}")

    metrics = {
        # --- shared contract keys the E&W path also provides ---
        "total":      total,               # CRIMES ONLY (groups 1-5) — the headline
        "categories": categories,          # native CRIME categories -> recorded counts
        # --- crime/offence split (the fix) ---
        "by_group":            by_group,              # crime group -> recorded
        "total_offences":      total_offences,        # groups 6-8, separate from `total`
        "offence_categories":  offence_categories,    # native offence categories -> recorded
        "offence_by_group":    offence_by_group,      # offence group -> recorded
        "unclassified":        unclassified,          # never in either total
        # --- Scotland-specific additions ---
        "detected":            detected,              # crimes detected (native category)
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
