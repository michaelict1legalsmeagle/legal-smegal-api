"""
scotland_crime_mapping.py
=========================
Configurable mapping from native Scottish crime classifications to a LegalSmegal
`display_category` plus a `comparison_quality`, per the `scotland-crime-rates-for-area`
skill's mapping table.

Design rules (from the skill + core governance):
  * Native Scottish categories are ALWAYS preserved verbatim by the caller. This
    module only ADDS a display label + a comparability grade; it never rewrites
    the native value.
  * The mapping is data-driven (the RULES list below), not hard-coded through the
    app. To adjust a mapping, edit one row here.
  * When a native category cannot be confidently mapped, the result is
    ("<native as-is>", "not_comparable"). We NEVER assert comparability we are not
    sure of — an unmapped Scottish category is shown in Scottish terms only.

comparison_quality is one of: "very_good" | "good" | "approximate" | "not_comparable".

This module is pure (no imports from app.py, no network, no DB) so it can be unit
tested in isolation and imported by both the importer (load-time tagging) and the
provider.
"""

from __future__ import annotations
from typing import List, Tuple

# Each rule: (list_of_lowercase_keyword_phrases, display_category, comparison_quality)
# Matched against the lowercased "native_group | native_category" text.
# ORDER MATTERS — more specific rules first. First match wins.
RULES: List[Tuple[List[str], str, str]] = [
    # --- very good (near like-for-like) ---
    (["shoplifting"],                              "Shoplifting",                "very_good"),
    (["robbery"],                                  "Robbery",                    "very_good"),

    # --- vehicle: pedal cycle BEFORE motor vehicle (both contain "theft of") ---
    (["pedal cycle", "pedal cycles", "bicycle", "cycle theft"],
                                                   "Bicycle theft",             "approximate"),
    (["theft of a motor vehicle", "theft of motor vehicle",
      "theft from a motor vehicle", "theft from motor vehicle",
      "theft of/from motor vehicle", "motor vehicle"],
                                                   "Vehicle crime",             "good"),

    # --- housebreaking (Scotland's nearest to burglary) ---
    (["housebreaking"],                            "Burglary / Housebreaking",  "good"),

    # --- drugs ---
    (["drug"],                                     "Drugs",                     "good"),

    # --- weapons ---
    (["weapon", "bladed", "offensive weapon", "carrying"],
                                                   "Weapons",                   "good"),

    # --- damage / arson ---
    (["fire-raising", "fireraising", "fire raising"],
                                                   "Criminal damage / arson",   "approximate"),
    (["vandalism", "malicious mischief", "reckless"],
                                                   "Criminal damage / arson",   "approximate"),

    # --- sexual ---
    (["sexual", "rape", "indecent"],               "Violence & sexual offences","good"),

    # --- violence (non-sexual) ---
    (["non-sexual crimes of violence", "murder", "homicide", "culpable homicide",
      "attempted murder", "serious assault", "common assault", "assault",
      "violence", "threats and extortion"],
                                                   "Violence & sexual offences","good"),

    # --- theft variants ---
    (["theft from the person", "theft from a person", "pickpocket"],
                                                   "Theft from person",         "approximate"),
    (["other theft", "theft by", "theft (other)", "other dishonesty"],
                                                   "Other theft",               "approximate"),

    # --- public order / antisocial: NOT safely comparable ---
    (["antisocial", "anti-social", "threatening or abusive", "breach of the peace",
      "public order", "racially aggravated", "conduct"],
                                                   "Public order / ASB",        "not_comparable"),
]


def _norm(*parts: str) -> str:
    return " | ".join((p or "").strip().lower() for p in parts if p is not None)


def map_scotland_category(native_group: str, native_category: str) -> Tuple[str, str]:
    """Return (display_category, comparison_quality) for a native Scottish
    (group, category). Unmatched -> (native_category or native_group, 'not_comparable').

    The caller keeps the native value regardless; this is only the display/comparability
    overlay. Never raises.
    """
    hay = _norm(native_group, native_category)
    for keywords, display, quality in RULES:
        for kw in keywords:
            if kw in hay:
                return display, quality
    # No confident mapping — show it in Scottish terms only, not comparable.
    fallback = (native_category or native_group or "").strip() or "Unclassified"
    return fallback, "not_comparable"


# Convenience for callers that only have a single label string.
def map_label(label: str) -> Tuple[str, str]:
    return map_scotland_category("", label)
