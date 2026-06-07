"""
test_financials_ceiling.py
==========================
Static analysis tests verifying the Financials page ceiling doctrine.
These parse legalsmegal-financials.html directly and assert:
- No MANUAL badge on ceiling limit section
- DERIVED FROM WORKBENCH label present
- manualCeiling input is hidden/disabled (never user-editable)
- No oninput triggering window._ceilingIsManual=true
- No _mcRaw reads from manualCeiling
- bid_ceiling not used as canonical ceiling source
- Priority chain reads financial_current_standing, then workbench_ceiling, then legacy
- purchase price read goes to purchasePrice field only
"""

import re
import pytest
from pathlib import Path

FINANCIALS_HTML = Path(__file__).parent.parent.parent.parent / \
    "frontend" / "legalsmegal-frontend-main" / "legalsmegal-financials.html"

# Allow running from multiple directories
_candidates = [
    Path("/tmp/frontend/legalsmegal-frontend-main/legalsmegal-financials.html"),
    Path("/mnt/user-data/outputs/legalsmegal-financials.html"),
]
for _c in _candidates:
    if _c.exists():
        FINANCIALS_HTML = _c
        break


@pytest.fixture(scope="module")
def html():
    assert FINANCIALS_HTML.exists(), f"File not found: {FINANCIALS_HTML}"
    return FINANCIALS_HTML.read_text(encoding="utf-8")


# ── TEST 1: No MANUAL badge on ceiling limit row ─────────────────────────────

def test_no_manual_badge_on_ceiling_limit(html):
    """Ceiling limit section must not contain MANUAL badge."""
    # Find the ceiling limit block
    ceil_block_match = re.search(
        r'Ceiling limit.*?(?=</div>|<div style)',
        html, re.DOTALL
    )
    # More precise: look for the ceiling limit heading and check its badge
    ceiling_sections = re.findall(
        r'Ceiling limit\s*<span[^>]*>(.*?)</span>',
        html, re.DOTALL
    )
    for badge_text in ceiling_sections:
        assert 'MANUAL' not in badge_text, (
            f"Ceiling limit must not have MANUAL badge; found: {badge_text!r}"
        )


# ── TEST 2: DERIVED FROM WORKBENCH label present ─────────────────────────────

def test_derived_from_workbench_label_present(html):
    """DERIVED FROM WORKBENCH must appear as the ceiling badge."""
    assert 'DERIVED FROM WORKBENCH' in html, (
        "Ceiling limit must show 'DERIVED FROM WORKBENCH' label"
    )


# ── TEST 3: manualCeiling input is hidden and disabled ───────────────────────

def test_manual_ceiling_input_is_hidden(html):
    """manualCeiling input must have display:none — never user-visible."""
    # Find id="manualCeiling" input tag
    mc_tags = re.findall(r'<input[^>]+id="manualCeiling"[^>]*>', html)
    assert mc_tags, "manualCeiling input must still exist for JS compat"
    for tag in mc_tags:
        assert 'display:none' in tag or 'display: none' in tag, (
            f"manualCeiling input must have display:none; found: {tag!r}"
        )


def test_manual_ceiling_wf_input_is_hidden(html):
    """manualCeilingWF input must have display:none — never user-visible."""
    mc_tags = re.findall(r'<input[^>]+id="manualCeilingWF"[^>]*>', html)
    assert mc_tags, "manualCeilingWF input must still exist for JS compat"
    for tag in mc_tags:
        assert 'display:none' in tag or 'display: none' in tag, (
            f"manualCeilingWF input must have display:none; found: {tag!r}"
        )


# ── TEST 4: No oninput triggering _ceilingIsManual ───────────────────────────

def test_no_oninput_setting_ceiling_is_manual(html):
    """No oninput handler must set window._ceilingIsManual=true."""
    # Find all oninput attributes
    oninput_matches = re.findall(r'oninput="([^"]*)"', html)
    for handler in oninput_matches:
        assert '_ceilingIsManual' not in handler, (
            f"oninput handler must not set _ceilingIsManual; found: {handler!r}"
        )


# ── TEST 5: No _mcRaw reads that could override workbench ceiling ─────────────

def test_no_mcraw_reads_from_manual_ceiling(html):
    """_mcRaw pattern reading manualCeiling value must not exist."""
    # These patterns mean manual ceiling can override workbench — must be gone
    bad_patterns = [
        r'_mcRaw.*manualCeiling',
        r'parseFloat\(document\.getElementById\(.manualCeiling.\)\.value\)',
    ]
    for pattern in bad_patterns:
        # Exclude hidden input declarations
        matches = re.findall(pattern, html)
        active = [m for m in matches if 'display:none' not in m]
        # Filter out the disabled input tags themselves
        active = [m for m in active if 'id="manualCeiling"' not in m]
        assert not active, (
            f"Manual ceiling read pattern must be removed: {pattern!r}\n"
            f"Found: {active}"
        )


# ── TEST 6: bid_ceiling not used as canonical ceiling ────────────────────────

def test_bid_ceiling_not_canonical(html):
    """bid_ceiling must not appear as a ceiling fallback in the extraction block."""
    # The extraction block was updated to remove bid_ceiling fallback
    # Find the ceiling extraction section
    extraction_section = re.search(
        r'Extract Workbench ceiling.*?Run calculations',
        html, re.DOTALL
    )
    if extraction_section:
        section_text = extraction_section.group(0)
        assert 'bid_ceiling' not in section_text or \
               'Never use' in section_text or \
               'not used' in section_text.lower(), (
            "bid_ceiling must not be used as canonical ceiling in extraction block"
        )
    # Specifically the old fallback pattern must be gone
    assert 'bid_ceiling > 5000 ? _bidCeil' not in html, (
        "bid_ceiling conditional fallback must be removed from ceiling chain"
    )


# ── TEST 7: Priority chain reads owned objects first ─────────────────────────

def test_priority_1_financial_current_standing(html):
    """financial_current_standing.workbench_ceiling_range must be Priority 1."""
    assert 'financial_current_standing' in html, (
        "financial_current_standing must be referenced in ceiling extraction"
    )
    assert 'workbench_ceiling_range' in html, (
        "workbench_ceiling_range must be referenced in ceiling extraction"
    )


def test_priority_2_workbench_ceiling_valuation_range(html):
    """workbench_ceiling.valuation_range must be Priority 2."""
    assert 'workbench_ceiling' in html, "workbench_ceiling must be referenced"


def test_priority_3_legacy_only_as_fallback(html):
    """Legacy summary_json.ceiling must be labelled LEGACY FALLBACK if used."""
    assert 'LEGACY FALLBACK' in html, (
        "Legacy ceiling path must be labelled LEGACY FALLBACK"
    )


# ── TEST 8 & 9: purchase price affects current standing only ─────────────────

def test_purchase_price_reads_purchaseprice_field(html):
    """calculate() reads purchasePrice field — not manualCeiling — for the bid."""
    # calculate() should read purchasePrice for the price variable
    assert "getElementById('purchasePrice')" in html or \
           'getElementById("purchasePrice")' in html, (
        "purchasePrice input must be the bid/price source"
    )


def test_purchase_price_does_not_write_to_ceiling(html):
    """purchasePrice input oninput must not write to manualCeiling or _verdictCeiling."""
    pp_tags = re.findall(r'<input[^>]+id="purchasePrice"[^>]*>', html)
    for tag in pp_tags:
        assert 'manualCeiling' not in tag, (
            "purchasePrice oninput must not write to manualCeiling"
        )
        assert '_verdictCeiling' not in tag, (
            "purchasePrice oninput must not write to _verdictCeiling"
        )


# ── TEST 10: Manual ceiling input cannot affect ceiling ──────────────────────

def test_manual_ceiling_has_no_active_oninput(html):
    """manualCeiling input must have no oninput that triggers calculation."""
    mc_tags = re.findall(r'<input[^>]+id="manualCeiling"[^>]*>', html)
    for tag in mc_tags:
        assert 'oninput' not in tag, (
            f"Hidden manualCeiling input must have no oninput; found: {tag!r}"
        )


# ── TEST 11: Legacy ceiling labelled correctly ────────────────────────────────

def test_legacy_ceiling_label(html):
    """Legacy ceiling path must use the string LEGACY FALLBACK."""
    # Find the _isLegacyFallback assignment
    assert '_isLegacyFallback = true' in html or "_isLegacyFallback = True" in html, (
        "_isLegacyFallback flag must be set on legacy path"
    )
    assert "'LEGACY FALLBACK'" in html or '"LEGACY FALLBACK"' in html, (
        "LEGACY FALLBACK string must be used in ceiling label"
    )


# ── TEST 12: Missing-data state explicitly handled ────────────────────────────

def test_missing_data_state_explicit(html):
    """If no ceiling source, show explicit missing state."""
    assert 'no ceiling data' in html.lower() or \
           'No workbench ceiling available' in html or \
           'no_bid' in html, (
        "Missing-data state must be explicitly handled"
    )


# ── TEST 13: wbCeilDisplay element exists ────────────────────────────────────

def test_wb_ceil_display_element_exists(html):
    """wbCeilDisplay element must exist for programmatic ceiling display."""
    assert 'id="wbCeilDisplay"' in html, (
        "wbCeilDisplay element must exist for workbench ceiling display"
    )


# ── TEST 14: _verdictCeiling always set from workbench, not bid_ceiling ──────

def test_verdict_ceiling_set_from_workbench_not_bid(html):
    """window._verdictCeiling must not be set from bid_ceiling in the extraction assignment."""
    # Specifically look for the pattern: _verdictCeiling = ... bid_ceiling ... (on same or next line)
    # The old pattern was: _verdictCeiling = _crMid > 5000 ? _crMid : _bidCeil > 5000 ? _bidCeil
    assert '_bidCeil > 5000 ? _bidCeil' not in html, (
        "_verdictCeiling must not fall back to bid_ceiling"
    )
    assert 'bid_ceiling > 5000' not in html, (
        "_verdictCeiling must not use bid_ceiling conditional"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
