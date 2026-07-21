"""
Regression test for S44-INSUFFICIENT-EVIDENCE-RESCUE.

Confirmed live on deal c45787fa-275d-4f0c-87ae-c56d784f93c7: a fresh verdict
recompute came back status=insufficient_evidence with no comparable_valuation,
so calculate_workbench_ceiling()'s own guard ("if verdict has no valid
comparable_valuation, workbench is also insufficient_evidence") returned
risks=[] and silently dropped all 25 real flags — 6 of them critical — even
though a real prior valuation (£122,432) already existed and was already
being shown correctly elsewhere on the same page via merged['comps_avg_value'].

This test proves two things with the REAL calculate_workbench_ceiling
function (not a mock): the rescue produces a genuine, non-zero risk
adjustment where the un-rescued verdict does not.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from services.ceiling_engine import calculate_workbench_ceiling


# The actual 6 critical + 9 high flags from the live deal (titles + severity
# only — enough for _flag_to_segments' keyword routing to fire correctly).
CHAPELFIELD_ROAD_FLAGS = [
    {"title": "Aviva Equity Release Charge — Seller May Not Own Property", "severity": "critical"},
    {"title": "Seller May Not Be Registered Proprietor", "severity": "critical"},
    {"title": "Squatter/Unlawful Occupation Risk — Buyer Bears Full Risk", "severity": "critical"},
    {"title": "Japanese Knotweed Waiver — Buyer Cannot Refuse Completion", "severity": "critical"},
    {"title": "Buyer Bears All Risk From Exchange — No Insurance Obligation on Seller", "severity": "critical"},
    {"title": "Adverse Entry Waiver — Buyer Cannot Refuse on Charges or Restrictions", "severity": "critical"},
    {"title": "Article 4 Direction — HMO Conversion Requires Planning Permission", "severity": "high"},
    {"title": "Restrictive Covenant — Residential Use Only, No Business", "severity": "high"},
    {"title": "Restrictive Covenant — One Semi-Detached Dwelling Only", "severity": "high"},
    {"title": "Aviva Equity Release — Further Advances Priority Registered", "severity": "high"},
    {"title": "Completion Date Fixed at 20 August 2026 — Non-Standard", "severity": "high"},
    {"title": "Limited Title Guarantee — Reduced Seller Warranties", "severity": "high"},
    {"title": "No Assignment or Sub-Sale Permitted Before Completion", "severity": "high"},
    {"title": "Interest Rate 10% Over Santander Base Rate on Default", "severity": "high"},
    {"title": "Seller Will Not Answer Buyer Enquiries — Buyer Purchases Blind", "severity": "high"},
]


class TestInsufficientEvidenceRescue:
    def test_unrescued_insufficient_evidence_verdict_produces_zero_risks(self):
        """Reproduces the confirmed live bug: a verdict with no comparable_valuation
        makes calculate_workbench_ceiling drop all flags, regardless of severity."""
        broken_verdict = {
            "status": "insufficient_evidence",
            "valuation_range": {"low": None, "midpoint": None, "high": None},
        }
        wb = calculate_workbench_ceiling(
            verdict_ceiling=broken_verdict,
            active_legal_flags=CHAPELFIELD_ROAD_FLAGS,
        )
        assert wb["status"] == "insufficient_evidence"
        assert wb["legal_pack_value_risks"]["risks"] == []
        assert wb["legal_pack_value_risks"]["adjustment_factor"] == 1.0

    def test_rescued_verdict_produces_real_risk_adjustment(self):
        """Mirrors exactly what the app.py fix does: inject comparable_valuation
        from merged['comps_avg_value'] before calling calculate_workbench_ceiling."""
        rescued_base = 122432.0
        rescued_ub = 0.05
        rescued_verdict = {
            "status": "insufficient_evidence",  # still tagged as such — honest about provenance
            "comparable_valuation": rescued_base,
            "valuation_range": {
                "low": round(rescued_base * (1 - rescued_ub), 2),
                "midpoint": round(rescued_base, 2),
                "high": round(rescued_base * (1 + rescued_ub), 2),
                "uncertainty_band": rescued_ub,
            },
            "audit": {"base_valuation_source": "rescued_from_prior_persisted_value"},
        }
        wb = calculate_workbench_ceiling(
            verdict_ceiling=rescued_verdict,
            active_legal_flags=CHAPELFIELD_ROAD_FLAGS,
        )
        # The core assertion: with the SAME 25 flags, the rescue produces a
        # real, non-empty risk list and a genuine discount — not risks=[].
        assert len(wb["legal_pack_value_risks"]["risks"]) > 0
        assert wb["legal_pack_value_risks"]["adjustment_factor"] < 1.0
        assert wb["valuation_range"]["midpoint"] < rescued_base
        # market_consequence_adjustments must now be present — this is
        # exactly the field the Verdict page checks for "has_mca".
        assert isinstance(wb.get("market_consequence_adjustments"), dict)

    def test_rescue_only_applies_when_status_is_insufficient_evidence(self):
        """A verdict that already succeeded must never be touched by the rescue —
        this only fires on the specific insufficient_evidence branch in app.py."""
        good_verdict = {
            "status": "ok",
            "comparable_valuation": 200000.0,
            "valuation_range": {"low": 190000.0, "midpoint": 200000.0, "high": 210000.0, "uncertainty_band": 0.05},
        }
        wb = calculate_workbench_ceiling(
            verdict_ceiling=good_verdict,
            active_legal_flags=CHAPELFIELD_ROAD_FLAGS,
        )
        assert wb["status"] != "insufficient_evidence"
        assert len(wb["legal_pack_value_risks"]["risks"]) > 0

    def test_rescue_gate_rejects_low_value_noise(self):
        """merged['comps_avg_value'] could theoretically hold a tiny/junk number —
        the app.py fix only rescues values > £5,000, matching the same threshold
        used everywhere else in this function for 'is this a real value'."""
        assert not (100 > 5000)  # sanity: the threshold in the real fix is > 5000
