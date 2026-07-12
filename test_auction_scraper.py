"""
Regression test for the auction_scraper NameError (S-AUDIT-1).

_extract_listing_from_element() previously called `_extract_lot_image(element)`
but the function parameter was named `el` — a guaranteed NameError on every
successful selector match. This test locks in the fix.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "services"))

from bs4 import BeautifulSoup
from services.auction_scraper import _extract_listing_from_element, _extract_lot_image


SAMPLE_HTML = """
<div class="lot">
  <a class="detail-link" href="/lots/123">View lot</a>
  <span class="lot-number">Lot 12</span>
  <span class="address">42 Example Street, Leeds, LS1 1AA</span>
  <span class="guide-price">Guide £85,000</span>
  <img src="https://example.com/photos/lot12.jpg">
</div>
"""

SELECTORS = {
    "detail_link": "a.detail-link",
    "lot_number":  "span.lot-number",
    "address":     "span.address",
    "guide_price": "span.guide-price",
}


def _sample_element():
    soup = BeautifulSoup(SAMPLE_HTML, "html.parser")
    return soup.select_one("div.lot")


class TestExtractListingFromElement:
    def test_does_not_raise_nameerror(self):
        """The original bug: _extract_lot_image(element) — `element` didn't exist."""
        el = _sample_element()
        result = _extract_listing_from_element(
            el, SELECTORS, "https://example.com/auctions", {"id": "src1", "name": "Test House"}
        )
        assert result is not None

    def test_extracts_expected_fields(self):
        el = _sample_element()
        result = _extract_listing_from_element(
            el, SELECTORS, "https://example.com/auctions", {"id": "src1", "name": "Test House"}
        )
        assert result["_raw_lot_number"] == "Lot 12"
        assert result["_raw_address"] == "42 Example Street, Leeds, LS1 1AA"
        assert result["_raw_source_url"] == "https://example.com/lots/123"
        assert result["_source_id"] == "src1"

    def test_image_url_extracted_via_el_param(self):
        """Confirms _extract_lot_image is called with the correct (el) reference."""
        el = _sample_element()
        result = _extract_listing_from_element(
            el, SELECTORS, "https://example.com/auctions", {"id": "src1", "name": "Test House"}
        )
        assert result["_raw_image_url"] == "https://example.com/photos/lot12.jpg"

    def test_matches_direct_extract_lot_image_call(self):
        el = _sample_element()
        assert _extract_lot_image(el) == "https://example.com/photos/lot12.jpg"

    def test_missing_image_returns_none_not_error(self):
        soup = BeautifulSoup('<div class="lot"><span class="address">1 No Photo Rd</span></div>', "html.parser")
        el = soup.select_one("div.lot")
        result = _extract_listing_from_element(el, SELECTORS, "https://example.com", {"id": "s", "name": "n"})
        assert result["_raw_image_url"] is None
