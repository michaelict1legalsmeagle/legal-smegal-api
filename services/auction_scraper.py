"""
services/auction_scraper.py
────────────────────────────────────────────────────────────────────────────
Auction listing scraper — Phase 1 (discovery only, no enrichment).

Design contracts:
  - Stateless: no DB calls, no globals mutated.
  - Fault-isolated: every function catches its own exceptions and returns []
    rather than raising. The cron caller decides whether to log/continue.
  - Testable: scrape_source() accepts a plain dict, returns a list of dicts.
  - No LLM calls. No enrichment. Extraction only.

scrape_method values (stored in auction_sources.selectors):
  "http"       → requests + BeautifulSoup (static HTML)
  "firecrawl"  → Firecrawl Extract API (JS-rendered pages)

Selector contract (auction_sources.selectors jsonb):
  For http:
    listing_container  — CSS selector for each lot element
    lot_number         — within container (optional)
    address            — within container (required for useful data)
    guide_price        — within container (optional)
    auction_date       — within container (optional)
    detail_link        — href to detail page (optional, used as source_url)
    legal_pack_link    — href to legal pack PDF (optional)
    property_type      — within container (optional)
    pagination_next    — next-page link selector (optional, enables pagination)

  For firecrawl:
    firecrawl_prompt   — plain-English extraction instruction

All selectors are optional. Missing = field skipped. No crashes on partial config.
"""

import os
import re
import time
import logging
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import requests

log = logging.getLogger(__name__)

# ── CONSTANTS ────────────────────────────────────────────────────────────────

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")
FIRECRAWL_BASE    = "https://api.firecrawl.dev/v1"

HTTP_TIMEOUT   = 30    # seconds per request
MAX_PAGES      = 8     # max pages scraped per source per run
PAGE_DELAY_S   = 2.0   # polite delay between page requests

USER_AGENT = (
    "Mozilla/5.0 (compatible; LegalSmegal-Discovery/1.0; "
    "property research; contact=hello@legalsmegal.com)"
)

_SESSION: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """Return a shared requests.Session with consistent headers."""
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-GB,en;q=0.9",
        })
    return _SESSION


# ── PUBLIC ENTRY POINT ───────────────────────────────────────────────────────

def scrape_source(source: dict) -> list[dict]:
    """
    Scrape all listings for one auction_sources row.

    Args:
        source: dict matching auction_sources schema
                {id, name, slug, listings_url, scrape_method, selectors}

    Returns:
        List of normalised listing dicts. Empty list on any failure.
        Never raises.
    """
    slug   = source.get("slug", "unknown")
    method = (source.get("scrape_method") or "http").strip().lower()

    log.info("[SCAN:%s] Starting %s scrape — url: %s", slug, method, source.get("listings_url"))

    try:
        if method == "firecrawl":
            results = _scrape_firecrawl(source)
        else:
            results = _scrape_http(source)
    except Exception as exc:
        log.error("[SCAN:%s] Unhandled exception in scraper: %s", slug, exc, exc_info=True)
        return []

    log.info("[SCAN:%s] Raw extraction complete — %d listings before normalisation", slug, len(results))

    normalised = []
    for raw in results:
        item = _normalise_listing(raw, source)
        if item:
            normalised.append(item)

    log.info("[SCAN:%s] Normalised: %d usable listings", slug, len(normalised))
    return normalised




def _extract_listings_from_text(soup: Any, source: dict) -> list[dict]:
    """
    Special parser for pages where all lot data is in anchor text (Auction House UK).
    Finds all <a title="View property details"> links and parses their text content.
    """
    page_url = source.get("listings_url", "")
    results = []

    containers = soup.select('a[title="View property details"]')
    if not containers:
        # Fallback: any lot-looking links
        containers = soup.select('a[href*="/auction/lot/"]')

    for el in containers:
        href = el.get("href", "")
        if not href:
            continue
        detail_url = urljoin(page_url, href)
        text = el.get_text(separator=" ", strip=True)

        # Extract lot number: "Lot 42" or "Lot 1"
        lot_match = re.search(r"\bLot\s+(\d+)", text, re.IGNORECASE)
        lot_number = f"Lot {lot_match.group(1)}" if lot_match else None

        # Extract guide price: "*Guide | £35,000" or "£35,000 - £40,000"
        price_match = re.search(r"[£*]\s*Guide\s*\|?\s*£?([\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
        if not price_match:
            price_match = re.search(r"£([\d,]+(?:\.\d+)?)", text)
        guide_price_raw = price_match.group(1).replace(",", "") if price_match else None

        # Extract address: text before "Lot N" — take the part after " - " if present
        address = None
        if " - " in text:
            addr_part = text.split(" - ", 1)[1]
            # Remove everything from "Lot N" onward
            if lot_match:
                addr_part = addr_part[:addr_part.upper().find("LOT ")].strip()
            address = _clean_text(addr_part) if addr_part else None

        # Extract property type
        type_match = re.search(
            r"\b(\d+\s+Bed\s+)?(Terraced House|Detached House|Semi-Detached House|"
            r"Bungalow|Flat|Apartment|Land|Commercial|Mixed Use|HMO|Other)\b",
            text, re.IGNORECASE
        )
        property_type = type_match.group(0).strip() if type_match else None

        results.append({
            "_raw_source_url":     detail_url,
            "_raw_lot_number":     lot_number,
            "_raw_address":        address,
            "_raw_guide_price":    guide_price_raw,
            "_raw_auction_date":   None,  # not in list view — on detail page
            "_raw_property_type":  property_type,
            "_raw_legal_pack_url": None,
            "_source_id":          source.get("id"),
            "_auction_house":      source.get("name"),
        })

    return results


# ── HTTP SCRAPER ─────────────────────────────────────────────────────────────

def _scrape_http(source: dict) -> list[dict]:
    """
    Fetch static HTML pages and extract listings with BeautifulSoup.
    Handles pagination up to MAX_PAGES.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        log.error("[SCAN] beautifulsoup4 not installed — add to requirements.txt")
        return []

    selectors  = source.get("selectors") or {}
    slug       = source.get("slug", "unknown")
    base_url   = source.get("listings_url", "")
    session    = _get_session()

    container_sel  = selectors.get("listing_container", ".lot, .property, article")
    pagination_sel = selectors.get("pagination_next")

    all_listings: list[dict] = []
    current_url  = base_url
    pages_fetched = 0

    while current_url and pages_fetched < MAX_PAGES:
        if pages_fetched > 0:
            time.sleep(PAGE_DELAY_S)

        try:
            resp = session.get(current_url, timeout=HTTP_TIMEOUT, allow_redirects=True)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            log.warning("[SCAN:%s] HTTP %s on page %d: %s", slug, e.response.status_code, pages_fetched + 1, current_url)
            break
        except requests.exceptions.RequestException as e:
            log.warning("[SCAN:%s] Request failed on page %d: %s", slug, pages_fetched + 1, e)
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        # Check for text_parse mode (Auction House style — data in link text)
        if selectors.get("text_parse"):
            page_listings = _extract_listings_from_text(soup, source)
            all_listings.extend(page_listings)
            log.info("[SCAN:%s] Page %d (text_parse): %d extracted", slug, pages_fetched + 1, len(page_listings))
            pages_fetched += 1
            current_url = None  # no pagination in text_parse mode for now
            continue

        containers = soup.select(container_sel)

        if not containers:
            log.warning("[SCAN:%s] No containers matched '%s' on page %d — check selectors", slug, container_sel, pages_fetched + 1)
            break

        page_listings = [_extract_listing_from_element(el, selectors, current_url, source) for el in containers]
        page_listings = [l for l in page_listings if l]  # drop None
        all_listings.extend(page_listings)

        log.info("[SCAN:%s] Page %d: %d containers → %d extracted", slug, pages_fetched + 1, len(containers), len(page_listings))

        pages_fetched += 1

        # Pagination
        next_url = None
        if pagination_sel:
            next_el = soup.select_one(pagination_sel)
            if next_el:
                href = next_el.get("href", "")
                if href and not href.startswith("#"):
                    next_url = urljoin(current_url, href)
                    if next_url == current_url:
                        next_url = None

        current_url = next_url

    return all_listings


def _extract_listing_from_element(el: Any, selectors: dict, page_url: str, source: dict) -> Optional[dict]:
    """Extract one listing dict from a BeautifulSoup element using the selector map."""
    def _text(sel: Optional[str]) -> Optional[str]:
        if not sel:
            return None
        found = el.select_one(sel)
        return found.get_text(strip=True) if found else None

    def _href(sel: Optional[str]) -> Optional[str]:
        if not sel:
            return None
        found = el.select_one(sel)
        if found:
            href = found.get("href", "")
            return urljoin(page_url, href) if href else None
        return None

    detail_url = _href(selectors.get("detail_link"))
    # If no detail link extracted, try the element itself if it's an <a>
    if not detail_url and el.name == "a":
        href = el.get("href", "")
        if href:
            detail_url = urljoin(page_url, href)

    return {
        "_raw_source_url":     detail_url,
        "_raw_lot_number":     _text(selectors.get("lot_number")),
        "_raw_address":        _text(selectors.get("address")),
        "_raw_guide_price":    _text(selectors.get("guide_price")),
        "_raw_auction_date":   _text(selectors.get("auction_date")),
        "_raw_property_type":  _text(selectors.get("property_type")),
        "_raw_legal_pack_url": _href(selectors.get("legal_pack_link")),
        "_source_id":          source.get("id"),
        "_auction_house":      source.get("name"),
    }


# ── FIRECRAWL SCRAPER ─────────────────────────────────────────────────────────

def _scrape_firecrawl(source: dict) -> list[dict]:
    """
    Use Firecrawl Extract API to scrape JS-rendered pages.
    Returns raw listing dicts using the same _raw_ key convention.
    """
    slug    = source.get("slug", "unknown")
    url     = source.get("listings_url", "")
    prompt  = (source.get("selectors") or {}).get(
        "firecrawl_prompt",
        "Extract all auction property listings. For each return: lot_number, address, guide_price as a number, auction_date as YYYY-MM-DD, property_type, detail_url."
    )

    if not FIRECRAWL_API_KEY:
        log.error("[SCAN:%s] FIRECRAWL_API_KEY not set — cannot scrape firecrawl source", slug)
        return []

    schema = {
        "type": "object",
        "properties": {
            "listings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "lot_number":    {"type": "string"},
                        "address":       {"type": "string"},
                        "guide_price":   {"type": "string"},
                        "auction_date":  {"type": "string"},
                        "property_type": {"type": "string"},
                        "detail_url":    {"type": "string"},
                        "legal_pack_url": {"type": "string"},
                    }
                }
            }
        }
    }

    payload = {
        "url": url,
        "formats": ["extract"],
        "extract": {
            "schema": schema,
            "systemPrompt": prompt,
        }
    }

    try:
        resp = requests.post(
            f"{FIRECRAWL_BASE}/scrape",
            json=payload,
            headers={
                "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        log.error("[SCAN:%s] Firecrawl request failed: %s", slug, e)
        return []
    except Exception as e:
        log.error("[SCAN:%s] Firecrawl response parse failed: %s", slug, e)
        return []

    raw_listings = (
        (data.get("data") or {})
        .get("extract", {})
        .get("listings", [])
    )

    if not isinstance(raw_listings, list):
        log.warning("[SCAN:%s] Firecrawl returned unexpected shape: %s", slug, type(raw_listings))
        return []

    # Remap to _raw_ convention for consistent normalisation
    results = []
    for item in raw_listings:
        if not isinstance(item, dict):
            continue
        results.append({
            "_raw_source_url":     item.get("detail_url"),
            "_raw_lot_number":     item.get("lot_number"),
            "_raw_address":        item.get("address"),
            "_raw_guide_price":    str(item.get("guide_price", "")) if item.get("guide_price") else None,
            "_raw_auction_date":   item.get("auction_date"),
            "_raw_property_type":  item.get("property_type"),
            "_raw_legal_pack_url": item.get("legal_pack_url"),
            "_source_id":          source.get("id"),
            "_auction_house":      source.get("name"),
        })

    return results


# ── NORMALISATION ─────────────────────────────────────────────────────────────

def _normalise_listing(raw: dict, source: dict) -> Optional[dict]:
    """
    Convert a _raw_ dict to the auction_listings column shape.
    Returns None if source_url is missing or clearly invalid (minimum required).
    """
    source_url = (raw.get("_raw_source_url") or "").strip()
    if not source_url or not source_url.startswith("http"):
        return None

    # Ensure URL is on the same domain as the source (basic sanity check)
    source_domain = urlparse(source.get("listings_url", "")).netloc
    listing_domain = urlparse(source_url).netloc
    if source_domain and listing_domain and source_domain not in listing_domain and listing_domain not in source_domain:
        log.debug("[SCRAPER] Skipping off-domain URL: %s (source domain: %s)", source_url, source_domain)
        return None

    return {
        "source_id":       raw.get("_source_id"),
        "source_url":      source_url,
        "auction_house":   raw.get("_auction_house"),
        "lot_number":      _clean_text(raw.get("_raw_lot_number")),
        "address":         _clean_text(raw.get("_raw_address")),
        "postcode":        _extract_postcode(raw.get("_raw_address")),
        "guide_price":     _parse_price(raw.get("_raw_guide_price")),
        "auction_date":    _parse_date(raw.get("_raw_auction_date")) if _is_valid_date(_parse_date(raw.get("_raw_auction_date"))) else None,
        "property_type":   _clean_text(raw.get("_raw_property_type")),
        "legal_pack_url":  _clean_url(raw.get("_raw_legal_pack_url")),
        "status":          "active",
    }


# ── FIELD PARSERS ─────────────────────────────────────────────────────────────

def _clean_text(v: Any) -> Optional[str]:
    """Strip and return text, or None if empty."""
    if not v:
        return None
    s = str(v).strip()
    # Collapse internal whitespace
    s = re.sub(r"\s+", " ", s)
    return s if s else None


def _clean_url(v: Any) -> Optional[str]:
    """Return URL string or None."""
    if not v:
        return None
    s = str(v).strip()
    return s if s.startswith("http") else None


def _extract_postcode(address: Any) -> Optional[str]:
    """
    Extract UK postcode from an address string.
    UK postcode regex covers all standard formats.
    Returns uppercase postcode with space, or None.
    """
    if not address:
        return None
    pattern = r"\b([A-Z]{1,2}\d{1,2}[A-Z]?)\s*(\d[A-Z]{2})\b"
    match = re.search(pattern, str(address).upper())
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return None


def _parse_price(v: Any) -> Optional[float]:
    """
    Parse guide price from various string formats to float.
    Handles: £95,000 / £95k / 95000 / "£95,000 - £100,000" (takes lower bound)
    """
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None

    # Take the first number if a range is given
    s = s.split("-")[0].split("–")[0].strip()

    # Remove currency symbols, commas, spaces
    s = re.sub(r"[£$,\s]", "", s)

    # Handle 'k' suffix (e.g. "95k")
    if s.lower().endswith("k"):
        try:
            return float(s[:-1]) * 1000
        except ValueError:
            return None

    try:
        f = float(s)
        # Sanity: UK property price range
        if 1_000 <= f <= 50_000_000:
            return f
        return None
    except ValueError:
        return None


def _parse_date(v: Any) -> Optional[str]:
    """
    Parse auction date to ISO date string (YYYY-MM-DD).
    Handles: "15 June 2026", "2026-06-15", "15/06/2026", "June 2026"
    Returns None if unparseable.
    """
    if not v:
        return None
    s = str(v).strip()
    if not s:
        return None

    # Already ISO format
    iso_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if iso_match:
        return s

    # DD/MM/YYYY or DD-MM-YYYY
    dmy_match = re.match(r"^(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})$", s)
    if dmy_match:
        d, m, y = dmy_match.groups()
        return f"{y}-{int(m):02d}-{int(d):02d}"

    # "15 June 2026" or "15th June 2026"
    MONTHS = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    long_match = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})", s)
    if long_match:
        d, mon, y = long_match.groups()
        m = MONTHS.get(mon[:3].lower())
        if m:
            return f"{y}-{m:02d}-{int(d):02d}"

    # "June 2026" — no day, use 1st as placeholder
    mon_year = re.search(r"([A-Za-z]+)\s+(\d{4})", s)
    if mon_year:
        mon, y = mon_year.groups()
        m = MONTHS.get(mon[:3].lower())
        if m:
            return f"{y}-{m:02d}-01"

    return None


def _is_valid_date(date_str: Optional[str]) -> bool:
    """Reject dates outside the plausible auction window (2024-2028)."""
    if not date_str:
        return False
    try:
        yr = int(date_str[:4])
        return 2024 <= yr <= 2028
    except (ValueError, TypeError):
        return False
