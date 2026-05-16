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
from datetime import datetime, date, timezone
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

def scrape_source(source: dict, meta: dict | None = None) -> list[dict]:
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

    _meta: dict = {}  # accumulates observability data from sub-scrapers

    try:
        if method == "firecrawl":
            results = _scrape_firecrawl(source, _meta)
        elif method == "http_seed":
            results = _scrape_http_seed(source, _meta)
        elif method == "savills":
            results = _scrape_savills(source, _meta)
        else:
            results = _scrape_http(source)
    except Exception as exc:
        log.error("[SCAN:%s] Unhandled exception in scraper: %s", slug, exc, exc_info=True)
        if meta is not None:
            meta["partial_reason"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return []

    log.info("[SCAN:%s] Raw extraction complete — %d listings before normalisation", slug, len(results))

    normalised = []
    for raw in results:
        item = _normalise_listing(raw, source)
        if item:
            normalised.append(item)

    _meta["listings_skipped"] = len(results) - len(normalised)
    if not normalised and "partial_reason" not in _meta:
        _meta["partial_reason"] = "parse_zero"

    log.info("[SCAN:%s] Normalised: %d usable listings", slug, len(normalised))
    if meta is not None:
        meta.update(_meta)
    return normalised




def _extract_lot_image(container) -> "str | None":
    """
    Extract primary property image URL from a BeautifulSoup lot container.
    Handles standard src, lazy-load (data-src / data-original / data-lazy),
    og:image meta, and srcset. Never raises.
    """
    if container is None:
        return None
    SKIP = ("data:", ".svg", ".gif", "logo", "icon", "placeholder",
            "blank", "spacer", "1x1", "nophoto", "noimage", "default")
    # Attributes to check in priority order (covers lazy-load patterns)
    SRC_ATTRS = ("data-src", "data-original", "data-lazy", "data-lazy-src",
                 "data-url", "data-image", "src")
    try:
        # 1. og:image meta inside container (rare but covers full-page passes)
        og = (container.find("meta", property="og:image")
              or container.find("meta", attrs={"name": "og:image"}))
        if og and og.get("content", "").strip().startswith("http"):
            return og["content"].strip()

        # 2. img tag — check all lazy-load attributes before falling to src
        for img in container.find_all("img"):
            for attr in SRC_ATTRS:
                val = (img.get(attr) or "").strip()
                if not val:
                    continue
                if val.startswith("//"):
                    val = "https:" + val
                if not val.startswith("http"):
                    continue
                if any(s in val.lower() for s in SKIP):
                    continue
                return val

        # 3. srcset — take first entry (smallest, still the real image)
        for img in container.find_all("img", srcset=True):
            first = img["srcset"].split(",")[0].strip().split()[0]
            if first.startswith("http") and not any(s in first.lower() for s in SKIP):
                return first

    except Exception:
        pass
    return None


def _fetch_detail_og_image(source_url: str, session) -> "str | None":
    """
    Fetch og:image from a lot detail page as a fallback.
    Only called when container extraction fails.
    Uses a short timeout — non-blocking on failure.
    """
    if not source_url:
        return None
    try:
        r = session.get(source_url, timeout=8)
        if r.status_code != 200:
            return None
        from bs4 import BeautifulSoup as _BS
        soup = _BS(r.text, "html.parser")
        og = soup.find("meta", property="og:image")
        if og and og.get("content", "").strip().startswith("http"):
            return og["content"].strip()
    except Exception:
        pass
    return None


def _extract_listings_from_text(soup: Any, source: dict) -> list[dict]:
    """
    Parser for Auction House UK pages.
    National search page: all data is inside the <a> link text.
    Regional pages: lot number is in the <a> link; address/price are in sibling elements.
    Solution: use the PARENT element text (which contains both link + siblings).
    """
    page_url = source.get("listings_url", "")
    results = []

    containers = soup.select('a[title="View property details"]')
    if not containers:
        containers = soup.select('a[href*="/auction/lot/"]')

    for el in containers:
        href = el.get("href", "")
        if not href:
            continue
        detail_url = urljoin(page_url, href)

        # Use parent element text — captures address/price from sibling elements
        # on regional pages where the <a> only contains the lot number.
        link_text   = el.get_text(separator=" ", strip=True)
        parent      = el.parent
        parent_text = parent.get_text(separator=" ", strip=True) if parent else link_text
        # Use parent text if it's meaningfully richer (has address/price data)
        text = parent_text if len(parent_text) > len(link_text) + 8 else link_text

        # Detect past/sold lots (AH regional pages mix past results with upcoming)
        # These start with "Sold", "Postponed", "Withdrawn" in the parent text
        _is_sold = bool(re.match(
            r'^(?:Sold|Postponed|Withdrawn|Withheld)',
            text.strip(), re.IGNORECASE
        ))

        # Extract lot number: "Lot 42" or "Lot 1"
        lot_match = re.search(r"\bLot\s+(\d+)", text, re.IGNORECASE)
        lot_number = f"Lot {lot_match.group(1)}" if lot_match else None

        # Extract guide price: "*Guide | £35,000" or "£35,000 - £40,000"
        price_match = re.search(r"[£*]\s*Guide\s*\|?\s*£?([\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
        if not price_match:
            price_match = re.search(r"£([\d,]+(?:\.\d+)?)", text)
        guide_price_raw = price_match.group(1).replace(",", "") if price_match else None

        # Extract address — handles 6 format variations:
        # A: "Property for Auction in REGION - ADDRESS Lot N..."   (national search)
        # B: "Property for Auction in REGION ADDRESS Lot N..."     (no dash)
        # C: "ADDRESS Lot N..."                                    (no prefix)
        # D: "Lot N *Guide | £X (plus fees) TYPE ADDRESS"          (event pages)
        # E: "Lot N TYPE ADDRESS"                                   (some regional)
        # F: postcode-bounded fallback
        address = None

        if lot_match:
            lot_start = text.find(lot_match.group(0))
            before_lot = text[:lot_start]
            after_lot  = text[lot_start + len(lot_match.group(0)):].strip()

            # Try before_lot path (national search: address before lot number)
            before_lot = re.sub(
                r"^Property\s+(?:for\s+)?(?:Auction|Sale)\s+in\s+[A-Za-z,\s&']+?"
                r"(?:\s*-\s*|\s{2,}|\s+(?=[A-Z]\d|[A-Z]{2}\d|\d))",
                "", before_lot, flags=re.IGNORECASE
            ).strip()
            before_lot = re.sub(r"\s*[-\u2013]\s*$", "", before_lot).strip()

            if len(before_lot) > 8:
                address = _clean_text(before_lot)
            else:
                # Event page path: address is AFTER lot number
                # Strip "*Guide | £X - £Y (plus fees)" price prefix
                after_lot = re.sub(
                    r"^\*?Guide\s*\|\s*£[\d,]+(?:\s*[–\-]\s*£[\d,]+)?\s*(?:\(plus fees\))?\s*",
                    "", after_lot, flags=re.IGNORECASE
                ).strip()
                # Strip leading property type descriptor
                after_lot = re.sub(
                    r"^(?:\d+\s+(?:Bed|Bedroom)\s+)?(?:Terraced|Semi-Detached|Detached|"
                    r"End-Terraced|Flat|Apartment|Bungalow|Land|Commercial|Mixed Use|"
                    r"Studio|HMO|Maisonette)(?:\s+House|\s+Flat|\s+Bungalow)?\s+",
                    "", after_lot, flags=re.IGNORECASE
                ).strip()
                if len(after_lot) > 8:
                    address = _clean_text(after_lot[:200])

        # Postcode-bounded fallback
        if not address:
            pc_m = re.search(r"([A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2})", text)
            if pc_m:
                candidate = text[:pc_m.end()]
                candidate = re.sub(
                    r"^Property\s+(?:for\s+)?(?:Auction|Sale)\s+in\s+[A-Za-z,\s&']+?"
                    r"(?:\s*-\s*|\s{2,}|\s+(?=[A-Z]\d|[A-Z]{2}\d|\d))",
                    "", candidate, flags=re.IGNORECASE
                ).strip()
                # Also strip guide prefix from postcode candidate
                candidate = re.sub(
                    r"^\*?Guide\s*\|\s*£[\d,]+(?:\s*[–\-]\s*£[\d,]+)?\s*(?:\(plus fees\))?\s*",
                    "", candidate, flags=re.IGNORECASE
                ).strip()
                if len(candidate) > 8:
                    address = _clean_text(candidate)

        # Strip leading "Lot N" that may have leaked into address
        if address and lot_match:
            address = re.sub(r"^Lot\s+\d+\s*", "", address, flags=re.IGNORECASE).strip() or None

        # Extract property type
        type_match = re.search(
            r"\b(\d+\s+Bed\s+)?(Terraced House|Detached House|Semi-Detached House|"
            r"Bungalow|Flat|Apartment|Land|Commercial|Mixed Use|HMO|Other)\b",
            text, re.IGNORECASE
        )
        property_type = type_match.group(0).strip() if type_match else None

        # ── Legal pack link extraction (zero extra requests) ────────────────
        # Check lot container for any link to a legal/document download.
        # AH event pages sometimes include a "Legal Pack" button per lot;
        # if absent, _raw_legal_pack_url remains None — non-blocking.
        legal_pack_raw = None
        if parent:
            lp_keywords_href = ['legal', 'document', 'pack', '.pdf']
            lp_keywords_text = ['legal pack', 'legal document', 'download pack',
                                 'view pack', 'legal docs']
            for a_el in parent.find_all('a', href=True):
                a_href  = (a_el.get('href') or '').lower()
                a_text  = a_el.get_text(strip=True).lower()
                a_title = (a_el.get('title') or '').lower()
                href_abs = urljoin(page_url, a_el.get('href', ''))
                # Skip if it's the same as the lot detail link
                if href_abs == detail_url:
                    continue
                if (any(kw in a_href for kw in lp_keywords_href) or
                        any(kw in a_text or kw in a_title for kw in lp_keywords_text)):
                    legal_pack_raw = href_abs
                    break

        results.append({
            "_raw_source_url":     detail_url,
            "_raw_lot_number":     lot_number,
            "_raw_address":        address,
            "_raw_guide_price":    guide_price_raw,
            "_raw_auction_date":   None,  # not in list view — on detail page
            "_raw_property_type":  property_type,
            "_raw_legal_pack_url": legal_pack_raw,
            "_raw_image_url":      _extract_lot_image(parent),
            "_source_id":          source.get("id"),
            "_auction_house":      source.get("name"),
            "_is_sold":            _is_sold,
        })

    return results




def _scrape_savills(source: dict, meta: dict | None = None) -> list:
    """
    Savills auction scraper.

    Step 1 — HTTP GET the seed URL (upcoming-auctions page) to find the
             current active catalogue URL. Savills auction IDs rotate monthly;
             the seed page always lists the next available 'View catalogue'.

    Step 2 — Append /quantity-100 to get 100 lots per page, then call
             _scrape_firecrawl() which renders JS and extracts lot data
             via the firecrawl_prompt stored in source.selectors.

    The source.listings_url stores the seed URL:
      https://auctions.savills.co.uk/upcoming-auctions
    """
    import re as _re
    slug = source.get("slug", "savills")
    seed = source.get("listings_url") or "https://auctions.savills.co.uk/upcoming-auctions"

    # ── Step 1: resolve active catalogue URL from seed ─────────────────────
    catalogue_url = None
    try:
        import requests as _req
        from bs4 import BeautifulSoup as _BS

        r = _req.get(seed, timeout=15,
                     headers={"User-Agent": "Mozilla/5.0 (compatible; LegalSmegal/1.0)"})
        r.raise_for_status()
        soup = _BS(r.text, "html.parser")

        # Find <a> with href matching /auctions/DD-MONTH-YYYY-NNN
        _pat = _re.compile(r"/auctions/\d+-\w+-\d{4}-\d+$")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if _pat.search(href):
                txt = a.get_text(strip=True).lower()
                if "catalogue" in txt or "lot" in txt:
                    catalogue_url = href
                    if not catalogue_url.startswith("http"):
                        catalogue_url = "https://auctions.savills.co.uk" + catalogue_url
                    break

        if not catalogue_url:
            log.warning("[SCAN:%s] No active catalogue found on Savills seed page", slug)
            if meta is not None:
                meta["partial_reason"] = "no_savills_catalogue"
            return []

        log.info("[SCAN:%s] Savills catalogue resolved: %s", slug, catalogue_url)

    except Exception as exc:
        log.error("[SCAN:%s] Savills seed fetch failed: %s", slug, exc)
        if meta is not None:
            meta["partial_reason"] = f"savills_seed_error: {exc}"
        return []

    # ── Step 2: Firecrawl the catalogue (100 lots per page) ────────────────
    # Append /quantity-100 so Firecrawl sees a full page of lots
    firecrawl_url = catalogue_url.rstrip("/") + "/quantity-100"
    source_override = {**source, "listings_url": firecrawl_url}

    results = _scrape_firecrawl(source_override, meta=meta)

    # Post-process: inject auction_date from catalogue URL slug if missing
    # URL pattern: /auctions/20-may-2026-223 → "2026-05-20"
    _date_m = _re.search(r"/(\d+)-(\w+)-(\d{4})-\d+", catalogue_url)
    if _date_m:
        _day, _mon_str, _yr = _date_m.group(1), _date_m.group(2).lower(), _date_m.group(3)
        _MONTHS = {"jan":"01","feb":"02","mar":"03","apr":"04","may":"05","jun":"06",
                   "jul":"07","aug":"08","sep":"09","oct":"10","nov":"11","dec":"12"}
        _mon = _MONTHS.get(_mon_str[:3])
        if _mon:
            _auction_date = f"{_yr}-{_mon}-{_day.zfill(2)}"
            for lot in results:
                if not lot.get("_raw_auction_date"):
                    lot["_raw_auction_date"] = _auction_date

    return results


def _scrape_http_seed(source: dict, meta: dict | None = None) -> list[dict]:
    """
    Two-step scraper for Auction House future-auction-dates approach.
    Step 1: Scrape the diary page (listings_url) — extract future event lot URLs + dates.
    Step 2: For each future event URL, scrape lots using text_parse.
    Guarantees ONLY upcoming lots are returned (diary only lists future events).
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        log.error("[SCAN] beautifulsoup4 not installed")
        return []

    slug     = source.get("slug", "unknown")
    base_url = "https://www.auctionhouse.co.uk"
    diary_url = source.get("listings_url", "")
    session  = _get_session()
    today    = datetime.now(timezone.utc).date()

    # ── Step 1: Fetch diary and extract future event lot URLs ────────────
    try:
        resp = session.get(diary_url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        log.error("[SCAN:%s] Diary fetch failed: %s", slug, e)
        if meta is not None:
            meta["partial_reason"] = "seed_fetch_error"
            meta["seed_urls_found"] = 0
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    # Diary table rows: each row has date text + "View Lots" link
    # Link pattern: /{region}/auction/lots/{id} or /online/auction/{year}/{m}/{d}
    event_entries: list[dict] = []
    for link in soup.select("a[href*='/auction/lots/'], a[href*='/online/auction/']"):
        href = link.get("href", "")
        if not href:
            continue
        full_url = href if href.startswith("http") else base_url + href

        # Extract date from the surrounding table row
        row = link.find_parent("tr")
        row_text = row.get_text(separator=" ", strip=True) if row else ""

        # Parse date from row: formats "12/05/2026" or "Tue 12/05/2026"
        date_match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", row_text)
        event_date = None
        if date_match:
            try:
                d, m, y = int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3))
                event_date = date(y, m, d)
            except (ValueError, TypeError):
                pass

        # Skip events in the past (keep today and future)
        if event_date and event_date < today:
            continue

        # Extract auctioneer name from row
        auctioneer = ""
        td_cells = row.select("td") if row else []
        if len(td_cells) >= 3:
            auctioneer = td_cells[2].get_text(strip=True)

        event_entries.append({
            "url":         full_url,
            "event_date":  event_date.isoformat() if event_date else None,
            "auctioneer":  auctioneer or source.get("name", "Auction House"),
        })

    log.info("[SCAN:%s] Found %d future auction events on diary", slug, len(event_entries))
    if meta is not None:
        meta["seed_urls_found"] = len(event_entries)
    if not event_entries:
        log.warning("[SCAN:%s] Diary returned 0 future events — no lots to scrape", slug)
        if meta is not None:
            meta["partial_reason"] = "seed_empty"
        return []

    # ── Step 2: Scrape each event lot page ───────────────────────────────
    all_results: list[dict] = []
    selectors = source.get("selectors") or {}

    for event in event_entries[:MAX_PAGES]:  # cap at MAX_PAGES events per run
        time.sleep(PAGE_DELAY_S)
        try:
            eresp = session.get(event["url"], timeout=HTTP_TIMEOUT)
            eresp.raise_for_status()
        except Exception as e:
            log.warning("[SCAN:%s] Event page failed %s: %s", slug, event["url"], e)
            continue

        esoup = BeautifulSoup(eresp.text, "html.parser")

        # Use existing text_parse extraction on event page
        raw_lots = _extract_listings_from_text(esoup, {
            **source,
            "listings_url": event["url"],
            "name": event.get("auctioneer") or source.get("name", "Auction House"),
        })

        # Inject auction_date from diary (more reliable than extracting from lot text)
        for lot in raw_lots:
            if event.get("event_date"):
                lot["_raw_auction_date"] = event["event_date"]

        all_results.extend(raw_lots)
        log.info("[SCAN:%s] Event %s: %d lots", slug, event["url"][-30:], len(raw_lots))

    return all_results


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
        "_raw_image_url":      _extract_lot_image(element),
        "_source_id":          source.get("id"),
        "_auction_house":      source.get("name"),
    }


# ── FIRECRAWL SCRAPER ─────────────────────────────────────────────────────────

def _scrape_firecrawl(source: dict, meta: dict | None = None) -> list[dict]:
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
        if meta is not None:
            meta["firecrawl_status_code"] = resp.status_code
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        code = getattr(e.response, "status_code", None)
        if meta is not None:
            meta["firecrawl_status_code"] = code
            meta["partial_reason"] = (
                "firecrawl_rate_limit" if code == 429 else f"firecrawl_http_{code}"
            )
        log.error("[SCAN:%s] Firecrawl HTTP error %s: %s", slug, code, e)
        return []
    except requests.exceptions.RequestException as e:
        if meta is not None:
            meta["partial_reason"] = "firecrawl_network_error"
        log.error("[SCAN:%s] Firecrawl request failed: %s", slug, e)
        return []
    except Exception as e:
        if meta is not None:
            meta["partial_reason"] = "firecrawl_parse_error"
        log.error("[SCAN:%s] Firecrawl response parse failed: %s", slug, e)
        return []

    raw_listings = (
        (data.get("data") or {})
        .get("extract", {})
        .get("listings", [])
    )

    if not isinstance(raw_listings, list) or not raw_listings:
        if not isinstance(raw_listings, list):
            log.warning("[SCAN:%s] Firecrawl returned unexpected shape: %s", slug, type(raw_listings))
        else:
            log.warning("[SCAN:%s] Firecrawl returned 0 listings — possible extraction failure", slug)
        if meta is not None and "partial_reason" not in meta:
            meta["partial_reason"] = "firecrawl_empty"
        return []

    # Page-level og:image from Firecrawl metadata (reliable fallback per source page)
    page_og_image = (
        (data.get("data") or {})
        .get("metadata", {})
        .get("og:image") or None
    )

    # Remap to _raw_ convention for consistent normalisation
    results = []
    for item in raw_listings:
        if not isinstance(item, dict):
            continue
        # Per-lot image preferred; fall back to page og:image (same for all lots)
        img = item.get("image_url") or page_og_image or None
        results.append({
            "_raw_source_url":     item.get("detail_url"),
            "_raw_lot_number":     item.get("lot_number"),
            "_raw_address":        item.get("address"),
            "_raw_guide_price":    str(item.get("guide_price", "")) if item.get("guide_price") else None,
            "_raw_auction_date":   item.get("auction_date"),
            "_raw_property_type":  item.get("property_type"),
            "_raw_legal_pack_url": item.get("legal_pack_url"),
            "_raw_image_url":      img,
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

    # Skip sold/postponed lots from Auction House regional pages
    # These are past results and not useful for discovery
    if raw.get("_is_sold"):
        return None

    # Clean address — strip "Sold £X", "Postponed", "Sold Prior" prefixes
    raw_addr = raw.get("_raw_address") or ""
    clean_addr = re.sub(
        r"^(?:Sold\s+(?:Prior\s+)?(?:£[\d,]+\+?\s*)?|Postponed\s+|Withdrawn\s+|Withheld\s+)",
        "", raw_addr, flags=re.IGNORECASE
    ).strip()
    # Also strip leading property type descriptors (e.g. "2 Bed Semi-Detached House ")
    clean_addr = re.sub(
        r"^(?:\d+\s+(?:Bed|Bedroom)?\s+)?(?:Terraced|Semi-Detached|Detached|End-Terraced|"
        r"Flat|Apartment|Bungalow|Land|Commercial|Mixed Use|Studio|HMO)(?:\s+House|\s+Bungalow)?\s+",
        "", clean_addr, flags=re.IGNORECASE
    ).strip()

    return {
        "source_id":       raw.get("_source_id"),
        "source_url":      source_url,
        "auction_house":   raw.get("_auction_house"),
        "lot_number":      _clean_text(raw.get("_raw_lot_number")),
        "address":         _clean_text(clean_addr) if clean_addr else _clean_text(raw_addr),
        "postcode":        _extract_postcode(raw.get("_raw_address")),
        "guide_price":     _parse_price(raw.get("_raw_guide_price")),
        "auction_date":    _parse_date(raw.get("_raw_auction_date")) if _is_valid_date(_parse_date(raw.get("_raw_auction_date"))) else None,
        "property_type":   _clean_text(raw.get("_raw_property_type")),
        "legal_pack_url":  _clean_url(raw.get("_raw_legal_pack_url")),
        "image_url":       _clean_url(raw.get("_raw_image_url")),
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

    # Strip trailing annotations: "(plus fees)", "plus fees", "guide price", etc.
    s = re.sub(r"\s*\(.*?\).*$", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s*(plus fees|guide|plus|fees).*$", "", s, flags=re.IGNORECASE).strip()
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
