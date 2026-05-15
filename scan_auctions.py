"""
scan_auctions.py
────────────────────────────────────────────────────────────────────────────
Render cron entry point — auction discovery scan.

Runs daily at 07:00 UTC (configured in render.yaml).

What it does:
  1. Load active sources from auction_sources table
  2. For each source: scrape → upsert → log
  3. Write one row to auction_scan_log per source
  4. Update auction_sources.last_scanned_at

What it does NOT do:
  - Enrichment (Phase 2)
  - LLM calls (Phase 3)
  - Any modification to the deals pipeline

Exit codes:
  0 — all sources completed (some may have partial results)
  1 — fatal error before any scraping started (DB unavailable, etc.)

Environment variables required:
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY

Environment variables optional:
  FIRECRAWL_API_KEY  (required only if any source has scrape_method='firecrawl')
"""

import logging
import os
import sys
import time
from datetime import datetime, timezone, date, timedelta

from supabase import create_client, Client

# ── Logging — stdout only, structured prefix for Render log capture ──────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    stream=sys.stdout,
)
log = logging.getLogger("scan_auctions")

# Import scraper (sibling module when run from repo root)
sys.path.insert(0, os.path.dirname(__file__))
try:
    from services.auction_scraper import scrape_source
except ImportError as e:
    log.error("Failed to import auction_scraper: %s", e)
    sys.exit(1)

try:
    from services.auction_enrichment import enrich_pass
    _enrichment_available = True
except ImportError as e:
    log.warning("auction_enrichment not available — enrichment skipped: %s", e)
    enrich_pass = None
    _enrichment_available = False


# ── Supabase connection ──────────────────────────────────────────────────────

def _get_supabase() -> Client:
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
    return create_client(url, key)

# ── Circuit breaker ──────────────────────────────────────────────────────────
# Prevent chronic partial-failure sources from burning Firecrawl credits.
# A source whose last N scans were ALL non-ok gets skipped for that run.
# The circuit is re-checked every run — it auto-closes if a scan succeeds.

CIRCUIT_BREAKER_SLUGS     = {"allsop"}  # extend as needed
CIRCUIT_BREAKER_THRESHOLD = 5           # consecutive non-ok runs to open


def _check_circuit_open(supabase: Client, slug: str, threshold: int) -> bool:
    """Return True if slug has had `threshold` consecutive non-ok scans.
    Fails open on error — allows the scrape if the check itself errors.
    """
    try:
        res = (
            supabase.table("auction_scan_log")
            .select("status")
            .eq("source_slug", slug)
            .order("scanned_at", desc=True)
            .limit(threshold)
            .execute()
        )
        rows = res.data or []
        if len(rows) < threshold:
            return False  # not enough history to be confident — allow
        return all(r["status"] != "ok" for r in rows)
    except Exception as e:
        log.warning("[SCAN:%s] Circuit breaker check failed: %s", slug, e)
        return False  # fail open



# ── Upsert listings ──────────────────────────────────────────────────────────

def _upsert_listings(supabase: Client, listings: list[dict]) -> tuple[int, int]:
    """
    Upsert listings into auction_listings.
    ON CONFLICT (source_url): update last_seen_at, guide_price, legal_pack_url.

    Returns (new_count, updated_count).
    new_count is approximate (Supabase upsert doesn't distinguish insert vs update
    without a returning clause, so we query before to count pre-existing).
    """
    if not listings:
        return 0, 0

    source_urls = [l["source_url"] for l in listings]

    # Count how many already exist
    try:
        existing_res = supabase.table("auction_listings") \
            .select("source_url") \
            .in_("source_url", source_urls) \
            .execute()
        existing_urls = {r["source_url"] for r in (existing_res.data or [])}
    except Exception as e:
        log.warning("Could not pre-count existing listings: %s", e)
        existing_urls = set()

    new_count = 0
    updated_count = 0

    for listing in listings:
        is_new = listing["source_url"] not in existing_urls

        # Build upsert row — exclude fields that shouldn't overwrite on conflict
        row = {
            "source_id":       listing.get("source_id"),
            "source_url":      listing["source_url"],
            "auction_house":   listing.get("auction_house"),
            "lot_number":      listing.get("lot_number"),
            "address":         listing.get("address"),
            "postcode":        listing.get("postcode"),
            "guide_price":     listing.get("guide_price"),
            "auction_date":    listing.get("auction_date"),
            "property_type":   listing.get("property_type"),
            "legal_pack_url":  listing.get("legal_pack_url"),
            "status":          "active",
            "last_seen_at":    datetime.now(timezone.utc).isoformat(),
        }
        # Only set first_seen_at for new rows (upsert preserves existing value on conflict)
        if is_new:
            row["first_seen_at"] = row["last_seen_at"]

        # Fields that hold persistent value — never overwrite with null.
        # guide_price, legal_pack_url etc. may become null on a rescan when
        # an auction has passed (page structure changes) but the original value
        # remains correct and useful for investors.
        _preserve = {"guide_price", "legal_pack_url", "auction_date",
                     "address", "postcode", "property_type", "tenure", "lot_number"}

        try:
            if is_new:
                supabase.table("auction_listings").insert(row).execute()
                new_count += 1
            else:
                # Only send fields that are non-null, PLUS always-update fields
                _update_row = {
                    k: v for k, v in row.items()
                    if v is not None or k not in _preserve
                }
                supabase.table("auction_listings")                     .update(_update_row)                     .eq("source_url", listing["source_url"])                     .execute()
                updated_count += 1
        except Exception as e:
            log.warning("Upsert failed for %s: %s", listing["source_url"], e)

    return new_count, updated_count


# ── Log scan result ──────────────────────────────────────────────────────────

def _log_scan(
    supabase: Client,
    source: dict,
    status: str,
    duration_s: float,
    listings_found: int,
    listings_new: int,
    listings_updated: int,
    error_msg: str | None = None,
    partial_reason: str | None = None,
    seed_urls_found: int | None = None,
    firecrawl_status_code: int | None = None,
    listings_skipped: int = 0,
) -> None:
    try:
        row: dict = {
            "source_id":        source.get("id"),
            "source_slug":      source.get("slug", "unknown"),
            "status":           status,
            "duration_s":       round(duration_s, 2),
            "listings_found":   listings_found,
            "listings_new":     listings_new,
            "listings_updated": listings_updated,
            "listings_skipped": listings_skipped,
            "error_msg":        error_msg,
        }
        # New observability columns — only included when set.
        # Migration must be run before deployment; omitting here is safe
        # because Postgres INSERT ignores columns not in the statement.
        if partial_reason is not None:
            row["partial_reason"] = partial_reason
        if seed_urls_found is not None:
            row["seed_urls_found"] = seed_urls_found
        if firecrawl_status_code is not None:
            row["firecrawl_status_code"] = firecrawl_status_code
        supabase.table("auction_scan_log").insert(row).execute()
    except Exception as e:
        # Non-fatal: if logging fails, we continue
        log.warning("Failed to write scan log for %s: %s", source.get("slug"), e)


# ── Update source last_scanned_at ────────────────────────────────────────────

def _touch_source(supabase: Client, source_id: str) -> None:
    try:
        supabase.table("auction_sources").update({
            "last_scanned_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", source_id).execute()
    except Exception as e:
        log.warning("Failed to update last_scanned_at for source %s: %s", source_id, e)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    run_start = time.time()
    log.info("[SCAN] ════ Auction scan starting ════")
    log.info("[SCAN] Run time: %s UTC", datetime.utcnow().isoformat())

    # Connect
    try:
        supabase = _get_supabase()
        log.info("[SCAN] Supabase connected")
    except Exception as e:
        log.error("[SCAN] Fatal: cannot connect to Supabase — %s", e)
        sys.exit(1)

    # Load active sources
    try:
        sources_res = supabase.table("auction_sources") \
            .select("id,name,slug,listings_url,scrape_method,selectors") \
            .eq("active", True) \
            .execute()
        sources = sources_res.data or []
    except Exception as e:
        log.error("[SCAN] Fatal: cannot load auction_sources — %s", e)
        sys.exit(1)

    if not sources:
        log.warning("[SCAN] No active sources found — exiting")
        return

    log.info("[SCAN] Sources to scan: %d", len(sources))

    # Track totals
    total_found   = 0
    total_new     = 0
    total_updated = 0
    source_results = []

    for source in sources:
        slug        = source.get("slug", "unknown")
        source_start = time.time()
        status      = "ok"
        error_msg   = None
        listings    = []
        new_c = updated_c = 0
        _meta: dict = {}  # observability data from scraper

        # ── Circuit breaker ──────────────────────────────────────────────
        if slug in CIRCUIT_BREAKER_SLUGS and _check_circuit_open(supabase, slug, CIRCUIT_BREAKER_THRESHOLD):
            log.warning(
                "[SCAN:%s] Circuit open — %d consecutive non-ok runs — skipping",
                slug, CIRCUIT_BREAKER_THRESHOLD,
            )
            _log_scan(
                supabase=supabase, source=source, status="partial",
                duration_s=0.0, listings_found=0, listings_new=0, listings_updated=0,
                error_msg=f"circuit_open after {CIRCUIT_BREAKER_THRESHOLD} consecutive failures",
                partial_reason="circuit_open",
            )
            _touch_source(supabase, source["id"])
            source_results.append({"slug": slug, "status": "partial"})
            continue
        # ────────────────────────────────────────────────────────────────

        try:
            # ── SCRAPE ──────────────────────────────────────────────────
            listings = scrape_source(source, meta=_meta)

            if not listings:
                log.warning("[SCAN:%s] WARN: 0 listings found — check selectors or source health", slug)
                status = "partial"
                if not error_msg:
                    error_msg = _meta.get("partial_reason", "zero_listings")

            # ── UPSERT ──────────────────────────────────────────────────
            new_c, updated_c = _upsert_listings(supabase, listings)

        except Exception as exc:
            log.error("[SCAN:%s] Unhandled exception: %s", slug, exc, exc_info=True)
            status    = "failed"
            error_msg = str(exc)[:500]

        duration = time.time() - source_start

        # ── LOG ─────────────────────────────────────────────────────────
        _log_scan(
            supabase=supabase,
            source=source,
            status=status,
            duration_s=duration,
            listings_found=len(listings),
            listings_new=new_c,
            listings_updated=updated_c,
            error_msg=error_msg,
            partial_reason=_meta.get("partial_reason"),
            seed_urls_found=_meta.get("seed_urls_found"),
            firecrawl_status_code=_meta.get("firecrawl_status_code"),
            listings_skipped=_meta.get("listings_skipped", 0),
        )
        _touch_source(supabase, source["id"])

        total_found   += len(listings)
        total_new     += new_c
        total_updated += updated_c

        log.info(
            "[SCAN:%s] %s — found: %d, new: %d, updated: %d, duration: %.1fs",
            slug, status.upper(), len(listings), new_c, updated_c, duration,
        )
        source_results.append({"slug": slug, "status": status})

    # Summary
    total_duration = time.time() - run_start
    all_ok = all(r["status"] == "ok" for r in source_results)
    any_failed = any(r["status"] == "failed" for r in source_results)

    log.info(
        "[SCAN] ════ Complete ════ sources: %d, total found: %d, new: %d, updated: %d, duration: %.1fs",
        len(sources), total_found, total_new, total_updated, total_duration,
    )

    if any_failed:
        log.warning("[SCAN] One or more sources failed — check individual logs above")

    if not all_ok:
        log.warning("[SCAN] Source statuses: %s", {r["slug"]: r["status"] for r in source_results})

    # ── ENRICHMENT PASS ──────────────────────────────────────────────────────
    # Runs after all sources complete. Processes pending listings up to limit.
    # Enrichment failure is non-fatal — scan exit code is unaffected.
    if _enrichment_available and enrich_pass is not None:
        hetzner_url = os.environ.get("DATA_DATABASE_URL", "").strip()
        if hetzner_url:
            log.info("[ENRICH] Starting enrichment pass")
            try:
                enrich_summary = enrich_pass(
                    supabase_client=supabase,
                    hetzner_url=hetzner_url,
                    limit=100,
                )
                log.info(
                    "[ENRICH] Pass complete — enriched: %d failed: %d duration: %.1fs",
                    enrich_summary.get("enriched", 0),
                    enrich_summary.get("failed", 0),
                    enrich_summary.get("duration_s", 0.0),
                )
            except Exception as enrich_exc:
                log.error("[ENRICH] Enrichment pass raised: %s", enrich_exc, exc_info=True)
        else:
            log.warning("[ENRICH] DATA_DATABASE_URL not set — enrichment skipped")
    else:
        log.info("[ENRICH] Enrichment module not available — skipped")

    # ── EXPIRY PASS ──────────────────────────────────────────────────────────
    # Runs after scan + enrichment. Marks past-date listings as expired so
    # the discovery feed only shows upcoming / live opportunities.
    try:
        expire_result = _expire_stale_listings(supabase)
        if expire_result["expired"]:
            log.info("[EXPIRE] Pass complete — %d lots marked expired", expire_result["expired"])
    except Exception as expire_exc:
        log.error("[EXPIRE] Expiry pass raised: %s", expire_exc, exc_info=True)


# ── Stale listing expiry ─────────────────────────────────────────────────────

def _expire_stale_listings(supabase: Client) -> dict:
    """
    Mark auction_listings as 'expired' when their auction date has passed.

    Traditional sources: auction_date < today → expired.
      PostgreSQL NULL handling: lt() filter excludes NULL rows, so null-date
      listings (iamsold) are not touched by this branch.

    iamsold (null auction_date): expire after 30 days from first_seen_at.
      iamsold uses Modern Method of Auction with rolling bid deadlines;
      we have no reliable auction_date, so time-based expiry is used.

    Returns count of expired rows across both branches.
    """
    today      = date.today().isoformat()
    cutoff_mma = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    expired    = 0

    # Branch 1: any source with a past auction_date
    try:
        res = supabase.table("auction_listings")             .update({"status": "expired"})             .eq("status", "active")             .lt("auction_date", today)             .execute()
        n = len(res.data or [])
        expired += n
        if n:
            log.info("[EXPIRE] %d lots expired (past auction_date < %s)", n, today)
    except Exception as e:
        log.warning("[EXPIRE] Traditional expiry failed: %s", e)

    # Branch 2: iamsold — null auction_date, expire after 30 days
    try:
        res2 = supabase.table("auction_listings")             .update({"status": "expired"})             .eq("status", "active")             .eq("auction_house", "iamsold")             .lt("first_seen_at", cutoff_mma)             .execute()
        n2 = len(res2.data or [])
        expired += n2
        if n2:
            log.info("[EXPIRE] %d iamsold lots expired (first_seen_at > 30 days)", n2)
    except Exception as e:
        log.warning("[EXPIRE] iamsold expiry failed: %s", e)

    return {"expired": expired}


if __name__ == "__main__":
    main()
