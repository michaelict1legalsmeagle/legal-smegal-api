# Integration & deployment notes — Scotland engine

This module is delivered as a **reviewable, isolated** unit. The steps below are **your reviewed
integration**, not automated changes. Nothing here touches a locked area until you decide it does.

## Governance boundary (what this delivery does NOT do)

Per core-governance, these stay untouched unless you explicitly authorise a change:
ceiling engine · legal-pack engine · comp science · database schema · API contracts · Verdict · D1–D4.

Integration must be **additive and isolated**: a Scottish deal routes into this path; an E&W deal is
untouched. Capture the E&W regression baseline (deal output byte-identical before/after) before wiring.

## Routing (the one wiring decision)

At the point a deal's nation is known (from legal-pack extraction / normalised address):

```
if deal.nation == "Scotland":
    band = scotland_band_engine.build_band(deal.postcode, REF)   # this module
    anchor = scotland_band_engine.extract_home_report(pack_pdf)  # if Home Report present
else:
    ... existing E&W ceiling engine (unchanged) ...
```

Scotland output is **method-labelled and visually distinct** from the E&W comparable-derived ceiling —
it is an area band + surveyor anchor, not a comps ceiling. Do not present them as the same figure.

## Deployment concerns (address before it serves traffic)

1. **Startup cost — SOLVED via Postgres (recommended path).** Run `build_scotland_reference.py` once
   per data refresh to load the three files into `scotland_postcode / scotland_area_price /
   scotland_la_median`. At worker startup call `load_reference_pg(dsn)` — it loads only the tiny price
   (~7.5k) + LA (~33) tables into memory and queries the 157k postcode table lazily per request
   (indexed PK lookup). No 100 MB parse, no big RAM dict. Verified: DB path returns **identical** bands
   to the file path (see tests). One psycopg connection per process worker (see module docstring).
   *(Standalone alternative: `load_reference()` parses the files in-process — fine for batch, not for a
   multi-worker web service.)*
2. **`pdftotext` is a system dependency.** Ensure `poppler-utils` is in the container/host image. The
   extractor already guards untrusted uploads (size cap, timeout, graceful failure) — keep those.
3. **Runtime xlsx dependency is optional.** `openpyxl` is only needed to parse the RoS workbook. If you
   pre-extract the LA medians to a small JSON/table at build time, the request path needs no xlsx lib.
4. **Refresh cadence.** Small-area price = annual (next early 2026/2027). RoS LA = monthly. SPD =
   ~quarterly. SIMD = 2020 now, refresh expected late 2026 on 2022 zones. Version each load.

## Reference load (Postgres)

```
python3 build_scotland_reference.py --dsn <postgres_dsn> --ref-dir /data/scotland
```
Creates & populates the three tables (idempotent — drops & rebuilds) and prints a verification
summary. It reuses the engine's verified parsers, so DB rows match the file-path values exactly.
Re-run per refresh (SPD ~quarterly, price annual, RoS monthly). To unlock LQ/UQ, drop the full price
download (all measures/years) in the same folder before running — the loader reads any measure.

## Open items (honest — none block review, some block go-live)

- **Enrichment extraction** — property type 4/5, date 3/5, EPC 4/5 on the sample, due to Home Report
  template variance. Fields return `None`, never a wrong value. **Fix = a tested extractor against a
  corpus of packs**, not more regex on five. Until then, the type-shape adjustment simply doesn't run
  where type is `None` (honest degradation).
- **Divergence threshold** — `assess_lot(divergence_pct=40)` is a labelled `[CALIBRATE]` placeholder.
  Five real lots show normal property-vs-area variation reaches ±~35%; set the production threshold
  above that from more lots (MEASURE→L4). Not a claimed value.
- **Full price download** — swap the Median-2023 slice for the complete dataset to expose LQ/UQ/count.
- **ScotLIS subject last-sold** — optional enrichment; confirm terms permit automated lookup before
  wiring. Anchor + band stand without it.
- **Licence display** — surface `ATTRIBUTION.md` wherever the data renders (OGL requirement).
- **Vintage watch** — migrate to 2022 data zones only when the price series or SIMD re-bases.

## Not in this slice

Address-level Scottish comparables (licence-only, revenue-gated) · the Scottish **legal-pack flag**
engine (missives / Standard Securities / title sheet — different taxonomy from the E&W pack) ·
commercial Scotland · Northern Ireland.
