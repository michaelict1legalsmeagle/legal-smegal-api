```# LegalSmegal — Scotland Valuation Band + Anchor Engine (Slice 1)

An **isolated** module that produces an evidenced indicative value for a Scottish property from
**free official data only** — £0 data-licensing cost. It is the Scotland counterpart to the E&W
comparable engine, built to the same doctrine: **Data → Context → Visuals → User Inference**, no
fabricated values.

> **Not a comparable engine.** It returns an *area price band* (official small-area statistics),
> anchored on the Home Report surveyor valuation, with explicit insufficient-evidence states.
> Address-level Scottish comparables are licence-only and are a separate, revenue-gated slice.

## What it does

```
Home Report PDF ──▶ anchor (market value, date, type, EPC, condition)
lot postcode    ──▶ 2011 data zone ──▶ tiered band (Data Zone → Intermediate Zone → LA)
                                        └▶ RoS LA time-adjust 2023 → current
                                        └▶ SIMD 2020 decile, urban–rural, lat/long
                     ──▶ sourced output + honesty states + anchor-vs-band divergence flag
```

Every output figure carries **value + source + year + geography level + vintage**. Missing or
suppressed data returns an explicit *unavailable* / coarser-tier state — **never a guessed number**.

## Verified inputs (all confirmed against the real files)

| File | Source (OGL) | Role |
|---|---|---|
| `residential-properties-sales-and-price.csv` | statistics.gov.scot (RoS data) | small-area band — Count/LQ/Mean/Median/UQ, annual, **2011 data zone** |
| `SmallUser.csv` (from `spd_postcodeindex_cut`) | NRS Scottish Postcode Directory | postcode → DZ2011 / IZ2011 / council / SIMD2020 / lat-long |
| `ros_all_stats_June_2026.xlsx` | Registers of Scotland | current LA median (recency) + 2023 baseline (time-adjust) + house-type shape |
| Home Report PDF | per-lot pack | anchor: market value + date, property type, EPC, condition |

**Vintage:** price data and SIMD 2020 are **both 2011 data zones** → they join directly on
`DataZone2011Code`; no cross-vintage bridge. (2022 zones apply only after a future re-base.)

## Run (standalone / batch)

```bash
python3 scotland_band_engine.py <reference_dir> <home_reports_dir>
```
`<reference_dir>` holds the three data files above. Requires `poppler-utils` (`pdftotext`) on PATH
and `openpyxl` (see `requirements.txt`).

## Tests

```bash
export LS_SCOT_REF=<reference_dir>
export LS_SCOT_HR=<home_reports_dir>   # optional
python -m pytest test_scotland_band_engine.py -v
```
The suite locks in the verification audit: **2011-vintage guard**, **LA-code consistency across all
three sources** (the Glasgow/N.Lanarkshire code trap), **IZ fallback fires on a real suppressed data
zone**, **unknown postcode → unavailable (never a number)**, and **bad PDF → graceful error**.

## Isolation guarantee (governance)

This module imports and touches **nothing** in the live platform. It does **not** read the E&W
ceiling engine, comps tables, the dead Supabase price-paid objects, the database, or the API.
Wiring it into the deal package (deal_id / Area Intelligence) is a **separate, reviewed** step —
see `INTEGRATION.md`.

## Attribution

Displaying this data requires the OGL attributions in `ATTRIBUTION.md`.

## Status & open items

See `INTEGRATION.md` §Open items. In short: core pipeline is verified end-to-end on real packs;
enrichment fields (type/date/EPC) have Home-Report **template variance** and return `None` rather
than a wrong value (hardening = a tested extractor corpus, not more regex); the divergence threshold
is a labelled `[CALIBRATE]` placeholder pending more real lots.
```
