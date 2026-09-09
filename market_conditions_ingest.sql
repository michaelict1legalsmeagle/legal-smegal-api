-- LegalSmegal — Current Market Data: ENRICHMENT feeds (review, then run)
-- The read ALREADY fires on 3 Supabase-native signals (momentum + deceleration
-- + auction). These add votes 4–6. market_data.py readers consume them the
-- instant they exist — no code change needed after loading.
-- Isolated, additive tables. No existing object touched. Ceiling untouched.

-- ── 1. BoE rate-trend (enables the rate_trend vote) ──────────────────────────
create table if not exists bench_rate_history (
  series_code text  not null,
  as_of       date  not null,
  rate_pct    numeric not null,
  source      text default 'BoE',
  primary key (series_code, as_of)
);

-- Verified seed (Bank Rate held at 3.75% since the 18 Dec 2025 cut; unanimous/
-- majority holds Feb/Mar/Apr/Jun/Jul 2026 — BoE MPC). 6-month delta = 0 =>
-- rate_trend votes NEUTRAL (honest: a flat-rate environment). Replace/extend
-- with the full BoE IADB series (IUDBEDR, IUMBV34, IUMBV42) when you backfill.
insert into bench_rate_history (series_code, as_of, rate_pct) values
  ('IUDBEDR','2026-03-01',3.75),
  ('IUDBEDR','2026-06-01',3.75),
  ('IUDBEDR','2026-09-04',3.75)
on conflict (series_code, as_of) do update set rate_pct = excluded.rate_pct;

-- Full backfill (you run — reaches BoE from your box): BoE IADB CSV export for
-- SeriesCodes=IUDBEDR,IUMBV34,IUMBV42 -> COPY into bench_rate_history. Once the
-- fix series are in, point _read_rate_trend at IUMBV34 for a sharper vote.

-- ── 2. ONS affordability (enables the affordability vote) ────────────────────
create table if not exists ons_affordability (
  area_code text    not null,   -- LAD code, matches uk_hpi_monthly.area_code
  year      int     not null,
  ratio     numeric not null,   -- median house price / median workplace earnings
  lr_avg    numeric,            -- long-run (all-years) average ratio for the LAD
  source    text default 'ONS',
  primary key (area_code, year)
);
create index if not exists idx_ons_afford_area on ons_affordability(area_code);

-- Loader (you run — reaches ONS from your box):
--   1. Download ONS "Ratio of house price to workplace-based earnings by local
--      authority" (median), E&W, all years. Free, LAD-level.
--   2. For each LAD: insert the latest year's ratio, and set lr_avg = the mean
--      ratio across all available years (that LAD's long-run anchor).
--   The reader computes vs_lr = ratio / lr_avg; >1.05 stretched (buyer-leaning),
--   <0.95 cheap (seller-leaning). Covers the 42 English book LADs (all E-codes).

-- ── 3. Volume (Hetzner / prod only — no table needed) ────────────────────────
-- _read_volume runs against Hetzner price_paid_raw_2025 in production. It cannot
-- be reached from the build box, so it is the one feed not verifiable offline —
-- it fails safe to "unavailable" and adds a vote automatically in prod.
