-- RENT-PIPR-1 (10 Oct 2026): uk_prms_monthly gains the ONS PIPR area and region names.
-- Additive and nullable: existing readers are unaffected. Run BEFORE deploying the
-- RENT-PIPR-1 API commit (the refresh writes these columns; the regional rent
-- benchmark joins a local authority to its region by region_name).
ALTER TABLE public.uk_prms_monthly
  ADD COLUMN IF NOT EXISTS area_name   text,
  ADD COLUMN IF NOT EXISTS region_name text;

CREATE INDEX IF NOT EXISTS uk_prms_monthly_area_period_idx
  ON public.uk_prms_monthly (area_code, period DESC);
CREATE INDEX IF NOT EXISTS uk_prms_monthly_name_period_idx
  ON public.uk_prms_monthly (area_name, period);
