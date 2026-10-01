-- FIN-CORE (30 Sep 2026): seed the two comparison benchmark rows so
-- boe_rates_sync.py (update-by-series_code) can fill them. rate_pct / as_of stay
-- NULL until the first sync run — the Financials page shows "—" until then.
-- Idempotent.
INSERT INTO public.bench_rates (series_code, label, rate_pct, as_of, source)
SELECT 'IUMB6RH', '2yr fixed-rate savings bond (households, quoted)', NULL, NULL, 'Bank of England IADB'
WHERE NOT EXISTS (SELECT 1 FROM public.bench_rates WHERE series_code = 'IUMB6RH');
INSERT INTO public.bench_rates (series_code, label, rate_pct, as_of, source)
SELECT 'IUMAMNPY', '10yr gilt nominal par yield (monthly average)', NULL, NULL, 'Bank of England IADB'
WHERE NOT EXISTS (SELECT 1 FROM public.bench_rates WHERE series_code = 'IUMAMNPY');
-- Check after the sync has run (expect both rows with a rate and a recent as_of):
-- SELECT series_code, rate_pct, as_of FROM public.bench_rates ORDER BY series_code;
