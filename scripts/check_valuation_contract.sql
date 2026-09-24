-- check_valuation_contract.sql — V-SSOT data-contract check (read-only).
-- The frontend shows numbers ONLY for engine state 'ok' (status ok / degraded_low_comps)
-- and shows none for insufficient_evidence. That is only safe if the engine's output
-- is internally consistent. Every row returned here is a contract breach to fix in
-- the engine, not in a page. Expected result: ZERO rows.
-- Run in Supabase SQL editor (project "Final") or via the Supabase MCP.
-- Baseline 2026-09-23: 0 breaches across 92 deals × 3 ceiling objects.
SELECT d.id, d.address, v.k AS ceiling_object, v.s->>'status' AS status,
       v.s->>'comparable_valuation' AS comparable_valuation,
       CASE
         WHEN v.s->>'status' IN ('ok','degraded_low_comps')
              AND COALESCE(NULLIF(v.s->>'comparable_valuation','')::numeric,0) <= 5000
           THEN 'value-status with no comparable_valuation'
         WHEN v.s->>'status' = 'insufficient_evidence'
              AND COALESCE(NULLIF(v.s->>'comparable_valuation','')::numeric,0) > 0
           THEN 'insufficient_evidence carrying a comparable_valuation'
         WHEN v.s->>'status' = 'insufficient_evidence'
              AND COALESCE(NULLIF(v.s#>>'{valuation_range,low}','')::numeric,0) > 0
           THEN 'insufficient_evidence carrying a valuation_range'
         WHEN v.s->>'status' IS NULL AND NOT (v.s ? 'error')
           THEN 'ceiling object with no status'
       END AS breach
FROM deals d
CROSS JOIN LATERAL (VALUES
  ('verdict_ceiling',   d.summary_json::jsonb->'verdict_ceiling'),
  ('ceiling',           d.summary_json::jsonb->'ceiling'),
  ('workbench_ceiling', d.summary_json::jsonb->'workbench_ceiling')) v(k, s)
WHERE jsonb_typeof(v.s) = 'object'
  AND (
    (v.s->>'status' IN ('ok','degraded_low_comps')
       AND COALESCE(NULLIF(v.s->>'comparable_valuation','')::numeric,0) <= 5000)
 OR (v.s->>'status' = 'insufficient_evidence'
       AND (COALESCE(NULLIF(v.s->>'comparable_valuation','')::numeric,0) > 0
         OR COALESCE(NULLIF(v.s#>>'{valuation_range,low}','')::numeric,0) > 0))
 OR (v.s->>'status' IS NULL AND NOT (v.s ? 'error'))
  )
ORDER BY d.updated_at DESC;
