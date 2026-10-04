-- DOCS-1 (2026-10-04): the list of files the user selected for a deal
-- ({files:[{name, stored_name, size, parts[]}], created_at, accepted_missing_at?}).
-- Summarise compares it with the stored documents so a file that never reached
-- the server is named as "not received" instead of disappearing (Lot 6: 6 selected,
-- 3 stored, analysed as complete). Additive and nullable: no existing row or
-- reader changes. Rollback: alter table public.deals drop column pack_manifest;
alter table public.deals add column if not exists pack_manifest jsonb;
