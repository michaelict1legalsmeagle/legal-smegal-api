-- ============================================================================
-- S33-TYPE-MATCH  (2026-06-23)
-- Like-for-like comp prioritisation: add optional in_property_type to the comp
-- functions so same-type sales (terraced↔terraced, semi↔semi, detached↔detached)
-- are sorted to the top of the radius window BEFORE the limit truncates it.
--
-- WHY: get_radius_comps orders by date_of_transfer DESC then distance, then
-- LIMITs. In a terrace-heavy sector the most-recent N sales can be almost all
-- terraces, crowding out same-type comps before app.py's client-side filter
-- ever sees them. Live failure: 104 Village St DE23 (a SEMI) was valued off
-- 5 terraces → £160k median → hammer was £216k.
--
-- DESIGN: PRIORITISE, do not EXCLUDE. When in_property_type is supplied, same-
-- type rows sort first; everything else still returns (graceful on thin
-- sectors). When NULL, behaviour is byte-for-byte identical to today, so all
-- existing callers are unaffected. The new param has a DEFAULT, so the old
-- 4-arg / 3-arg call signatures keep working.
--
-- ============================================================================
-- DEPLOYMENT RECORD — APPLIED TO PROD (project "Final", qdgxmwdvmfrcicgpukhs)
-- on 2026-06-23. This file is the faithful record of what is live.
--
-- IMPORTANT — applied in TWO parts (PART A then PART B below). BOTH are required.
-- Adding the DEFAULT-param overloads alone (PART A) created a "function is not
-- unique" ambiguity: a 3-arg call (every existing caller, incl. app.py's named
-- {in_postcode,in_radius_miles,in_limit}) could match EITHER the old 3-arg
-- version OR the new 4-arg-with-default version, and Postgres refused to choose
-- ("42725: function ... is not unique"). Caught on prod verification BEFORE the
-- app.py deploy — had it shipped, every comp query would have failed. PART B
-- drops the old signatures, leaving only the new versions; their DEFAULT NULL
-- means 3-arg callers resolve cleanly with byte-for-byte old behaviour.
--
-- VERIFIED ON PROD against 104 Village St, DE23 8DF (a SEMI, hammer £216,000):
--   untyped 3-arg call .... T=52/S=36/D=12, medians 130k/213.5k/222.5k (unchanged)
--   typed  'S' call ....... 50 rows, 100% S, median £205,000 (was £160k contaminated)
--
-- DEPLOY ORDER: this migration is ALREADY LIVE. Deploy the app.py that sends
-- in_property_type next. (app.py also has a fallback that strips the param if a
-- typed call ever errors, so it degrades gracefully rather than breaking comps.)
-- ============================================================================

-- ╔═══════════════════════════════════════════════════════════════════════════
-- ║ PART A — CREATE OR REPLACE with new optional in_property_type param
-- ╚═══════════════════════════════════════════════════════════════════════════

-- ── 1. Inner function: get_radius_comps (live 4-arg overload) ───────────────
CREATE OR REPLACE FUNCTION public.get_radius_comps(
  in_pcd text,
  in_radius_m integer DEFAULT 750,
  in_months integer DEFAULT 18,
  in_limit integer DEFAULT 100,
  in_property_type text DEFAULT NULL          -- NEW, optional; NULL = old behaviour
)
RETURNS TABLE(date_of_transfer date, price integer, property_type text,
              postcode text, town_city text, meters integer,
              duration text, ppd_category_type text, old_new text)
LANGUAGE sql STABLE
AS $function$
with subject as (
  select lat, lng
  from public.nspl_postcodes
  where pcd_nospace = regexp_replace(upper(in_pcd), '\s+', '', 'g')
  limit 1
)
select
  g.date_of_transfer,
  g.price,
  g.property_type,
  g.postcode,
  g.town_city,
  round(earth_distance(ll_to_earth(g.nspl_lat, g.nspl_lng), ll_to_earth(s.lat, s.lng)))::int as meters,
  g.duration,
  g.ppd_category_type,
  g.old_new
from public.price_paid_geo g
cross join subject s
where g.nspl_lat is not null
  and g.date_of_transfer >= (current_date - (in_months || ' months')::interval)::date
  and earth_box(ll_to_earth(s.lat, s.lng), in_radius_m) @> ll_to_earth(g.nspl_lat, g.nspl_lng)
  and earth_distance(ll_to_earth(g.nspl_lat, g.nspl_lng), ll_to_earth(s.lat, s.lng)) <= in_radius_m
order by
  -- NEW: same-type first when a type is supplied; NULL leaves order unchanged
  (in_property_type is not null and upper(g.property_type) = upper(in_property_type)) desc,
  g.date_of_transfer desc,
  meters asc
limit in_limit;
$function$;

-- ── 2. Wrapper: housing_comps_v1 (live named-arg overload) ───────────────────
CREATE OR REPLACE FUNCTION public.housing_comps_v1(
  in_postcode text,
  in_radius_miles numeric,
  in_limit integer,
  in_property_type text DEFAULT NULL          -- NEW, optional; forwarded down
)
RETURNS TABLE(price integer, date date, property_type text, postcode text,
              lat double precision, lng double precision, miles double precision,
              address text, town text, duration text, ppd_category_type text, old_new text)
LANGUAGE sql STABLE SECURITY DEFINER
AS $function$
with g as (
  select * from public.get_radius_comps(
    regexp_replace(upper(in_postcode), '\s+', '', 'g')::text,
    greatest(1, round(in_radius_miles * 1609.344)::int)::integer,
    24::integer,
    greatest(1, in_limit)::integer,
    in_property_type                            -- NEW: forwarded
  )
)
select
  g.price::integer,
  g.date_of_transfer::date as date,
  g.property_type::text,
  g.postcode::text,
  n.lat::double precision,
  n.lng::double precision,
  (g.meters::double precision / 1609.344) as miles,
  null::text as address,
  g.town_city::text as town,
  g.duration::text,
  g.ppd_category_type::text,
  g.old_new::text
from g
left join public.nspl_lookup n
  on n.pcd_nospace = regexp_replace(upper(g.postcode), '\s+', '', 'g');
$function$;

-- ============================================================================
-- ╔═══════════════════════════════════════════════════════════════════════════
-- ║ PART B — DROP the original signatures (REQUIRED — resolves the ambiguity)
-- ╚═══════════════════════════════════════════════════════════════════════════
-- Drops ONLY the exact original signatures (without in_property_type). The new
-- 4th-arg versions from PART A remain and absorb all existing call styles.
DROP FUNCTION IF EXISTS public.housing_comps_v1(in_postcode text, in_radius_miles numeric, in_limit integer);
DROP FUNCTION IF EXISTS public.get_radius_comps(in_pcd text, in_radius_m integer, in_months integer, in_limit integer);

-- ============================================================================
-- VERIFY (these were RUN ON PROD and PASSED — re-run any time to confirm):
--
--   -- untyped 3-arg call (existing callers) — must be unchanged:
--   select property_type, count(*),
--          round(percentile_cont(0.5) within group (order by price)) median
--   from housing_comps_v1(in_postcode => 'DE23 8DF', in_radius_miles => 0.75, in_limit => 100)
--   group by property_type order by count(*) desc;
--   -- EXPECT: T=52/S=36/D=12, medians 130000/213500/222500
--
--   -- typed 'S' call (new app.py path) — semis prioritised to top of window:
--   select property_type, count(*),
--          round(percentile_cont(0.5) within group (order by price)) median
--   from housing_comps_v1(in_postcode => 'DE23 8DF', in_radius_miles => 0.75, in_limit => 50, in_property_type => 'S')
--   group by property_type order by count(*) desc;
--   -- EXPECT: 50 rows, 100% S, median 205000
-- ============================================================================

-- ── ROLLBACK ────────────────────────────────────────────────────────────────
-- To revert: recreate the ORIGINAL signatures (3-arg housing_comps_v1, 4-arg
-- get_radius_comps WITHOUT in_property_type) from the definitions captured in
-- Supabase_Snippet_Untitled_query.csv, THEN drop the in_property_type versions:
--   DROP FUNCTION IF EXISTS public.housing_comps_v1(text, numeric, integer, text);
--   DROP FUNCTION IF EXISTS public.get_radius_comps(text, integer, integer, integer, text);
-- Note: because PART B dropped the originals, a clean rollback must RECREATE
-- them first (not just CREATE OR REPLACE) — keep the CSV definitions as the
-- source of truth for the pre-migration bodies.
