-- S35-SIZE-MATCH migration — APPLIED TO PROD 2026-06-24
-- Supabase project "Final" (qdgxmwdvmfrcicgpukhs), name: comps_paon_passthrough_20260624
-- Adds paon + street passthrough to comp RPCs → enables address-level EPC
-- floor-area matching (size normalisation). Purely additive, backward-compatible.

DROP FUNCTION IF EXISTS public.housing_comps_v1(text, numeric, integer, text);
DROP FUNCTION IF EXISTS public.get_radius_comps(text, integer, integer, integer, text);

CREATE FUNCTION public.get_radius_comps(
  in_pcd text, in_radius_m integer DEFAULT 750, in_months integer DEFAULT 18,
  in_limit integer DEFAULT 100, in_property_type text DEFAULT NULL::text)
 RETURNS TABLE(date_of_transfer date, price integer, property_type text, postcode text,
   town_city text, meters integer, duration text, ppd_category_type text, old_new text,
   paon text, street text)
 LANGUAGE sql STABLE
AS $function$
with subject as (
  select lat, lng from public.nspl_postcodes
  where pcd_nospace = regexp_replace(upper(in_pcd), '\s+', '', 'g') limit 1
)
select g.date_of_transfer, g.price, g.property_type, g.postcode, g.town_city,
  round(earth_distance(ll_to_earth(g.nspl_lat, g.nspl_lng), ll_to_earth(s.lat, s.lng)))::int as meters,
  g.duration, g.ppd_category_type, g.old_new, g.paon, g.street
from public.price_paid_geo g cross join subject s
where g.nspl_lat is not null
  and g.date_of_transfer >= (current_date - (in_months || ' months')::interval)::date
  and earth_box(ll_to_earth(s.lat, s.lng), in_radius_m) @> ll_to_earth(g.nspl_lat, g.nspl_lng)
  and earth_distance(ll_to_earth(g.nspl_lat, g.nspl_lng), ll_to_earth(s.lat, s.lng)) <= in_radius_m
order by (in_property_type is not null and upper(g.property_type) = upper(in_property_type)) desc,
  g.date_of_transfer desc, meters asc
limit in_limit;
$function$;

CREATE FUNCTION public.housing_comps_v1(
  in_postcode text, in_radius_miles numeric, in_limit integer, in_property_type text DEFAULT NULL::text)
 RETURNS TABLE(price integer, date date, property_type text, postcode text,
   lat double precision, lng double precision, miles double precision, address text,
   town text, duration text, ppd_category_type text, old_new text, paon text, street text)
 LANGUAGE sql STABLE SECURITY DEFINER
AS $function$
with g as (
  select * from public.get_radius_comps(
    regexp_replace(upper(in_postcode), '\s+', '', 'g')::text,
    greatest(1, round(in_radius_miles * 1609.344)::int)::integer,
    24::integer, greatest(1, in_limit)::integer, in_property_type)
)
select g.price::integer, g.date_of_transfer::date as date, g.property_type::text, g.postcode::text,
  n.lat::double precision, n.lng::double precision, (g.meters::double precision / 1609.344) as miles,
  g.paon::text as address, g.town_city::text as town, g.duration::text,
  g.ppd_category_type::text, g.old_new::text, g.paon::text, g.street::text
from g left join public.nspl_lookup n on n.pcd_nospace = regexp_replace(upper(g.postcode), '\s+', '', 'g');
$function$;
