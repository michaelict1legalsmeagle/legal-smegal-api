-- CRIME-EVID-1 (8 Oct 2026): Home Office police recorded crime by Community
-- Safety Partnership, England & Wales, on Hetzner (legalsmegal_data).
-- Isolated schema `ew`, additive only. Run ON THE BOX as postgres:
--   sudo -u postgres psql -P pager=off -d legalsmegal_data -v ON_ERROR_STOP=1 -f 20261008_ew_crime_by_csp.sql
-- Data is loaded separately by load_crime_by_csp.py (and refreshed by the
-- monthly refresh_data.py cron when gov.uk publishes a newer file).
BEGIN;

CREATE SCHEMA IF NOT EXISTS ew;

CREATE TABLE IF NOT EXISTS ew.crime_by_csp (
    financial_year  text        NOT NULL,   -- e.g. '2025/26'
    police_force    text        NOT NULL,   -- e.g. 'Greater Manchester'
    csp_name        text        NOT NULL,   -- e.g. 'Manchester' (= local authority name)
    offence_group   text        NOT NULL,   -- Home Office offence group
    offences        integer     NOT NULL,   -- sum of the year's quarters
    quarters        text        NOT NULL,   -- quarters present, e.g. '1,2,3,4'
    usable          boolean     NOT NULL,   -- false = Home Office note / 'Unassigned' group
    quality_note    text,                   -- verbatim Home Office note when unusable
    source_file     text        NOT NULL,   -- e.g. 'prc-csp-mar2021-mar2026-tables-230726.ods'
    published       text,                   -- 'Updated:' date from the file's Notes sheet
    loaded_at       timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (financial_year, police_force, csp_name, offence_group)
);

CREATE INDEX IF NOT EXISTS crime_by_csp_csp_idx ON ew.crime_by_csp (csp_name, financial_year);

-- App role: read for the API; insert/delete for the monthly refresh cron.
GRANT USAGE ON SCHEMA ew TO legalsmegal;
GRANT SELECT, INSERT, DELETE ON ew.crime_by_csp TO legalsmegal;

COMMIT;
