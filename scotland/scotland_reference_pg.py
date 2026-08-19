"""
scotland_reference_pg.py  —  runtime reference backed by Postgres.

Returns the SAME REF shape the engine expects, but:
  * the 157k-row postcode index is queried lazily per postcode (indexed PK lookup),
    so there is NO 100 MB startup parse and no big dict held in RAM;
  * the tiny price (~7.5k) and LA-median (~32) tables are loaded once into dicts.

The engine (scotland_band_engine) is untouched: build_band() calls ref["pc_index"].get(pc),
and PgPostcodeIndex.get() satisfies that interface from the database.

    from scotland_reference_pg import load_reference_pg
    REF = load_reference_pg(dsn)              # once per worker at startup
    ...
    band = scotland_band_engine.build_band(deal_postcode, REF)   # per request

Concurrency: one connection per process worker (psycopg connections are not shared
across threads). For threaded servers, pass a connection pool's getconn or wrap get().
"""
import psycopg


class PgPostcodeIndex:
    """Lazy dict-like over scotland_postcode. Implements .get(postcode) and len()."""
    def __init__(self, conn):
        self._conn = conn
        self._count = None

    def get(self, postcode, default=None):
        pc = (postcode or "").replace(" ", "").upper()
        with self._conn.cursor() as c:
            c.execute("SELECT dz2011, iz2011, la_code, simd2020_rank, ur6, lat, lng "
                      "FROM scotland_postcode WHERE postcode = %s", (pc,))
            row = c.fetchone()
        if not row:
            return default
        return {"dz2011": row[0], "iz2011": row[1], "la_code": row[2],
                "simd2020_rank": row[3], "ur6": row[4], "lat": row[5], "long": row[6]}

    def __len__(self):
        if self._count is None:
            with self._conn.cursor() as c:
                c.execute("SELECT count(*) FROM scotland_postcode")
                self._count = c.fetchone()[0]
        return self._count


def load_reference_pg(dsn):
    """Return REF backed by Postgres. Keep the returned REF for the worker's lifetime."""
    conn = psycopg.connect(dsn, autocommit=True)

    with conn.cursor() as c:
        c.execute("SELECT area_code, median_2023 FROM scotland_area_price")
        price = {code: med for code, med in c.fetchall()}

        c.execute("SELECT la_code, current_median, current_period, median_2023 "
                  "FROM scotland_la_median")
        cur, y2023, period = {}, {}, None
        for la, cur_med, per, m2023 in c.fetchall():
            if cur_med is not None:
                cur[la] = cur_med
                period = period or per
            if m2023 is not None:
                y2023[la] = m2023

    return {"pc_index": PgPostcodeIndex(conn), "price_2023": price,
            "ros_cur": cur, "ros_period": period, "ros_2023": y2023, "_conn": conn}
