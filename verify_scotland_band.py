import sys, psycopg
DSN = "dbname=legalsmegal_data"
AB16_EXPECT = 119571
N_DZ2011 = 6976
PACKS = ["KA24 4AA", "G33 1FD", "G73 5PL", "KA5 6PW", "G13 1UQ"]
def simd_decile(rank):
    if not rank: return None
    return ((rank - 1) * 10 // N_DZ2011) + 1
def load_ref(conn):
    with conn.cursor() as c:
        c.execute("SELECT area_code, median_2023 FROM scotland_area_price")
        price = dict(c.fetchall())
        c.execute("SELECT la_code, current_median, current_period, median_2023 FROM scotland_la_median")
        cur, y2023, period = {}, {}, None
        for la, cm, per, m23 in c.fetchall():
            if cm is not None: cur[la] = cm; period = period or per
            if m23 is not None: y2023[la] = m23
    return price, cur, y2023, period
def geo(conn, pc):
    pc = (pc or "").replace(" ", "").upper()
    with conn.cursor() as c:
        c.execute("SELECT dz2011, iz2011, la_code, simd2020_rank, ur6, lat, lng "
                  "FROM scotland_postcode WHERE postcode=%s", (pc,))
        return c.fetchone()
def build_band(conn, price, cur, y2023, period, pc):
    g = geo(conn, pc)
    if not g: return {"status": "unavailable", "reason": "not in SPD index"}
    dz, iz, la, rank, ur6, lat, lng = g
    tier = None
    if dz in price: tier = ("Data Zone (2011)", dz, price[dz])
    elif iz in price: tier = ("Intermediate Zone (2011)", iz, price[iz])
    elif la in price: tier = ("Local Authority", la, price[la])
    if not tier: return {"status": "unavailable", "reason": "no unsuppressed area figure"}
    band_2023 = tier[2]; factor = adj = None
    if la in cur and la in y2023 and y2023[la]:
        factor = cur[la] / y2023[la]; adj = round(band_2023 * factor)
    return {"status": "ok", "geography_level": tier[0], "area_code": tier[1],
            "band_median_2023": band_2023, "time_adjust_factor": round(factor, 4) if factor else None,
            "band_median_current": adj, "simd2020_decile": simd_decile(rank), "ur6": ur6}
conn = psycopg.connect(DSN, autocommit=True, options="-c search_path=scotland")
price, cur, y2023, period = load_ref(conn)
fails = 0
b = build_band(conn, price, cur, y2023, period, "AB16 5TL")
ok1 = b.get("status") == "ok" and b.get("band_median_current") == AB16_EXPECT
print("[gate1] AB16 5TL band_current=%s (expect %s) -> %s" % (b.get("band_median_current"), AB16_EXPECT, "PASS" if ok1 else "FAIL"))
fails += 0 if ok1 else 1
b2 = build_band(conn, price, cur, y2023, period, "E6 2AU")
ok2 = b2.get("status") != "ok"
print("[gate2] E6 2AU (English) status=%s -> %s" % (b2.get("status"), "PASS (no band)" if ok2 else "FAIL"))
fails += 0 if ok2 else 1
print("[gate3] test-pack postcodes -> real bands:")
for pc in PACKS:
    bb = build_band(conn, price, cur, y2023, period, pc)
    if bb.get("status") == "ok":
        print("    %-9s %-24s current=%s" % (pc, bb["geography_level"], bb["band_median_current"]))
    else:
        print("    %-9s UNAVAILABLE: %s" % (pc, bb.get("reason")))
print("RESULT:", "GATES 1&2 PASS" if fails == 0 else ("%d FAILED" % fails))
sys.exit(1 if fails else 0)
