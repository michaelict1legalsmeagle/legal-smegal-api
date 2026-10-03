"""PP-DELETE (2026-10-02): refresh_data._upsert_pp_rows_from_csv must DELETE
record_status 'D' transactions (bare and braced ID) instead of storing them,
and must apply rows in file order. No network, no database: the Supabase
client and the Hetzner connection are stand-ins that record calls."""
import importlib, os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _load(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "http://stub")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "stub")
    monkeypatch.setenv("DATA_DATABASE_URL", "postgresql://stub")
    fake = types.ModuleType("supabase")
    fake.create_client = lambda *a, **k: object()
    fake.Client = object
    monkeypatch.setitem(sys.modules, "supabase", fake)
    pg = types.ModuleType("psycopg"); pg.connect = lambda *a, **k: None
    pgr = types.ModuleType("psycopg.rows"); pgr.dict_row = object(); pg.rows = pgr
    monkeypatch.setitem(sys.modules, "psycopg", pg)
    monkeypatch.setitem(sys.modules, "psycopg.rows", pgr)
    sys.modules.pop("refresh_data", None)
    return importlib.import_module("refresh_data")

class Cur:
    def __init__(self, log): self.log = log; self.rowcount = 0
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def executemany(self, sql, rows): self.log.append(("upsert", [r["transaction_unique_identifier"] for r in rows]))
    def execute(self, sql, params):
        assert "DELETE FROM public.price_paid_raw_2025" in sql
        self.log.append(("delete", list(params[0]))); self.rowcount = len(params[0]) // 2
class Conn:
    def __init__(self): self.log = []; self.commits = 0; self.closed = False
    def cursor(self): return Cur(self.log)
    def commit(self): self.commits += 1
    def close(self): self.closed = True

def row(tid, status, price="250000"):
    return ",".join(f'"{v}"' for v in [tid, price, "2026-08-01 00:00", "NN8 1SF", "T", "N", "F",
        "12", "", "HIGH STREET", "", "TOWN", "DISTRICT", "COUNTY", "A", status])

def run(monkeypatch, lines):
    rd = _load(monkeypatch); conn = Conn()
    monkeypatch.setattr(rd, "_get_hetzner_conn", lambda: conn)
    count, skipped = rd._upsert_pp_rows_from_csv(iter(lines), label="test")
    return conn, count, skipped

def test_d_rows_are_deleted_not_stored(monkeypatch):
    conn, count, skipped = run(monkeypatch, [row("{AAA}", "A"), row("{BBB}", "D"), row("{CCC}", "C")])
    assert conn.log == [("upsert", ["AAA"]), ("delete", ["BBB", "{BBB}"]), ("upsert", ["CCC"])]
    assert count == 2 and skipped == 0 and conn.closed

def test_d_row_without_price_still_deletes(monkeypatch):
    conn, count, skipped = run(monkeypatch, [row("{DDD}", "D", price="")])
    assert conn.log == [("delete", ["DDD", "{DDD}"])] and count == 0 and skipped == 0

def test_no_d_rows_behaves_as_before(monkeypatch):
    conn, count, skipped = run(monkeypatch, [row("{E1}", "A"), row("{E2}", "C")])
    assert conn.log == [("upsert", ["E1", "E2"])] and count == 2 and skipped == 0

def test_batches_split_at_batch_size(monkeypatch):
    rd = _load(monkeypatch)
    n = rd.BATCH_SIZE + 3
    conn, count, _ = run(monkeypatch, [row("{X%d}" % i, "D") for i in range(n)])
    assert [k for k, _ in conn.log] == ["delete", "delete"]
    assert len(conn.log[0][1]) == 2 * rd.BATCH_SIZE and len(conn.log[1][1]) == 6
