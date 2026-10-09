"""HEALTH-1 (10 Oct 2026) — /healthz liveness probe for Render's HTTP health check.

Before: the Render service had no health check path, so only the TCP port was probed.
These locks keep the probe cheap and dependency-free (a restart cannot fix an outside
outage and would kill in-flight background analyses) and keep it out of the rate limit.
"""
import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()


def _healthz():
    for n in ast.walk(ast.parse(APP)):
        if isinstance(n, ast.FunctionDef) and n.name == "healthz":
            return n
    raise AssertionError("healthz route missing")


def test_route_registered_get_and_head():
    fn = _healthz()
    decos = [ast.get_source_segment(APP, d) for d in fn.decorator_list]
    assert '@app.route("/healthz", methods=["GET", "HEAD"])'.lstrip("@") in decos
    assert "limiter.exempt" in decos
    assert not any("require_auth" in d for d in decos)


def test_probe_touches_no_dependency():
    body = ast.get_source_segment(APP, _healthz())
    for dep in ("supabase", "data_query", "requests.", "anthropic", "psycopg", "stripe", "Hetzner"):
        assert dep not in body, dep
    assert 'jsonify({"status": "ok"}), 200' in body


def test_probe_answers_ok_through_flask():
    """Exercise the real route on a minimal Flask app built from its source."""
    import types
    from flask import Flask, jsonify
    fn = _healthz()
    src = ast.get_source_segment(APP, fn)
    lines = [l for l in src.splitlines() if not l.startswith("@")]
    a = Flask("t")
    g = {"jsonify": jsonify}
    exec("\n".join(lines), g)
    a.add_url_rule("/healthz", "healthz", g["healthz"], methods=["GET", "HEAD"])
    c = a.test_client()
    r = c.get("/healthz")
    assert r.status_code == 200 and r.get_json() == {"status": "ok"}
    assert c.head("/healthz").status_code == 200
