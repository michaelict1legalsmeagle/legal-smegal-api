"""
ocr_fallback_service.py — second OCR engine for LegalSmegal (V-OCR-FALLBACK, 2026-09-24/26)

Runs on the Hetzner box (NOT on Render). Called by docai_ocr.py only after
Google Document AI has failed (sync and batch). Open-source Tesseract 5 via
poppler page rendering. No text is invented: output is exactly what Tesseract
reads, page-marked in the same "=== PAGE n ===" format as Document AI.

Evidence it is needed: 23 documents 'empty' after Document AI, 2 stuck
'processing' (live, 26 Sep; all application/pdf). One of them,
Lot_34_Rent_Statements_-_redacted.pdf, reads correctly with Tesseract 5.3.4
(11 pages, "Rents received for the Period: £850.00 … 15/09/2025-14/10/2025").

API
  POST /ocr     header X-OCR-Secret: <OCR_FALLBACK_SECRET>
                body: raw PDF bytes (Content-Type: application/pdf), max 40 MB
  -> 200 {"ok": true, "text": "...", "pages": n, "engine": "tesseract <ver>", "dpi": 300}
  -> 4xx/5xx {"ok": false, "error": "..."}
  GET  /health  -> {"ok": true, "engine": "tesseract <ver>"}

Memory-bounded: one page rendered and OCR'd at a time; one job at a time
(lock) — the box also runs production Postgres.
"""
import glob
import hmac
import os
import subprocess
import tempfile
import threading

from flask import Flask, jsonify, request

MAX_BYTES = 40 * 1024 * 1024
MAX_PAGES = 400
DPI = int(os.environ.get("OCR_DPI", "300"))
PAGE_TIMEOUT = int(os.environ.get("OCR_PAGE_TIMEOUT", "120"))
SECRET = os.environ.get("OCR_FALLBACK_SECRET", "")
_LOCK = threading.Lock()
app = Flask(__name__)


def _tess_version() -> str:
    try:
        return subprocess.run(["tesseract", "--version"], capture_output=True, text=True,
                              timeout=10).stdout.split("\n")[0]
    except Exception:
        return "tesseract (version unknown)"


def _page_count(pdf_path: str) -> int:
    out = subprocess.run(["pdfinfo", pdf_path], capture_output=True, text=True, timeout=30).stdout
    for line in out.splitlines():
        if line.startswith("Pages:"):
            return int(line.split()[1])
    raise RuntimeError("pdfinfo could not read page count (encrypted or corrupt PDF?)")


def ocr_pdf(pdf_bytes: bytes) -> dict:
    with tempfile.TemporaryDirectory() as td:
        pdf = os.path.join(td, "in.pdf")
        with open(pdf, "wb") as f:
            f.write(pdf_bytes)
        n = _page_count(pdf)
        if n < 1 or n > MAX_PAGES:
            raise RuntimeError(f"page count {n} outside 1..{MAX_PAGES}")
        parts = []
        for p in range(1, n + 1):
            base = os.path.join(td, f"p{p}")
            subprocess.run(["pdftoppm", "-r", str(DPI), "-gray", "-png", "-f", str(p), "-l", str(p), pdf, base],
                           check=True, capture_output=True, timeout=PAGE_TIMEOUT)
            imgs = sorted(glob.glob(base + "*.png"))
            if not imgs:
                raise RuntimeError(f"page {p} did not render")
            txt = subprocess.run(["tesseract", imgs[0], "-", "--psm", "3", "-l", "eng"],
                                 capture_output=True, text=True, timeout=PAGE_TIMEOUT).stdout
            for im in imgs:
                os.remove(im)
            parts.append(f"\n\n=== PAGE {p} ===\n\n{txt.strip()}")
        return {"text": "".join(parts), "pages": n}


@app.get("/health")
def health():
    return jsonify({"ok": True, "engine": _tess_version()})


@app.post("/ocr")
def ocr():
    if not SECRET or not hmac.compare_digest(request.headers.get("X-OCR-Secret", ""), SECRET):
        return jsonify({"ok": False, "error": "unauthorised"}), 401
    data = request.get_data(cache=False)
    if not data or len(data) > MAX_BYTES:
        return jsonify({"ok": False, "error": "empty or oversized body"}), 413
    if data[:5] != b"%PDF-":
        return jsonify({"ok": False, "error": "not a PDF"}), 415
    if not _LOCK.acquire(timeout=600):
        return jsonify({"ok": False, "error": "busy"}), 503
    try:
        r = ocr_pdf(data)
        return jsonify({"ok": True, "text": r["text"], "pages": r["pages"], "engine": _tess_version(), "dpi": DPI})
    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 422
    finally:
        _LOCK.release()
