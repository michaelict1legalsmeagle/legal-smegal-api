"""
V-OCR guards (2026-09-24): every uploaded document must be read.
Run: python3 -m pytest tests/test_ocr_reliability.py -q
"""
import os, re, sys
from unittest.mock import MagicMock, patch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import docai_ocr

APP = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()


def test_batch_timeout_scales_with_size():
    assert docai_ocr._batch_timeout(745_132) >= 140            # Lot 34 rent statements (0.71 MB)
    assert docai_ocr._batch_timeout(int(14.7 * 1048576)) >= 560 # 299-page official-copy transfer
    assert docai_ocr._batch_timeout(10**9) == 900


def test_sync_first_then_batch_fallback():
    with patch.object(docai_ocr, "_extract_text_sync", side_effect=RuntimeError("page limit")) as s, \
         patch.object(docai_ocr, "_extract_text_batch", return_value="\n\n=== PAGE 1 ===\n\nX") as b:
        assert docai_ocr.extract_text_via_docai(b"%PDF") == "\n\n=== PAGE 1 ===\n\nX"
        s.assert_called_once(); b.assert_called_once()
    with patch.object(docai_ocr, "_extract_text_sync", return_value="\n\n=== PAGE 1 ===\n\nY") as s, \
         patch.object(docai_ocr, "_extract_text_batch") as b:
        assert docai_ocr.extract_text_via_docai(b"%PDF").endswith("Y")
        b.assert_not_called()


def test_page_parser_accepts_library_json_shape():
    shard = {"text": "Page one text Page two",
             "pages": [{"pageNumber": 1, "layout": {"textAnchor": {"textSegments": [{"startIndex": "0", "endIndex": "14"}]}}},
                       {"pageNumber": 2, "layout": {"textAnchor": {"textSegments": [{"startIndex": "14", "endIndex": "22"}]}}}]}
    text, n = docai_ocr._marked_text_from_shards([shard])
    assert n == 2 and "=== PAGE 2 ===" in text and text.endswith("Page two")


def test_concurrency_is_bounded():
    assert isinstance(docai_ocr._OCR_SLOTS, type(__import__("threading").BoundedSemaphore(1)))


def test_worker_is_module_level_and_recovery_exists():
    assert re.search(r"^def _run_ocr_for_document\(", APP, re.M)
    assert re.search(r"^def _recover_unread_documents\(", APP, re.M)
    assert APP.count("_requeued = _recover_unread_documents(deal_id, request.user_id)") == 2
    assert "Do not report them as missing." in APP


def test_upload_routes_unusable_text_to_ocr():
    assert "needs_ocr = (docai_ocr is not None) and _text_unusable(extracted_text, page_count)" in APP


def test_tesseract_fallback_after_document_ai_fails(monkeypatch):     # V-OCR-FALLBACK
    monkeypatch.setenv("OCR_FALLBACK_URL", "https://ocr.example")
    with patch.object(docai_ocr, "_extract_text_sync", side_effect=RuntimeError("sync")), \
         patch.object(docai_ocr, "_extract_text_batch", side_effect=TimeoutError("batch")), \
         patch.object(docai_ocr, "_extract_text_fallback", return_value="\n\n=== PAGE 1 ===\n\nT") as fb:
        assert docai_ocr.extract_text_via_docai(b"%PDF").endswith("T")
        fb.assert_called_once()
    monkeypatch.delenv("OCR_FALLBACK_URL")
    with patch.object(docai_ocr, "_extract_text_sync", side_effect=RuntimeError("sync")), \
         patch.object(docai_ocr, "_extract_text_batch", side_effect=TimeoutError("batch")):
        try:
            docai_ocr.extract_text_via_docai(b"%PDF"); assert False, "should raise"
        except TimeoutError:
            pass


def test_fallback_service_reads_a_scanned_pdf():
    """Runs the real service on an image-only PDF (built here: text drawn onto an
    image, no text layer). Skipped where tesseract/poppler are not installed (CI)."""
    import shutil, pytest
    if not (shutil.which("tesseract") and shutil.which("pdftoppm") and shutil.which("pdfinfo")):
        pytest.skip("tesseract/poppler not available here")
    PIL = pytest.importorskip("PIL")
    from PIL import Image, ImageDraw, ImageFont
    import io, importlib
    img = Image.new("L", (1700, 2200), 255)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 48)
    except Exception:
        pytest.skip("no truetype font for the synthetic scan")
    d.text((150, 300), "Rents received for the Period: 850.00", fill=0, font=font)
    d.text((150, 400), "15/09/2025-14/10/2025", fill=0, font=font)
    buf = io.BytesIO(); img.save(buf, format="PDF", resolution=200)
    sys.path.insert(0, os.path.join(ROOT, "hetzner", "ocr_fallback"))
    os.environ["OCR_FALLBACK_SECRET"] = "t"
    import ocr_fallback_service as svc
    importlib.reload(svc)
    c = svc.app.test_client()
    assert c.post("/ocr", data=buf.getvalue(), headers={"X-OCR-Secret": "bad"}).status_code == 401
    assert c.post("/ocr", data=b"not a pdf", headers={"X-OCR-Secret": "t"}).status_code == 415
    r = c.post("/ocr", data=buf.getvalue(), headers={"X-OCR-Secret": "t"})
    assert r.status_code == 200 and r.json["pages"] == 1
    assert "=== PAGE 1 ===" in r.json["text"] and "Rents received for the Period" in r.json["text"]
