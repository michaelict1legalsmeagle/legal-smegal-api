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
