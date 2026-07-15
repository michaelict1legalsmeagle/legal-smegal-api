"""
Regression test for guest_routes._extract_text (S42-GUEST-EXTRACT-UNIFY).

Verifies the control flow only — extract_pdf_text() and Document AI OCR
both make real network calls, so those are mocked here rather than hit
live. What this locks in:
  - non-empty extraction never triggers OCR
  - empty extraction + OCR available triggers OCR
  - empty extraction + OCR unavailable returns empty, no crash
  - a transient OCR failure is retried (up to _OCR_MAX_ATTEMPTS), not an
    immediate hard failure
  - OCR succeeding on a later attempt returns that text
  - OCR returning empty text (not raising) does NOT retry — that's a
    real "nothing to extract" result, not a transient failure

app.py itself needs supabase/anthropic/stripe/etc. to import, none of
which are installed in this test environment — and _extract_text's
`from app import extract_pdf_text` is a deferred import specifically so
guest_routes.py doesn't need app.py's dependencies at module load time.
So rather than importing the real app.py, a lightweight fake module is
injected into sys.modules['app'] before each call — this still exercises
the real deferred-import line in guest_routes.py, just resolving it to a
test double instead of the real (heavy) module, which is exactly what the
deferred-import pattern is designed to allow.
"""
import sys
import os
import types
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(__file__))

import guest_routes


def _fake_app_module(extract_pdf_text_fn):
    """Builds a minimal stand-in for app.py exposing only extract_pdf_text,
    and installs it into sys.modules so `from app import extract_pdf_text`
    resolves to it instead of trying to import the real, dependency-heavy
    app.py."""
    fake = types.ModuleType("app")
    fake.extract_pdf_text = extract_pdf_text_fn
    return fake


class TestGuestExtractText:
    def test_non_empty_extraction_never_triggers_ocr(self):
        fake_app = _fake_app_module(lambda b: ("some real text", 3))
        with patch.dict(sys.modules, {"app": fake_app}), \
             patch.object(guest_routes, "_docai_ocr", MagicMock()) as mock_ocr:
            text, pages = guest_routes._extract_text(b"fake pdf bytes", "doc.pdf")
            assert text == "some real text"
            assert pages == 3
            mock_ocr.extract_text_via_docai.assert_not_called()

    def test_empty_extraction_triggers_ocr_when_available(self):
        fake_app = _fake_app_module(lambda b: ("", 2))
        mock_ocr = MagicMock()
        mock_ocr.extract_text_via_docai.return_value = "ocr recovered text"
        with patch.dict(sys.modules, {"app": fake_app}), \
             patch.object(guest_routes, "_docai_ocr", mock_ocr):
            text, pages = guest_routes._extract_text(b"scanned pdf bytes", "scan.pdf")
            assert text == "ocr recovered text"
            assert pages == 2
            mock_ocr.extract_text_via_docai.assert_called_once()

    def test_empty_extraction_no_ocr_available_returns_empty(self):
        fake_app = _fake_app_module(lambda b: ("", 0))
        with patch.dict(sys.modules, {"app": fake_app}), \
             patch.object(guest_routes, "_docai_ocr", None):
            text, pages = guest_routes._extract_text(b"bytes", "doc.pdf")
            assert text == ""
            assert pages == 0

    def test_transient_ocr_failure_is_retried(self):
        fake_app = _fake_app_module(lambda b: ("", 1))
        mock_ocr = MagicMock()
        # Fails twice, succeeds on the third attempt
        mock_ocr.extract_text_via_docai.side_effect = [
            Exception("transient network blip"),
            Exception("quota hiccup"),
            "recovered on attempt 3",
        ]
        with patch.dict(sys.modules, {"app": fake_app}), \
             patch.object(guest_routes, "_docai_ocr", mock_ocr), \
             patch("guest_routes.time.sleep") as mock_sleep:  # don't actually wait in tests
            text, pages = guest_routes._extract_text(b"bytes", "doc.pdf")
            assert text == "recovered on attempt 3"
            assert mock_ocr.extract_text_via_docai.call_count == 3
            assert mock_sleep.call_count == 2  # backoff between attempts 1->2 and 2->3

    def test_ocr_fails_all_attempts_returns_empty_not_crash(self):
        fake_app = _fake_app_module(lambda b: ("", 0))
        mock_ocr = MagicMock()
        mock_ocr.extract_text_via_docai.side_effect = Exception("persistent failure")
        with patch.dict(sys.modules, {"app": fake_app}), \
             patch.object(guest_routes, "_docai_ocr", mock_ocr), \
             patch("guest_routes.time.sleep"):
            text, pages = guest_routes._extract_text(b"bytes", "doc.pdf")
            assert text == ""
            assert mock_ocr.extract_text_via_docai.call_count == guest_routes._OCR_MAX_ATTEMPTS

    def test_ocr_returning_empty_text_does_not_retry(self):
        # OCR succeeding (no exception) but genuinely finding no text is a
        # real result, not a transient failure — must not be retried.
        fake_app = _fake_app_module(lambda b: ("", 0))
        mock_ocr = MagicMock()
        mock_ocr.extract_text_via_docai.return_value = ""
        with patch.dict(sys.modules, {"app": fake_app}), \
             patch.object(guest_routes, "_docai_ocr", mock_ocr), \
             patch("guest_routes.time.sleep") as mock_sleep:
            text, pages = guest_routes._extract_text(b"bytes", "doc.pdf")
            assert text == ""
            assert mock_ocr.extract_text_via_docai.call_count == 1
            mock_sleep.assert_not_called()

    def test_extract_pdf_text_exception_falls_through_gracefully(self):
        # extract_pdf_text() itself raising must not crash the caller —
        # matches the try/except already in place.
        def _raise(b):
            raise Exception("hetzner unreachable and spawn failed")
        fake_app = _fake_app_module(_raise)
        with patch.dict(sys.modules, {"app": fake_app}), \
             patch.object(guest_routes, "_docai_ocr", None):
            text, pages = guest_routes._extract_text(b"bytes", "doc.pdf")
            assert text == ""
            assert pages == 0

