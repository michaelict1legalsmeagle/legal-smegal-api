"""
Guard: no in-process PDF parsing on the 512MB Render web dyno (H-NOFITZ, 2026-09-23).
Every local fitz/pdfplumber open in the web service has been an OOM that took the
whole site down. Extraction = Hetzner service; OCR = Document AI. This test fails
the build if either library is opened again in the web-service modules.
Also guards that both products share the single classifier.
Run: python3 -m pytest tests/test_no_local_pdf_parse.py -q   (from repo root)
"""
import os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_MODULES = ["app.py", "guest_routes.py"]
FORBIDDEN = [r"\bfitz\.open\(", r"\bpdfplumber\.open\(", r"\b_pdp\.open\(", r"\b_fitz\.open\(",
             r"^\s*import fitz\b", r"^\s*import fitz as\b"]


def _code_lines(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.lstrip().startswith("#"):
                continue
            yield n, line


def test_no_local_pdf_open_in_web_modules():
    hits = []
    for mod in WEB_MODULES:
        for n, line in _code_lines(mod):
            for pat in FORBIDDEN:
                if re.search(pat, line):
                    hits.append(f"{mod}:{n}: {line.strip()}")
    assert not hits, "Local PDF parsing on the web dyno:\n" + "\n".join(hits)


def test_guest_pipeline_uses_shared_classifier():
    src = open(os.path.join(ROOT, "guest_routes.py"), encoding="utf-8").read()
    assert "from doc_classifier import classify_document" in src


def test_docai_page_marker_count():
    # same expression used in app.py background OCR and guest_routes._extract_text
    t = "\n\n=== PAGE 1 ===\n\nabc\n\n=== PAGE 2 ===\n\n\n\n=== PAGE 3 ===\n\nx"
    assert len(re.findall(r"=== PAGE \S+ ===", t)) == 3
