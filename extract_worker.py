"""
extract_worker.py — spawn-context PDF text extraction worker.

Imported ONLY by the spawn child process created in extract_pdf_text()
(app.py). Intentionally imports nothing from app.py. The spawn child
starts a fresh Python interpreter and only loads this file and its
direct dependencies — fitz (~20MB) vs the full gunicorn worker
(~175MB with Flask + all app.py imports). This is why spawn was chosen
over fork: the child memory footprint drops from ~175MB to ~25MB,
which eliminates the OOM pressure from concurrent extractions on
Render's 512MB Starter plan.

Why docai_ocr is NOT here (unlike the old fork worker):
  The caller (upload_document in app.py) already ran is_image_only_pdf()
  before calling extract_pdf_text(). needs_ocr=False means the PDF has
  a confirmed embedded text layer — pymupdf should succeed. The
  docai_ocr fallback inside the old fork worker pre-dated the
  is_image_only_pdf gate and was belt-and-suspenders that in practice
  was never reached on the fast path. If pymupdf fails here, the result
  is empty text — the same outcome as the current "both methods failed"
  path, and no worse than Document AI returning nothing on a malformed
  PDF. 2026-07-01 (spawn refactor, H-SPAWN).
"""


def extract_pdf_text_worker(file_bytes: bytes, result_queue) -> None:
    """
    Run in a spawn child process. Extracts text from a PDF using pymupdf.
    Puts (text: str, page_count: int) onto result_queue.
    Always puts a result — never raises out of the function, so the
    parent's q.get_nowait() is always safe after a clean exit.
    """
    page_count = 0
    try:
        import fitz  # pymupdf — the only import this worker needs
        doc        = fitz.open(stream=file_bytes, filetype="pdf")
        page_count = len(doc)
        text_parts = []
        for page in doc:
            text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            if text and text.strip():
                text_parts.append(text)
        doc.close()
        combined = "\n\n".join(text_parts)
        if combined.strip() and len(combined) > page_count * 50:
            print(
                f"⏱️ [extract_pdf_text] pymupdf extracted "
                f"{len(combined):,} chars from {page_count} pages"
            )
            result_queue.put((combined, page_count))
            return
        print(
            f"⏱️ [extract_pdf_text] pymupdf low yield "
            f"({len(combined)} chars, {page_count} pages) — returning empty"
        )
    except ImportError:
        print("⏱️ [extract_pdf_text] pymupdf not available — returning empty")
    except Exception as exc:
        print(f"⏱️ [extract_pdf_text] pymupdf failed: {exc} — returning empty")

    result_queue.put(("", page_count))
