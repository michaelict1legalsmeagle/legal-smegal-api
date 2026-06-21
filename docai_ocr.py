"""
docai_ocr.py — Google Document AI OCR fallback for scanned/image-only PDFs.

WHY THIS EXISTS (2026-06-20):
extract_pdf_text() in app.py was OOM-crashing the Render web service (512MB
limit) on legitimately scanned/image-only legal-pack PDFs — e.g. a Local
Authority Search where the search firm flattened their output to a scanned
image per page rather than a native digital document. On such files:

  - pymupdf returns 0 chars on every page (correctly — there is no text
    layer to extract; confirmed via page.get_text('dict') showing 100%
    image-type blocks, 0 text-type blocks, across all pages).
  - pdfplumber's double-pass retry (strict tolerance, then loose tolerance)
    was then attempted anyway, burning CPU and leaking ~30-40MB of memory
    per page with zero chance of success, crossing the 512MB ceiling at
    roughly the 28-second mark on a 32-page document — a near-exact race
    against the 25s extract_pdf_text() timeout, and a race the timeout can
    easily lose once real Flask/gunicorn process overhead is added on top
    of the bare-metal measurement this was based on.

A "skip it" fallback was explicitly rejected — buyer-liability documents
like Local Authority Searches must be read, not silently dropped, per the
upload-report doctrine (missing a land charge or planning enforcement
entry is a real liability gap, not a cosmetic one). So this module sends
genuinely image-only PDFs to Google Document AI's OCR processor instead of
ever entering the pdfplumber retry loop.

ARCHITECTURE:
This runs entirely as outbound HTTPS calls from the Render process — no
local rasterization, no Tesseract, no CPU/memory-heavy work happens in the
Render container itself. That keeps the 512MB ceiling out of the picture
for this code path regardless of document page count.

Document AI's synchronous `:process` endpoint is capped at 15 pages. Many
real legal-pack documents (this trigger file is 32 pages) exceed that, so
this module always uses the asynchronous batchProcess flow:
  1. Upload the PDF bytes to a GCS bucket (input/ prefix)
  2. Call batchProcess, which writes results to the same bucket (output/
     prefix) as one or more JSON files
  3. Poll the long-running operation until done
  4. Read and concatenate the output JSON files' text fields
  5. Clean up (delete) the input/output objects — this is OCR for a single
     one-off request, not a permanent archive; nothing should accumulate
     in the bucket over time.

CONFIGURATION (env vars expected on Render):
  GOOGLE_APPLICATION_CREDENTIALS_JSON   - the full service-account JSON key,
                                            as a single-line string (Render
                                            secret env var, not a file path)
  DOCAI_PROJECT_NUMBER                  - e.g. "209319147960"
  DOCAI_LOCATION                        - e.g. "eu"
  DOCAI_PROCESSOR_ID                    - e.g. "7f6465f261336101"
  DOCAI_GCS_BUCKET                     - e.g. "legalsmegal-docai-eu"
"""

import io
import json
import logging
import os
import tempfile
import time
import uuid

logger = logging.getLogger(__name__)

_DOCAI_TIMEOUT_SECONDS = 90  # hard ceiling on the whole OCR round-trip
_POLL_INTERVAL_SECONDS = 3

_docai_client = None
_storage_client = None
_creds_initialized = False


def _ensure_credentials():
    """Writes the GOOGLE_APPLICATION_CREDENTIALS_JSON env var out to a temp
    file and points GOOGLE_APPLICATION_CREDENTIALS at it, exactly once per
    process. Google's client libraries only know how to read credentials
    from a file path (or ADC), not from a raw JSON string in an env var —
    this bridges Render's "secret env var" model to that expectation."""
    global _creds_initialized
    if _creds_initialized:
        return
    raw = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not raw:
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS_JSON is not set — Document AI "
            "OCR fallback cannot authenticate."
        )
    # Validate it's well-formed JSON before writing, so a misconfigured env
    # var fails loudly here rather than producing a cryptic auth error later.
    json.loads(raw)
    fd, path = tempfile.mkstemp(prefix="docai-creds-", suffix=".json")
    with os.fdopen(fd, "w") as f:
        f.write(raw)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
    _creds_initialized = True


def _get_docai_client():
    global _docai_client
    if _docai_client is None:
        _ensure_credentials()
        from google.cloud import documentai
        location = os.environ.get("DOCAI_LOCATION", "eu")
        api_endpoint = f"{location}-documentai.googleapis.com"
        _docai_client = documentai.DocumentProcessorServiceClient(
            client_options={"api_endpoint": api_endpoint}
        )
    return _docai_client


def _get_storage_client():
    global _storage_client
    if _storage_client is None:
        _ensure_credentials()
        from google.cloud import storage
        _storage_client = storage.Client()
    return _storage_client


def is_image_only_pdf(file_bytes: bytes) -> bool:
    """True if pymupdf finds zero text-type blocks across every page —
    i.e. this PDF has no extractable text layer at all (a genuine
    scanned/flattened-image document), as opposed to a normal PDF where
    pymupdf merely garbled some encoding.

    This check is what decides whether we skip straight to Document AI
    OCR instead of ever entering the pdfplumber double-pass retry loop —
    see the module docstring for why that distinction matters for the
    512MB OOM issue.
    """
    try:
        import fitz  # pymupdf
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            for page in doc:
                d = page.get_text("dict")
                if any(b.get("type") == 0 for b in d.get("blocks", [])):
                    # Found at least one real text block somewhere — this
                    # is not a pure scanned document, so let the normal
                    # pymupdf/pdfplumber flow continue to handle it.
                    return False
            return True
        finally:
            doc.close()
    except Exception as e:
        logger.warning(f"is_image_only_pdf: detection failed ({e}) — "
                        f"assuming not image-only, falling through to "
                        f"normal extraction path")
        return False


def extract_text_via_docai(file_bytes: bytes) -> str:
    """Sends file_bytes through Google Document AI's batchProcess OCR flow
    and returns the extracted text. Raises on any failure — callers should
    catch and fall through to returning empty text (the same degrade-
    gracefully behaviour the rest of extract_pdf_text already uses for
    genuinely unreadable documents)."""
    t0 = time.time()
    bucket_name = os.environ["DOCAI_GCS_BUCKET"]
    project_number = os.environ["DOCAI_PROJECT_NUMBER"]
    location = os.environ.get("DOCAI_LOCATION", "eu")
    processor_id = os.environ["DOCAI_PROCESSOR_ID"]

    job_id = uuid.uuid4().hex
    input_blob_name = f"input/{job_id}.pdf"
    output_prefix = f"output/{job_id}/"

    storage_client = _get_storage_client()
    bucket = storage_client.bucket(bucket_name)

    input_blob = bucket.blob(input_blob_name)
    output_blobs = []
    try:
        input_blob.upload_from_string(file_bytes, content_type="application/pdf")

        from google.cloud import documentai

        docai_client = _get_docai_client()
        processor_name = (
            f"projects/{project_number}/locations/{location}"
            f"/processors/{processor_id}"
        )

        gcs_document = documentai.GcsDocument(
            gcs_uri=f"gs://{bucket_name}/{input_blob_name}",
            mime_type="application/pdf",
        )
        input_config = documentai.BatchDocumentsInputConfig(
            gcs_documents=documentai.GcsDocuments(documents=[gcs_document])
        )
        output_config = documentai.DocumentOutputConfig(
            gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(
                gcs_uri=f"gs://{bucket_name}/{output_prefix}"
            )
        )
        request = documentai.BatchProcessRequest(
            name=processor_name,
            input_documents=input_config,
            document_output_config=output_config,
        )

        operation = docai_client.batch_process_documents(request)

        # Poll rather than operation.result(timeout=...) directly, so we can
        # enforce our own hard ceiling and log progress — a hung operation
        # should not be able to hold this request open indefinitely.
        while not operation.done():
            if time.time() - t0 > _DOCAI_TIMEOUT_SECONDS:
                raise TimeoutError(
                    f"Document AI batchProcess did not complete within "
                    f"{_DOCAI_TIMEOUT_SECONDS}s"
                )
            time.sleep(_POLL_INTERVAL_SECONDS)

        # Surfaces any error captured in the operation itself (e.g. a
        # malformed/corrupt PDF that even Document AI can't open).
        operation.result()

        output_blobs = list(storage_client.list_blobs(
            bucket_name, prefix=output_prefix
        ))
        json_blobs = [b for b in output_blobs if b.name.endswith(".json")]
        if not json_blobs:
            raise RuntimeError(
                "Document AI batchProcess completed but produced no JSON "
                "output — treating as extraction failure."
            )

        text_parts = []
        for blob in sorted(json_blobs, key=lambda b: b.name):
            doc_json = json.loads(blob.download_as_text())
            text = doc_json.get("text", "")
            if text:
                text_parts.append(text)

        combined = "\n\n".join(text_parts)
        logger.info(
            f"docai_ocr: extracted {len(combined):,} chars via Document AI "
            f"in {time.time() - t0:.1f}s (job {job_id})"
        )
        return combined

    finally:
        # Best-effort cleanup — this bucket is a transient OCR scratch
        # space, not a document archive. Failures here are logged but
        # never raised, so a cleanup hiccup can't turn a successful OCR
        # result into a failed request.
        try:
            input_blob.delete()
        except Exception as e:
            logger.warning(f"docai_ocr: failed to delete input blob: {e}")
        for blob in output_blobs:
            try:
                blob.delete()
            except Exception as e:
                logger.warning(f"docai_ocr: failed to delete output blob "
                                f"{blob.name}: {e}")
