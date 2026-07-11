"""
docai_ocr.py — Google Document AI OCR fallback for scanned/image-only or
text-layer-broken PDFs.

WHY THIS EXISTS (2026-06-20):
extract_pdf_text() in app.py was OOM-crashing the Render web service (512MB
limit) on legitimately scanned/image-only legal-pack PDFs — e.g. a Local
Authority Search where the search firm flattened their output to a scanned
image per page rather than a native digital document. A "skip it" fallback
was explicitly rejected — buyer-liability documents like Local Authority
Searches must be read, not silently dropped, per the upload-report
doctrine. So this module sends genuinely image-only (or otherwise
unreadable) PDFs to Google Document AI's OCR processor instead of ever
entering the pdfplumber retry loop.

ARCHITECTURE:
This runs entirely as outbound HTTPS calls from the Render process (or,
since H-EXTRACT, from the Hetzner extraction microservice) — no local
rasterization, no Tesseract, no CPU/memory-heavy work happens in the
caller's own process.

Document AI's synchronous `:process` endpoint is capped at 15 pages. Many
real legal-pack documents exceed that, so this module always uses the
asynchronous batchProcess flow:
  1. Upload the PDF bytes to a GCS bucket (input/ prefix)
  2. Call batchProcess, which writes results to the same bucket (output/
     prefix) as one or more JSON files
  3. Poll the long-running operation until done
  4. Read each output JSON file and build page-marked text from it (see
     S-COMM-P1 below) — NOT just the flat top-level `text` field
  5. Clean up (delete) the input/output objects — this is OCR for a single
     one-off request, not a permanent archive.

S-COMM-P1 (2026-07-11): PAGE BOUNDARIES WERE BEING DISCARDED.
  The previous version of this module read only `doc_json.get("text", "")`
  from each Document AI output shard — the single flat text field — and
  joined shards with "\n\n". Document AI's own response has per-page
  structure (`document.pages[].layout.textAnchor.textSegments`, offsets
  into that shard's own `text` field) that was being thrown away. That
  made it structurally impossible for anything downstream to cite a real
  page number for OCR'd content — exactly the documents (scanned leases,
  scanned tenancy agreements) where a page citation matters most, since
  they're the ones carrying the commercial terms P1 needs to verify.
  extract_text_via_docai() now slices each shard's `text` per page using
  its own textAnchor segments and returns "=== PAGE N ==="-marked combined
  text, using each page's own `pageNumber` field (falling back to a
  running counter only if that field is absent) so page numbers stay
  correct across multiple output shards.

CONFIGURATION (env vars expected on Render / Hetzner):
  GOOGLE_APPLICATION_CREDENTIALS_JSON   - the full service-account JSON key,
                                            as a single-line string (secret
                                            env var, not a file path) --
                                            OR GOOGLE_APPLICATION_CREDENTIALS
                                            pointing directly at a file path
                                            (Hetzner's systemd unit uses this
                                            form; see extraction_service.service)
  DOCAI_PROJECT_NUMBER                  - e.g. "209319147960"
  DOCAI_LOCATION                        - e.g. "eu"
  DOCAI_PROCESSOR_ID                    - e.g. "7f6465f261336101"
  DOCAI_GCS_BUCKET                      - e.g. "legalsmegal-docai-eu"
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
    """Writes GOOGLE_APPLICATION_CREDENTIALS_JSON out to a temp file and
    points GOOGLE_APPLICATION_CREDENTIALS at it, exactly once per process
    -- but only if GOOGLE_APPLICATION_CREDENTIALS isn't already set to a
    file path (Hetzner's systemd unit sets it directly; Render supplies the
    JSON string form instead). Google's client libraries only read
    credentials from a file path (or ADC), not a raw JSON string in an env
    var -- this bridges whichever form is present to that expectation."""
    global _creds_initialized
    if _creds_initialized:
        return
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        # Already a file path (Hetzner systemd unit form) -- nothing to do.
        _creds_initialized = True
        return
    raw = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not raw:
        raise RuntimeError(
            "Neither GOOGLE_APPLICATION_CREDENTIALS nor "
            "GOOGLE_APPLICATION_CREDENTIALS_JSON is set — Document AI OCR "
            "fallback cannot authenticate."
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
    i.e. this PDF has no extractable text layer at all. Retained for any
    caller that still wants a quick pre-check, but note the extraction
    service (extraction_service_hetzner.py) no longer uses this function
    to decide OCR routing — it triggers OCR on non-whitespace YIELD from
    its own pymupdf pass instead, because block presence alone can't tell
    a genuinely scanned page apart from one with a corrupted text layer
    (see extraction_service_hetzner.py module docstring, FIX 2)."""
    try:
        import fitz  # pymupdf
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            for page in doc:
                d = page.get_text("dict")
                if any(b.get("type") == 0 for b in d.get("blocks", [])):
                    return False
            return True
        finally:
            doc.close()
    except Exception as e:
        logger.warning(f"is_image_only_pdf: detection failed ({e}) — "
                        f"assuming not image-only, falling through to "
                        f"normal extraction path")
        return False


def _page_text_from_shard(doc_json: dict) -> list:
    """Slice a single Document AI output shard's flat `text` field into
    per-page strings using each page's own layout.textAnchor.textSegments.
    Returns a list of (page_number, page_text) tuples. page_number comes
    from the shard's own pageNumber field when present; falls back to a
    1-indexed running counter within the shard otherwise (Document AI
    populates pageNumber in every real response observed, but this keeps
    the function from crashing if a future API version omits it)."""
    full_text = doc_json.get("text", "")
    pages = doc_json.get("pages", [])
    out = []
    for i, page in enumerate(pages):
        page_number = page.get("pageNumber") or (i + 1)
        segments = (
            page.get("layout", {})
            .get("textAnchor", {})
            .get("textSegments", [])
        )
        if not segments:
            out.append((page_number, ""))
            continue
        parts = []
        for seg in segments:
            start = int(seg.get("startIndex", 0) or 0)
            end = int(seg.get("endIndex", 0) or 0)
            parts.append(full_text[start:end])
        out.append((page_number, "".join(parts)))
    if not pages and full_text:
        # Defensive fallback: a shard with text but no page structure at
        # all (shouldn't happen with batchProcess, but don't silently drop
        # the text if it does) -- surface it as a single unnumbered page.
        out.append((None, full_text))
    return out


def extract_text_via_docai(file_bytes: bytes) -> str:
    """Sends file_bytes through Google Document AI's batchProcess OCR flow
    and returns page-marked text ("=== PAGE N ===" before each page, using
    Document AI's own page numbering — see _page_text_from_shard). Raises
    on any failure — callers should catch and fall through to returning
    empty text (the same degrade-gracefully behaviour the rest of the
    extraction pipeline already uses for genuinely unreadable documents)."""
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

        # S-COMM-P1: build page-marked text across all shards, using each
        # shard's own per-page textAnchor segmentation rather than each
        # shard's flat `text` field. Shards are processed in filename order
        # (Document AI names them sequentially), and each page keeps its
        # own real pageNumber, so numbering stays correct even when a large
        # document is split across multiple output shards.
        marked_parts = []
        running_counter = 0
        for blob in sorted(json_blobs, key=lambda b: b.name):
            doc_json = json.loads(blob.download_as_text())
            for page_number, page_text in _page_text_from_shard(doc_json):
                running_counter += 1
                label = page_number if page_number is not None else running_counter
                marked_parts.append(f"\n\n=== PAGE {label} ===\n\n{page_text}")

        combined = "".join(marked_parts)
        logger.info(
            f"docai_ocr: extracted {len(combined):,} chars ({running_counter} "
            f"pages) via Document AI in {time.time() - t0:.1f}s (job {job_id})"
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
