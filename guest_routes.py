"""
guest_routes.py — LegalSmegal one-off report pipeline (Flow 2)
================================================================
Completely independent of the subscriber pipeline.
Sessions backed by Supabase (guest_sessions table) — survives
restarts, scales across multiple workers, no Redis required.

Flow:
  1. POST /api/guest2/create-session   — create Supabase session row
  2. POST /api/guest2/upload           — append docs to session
  3. POST /api/guest2/checkout         — Stripe checkout, lock session
  4. POST /api/webhooks/stripe-guest   — payment confirmed, fire analysis
  5. GET  /api/guest2/status           — token-free poll (returns token when ready)
  6. GET  /api/guest2/report           — token-gated full report fetch

Environment variables (all already on Render except STRIPE_GUEST_WEBHOOK_SECRET):
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY  (or SUPABASE_KEY)
  STRIPE_SECRET_KEY
  STRIPE_GUEST_WEBHOOK_SECRET
  RESEND_API_KEY
  RESEND_FROM
  ANTHROPIC_API_KEY
  REPORT_PRICE_GBP            (default: 29)
  FRONTEND_BASE_URL
"""

import io
import json
import os
import re
import threading
import time
import hmac
import hashlib
import secrets
import logging

import requests
from flask import Blueprint, request, jsonify

# ── Optional PDF extraction ──────────────────────────────────────────────────
try:
    import pdfplumber as _pdfplumber
except ImportError:
    _pdfplumber = None

try:
    import docai_ocr as _docai_ocr
except ImportError:
    _docai_ocr = None

try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None

try:
    from supabase import create_client as _supa_create
except ImportError:
    _supa_create = None

# ── Blueprint ────────────────────────────────────────────────────────────────
guest_bp = Blueprint("guest2", __name__)
logger   = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
STRIPE_SECRET       = (os.getenv("STRIPE_SECRET_KEY")            or "").strip()
STRIPE_WH_SECRET    = (os.getenv("STRIPE_GUEST_WEBHOOK_SECRET")  or "").strip()
RESEND_API_KEY      = (os.getenv("RESEND_API_KEY")               or "").strip()
RESEND_FROM         = (os.getenv("RESEND_FROM", "noreply@legalsmegal.com")).strip()
REPORT_PRICE_GBP    = int(os.getenv("REPORT_PRICE_GBP", "29"))
FRONTEND_BASE       = (os.getenv("FRONTEND_BASE_URL", "https://legalsmegal-frontend.onrender.com")).strip()
ANTHROPIC_API_KEY   = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
SUPA_URL            = (os.getenv("SUPABASE_URL") or "").strip()
SUPA_KEY            = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY") or "").strip()

SESSION_TTL_HOURS   = 2
REPORT_TOKEN_TTL    = 72 * 3600   # 72-hour viewer link
MAX_FILE_BYTES      = 20 * 1024 * 1024
MAX_FILES           = 10

# ── Supabase client (independent of main app) ────────────────────────────────
_supa_client = None
_supa_lock   = threading.Lock()

def _get_supa():
    global _supa_client
    if _supa_client is not None:
        return _supa_client
    with _supa_lock:
        if _supa_client is None:
            if not _supa_create:
                raise RuntimeError("supabase-py not installed")
            if not SUPA_URL or not SUPA_KEY:
                raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set")
            _supa_client = _supa_create(SUPA_URL, SUPA_KEY)
        return _supa_client

# ── Supabase session helpers ─────────────────────────────────────────────────

def _session_create(session_id: str, email: str) -> bool:
    try:
        _get_supa().table("guest_sessions").insert({
            "session_id": session_id,
            "email":      email,
        }).execute()
        return True
    except Exception as e:
        logger.error(f"[guest2] session_create failed: {e}")
        return False

def _session_get(session_id: str) -> dict | None:
    try:
        r = _get_supa().table("guest_sessions")\
            .select("*")\
            .eq("session_id", session_id)\
            .gt("expires_at", "now()")\
            .single()\
            .execute()
        return r.data if r.data else None
    except Exception as e:
        logger.warning(f"[guest2] session_get {session_id}: {e}")
        return None

def _session_update(session_id: str, fields: dict) -> bool:
    try:
        _get_supa().table("guest_sessions")\
            .update(fields)\
            .eq("session_id", session_id)\
            .execute()
        return True
    except Exception as e:
        logger.error(f"[guest2] session_update {session_id}: {e}")
        return False

def _session_claim_paid(session_id: str) -> bool:
    """Atomically mark a session as paid, but only if it is not already paid.

    This is a single conditional UPDATE ... WHERE paid = false sent to
    PostgREST — not a read-then-write. Two concurrent callers (e.g. two
    near-simultaneous webhook deliveries for the same session) will both
    issue this UPDATE; PostgreSQL serialises the two statements, so only
    one of them actually matches a row (the other's WHERE clause finds
    nothing, because the first writer already flipped paid to true).

    Returns True only for the caller that actually won the claim — i.e.
    only one caller, ever, gets True for a given session_id. All other
    callers (including genuine duplicate webhook deliveries) get False
    and must not proceed to start analysis or send email.
    """
    try:
        r = _get_supa().table("guest_sessions")\
            .update({"paid": True})\
            .eq("session_id", session_id)\
            .eq("paid", False)\
            .execute()
        won = bool(r.data)
        if not won:
            logger.info(f"[guest2] Claim lost for {session_id} — already paid by another writer")
        return won
    except Exception as e:
        logger.error(f"[guest2] session_claim_paid {session_id}: {e}")
        return False

def _session_append_doc(session_id: str, doc: dict) -> bool:
    """Append a document to the session's documents JSONB array."""
    try:
        # Fetch current docs then append — Supabase JSONB array append
        r = _get_supa().table("guest_sessions")\
            .select("documents")\
            .eq("session_id", session_id)\
            .single()\
            .execute()
        current = r.data.get("documents") or []
        current.append(doc)
        _get_supa().table("guest_sessions")\
            .update({"documents": current})\
            .eq("session_id", session_id)\
            .execute()
        return True
    except Exception as e:
        logger.error(f"[guest2] session_append_doc {session_id}: {e}")
        return False

def _session_purge_doc_text(session_id: str):
    """Remove extracted_text from all docs after analysis — keep metadata only."""
    try:
        r = _get_supa().table("guest_sessions")\
            .select("documents")\
            .eq("session_id", session_id)\
            .single()\
            .execute()
        docs = r.data.get("documents") or []
        stripped = [
            {k: v for k, v in d.items() if k != "extracted_text"}
            for d in docs
        ]
        _get_supa().table("guest_sessions")\
            .update({"documents": stripped, "doc_text_purged": True})\
            .eq("session_id", session_id)\
            .execute()
    except Exception as e:
        logger.warning(f"[guest2] purge_doc_text {session_id}: {e}")

# ── Anthropic client ─────────────────────────────────────────────────────────
_anthropic_client = None
_anthropic_lock   = threading.Lock()

def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is not None:
        return _anthropic_client
    with _anthropic_lock:
        if _anthropic_client is None:
            if not _anthropic:
                raise RuntimeError("anthropic package not installed")
            if not ANTHROPIC_API_KEY:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            _anthropic_client = _anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        return _anthropic_client

# ── PDF text extraction ──────────────────────────────────────────────────────
# S31 — Hard timeout on extraction. pdfplumber/pdfminer has no internal timeout
# and can hang indefinitely on certain malformed/pathological PDFs (complex
# embedded fonts, corrupted content streams). A single such file can block a
# Gunicorn worker for up to the full 180s worker timeout — with only 2 workers
# configured, that's enough to make every other concurrent request on this
# service fail (observed live on 2026-06-20: WORKER TIMEOUT after 188s,
# producing 502s on unrelated /api/guest2/upload and /api/deals/* requests
# that browsers then misreport as CORS failures, since there's no response
# to read CORS headers from). Run extraction in a daemon thread with a join
# timeout; if it doesn't finish in time, give up on that file and return
# empty text rather than hanging the worker. Callers already handle empty
# text gracefully (doc_type falls back to "unknown", upload still succeeds).
_EXTRACT_TEXT_TIMEOUT_SECONDS = 100
# S33 — raised from 25s for the same reason as app.py's
# _EXTRACT_PDF_TEXT_TIMEOUT_SECONDS: image-only PDFs now fall through to
# Document AI's batchProcess OCR flow (docai_ocr.py, internal ceiling 90s),
# a real network round-trip rather than a local hang. 25s would cut off an
# OCR call that was about to succeed.

def _extract_text_impl(file_bytes: bytes, filename: str, _result: dict) -> None:
    text, pages = "", 0
    if _pdfplumber:
        try:
            with _pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                pages = len(pdf.pages)
                parts = [pg.extract_text() or "" for pg in pdf.pages]
                text  = "\n\n".join(p for p in parts if p.strip())
        except Exception as e:
            logger.warning(f"[guest2] pdfplumber: {e}")
    if not text:
        try:
            import fitz as _fitz
            doc = _fitz.open(stream=file_bytes, filetype="pdf")
            pages = doc.page_count
            text  = "\n\n".join(pg.get_text() for pg in doc if pg.get_text().strip())
            doc.close()
        except Exception as e:
            logger.warning(f"[guest2] fitz: {e}")
    if not text and _docai_ocr is not None:
        # S33 — both standard extraction methods came back empty. Before
        # accepting that as "scanned, no text available", check whether
        # this is genuinely an image-only PDF (no text layer at all) and,
        # if so, route it through Document AI OCR rather than silently
        # giving up. Buyer-liability documents (e.g. Local Authority
        # Searches) must be read, not skipped — see app.py's
        # extract_pdf_text for the fuller rationale and the original
        # discovery of this gap.
        try:
            if _docai_ocr.is_image_only_pdf(file_bytes):
                logger.info(
                    f"[guest2] {filename!r} detected as image-only PDF — "
                    f"routing to Document AI OCR"
                )
                ocr_text = _docai_ocr.extract_text_via_docai(file_bytes)
                if ocr_text.strip():
                    text = ocr_text
        except Exception as e:
            logger.warning(f"[guest2] Document AI OCR failed for "
                            f"{filename!r}: {e}")
    _result["text"]  = text
    _result["pages"] = pages

def _extract_text(file_bytes: bytes, filename: str) -> tuple[str, int]:
    result = {"text": "", "pages": 0}
    t = threading.Thread(
        target=_extract_text_impl,
        args=(file_bytes, filename, result),
        daemon=True,
    )
    t.start()
    t.join(timeout=_EXTRACT_TEXT_TIMEOUT_SECONDS)
    if t.is_alive():
        # Extraction is still running in the background thread (it's a daemon
        # thread, so it will not block process/worker shutdown). We give up
        # waiting for it and return empty text so the request can complete.
        logger.warning(
            f"[guest2] _extract_text: TIMEOUT after {_EXTRACT_TEXT_TIMEOUT_SECONDS}s "
            f"on {filename!r} — returning empty text, doc_type will fall back to 'unknown'."
        )
        return "", 0
    return result["text"], result["pages"]

def _infer_doc_type(filename: str, text: str) -> str:
    fn = filename.lower(); tx = text.lower()
    if "special" in fn or "special conditions" in tx[:500]: return "special_conditions"
    if "title" in fn and "plan" in fn:   return "title_plan"
    if "title" in fn or "register" in fn: return "title_register"
    if "lease" in fn or "leasehold" in tx[:300]: return "lease"
    if "search" in fn and "local" in fn: return "local_auth_search"
    if "environmental" in fn:            return "environmental"
    if "epc" in fn or "energy performance" in tx[:300]: return "epc"
    if "tenancy" in fn or "ast" in fn:   return "tenancy_ast"
    if "auction" in fn and ("tc" in fn or "condition" in fn): return "auction_tcs"
    if "addendum" in fn or "amendment" in fn: return "addendum"
    return "unknown"

# ── Report token ─────────────────────────────────────────────────────────────
def _sign_report_token(session_id: str) -> str:
    import base64
    secret  = os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY") or "guest2-secret"
    expires = int(time.time()) + REPORT_TOKEN_TTL
    payload = f"{session_id}:{expires}"
    sig     = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    encoded = base64.urlsafe_b64encode(payload.encode()).decode()
    return f"{encoded}.{sig}"

def _verify_report_token(token: str) -> str | None:
    import base64
    try:
        secret = os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY") or "guest2-secret"
        encoded, sig = token.rsplit(".", 1)
        payload  = base64.urlsafe_b64decode(encoded.encode()).decode()
        expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, sig):
            return None
        session_id, expires_str = payload.split(":", 1)
        if time.time() > int(expires_str):
            return None
        return session_id
    except Exception:
        return None

# ── LLM analysis ─────────────────────────────────────────────────────────────
ANALYSIS_SYSTEM = """You are a UK auction property legal analyst. Your job is to FIND EVERY RISK in this legal pack. Be aggressive and thorough — an investor's money is at stake.

Return ONLY valid JSON. No prose, no markdown fences. Exactly this structure (flags MUST come first):
{
  "flags": [
    {
      "severity": "critical|high|missing|note",
      "title": "specific risk title — max 10 words",
      "summation": "one sentence: what this means for the investor",
      "evidence": "verbatim quote from document — max 30 words",
      "implication": "financial or legal impact — max 20 words",
      "action": "what investor must do — max 15 words",
      "source_document": "document filename",
      "source_clause": "clause number or null",
      "source_page": null,
      "legal_risk_weight": 7
    }
  ],
  "flag_counts": {"critical": 0, "high": 0, "missing": 0, "note": 0},
  "deal_score": 0,
  "viability_statement": "2-3 sentences: investor verdict",
  "property": {"address": "full address", "postcode": "postcode", "lot_number": "lot", "type": "BTL/HMO/Commercial/etc", "physical_type": "Flat/Detached/Semi-Detached/Terraced/Other", "tenure": "Freehold/Leasehold", "lease_years": null, "guide_price_pence": null},
  "completion_terms": {"deposit_pct": null, "deposit_refundable": null, "completion_days": null, "completion_type": "working", "buyers_premium_pct": null, "vacant_possession": null},
  "special_conditions": {
    "buyers_premium_pct": null, "buyers_premium_gbp": null, "admin_fee_gbp": null,
    "vat_elected": false, "seller_legal_costs_gbp": null, "search_fee_reimbursement": false,
    "completion_days": null, "deposit_pct": null, "non_refundable_deposit": false,
    "conditional_sale": false, "overage_clause": false, "addendum_present": false,
    "addendum_date": null, "addendum_notes": null, "unusual_clauses": [],
    "true_cost_additions_notes": null, "special_conditions_present": false,
    "special_conditions_missing": false
  },
  "pack_completeness": {"completeness_pct": 0, "present_count": 0, "total": 13},
  "documents_processed": 0
}

FLAG EXTRACTION RULES:
1. NEVER return an empty flags array. Every legal pack has risks.
2. Flag EVERY: restrictive covenants, chancel repair, mining/subsidence, flood risk, Japanese knotweed, Article 4, HMO licensing, short lease (<85 years), ground rent escalation, service charge >£2500/yr, absent landlord, possessory title, missing searches, auction clauses, tenancy issues, planning enforcement.
3. Flag MISSING documents: Special Conditions, Title Register, Local Search, Environmental, EPC — each is a MISSING flag.
4. Minimum 10-20 flags total.
5. Scoring: Start 100. Deduct critical=-12, high=-6, missing=-4, note=-1.
6. Evidence quotes MAX 30 words.
7. A blank flags array is a SYSTEM FAILURE. Minimum 3 flags required."""

def _run_llm_analysis(documents: list) -> dict:
    PRIORITY = ["special_conditions","addendum","title_register","lease","title_plan",
                "deed","freehold","tenancy_ast","local_auth_search","environmental",
                "epc","survey","auction_tcs","unknown"]
    docs_sorted = sorted(documents,
        key=lambda d: PRIORITY.index(d.get("doc_type","unknown"))
                      if d.get("doc_type","unknown") in PRIORITY else 99)

    parts, total = [], 0
    HARD_CAP, PER_DOC = 40000, 6000
    for doc in docs_sorted:
        txt = (doc.get("extracted_text") or "").strip()
        if not txt: continue
        label = f"=== {doc.get('doc_type','unknown').upper()}: {doc.get('file_name','')} ===\n"
        capped = txt[:PER_DOC] + ("\n[...truncated...]" if len(txt) > PER_DOC else "")
        chunk  = label + capped + "\n\n"
        if total + len(chunk) > HARD_CAP:
            rem = HARD_CAP - total - len(label) - 20
            if rem > 300:
                parts.append(label + txt[:rem] + "\n[...truncated...]\n\n")
            break
        parts.append(chunk); total += len(chunk)

    if not "".join(parts).strip():
        raise ValueError("no_text_extracted")

    client  = _get_anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=8192, temperature=0.1,
        system=ANALYSIS_SYSTEM,
        messages=[{"role": "user", "content": f"Analyse this auction legal pack:\n\n{''.join(parts)}"}],
    )
    content = message.content[0].text if message.content else ""
    logger.info(f"[guest2-llm] stop_reason={message.stop_reason}")

    result = None
    try:
        result = json.loads(content.strip())
    except Exception:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(),
                         flags=re.IGNORECASE | re.MULTILINE).strip()
        try:
            result = json.loads(cleaned)
        except Exception:
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if m: result = json.loads(m.group(0))
            else: raise ValueError(f"LLM non-JSON: {content[:200]}")

    if not isinstance(result.get("flags"), list):
        result["flags"] = []
    if not result["flags"]:
        result["flags"] = [{"severity":"note","title":"No specific flags raised",
            "summation":"Always commission a solicitor review before bidding.",
            "evidence":"System generated","implication":"No automated flags does not guarantee clean pack",
            "action":"Commission independent solicitor review",
            "source_document":"System","source_clause":None,"source_page":None,"legal_risk_weight":1}]
    result["flag_counts"] = {
        "critical": sum(1 for f in result["flags"] if (f.get("severity") or "").lower()=="critical"),
        "high":     sum(1 for f in result["flags"] if (f.get("severity") or "").lower()=="high"),
        "missing":  sum(1 for f in result["flags"] if (f.get("severity") or "").lower()=="missing"),
        "note":     sum(1 for f in result["flags"] if (f.get("severity") or "").lower()=="note"),
    }
    return result

# ── PDF generation (ReportLab) ───────────────────────────────────────────────
def _generate_pdf_local(summary_json: dict, docs: list) -> bytes:
    """Local ReportLab PDF generation — fallback only."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable)
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT
    import io as _io
    from datetime import date

    buf = _io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm, topMargin=16*mm, bottomMargin=20*mm)

    C_BLACK = colors.HexColor("#1a1a1a"); C_MUTED = colors.HexColor("#7a8fa3")
    C_CRIT  = colors.HexColor("#e05c5c"); C_HIGH  = colors.HexColor("#e8943a")
    C_MISS  = colors.HexColor("#4a9eff"); C_NOTE  = colors.HexColor("#7a8fa3")
    C_GREEN = colors.HexColor("#3ecf8e"); C_WHITE = colors.white
    MONO = "Courier"; SANS = "Helvetica"; W = A4[0] - 36*mm

    def sty(name, **kw):
        return ParagraphStyle(name, parent=getSampleStyleSheet()["Normal"], **kw)

    s_logo = sty("logo2", fontName=MONO, fontSize=16, textColor=C_BLACK, spaceAfter=2)
    s_tag  = sty("tag2",  fontName=MONO, fontSize=7,  textColor=C_MUTED, spaceAfter=14, leading=10)
    s_sec  = sty("sec2",  fontName=MONO, fontSize=7,  textColor=C_MUTED, spaceBefore=10, spaceAfter=5, leading=9)
    s_kv_k = sty("kvk2",  fontName=SANS, fontSize=9,  textColor=C_MUTED)
    s_kv_v = sty("kvv2",  fontName=SANS+"-Bold", fontSize=9, textColor=C_BLACK, alignment=TA_RIGHT)
    s_body = sty("body2", fontName=SANS, fontSize=8,  textColor=C_BLACK, leading=11)
    s_bold = sty("bold2", fontName=SANS+"-Bold", fontSize=8, textColor=C_BLACK, leading=11)
    s_small= sty("sml2",  fontName=SANS, fontSize=7,  textColor=C_MUTED, leading=10)
    s_mono = sty("mono2", fontName=MONO, fontSize=7,  textColor=C_MUTED, leading=10)
    s_li   = sty("li2",   fontName=SANS, fontSize=8,  textColor=C_BLACK, leading=12, leftIndent=8)
    s_ntc  = sty("ntc2",  fontName=SANS, fontSize=7,  textColor=colors.HexColor("#555555"), leading=10)

    sj = summary_json or {}
    flags = sj.get("flags") or []; prop = sj.get("property") or {}
    counts = sj.get("flag_counts") or {}; sc = sj.get("special_conditions") or {}
    score = sj.get("deal_score","?"); address = prop.get("address") or "?"
    postcode = prop.get("postcode") or ""; tenure = prop.get("tenure") or "?"
    lease = prop.get("lease_years"); guide_p = prop.get("guide_price_pence")
    guide = f"\u00a3{int(guide_p)//100:,}" if guide_p else "?"
    viability = sj.get("viability_statement") or ""
    today = date.today().strftime("%d %B %Y")

    story = []

    def section(t):
        story.append(Spacer(1, 3*mm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=C_MUTED))
        story.append(Paragraph(t.upper(), s_sec))

    def kv(key, val):
        t = Table([[Paragraph(key, s_kv_k), Paragraph(str(val or ""), s_kv_v)]],
                  colWidths=[W*0.55, W*0.45])
        t.setStyle(TableStyle([("BOTTOMPADDING",(0,0),(-1,-1),3),("TOPPADDING",(0,0),(-1,-1),3),
            ("LINEBELOW",(0,0),(-1,-1),0.3,colors.HexColor("#f0f0f0")),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
        story.append(t)

    def sev_col(sev):
        return {"critical":C_CRIT,"high":C_HIGH,"missing":C_MISS}.get((sev or "").lower(),C_NOTE)

    def sev_lbl(sev):
        return {"critical":"CRITICAL","high":"HIGH","missing":"MISSING","note":"NOTE"}.get((sev or "").lower(),"NOTE")

    def flag_table(flag_list):
        if not flag_list:
            story.append(Paragraph("No flags in this category.", s_small)); return
        rows = [[Paragraph("SEV",s_mono),Paragraph("FLAG",s_mono),Paragraph("SOURCE",s_mono)]]
        for f in flag_list:
            sev = (f.get("severity") or "note").lower()
            col = sev_col(sev); lbl = sev_lbl(sev)
            hex_str = col.hexval().lstrip("#x")
            sp = Paragraph("<font color=\"#"+hex_str+"\">"+lbl+"</font>",
                           sty("sv2"+sev,fontName=MONO,fontSize=6,textColor=col))
            fc = [Paragraph("<b>"+(f.get("title") or "")+"</b>",s_bold),
                  Paragraph(f.get("summation","") or "",s_body),
                  Paragraph(f.get("action","") or "",s_small)]
            rows.append([sp,fc,Paragraph((f.get("source_document") or "")[:40],s_small)])
        tbl = Table(rows,colWidths=[20*mm,W-20*mm-26*mm,26*mm],repeatRows=1)
        tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#f5f5f5")),
            ("LINEBELOW",(0,0),(-1,-1),0.3,colors.HexColor("#eeeeee")),
            ("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),4),
            ("BOTTOMPADDING",(0,0),(-1,-1),4),("LEFTPADDING",(0,0),(-1,-1),3),
            ("RIGHTPADDING",(0,0),(-1,-1),3)]))
        story.append(tbl)

    story.append(Paragraph("LegalSmegal",s_logo))
    story.append(Paragraph("AUCTION LEGAL PACK INTELLIGENCE REPORT",s_tag))
    section("1. Report Snapshot")
    kv("Property",address); kv("Postcode",postcode)
    kv("Tenure",tenure+(f" - {lease} years remaining" if lease else ""))
    kv("Guide price",guide); kv("Report date",today)
    score_c = C_GREEN if (isinstance(score,(int,float)) and score>=70) else C_HIGH if (isinstance(score,(int,float)) and score>=50) else C_CRIT
    sc_t = Table([[Paragraph(f"<b>{score}</b>",sty("scr2",fontName=MONO,fontSize=18,textColor=C_WHITE,alignment=TA_CENTER)),
        Paragraph(f"<b>Pack Score / 100</b><br/>{viability}",sty("scv2",fontName=SANS,fontSize=8,textColor=C_BLACK,leading=11))]],
        colWidths=[22*mm,W-22*mm])
    sc_t.setStyle(TableStyle([("BACKGROUND",(0,0),(0,0),score_c),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    story.append(Spacer(1,3*mm)); story.append(sc_t)
    section("2. Pack Status")
    st = Table([[Paragraph(f"<b>{counts.get('critical',0)}</b>",sty("s12",fontName=MONO,fontSize=16,textColor=C_CRIT,alignment=TA_CENTER)),
        Paragraph(f"<b>{counts.get('high',0)}</b>",sty("s22",fontName=MONO,fontSize=16,textColor=C_HIGH,alignment=TA_CENTER)),
        Paragraph(f"<b>{counts.get('missing',0)}</b>",sty("s32",fontName=MONO,fontSize=16,textColor=C_MISS,alignment=TA_CENTER)),
        Paragraph(f"<b>{counts.get('note',0)}</b>",sty("s42",fontName=MONO,fontSize=16,textColor=C_NOTE,alignment=TA_CENTER))],
        [Paragraph("CRITICAL",s_mono),Paragraph("HIGH",s_mono),Paragraph("MISSING",s_mono),Paragraph("NOTES",s_mono)]],
        colWidths=[W/4]*4)
    st.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    story.append(st)
    section("3. Special Conditions")
    sc_lines = []
    if sc.get("buyers_premium_pct"): sc_lines.append(f"Buyer premium: {sc['buyers_premium_pct']}%")
    if sc.get("vat_elected"):        sc_lines.append("VAT ELECTED - purchase price +20%")
    if sc.get("non_refundable_deposit"): sc_lines.append("Deposit is non-refundable")
    if sc.get("completion_days"):    sc_lines.append(f"Completion: {sc['completion_days']} days")
    if not sc_lines: sc_lines = ["No special conditions extracted"]
    for line in sc_lines: story.append(Paragraph(f"- {line}",s_li))
    section("4. Buyer Cost Exposure")
    flag_table([f for f in flags if any(k in ((f.get("title") or "")+(f.get("summation") or "")).lower() for k in ["premium","fee","cost","charge","vat","arrear","deposit","indemnit"])])
    section("5. Missing Evidence")
    flag_table([f for f in flags if (f.get("severity") or "").lower()=="missing"])
    section(f"6. Full Flag Register ({len(flags)} flags)")
    flag_table(flags)
    section("7. Document Inventory")
    if docs:
        dr = [[Paragraph("FILE",s_mono),Paragraph("PG",s_mono),Paragraph("TYPE",s_mono)]]
        for d in docs:
            dr.append([Paragraph((d.get("file_name") or "")[:50],s_body),Paragraph(str(d.get("page_count") or ""),s_body),Paragraph((d.get("doc_type") or "").replace("_"," ").title(),s_body)])
        dt = Table(dr,colWidths=[W*0.55,W*0.1,W*0.35],repeatRows=1)
        dt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#f5f5f5")),("LINEBELOW",(0,0),(-1,-1),0.3,colors.HexColor("#eeeeee")),
            ("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),("LEFTPADDING",(0,0),(-1,-1),3)]))
        story.append(dt)
    story.append(Spacer(1,5*mm))
    disc = Table([[Paragraph("This report is produced by LegalSmegal Technologies Ltd for investor decision-support purposes only. It does not constitute legal advice. Always instruct a qualified solicitor before bidding at auction.",s_ntc)]],colWidths=[W])
    disc.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),("LINEABOVE",(0,0),(-1,-1),1,C_BLACK),("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#f9f9f9"))]))
    story.append(disc)

    def _footer(canvas, doc):
        canvas.saveState(); canvas.setFont(MONO,7); canvas.setFillColor(C_MUTED)
        canvas.drawString(18*mm,12*mm,"LegalSmegal Technologies Ltd")
        canvas.drawCentredString(A4[0]/2,12*mm,"Not legal advice - investor decision support only")
        canvas.drawRightString(A4[0]-18*mm,12*mm,today)
        canvas.setStrokeColor(colors.HexColor("#e5e5e5"))
        canvas.line(18*mm,15*mm,A4[0]-18*mm,15*mm)
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


def _generate_pdf_bytes(summary_json: dict, docs: list) -> bytes:
    """Call Hetzner PDF microservice. Falls back to local generation if unavailable."""
    pdf_url    = (os.getenv("PDF_SERVICE_URL") or "").strip()
    pdf_secret = (os.getenv("PDF_SECRET") or "").strip()

    if pdf_url:
        try:
            resp = requests.post(
                pdf_url,
                headers={"X-PDF-Secret": pdf_secret, "Content-Type": "application/json"},
                json={"summary_json": summary_json, "docs": docs},
                timeout=90,
            )
            if resp.status_code == 200:
                logger.info(f"[guest2] PDF from Hetzner: {len(resp.content):,} bytes")
                return resp.content
            logger.warning(f"[guest2] Hetzner PDF {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"[guest2] Hetzner PDF call failed: {e} — falling back to local")

    # Local fallback (used if Hetzner unreachable)
    logger.info("[guest2] Generating PDF locally (fallback)")
    return _generate_pdf_local(summary_json, docs)



def _send_report_email(to_email: str, address: str, report_url: str, pdf_bytes: bytes) -> bool:
    if not RESEND_API_KEY:
        logger.warning("[guest2] RESEND_API_KEY not set"); return False
    import base64
    try:
        resp = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
            json={
                "from": RESEND_FROM, "to": [to_email],
                "subject": f"Your LegalSmegal Report — {address or 'Legal Pack'}",
                "html": f"""<div style="font-family:'IBM Plex Sans',sans-serif;max-width:560px;margin:0 auto;padding:32px 24px;background:#0d1219;color:#e8edf2">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:18px;font-weight:600;margin-bottom:28px">Legal<span style="color:#c8a84b">Smegal</span></div>
  <div style="font-size:14px;font-weight:600;margin-bottom:8px">Your report is ready</div>
  <div style="font-size:13px;color:#7a8fa3;margin-bottom:24px">{address or 'Legal Pack'}</div>
  <a href="{report_url}" style="display:inline-block;padding:12px 24px;background:#c8a84b;color:#080c10;font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;text-decoration:none;border-radius:4px">View Online &rarr;</a>
  <div style="margin-top:24px;font-family:'IBM Plex Mono',monospace;font-size:9px;color:#3d5068">PDF attached. Link valid 72 hours. Not legal advice.</div>
</div>""",
                "attachments": [{"filename": "LegalSmegal-Report.pdf",
                                  "content": base64.b64encode(pdf_bytes).decode()}],
            }, timeout=30,
        )
        if resp.status_code in (200, 201):
            logger.info(f"[guest2] Email sent to {to_email}"); return True
        logger.warning(f"[guest2] Resend {resp.status_code}: {resp.text[:200]}"); return False
    except Exception as e:
        logger.warning(f"[guest2] Email error: {e}"); return False

# ── Background analysis + delivery ──────────────────────────────────────────
def _run_analysis_and_deliver(session_id: str):
    session = _session_get(session_id)
    if not session:
        logger.error(f"[guest2] Session {session_id} not found"); return

    docs  = session.get("documents") or []
    email = session.get("email") or ""
    logger.info(f"[guest2] Analysis starting: {session_id} ({len(docs)} docs)")

    try:
        summary_json = _run_llm_analysis(docs)
    except Exception as e:
        logger.error(f"[guest2] LLM failed: {e}")
        _session_update(session_id, {"status": "failed"})
        # Send failure email so user isn't left in limbo
        try:
            requests.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
                json={
                    "from": RESEND_FROM,
                    "to": [email],
                    "subject": "LegalSmegal — Report generation failed",
                    "html": (
                        "<p>We're sorry — your legal pack report could not be generated due to a technical error.</p>"
                        "<p>You have not been charged. If payment was taken, it will be automatically refunded within 5 working days.</p>"
                        "<p>Please try again or contact support at support@legalsmegal.com.</p>"
                    ),
                },
                timeout=15,
            )
        except Exception as mail_err:
            logger.warning(f"[guest2] Failure email error: {mail_err}")
        return

    token      = _sign_report_token(session_id)
    report_url = f"{FRONTEND_BASE}/legalsmegal-report.html?guest_session={session_id}&token={token}"

    # Store summary_json + token, purge doc text
    _session_update(session_id, {"summary_json": summary_json, "report_token": token})
    _session_purge_doc_text(session_id)

    address = (summary_json.get("property") or {}).get("address") or ""

    try:
        pdf_bytes = _generate_pdf_bytes(summary_json, [
            {k: v for k, v in d.items() if k != "extracted_text"} for d in docs
        ])
        logger.info(f"[guest2] PDF generated ({len(pdf_bytes):,} bytes)")
        _send_report_email(email, address, report_url, pdf_bytes)
    except Exception as e:
        logger.error(f"[guest2] PDF/email failed: {e}")
        # Fallback: link-only email
        try:
            requests.post("https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
                json={"from": RESEND_FROM, "to": [email],
                      "subject": f"Your LegalSmegal Report — {address}",
                      "html": f'<a href="{report_url}">View your report</a>'}, timeout=15)
        except Exception: pass

    logger.info(f"[guest2] Delivery complete: {session_id}")

# ════════════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════════════

@guest_bp.route("/api/guest2/create-session", methods=["POST", "OPTIONS"])
def guest2_create_session():
    if request.method == "OPTIONS": return "", 204
    data  = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    if not email or "@" not in email:
        return jsonify({"error": "Valid email required"}), 400
    session_id = secrets.token_hex(16)
    if not _session_create(session_id, email):
        return jsonify({"error": "Could not create session"}), 500
    logger.info(f"[guest2] Session created: {session_id} for {email}")
    return jsonify({"ok": True, "session_id": session_id}), 201


@guest_bp.route("/api/guest2/upload", methods=["POST", "OPTIONS"])
def guest2_upload():
    if request.method == "OPTIONS": return "", 204
    session_id = (request.form.get("session_id") or "").strip()
    session    = _session_get(session_id)
    if not session:      return jsonify({"error": "Session not found or expired"}), 404
    if session.get("locked"): return jsonify({"error": "Session locked"}), 409

    files = request.files.getlist("file")
    if not files: return jsonify({"error": "No files provided"}), 400

    current_count = len(session.get("documents") or [])
    if current_count + len(files) > MAX_FILES:
        return jsonify({"error": f"Maximum {MAX_FILES} files per session"}), 400

    uploaded = []
    for f in files:
        raw = f.read()
        if len(raw) > MAX_FILE_BYTES:
            uploaded.append({"file_name": f.filename, "ok": False, "error": "File too large"}); continue
        text, pages = _extract_text(raw, f.filename)
        doc_type    = _infer_doc_type(f.filename, text)
        doc = {"file_name": f.filename, "extracted_text": text,
               "page_count": pages, "doc_type": doc_type}
        if _session_append_doc(session_id, doc):
            uploaded.append({"file_name": f.filename, "ok": True,
                             "doc_type": doc_type, "page_count": pages})
            logger.info(f"[guest2] Upload: {f.filename} ({doc_type}) → {session_id}")
        else:
            uploaded.append({"file_name": f.filename, "ok": False, "error": "Storage error"})

    return jsonify({"ok": True, "uploaded": uploaded}), 200


@guest_bp.route("/api/guest2/checkout", methods=["POST", "OPTIONS"])
def guest2_checkout():
    if request.method == "OPTIONS": return "", 204
    if not STRIPE_SECRET:
        return jsonify({"error": "Payment not configured (STRIPE_SECRET_KEY missing)"}), 503
    data       = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or "").strip()
    session    = _session_get(session_id)
    if not session:                    return jsonify({"error": "Session not found or expired"}), 404
    if not session.get("documents"):   return jsonify({"error": "No documents uploaded"}), 400

    # S30 — Duplicate-purchase guard. If this session has already been paid,
    # refuse to create a second Checkout Session — the customer would be
    # charged twice for the same upload and the second payment would spawn
    # a second, independent analysis run (duplicate summary_json, duplicate
    # email, divergent flag counts vs the first). This is the correct place
    # to stop it: before Stripe is asked to take a second payment at all.
    if session.get("paid"):
        return jsonify({
            "error": "already_paid",
            "message": "This upload has already been paid for. Check your email for the report, or refresh the report page.",
        }), 409

    # NOTE: we deliberately do NOT add a second guard here based on the local
    # "locked" flag. A flag like that goes stale the moment a customer cancels
    # checkout or their connection drops mid-payment — it would then trap them,
    # unable to ever pay for this upload again. The idempotency key below is
    # the correct fix: it makes a second checkout-creation call for the same
    # session_id safe by construction, because Stripe — not a local flag —
    # is the source of truth on whether a session already exists.

    # S28 — Scanned PDF detection: block checkout if no document has extractable text.
    # A legal pack where every PDF is a scanned image will produce an empty LLM analysis.
    # Better to catch this before payment than after.
    docs = session.get("documents") or []
    all_empty = all(not (d.get("extracted_text") or "").strip() for d in docs)
    if all_empty:
        return jsonify({
            "error": "scanned_pdf",
            "message": (
                "Your documents appear to be scanned images with no text layer. "
                "LegalSmegal cannot analyse image-only PDFs. Please upload text-based PDFs "
                "or OCR your documents before uploading."
            ),
        }), 422

    email = session["email"]
    try:
        resp = requests.post(
            "https://api.stripe.com/v1/checkout/sessions",
            auth=(STRIPE_SECRET, ""),
            # S30 — Idempotency key, deterministic per session_id. If this
            # endpoint is called more than once for the same session_id
            # (double-click, browser back/forward, dropped connection and
            # client-side retry, the Cancel-then-resubmit path on the upload
            # page), Stripe recognises the repeated key and returns the
            # *original* Checkout Session instead of creating a second one.
            # This is Stripe's own documented mechanism for exactly this
            # problem — see https://stripe.com/docs/api/idempotent_requests —
            # and is strictly stronger than any local "locked" flag, because
            # Stripe enforces it server-side regardless of what our own
            # database thinks happened.
            headers={"Idempotency-Key": f"guest2-checkout-{session_id}"},
            data={
                "mode": "payment",
                "line_items[0][price_data][currency]":                "gbp",
                "line_items[0][price_data][unit_amount]":             str(REPORT_PRICE_GBP * 100),
                "line_items[0][price_data][product_data][name]":      "LegalSmegal Legal Pack Report",
                "line_items[0][price_data][product_data][description]": "One-off auction legal pack intelligence report",
                "line_items[0][quantity]": "1",
                "customer_email": email,
                "metadata[guest2_session_id]": session_id,
                "success_url": f"{FRONTEND_BASE}/legalsmegal-report.html?guest_session={session_id}&processing=1",
                "cancel_url":  f"{FRONTEND_BASE}/legalsmegal-upload-report.html?cancelled=1",
            }, timeout=15,
        )
        if resp.status_code != 200:
            return jsonify({"error": "Payment setup failed"}), 502
        stripe_session = resp.json()
        checkout_url   = stripe_session.get("url")
        if not checkout_url: return jsonify({"error": "No checkout URL"}), 502
        _session_update(session_id, {"locked": True, "stripe_session_id": stripe_session.get("id","")})
        return jsonify({"ok": True, "checkout_url": checkout_url}), 200
    except Exception as e:
        logger.exception("guest2_checkout failed")
        return jsonify({"error": "Payment setup failed"}), 500


@guest_bp.route("/api/webhooks/stripe-guest", methods=["POST"])
def guest2_stripe_webhook():
    payload = request.get_data()
    sig     = request.headers.get("Stripe-Signature", "")
    if not STRIPE_WH_SECRET:
        logger.error("[guest2-wh] STRIPE_GUEST_WEBHOOK_SECRET not set — rejecting all webhook events")
        return jsonify({"error": "Webhook not configured"}), 503
    try:
        parts    = {k: v for k, v in (p.split("=", 1) for p in sig.split(",") if "=" in p)}
        ts       = parts.get("t", "0"); v1 = parts.get("v1", "")
        signed   = f"{ts}.{payload.decode('utf-8')}"
        expected = hmac.new(STRIPE_WH_SECRET.encode(), signed.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, v1):
            return jsonify({"error": "Invalid signature"}), 400
        if abs(int(time.time()) - int(ts)) > 300:
            return jsonify({"error": "Timestamp too old"}), 400
    except Exception as e:
        logger.warning(f"[guest2-wh] Sig check failed: {e}")
        return jsonify({"error": "Signature error"}), 400

    event      = request.get_json(force=True, silent=True) or {}
    event_type = event.get("type", "")
    logger.info(f"[guest2-wh] Event: {event_type}")

    if event_type == "checkout.session.completed":
        obj        = (event.get("data") or {}).get("object") or {}
        session_id = (obj.get("metadata") or {}).get("guest2_session_id", "")
        event_id   = event.get("id", "")
        if not session_id:
            return jsonify({"ok": True}), 200

        if obj.get("payment_status") != "paid":
            return jsonify({"ok": True}), 200

        # S30 — Atomic claim. The previous version of this check did
        # _session_get() (read) followed by _session_update() (write) as two
        # separate calls. That leaves a race window: two near-simultaneous
        # deliveries of either the same event (Stripe redelivery) or two
        # genuinely different completed checkouts for the same session_id
        # (e.g. the Cancel-then-resubmit path on the upload page) could both
        # read paid=false before either has written paid=true, and both
        # would then proceed to start an analysis thread and send an email.
        #
        # _session_claim_paid() sends a single conditional UPDATE ... WHERE
        # paid = false to PostgREST. PostgreSQL serialises the two UPDATE
        # statements at the database level — only one can possibly match
        # the WHERE clause, because the other writer already flipped the
        # row. This makes "only one thread ever starts analysis for a given
        # session_id" a guarantee enforced by the database, not by
        # application-level timing.
        won_claim = _session_claim_paid(session_id)
        if not won_claim:
            logger.info(
                f"[guest2-wh] event={event_id} session={session_id} — "
                f"claim already held, not starting a second analysis run"
            )
            return jsonify({"ok": True}), 200

        t = threading.Thread(target=_run_analysis_and_deliver,
                             args=(session_id,), daemon=True,
                             name=f"guest2-{session_id[:8]}")
        t.start()
        logger.info(f"[guest2-wh] event={event_id} session={session_id} — analysis thread started")

    return jsonify({"ok": True}), 200


@guest_bp.route("/api/guest2/status", methods=["GET", "OPTIONS"])
def guest2_status():
    """Token-free poll. Returns token once complete."""
    if request.method == "OPTIONS": return "", 204
    session_id = (request.args.get("guest_session") or "").strip()
    if not session_id: return jsonify({"error": "guest_session required"}), 400
    session = _session_get(session_id)
    if not session:              return jsonify({"status": "expired"}), 404
    if not session.get("paid"): return jsonify({"status": "unpaid"}), 402
    if session.get("status") == "failed":  return jsonify({"status": "failed"}), 200
    if not session.get("summary_json"): return jsonify({"status": "processing"}), 202
    token = session.get("report_token") or _sign_report_token(session_id)
    return jsonify({"status": "complete", "token": token}), 200


@guest_bp.route("/api/guest2/report", methods=["GET", "OPTIONS"])
def guest2_get_report():
    """Token-gated full report fetch."""
    if request.method == "OPTIONS": return "", 204
    session_id = (request.args.get("guest_session") or "").strip()
    token      = (request.args.get("token") or "").strip()
    if not session_id or not token: return jsonify({"error": "session and token required"}), 401
    verified_id = _verify_report_token(token)
    if not verified_id or verified_id != session_id:
        return jsonify({"error": "Invalid or expired token"}), 401
    session = _session_get(session_id)
    if not session:              return jsonify({"error": "Session expired"}), 404
    if not session.get("paid"): return jsonify({"error": "Payment not confirmed"}), 402
    summary_json = session.get("summary_json")
    if not summary_json:         return jsonify({"ok": True, "status": "processing"}), 202
    docs = [
        {k: v for k, v in d.items() if k != "extracted_text"}
        for d in (session.get("documents") or [])
    ]
    return jsonify({"ok": True, "status": "complete",
                    "summary_json": summary_json, "documents": docs}), 200
