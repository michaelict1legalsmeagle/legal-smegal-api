"""
guest_routes.py — LegalSmegal one-off report pipeline (Flow 2)
================================================================
Completely independent of the subscriber pipeline. No shared state,
no Supabase writes, no auth system. Stripe is the gate.

Flow:
  1. POST /api/guest2/create-session   — create in-memory session, return session_id
  2. POST /api/guest2/upload           — attach PDFs to session (extract text in-memory)
  3. POST /api/guest2/checkout         — create Stripe session, lock upload
  4. POST /api/webhooks/stripe-guest   — Stripe calls this on payment; triggers analysis + email
  5. GET  /api/guest2/report           — token-gated viewer endpoint

All document text is held in a module-level dict keyed by session_id.
Sessions expire after SESSION_TTL_SECONDS. Memory is cleaned by a
background thread on a fixed interval.

Environment variables required (set on Render):
  STRIPE_SECRET_KEY
  STRIPE_GUEST_WEBHOOK_SECRET
  RESEND_API_KEY
  RESEND_FROM              (e.g. noreply@legalsmegal.com)
  ANTHROPIC_API_KEY        (already set for subscriber pipeline)
  REPORT_PRICE_GBP         (default: 29)
  FRONTEND_BASE_URL        (already set)
"""

import io
import json
import os
import re
import threading
import time
import hmac
import hashlib
import logging

import requests
from flask import Blueprint, request, jsonify, current_app

# ── Optional PDF extraction (same as main app) ──────────────────────────────
try:
    import pdfplumber as _pdfplumber
except ImportError:
    _pdfplumber = None

try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None

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
ANTHROPIC_API_KEY   = (os.getenv("ANTHROPIC_API_KEY")            or "").strip()

SESSION_TTL_SECONDS = 7200   # 2 hours — covers upload → payment → analysis window
MAX_FILE_BYTES      = 20 * 1024 * 1024   # 20 MB per file
MAX_FILES_PER_SESSION = 10
REPORT_TOKEN_TTL    = 72 * 3600          # 72-hour viewer link

# ── In-memory session store ──────────────────────────────────────────────────
# Structure:
# _sessions[session_id] = {
#   "email":      str,
#   "created_at": float,
#   "locked":     bool,           # True after checkout created — no more uploads
#   "paid":       bool,
#   "documents":  [               # list of dicts
#     { "file_name": str, "extracted_text": str, "page_count": int, "doc_type": str }
#   ],
#   "summary_json": dict | None,  # populated after analysis
#   "report_token": str | None,   # signed token for viewer
#   "stripe_session_id": str,
# }
_sessions: dict = {}
_sessions_lock  = threading.Lock()

# ── Anthropic client (independent — does not share main app's singleton) ────
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


# ── Session helpers ──────────────────────────────────────────────────────────
def _new_session_id() -> str:
    import secrets
    return secrets.token_hex(16)

def _get_session(session_id: str) -> dict | None:
    with _sessions_lock:
        s = _sessions.get(session_id)
        if s is None:
            return None
        if time.time() - s["created_at"] > SESSION_TTL_SECONDS:
            del _sessions[session_id]
            return None
        return s

def _purge_expired():
    """Remove expired sessions. Called by background thread."""
    now = time.time()
    with _sessions_lock:
        expired = [k for k, v in _sessions.items()
                   if now - v["created_at"] > SESSION_TTL_SECONDS]
        for k in expired:
            del _sessions[k]
    if expired:
        logger.info(f"[guest2] Purged {len(expired)} expired sessions")

def _start_cleanup_thread():
    def _loop():
        while True:
            time.sleep(600)   # every 10 minutes
            try:
                _purge_expired()
            except Exception as e:
                logger.warning(f"[guest2] Cleanup error: {e}")
    t = threading.Thread(target=_loop, daemon=True, name="guest2-cleanup")
    t.start()

_start_cleanup_thread()


# ── PDF text extraction ──────────────────────────────────────────────────────
def _extract_text(file_bytes: bytes, filename: str) -> tuple[str, int]:
    """Returns (extracted_text, page_count). Never raises."""
    text = ""
    pages = 0
    if _pdfplumber:
        try:
            with _pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                pages = len(pdf.pages)
                parts = []
                for pg in pdf.pages:
                    t = pg.extract_text() or ""
                    if t.strip():
                        parts.append(t)
                text = "\n\n".join(parts)
        except Exception as e:
            logger.warning(f"[guest2] pdfplumber failed for {filename}: {e}")
    if not text:
        try:
            import fitz as _fitz
            doc = _fitz.open(stream=file_bytes, filetype="pdf")
            pages = doc.page_count
            parts = []
            for pg in doc:
                t = pg.get_text()
                if t.strip():
                    parts.append(t)
            text = "\n\n".join(parts)
            doc.close()
        except Exception as e:
            logger.warning(f"[guest2] fitz failed for {filename}: {e}")
    return text, pages


def _infer_doc_type(filename: str, text: str) -> str:
    """Lightweight doc type inference — mirrors main app logic."""
    fn = filename.lower()
    tx = text.lower()
    if "special" in fn or "special conditions" in tx[:500]:
        return "special_conditions"
    if "title" in fn and "plan" in fn:
        return "title_plan"
    if "title" in fn or "register" in fn:
        return "title_register"
    if "lease" in fn or "leasehold" in tx[:300]:
        return "lease"
    if "search" in fn and "local" in fn:
        return "local_auth_search"
    if "environmental" in fn:
        return "environmental"
    if "epc" in fn or "energy performance" in tx[:300]:
        return "epc"
    if "tenancy" in fn or "ast" in fn:
        return "tenancy_ast"
    if "auction" in fn and ("tc" in fn or "condition" in fn):
        return "auction_tcs"
    if "addendum" in fn or "amendment" in fn:
        return "addendum"
    return "unknown"


# ── LLM analysis (identical prompt to subscriber pipeline) ───────────────────
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
    "buyers_premium_pct": null,
    "buyers_premium_gbp": null,
    "admin_fee_gbp": null,
    "vat_elected": false,
    "seller_legal_costs_gbp": null,
    "search_fee_reimbursement": false,
    "completion_days": null,
    "deposit_pct": null,
    "non_refundable_deposit": false,
    "conditional_sale": false,
    "overage_clause": false,
    "addendum_present": false,
    "addendum_date": null,
    "addendum_notes": null,
    "unusual_clauses": [],
    "true_cost_additions_notes": null,
    "special_conditions_present": false,
    "special_conditions_missing": false
  },
  "pack_completeness": {"completeness_pct": 0, "present_count": 0, "total": 13},
  "documents_processed": 0
}

FLAG EXTRACTION RULES — YOU MUST FOLLOW ALL OF THEM:
1. NEVER return an empty flags array. Every legal pack has risks. If a pack seems clean, flag what is MISSING.
2. Flag EVERY one of these if present: restrictive covenants, chancel repair, mining/subsidence, flood risk, Japanese knotweed, Article 4 directions, HMO licensing, short lease (<85 years), ground rent escalation, service charge >£2500/yr, absent landlord, possessory title, missing searches, auction clauses (non-refundable deposit, 28-day completion, buyers premium), tenancy issues (sitting tenant, AST expiry, rent arrears), planning enforcement notices.
3. Flag MISSING documents: if Special Conditions, Title Register, Local Search, Environmental Search, EPC are absent — each is a MISSING flag.
4. Minimum flags: generate at least 1 flag per document that contains a clause. Aim for 10-20 flags total.
5. Scoring: Start at 100. Deduct critical=-12, high=-6, missing=-4, note=-1.
6. Keep evidence quotes SHORT (max 30 words) — critical for fitting all flags within token budget.
7. The flags array MUST be complete before flag_counts. Do not close the JSON until all flags are written.

FEW-SHOT EXAMPLE — this is exactly what one flag object must look like:
{"severity": "critical", "title": "Missing Local Authority Search", "summation": "No local search in pack — planning restrictions and enforcement notices unknown.", "evidence": "Document not present in legal pack", "implication": "Unknown planning restrictions could prevent intended use", "action": "Order local search before bidding — allow 5-10 working days", "source_document": "Not present", "source_clause": null, "source_page": null, "legal_risk_weight": 9}

A blank flags array is a SYSTEM FAILURE. Minimum 3 flags required even for a clean pack.

SPECIAL CONDITIONS EXTRACTION — populate the special_conditions object precisely.
PROPERTY TYPE EXTRACTION — physical_type must be exactly: Flat, Detached, Semi-Detached, Terraced, or Other."""


def _run_llm_analysis(documents: list) -> dict:
    """Run LLM analysis on extracted document texts. Returns summary_json dict."""
    PRIORITY = ['special_conditions','addendum','title_register','lease',
                'title_plan','deed','freehold','tenancy_ast',
                'local_auth_search','environmental','epc','survey','auction_tcs','unknown']
    docs_sorted = sorted(documents,
        key=lambda d: PRIORITY.index(d.get("doc_type","unknown"))
                      if d.get("doc_type","unknown") in PRIORITY else 99)

    parts  = []
    total  = 0
    HARD_CAP = 40000
    PER_DOC  = 6000

    for doc in docs_sorted:
        txt = (doc.get("extracted_text") or "").strip()
        if not txt:
            continue
        label   = f"=== {doc.get('doc_type','unknown').upper()}: {doc.get('file_name','')} ===\n"
        capped  = txt[:PER_DOC] + ("\n[...truncated...]" if len(txt) > PER_DOC else "")
        chunk   = label + capped + "\n\n"
        if total + len(chunk) > HARD_CAP:
            rem = HARD_CAP - total - len(label) - 20
            if rem > 300:
                parts.append(label + txt[:rem] + "\n[...truncated...]\n\n")
            break
        parts.append(chunk)
        total += len(chunk)

    truncated = "".join(parts)
    if not truncated.strip():
        raise ValueError("no_text_extracted")

    client = _get_anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        temperature=0.1,
        system=ANALYSIS_SYSTEM,
        messages=[{"role": "user", "content": f"Analyse this auction legal pack:\n\n{truncated}"}],
    )
    content = message.content[0].text if message.content else ""
    logger.info(f"[guest2-llm] stop_reason={message.stop_reason} out_tokens={getattr(getattr(message,'usage',None),'output_tokens',0)}")

    # Parse JSON
    result = None
    try:
        result = json.loads(content.strip())
    except Exception:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(),
                         flags=re.IGNORECASE | re.MULTILINE).strip()
        try:
            result = json.loads(cleaned)
        except Exception:
            # Last resort: extract JSON object
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if m:
                result = json.loads(m.group(0))
            else:
                raise ValueError(f"LLM returned non-JSON: {content[:200]}")

    # Enforce schema
    if not isinstance(result.get("flags"), list):
        result["flags"] = []

    if len(result["flags"]) == 0:
        result["flags"] = [{
            "severity": "note",
            "title": "Analysis complete — no specific flags raised",
            "summation": "No specific risk flags identified. Always commission a solicitor review before bidding.",
            "evidence": "System generated",
            "implication": "No automated flags does not guarantee a clean legal pack",
            "action": "Commission independent solicitor review",
            "source_document": "System",
            "source_clause": None,
            "source_page": None,
            "legal_risk_weight": 1,
        }]

    result["flag_counts"] = {
        "critical": sum(1 for f in result["flags"] if (f.get("severity") or "").lower() == "critical"),
        "high":     sum(1 for f in result["flags"] if (f.get("severity") or "").lower() == "high"),
        "missing":  sum(1 for f in result["flags"] if (f.get("severity") or "").lower() == "missing"),
        "note":     sum(1 for f in result["flags"] if (f.get("severity") or "").lower() == "note"),
    }

    return result


# ── Report token (signed, TTL-bearing) ──────────────────────────────────────
def _sign_report_token(session_id: str) -> str:
    """Create a signed token: base64(session_id:expires_at):hmac"""
    import base64, secrets
    secret = os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY") or "guest2-fallback-secret"
    expires = int(time.time()) + REPORT_TOKEN_TTL
    payload = f"{session_id}:{expires}"
    sig     = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    encoded = base64.urlsafe_b64encode(payload.encode()).decode()
    return f"{encoded}.{sig}"

def _verify_report_token(token: str) -> str | None:
    """Returns session_id if valid, None otherwise."""
    import base64
    try:
        secret = os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY") or "guest2-fallback-secret"
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


# ── PDF generation (WeasyPrint) ──────────────────────────────────────────────
def _build_report_html(summary_json: dict, docs: list) -> str:
    """Render summary_json into the same HTML as legalsmegal-report.html — for PDF generation."""
    sj      = summary_json or {}
    flags   = sj.get("flags") or []
    prop    = sj.get("property") or {}
    counts  = sj.get("flag_counts") or {}
    sc      = sj.get("special_conditions") or {}
    score   = sj.get("deal_score", "—")

    address     = prop.get("address") or "—"
    postcode    = prop.get("postcode") or ""
    tenure      = prop.get("tenure") or "—"
    lease_yrs   = prop.get("lease_years")
    guide_p     = prop.get("guide_price_pence")
    guide_fmt   = f"£{int(guide_p)//100:,}" if guide_p else "—"
    viability   = sj.get("viability_statement") or ""

    from datetime import date
    today_str = date.today().strftime("%-d %B %Y")

    def sev_badge(sev: str) -> str:
        s = (sev or "note").lower()
        colours = {
            "critical": "#e05c5c",
            "high":     "#e8943a",
            "missing":  "#4a9eff",
            "note":     "#7a8fa3",
        }
        labels = {
            "critical": "Critical disclosure",
            "high":     "Requires clarification",
            "missing":  "Missing evidence",
            "note":     "Note",
        }
        c = colours.get(s, "#7a8fa3")
        l = labels.get(s, s.upper())
        return (f'<span style="display:inline-block;padding:2px 8px;border-radius:3px;'
                f'background:{c}20;border:1px solid {c};color:{c};'
                f'font-size:9px;font-weight:600;font-family:monospace;'
                f'text-transform:uppercase;letter-spacing:.06em">{l}</span>')

    def esc(s: str) -> str:
        return (str(s or "")
                .replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))

    def flag_row(f: dict) -> str:
        badge = sev_badge(f.get("severity", "note"))
        return f"""
        <tr style="border-bottom:1px solid #f0f0f0">
          <td style="padding:10px 8px;vertical-align:top;width:160px">{badge}</td>
          <td style="padding:10px 8px;vertical-align:top">
            <div style="font-weight:600;font-size:11px;color:#1a1a1a;margin-bottom:3px">{esc(f.get('title',''))}</div>
            <div style="font-size:10px;color:#555;margin-bottom:4px">{esc(f.get('summation',''))}</div>
            <div style="font-size:9px;color:#888;font-style:italic;margin-bottom:3px">"{esc(f.get('evidence',''))}"</div>
            <div style="font-size:9px;color:#333"><strong>Action:</strong> {esc(f.get('action',''))}</div>
          </td>
          <td style="padding:10px 8px;vertical-align:top;width:120px;font-size:9px;color:#888">{esc(f.get('source_document',''))}</td>
        </tr>"""

    all_flag_rows = "".join(flag_row(f) for f in flags)

    # Buyer cost exposure
    cost_flags = []
    for f in flags:
        t = (f.get("title") or "").lower()
        s = (f.get("summation") or "").lower()
        if any(k in t+s for k in ["premium","fee","cost","charge","vat","arrear","deposit","indemnit","search"]):
            cost_flags.append(f)

    cost_rows = "".join(flag_row(f) for f in cost_flags) if cost_flags else \
        '<tr><td colspan="3" style="padding:10px;color:#888;font-size:10px">No specific cost exposure flags identified.</td></tr>'

    # Missing evidence
    missing_flags = [f for f in flags if (f.get("severity") or "").lower() == "missing"]
    missing_rows  = "".join(flag_row(f) for f in missing_flags) if missing_flags else \
        '<tr><td colspan="3" style="padding:10px;color:#888;font-size:10px">No missing evidence flags.</td></tr>'

    # Document inventory
    doc_rows = ""
    for d in (docs or []):
        fn  = esc(d.get("file_name") or d.get("doc_type") or "—")
        pg  = d.get("page_count") or "—"
        dt  = esc((d.get("doc_type") or "document").replace("_", " ").title())
        doc_rows += f'<tr style="border-bottom:1px solid #f0f0f0"><td style="padding:6px 8px;font-size:10px">{fn}</td><td style="padding:6px 8px;font-size:10px;text-align:center">{pg}</td><td style="padding:6px 8px;font-size:10px">{dt}</td></tr>'
    if not doc_rows:
        doc_rows = '<tr><td colspan="3" style="padding:10px;color:#888;font-size:10px">No documents recorded.</td></tr>'

    score_colour = "#3ecf8e" if (isinstance(score, (int,float)) and score >= 70) else \
                   "#e8943a" if (isinstance(score, (int,float)) and score >= 50) else "#e05c5c"

    sc_items = ""
    if sc.get("buyers_premium_pct"):
        sc_items += f'<li>Buyer\'s premium: {sc["buyers_premium_pct"]}%</li>'
    if sc.get("buyers_premium_gbp"):
        sc_items += f'<li>Buyer\'s premium (fixed): £{sc["buyers_premium_gbp"]:,}</li>'
    if sc.get("admin_fee_gbp"):
        sc_items += f'<li>Admin fee: £{sc["admin_fee_gbp"]:,}</li>'
    if sc.get("vat_elected"):
        sc_items += "<li><strong>VAT elected — purchase price +20%</strong></li>"
    if sc.get("seller_legal_costs_gbp"):
        sc_items += f'<li>Seller\'s legal costs payable by buyer: £{sc["seller_legal_costs_gbp"]:,}</li>'
    if sc.get("non_refundable_deposit"):
        sc_items += "<li>Deposit is non-refundable</li>"
    if sc.get("completion_days"):
        sc_items += f'<li>Completion: {sc["completion_days"]} days</li>'
    if sc.get("special_conditions_missing"):
        sc_items += "<li><em>Special Conditions of Sale not present in pack</em></li>"
    if not sc_items:
        sc_items = "<li>No special conditions extracted</li>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'IBM Plex Sans',sans-serif; font-size:12px; background:#d4d4d4; color:#1a1a1a; -webkit-font-smoothing:antialiased; }}
  .page {{ width:794px; min-height:1123px; background:#fff; margin:0 auto 16px; padding:44px 48px 64px; position:relative; page-break-after:always; }}
  .page:last-child {{ page-break-after:avoid; }}
  .stripe {{ position:absolute; top:0; left:0; right:0; height:3px; background:#1a1a1a; }}
  .logo {{ font-family:'IBM Plex Mono',monospace; font-size:16px; font-weight:600; margin-bottom:4px; }}
  .logo span {{ color:#c8a84b; }}
  .tagline {{ font-family:'IBM Plex Mono',monospace; font-size:9px; color:#7a8fa3; letter-spacing:.1em; text-transform:uppercase; margin-bottom:32px; }}
  .section {{ font-family:'IBM Plex Mono',monospace; font-size:9px; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:#7a8fa3; border-bottom:1px solid #e5e5e5; padding-bottom:6px; margin:28px 0 16px; }}
  .kv {{ display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #f5f5f5; font-size:11px; }}
  .kv .k {{ color:#7a8fa3; }}
  .kv .v {{ font-weight:600; text-align:right; }}
  .score-pill {{ display:inline-block; padding:4px 16px; border-radius:20px; font-family:'IBM Plex Mono',monospace; font-size:22px; font-weight:600; color:#fff; }}
  table {{ width:100%; border-collapse:collapse; }}
  th {{ font-family:'IBM Plex Mono',monospace; font-size:9px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:#7a8fa3; padding:6px 8px; text-align:left; border-bottom:2px solid #e5e5e5; }}
  .footer {{ position:absolute; bottom:24px; left:48px; right:48px; display:flex; justify-content:space-between; font-family:'IBM Plex Mono',monospace; font-size:8px; color:#aaa; border-top:1px solid #e5e5e5; padding-top:8px; }}
  @media print {{
    body {{ background:white; }}
    .page {{ width:100%; margin:0; border:none; box-shadow:none; page-break-after:always; }}
    .page:last-child {{ page-break-after:avoid; }}
    @page {{ size:A4 portrait; margin:0; }}
  }}
</style>
</head>
<body>

<!-- PAGE 1: Cover + Snapshot + Pack Status -->
<div class="page">
  <div class="stripe"></div>
  <div class="logo">Legal<span>Smegal</span></div>
  <div class="tagline">Auction Legal Pack Intelligence Report</div>

  <div class="section">1. Report Snapshot</div>
  <div class="kv"><span class="k">Property</span><span class="v">{esc(address)}</span></div>
  <div class="kv"><span class="k">Postcode</span><span class="v">{esc(postcode)}</span></div>
  <div class="kv"><span class="k">Tenure</span><span class="v">{esc(tenure)}{f" — {lease_yrs} years remaining" if lease_yrs else ""}</span></div>
  <div class="kv"><span class="k">Guide price</span><span class="v">{guide_fmt}</span></div>
  <div class="kv"><span class="k">Report date</span><span class="v">{today_str}</span></div>
  <div style="margin-top:20px;display:flex;align-items:center;gap:16px">
    <span class="score-pill" style="background:{score_colour}">{score}</span>
    <div>
      <div style="font-family:monospace;font-size:9px;color:#7a8fa3;letter-spacing:.06em;text-transform:uppercase">Pack Score / 100</div>
      <div style="font-size:11px;color:#555;margin-top:4px;max-width:500px">{esc(viability)}</div>
    </div>
  </div>

  <div class="section">2. Pack Status</div>
  <div style="display:flex;gap:24px;margin-bottom:16px">
    <div style="text-align:center"><div style="font-size:22px;font-weight:700;color:#e05c5c">{counts.get('critical',0)}</div><div style="font-size:9px;color:#7a8fa3;font-family:monospace;text-transform:uppercase;letter-spacing:.06em">Critical</div></div>
    <div style="text-align:center"><div style="font-size:22px;font-weight:700;color:#e8943a">{counts.get('high',0)}</div><div style="font-size:9px;color:#7a8fa3;font-family:monospace;text-transform:uppercase;letter-spacing:.06em">High</div></div>
    <div style="text-align:center"><div style="font-size:22px;font-weight:700;color:#4a9eff">{counts.get('missing',0)}</div><div style="font-size:9px;color:#7a8fa3;font-family:monospace;text-transform:uppercase;letter-spacing:.06em">Missing</div></div>
    <div style="text-align:center"><div style="font-size:22px;font-weight:700;color:#7a8fa3">{counts.get('note',0)}</div><div style="font-size:9px;color:#7a8fa3;font-family:monospace;text-transform:uppercase;letter-spacing:.06em">Notes</div></div>
  </div>

  <div class="section">3. Special Conditions Summary</div>
  <ul style="padding-left:16px;font-size:11px;line-height:2;color:#333">{sc_items}</ul>

  <div class="footer">
    <span>LegalSmegal Technologies Ltd</span>
    <span>Not legal advice — for investor decision support only</span>
    <span>{today_str}</span>
  </div>
</div>

<!-- PAGE 2: Buyer Cost Exposure + Missing Evidence -->
<div class="page">
  <div class="stripe"></div>
  <div class="section">4. Buyer Cost Exposure</div>
  <table>
    <tr><th>Severity</th><th>Flag</th><th>Source</th></tr>
    {cost_rows}
  </table>

  <div class="section">5. Missing Evidence</div>
  <table>
    <tr><th>Severity</th><th>Flag</th><th>Source</th></tr>
    {missing_rows}
  </table>

  <div class="footer">
    <span>LegalSmegal Technologies Ltd</span>
    <span>Not legal advice — for investor decision support only</span>
    <span>{today_str}</span>
  </div>
</div>

<!-- PAGE 3: Full Flag Register -->
<div class="page">
  <div class="stripe"></div>
  <div class="section">6. Full Flag Register ({len(flags)} flags)</div>
  <table>
    <tr><th>Severity</th><th>Flag</th><th>Source</th></tr>
    {all_flag_rows}
  </table>

  <div class="footer">
    <span>LegalSmegal Technologies Ltd</span>
    <span>Not legal advice — for investor decision support only</span>
    <span>{today_str}</span>
  </div>
</div>

<!-- PAGE 4: Document Inventory -->
<div class="page">
  <div class="stripe"></div>
  <div class="section">7. Document Inventory</div>
  <table>
    <tr><th>File</th><th style="text-align:center">Pages</th><th>Type</th></tr>
    {doc_rows}
  </table>

  <div style="margin-top:32px;padding:16px;background:#f9f9f9;border-left:3px solid #1a1a1a">
    <div style="font-family:monospace;font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px">Important Notice</div>
    <div style="font-size:10px;color:#555;line-height:1.6">
      This report is produced by LegalSmegal Technologies Ltd for investor decision-support purposes only.
      It does not constitute legal advice. You should always instruct a qualified solicitor to review
      the full legal pack before bidding at auction. LegalSmegal accepts no liability for any loss
      arising from reliance on this report.
    </div>
  </div>

  <div class="footer">
    <span>LegalSmegal Technologies Ltd</span>
    <span>Not legal advice — for investor decision support only</span>
    <span>{today_str}</span>
  </div>
</div>

</body>
</html>"""


def _generate_pdf_bytes(summary_json: dict, docs: list) -> bytes:
    """Generate PDF using ReportLab (pure Python, no system deps)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT
    import io as _io, re as _re, json as _json
    from datetime import date

    buf = _io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=16*mm, bottomMargin=20*mm)

    C_BLACK = colors.HexColor("#1a1a1a")
    C_GOLD  = colors.HexColor("#c8a84b")
    C_MUTED = colors.HexColor("#7a8fa3")
    C_CRIT  = colors.HexColor("#e05c5c")
    C_HIGH  = colors.HexColor("#e8943a")
    C_MISS  = colors.HexColor("#4a9eff")
    C_NOTE  = colors.HexColor("#7a8fa3")
    C_GREEN = colors.HexColor("#3ecf8e")
    C_WHITE = colors.white
    MONO = "Courier"
    SANS = "Helvetica"
    W = A4[0] - 36*mm

    def sty(name, **kw):
        from reportlab.lib.styles import getSampleStyleSheet
        base = getSampleStyleSheet()
        return ParagraphStyle(name, parent=base["Normal"], **kw)

    s_logo  = sty("logo", fontName=MONO,  fontSize=16, textColor=C_BLACK, spaceAfter=2)
    s_tag   = sty("tag",  fontName=MONO,  fontSize=7,  textColor=C_MUTED, spaceAfter=14, leading=10)
    s_sec   = sty("sec",  fontName=MONO,  fontSize=7,  textColor=C_MUTED, spaceBefore=10, spaceAfter=5, leading=9)
    s_kv_k  = sty("kvk",  fontName=SANS,  fontSize=9,  textColor=C_MUTED)
    s_kv_v  = sty("kvv",  fontName=SANS+"-Bold", fontSize=9, textColor=C_BLACK, alignment=TA_RIGHT)
    s_body  = sty("body", fontName=SANS,  fontSize=8,  textColor=C_BLACK, leading=11)
    s_bold  = sty("bold", fontName=SANS+"-Bold", fontSize=8, textColor=C_BLACK, leading=11)
    s_small = sty("sml",  fontName=SANS,  fontSize=7,  textColor=C_MUTED, leading=10)
    s_mono  = sty("mono", fontName=MONO,  fontSize=7,  textColor=C_MUTED, leading=10)
    s_li    = sty("li",   fontName=SANS,  fontSize=8,  textColor=C_BLACK, leading=12, leftIndent=8)
    s_ntc   = sty("ntc",  fontName=SANS,  fontSize=7,  textColor=colors.HexColor("#555"), leading=10)

    sj       = summary_json or {}
    flags    = sj.get("flags") or []
    prop     = sj.get("property") or {}
    counts   = sj.get("flag_counts") or {}
    sc       = sj.get("special_conditions") or {}
    score    = sj.get("deal_score", "?")
    address  = prop.get("address") or "?"
    postcode = prop.get("postcode") or ""
    tenure   = prop.get("tenure") or "?"
    lease    = prop.get("lease_years")
    guide_p  = prop.get("guide_price_pence")
    guide    = f"£{int(guide_p)//100:,}" if guide_p else "?"
    viability= sj.get("viability_statement") or ""
    today    = date.today().strftime("%d %B %Y")

    story = []

    def section(title):
        story.append(Spacer(1, 3*mm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=C_MUTED))
        story.append(Paragraph(title.upper(), s_sec))

    def kv(key, val):
        t = Table([[Paragraph(key, s_kv_k), Paragraph(str(val or ""), s_kv_v)]],
                  colWidths=[W*0.55, W*0.45])
        t.setStyle(TableStyle([
            ("BOTTOMPADDING",(0,0),(-1,-1),3),("TOPPADDING",(0,0),(-1,-1),3),
            ("LINEBELOW",(0,0),(-1,-1),0.3,colors.HexColor("#f0f0f0")),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ]))
        story.append(t)

    def sev_col(sev):
        return {"critical":C_CRIT,"high":C_HIGH,"missing":C_MISS}.get((sev or "").lower(), C_NOTE)

    def sev_lbl(sev):
        return {"critical":"CRITICAL","high":"HIGH","missing":"MISSING","note":"NOTE"}.get((sev or "").lower(), "NOTE")

    # Logo
    story.append(Paragraph("LegalSmegal", s_logo))
    story.append(Paragraph("AUCTION LEGAL PACK INTELLIGENCE REPORT", s_tag))

    section("1. Report Snapshot")
    kv("Property", address)
    kv("Postcode", postcode)
    kv("Tenure", tenure + (f" - {lease} years remaining" if lease else ""))
    kv("Guide price", guide)
    kv("Report date", today)

    score_c = C_GREEN if (isinstance(score,(int,float)) and score>=70) else C_HIGH if (isinstance(score,(int,float)) and score>=50) else C_CRIT
    sc_t = Table([[
        Paragraph(f"<b>{score}</b>", sty("scr",fontName=MONO,fontSize=18,textColor=C_WHITE,alignment=TA_CENTER)),
        Paragraph(f"<b>Pack Score / 100</b><br/>{viability}", sty("scv",fontName=SANS,fontSize=8,textColor=C_BLACK,leading=11)),
    ]], colWidths=[22*mm, W-22*mm])
    sc_t.setStyle(TableStyle([("BACKGROUND",(0,0),(0,0),score_c),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    story.append(Spacer(1, 3*mm))
    story.append(sc_t)

    section("2. Pack Status")
    st = Table([[
        Paragraph(f"<b>{counts.get("critical",0)}</b>", sty("s1",fontName=MONO,fontSize=16,textColor=C_CRIT,alignment=TA_CENTER)),
        Paragraph(f"<b>{counts.get("high",0)}</b>",     sty("s2",fontName=MONO,fontSize=16,textColor=C_HIGH,alignment=TA_CENTER)),
        Paragraph(f"<b>{counts.get("missing",0)}</b>",  sty("s3",fontName=MONO,fontSize=16,textColor=C_MISS,alignment=TA_CENTER)),
        Paragraph(f"<b>{counts.get("note",0)}</b>",     sty("s4",fontName=MONO,fontSize=16,textColor=C_NOTE,alignment=TA_CENTER)),
    ],[
        Paragraph("CRITICAL",s_mono),Paragraph("HIGH",s_mono),Paragraph("MISSING",s_mono),Paragraph("NOTES",s_mono),
    ]], colWidths=[W/4]*4)
    st.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    story.append(st)

    section("3. Special Conditions")
    sc_lines = []
    if sc.get("buyers_premium_pct"):     sc_lines.append(f"Buyer premium: {sc["buyers_premium_pct"]}%")
    if sc.get("buyers_premium_gbp"):     sc_lines.append(f"Buyer premium (fixed): £{sc["buyers_premium_gbp"]:,}")
    if sc.get("admin_fee_gbp"):          sc_lines.append(f"Admin fee: £{sc["admin_fee_gbp"]:,}")
    if sc.get("vat_elected"):            sc_lines.append("VAT ELECTED - purchase price +20%")
    if sc.get("non_refundable_deposit"): sc_lines.append("Deposit is non-refundable")
    if sc.get("completion_days"):        sc_lines.append(f"Completion: {sc["completion_days"]} days")
    if sc.get("special_conditions_missing"): sc_lines.append("Special Conditions not present in pack")
    if not sc_lines: sc_lines = ["No special conditions extracted"]
    for line in sc_lines:
        story.append(Paragraph(f"- {line}", s_li))

    def flag_table(flag_list):
        if not flag_list:
            story.append(Paragraph("No flags in this category.", s_small))
            return
        rows = [[Paragraph("SEV",s_mono), Paragraph("FLAG",s_mono), Paragraph("SOURCE",s_mono)]]
        for f in flag_list:
            sev = (f.get("severity") or "note").lower()
            col = sev_col(sev)
            lbl = sev_lbl(sev)
            rows.append([
                Paragraph(f"<font color="#{col.hexval()[1:]}">{lbl}</font>", sty(f"sv{sev}",fontName=MONO,fontSize=6,textColor=col)),
                [Paragraph(f"<b>{f.get("title","")}</b>", s_bold),
                 Paragraph(f.get("summation","") or "", s_body),
                 Paragraph(f.get("action","") or "", s_small)],
                Paragraph((f.get("source_document") or "")[:40], s_small),
            ])
        cw = [20*mm, W-20*mm-26*mm, 26*mm]
        tbl = Table(rows, colWidths=cw, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#f5f5f5")),
            ("LINEBELOW",(0,0),(-1,-1),0.3,colors.HexColor("#eeeeee")),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
            ("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3),
        ]))
        story.append(tbl)

    section("4. Buyer Cost Exposure")
    flag_table([f for f in flags if any(k in ((f.get("title") or "")+(f.get("summation") or "")).lower() for k in ["premium","fee","cost","charge","vat","arrear","deposit","indemnit"])])

    section("5. Missing Evidence")
    flag_table([f for f in flags if (f.get("severity") or "").lower() == "missing"])

    section(f"6. Full Flag Register ({len(flags)} flags)")
    flag_table(flags)

    section("7. Document Inventory")
    if docs:
        dr = [[Paragraph("FILE",s_mono), Paragraph("PG",s_mono), Paragraph("TYPE",s_mono)]]
        for d in docs:
            dr.append([Paragraph((d.get("file_name") or "")[:50],s_body), Paragraph(str(d.get("page_count") or ""),s_body), Paragraph((d.get("doc_type") or "").replace("_"," ").title(),s_body)])
        dt = Table(dr, colWidths=[W*0.55, W*0.1, W*0.35], repeatRows=1)
        dt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#f5f5f5")),("LINEBELOW",(0,0),(-1,-1),0.3,colors.HexColor("#eeeeee")),("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),("LEFTPADDING",(0,0),(-1,-1),3)]))
        story.append(dt)
    else:
        story.append(Paragraph("No documents recorded.", s_small))

    story.append(Spacer(1,5*mm))
    disc = Table([[Paragraph("This report is produced by LegalSmegal Technologies Ltd for investor decision-support purposes only. It does not constitute legal advice. Always instruct a qualified solicitor before bidding at auction.", s_ntc)]], colWidths=[W])
    disc.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),("LINEABOVE",(0,0),(-1,-1),1,C_BLACK),("BACKGROUND",(0,0),(-1,-1),colors.HexColor("#f9f9f9"))]))
    story.append(disc)

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(MONO, 7)
        canvas.setFillColor(C_MUTED)
        canvas.drawString(18*mm, 12*mm, "LegalSmegal Technologies Ltd")
        canvas.drawCentredString(A4[0]/2, 12*mm, "Not legal advice - investor decision support only")
        canvas.drawRightString(A4[0]-18*mm, 12*mm, today)
        canvas.setStrokeColor(colors.HexColor("#e5e5e5"))
        canvas.line(18*mm, 15*mm, A4[0]-18*mm, 15*mm)
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


# ── Email delivery ───────────────────────────────────────────────────────────
def _send_report_email(to_email: str, address: str, report_url: str, pdf_bytes: bytes) -> bool:
    """Send PDF as email attachment via Resend. Returns True on success."""
    if not RESEND_API_KEY:
        logger.warning("[guest2] RESEND_API_KEY not set — skipping email")
        return False
    import base64
    from datetime import date
    subject_addr = address or "Your Legal Pack"
    try:
        payload = {
            "from":    RESEND_FROM,
            "to":      [to_email],
            "subject": f"Your LegalSmegal Report — {subject_addr}",
            "html": f"""
<div style="font-family:'IBM Plex Sans',sans-serif;max-width:560px;margin:0 auto;padding:32px 24px;background:#0d1219;color:#e8edf2">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:18px;font-weight:600;margin-bottom:4px">Legal<span style="color:#c8a84b">Smegal</span></div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#3d5068;letter-spacing:.1em;text-transform:uppercase;margin-bottom:28px">Auction Legal Pack Intelligence</div>
  <div style="font-size:14px;color:#e8edf2;margin-bottom:8px;font-weight:600">Your report is ready</div>
  <div style="font-size:13px;color:#7a8fa3;margin-bottom:8px;line-height:1.6">{subject_addr}</div>
  <div style="font-size:12px;color:#7a8fa3;margin-bottom:24px">Your full legal pack intelligence report is attached as a PDF. You can also view it online using the link below.</div>
  <a href="{report_url}" style="display:inline-block;padding:12px 24px;background:#c8a84b;color:#080c10;font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;text-decoration:none;border-radius:4px">View Online →</a>
  <div style="margin-top:24px;font-family:'IBM Plex Mono',monospace;font-size:9px;color:#3d5068;line-height:1.7">
    Online link valid for 72 hours. PDF attached for permanent reference.<br>
    Not legal advice. LegalSmegal Technologies Ltd.
  </div>
</div>""",
            "attachments": [{
                "filename": "LegalSmegal-Report.pdf",
                "content":  base64.b64encode(pdf_bytes).decode(),
            }],
        }
        resp = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        if resp.status_code in (200, 201):
            logger.info(f"[guest2] Email sent to {to_email}")
            return True
        logger.warning(f"[guest2] Resend HTTP {resp.status_code}: {resp.text[:200]}")
        return False
    except Exception as e:
        logger.warning(f"[guest2] Email exception: {e}")
        return False


# ── Background analysis + delivery ──────────────────────────────────────────
def _run_analysis_and_deliver(session_id: str):
    """Called in a daemon thread after Stripe payment confirmed."""
    session = _get_session(session_id)
    if not session:
        logger.error(f"[guest2] Session {session_id} not found for analysis")
        return

    docs  = session.get("documents") or []
    email = session.get("email") or ""
    logger.info(f"[guest2] Starting analysis for session {session_id} ({len(docs)} docs)")

    try:
        summary_json = _run_llm_analysis(docs)
    except ValueError as e:
        logger.error(f"[guest2] LLM analysis failed (no text?): {e}")
        return
    except Exception as e:
        logger.error(f"[guest2] LLM analysis exception: {e}")
        return

    # Sign report token
    token      = _sign_report_token(session_id)
    report_url = f"{FRONTEND_BASE}/legalsmegal-report.html?guest_session={session_id}&token={token}"

    # Store in session
    with _sessions_lock:
        s = _sessions.get(session_id)
        if s:
            s["summary_json"] = summary_json
            s["report_token"] = token

    # Generate PDF
    address = (summary_json.get("property") or {}).get("address") or "Legal Pack"
    try:
        pdf_bytes = _generate_pdf_bytes(summary_json, docs)
        logger.info(f"[guest2] PDF generated ({len(pdf_bytes):,} bytes) for session {session_id}")
    except Exception as e:
        logger.error(f"[guest2] PDF generation failed: {e}")
        # Still send the email with just the link
        pdf_bytes = None

    # Email
    if pdf_bytes:
        _send_report_email(email, address, report_url, pdf_bytes)
    else:
        # Fallback: link-only email if PDF failed
        _send_link_only_email(email, address, report_url)

    logger.info(f"[guest2] Delivery complete for session {session_id}")


def _send_link_only_email(to_email: str, address: str, report_url: str):
    """Fallback: email without PDF attachment."""
    if not RESEND_API_KEY:
        return
    try:
        requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
            json={
                "from":    RESEND_FROM,
                "to":      [to_email],
                "subject": f"Your LegalSmegal Report — {address}",
                "html": f"""
<div style="font-family:'IBM Plex Sans',sans-serif;max-width:560px;margin:0 auto;padding:32px 24px;background:#0d1219;color:#e8edf2">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:18px;font-weight:600;margin-bottom:28px">Legal<span style="color:#c8a84b">Smegal</span></div>
  <div style="font-size:14px;color:#e8edf2;font-weight:600;margin-bottom:8px">Your report is ready</div>
  <div style="font-size:13px;color:#7a8fa3;margin-bottom:24px">{address}</div>
  <a href="{report_url}" style="display:inline-block;padding:12px 24px;background:#c8a84b;color:#080c10;font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;text-decoration:none;border-radius:4px">View Report →</a>
  <div style="margin-top:24px;font-family:'IBM Plex Mono',monospace;font-size:9px;color:#3d5068">Link valid for 72 hours. Not legal advice.</div>
</div>""",
            },
            timeout=15,
        )
    except Exception as e:
        logger.warning(f"[guest2] Link-only email failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════════════

@guest_bp.route("/api/guest2/create-session", methods=["POST", "OPTIONS"])
def guest2_create_session():
    """No auth. Create in-memory session. Body: { email }"""
    if request.method == "OPTIONS":
        return "", 204
    data  = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    if not email or "@" not in email:
        return jsonify({"error": "Valid email required"}), 400

    session_id = _new_session_id()
    with _sessions_lock:
        _sessions[session_id] = {
            "email":            email,
            "created_at":       time.time(),
            "locked":           False,
            "paid":             False,
            "documents":        [],
            "summary_json":     None,
            "report_token":     None,
            "stripe_session_id": "",
        }
    logger.info(f"[guest2] Session created: {session_id} for {email}")
    return jsonify({"ok": True, "session_id": session_id}), 201


@guest_bp.route("/api/guest2/upload", methods=["POST", "OPTIONS"])
def guest2_upload():
    """No auth. Upload PDF(s) to session.
    Multipart: file(s) + session_id field."""
    if request.method == "OPTIONS":
        return "", 204

    session_id = (request.form.get("session_id") or "").strip()
    session    = _get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found or expired"}), 404
    if session.get("locked"):
        return jsonify({"error": "Session locked after payment initiated"}), 409

    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "No files provided"}), 400

    if len(session["documents"]) + len(files) > MAX_FILES_PER_SESSION:
        return jsonify({"error": f"Maximum {MAX_FILES_PER_SESSION} files per session"}), 400

    uploaded = []
    for f in files:
        raw = f.read()
        if len(raw) > MAX_FILE_BYTES:
            uploaded.append({"file_name": f.filename, "ok": False, "error": "File too large (max 20MB)"})
            continue
        text, pages = _extract_text(raw, f.filename)
        doc_type    = _infer_doc_type(f.filename, text)
        with _sessions_lock:
            s = _sessions.get(session_id)
            if s:
                s["documents"].append({
                    "file_name":      f.filename,
                    "extracted_text": text,
                    "page_count":     pages,
                    "doc_type":       doc_type,
                })
        uploaded.append({"file_name": f.filename, "ok": True, "doc_type": doc_type, "page_count": pages})
        logger.info(f"[guest2] Uploaded {f.filename} ({pages}pp, {doc_type}) to {session_id}")

    return jsonify({"ok": True, "uploaded": uploaded}), 200


@guest_bp.route("/api/guest2/checkout", methods=["POST", "OPTIONS"])
def guest2_checkout():
    """No auth. Create Stripe Checkout session.
    Body: { session_id }. Returns { checkout_url }."""
    if request.method == "OPTIONS":
        return "", 204
    if not STRIPE_SECRET:
        return jsonify({"error": "Payment not configured (STRIPE_SECRET_KEY missing)"}), 503

    data       = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or "").strip()
    session    = _get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found or expired"}), 404
    if not session.get("documents"):
        return jsonify({"error": "No documents uploaded"}), 400

    email = session["email"]

    try:
        resp = requests.post(
            "https://api.stripe.com/v1/checkout/sessions",
            auth=(STRIPE_SECRET, ""),
            data={
                "mode":                                               "payment",
                "line_items[0][price_data][currency]":                "gbp",
                "line_items[0][price_data][unit_amount]":             str(REPORT_PRICE_GBP * 100),
                "line_items[0][price_data][product_data][name]":      "LegalSmegal Legal Pack Report",
                "line_items[0][price_data][product_data][description]": "One-off auction legal pack intelligence report",
                "line_items[0][quantity]":                            "1",
                "customer_email":                                     email,
                "metadata[guest2_session_id]":                        session_id,
                "success_url": f"{FRONTEND_BASE}/legalsmegal-report.html?guest_session={session_id}&processing=1",
                "cancel_url":  f"{FRONTEND_BASE}/legalsmegal-upload-report.html?cancelled=1",
            },
            timeout=15,
        )
        if resp.status_code != 200:
            logger.error(f"[guest2] Stripe error: {resp.text[:300]}")
            return jsonify({"error": "Payment setup failed"}), 502

        stripe_session = resp.json()
        checkout_url   = stripe_session.get("url")
        if not checkout_url:
            return jsonify({"error": "No checkout URL returned"}), 502

        # Lock session against further uploads
        with _sessions_lock:
            s = _sessions.get(session_id)
            if s:
                s["locked"]            = True
                s["stripe_session_id"] = stripe_session.get("id", "")

        return jsonify({"ok": True, "checkout_url": checkout_url}), 200

    except Exception as e:
        logger.exception("guest2_checkout failed")
        return jsonify({"error": str(e)}), 500


@guest_bp.route("/api/webhooks/stripe-guest", methods=["POST"])
def guest2_stripe_webhook():
    """Stripe webhook for guest flow. Verifies signature, fires analysis thread."""
    payload = request.get_data()
    sig     = request.headers.get("Stripe-Signature", "")

    if STRIPE_WH_SECRET:
        try:
            parts    = {k: v for k, v in (p.split("=", 1) for p in sig.split(",") if "=" in p)}
            ts       = parts.get("t", "0")
            v1       = parts.get("v1", "")
            signed   = f"{ts}.{payload.decode('utf-8')}"
            expected = hmac.new(STRIPE_WH_SECRET.encode(), signed.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, v1):
                return jsonify({"error": "Invalid signature"}), 400
            if abs(int(time.time()) - int(ts)) > 300:
                return jsonify({"error": "Timestamp too old"}), 400
        except Exception as e:
            logger.warning(f"[guest2-wh] Signature check failed: {e}")
            return jsonify({"error": "Signature error"}), 400

    try:
        event = request.get_json(force=True, silent=True) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    event_type = event.get("type", "")
    logger.info(f"[guest2-wh] Event: {event_type}")

    if event_type == "checkout.session.completed":
        obj        = (event.get("data") or {}).get("object") or {}
        session_id = (obj.get("metadata") or {}).get("guest2_session_id", "")
        paid       = obj.get("payment_status") == "paid"

        if not session_id:
            logger.warning("[guest2-wh] No guest2_session_id in metadata")
            return jsonify({"ok": True}), 200

        if paid:
            with _sessions_lock:
                s = _sessions.get(session_id)
                if s:
                    s["paid"] = True

            # Fire analysis in background — webhook must return fast
            t = threading.Thread(
                target=_run_analysis_and_deliver,
                args=(session_id,),
                daemon=True,
                name=f"guest2-analysis-{session_id[:8]}",
            )
            t.start()
            logger.info(f"[guest2-wh] Analysis thread started for session {session_id}")

    return jsonify({"ok": True}), 200


@guest_bp.route("/api/guest2/report", methods=["GET", "OPTIONS"])
def guest2_get_report():
    """Token-gated. Returns summary_json for the viewer page.
    Query: ?guest_session=<id>&token=<signed_token>"""
    if request.method == "OPTIONS":
        return "", 204

    session_id = (request.args.get("guest_session") or "").strip()
    token      = (request.args.get("token") or "").strip()

    if not session_id or not token:
        return jsonify({"error": "session_id and token required"}), 401

    verified_id = _verify_report_token(token)
    if not verified_id or verified_id != session_id:
        return jsonify({"error": "Invalid or expired token"}), 401

    session = _get_session(session_id)
    if not session:
        return jsonify({"error": "Session expired — report no longer available"}), 404
    if not session.get("paid"):
        return jsonify({"error": "Payment not confirmed"}), 402

    summary_json = session.get("summary_json")
    if not summary_json:
        # Analysis still running
        return jsonify({"ok": True, "status": "processing"}), 202

    docs = [
        {"file_name": d["file_name"], "doc_type": d["doc_type"], "page_count": d["page_count"]}
        for d in (session.get("documents") or [])
    ]

    return jsonify({
        "ok":           True,
        "status":       "complete",
        "summary_json": summary_json,
        "documents":    docs,
    }), 200


@guest_bp.route("/api/guest2/status", methods=["GET", "OPTIONS"])
def guest2_status():
    """Poll endpoint for the viewer page while analysis runs.
    Query: ?guest_session=<id>&token=<signed_token>
    Returns: { status: 'processing' | 'complete' | 'error' }"""
    if request.method == "OPTIONS":
        return "", 204
    return guest2_get_report()
