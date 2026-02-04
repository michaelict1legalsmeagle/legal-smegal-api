# services/llm_openrouter.py

import os
import json
import re
import time
from typing import Any, Dict, Optional, List

import requests
from requests import Response

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_DEFAULT_MODEL = (os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini") or "openai/gpt-4o-mini").strip()
_TIMEOUT_SECONDS = int((os.getenv("OPENROUTER_TIMEOUT_SECONDS", "20") or "20").strip())
_MAX_RESPONSE_CHARS = int((os.getenv("OPENROUTER_MAX_RESPONSE_CHARS", "200000") or "200000").strip())

# Retry policy (tight + intentional)
_MAX_RETRIES = int((os.getenv("OPENROUTER_MAX_RETRIES", "2") or "2").strip())  # total attempts = 1 + retries
_BASE_BACKOFF_SECONDS = float((os.getenv("OPENROUTER_BACKOFF_SECONDS", "0.8") or "0.8").strip())

# Optional: ask provider for JSON mode when supported.
# OpenRouter will pass-through for some providers/models; harmless if ignored.
_ENABLE_JSON_MODE = (os.getenv("OPENROUTER_JSON_MODE", "1") or "1").strip().lower() in ("1", "true", "yes")


class OpenRouterError(RuntimeError):
    """Deterministic wrapper for upstream failures."""


def _safe_trim(text: str, max_chars: int) -> str:
    if not isinstance(text, str):
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _extract_json(text: str) -> Optional[Any]:
    """
    Best-effort JSON extraction from a model response.
    Bounded to avoid pathological regex/runtime on huge responses.
    """
    if not isinstance(text, str):
        return None

    s = _safe_trim(text.strip(), _MAX_RESPONSE_CHARS)
    if not s:
        return None

    # 1) Direct JSON
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) Strip markdown fences (start/end only)
    s2 = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", s, flags=re.IGNORECASE).strip()
    if s2:
        try:
            return json.loads(s2)
        except Exception:
            pass

    # 3) Find first JSON object/array in text, but bounded and non-greedy
    #    Avoid catastrophic backtracking by keeping it simple.
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", s2 or s)
    if not m:
        return None

    candidate = _safe_trim(m.group(1).strip(), _MAX_RESPONSE_CHARS)
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _headers() -> Dict[str, str]:
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        raise OpenRouterError("OPENROUTER_API_KEY not set")

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # OpenRouter recommends these; omit if empty to avoid sending junk.
    referer = (os.getenv("OPENROUTER_HTTP_REFERER") or "").strip()
    title = (os.getenv("OPENROUTER_APP_NAME") or "legal-smegal-api").strip()

    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    return headers


def _should_retry(resp: Optional[Response], exc: Optional[BaseException]) -> bool:
    if exc is not None:
        # Network hiccups/timeouts are retryable.
        return isinstance(exc, (requests.Timeout, requests.ConnectionError))
    if resp is None:
        return False
    # Retry on rate limit + transient upstream/server errors
    return resp.status_code in (429, 500, 502, 503, 504)


def _post_with_retries(payload: Dict[str, Any]) -> Dict[str, Any]:
    last_exc: Optional[BaseException] = None
    last_resp: Optional[Response] = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers=_headers(),
                json=payload,
                timeout=_TIMEOUT_SECONDS,
            )
            last_resp = resp

            if _should_retry(resp, None) and attempt < _MAX_RETRIES:
                sleep_s = _BASE_BACKOFF_SECONDS * (2 ** attempt)
                time.sleep(sleep_s)
                continue

            resp.raise_for_status()
            return resp.json()

        except Exception as e:
            last_exc = e
            if _should_retry(None, e) and attempt < _MAX_RETRIES:
                sleep_s = _BASE_BACKOFF_SECONDS * (2 ** attempt)
                time.sleep(sleep_s)
                continue
            break

    # Deterministic error output (no mystery)
    if last_resp is not None:
        body = ""
        try:
            body = _safe_trim(last_resp.text or "", 2000)
        except Exception:
            body = ""
        raise OpenRouterError(
            f"OpenRouter request failed (status={getattr(last_resp, 'status_code', 'n/a')}): {body}"
        ) from last_exc

    raise OpenRouterError("OpenRouter request failed (no response)") from last_exc


def _openrouter_chat(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    model = _DEFAULT_MODEL.strip() or "openai/gpt-4o-mini"

    req: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
    }

    # Ask for strict JSON object when possible. If provider ignores, extraction still works.
    if _ENABLE_JSON_MODE:
        req["response_format"] = {"type": "json_object"}

    data = _post_with_retries(req)

    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise OpenRouterError("Malformed OpenRouter response (missing choices/message/content)") from e


def llm_json(*, system: str, prompt: str, temperature: float = 0.2) -> Dict[str, Any]:
    """
    Return a JSON object for backend consumption.
    IMPORTANT: This function must exist because app.py imports it at module import time.

    Contract:
    - If prompt is empty/blank -> return {ok:false, error:"prompt_required"} (no silent defaults).
    - If model returns non-JSON -> return {ok:false, error:"non_json_response", raw:"..."}.
    """
    system_prompt = (system or "").strip() or "Return ONLY valid JSON. No prose."
    user_prompt = (prompt or "").strip()

    if not user_prompt:
        return {"ok": False, "error": "prompt_required"}

    content = _openrouter_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )

    parsed = _extract_json(content)
    if parsed is None:
        return {"ok": False, "error": "non_json_response", "raw": _safe_trim(content, 4000)}

    # Preserve the actual JSON type; callers can enforce dict if they need.
    return {"ok": True, "data": parsed}


def rephrase_for_user(playbook: Dict[str, Any], question: Optional[str] = None) -> str:
    """
    Uses OpenRouter ONLY to improve clarity and tone.
    It must not add facts.
    """
    system_prompt = (
        "You are a calm, experienced UK property solicitor explaining a single issue. "
        "Do not give advice. Do not introduce new risks. "
        "Explain clearly and concisely in plain English."
    )

    issue = (playbook or {}).get("what_it_means", "")
    why = (playbook or {}).get("why_it_matters", "")
    costs = (playbook or {}).get("cost_implications", "")
    q = (question or "").strip()

    user_prompt = f"""Issue explanation:
{issue}

Why it matters:
{why}

Costs:
{costs}
"""
    if q:
        user_prompt += f"\nUser question:\n{q}\n"

    return _openrouter_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
