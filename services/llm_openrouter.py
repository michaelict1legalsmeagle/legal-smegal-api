# services/llm_openrouter.py

import os
import json
import re
import time
from typing import Any, Dict, Optional, List, Union

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

    referer = (os.getenv("OPENROUTER_HTTP_REFERER") or "").strip()
    title = (os.getenv("OPENROUTER_APP_NAME") or "legal-smegal-api").strip()

    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    return headers


def _should_retry(resp: Optional[Response], exc: Optional[BaseException]) -> bool:
    if exc is not None:
        return isinstance(exc, (requests.Timeout, requests.ConnectionError))
    if resp is None:
        return False
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

    if _ENABLE_JSON_MODE:
        req["response_format"] = {"type": "json_object"}

    data = _post_with_retries(req)

    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise OpenRouterError("Malformed OpenRouter response (missing choices/message/content)") from e


def _coerce_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        return str(v)
    except Exception:
        return ""


def _extract_prompt_from_payload(payload: Dict[str, Any]) -> str:
    """
    Adapter: accept prompt from common shapes without requiring callers to match exactly.
    Priority:
      1) payload["prompt"]
      2) payload["options"]["prompt"]
      3) payload["input"] / payload["text"]
      4) payload["messages"] -> last user content
    """
    if not isinstance(payload, dict):
        return ""

    p = _coerce_str(payload.get("prompt")).strip()
    if p:
        return p

    opts = payload.get("options")
    if isinstance(opts, dict):
        p2 = _coerce_str(opts.get("prompt")).strip()
        if p2:
            return p2

    p3 = _coerce_str(payload.get("input") or payload.get("text")).strip()
    if p3:
        return p3

    msgs = payload.get("messages")
    if isinstance(msgs, list):
        # Find last user message content
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                c = _coerce_str(m.get("content")).strip()
                if c:
                    return c

    return ""


def _extract_system_from_payload(payload: Dict[str, Any], fallback: str) -> str:
    if not isinstance(payload, dict):
        return (fallback or "").strip()
    s = _coerce_str(payload.get("system")).strip()
    if s:
        return s
    # Some callers may place system inside options
    opts = payload.get("options")
    if isinstance(opts, dict):
        s2 = _coerce_str(opts.get("system")).strip()
        if s2:
            return s2
    return (fallback or "").strip()


def llm_json(*, system: str, prompt: Any, temperature: float = 0.2) -> Dict[str, Any]:
    """
    Return a JSON object for backend consumption.
    IMPORTANT: This function must exist because app.py imports it at module import time.

    Contract (preserved):
    - If prompt is empty/blank -> return {ok:false, error:"prompt_required"} (no silent defaults).
    - If model returns non-JSON -> return {ok:false, error:"non_json_response", raw:"..."}.

    Adapter (added, production-safe):
    - If `prompt` is a dict payload, extract prompt from {prompt, options.prompt, messages, ...}.
    - If `prompt` is a list (messages), try to derive the last user prompt.
    """
    # Allow callers to pass a full payload dict instead of a string prompt.
    payload_dict: Optional[Dict[str, Any]] = prompt if isinstance(prompt, dict) else None

    system_prompt = (system or "").strip()
    if payload_dict is not None:
        system_prompt = _extract_system_from_payload(payload_dict, system_prompt)

    system_prompt = system_prompt or "Return ONLY valid JSON. No prose."

    user_prompt = ""
    if payload_dict is not None:
        user_prompt = _extract_prompt_from_payload(payload_dict)
    elif isinstance(prompt, list):
        # Some callers might pass messages directly.
        user_prompt = _extract_prompt_from_payload({"messages": prompt})
    else:
        user_prompt = _coerce_str(prompt).strip()

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
