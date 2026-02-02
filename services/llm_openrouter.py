# services/llm_openrouter.py

import os
import json
import re
from typing import Any, Dict, Optional

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip() or "openai/gpt-4o-mini"
_TIMEOUT_SECONDS = int(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "20") or "20")

def _extract_json(text: str) -> Optional[Any]:
    """Best-effort JSON extraction from a model response."""
    if not isinstance(text, str):
        return None
    s = text.strip()
    # Direct JSON
    try:
        return json.loads(s)
    except Exception:
        pass
    # Strip markdown fences
    s2 = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        return json.loads(s2)
    except Exception:
        pass
    # Find first {...} or [...]
    m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None

def _openrouter_chat(messages, temperature: float = 0.2) -> str:
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Optional but recommended by OpenRouter; safe if unset in env
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", ""),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "legal-smegal-api"),
        },
        json={
            "model": _DEFAULT_MODEL,
            "messages": messages,
            "temperature": float(temperature),
        },
        timeout=_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def llm_json(*, system: str, prompt: str, temperature: float = 0.2) -> Dict[str, Any]:
    """
    Return a JSON object for backend consumption.
    IMPORTANT: This function must exist because app.py imports it at module import time.
    """
    system_prompt = system or "Return ONLY valid JSON. No prose."
    user_prompt = prompt or ""

    content = _openrouter_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )

    parsed = _extract_json(content)
    if parsed is None:
        # Keep it non-fatal for callers; expose raw for debugging.
        return {"ok": False, "error": "non_json_response", "raw": content}

    # Ensure dict response shape for the API
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

    user_prompt = f"""Issue explanation:
{issue}

Why it matters:
{why}

Costs:
{costs}
"""

    return _openrouter_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
