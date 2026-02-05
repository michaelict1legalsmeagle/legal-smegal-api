# services/llm_openrouter.py

import os
import json
import re
from typing import Any, Dict, Optional, List, Union

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_DEFAULT_MODEL = (os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini") or "").strip() or "openai/gpt-4o-mini"
_TIMEOUT_SECONDS = int(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "30") or "30")


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


def _openrouter_chat(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Optional but recommended by OpenRouter; safe if unset
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


def _normalize_messages(
    *,
    system: Optional[str],
    prompt: Optional[str],
    messages: Optional[List[Dict[str, str]]],
) -> List[Dict[str, str]]:
    """
    Adapter: accept either:
      - system + prompt
      - messages (optionally plus system)
    Produces a valid OpenRouter chat-completions message list.
    """
    sys_text = (system or "Return ONLY valid JSON. No prose.").strip()

    if messages and isinstance(messages, list):
        out: List[Dict[str, str]] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role in ("system", "user", "assistant") and isinstance(content, str):
                out.append({"role": role, "content": content})
        # Ensure a system message exists (prepend if missing)
        if not any(m.get("role") == "system" for m in out):
            out.insert(0, {"role": "system", "content": sys_text})
        return out

    user_text = (prompt or "").strip()
    if not user_text:
        # Caller provided neither messages nor prompt -> this is a caller error,
        # but we fail in a controlled way for API stability.
        return [{"role": "system", "content": sys_text}, {"role": "user", "content": ""}]

    return [{"role": "system", "content": sys_text}, {"role": "user", "content": user_text}]


def llm_json(
    *,
    system: Optional[str] = None,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Contract-preserving JSON producer.

    Back-compat:
      - existing callers pass (system, prompt)
    New-compat:
      - callers may pass messages=[{role, content}, ...] (system optional)

    Returns:
      {"ok": True, "data": <parsed_json>} OR {"ok": False, "error": "...", "raw": "..."}
    """
    msg_list = _normalize_messages(system=system, prompt=prompt, messages=messages)

    content = _openrouter_chat(msg_list, temperature=float(temperature))

    parsed = _extract_json(content)
    if parsed is None:
        return {"ok": False, "error": "non_json_response", "raw": content}

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
