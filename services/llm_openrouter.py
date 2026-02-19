# services/llm_openrouter.py
# CONTRACT-ENFORCED LLM JSON LAYER
# Guarantees:
# - Deterministic JSON extraction
# - Auto-unwrap common wrapper shapes (ok/data, analysis, result)
# - Enforce analysis contract at ROOT (score + summary always present or hard fail)

import os
import json
import re
from typing import Any, Dict, Optional, List

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_DEFAULT_MODEL = (os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini") or "").strip() or "openai/gpt-4o-mini"
_TIMEOUT_SECONDS = int(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "30") or "30")

LLM_LAYER_VERSION = os.getenv("LLM_LAYER_VERSION", "llm_openrouter_v2_contract_root")


def _extract_json(text: str) -> Optional[Any]:
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

    # First JSON block
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
        if not any(m.get("role") == "system" for m in out):
            out.insert(0, {"role": "system", "content": sys_text})
        return out

    return [{"role": "system", "content": sys_text}, {"role": "user", "content": (prompt or "").strip()}]


def _unwrap_analysis_obj(obj: Any) -> Any:
    """
    Accepts common wrapper shapes and returns the likely analysis dict.
    Handles cases like:
      - {ok:true, data:{...}}
      - {data:{...}}
      - {analysis:{...}}
      - {result:{...}}
    If no wrapper found, returns obj as-is.
    """
    if not isinstance(obj, dict):
        return obj

    # Highest confidence wrappers first
    for key in ("data", "analysis", "result", "output"):
        inner = obj.get(key)
        if isinstance(inner, dict):
            return inner

    return obj


def _validate_analysis_contract(parsed: Any) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError("Analysis output is not a JSON object")

    if "score" not in parsed:
        raise ValueError("Analysis contract violation: missing 'score'")

    if "summary" not in parsed:
        raise ValueError("Analysis contract violation: missing 'summary'")

    if not isinstance(parsed["score"], (int, float)):
        raise ValueError("Analysis contract violation: score must be numeric")

    if not isinstance(parsed["summary"], str):
        raise ValueError("Analysis contract violation: summary must be string")

    # Normalize optional arrays for UI stability
    if "positives" not in parsed or not isinstance(parsed.get("positives"), list):
        parsed["positives"] = []
    if "risks" not in parsed or not isinstance(parsed.get("risks"), list):
        parsed["risks"] = []

    return parsed


def llm_json(
    *,
    system: Optional[str] = None,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    msg_list = _normalize_messages(system=system, prompt=prompt, messages=messages)

    content = _openrouter_chat(msg_list, temperature=float(temperature))

    parsed = _extract_json(content)
    if parsed is None:
        raise ValueError("Model returned non-JSON response")

    # Auto-unwrap wrappers so root has score/summary
    candidate = _unwrap_analysis_obj(parsed)

    # If still missing score but original had a dict, attempt one more unwrap layer
    if isinstance(candidate, dict) and ("score" not in candidate or "summary" not in candidate):
        candidate2 = _unwrap_analysis_obj(candidate)
        candidate = candidate2

    validated = _validate_analysis_contract(candidate)

    # Stamp for deploy verification (safe extra key; frontend ignores)
    validated["_llm_layer_version"] = LLM_LAYER_VERSION

    return validated


def rephrase_for_user(playbook: Dict[str, Any], question: Optional[str] = None) -> str:
    system_prompt = (
        "You are a calm, experienced UK property solicitor explaining a single issue. "
        "Do not introduce new facts. Explain clearly and concisely."
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
