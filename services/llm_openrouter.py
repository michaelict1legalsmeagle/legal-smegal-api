# services/llm_openrouter.py
# CONTRACT-ENFORCED LLM JSON LAYER
# Deterministic. No silent partial payloads.

import os
import json
import re
from typing import Any, Dict, Optional, List

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_DEFAULT_MODEL = (os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini") or "").strip() or "openai/gpt-4o-mini"
_TIMEOUT_SECONDS = int(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "30") or "30")


def _extract_json(text: str) -> Optional[Any]:
    if not isinstance(text, str):
        return None

    s = text.strip()

    # Direct parse
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

    # Extract first JSON block
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

    sys_text = (system or (
        "Return ONLY valid JSON (no markdown, no prose).\n"
        "Schema (MUST include all keys):\n"
        "{\n"
        '  "score": <number 0-100>,\n'
        '  "summary": <string>,\n'
        '  "positives": <array of strings>,\n'
        '  "risks": <array of strings>\n'
        "}\n"
        "Do not include any other keys."
    )).strip()

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

    return [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": (prompt or "").strip()},
    ]


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

    parsed.setdefault("positives", [])
    parsed.setdefault("risks", [])

    # Normalise array types (strict on required keys, tolerant on optional arrays)
    if not isinstance(parsed.get("positives"), list):
        parsed["positives"] = []
    else:
        parsed["positives"] = [str(x) for x in parsed["positives"] if isinstance(x, (str, int, float))]

    if not isinstance(parsed.get("risks"), list):
        parsed["risks"] = []
    else:
        parsed["risks"] = [str(x) for x in parsed["risks"] if isinstance(x, (str, int, float))]

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

    validated = _validate_analysis_contract(parsed)

    return validated


def rephrase_for_user(playbook: Dict[str, Any], question: Optional[str] = None) -> str:

    system_prompt = (
        "You are a calm UK property solicitor explaining one issue clearly."
    )

    issue = (playbook or {}).get("what_it_means", "")
    why = (playbook or {}).get("why_it_matters", "")
    costs = (playbook or {}).get("cost_implications", "")

    user_prompt = f"Issue: {issue}\nWhy: {why}\nCosts: {costs}"

    return _openrouter_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
