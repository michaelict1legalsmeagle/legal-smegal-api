# services/llm_openrouter.py

import os
import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def rephrase_for_user(playbook, question=None):
    """
    Uses OpenRouter ONLY to improve clarity and tone.
    It must not add facts.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    system_prompt = (
        "You are a calm, experienced UK property solicitor explaining a single issue. "
        "Do not give advice. Do not introduce new risks. "
        "Explain clearly and concisely in plain English."
    )

    user_prompt = f"""
Issue explanation:
{playbook['what_it_means']}

Why it matters:
{playbook['why_it_matters']}

Costs:
{playbook['cost_implications']}
"""

    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2
        },
        timeout=15
    )

    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
