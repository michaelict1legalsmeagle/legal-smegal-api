import os
import requests
import json

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("Missing OPENROUTER_API_KEY — export it first.")

def ask_openrouter(prompt: str):
    """Directly call OpenRouter API and handle any response safely."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://legal-smegal.ai",
        "X-Title": "Legal Smegal"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        payload = response.json()

        # Handle OpenRouter-style responses
        if "choices" in payload and len(payload["choices"]) > 0:
            return payload["choices"][0]["message"]["content"]
        else:
            return f"[Unexpected response format]\n{json.dumps(payload, indent=2)}"

    except requests.exceptions.RequestException as e:
        return f"[HTTP error] {str(e)}"
    except json.JSONDecodeError:
        return f"[Invalid JSON] Raw response:\n{response.text}"

print("⚖️  Legal Smegal is ready. Type your question (or 'exit' to quit).")

while True:
    q = input("\nYou: ").strip()
    if q.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break
    result = ask_openrouter(q)
    print("\nSolicitor:", result)
