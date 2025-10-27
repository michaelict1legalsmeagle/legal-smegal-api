from flask import Flask, request, jsonify
import os
import requests
import json
from datetime import datetime

app = Flask(__name__)

# --- Configuration ---
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY — export it first.")

# Optional: simple API key auth for subscribers
SUBSCRIBER_KEY = os.getenv("LEGAL_SMEGAL_SUB_KEY", "demo-key")

# --- Helper: call OpenRouter directly ---
def ask_openrouter(prompt: str):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
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

        if "choices" in payload and len(payload["choices"]) > 0:
            return payload["choices"][0]["message"]["content"]
        else:
            return f"[Unexpected response format]\n{json.dumps(payload, indent=2)}"

    except requests.exceptions.RequestException as e:
        return f"[HTTP error] {str(e)}"
    except json.JSONDecodeError:
        return f"[Invalid JSON] Raw response:\n{response.text}"


# --- API Routes ---
@app.route("/ask", methods=["POST"])
def ask():
    # Authentication
    client_key = request.headers.get("x-api-key")
    if client_key != SUBSCRIBER_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(silent=True)
    if not data or "question" not in data:
        return jsonify({"error": "Missing question"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    answer = ask_openrouter(question)

    # Log each query to a simple text file
    with open("queries.log", "a") as f:
        f.write(f"[{datetime.now().isoformat()}] Q: {question}\nA: {answer}\n\n")

    return jsonify({"answer": answer})


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Legal Smegal API",
        "status": "active",
        "usage": "POST /ask with JSON {'question': '...'} and header x-api-key"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)

