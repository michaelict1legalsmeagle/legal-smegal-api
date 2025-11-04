from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def call_openrouter(messages, model="gpt-4o-mini"):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://legal-smegal-api-final.onrender.com",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 600,
    }

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers, json=data, timeout=60
        )
        resp.raise_for_status()
        j = resp.json()
        return j["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error contacting AI model: {str(e)}"

@app.route("/", methods=["GET"])
def home():
    return jsonify({"service": "Legal Smegal API v1", "status": "active"})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    question = data.get("question", "")
    mode = data.get("mode", "concise")

    if not question:
        return jsonify({"error": "Missing 'question' field"}), 400

    prompt = f"You are a UK property solicitor AI. Mode: {mode}. Question: {question}"

    answer = call_openrouter([
        {"role": "system", "content": "You are Legal Smegal, a UK solicitor AI."},
        {"role": "user", "content": prompt}
    ])

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
