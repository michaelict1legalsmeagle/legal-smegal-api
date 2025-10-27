from flask import Flask, request, jsonify
import os
import requests

app = Flask(__name__)

# --- Root route (status check) ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Legal Smegal API",
        "status": "active",
        "usage": "POST /ask with JSON {'question': '...'} and header x-api-key"
    })

# --- Main /ask endpoint ---
@app.route("/ask", methods=["POST"])
def ask():
    # Validate subscription key
    client_key = request.headers.get("x-api-key")
    server_key = os.getenv("LEGAL_SMEGAL_SUB_KEY")

    if not client_key or client_key != server_key:
        return jsonify({"error": "Unauthorized: invalid or missing API key"}), 401

    # Parse JSON input
    data = request.get_json(silent=True)
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field in JSON body"}), 400

    question = data["question"]

    # Prepare OpenRouter call
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are Legal Smegal, a UK legal explainer bot."},
                {"role": "user", "content": question}
            ]
        }

        # Send request to OpenRouter API
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        data = response.json()

        # Handle valid and error responses
        if response.status_code == 200 and "choices" in data:
            answer = data["choices"][0]["message"]["content"]
            return jsonify({"answer": answer})
        else:
            return jsonify({
                "error": "OpenRouter API error",
                "details": data
            }), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Run Flask app ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # default for Render
    app.run(host="0.0.0.0", port=port)
