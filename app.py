from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, os, json
from supabase import create_client, Client

# ============================================================
# INITIALISE APP
# ============================================================
app = Flask(__name__)
CORS(app)

# --- Environment ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ============================================================
# OPENROUTER CALL WRAPPER
# ============================================================
def call_openrouter(messages, model="gpt-4o-mini"):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://legal-smegal-api-final.onrender.com",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 800,
    }

    try:
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        return content
    except Exception as e:
        return f"⚠️ Error contacting AI model: {str(e)}"

# ============================================================
# ROUTES
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Legal Smegal API v2",
        "status": "active",
        "routes": ["/ask", "/generate-email", "/save-analysis"]
    })

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    question = data.get("question", "")
    mode = data.get("mode", "concise")

    if not question:
        return jsonify({"error": "Missing question field"}), 400

    prompt = f"You are a UK property solicitor AI. Mode: {mode}. Question: {question}"

    answer = call_openrouter([
        {"role": "system", "content": "You are Legal Smegal, a UK solicitor AI providing clear, plain-English legal insights."},
        {"role": "user", "content": prompt}
    ])

    return jsonify({"answer": answer})

@app.route("/generate-email", methods=["POST"])
def generate_email():
    data = request.get_json() or {}
    analysis = data.get("analysisData", "")
    mode = data.get("mode", "professional")

    prompt = f"""
You are Legal Smegal, a UK solicitor AI.
Draft a formal, plain-English email summarizing key risks, findings, and recommendations from this property analysis.

Mode: {mode.upper()}
Analysis:
{analysis}

Close with: "Please review and advise next steps."
    """

    email_text = call_openrouter([
        {"role": "system", "content": "You are Legal Smegal, a UK property solicitor AI."},
        {"role": "user", "content": prompt}
    ])

    return jsonify({"emailDraft": email_text})

@app.route("/save-analysis", methods=["POST"])
def save_analysis():
    data = request.get_json() or {}
    title = data.get("title", "Untitled Analysis")
    user_id = data.get("user_id", "anonymous")
    summary = data.get("summary", "")
    score = data.get("score", 0)
    risks = data.get("risks", "")
    analysis = json.dumps(data.get("analysis", {}))

    try:
        supabase.table("analyses").insert({
            "title": title,
            "user_id": user_id,
            "summary": summary,
            "score": score,
            "risks": risks,
            "analysis": analysis
        }).execute()

        return jsonify({"status": "success", "message": "Analysis saved safely."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
