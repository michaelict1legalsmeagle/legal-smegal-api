from flask import Flask, request, jsonify
import requests
import os
import json
import uuid
from flask_cors import CORS
from supabase import create_client, Client

# ============================================================
#  FLASK APP INITIALIZATION
# ============================================================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ============================================================
#  ENVIRONMENT VARIABLES
# ============================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================================
#  PROMPT BUILDERS
# ============================================================

def build_email_prompt(analysis_text: str, mode: str) -> str:
    """Solicitor-style email drafting prompt."""
    return f"""
You are Legal Smegal, a UK property and auction law AI assistant for solicitors,
investors, and pension trustees. Write a formal email summarizing deal findings,
risks, and recommendations from the analysis below.

Rules:
- Formal plain-English solicitor tone.
- No markdown, emojis, HTML, or code formatting.
- Output only the email text body.
- Do NOT alter any UI or layout.
- Preserve professional legal phrasing and structure.

MODE SPECIFICATIONS:
1. concise → ≤ 100 words or 3–5 bullet points.
2. professional → 150–200 words, structured, formal tone.
3. detailed → 250–400 words, full due-diligence analysis with clause references
   and mitigation strategies.

Include clear identification of potential risks (title, covenant, rights of way,
planning, overage, restrictions) and end with:
“Please review and advise next steps.”

Selected Mode: {mode.upper()}

Analysis Context:
{analysis_text}
"""


def build_qa_prompt(question: str, context: str = "", mode: str = "concise") -> str:
    """Legal Q&A prompt tuned for concise, adaptive answers."""
    base_prompt = f"""
You are Legal Smegal, a UK property solicitor AI.
Answer clearly, accurately, and in plain English.
Avoid unnecessary legal padding, citations, or long-winded phrasing.
"""

    if mode == "concise":
        base_prompt += "\nKeep your response under 120 words. Focus only on the main legal issue and risk mitigation."
    elif mode == "professional":
        base_prompt += "\nProvide a well-reasoned solicitor-style answer, around 150–200 words, structured with logic and clarity."
    elif mode == "detailed":
        base_prompt += "\nProvide a comprehensive analysis (250–400 words), including relevant clauses, solicitor reasoning, and protective recommendations."

    return f"""{base_prompt}

Question: {question}
Context: {context}
"""

# ============================================================
#  OPENROUTER API WRAPPER
# ============================================================

def call_openrouter(messages, model="gpt-4o-mini"):
    """Generic OpenRouter API call wrapper."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://legal-smegal-api-final.onrender.com",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 700,
    }
    try:
        print("🔹 Sending request to OpenRouter...")
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers, json=data, timeout=60
        )
        print(f"🔹 OpenRouter response status: {resp.status_code}")
        resp.raise_for_status()
        j = resp.json()
        content = j.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return content or "No response from Legal Smegal AI."
    except requests.Timeout:
        return "⏱ Legal Smegal service timed out. Please retry shortly."
    except Exception as e:
        print("❌ OpenRouter API error:", str(e))
        return "An internal AI communication error occurred."

# ============================================================
#  ROUTES
# ============================================================

@app.route("/generate-email", methods=["POST"])
def generate_email():
    data = request.get_json() or {}
    analysis_text = data.get("analysisData", "")
    mode = data.get("mode", "professional").lower()
    try:
        prompt = build_email_prompt(analysis_text, mode)
        email_text = call_openrouter([
            {"role": "system", "content": "You are Legal Smegal, a professional UK solicitor AI."},
            {"role": "user", "content": prompt}
        ])
        return jsonify({"emailDraft": email_text})
    except Exception as e:
        print("❌ Error in /generate-email:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json() or {}
    question = data.get("question", "")
    context = data.get("context", "")
    mode = data.get("mode", "concise").lower()
    try:
        prompt = build_qa_prompt(question, context, mode)
        answer = call_openrouter([
            {"role": "system", "content": "You are Legal Smegal, a UK property solicitor AI providing clear, plain-English insights."},
            {"role": "user", "content": prompt}
        ])
        return jsonify({"answer": answer})
    except Exception as e:
        print("❌ Error in /ask:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/save-analysis", methods=["POST"])
def save_analysis():
    """Save analysis data to Supabase with UUID auto-generation."""
    data = request.get_json() or {}
    title = data.get("title", "Untitled Analysis")
    user_id = str(uuid.uuid4())  # ✅ Auto-generate valid UUID
    summary = data.get("summary", "")
    score = data.get("score", 0)
    risks = data.get("risks", "")
    analysis = json.dumps(data.get("analysis", {}))

    try:
        response = supabase.table("analyses").insert({
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "title": title,
            "summary": summary,
            "score": score,
            "risks": risks,
            "analysis": analysis
        }).execute()

        print("✅ Supabase insert response:", response)
        return jsonify({"status": "success", "message": "Analysis saved safely."})
    except Exception as e:
        print("❌ Error in /save-analysis:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Legal Smegal API Final",
        "status": "active",
        "routes": {
            "POST /ask": "{ 'question': '...', 'context': 'optional', 'mode': 'concise|professional|detailed' }",
            "POST /generate-email": "{ 'analysisData': '...', 'mode': 'professional' }",
            "POST /save-analysis": "{ 'title': '...', 'summary': '...', 'score': 0, 'risks': '...', 'analysis': {...} }"
        }
    })

# ============================================================
#  ENTRY POINT
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
