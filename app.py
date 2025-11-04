# ============================================================
#  IMPORTS
# ============================================================
from flask import Flask, request, jsonify
import requests
import os
import json
import uuid
import time
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
print("🟢 Using Supabase Key Prefix:", SUPABASE_KEY[:20])

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
#  OPENROUTER API WRAPPER (HARDENED)
# ============================================================

def call_openrouter(messages, model="gpt-4o-mini"):
    """Robust OpenRouter API call with retry + fallback models."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://legal-smegal-api-final.onrender.com",
        "Content-Type": "application/json",
    }

    model_chain = [model, "gpt-4o", "mistral-nemo:latest"]
    data = {
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 700,
    }

    for current_model in model_chain:
        data["model"] = current_model
        for attempt in range(3):
            try:
                print(f"🔹 Trying model {current_model} (attempt {attempt+1})...")
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers, json=data, timeout=60
                )

                if resp.status_code == 200:
                    j = resp.json()
                    content = (
                        j.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                    )
                    if content:
                        return content

                elif resp.status_code == 503:
                    print("⚠️ Model overloaded (503). Retrying...")
                    time.sleep(2 ** attempt)
                    continue

                else:
                    print(f"❌ Model {current_model} returned {resp.status_code}")
                    break

            except requests.Timeout:
                print(f"⏱ Timeout from {current_model}, retrying...")
                time.sleep(2 ** attempt)
                continue

            except Exception as e:
                print(f"⚠️ Unexpected error with {current_model}: {e}")
                break

        print(f"❌ All retries failed for {current_model}, trying next model...")

    fallback_error = {
        "error": {
            "code": 503,
            "message": "All AI models are currently overloaded. Please retry shortly.",
            "source": "openrouter",
            "status": "unavailable",
        }
    }
    print("🚨 All models failed. Returning fallback JSON.")
    return json.dumps(fallback_error)

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
    user_id = str(uuid.uuid4())
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


@app.route("/get-analyses", methods=["GET"])
def get_analyses():
    """Retrieve recent analyses from Supabase."""
    try:
        response = supabase.table("analyses").select("*").order("created_at", desc=True).limit(10).execute()
        data = response.data
        if not data:
            return jsonify({"analyses": [], "message": "No analyses found", "status": "empty"}), 200
        return jsonify({"analyses": data, "status": "success"}), 200
    except Exception as e:
        print("❌ Error in /get-analyses:", str(e))
        return jsonify({"error": str(e), "status": "failed"}), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Legal Smegal API Final",
        "status": "active",
        "routes": {
            "POST /ask": "{ 'question': '...', 'context': 'optional', 'mode': 'concise|professional|detailed' }",
            "POST /generate-email": "{ 'analysisData': '...', 'mode': 'professional' }",
            "POST /save-analysis": "{ 'title': '...', 'summary': '...', 'score': 0, 'risks': '...', 'analysis': {...} }",
            "GET /get-analyses": "Retrieves the 10 most recent analyses from Supabase"
        }
    })

# ============================================================
#  ENTRY POINT
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
