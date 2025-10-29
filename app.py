from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

# --- Environment Variables ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

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


def build_qa_prompt(question: str, context: str = "") -> str:
    """Legal Q&A prompt for dataset-driven reasoning."""
    return f"""
You are Legal Smegal, a specialist UK property solicitor AI.
Answer user questions about auction properties and legal packs accurately and
professionally.

If context from the dataset is provided, use it.
Cite relevant clauses, legal principles, or typical solicitor reasoning when applicable.
Respond in plain legal English suitable for investors or trustees.

Question: {question}
Context: {context}
"""

# ============================================================
#  CORE FUNCTIONS
# ============================================================

def call_openrouter(messages, model="gpt-4o-mini"):
    """Generic OpenRouter API call wrapper."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://legal-smegal-api.onrender.com",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 700,
    }
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
                         headers=headers, json=data, timeout=60)
    resp.raise_for_status()
    j = resp.json()
    return j.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

# ============================================================
#  ROUTES
# ============================================================

@app.route("/generate-email", methods=["POST"])
def generate_email():
    """Generate solicitor-style email draft."""
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
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask_question():
    """Interactive Q&A endpoint using dataset context."""
    data = request.get_json() or {}
    question = data.get("question", "")
    context = data.get("context", "")  # optional field for embedding lookups
    try:
        prompt = build_qa_prompt(question, context)
        answer = call_openrouter([
            {"role": "system", "content": "You are Legal Smegal, a UK property solicitor AI providing legal insights."},
            {"role": "user", "content": prompt}
        ])
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    """Health check."""
    return jsonify({
        "service": "Legal Smegal API v1",
        "status": "active",
        "routes": {
            "POST /ask": "{ 'question': '...', 'context': 'optional' }",
            "POST /generate-email": "{ 'analysisData': '...', 'mode': 'professional' }"
        }
    })

# ============================================================
#  ENTRY POINT
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
