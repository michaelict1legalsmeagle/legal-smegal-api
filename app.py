from flask import Flask, request, jsonify
import requests
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service": "Legal Smegal API v1",
        "status": "active",
        "routes": {
            "POST /ask": "{ 'question': '...', 'context': 'optional' }",
            "POST /generate-email": "{ 'analysisData': '...', 'mode': 'professional' }"
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
