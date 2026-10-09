import os
from flask import Flask, request, jsonify
from groq import Groq

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "healthy",
        "service": "Kvasir-HealthAI API",
        "model": "llama-3.1-8b-instant",
        "groq_api_key_configured": bool(os.environ.get("GROQ_API_KEY"))
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json() or {}
    prompt = data.get("prompt", "Analyze gastrointestinal image findings.")
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return jsonify({"error": "GROQ_API_KEY environment variable is missing"}), 400
        
    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are Kvasir-HealthAI, an expert medical AI assistant for gastrointestinal diagnosis."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant"
        )
        report = completion.choices[0].message.content
        return jsonify({
            "result": report,
            "model": "llama-3.1-8b-instant"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
