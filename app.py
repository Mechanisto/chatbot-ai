from flask import Flask, request, jsonify
from response import generate_response
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
@app.route("/", methods=["GET"])
def home():
    return "Chatbot API is running! Use POST /chat to talk to me."

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "GET":
        return "Chatbot API is working. Please send a POST request with JSON: {\"message\": \"your message\"}"

    # Cek kalau JSON ada
    if not request.is_json:
        return jsonify({"error": "Invalid request format, JSON expected."}), 400

    data = request.get_json()
    user_input = data.get("message")

    if not user_input:
        return jsonify({"error": "No message provided."}), 400

    response = generate_response(user_input)
    return jsonify({"response": response})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # default ke 5000 jika PORT tidak tersedia
    app.run(host="0.0.0.0", port=port)