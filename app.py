from flask import Flask, request, jsonify
from response import generate_response
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Chatbot API is running! Use POST /chat to talk to me."

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    response = generate_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # default ke 5000 jika PORT tidak tersedia
    app.run(host="0.0.0.0", port=port)