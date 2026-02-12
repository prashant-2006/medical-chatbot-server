import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_engine import get_answer  # Import our engine

app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('query')

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    print(f"📩 Received query: {user_query}")
    
    # Get answer from RAG engine
    response = get_answer(user_query)
    
    print(f"📤 Sending response: {response}")
    return jsonify({"answer": response})


@app.route('/')
def home():
    return jsonify({"status": "Server is running"})


if __name__ == '__main__':
    print("🚀 Server starting...")
    
    # Railway provides PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    
    # Must bind to 0.0.0.0 for Railway
    app.run(host='0.0.0.0', port=port, debug=False)
