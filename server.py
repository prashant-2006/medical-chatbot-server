import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from rag_engine import process_document, get_answer

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def cleanup_old_files():
    current_time = time.time()
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(file_path):
            if current_time - os.path.getmtime(file_path) > 7200:
                os.remove(file_path)

@app.route('/upload', methods=['POST'])
def upload_file():
    cleanup_old_files() 
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    session_id = request.form.get('session_id')
    
    if file.filename == '' or not session_id:
        return jsonify({"error": "Missing file or session_id"}), 400
        
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(file_path)
        
        try:
            process_document(file_path, session_id)
            return jsonify({"message": "File processed successfully"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    return jsonify({"error": "Invalid file type. Only PDFs allowed."}), 400

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get('query')
    session_id = data.get('session_id')
    raw_history = data.get('history', []) # Get history from frontend

    if not user_query or not session_id:
        return jsonify({"error": "Missing query or session_id"}), 400

    # Format history into a readable string for the AI
    formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in raw_history])

    try:
        response = get_answer(user_query, session_id, formatted_history)
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Server starting on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)