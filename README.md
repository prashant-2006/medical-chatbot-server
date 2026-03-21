# 🩺 AI Medical Assistant - RAG Backend

This is the Python/Flask backend for an AI-powered Medical Assistant application. It uses **Retrieval-Augmented Generation (RAG)** to allow users to upload PDF documents and ask questions about them. The AI intelligently retrieves relevant information from the document to provide accurate, cited answers.

🌐 **Live Demo:** https://medical-chatbot-frontend.vercel.app/
💻 **Frontend Repository:** https://github.com/prashant-2006/medical-chatbot-frontend

---

## ✨ Features

* **PDF Processing:** Extracts text from uploaded PDF files.
* **Smart Chunking:** Splits large documents into manageable pieces while retaining context.
* **Vector Storage:** Converts text into embeddings and stores them securely in Pinecone.
* **Context-Aware AI Chat:** Uses Google's Gemini AI to answer questions based *strictly* on the uploaded document, including page number citations.
* **Session Management:** Keeps different users' files and chats isolated using unique session IDs.
* **Auto-Cleanup:** Automatically deletes temporary PDF uploads after 2 hours to save server space.

---

## 🛠️ Tech Stack

* **Framework:** Python, Flask
* **AI/LLM:** Google Gemini (`gemini-2.5-flash`) via LangChain
* **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **Vector Database:** Pinecone
* **Document Processing:** PyMuPDF (fitz)
* **Frontend:** Next.js (Separate repository)

---

## 🚀 How to Run Locally

If you want to run this backend on your own computer, follow these step-by-step instructions.

### 1. Prerequisites
Make sure you have Python installed on your computer. You will also need accounts with **Google AI Studio** (for the Gemini API key) and **Pinecone** (for the vector database).

### 2. Clone the Repository
```bash
git clone https://github.com/prashant-2006/medical-chatbot-server
cd medical-chatbot-server
```

### 3. Create a Virtual Environment
It is highly recommended to use a virtual environment so the libraries for this project don't conflict with other Python projects on your computer.

* **On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```
* **On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
Install all the required Python libraries needed to run the engine.
```bash
pip install -r requirements.txt
```

### 5. Set Up Environment Variables
Create a new file in the root folder named `.env` and add your secret API keys. **Never share this file or upload it to GitHub!**

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=your_pinecone_index_name_here
```

### 6. Start the Server
Run the Flask application.
```bash
python app.py
```
The server should now be running locally at `http://127.0.0.1:5000`.

---

## 📡 API Endpoints

This backend provides the following REST API endpoints for the frontend to communicate with:

* **`POST /upload`**: Accepts a `.pdf` file and a `session_id`. It processes the file and stores the embeddings in Pinecone.
* **`POST /chat`**: Accepts a JSON payload containing a `query`, `session_id`, and chat `history`. Returns the AI's answer.

---

## 🤝 Acknowledgments
Built with ❤️ using LangChain, Flask, and Google Gemini.
