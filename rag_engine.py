import os
from operator import itemgetter
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Load Environment Variables
load_dotenv()

# Global initialization of models
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
index_name = os.getenv("PINECONE_INDEX_NAME")

def process_document(file_path, session_id):
    """Reads the uploaded PDF, chunks it, and stores it in Pinecone."""
    print(f"📄 Processing document for session: {session_id}")
    
    loader = PyMuPDFLoader(file_path)
    raw_docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(raw_docs)
    
    # Store in Pinecone using the session_id to keep it isolated
    PineconeVectorStore.from_documents(
        documents, 
        embeddings, 
        index_name=index_name,
        namespace=session_id 
    )
    print("✅ Document embedded and stored in Vector DB!")

def get_answer(query, session_id, chat_history):
    """Retrieves answers, handles memory, and manages citations."""
    vectorstore = PineconeVectorStore(
        index_name=index_name, 
        embedding=embeddings,
        namespace=session_id
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # NEW PROMPT: Handles memory, outside knowledge, and citations
    template = """You are a helpful AI Medical Assistant. 
    You have access to a specific uploaded document (Context) and the recent conversation history.

    Chat History (Last 5 messages):
    {chat_history}

    Uploaded Document Context:
    {context}

    User Question: {question}

    Instructions:
    1. First, try to answer the User Question using ONLY the "Uploaded Document Context". 
    2. If you find the answer in the Context, you MUST append the source page at the end of your answer, like: "[Source: Uploaded PDF, Page X]".
    3. If the answer is NOT in the Context, you MUST start your answer with: "This information is not from the uploaded document." and then provide the best answer you can from your general knowledge.
    4. Use the Chat History to understand follow-up questions (e.g., if the user says "Why?", look at the history to know what they mean).
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Inject Page Numbers into the context string
    def format_docs(docs):
        formatted_chunks = []
        for doc in docs:
            # PyMuPDF pages are 0-indexed, so we add 1 for human readability
            page_num = doc.metadata.get('page', 0) + 1 
            formatted_chunks.append(f"--- Page {page_num} ---\n{doc.page_content}")
        return "\n\n".join(formatted_chunks)

    # The itemgetter fix correctly routes the inputs so it doesn't crash
    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs, 
            "chat_history": itemgetter("chat_history"), 
            "question": itemgetter("question")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Pass the dictionary containing both the query and the formatted history
    return rag_chain.invoke({
        "question": query,
        "chat_history": chat_history
    })