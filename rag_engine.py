import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Load Environment Variables
load_dotenv()

# Global variables to store the chain (so we don't reload it every request)
rag_chain = None

def initialize_chain():
    global rag_chain
    print("🤖 Initializing RAG Chain...")

    # 2. Setup Embeddings (Matches ingest.py)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 3. Connect to Pinecone
    index_name = os.getenv("PINECONE_INDEX_NAME")
    vectorstore = PineconeVectorStore(
        index_name=index_name, 
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. Setup LLM (Using gemini-pro as it is most stable for free tier)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    # 5. Create Prompt
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 6. Define Helper to Format Docs
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 7. Build Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("✅ RAG Chain Ready!")

def get_answer(query):
    """
    This function takes a query string and returns the AI's answer.
    """
    if rag_chain is None:
        initialize_chain()
    
    try:
        response = rag_chain.invoke(query)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize once when this file is imported
initialize_chain()