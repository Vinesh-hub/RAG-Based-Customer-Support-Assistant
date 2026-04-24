import os
from src.loader import load_pdf
from src.chunker import chunk_documents
from src.retriever import load_or_create_vectorstore, retrieve_chunks
from src.llm import get_llm

def build_rag_pipeline(file_path: str, db_path="./chroma_db"):
    """
    Optimized: Only indexes the PDF if the database folder doesn't exist.
    """
    if os.path.exists(db_path) and os.listdir(db_path):
        # Load existing database (Fast)
        return load_or_create_vectorstore(path=db_path)
    else:
        # Create new database (Slow, runs once)
        documents = load_pdf(file_path)
        chunks = chunk_documents(documents)
        vectorstore = load_or_create_vectorstore(chunks, path=db_path)
        return vectorstore

def generate_answer(vectorstore, query):
    docs = retrieve_chunks(vectorstore, query)
    context = "\n".join([doc.page_content for doc in docs])
    
    llm = get_llm("llama-3.1-8b-instant")
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    return llm.invoke(prompt).content