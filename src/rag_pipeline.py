import hashlib
import json
import shutil
from pathlib import Path

from src.loader import load_pdf
from src.chunker import chunk_documents
from src.retriever import load_or_create_vectorstore, retrieve_chunks
from src.llm import get_llm


def _source_signature(file_path: str) -> str:
    source = Path(file_path)
    stat = source.stat()
    payload = f"{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_rag_pipeline(file_path: str, db_path="./chroma_db"):
    """
    Load the vector store when it matches the source PDF, otherwise rebuild it.
    """
    source_signature = _source_signature(file_path)
    db = Path(db_path)
    metadata_path = db / "source.json"

    if db.is_dir() and any(db.iterdir()) and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("source_signature") == source_signature:
            return load_or_create_vectorstore(path=db_path)

    if db.exists():
        shutil.rmtree(db)

    documents = load_pdf(file_path)
    if not documents:
        raise ValueError(f"No documents were loaded from {file_path}.")

    chunks = chunk_documents(documents)
    vectorstore = load_or_create_vectorstore(chunks, path=db_path)
    metadata_path.write_text(
        json.dumps({"source_signature": source_signature}),
        encoding="utf-8",
    )
    return vectorstore

def generate_answer(vectorstore, query):
    docs = retrieve_chunks(vectorstore, query)
    context = "\n".join([doc.page_content for doc in docs])
    
    llm = get_llm("openai/gpt-oss-20b")
    prompt = (
        "You are a customer-support assistant. Answer the question using only "
        "the provided context. If the context does not contain the answer, "
        "respond with exactly ESCALATE so the request can be sent to a human. "
        "Do not invent or infer unsupported details.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    return llm.invoke(prompt).content