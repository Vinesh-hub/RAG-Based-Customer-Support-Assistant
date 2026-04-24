import os
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    """
    Returns an open-source embedding model.
    This replaces Google embeddings to use a local, free, and Groq-compatible workflow.
    """
    # Using a popular, high-performance small model
    model_name = "sentence-transformers/all-mpnet-base-v2"
    return HuggingFaceEmbeddings(model_name=model_name)