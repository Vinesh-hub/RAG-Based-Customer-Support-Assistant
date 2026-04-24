import os
from langchain_community.vectorstores import Chroma
from src.embedder import get_embedding_model

def load_or_create_vectorstore(chunks=None, path="./chroma_db"):
    """
    Handles the Vector Database lifecycle.
    1. If chunks are provided, it creates a new database and saves it to 'path'.
    2. If no chunks are provided, it tries to load the existing database from 'path'.
    """
    embeddings = get_embedding_model()
    
    if chunks:
        # Create and persist the database
        print(f"Creating new vectorstore at {path}...")
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=path
        )
        return vectorstore
    else:
        # Load from disk
        if os.path.exists(path):
            print(f"Loading existing vectorstore from {path}...")
            return Chroma(
                persist_directory=path, 
                embedding_function=embeddings
            )
        else:
            raise FileNotFoundError(f"No vectorstore found at {path}. Please provide chunks to create one.")

def retrieve_chunks(vectorstore, query, k=3):
    """
    Finds the top K most relevant text chunks based on the query.
    """
    return vectorstore.similarity_search(query, k=k)