from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path: str):
    """Loads a PDF and returns document objects."""
    try:
        loader = PyPDFLoader(file_path)
        return loader.load()
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []