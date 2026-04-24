import os
from langchain_groq import ChatGroq

def get_llm(model_name="llama-3.3-70b-versatile"):
    """Returns the Groq LLM instance."""
    api_key = os.getenv("GROQ_API_KEY")
    return ChatGroq(model=model_name, groq_api_key=api_key)