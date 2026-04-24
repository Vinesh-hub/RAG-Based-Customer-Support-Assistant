RAG-Based Customer Support Assistant

1. Project Overview

This project is a Retrieval-Augmented Generation (RAG) system designed to act as a Customer Support Assistant[cite: 4, 6]. [cite_start]It utilizes a graph-based workflow to process user queries, retrieve relevant information from a PDF knowledge base, and provide context-aware answers[cite: 7, 8, 9].

Key Features:

Modular Architecture: Separated concerns for loading, chunking, embedding, and retrieval[cite: 60, 105].

Graph Orchestration: Uses LangGraph to manage state and routing logic[cite: 17, 74].

Intent-Based Routing: Automatically classifies queries into 'greeting', 'support', or 'escalate'[cite: 19, 117].

Human-in-the-Loop (HITL): Escalates complex or out-of-scope queries to human agents[cite: 11, 21, 119].

Groq Integration: High-speed inference using Llama 3 models on Groq[cite: 50].

2. System Architecture

The system follows a High-Level Design (HLD) consisting of:

Document Ingestion: PDF documents are loaded and split into semantic chunks[cite: 29, 35, 36].

Vector Store: Chunks are embedded and stored in a local ChromaDB instance[cite: 15, 30, 38].

Graph Workflow:

Categorizer: Detects user intent[cite: 117].

RAG Agent: Retrieves context and generates answers[cite: 31, 39, 40].

Escalator: Handles low-confidence responses[cite: 43, 78, 81].

3. Tech Stack

LLM: Groq (Llama 3.3 70B & Llama 3.1 8B)[cite: 50].

Embeddings: HuggingFace (Local)[cite: 37, 108].

Orchestration: LangGraph[cite: 49, 113].

Vector Database: ChromaDB[cite: 48].

Environment: Python, Dotenv.

4. Installation & Setup

Prerequisites

Python 3.10 or higher.

A Groq API Key.

Steps

Clone the Repository:

git clone <your-repo-url>
cd RAG-Based-Customer-Support-Assistant


Create a Virtual Environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:

pip install -r requirements.txt


Configure Environment Variables:

Create a .env file in the root directory.

Add your Groq key: GROQ_API_KEY=your_key_here.

Add Knowledge Base:

Place your PDF file (e.g., DA_Notes.pdf) inside the data/ folder[cite: 7].

Update the pdf_path in main.py if necessary.

5. Usage

Run the assistant using:

python main.py


First Run: The system will index the PDF (this may take a few moments)[cite: 15].

Subsequent Runs: The system will load the local chroma_db for near-instant responses.

6. Project Structure

├── data/               # Source PDF files
├── src/                # Modular source code
│   ├── chunker.py      # Text splitting logic
│   ├── embedder.py     # Local embedding configuration
│   ├── llm.py          # Groq LLM setup
│   ├── loader.py       # PDF loading utilities
│   ├── rag_pipeline.py # RAG orchestration logic
│   └── retriever.py    # Vector store interactions
├── main.py             # LangGraph workflow and entry point
├── requirements.txt    # Project dependencies
└── .gitignore          # Git exclusion rules


7. Future Enhancements

Multi-document support[cite: 128].

Conversation memory integration[cite: 130].

Web-based User Interface[cite: 29].
