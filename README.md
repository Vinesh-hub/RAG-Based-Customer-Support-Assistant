RAG-Based Customer Support Assistant

1. Project Overview

This project is a Retrieval-Augmented Generation (RAG) system designed to act as a Customer Support Assistant. It utilizes a graph-based workflow to process user queries, retrieve relevant information from a PDF knowledge base, and provide context-aware answers.

Key Features:

Modular Architecture: Separated concerns for loading, chunking, embedding, and retrieval.

Graph Orchestration: Uses LangGraph to manage state and routing logic.

Intent-Based Routing: Automatically classifies queries into 'greeting', 'support', or 'escalate'.

Human-in-the-Loop (HITL): Escalates complex or out-of-scope queries to human agents.

Groq Integration: High-speed inference using Llama 3 models on Groq.

2. System Architecture

The system follows a High-Level Design (HLD) consisting of:

Document Ingestion: PDF documents are loaded and split into semantic chunks.

Vector Store: Chunks are embedded and stored in a local ChromaDB instance.

Graph Workflow:

Categorizer: Detects user intent].

RAG Agent: Retrieves context and generates answers.

Escalator: Handles low-confidence responses.

3. Tech Stack

LLM: Groq (Llama 3.3 70B & Llama 3.1 8B).

Embeddings: HuggingFace (Local).

Orchestration: LangGraph.

Vector Database: ChromaDB.

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

Place your PDF file (e.g., DA_Notes.pdf) inside the data/ folder.

Update the pdf_path in main.py if necessary.

5. Usage

Run the assistant using:

python main.py


First Run: The system will index the PDF (this may take a few moments).

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

Multi-document support.

Conversation memory integration].

Web-based User Interface.
