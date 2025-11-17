# main.py

"""
FastAPI Application - Stateless API Server

This script serves a RESTful API to interact with the RAG pipeline.

ARCHITECTURE CHANGE:
This API server is now STATELESS and follows the read-only pattern.
It ONLY serves queries via the `/ask` endpoint.

Data ingestion is handled separately by Celery workers (see ingestion_worker.py).
To ingest documents, use trigger_ingestion.py script.

This separation follows production best practices:
- Read path (query): Handled by this API
- Write path (ingestion): Handled by async workers
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers.cross_encoder import CrossEncoder
import chromadb

from rag_pipeline import (
    retrieve_and_rerank,
    generate_response
)

# SECTION 1: INITIALIZATION (Read-only resources)
load_dotenv()
print(" Initializing RAG API (Stateless Mode)...")
print("  - Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
print("  - Loading Cross-Encoder re-ranking model...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("  - Connecting to Vector Database...")

# Get configuration from environment variables
chroma_db_path = os.getenv('CHROMA_DB_PATH', './chroma_db')
collection_name = os.getenv('COLLECTION_NAME', 'collection')

client = chromadb.PersistentClient(path=chroma_db_path)
collection = client.get_or_create_collection(name=collection_name)

# Log the current state (but don't modify it)
chunk_count = collection.count()
if chunk_count == 0:
    print("    WARNING: Database is empty!")
    print("  - Run 'python trigger_ingestion.py data/' to populate the database")
    print("  - Or ensure the Celery worker is running and dispatch ingestion tasks")
else:
    print(f"   Database ready with {chunk_count} chunks")

print("   API server is ready (read-only mode)")


# SECTION 2: FASTAPI APPLICATION
app = FastAPI(
    title="RAG Chatbot API",
    description="An API for asking questions powered by a Retrieval-Augmented Generation pipeline.",
    version="2.0.0"
)

# Pydantic Models for Data Validation
class Question(BaseModel):
    query: str

class Answer(BaseModel):
    response: str

# API Endpoints
@app.get("/", summary="Check API Status")
def read_root():
    """A simple health check endpoint to confirm the API is running."""
    return {"status": "RAG Chatbot API is online and ready.", "version": "2.0.0"}

@app.post("/ask", response_model=Answer, summary="Ask a Question")
def ask_question(question: Question):
    """
    Receives a user query, runs it through the full RAG pipeline,
    and returns the generated answer.
    """
    user_query = question.query
    print(f"Received query: '{user_query}'")

    # 1. Retrieve and rerank
    print("  - Retrieving and re-ranking context...")
    retrieved_chunks = retrieve_and_rerank(user_query, embedding_model, collection, cross_encoder)
    retrieved_context = "\n\n".join(retrieved_chunks)
    
    # 2. Generate response
    print("  - Generating response...")
    final_answer = generate_response(retrieved_context, user_query)
    print("  - Response generated.")
    
    return {"response": final_answer}
