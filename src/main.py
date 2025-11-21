# main.py

"""
FastAPI Application - Stateless API Server

This script serves a RESTful API to interact with the RAG pipeline.

ARCHITECTURE CHANGE:
This API server is now STATELESS and follows the read-only pattern.
It ONLY serves queries via the `/ask` endpoint.

Data ingestion is handled separately by Celery workers (see ingestion_worker.py).
To ingest documents, use trigger_ingestion.py script.

This separation follows best practices:
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
from rank_bm25 import BM25Okapi
from typing import Optional, List

from rag_pipeline import (
    retrieve_and_rerank,
    retrieve_and_rerank_hybrid,
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

# Hybrid search configuration (optional)
ENABLE_HYBRID_SEARCH = os.getenv('ENABLE_HYBRID_SEARCH', 'false').lower() == 'true'
BM25_WEIGHT = float(os.getenv('BM25_WEIGHT', '0.3'))
VECTOR_WEIGHT = float(os.getenv('VECTOR_WEIGHT', '0.7'))
USE_DIVERSITY_FILTER = os.getenv('USE_DIVERSITY_FILTER', 'true').lower() == 'true'
DIVERSITY_THRESHOLD = float(os.getenv('DIVERSITY_THRESHOLD', '0.85'))

client = chromadb.PersistentClient(path=chroma_db_path)
collection = client.get_or_create_collection(name=collection_name)

# Initialize BM25 index if hybrid search is enabled
bm25_index: Optional[BM25Okapi] = None
all_documents: Optional[List[str]] = None

# Log the current state (but don't modify it)
chunk_count = collection.count()
if chunk_count == 0:
    print("    WARNING: Database is empty!")
    print("  - Run 'python trigger_ingestion.py data/' to populate the database")
    print("  - Or ensure the Celery worker is running and dispatch ingestion tasks")
else:
    print(f"   Database ready with {chunk_count} chunks")
    
    # Initialize BM25 if hybrid search is enabled and database has content
    if ENABLE_HYBRID_SEARCH and chunk_count > 0:
        print("  - Initializing BM25 index for hybrid search...")
        try:
            # Fetch all documents from ChromaDB for BM25
            all_results = collection.get(limit=chunk_count)
            all_documents = all_results.get("documents", [])
            
            if all_documents:
                # Tokenize documents for BM25
                tokenized_docs = [doc.lower().split() for doc in all_documents]
                bm25_index = BM25Okapi(tokenized_docs)
                print(f"   BM25 index initialized with {len(all_documents)} documents")
                print(f"   Hybrid search enabled (BM25 weight: {BM25_WEIGHT}, Vector weight: {VECTOR_WEIGHT})")
            else:
                print("    WARNING: Could not fetch documents for BM25 index")
                ENABLE_HYBRID_SEARCH = False
        except Exception as e:
            print(f"    ERROR initializing BM25: {e}")
            print("    Falling back to vector-only search")
            ENABLE_HYBRID_SEARCH = False
            bm25_index = None
            all_documents = None
    
    if USE_DIVERSITY_FILTER:
        print(f"   Diversity filter enabled (threshold: {DIVERSITY_THRESHOLD})")

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
    
    Uses hybrid search (BM25 + Vector) if enabled, otherwise uses enhanced vector search.
    Both methods include diversity filtering to reduce redundancy.
    """
    user_query = question.query
    print(f"Received query: '{user_query}'")

    # 1. Retrieve and rerank with appropriate method
    if ENABLE_HYBRID_SEARCH and bm25_index is not None:
        print("  - Using hybrid retrieval (BM25 + Vector) with diversity filtering...")
        retrieved_chunks = retrieve_and_rerank_hybrid(
            user_query, 
            embedding_model, 
            collection, 
            cross_encoder,
            bm25_index=bm25_index,
            documents=all_documents,
            n_initial=30,
            n_final=5,
            bm25_weight=BM25_WEIGHT,
            vector_weight=VECTOR_WEIGHT,
            use_diversity_filter=USE_DIVERSITY_FILTER,
            diversity_threshold=DIVERSITY_THRESHOLD
        )
    else:
        print("  - Using enhanced vector retrieval with diversity filtering...")
        retrieved_chunks = retrieve_and_rerank(
            user_query, 
            embedding_model, 
            collection, 
            cross_encoder,
            n_initial=20,
            n_final=5,
            use_diversity_filter=USE_DIVERSITY_FILTER,
            diversity_threshold=DIVERSITY_THRESHOLD
        )
    
    retrieved_context = "\n\n".join(retrieved_chunks)
    
    # 2. Generate response
    print("  - Generating response...")
    final_answer = generate_response(retrieved_context, user_query)
    print("  - Response generated.")
    
    return {"response": final_answer}
