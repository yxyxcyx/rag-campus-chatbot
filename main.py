# main.py

"""
FastAPI

This script serves a RESTful API to interact with the RAG pipeline.
It handles:
-   Initialization of the embedding model and vector database.
-   On-demand population of the database if it's empty, using the
    latest data processing pipeline from rag_pipeline.py.
-   An `/ask` endpoint to receive user queries and return model-generated answers.
"""

import os
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers.cross_encoder import CrossEncoder
import chromadb

from rag_pipeline import (
    load_documents_from_folder,
    chunk_text,
    embed_chunks,
    retrieve_and_rerank,
    generate_response
)

# SECTION 1: INITIALIZATION AND DATABASE SETUP
load_dotenv()
print("Initializing RAG API...")
print("  - Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
print("  - Loading Cross-Encoder re-ranking model...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("  - Connecting to Vector Database...")
client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "xmum_handbook"
collection = client.get_or_create_collection(name=collection_name)

# Check if the database is empty. If so, run the full data ingestion pipeline
if collection.count() == 0:
    print("  - Database is empty. Running one-time data ingestion pipeline...")
    
    documents_dict = load_documents_from_folder("data")
    
    if documents_dict:

        text_chunks_with_metadata = chunk_text(documents_dict)
        
        # Embed the chunks
        vector_embeddings = embed_chunks(text_chunks_with_metadata, embedding_model)
        ids = [f"chunk_{i}" for i in range(len(text_chunks_with_metadata))]
        
        collection.add(
            embeddings=vector_embeddings,
            documents=text_chunks_with_metadata,
            ids=ids
        )
        print("  - PIPELINE COMPLETE. Database is now populated.")
    else:
        print("  - WARNING: No text could be extracted from the 'data' folder.")
else:
    print(f"  - Database already populated with {collection.count()} chunks. API is ready.")


# SECTION 2: FASTAPI APPLICATION
app = FastAPI(
    title="XMUM RAG Chatbot API",
    description="An API for asking questions about the XMUM student handbook, powered by a Retrieval-Augmented Generation pipeline.",
    version="1.0.0"
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
    return {"status": "XMUM RAG Chatbot API is online and ready."}

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