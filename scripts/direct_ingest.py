#!/usr/bin/env python3
# direct_ingest.py

"""
Direct Document Ingestion (No Celery Required)

This script directly processes and ingests documents into the vector database
without requiring Celery/Redis infrastructure.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import get_settings
from logging_config import setup_logging, get_logger
from enhanced_document_loader import EnhancedDocumentLoader
from sentence_window_retrieval import chunk_text_with_sentence_windows
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb


def main():
    # Setup logging
    logger = setup_logging(level="INFO", json_output=False, app_name="direct-ingest")
    
    settings = get_settings()
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python direct_ingest.py <path_to_file_or_folder>")
        print("       python direct_ingest.py --stats")
        sys.exit(1)
    
    path = sys.argv[1]
    
    # Initialize ChromaDB
    logger.info("Connecting to ChromaDB", path=settings.chroma_db_path)
    client = chromadb.PersistentClient(path=settings.chroma_db_path)
    collection = client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Stats mode
    if path == "--stats":
        count = collection.count()
        logger.info("Collection statistics", document_count=count)
        print(f"\nCollection: {settings.collection_name}")
        print(f"Documents: {count}")
        return
    
    # Initialize embedding model (use model from config)
    logger.info("Loading embedding model", model=settings.embedding_model_name)
    embedding_model = HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Load documents
    logger.info("Loading documents", path=path)
    loader = EnhancedDocumentLoader(dpi=2.0, parallel_workers=2)
    
    if os.path.isfile(path):
        documents_dict = loader.load_document(path)
    elif os.path.isdir(path):
        documents_dict = {}
        for file in Path(path).glob("**/*"):
            if file.suffix.lower() in ['.pdf', '.docx', '.txt']:
                try:
                    doc = loader.load_document(str(file))
                    documents_dict.update(doc)
                except Exception as e:
                    logger.warning(f"Failed to load {file}: {e}")
    else:
        logger.error("Invalid path", path=path)
        sys.exit(1)
    
    if not documents_dict:
        logger.error("No documents found")
        sys.exit(1)
    
    logger.info("Documents loaded", count=len(documents_dict))
    total_chars = sum(len(text) for text in documents_dict.values())
    logger.info("Total content", characters=total_chars)
    
    # Create sentence windows
    logger.info("Creating sentence windows")
    central_sentences, windows = chunk_text_with_sentence_windows(
        documents_dict, 
        window_size=3
    )
    
    logger.info("Sentence windows created", count=len(windows))
    
    # Generate embeddings
    logger.info("Generating embeddings (this may take a while)")
    embeddings = embedding_model.embed_documents(central_sentences)
    
    logger.info("Embeddings generated", count=len(embeddings))
    
    # Store in ChromaDB
    logger.info("Storing in vector database")
    
    current_count = collection.count()
    ids = [f"sw_chunk_{current_count + i}" for i in range(len(windows))]
    
    # Add in batches
    batch_size = 100
    for i in range(0, len(windows), batch_size):
        end_idx = min(i + batch_size, len(windows))
        collection.add(
            embeddings=embeddings[i:end_idx],
            documents=windows[i:end_idx],
            ids=ids[i:end_idx]
        )
        logger.info(f"Added batch {i//batch_size + 1}/{(len(windows)-1)//batch_size + 1}")
    
    final_count = collection.count()
    logger.info("Ingestion complete", 
                documents_processed=len(documents_dict),
                windows_added=len(windows),
                total_in_db=final_count)
    
    print(f"\n{'='*60}")
    print("INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"Documents processed: {len(documents_dict)}")
    print(f"Sentence windows created: {len(windows)}")
    print(f"Total chunks in database: {final_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
