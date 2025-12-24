#!/usr/bin/env python3
# reingest_with_tables.py

"""
Re-ingestion Script with Table-Aware Processing

This script clears the existing vector database and re-ingests all documents
using the new table-aware extraction that properly handles:
- Fee structures and tabular data
- Preserves row-column relationships
- Creates semantic chunks for structured data

Usage:
    python scripts/reingest_with_tables.py [data_folder]
    
    If no folder is specified, defaults to 'data/'

This script can be run directly (without Celery) for quick testing.
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from typing import Dict, List
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

from config import get_settings
from logging_config import setup_logging, get_logger
from table_aware_loader import TableAwareLoader
from sentence_window_retrieval import chunk_text_with_sentence_windows
from enhanced_rag_engine import DocumentVersionManager


def reingest_with_tables(
    data_folder: str,
    clear_existing: bool = True,
    use_versioning: bool = True
) -> Dict:
    """
    Re-ingest documents with table-aware processing.
    
    Args:
        data_folder: Path to folder containing documents
        clear_existing: Whether to clear existing data first
        use_versioning: Whether to track document versions
        
    Returns:
        Dictionary with ingestion results
    """
    settings = get_settings()
    logger = get_logger(__name__)
    
    logger.info(f"Starting table-aware re-ingestion from: {data_folder}")
    
    # Initialize components
    logger.info("Initializing embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)
    
    logger.info("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=settings.chroma_db_path)
    
    # Clear existing collection if requested
    if clear_existing:
        logger.info("Clearing existing collection...")
        try:
            client.delete_collection(name=settings.collection_name)
            logger.info("Collection cleared")
        except Exception as e:
            logger.info(f"No existing collection to clear: {e}")
    
    collection = client.get_or_create_collection(name=settings.collection_name)
    
    # Initialize version manager
    version_manager = DocumentVersionManager() if use_versioning else None
    
    # Load documents with table-aware extraction
    logger.info("Loading documents with table-aware extraction...")
    loader = TableAwareLoader(dpi=2.0)
    documents_dict, table_chunks = loader.load_folder_with_tables(data_folder)
    
    if not documents_dict and not table_chunks:
        logger.warning("No documents found!")
        return {
            'status': 'warning',
            'message': 'No documents found',
            'chunks_added': 0
        }
    
    logger.info(f"Loaded {len(documents_dict)} documents with {len(table_chunks)} table chunks")
    
    # Create sentence windows from regular text
    logger.info("Creating sentence windows from text content...")
    central_sentences, windows = chunk_text_with_sentence_windows(documents_dict, window_size=3)
    logger.info(f"Created {len(windows)} sentence windows")
    
    # Prepare table chunks for embedding
    table_texts = [chunk['text'] for chunk in table_chunks]
    logger.info(f"Prepared {len(table_texts)} table chunks for embedding")
    
    # Combine all content
    all_chunks = windows + table_texts
    all_sentences = central_sentences + table_texts  # Table chunks are self-contained
    
    if not all_chunks:
        logger.warning("No chunks to embed!")
        return {
            'status': 'warning',
            'message': 'No content extracted from documents',
            'chunks_added': 0
        }
    
    # Embed all chunks
    logger.info(f"Embedding {len(all_sentences)} chunks...")
    embeddings = embedding_model.embed_documents(all_sentences)
    logger.info("Embedding complete")
    
    # Prepare IDs and metadata
    ids = [f"table_aware_chunk_{i}" for i in range(len(all_chunks))]
    
    metadatas = []
    for i, chunk in enumerate(all_chunks):
        if i < len(windows):
            metadatas.append({
                'type': 'sentence_window',
                'index': i,
                'technique': 'table-aware'
            })
        else:
            table_idx = i - len(windows)
            meta = {
                'type': 'table_row',
                'index': table_idx,
                'technique': 'table-aware'
            }
            if table_idx < len(table_chunks):
                chunk_meta = table_chunks[table_idx].get('metadata', {})
                meta['source'] = chunk_meta.get('source', '')
                meta['page'] = chunk_meta.get('page', 0)
            metadatas.append(meta)
    
    # Add to collection
    logger.info("Adding to vector database...")
    collection.add(
        embeddings=embeddings,
        documents=all_chunks,
        ids=ids,
        metadatas=metadatas
    )
    
    # Update version records
    if version_manager:
        for filename in os.listdir(data_folder):
            filepath = os.path.join(data_folder, filename)
            if os.path.isfile(filepath):
                version_manager.update_version(filepath)
    
    final_count = collection.count()
    
    result = {
        'status': 'success',
        'message': 'Table-aware re-ingestion complete',
        'documents_processed': len(documents_dict),
        'sentence_windows': len(windows),
        'table_chunks': len(table_texts),
        'total_chunks_added': len(all_chunks),
        'total_in_database': final_count,
        'technique': 'table-aware + sentence-window'
    }
    
    logger.info(f"Re-ingestion complete: {result}")
    
    return result


def preview_table_extraction(data_folder: str, max_chunks: int = 5):
    """
    Preview what the table extraction would produce without ingesting.
    
    Useful for debugging and verification.
    """
    logger = get_logger(__name__)
    
    logger.info(f"Previewing table extraction from: {data_folder}")
    
    loader = TableAwareLoader(dpi=2.0)
    documents_dict, table_chunks = loader.load_folder_with_tables(data_folder)
    
    print("\n" + "="*60)
    print("TABLE EXTRACTION PREVIEW")
    print("="*60)
    
    print(f"\nDocuments loaded: {len(documents_dict)}")
    print(f"Table chunks extracted: {len(table_chunks)}")
    
    if table_chunks:
        print(f"\nFirst {min(max_chunks, len(table_chunks))} table chunks:")
        print("-"*60)
        
        for i, chunk in enumerate(table_chunks[:max_chunks]):
            print(f"\n[Chunk {i+1}]")
            print(chunk['text'])
            print("-"*40)
    else:
        print("\nNo table chunks extracted.")
        print("This might mean the PDFs don't contain detectable tables,")
        print("or the table structure wasn't recognized.")
    
    return documents_dict, table_chunks


if __name__ == "__main__":
    # Setup logging
    setup_logging(level="INFO", json_output=False, app_name="reingest")
    
    # Get data folder from arguments or use default
    if len(sys.argv) > 1:
        if sys.argv[1] == "--preview":
            # Preview mode
            data_folder = sys.argv[2] if len(sys.argv) > 2 else "data/"
            preview_table_extraction(data_folder)
        else:
            data_folder = sys.argv[1]
            result = reingest_with_tables(data_folder)
            print(f"\nResult: {result}")
    else:
        # Default: re-ingest from data/
        data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
        if not os.path.exists(data_folder):
            data_folder = "data/"
        
        print(f"Re-ingesting from: {data_folder}")
        print("Use --preview to see what would be extracted without ingesting")
        print()
        
        result = reingest_with_tables(data_folder)
        print(f"\nResult: {result}")
