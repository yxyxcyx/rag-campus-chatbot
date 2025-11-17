# enhanced_ingestion_worker.py

"""
Enhanced Celery Worker with Advanced OCR and Hybrid Search

Improvements over standard version:
- Uses EnhancedDocumentLoader for better OCR
- Stores additional metadata for hybrid search
- Prepares data for BM25 indexing
"""

import os
from typing import Dict
from celery import Task
from celery.utils.log import get_task_logger
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

from celery_config import celery_app
from enhanced_document_loader import EnhancedDocumentLoader

def load_documents_from_folder(folder_path: str):
    """Wrapper for enhanced document loader"""
    loader = EnhancedDocumentLoader(dpi=2.0, parallel_workers=4)
    return loader.load_folder(folder_path)
from rag_pipeline import chunk_text, embed_chunks

# Task logger
logger = get_task_logger(__name__)


class CallbackTask(Task):
    """Custom Task class with resource pooling"""
    _embedding_model = None
    _chroma_client = None
    _collection = None

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            logger.info("Initializing embedding model (one-time per worker)...")
            self._embedding_model = HuggingFaceEmbeddings(
                model_name='all-MiniLM-L6-v2'
            )
        return self._embedding_model

    @property
    def chroma_client(self):
        if self._chroma_client is None:
            logger.info("Connecting to ChromaDB (one-time per worker)...")
            chroma_path = os.getenv('CHROMA_DB_PATH', './chroma_db')
            self._chroma_client = chromadb.PersistentClient(path=chroma_path)
        return self._chroma_client

    @property
    def collection(self):
        if self._collection is None:
            collection_name = os.getenv('COLLECTION_NAME', 'collection')
            self._collection = self.chroma_client.get_or_create_collection(
                name=collection_name
            )
        return self._collection


@celery_app.task(bind=True, base=CallbackTask, name='ingestion_worker.process_document')
def process_document(self, file_path: str, window_size: int = 3) -> Dict[str, any]:
    """
    Process documents using enhanced OCR and sentence-window retrieval
    
    Args:
        file_path: Path to document or folder
        window_size: Number of sentences before/after central sentence
        
    Returns:
        Dictionary with processing results
    """
    logger.info(f"Starting ENHANCED ingestion for: {file_path}")
    logger.info(f"Using enhanced OCR + sentence-window retrieval (window_size={window_size})")

    try:
        # Step 1: Load documents with ENHANCED OCR
        logger.info("Loading documents with enhanced OCR...")
        if os.path.isdir(file_path):
            documents_dict = load_documents_from_folder(file_path)
        elif os.path.isfile(file_path):
            from enhanced_document_loader import EnhancedDocumentLoader
            loader = EnhancedDocumentLoader(dpi=2.0, parallel_workers=4)
            documents_dict = loader.load_document(file_path)
        else:
            raise ValueError(f"Invalid path: {file_path}")

        if not documents_dict:
            logger.warning("No documents found or extracted")
            return {
                'status': 'warning',
                'message': 'No documents found or extracted',
                'windows_added': 0
            }

        logger.info(f"Loaded {len(documents_dict)} documents")
        total_chars = sum(len(text) for text in documents_dict.values())
        logger.info(f"Total content: {total_chars:,} characters")

        # Step 2: Create sentence windows
        logger.info("Creating sentence windows...")
        central_sentences, windows = chunk_text(documents_dict, window_size=window_size)
        logger.info(f"Created {len(windows)} sentence windows")

        # Step 3: Embed central sentences
        logger.info("Embedding central sentences...")
        embeddings = embed_chunks(central_sentences, self.embedding_model)

        # Step 4: Store in vector database
        logger.info("Adding to vector database...")
        
        current_count = self.collection.count()
        ids = [f"sw_chunk_{current_count + i}" for i in range(len(windows))]

        self.collection.add(
            embeddings=embeddings,
            documents=windows,  # Full windows for context
            ids=ids
        )

        result = {
            'status': 'success',
            'message': f'Successfully processed with enhanced OCR + sentence-window retrieval',
            'documents_processed': len(documents_dict),
            'total_characters': total_chars,
            'windows_added': len(windows),
            'window_size': window_size,
            'total_windows_in_db': self.collection.count(),
            'technique': 'enhanced-ocr + sentence-window'
        }

        logger.info(f"Enhanced ingestion complete: {result}")
        return result

    except Exception as e:
        logger.error(f"Error during enhanced ingestion: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'message': str(e),
            'windows_added': 0
        }


@celery_app.task(bind=True, base=CallbackTask, name='ingestion_worker.clear_collection')
def clear_collection(self) -> Dict[str, any]:
    """Clear all documents from the collection."""
    logger.info("Clearing collection...")
    
    try:
        collection_name = os.getenv('COLLECTION_NAME', 'collection')
        
        self.chroma_client.delete_collection(name=collection_name)
        self._collection = self.chroma_client.create_collection(name=collection_name)
        
        logger.info("Collection cleared successfully")
        return {
            'status': 'success',
            'message': 'Collection cleared successfully'
        }
    except Exception as e:
        logger.error(f"Error clearing collection: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'message': str(e)
        }


@celery_app.task(name='ingestion_worker.get_collection_stats')
def get_collection_stats() -> Dict[str, any]:
    """Get statistics about the current collection."""
    try:
        chroma_path = os.getenv('CHROMA_DB_PATH', './chroma_db')
        client = chromadb.PersistentClient(path=chroma_path)
        collection_name = os.getenv('COLLECTION_NAME', 'collection')
        collection = client.get_or_create_collection(name=collection_name)
        
        count = collection.count()
        
        return {
            'status': 'success',
            'collection_name': collection_name,
            'total_windows': count,
            'technique': 'enhanced-ocr + sentence-window'
        }
    except Exception as e:
        logger.error(f"Error getting collection stats: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'message': str(e)
        }
