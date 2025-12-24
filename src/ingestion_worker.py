# enhanced_ingestion_worker.py

"""
Enhanced Celery Worker with Advanced OCR and Hybrid Search

Improvements over standard version:
- Uses EnhancedDocumentLoader for better OCR
- Stores additional metadata for hybrid search
- Prepares data for BM25 indexing
- Uses centralized configuration
"""

import os
from typing import Dict
from celery import Task
from celery.utils.log import get_task_logger
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

from config import get_settings
from celery_config import celery_app
from enhanced_document_loader import EnhancedDocumentLoader
from table_aware_loader import TableAwareLoader
from enhanced_rag_engine import DocumentVersionManager

# Load validated configuration
settings = get_settings()

def load_documents_from_folder(folder_path: str):
    """Wrapper for enhanced document loader"""
    loader = EnhancedDocumentLoader(dpi=2.0, parallel_workers=4)
    return loader.load_folder(folder_path)


def load_documents_with_tables(folder_path: str):
    """Load documents with table-aware extraction for better fee/structured data handling"""
    loader = TableAwareLoader(dpi=2.0)
    return loader.load_folder_with_tables(folder_path)
from rag_pipeline import chunk_text, embed_chunks

# Task logger (Celery provides its own structured logger)
logger = get_task_logger(__name__)


class CallbackTask(Task):
    """Custom Task class with resource pooling"""
    _embedding_model = None
    _chroma_client = None
    _collection = None

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            logger.info(
                "Initializing embedding model (one-time per worker)",
                extra={"model": settings.embedding_model_name}
            )
            self._embedding_model = HuggingFaceEmbeddings(
                model_name=settings.embedding_model_name
            )
        return self._embedding_model

    @property
    def chroma_client(self):
        if self._chroma_client is None:
            logger.info(
                "Connecting to ChromaDB (one-time per worker)",
                extra={"path": settings.chroma_db_path}
            )
            self._chroma_client = chromadb.PersistentClient(path=settings.chroma_db_path)
        return self._chroma_client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.chroma_client.get_or_create_collection(
                name=settings.collection_name
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
    logger.info("Clearing collection", extra={"collection": settings.collection_name})
    
    try:
        self.chroma_client.delete_collection(name=settings.collection_name)
        self._collection = self.chroma_client.create_collection(name=settings.collection_name)
        
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
        client = chromadb.PersistentClient(path=settings.chroma_db_path)
        collection = client.get_or_create_collection(name=settings.collection_name)
        
        count = collection.count()
        
        return {
            'status': 'success',
            'collection_name': settings.collection_name,
            'total_windows': count,
            'technique': 'enhanced-ocr + sentence-window'
        }
    except Exception as e:
        logger.error(f"Error getting collection stats: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'message': str(e)
        }


@celery_app.task(bind=True, base=CallbackTask, name='ingestion_worker.process_with_tables')
def process_with_tables(self, file_path: str, window_size: int = 3, use_versioning: bool = True) -> Dict[str, any]:
    """
    Process documents with TABLE-AWARE extraction for structured data like fee tables.
    
    This task combines:
    1. Table-aware PDF extraction (preserves fee structure relationships)
    2. Standard sentence-window chunking for regular text
    3. Optional document versioning for incremental updates
    
    Args:
        file_path: Path to document or folder
        window_size: Number of sentences before/after central sentence
        use_versioning: Whether to use document versioning (only ingest changed files)
        
    Returns:
        Dictionary with processing results
    """
    logger.info(f"Starting TABLE-AWARE ingestion for: {file_path}")
    
    try:
        version_manager = DocumentVersionManager() if use_versioning else None
        
        # Step 1: Check for changes if versioning is enabled
        files_to_process = None
        if use_versioning and os.path.isdir(file_path):
            changes = version_manager.get_changed_documents(file_path)
            if not changes:
                logger.info("No document changes detected, skipping ingestion")
                return {
                    'status': 'success',
                    'message': 'No changes detected - database is up to date',
                    'windows_added': 0,
                    'documents_processed': 0
                }
            files_to_process = [f for f, change_type in changes.items() if change_type != 'deleted']
            logger.info(f"Found {len(changes)} changed documents: {changes}")
        
        # Step 2: Load documents with TABLE-AWARE extraction
        logger.info("Loading documents with table-aware extraction...")
        
        if os.path.isdir(file_path):
            documents_dict, table_chunks = load_documents_with_tables(file_path)
        elif os.path.isfile(file_path):
            loader = TableAwareLoader(dpi=2.0)
            text, table_chunks = loader.load_pdf_with_tables(file_path)
            documents_dict = {os.path.basename(file_path): text}
        else:
            raise ValueError(f"Invalid path: {file_path}")
        
        if not documents_dict and not table_chunks:
            logger.warning("No documents found or extracted")
            return {
                'status': 'warning',
                'message': 'No documents found or extracted',
                'windows_added': 0
            }
        
        logger.info(f"Loaded {len(documents_dict)} documents with {len(table_chunks)} table chunks")
        
        # Step 3: Create sentence windows from regular text
        logger.info("Creating sentence windows from text content...")
        central_sentences, windows = chunk_text(documents_dict, window_size=window_size)
        logger.info(f"Created {len(windows)} sentence windows")
        
        # Step 4: Add table chunks (these are already structured for retrieval)
        table_texts = [chunk['text'] for chunk in table_chunks]
        logger.info(f"Adding {len(table_texts)} structured table chunks")
        
        # Combine all chunks
        all_chunks = windows + table_texts
        all_sentences = central_sentences + table_texts  # Table chunks are self-contained
        
        # Step 5: Embed all chunks
        logger.info(f"Embedding {len(all_sentences)} chunks...")
        embeddings = embed_chunks(all_sentences, self.embedding_model)
        
        # Step 6: Store in vector database
        logger.info("Adding to vector database...")
        
        current_count = self.collection.count()
        ids = [f"enhanced_chunk_{current_count + i}" for i in range(len(all_chunks))]
        
        # Add metadata for table chunks
        metadatas = []
        for i, chunk in enumerate(all_chunks):
            if i < len(windows):
                metadatas.append({'type': 'sentence_window', 'index': i})
            else:
                table_idx = i - len(windows)
                if table_idx < len(table_chunks):
                    metadatas.append({
                        'type': 'table_row',
                        'source': table_chunks[table_idx].get('metadata', {}).get('source', ''),
                        'page': table_chunks[table_idx].get('metadata', {}).get('page', 0)
                    })
                else:
                    metadatas.append({'type': 'table_row', 'index': table_idx})
        
        self.collection.add(
            embeddings=embeddings,
            documents=all_chunks,
            ids=ids,
            metadatas=metadatas
        )
        
        # Step 7: Update version records
        if use_versioning and os.path.isdir(file_path):
            for filepath in os.listdir(file_path):
                full_path = os.path.join(file_path, filepath)
                if os.path.isfile(full_path):
                    version_manager.update_version(full_path)
        
        result = {
            'status': 'success',
            'message': 'Successfully processed with TABLE-AWARE extraction',
            'documents_processed': len(documents_dict),
            'sentence_windows_added': len(windows),
            'table_chunks_added': len(table_texts),
            'total_chunks_added': len(all_chunks),
            'total_chunks_in_db': self.collection.count(),
            'technique': 'table-aware + sentence-window',
            'versioning_enabled': use_versioning
        }
        
        logger.info(f"Table-aware ingestion complete: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error during table-aware ingestion: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'message': str(e),
            'windows_added': 0
        }


@celery_app.task(bind=True, base=CallbackTask, name='ingestion_worker.incremental_update')
def incremental_update(self, folder_path: str) -> Dict[str, any]:
    """
    Perform incremental update - only process changed documents.
    
    Uses document versioning to detect changes and only re-ingest
    documents that have been added or modified.
    """
    logger.info(f"Starting incremental update for: {folder_path}")
    
    try:
        version_manager = DocumentVersionManager()
        changes = version_manager.get_changed_documents(folder_path)
        
        if not changes:
            return {
                'status': 'success',
                'message': 'No changes detected - database is up to date',
                'changes': {}
            }
        
        # Process changes
        results = {
            'new': [],
            'modified': [],
            'deleted': []
        }
        
        for filepath, change_type in changes.items():
            if change_type == 'deleted':
                results['deleted'].append(filepath)
                version_manager.remove_version(filepath)
            else:
                results[change_type].append(filepath)
        
        # If there are new or modified files, process them
        files_to_process = results['new'] + results['modified']
        
        if files_to_process:
            # Process each file
            loader = TableAwareLoader(dpi=2.0)
            all_chunks = []
            all_sentences = []
            
            for filepath in files_to_process:
                if filepath.endswith('.pdf'):
                    text, table_chunks = loader.load_pdf_with_tables(filepath)
                    documents_dict = {os.path.basename(filepath): text}
                    
                    # Create sentence windows
                    central_sentences, windows = chunk_text(documents_dict, window_size=3)
                    
                    all_chunks.extend(windows)
                    all_sentences.extend(central_sentences)
                    
                    # Add table chunks
                    for chunk in table_chunks:
                        all_chunks.append(chunk['text'])
                        all_sentences.append(chunk['text'])
                    
                    # Update version
                    version_manager.update_version(filepath)
            
            if all_chunks:
                # Embed and store
                embeddings = embed_chunks(all_sentences, self.embedding_model)
                
                current_count = self.collection.count()
                ids = [f"incr_chunk_{current_count + i}" for i in range(len(all_chunks))]
                
                self.collection.add(
                    embeddings=embeddings,
                    documents=all_chunks,
                    ids=ids
                )
        
        return {
            'status': 'success',
            'message': f'Incremental update complete',
            'changes': {
                'new_files': len(results['new']),
                'modified_files': len(results['modified']),
                'deleted_files': len(results['deleted'])
            },
            'chunks_added': len(all_chunks) if files_to_process else 0,
            'total_chunks_in_db': self.collection.count()
        }
        
    except Exception as e:
        logger.error(f"Error during incremental update: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'message': str(e)
        }
