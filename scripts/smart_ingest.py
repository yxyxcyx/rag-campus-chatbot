#!/usr/bin/env python3
# smart_ingest.py

"""
Smart Document Ingestion - Unified Approach

This script automatically handles BOTH:
- Regular text documents (using sentence-window chunking)
- PDFs with tables (using table-aware extraction)

It intelligently detects which documents contain tables and applies
the appropriate extraction method.

Usage:
    python scripts/smart_ingest.py data/
    python scripts/smart_ingest.py data/ --clear    # Clear database first
    python scripts/smart_ingest.py --stats          # Show stats only
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import get_settings
from logging_config import setup_logging, get_logger


def detect_pdf_has_tables(file_path: str) -> bool:
    """
    Detect if a PDF likely contains tables by checking for table structures.
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        
        for page_num in range(min(5, len(doc))):  # Check first 5 pages
            page = doc[page_num]
            
            # Method 1: Try to find tables using PyMuPDF's table detection
            try:
                tables = page.find_tables()
                if tables and len(tables.tables) > 0:
                    doc.close()
                    return True
            except:
                pass
            
            # Method 2: Look for table-like patterns in text
            text = page.get_text()
            # Tables often have multiple columns with aligned numbers
            lines = text.split('\n')
            numeric_lines = 0
            for line in lines:
                # Count lines with multiple numbers (likely table rows)
                import re
                numbers = re.findall(r'\d+[,.]?\d*', line)
                if len(numbers) >= 2:
                    numeric_lines += 1
            
            # If many lines have multiple numbers, likely a table
            if numeric_lines > 5:
                doc.close()
                return True
        
        doc.close()
        return False
        
    except Exception as e:
        return False


def load_with_table_awareness(file_path: str, logger) -> Tuple[Dict[str, str], List[Dict]]:
    """
    Load a document with table awareness.
    Returns: (documents_dict, table_chunks)
    """
    from table_aware_loader import TableAwareLoader
    from enhanced_document_loader import EnhancedDocumentLoader
    
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.pdf':
        # Check if PDF has tables
        has_tables = detect_pdf_has_tables(file_path)
        
        if has_tables:
            logger.info(f"Tables detected in {Path(file_path).name}, using table-aware extraction")
            loader = TableAwareLoader(dpi=2.0)
            text_content, table_chunks = loader.load_pdf_with_tables(file_path)
            return {file_path: text_content}, table_chunks
        else:
            logger.info(f"No tables in {Path(file_path).name}, using standard extraction")
            loader = EnhancedDocumentLoader(dpi=2.0, parallel_workers=2)
            docs = loader.load_document(file_path)
            return docs, []
    else:
        # Non-PDF files: use standard loader
        loader = EnhancedDocumentLoader(dpi=2.0, parallel_workers=2)
        docs = loader.load_document(file_path)
        return docs, []


def smart_ingest(data_path: str, clear_first: bool = False):
    """
    Smart ingestion that automatically handles tables and regular documents.
    """
    logger = setup_logging(level="INFO", json_output=False, app_name="smart-ingest")
    settings = get_settings()
    
    from langchain_huggingface import HuggingFaceEmbeddings
    from sentence_window_retrieval import chunk_text_with_sentence_windows
    import chromadb
    
    # Initialize ChromaDB
    logger.info("Connecting to ChromaDB", path=settings.chroma_db_path)
    client = chromadb.PersistentClient(path=settings.chroma_db_path)
    
    # Clear if requested
    if clear_first:
        logger.info("Clearing existing collection...")
        try:
            client.delete_collection(name=settings.collection_name)
            logger.info("Collection cleared")
        except:
            logger.info("No existing collection to clear")
    
    collection = client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Initialize embedding model
    logger.info("Loading embedding model", model=settings.embedding_model_name)
    embedding_model = HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Collect all documents
    all_documents = {}
    all_table_chunks = []
    
    # Find all supported files
    data_path = Path(data_path)
    if data_path.is_file():
        files = [data_path]
    else:
        files = list(data_path.glob("**/*"))
        files = [f for f in files if f.suffix.lower() in ['.pdf', '.docx', '.txt']]
    
    if not files:
        logger.error("No documents found", path=str(data_path))
        print("\n❌ No documents found in the specified path.")
        print("Supported formats: .pdf, .docx, .txt")
        return
    
    logger.info(f"Found {len(files)} documents to process")
    
    # Process each file
    for file_path in files:
        try:
            logger.info(f"Processing: {file_path.name}")
            docs, table_chunks = load_with_table_awareness(str(file_path), logger)
            all_documents.update(docs)
            all_table_chunks.extend(table_chunks)
        except Exception as e:
            logger.warning(f"Failed to process {file_path.name}: {e}")
    
    if not all_documents and not all_table_chunks:
        logger.error("No content extracted from documents")
        return
    
    # Statistics
    total_chars = sum(len(text) for text in all_documents.values())
    logger.info(f"Total text content: {total_chars:,} characters")
    logger.info(f"Table chunks extracted: {len(all_table_chunks)}")
    
    # Create sentence windows from regular text
    logger.info("Creating sentence windows from text content...")
    central_sentences, windows = chunk_text_with_sentence_windows(
        all_documents, 
        window_size=3
    )
    logger.info(f"Created {len(windows)} sentence windows")
    
    # Prepare table chunks
    table_texts = [chunk['text'] for chunk in all_table_chunks]
    
    # Combine all content
    all_chunks = windows + table_texts
    all_sentences_for_embedding = central_sentences + table_texts
    
    if not all_chunks:
        logger.error("No chunks to embed")
        return
    
    # Generate embeddings
    logger.info(f"Generating embeddings for {len(all_sentences_for_embedding)} chunks...")
    embeddings = embedding_model.embed_documents(all_sentences_for_embedding)
    logger.info("Embeddings generated")
    
    # Prepare metadata
    ids = []
    metadatas = []
    
    for i in range(len(windows)):
        ids.append(f"smart_sw_{i}")
        metadatas.append({
            'type': 'sentence_window',
            'index': i
        })
    
    for i, chunk in enumerate(all_table_chunks):
        ids.append(f"smart_table_{i}")
        meta = {
            'type': 'table_row',
            'index': i
        }
        if 'metadata' in chunk:
            meta['source'] = chunk['metadata'].get('source', '')
            meta['page'] = chunk['metadata'].get('page', 0)
        metadatas.append(meta)
    
    # Store in ChromaDB
    logger.info("Storing in vector database...")
    
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        end_idx = min(i + batch_size, len(all_chunks))
        collection.add(
            embeddings=embeddings[i:end_idx],
            documents=all_chunks[i:end_idx],
            ids=ids[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )
        logger.info(f"Added batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}")
    
    final_count = collection.count()
    
    # Print summary
    print(f"\n{'='*60}")
    print("✅ SMART INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"Documents processed:     {len(all_documents)}")
    print(f"Sentence windows:        {len(windows)}")
    print(f"Table chunks:            {len(table_texts)}")
    print(f"Total chunks added:      {len(all_chunks)}")
    print(f"Total in database:       {final_count}")
    print(f"{'='*60}")
    
    if table_texts:
        print(f"\n📊 Table-aware extraction was used for documents with tables.")
    else:
        print(f"\n📝 No tables detected. Standard extraction was used.")
    
    return {
        'documents_processed': len(all_documents),
        'sentence_windows': len(windows),
        'table_chunks': len(table_texts),
        'total_in_db': final_count
    }


def show_stats():
    """Show database statistics."""
    settings = get_settings()
    import chromadb
    
    client = chromadb.PersistentClient(path=settings.chroma_db_path)
    
    try:
        collection = client.get_collection(name=settings.collection_name)
        count = collection.count()
        
        print(f"\n{'='*60}")
        print("DATABASE STATISTICS")
        print(f"{'='*60}")
        print(f"Collection: {settings.collection_name}")
        print(f"Total chunks: {count}")
        
        # Try to get sample to show types
        if count > 0:
            sample = collection.get(limit=min(100, count), include=['metadatas'])
            types = {}
            for meta in sample.get('metadatas', []):
                if meta:
                    t = meta.get('type', 'unknown')
                    types[t] = types.get(t, 0) + 1
            
            print(f"\nChunk types (sample of {len(sample.get('metadatas', []))}): ")
            for t, c in types.items():
                print(f"  - {t}: {c}")
        
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"No collection found or error: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("="*60)
        print("SMART DOCUMENT INGESTION")
        print("="*60)
        print("\nUsage:")
        print("  python scripts/smart_ingest.py data/           # Ingest all documents")
        print("  python scripts/smart_ingest.py data/ --clear   # Clear and re-ingest")
        print("  python scripts/smart_ingest.py --stats         # Show statistics")
        print("\nThis script automatically:")
        print("  ✓ Detects PDFs with tables and uses table-aware extraction")
        print("  ✓ Uses sentence-window chunking for regular text")
        print("  ✓ Combines both approaches for best results")
        sys.exit(1)
    
    if sys.argv[1] == "--stats":
        show_stats()
    else:
        data_path = sys.argv[1]
        clear_first = "--clear" in sys.argv
        
        if not os.path.exists(data_path):
            print(f"❌ Path not found: {data_path}")
            sys.exit(1)
        
        smart_ingest(data_path, clear_first=clear_first)
