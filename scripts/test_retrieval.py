#!/usr/bin/env python3
"""
Test script to verify the advanced retrieval features are working correctly.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
import chromadb
from rank_bm25 import BM25Okapi

from rag_pipeline import (
    retrieve_and_rerank,
    retrieve_and_rerank_hybrid,
    QueryProcessor,
    diversity_filter
)

load_dotenv()


def test_query_processor():
    """Test query preprocessing"""
    print("\n=== Testing Query Processor ===")
    processor = QueryProcessor()
    
    test_queries = [
        "  What   is   the   admission   process?  ",
        "how to apply for scholarship",
        "campus facilities"
    ]
    
    for query in test_queries:
        clean = processor.clean_query(query)
        expanded = processor.expand_query(query)
        print(f"Original: '{query}'")
        print(f"Cleaned:  '{clean}'")
        print(f"Expanded: {expanded}")
        print()


def test_diversity_filter():
    """Test diversity filtering"""
    print("\n=== Testing Diversity Filter ===")
    
    documents = [
        "The university offers various undergraduate programs in engineering.",
        "The university provides multiple undergraduate courses in engineering field.",
        "Students can choose from different scholarship options available.",
        "Various scholarship opportunities are available for students.",
        "The campus has modern facilities including library and labs."
    ]
    
    print(f"Original documents: {len(documents)}")
    for i, doc in enumerate(documents, 1):
        print(f"{i}. {doc[:70]}...")
    
    filtered = diversity_filter(documents, threshold=0.8)
    print(f"\nFiltered documents (threshold=0.8): {len(filtered)}")
    for i, doc in enumerate(filtered, 1):
        print(f"{i}. {doc[:70]}...")


def test_enhanced_retrieval():
    """Test enhanced retrieval with diversity filtering"""
    print("\n=== Testing Enhanced Retrieval ===")
    
    # Initialize components
    print("Loading models...")
    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Connect to ChromaDB
    chroma_db_path = os.getenv('CHROMA_DB_PATH', './chroma_db')
    collection_name = os.getenv('COLLECTION_NAME', 'collection')
    client = chromadb.PersistentClient(path=chroma_db_path)
    collection = client.get_or_create_collection(name=collection_name)
    
    if collection.count() == 0:
        print("WARNING: Database is empty. Please run ingestion first.")
        return
    
    test_query = "What are the admission requirements?"
    
    # Test standard retrieval (without diversity filter)
    print(f"\nQuery: '{test_query}'")
    print("\n1. Standard retrieval (no diversity filter):")
    results_standard = retrieve_and_rerank(
        test_query,
        embedding_model,
        collection,
        cross_encoder,
        n_initial=20,
        n_final=5,
        use_diversity_filter=False
    )
    print(f"   Retrieved {len(results_standard)} chunks")
    for i, chunk in enumerate(results_standard, 1):
        print(f"   {i}. {chunk[:100]}...")
    
    # Test with diversity filter
    print("\n2. Enhanced retrieval (with diversity filter):")
    results_filtered = retrieve_and_rerank(
        test_query,
        embedding_model,
        collection,
        cross_encoder,
        n_initial=20,
        n_final=5,
        use_diversity_filter=True,
        diversity_threshold=0.85
    )
    print(f"   Retrieved {len(results_filtered)} chunks")
    for i, chunk in enumerate(results_filtered, 1):
        print(f"   {i}. {chunk[:100]}...")
    
    # Compare results
    print("\n3. Comparison:")
    print(f"   Standard: {len(results_standard)} results")
    print(f"   Filtered: {len(results_filtered)} results")
    if len(results_standard) > len(results_filtered):
        print(f"   Diversity filter removed {len(results_standard) - len(results_filtered)} redundant results")


def test_hybrid_retrieval():
    """Test hybrid BM25 + Vector retrieval"""
    print("\n=== Testing Hybrid Retrieval ===")
    
    # Initialize components
    print("Loading models...")
    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Connect to ChromaDB
    chroma_db_path = os.getenv('CHROMA_DB_PATH', './chroma_db')
    collection_name = os.getenv('COLLECTION_NAME', 'collection')
    client = chromadb.PersistentClient(path=chroma_db_path)
    collection = client.get_or_create_collection(name=collection_name)
    
    chunk_count = collection.count()
    if chunk_count == 0:
        print("WARNING: Database is empty. Please run ingestion first.")
        return
    
    # Initialize BM25
    print("Initializing BM25 index...")
    all_results = collection.get(limit=min(chunk_count, 1000))  # Limit for testing
    all_documents = all_results.get("documents", [])
    
    if not all_documents:
        print("ERROR: Could not fetch documents")
        return
    
    tokenized_docs = [doc.lower().split() for doc in all_documents]
    bm25_index = BM25Okapi(tokenized_docs)
    print(f"BM25 index initialized with {len(all_documents)} documents")
    
    test_query = "scholarship application deadline"
    
    # Test hybrid retrieval
    print(f"\nQuery: '{test_query}'")
    print("\nHybrid retrieval (BM25 + Vector):")
    results = retrieve_and_rerank_hybrid(
        test_query,
        embedding_model,
        collection,
        cross_encoder,
        bm25_index=bm25_index,
        documents=all_documents,
        n_initial=30,
        n_final=5,
        bm25_weight=0.3,
        vector_weight=0.7,
        use_diversity_filter=True
    )
    
    print(f"Retrieved {len(results)} chunks")
    for i, chunk in enumerate(results, 1):
        print(f"{i}. {chunk[:150]}...")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Advanced Retrieval Features")
    print("=" * 60)
    
    # Test individual components
    test_query_processor()
    test_diversity_filter()
    
    # Test integrated retrieval
    test_enhanced_retrieval()
    
    # Test hybrid retrieval (optional - requires more memory)
    if os.getenv('TEST_HYBRID', 'false').lower() == 'true':
        test_hybrid_retrieval()
    else:
        print("\n=== Skipping Hybrid Retrieval Test ===")
        print("Set TEST_HYBRID=true to test hybrid retrieval")
    
    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
