#!/usr/bin/env python3
# test_enhanced_features.py

"""
Test Script for Enhanced RAG Features

Tests all the new features:
1. Table-aware document loading
2. Enhanced prompt engineering
3. Multi-part question handling
4. Ambiguous query detection
5. Conversation memory
6. Citation verification

Usage:
    python scripts/test_enhanced_features.py
"""

import os
import sys
import requests
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")


def test_api_health():
    """Test API is running."""
    print("\n" + "="*60)
    print("TEST 1: API Health Check")
    print("="*60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/")
        data = response.json()
        print(f"Status: {data.get('status')}")
        print(f"Version: {data.get('version')}")
        print(f"Database chunks: {data.get('database_chunks')}")
        return True
    except Exception as e:
        print(f"ERROR: Could not connect to API: {e}")
        print(f"Make sure the API is running at {API_BASE_URL}")
        return False


def test_basic_ask(query: str):
    """Test basic /ask endpoint."""
    print("\n" + "="*60)
    print("TEST 2: Basic Ask Endpoint")
    print("="*60)
    print(f"Query: {query}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={"query": query}
        )
        data = response.json()
        print(f"\nResponse:\n{data.get('response', data)}")
        return data
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def test_enhanced_ask(query: str, session_id: str = None):
    """Test enhanced /ask/enhanced endpoint."""
    print("\n" + "="*60)
    print("TEST 3: Enhanced Ask Endpoint")
    print("="*60)
    print(f"Query: {query}")
    if session_id:
        print(f"Session ID: {session_id}")
    
    try:
        payload = {"query": query}
        if session_id:
            payload["session_id"] = session_id
            
        response = requests.post(
            f"{API_BASE_URL}/ask/enhanced",
            json=payload
        )
        data = response.json()
        
        print(f"\n--- Response ---")
        print(f"Answer: {data.get('response', 'N/A')[:500]}...")
        print(f"\nConfidence: {data.get('confidence', 'N/A')}")
        print(f"Sources: {data.get('sources', [])}")
        print(f"Session ID: {data.get('session_id', 'N/A')}")
        print(f"Needs clarification: {data.get('needs_clarification', False)}")
        
        if data.get('query_analysis'):
            print(f"\n--- Query Analysis ---")
            analysis = data['query_analysis']
            print(f"  Multi-part: {analysis.get('is_multi_part', False)}")
            print(f"  Ambiguous: {analysis.get('is_ambiguous', False)}")
            print(f"  Intent: {analysis.get('intent', 'N/A')}")
            print(f"  Follow-up: {analysis.get('is_followup', False)}")
        
        return data
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def test_query_analysis(query: str):
    """Test query analysis endpoint."""
    print("\n" + "="*60)
    print("TEST 4: Query Analysis")
    print("="*60)
    print(f"Query: {query}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze",
            json={"query": query}
        )
        data = response.json()
        
        print(f"\n--- Analysis ---")
        print(f"Multi-part: {data.get('is_multi_part', False)}")
        print(f"Sub-questions: {data.get('sub_questions', [])}")
        print(f"Ambiguous: {data.get('is_ambiguous', False)}")
        print(f"Ambiguous terms: {data.get('ambiguous_terms', [])}")
        print(f"Intent: {data.get('intent', 'N/A')}")
        print(f"Confidence: {data.get('confidence', 'N/A')}")
        
        return data
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def test_conversation_memory():
    """Test conversation memory with follow-up questions."""
    print("\n" + "="*60)
    print("TEST 5: Conversation Memory")
    print("="*60)
    
    session_id = "test-session-123"
    
    # First question
    print("\n--- First Question ---")
    result1 = test_enhanced_ask(
        "What is the tuition fee for Bachelor of Data Science?",
        session_id
    )
    
    if result1:
        # Follow-up question
        print("\n--- Follow-up Question ---")
        result2 = test_enhanced_ask(
            "What about for international students?",
            session_id
        )
        
        # Check session info
        print("\n--- Session Info ---")
        try:
            response = requests.get(f"{API_BASE_URL}/session/{session_id}")
            data = response.json()
            print(f"History count: {data.get('history_count', 0)}")
            print(f"Topics: {data.get('topics', [])}")
            print(f"Entities: {data.get('entities', {})}")
        except Exception as e:
            print(f"ERROR: {e}")
        
        # Clear session
        print("\n--- Clearing Session ---")
        try:
            response = requests.delete(f"{API_BASE_URL}/session/{session_id}")
            print(f"Result: {response.json()}")
        except Exception as e:
            print(f"ERROR: {e}")


def test_multi_part_question():
    """Test multi-part question handling."""
    print("\n" + "="*60)
    print("TEST 6: Multi-Part Question")
    print("="*60)
    
    query = "What are the tuition fees AND what is the refund policy?"
    
    # First analyze
    analysis = test_query_analysis(query)
    
    if analysis and analysis.get('is_multi_part'):
        print("\n✓ Multi-part question detected correctly!")
    else:
        print("\n⚠ Multi-part question not detected")
    
    # Then ask
    result = test_enhanced_ask(query)
    
    return result


def test_fee_question():
    """Test the specific fee question that was failing."""
    print("\n" + "="*60)
    print("TEST 7: Fee Question (The Original Issue)")
    print("="*60)
    
    query = "What is the annual tuition fee for the Bachelor of Engineering in Data Science (Honours) for Malaysian students?"
    
    print(f"Query: {query}")
    print(f"\nExpected answer should include: RM 31,000, 4 years")
    
    result = test_enhanced_ask(query)
    
    if result:
        response = result.get('response', '').lower()
        if 'rm 31,000' in response or 'rm31,000' in response or '31000' in response or '31,000' in response:
            print("\n✓ SUCCESS: Fee information found in response!")
        else:
            print("\n⚠ Fee information may not be in response. Check if data was re-ingested.")
    
    return result


def test_table_extraction_preview():
    """Preview what table extraction produces."""
    print("\n" + "="*60)
    print("TEST 8: Table Extraction Preview (Local)")
    print("="*60)
    
    try:
        from table_aware_loader import TableAwareLoader
        
        data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
        if not os.path.exists(data_folder):
            data_folder = "data/"
        
        if not os.path.exists(data_folder):
            print(f"Data folder not found: {data_folder}")
            return
        
        loader = TableAwareLoader(dpi=2.0)
        
        # Check for fee-related PDF
        fee_pdf = os.path.join(data_folder, "Local Tuition Fees 2026.pdf")
        
        if os.path.exists(fee_pdf):
            print(f"Loading: {fee_pdf}")
            text, table_chunks = loader.load_pdf_with_tables(fee_pdf)
            
            print(f"\nExtracted {len(table_chunks)} table chunks")
            
            if table_chunks:
                print("\n--- Sample Table Chunks ---")
                for i, chunk in enumerate(table_chunks[:3]):
                    print(f"\n[Chunk {i+1}]")
                    print(chunk['text'][:300])
                    print("...")
        else:
            print(f"Fee PDF not found at: {fee_pdf}")
            print("Available files in data folder:")
            for f in os.listdir(data_folder):
                print(f"  - {f}")
                
    except ImportError as e:
        print(f"Import error (run from project root): {e}")
    except Exception as e:
        print(f"Error: {e}")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("ENHANCED RAG FEATURE TESTS")
    print("="*60)
    
    # Check API first
    if not test_api_health():
        print("\n⚠ API is not running. Some tests will be skipped.")
        print("Start the API with: make dev-start or uvicorn src.main:app --reload")
        
        # Run local tests only
        test_table_extraction_preview()
        return
    
    # Run all tests
    test_basic_ask("What programs does the university offer?")
    test_fee_question()
    test_multi_part_question()
    test_conversation_memory()
    test_table_extraction_preview()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--fee":
            test_fee_question()
        elif sys.argv[1] == "--tables":
            test_table_extraction_preview()
        elif sys.argv[1] == "--memory":
            if test_api_health():
                test_conversation_memory()
        elif sys.argv[1] == "--analyze":
            query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "What are the fees and deadlines?"
            if test_api_health():
                test_query_analysis(query)
        else:
            query = " ".join(sys.argv[1:])
            if test_api_health():
                test_enhanced_ask(query)
    else:
        run_all_tests()
