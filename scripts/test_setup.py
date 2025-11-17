#!/usr/bin/env python3
# test_setup.py - Verify all dependencies and configuration

import sys
import os

def test_imports():
    """Test all required imports"""
    print(" Testing imports...")
    failures = []
    
    tests = [
        ('langchain_huggingface', 'HuggingFaceEmbeddings'),
        ('chromadb', None),
        ('celery', 'Celery'),
        ('fitz', None),  # PyMuPDF
        ('nltk', None),
        ('redis', None),
        ('fastapi', None),
        ('streamlit', None),
    ]
    
    for module, attr in tests:
        try:
            mod = __import__(module)
            if attr:
                getattr(mod, attr)
            print(f"   {module}")
        except Exception as e:
            print(f"   {module}: {e}")
            failures.append(module)
    
    return len(failures) == 0

def test_nltk_data():
    """Test NLTK data availability"""
    print("\n Testing NLTK data...")
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
        print("   punkt tokenizer")
        return True
    except Exception as e:
        print(f"   punkt tokenizer: {e}")
        print("  Fix: python -c \"import nltk; nltk.download('punkt')\"")
        return False

def test_redis():
    """Test Redis connection"""
    print("\n Testing Redis connection...")
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379, db=0)
        client.ping()
        print("   Redis connection")
        return True
    except Exception as e:
        print(f"   Redis connection: {e}")
        print("  Fix: redis-server")
        return False

def test_file_structure():
    """Test required files exist"""
    print("\n Testing file structure...")
    required_files = [
        'main.py',
        'ingestion_worker.py',
        'celery_config.py',
        'rag_pipeline.py',
        'trigger_ingestion.py',
        '.env',
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"   {file}")
        else:
            print(f"   {file}")
            all_exist = False
    
    return all_exist

def test_env_vars():
    """Test environment variables"""
    print("\n Testing environment variables...")
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = ['GROQ_API_KEY']
    all_set = True
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"   {var}")
        else:
            print(f"    {var} not set")
            all_set = False
    
    return all_set

def main():
    print("="*60)
    print("RAG SYSTEM SETUP VERIFICATION")
    print("="*60)
    
    results = [
        test_imports(),
        test_nltk_data(),
        test_redis(),
        test_file_structure(),
        test_env_vars(),
    ]
    
    print("\n" + "="*60)
    if all(results):
        print(" ALL CHECKS PASSED!")
        print("="*60)
        print("\n You can now start the system:")
        print("  1. Terminal 1: ./start_worker.sh")
        print("  2. Terminal 2: uvicorn main:app --reload --port 8000")
        print("  3. Terminal 3: python trigger_ingestion.py data/")
        return 0
    else:
        print(" SOME CHECKS FAILED!")
        print("="*60)
        print("\nPlease fix the issues above before starting the system.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
