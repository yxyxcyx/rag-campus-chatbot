# Issues Fixed - Nov 12, 2025

## Issue 1: Worker Crash (SIGABRT)
**Symptom:** Celery worker crashed with signal 6 (SIGABRT) during ingestion

**Root Causes:**
1. NLTK punkt tokenizer data missing
2. Celery forking issues on macOS with ML libraries

**Fixes Applied:**
1. Downloaded NLTK punkt and punkt_tab tokenizers
2. Changed Celery worker pool from `prefork` to `solo` in `celery_config.py`
3. Created helper scripts: `start_worker.sh`, `test_setup.py`

**Status:** ✅ FIXED - Ingestion now works perfectly

---

## Issue 2: API 500 Error (UI)
**Symptom:** UI showed "Error: The server responded with status code 500"

**Root Cause:**
Groq model `llama3-8b-8192` was **decommissioned** by Groq

**Fixes Applied:**
Updated model in 3 files:
- `rag_pipeline.py` line 200
- `rag_pipeline_v2.py` line 224
- `evaluate.py` line 57

Changed from: `llama3-8b-8192`
Changed to: `llama-3.1-8b-instant`

**Status:** ✅ FIXED - API /ask endpoint now works

---

## Verification Results

### Setup Test (`python test_setup.py`)
```
✅ langchain_huggingface
✅ chromadb
✅ celery
✅ fitz (PyMuPDF)
✅ nltk
✅ redis
✅ fastapi
✅ streamlit
✅ punkt tokenizer
✅ Redis connection
✅ All required files
✅ GROQ_API_KEY
```

### Ingestion Test
```
✅ Documents processed: 2
✅ Chunks added: 43
✅ Total chunks in DB: 43
```

### API Test
```
✅ Retrieved 3 chunks
✅ Generated answer successfully
✅ /ask endpoint working
```

---

## Next Steps for User

### Restart API Server (if not already running)
```bash
# Terminal 1: API Server
cd /Users/chiayuxuan/Documents/rag-project/rag-campus-chatbot
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Test UI
```bash
# Terminal 2: Streamlit UI
cd /Users/chiayuxuan/Documents/rag-project/rag-campus-chatbot
source venv/bin/activate
streamlit run app.py
```

Then open http://localhost:8501 and ask questions!

---

## Files Modified

1. `celery_config.py` - Added `worker_pool='solo'`
2. `rag_pipeline.py` - Updated Groq model
3. `rag_pipeline_v2.py` - Updated Groq model
4. `evaluate.py` - Updated Groq model

## Files Created

1. `start_worker.sh` - Helper to start Celery worker correctly
2. `test_setup.py` - Verify all dependencies and configuration
3. `FIXED_ISSUES.md` - This file
