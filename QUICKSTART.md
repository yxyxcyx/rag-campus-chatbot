# Quick Start Guide

Get your RAG chatbot running in 5 minutes!

## Prerequisites Check

```bash
# Verify Python version (need 3.11+)
python3 --version

# Verify Git
git --version

# Verify you have the project
cd rag-campus-chatbot
```

---

## Step 1: Install Dependencies

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt

# Download NLTK data (required for sentence splitting)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## Step 2: Install Redis

### macOS
```bash
brew install redis
```

### Linux
```bash
sudo apt-get install redis-server
```

### Windows
Download from [Redis Windows](https://github.com/microsoftarchive/redis/releases)

---

## Step 3: Get Groq API Key

1. Go to [Groq Console](https://console.groq.com/)
2. Sign up or log in
3. Navigate to API Keys
4. Create a new API key
5. Copy it

---

## Step 4: Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit .env and add your key
nano .env  # or use any text editor

# Add this line:
GROQ_API_KEY=gsk_your_actual_api_key_here
```

---

## Step 5: Verify Setup

```bash
# Run verification script
python test_setup.py
```

You should see all ✅ checks passing.

---

## Step 6: Start Services

### Terminal 1: Redis
```bash
redis-server
```

Keep this running.

### Terminal 2: Worker
```bash
cd /path/to/rag-campus-chatbot
source venv/bin/activate
./start_worker.sh
```

You should see Celery start with task list.

### Terminal 3: API
```bash
cd /path/to/rag-campus-chatbot
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

You should see "Database ready with X chunks".

### Terminal 4: UI (Optional)
```bash
cd /path/to/rag-campus-chatbot
source venv/bin/activate
streamlit run app.py
```

---

## Step 7: Add Documents

```bash
# In a new terminal (Terminal 5)
cd /path/to/rag-campus-chatbot
source venv/bin/activate

# Copy your documents to data folder
cp /path/to/your/documents/*.pdf data/

# Trigger ingestion
python trigger_ingestion.py data/
```

Wait for "✅ INGESTION COMPLETE" message.

---

## Step 8: Test It!

### Option A: Web UI
1. Open browser: http://localhost:8501
2. Type a question about your documents
3. Hit Enter
4. Get answer!

### Option B: API
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this about?"}'
```

### Option C: API Docs
Open browser: http://localhost:8000/docs

---

## ✅ Success Checklist

- [x] Redis is running (Terminal 1)
- [x] Worker is running (Terminal 2 - shows "Ready")
- [x] API is running (Terminal 3 - shows "Database ready")
- [x] Documents ingested (43+ chunks)
- [x] UI loads (http://localhost:8501)
- [x] Query returns answer

---

## 🐛 Common Issues

### "Connection refused" error
**Fix**: Make sure Redis is running (`redis-server`)

### "Worker crashed" error
**Fix**: This was fixed! Verify `celery_config.py` has `worker_pool='solo'`

### "Database is empty" warning
**Fix**: Run `python trigger_ingestion.py data/`

### "ImportError" or "ModuleNotFoundError"
**Fix**: Activate venv: `source venv/bin/activate`

### "NLTK punkt not found"
**Fix**: `python -c "import nltk; nltk.download('punkt')"`

### "API returns 500" error
**Fix**: Check GROQ_API_KEY in `.env` is correct

---

## 🚀 Next Steps

### Add More Documents
```bash
cp more_documents.pdf data/
python trigger_ingestion.py data/
```

### Run Evaluation
```bash
python evaluate.py
```

### Check Performance
```bash
python check_metrics.py
```

### Clear Database
```bash
python -c "from ingestion_worker import clear_collection; print(clear_collection.delay().get())"
```

---

## 📖 More Information

- Full documentation: See `README.md`
- Architecture details: See `README.md#architecture`
- Troubleshooting: See `FIXED_ISSUES.md`
- API documentation: http://localhost:8000/docs

---

## 🎯 Pro Tips

1. **Keep Redis running**: Don't close Terminal 1
2. **Monitor worker logs**: Watch Terminal 2 for ingestion progress
3. **Use start_worker.sh**: It handles cleanup automatically
4. **Test with test_setup.py**: Run before starting if something breaks
5. **Check data folder**: Put all documents in `data/` folder

---

## 🔄 Restart Everything

If something goes wrong:

```bash
# Kill all processes
pkill -f redis-server
pkill -f celery
pkill -f uvicorn
pkill -f streamlit

# Start again from Step 6
```

---

**Ready to chat with your documents!** 🎉
