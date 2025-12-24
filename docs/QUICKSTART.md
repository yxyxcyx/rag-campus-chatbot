# Quick Start Guide for Beginners

This guide walks you through setting up and running the RAG Campus Chatbot from scratch.

## Prerequisites

- **Python 3.11+** installed
- **8GB RAM** minimum (the embedding model uses ~500MB)
- **Groq API Key** (free at https://console.groq.com)

---

## Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-username/rag-campus-chatbot.git
cd rag-campus-chatbot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements/dev.txt
```

## Step 2: Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Groq API key
# You can get a free key at https://console.groq.com
```

Edit `.env` file:
```
GROQ_API_KEY=your_api_key_here
```

## Step 3: Add Your Documents

```bash
# Create data folder if it doesn't exist
mkdir -p data

# Copy your PDF, DOCX, or TXT files
cp /path/to/your/documents/*.pdf data/
```

## Step 4: Ingest Documents

**This is the most important step!** Documents are NOT auto-ingested.

```bash
# Smart ingestion - automatically handles tables and regular text
make ingest

# Or if you need to clear existing data first:
make ingest-clear
```

Wait for ingestion to complete. You should see:
```
✅ SMART INGESTION COMPLETE
Documents processed:     5
Sentence windows:        1180
Table chunks:            45
Total chunks added:      1225
```

## Step 5: Start the Chatbot

### Option A: Local Development (Recommended for beginners)

You need 3 terminal windows:

**Terminal 1 - Start Redis:**
```bash
# If you have Redis installed
redis-server

# Or use Docker
docker run -p 6379:6379 redis:7-alpine
```

**Terminal 2 - Start API:**
```bash
source venv/bin/activate
make api
```

**Terminal 3 - Start UI:**
```bash
source venv/bin/activate
make ui
```

### Option B: Docker (All-in-one)

```bash
# Start everything
make up

# Wait for services to be ready (about 30 seconds)
# Then ingest documents
make ingest
```

## Step 6: Use the Chatbot

Open your browser:
- **UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

Ask questions like:
- "What are the tuition fees for Data Science?"
- "What is the refund policy?"
- "How do I apply for student housing?"

---

## Common Issues

### "I don't have enough information to answer..."

**Cause:** Documents not ingested properly.

**Solution:**
```bash
# Check if data is in database
make ingest-stats

# If count is 0, re-ingest
make ingest-clear
```

### "Connection refused" error

**Cause:** API server not running.

**Solution:**
```bash
# Make sure API is running
make api
```

### "Redis connection error"

**Cause:** Redis not running.

**Solution:**
```bash
# Start Redis
redis-server
# Or with Docker:
docker run -p 6379:6379 redis:7-alpine
```

---

## Quick Command Reference

| Command | What it does |
|---------|--------------|
| `make ingest` | Smart ingestion (handles tables automatically) |
| `make ingest-clear` | Clear database and re-ingest |
| `make ingest-stats` | Show database statistics |
| `make api` | Start API server |
| `make ui` | Start Streamlit UI |
| `make up` | Start all services with Docker |
| `make down` | Stop Docker services |
| `make test-api` | Check if API is running |

---

## Adding New Documents

1. Copy new documents to `data/` folder
2. Run `make ingest` (it will add to existing data)
3. Or run `make ingest-clear` to start fresh

---

## Free Deployment Options

See [DEPLOYMENT.md](DEPLOYMENT.md) for deploying to:
- Streamlit Cloud (Free, easiest)
- Railway (Free tier)
- Render (Free tier)
