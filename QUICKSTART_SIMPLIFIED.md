# Quick Start Guide - SIMPLIFIED (2 Commands!)

Get your RAG chatbot running in **2 minutes with Docker**!

---

## 🚀 Super Quick Start (Recommended)

### Prerequisites
- Docker Desktop installed ([Get it here](https://www.docker.com/products/docker-desktop))
- Your GROQ API key

### Step 1: One-time Setup (30 seconds)
```bash
cd rag-campus-chatbot

# Make sure .env has your API key
cp .env.example .env
# Edit .env and add: GROQ_API_KEY=your_actual_key

# Start everything automatically
./dev-start.sh
```

### Step 2: Daily Development (2 terminals)
```bash
# Terminal 1 - Backend
docker compose -f docker-compose.dev.yml up backend

# Terminal 2 - Frontend  
docker compose -f docker-compose.dev.yml up frontend
```

### Step 3: Access & Use
- **Frontend**: http://localhost:8501 (Chat interface)
- **Backend**: http://localhost:8000/docs (API docs)
- **Ready!** Ask questions about your documents

---

## 🛑 When You're Done
```bash
# Stop services (Ctrl+C in both terminals, then)
./dev-stop.sh
```

---

## ✅ What Runs Automatically

When you run `./dev-start.sh`, these services start automatically:
- ✅ Redis (message broker)
- ✅ ChromaDB (vector database)  
- ✅ Celery Worker (enhanced OCR for documents)
- ✅ Document ingestion (if database is empty)

You only need to manually start:
- 🎯 Backend API (Terminal 1)
- 🎯 Frontend UI (Terminal 2)

---

## 🔧 Alternative: Local Development (Advanced)

If you prefer local development without Docker:

### Prerequisites
- Python 3.11+
- Redis server
- Tesseract OCR

### Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Run (4 terminals needed)
```bash
# Terminal 1: Redis
redis-server

# Terminal 2: Worker
./start_worker.sh

# Terminal 3: Backend
uvicorn main:app --reload --port 8000

# Terminal 4: Frontend
streamlit run app.py
```

---

## 📊 Validation

### Check Database Status
```bash
# Docker version
docker compose -f docker-compose.dev.yml exec backend python -c "
import chromadb
client = chromadb.PersistentClient(path='/app/chroma_db')
collection = client.get_collection('collection')
print(f'Database has {collection.count()} chunks')
"

# Local version
python -c "
import chromadb
client = chromadb.PersistentClient(path='./chroma_db')
collection = client.get_collection('collection')
print(f'Database has {collection.count()} chunks')
"
```

**Expected**: 1000+ chunks from your PDF document

### Test Query
Visit http://localhost:8501 and ask:
- "What are the tuition fees?"
- "How do I apply for admission?"
- "What are the library hours?"

**Expected**: Actual answers with specific information (not "I don't have enough information")

---

## 🐛 Troubleshooting

### Docker Issues
```bash
# Docker not running
# → Start Docker Desktop

# Port conflicts
./dev-stop.sh
./dev-start.sh

# Check container status
docker compose -f docker-compose.dev.yml ps

# View logs
docker compose -f docker-compose.dev.yml logs backend
docker compose -f docker-compose.dev.yml logs worker
```

### Database Issues
```bash
# Reingest documents (Docker)
docker compose -f docker-compose.dev.yml exec worker python -c "
from enhanced_ingestion_worker import process_document
print(process_document.delay('/app/data').get())
"

# Reingest documents (Local)
python trigger_ingestion.py data/
```

### API Issues
```bash
# Check GROQ_API_KEY is set
grep GROQ_API_KEY .env

# Test API directly
curl http://localhost:8000/
```

---

## 🎯 Development Tips

### Edit Code Normally
- Edit files in VS Code as usual
- Docker automatically syncs changes
- Services auto-restart on file changes ✨

### Add New Documents
```bash
# Just copy to data folder
cp new_document.pdf data/

# Reingest (Docker)
docker compose -f docker-compose.dev.yml exec worker python -c "
from enhanced_ingestion_worker import process_document
print(process_document.delay('/app/data').get())
"

# Reingest (Local)
python trigger_ingestion.py data/
```

---

## 📚 More Information

- **Full Docker Guide**: See `DOCKER_DEVELOPMENT_GUIDE.md`
- **Architecture Details**: See `README.md`
- **Engineering Deep-dive**: See `ENGINEERING_IMPROVEMENTS.md`
- **Advanced Features**: See `PRINCIPAL_ENGINEER_SUMMARY.md`

---

## 🏆 Success Checklist

- [x] Docker running (`docker info`)
- [x] .env with GROQ_API_KEY
- [x] Services started with `./dev-start.sh`
- [x] Backend running (Terminal 1)
- [x] Frontend running (Terminal 2)
- [x] Database has 1000+ chunks
- [x] Frontend loads at http://localhost:8501
- [x] Can ask questions and get real answers

---

**You're ready to build amazing RAG applications!** 🚀

**Docker version = 2 commands, 2 terminals**  
**Local version = 4 terminals (if you prefer)**

**Happy coding!** ✨
