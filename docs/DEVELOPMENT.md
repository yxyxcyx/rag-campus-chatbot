# Development Guide

This guide provides comprehensive setup instructions for developing the RAG Campus Chatbot.

---

## Quick Setup

Choose your preferred development method:

### Option 1: Docker (Recommended)

**Prerequisites:**
- Docker Desktop installed ([Download here](https://www.docker.com/products/docker-desktop))
- `.env` file with `GROQ_API_KEY` configured

**One-command setup (background services):**
```bash
# Start Redis, ChromaDB, and the Celery worker automatically
./dev-start.sh
```

**Development workflow (after background services are running, open 2 terminals):**
```bash
# Terminal 1 - Backend API
docker compose -f docker-compose.dev.yml up backend

# Terminal 2 - Frontend UI  
docker compose -f docker-compose.dev.yml up frontend
```

**Access your services:**
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000/docs
- ChromaDB: http://localhost:8001
- Background services (Redis, Celery Worker) run automatically

**Stop services:**
```bash
./dev-stop.sh
```

#### Important notes

- `./dev-start.sh` only launches the background services (Redis, ChromaDB, Celery worker). Keep backend/frontend running in their own terminals for hot reload.
- Do not activate a local Python virtual environment when using Docker for development. All dependencies run inside the containers.
- On macOS and Linux, use `./dev-start.sh`; on Windows, use `dev-start.bat`.
- Avoid mixing Docker-based services with the same services started locally in a virtual environment.

### Option 2: Local Development

**Prerequisites:**
```bash
# Install Python 3.11+
# Install system dependencies
sudo apt-get install tesseract-ocr tesseract-ocr-eng  # Ubuntu/Debian
# or
brew install tesseract  # macOS

# Install Redis server
sudo apt-get install redis-server  # Ubuntu/Debian
# or 
brew install redis  # macOS
```

**Setup:**
```bash
# Install Python dependencies
pip install -r requirements/dev.txt

# Start Redis server
redis-server

# Set environment variables in .env
cp .env.example .env
# Edit .env with your GROQ_API_KEY
```

**Run services:**
```bash
# Terminal 1 - Start Celery worker
cd src && celery -A ingestion_worker worker -l info --pool=solo --concurrency=1

# Terminal 2 - Start FastAPI backend
cd src && uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 3 - Start Streamlit frontend
cd src && streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# Terminal 4 - Ingest documents (optional)
python scripts/trigger_ingestion.py data/
```

---

## Project Structure

After the refactoring, the project follows a clean structure:

```
rag-campus-chatbot/
├── src/                          # Core application code
│   ├── main.py                   # FastAPI backend
│   ├── app.py                    # Streamlit frontend
│   ├── rag_pipeline.py           # RAG pipeline
│   ├── ingestion_worker.py       # Enhanced Celery worker
│   ├── celery_config.py          # Celery configuration
│   ├── enhanced_document_loader.py
│   └── sentence_window_retrieval.py
├── scripts/                      # Utility scripts
│   ├── evaluate.py               # RAG evaluation
│   ├── trigger_ingestion.py      # Document ingestion trigger
│   ├── check_task_status.py      # Celery task monitoring
│   └── check_metrics.py          # Performance gating
├── docs/                         # Documentation
│   └── DEVELOPMENT.md           # This file
├── requirements/                 # Split requirements
│   ├── base.txt                 # Common dependencies
│   ├── api.txt                  # FastAPI service
│   ├── worker.txt               # Celery worker
│   ├── ui.txt                   # Streamlit UI
│   └── dev.txt                  # Development tools
├── tests/                       # Unit tests
├── data/                        # Document storage
└── Docker files & configs
```

---

## Development Workflow

### 1. Document Ingestion
```bash
# Using Docker dev environment (after ./dev-start.sh and backend/frontend are running)
python scripts/trigger_ingestion.py data/

# Using local setup (Option 2)
python scripts/trigger_ingestion.py data/
```

### 2. Testing RAG Pipeline
```bash
# Run evaluation
python scripts/evaluate.py

# Check performance metrics
python scripts/check_metrics.py
```

### 3. Monitoring Tasks
```bash
# Check Celery task status
python scripts/check_task_status.py <task_id>

# View collection statistics
python scripts/trigger_ingestion.py --stats
```

### 4. Code Quality
```bash
# Run linting
flake8 src/ scripts/ tests/

# Run tests
pytest tests/
```

---

## Docker Architecture

### Services Overview

**Development Stack:**
- **backend**: FastAPI API server with hot reload
- **frontend**: Streamlit UI with hot reload
- **worker**: Enhanced Celery worker with OCR
- **redis**: Message broker for task queue
- **chroma**: Vector database for embeddings

### Key Features

**Hot Reload:** Code changes automatically trigger service restarts  
**Volume Mounts:** Local `src/` directory mounted for live editing  
**Isolated Dependencies:** Each service uses optimized requirements  
**Persistent Data:** ChromaDB and Redis data persists between restarts  

### Container Optimization

The project uses **split requirements** for optimized builds:

- **base.txt**: Core shared dependencies.
- **api.txt**: FastAPI + LLM + vector database dependencies.
- **worker.txt**: Celery worker, OCR, document processing, and retrieval stack.
- **ui.txt**: Streamlit UI and HTTP client.
- **dev.txt**: Worker stack plus evaluation and development tools.

This helps to:
- Reduce build times
- Reduce image sizes
- Minimize dependency conflicts
- Limit the security surface area

---

## Troubleshooting

### Common Issues

**Port conflicts:**
```bash
# Check what's using port 8000/8501
lsof -i :8000
lsof -i :8501

# Kill existing processes
docker compose -f docker-compose.dev.yml down
```

**Docker build failures:**
```bash
# Clean rebuild
docker compose -f docker-compose.dev.yml build --no-cache backend
docker system prune -f  # Clean unused images
```

**Permission errors (Linux):**
```bash
# Fix Docker permissions
sudo usermod -aG docker $USER
newgrp docker
```

**Memory issues:**
```bash
# Increase Docker Desktop memory to 4GB+
# Docker Desktop > Settings > Resources > Memory
```

### Environment Variables

Required in `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here

# Optional overrides (for Docker containers; see comments in .env.example)
REDIS_HOST=redis
REDIS_PORT=6379
CHROMA_DB_PATH=/app/chroma_db
COLLECTION_NAME=collection
```

### Performance Optimization

**For large document collections:**
- Increase Celery worker memory: `--max-memory-per-child=1000000`
- Use SSD storage for ChromaDB persistence
- Monitor disk space in `/app/chroma_db`

**For development:**
- Use `--pool=solo --concurrency=1` for Celery (easier debugging)
- Enable hot reload for faster iteration
- Mount only changed files to reduce I/O

---

## Architecture Overview

### RAG Pipeline Features

**Enhanced Document Loading:**
- OCR for image-based PDFs
- Multi-format support (PDF, DOCX, TXT)
- Parallel processing for faster ingestion

**Sentence-Window Retrieval:**
- Chunking technique for better precision
- Context-aware window sizing
- Improved boundary handling

**Hybrid Search:**
- BM25 keyword search + Vector similarity
- Cross-encoder reranking
- Diversity filtering for result quality

**Production-Grade Features:**
- Semantic caching for repeated queries
- Query preprocessing and expansion
- Comprehensive evaluation metrics (RAGAs)
- Async task processing with Celery

### Evaluation Framework

The project includes comprehensive evaluation using RAGAs:

**Metrics:**
- **Faithfulness**: Factual consistency with context
- **Answer Relevancy**: Relevance to user question  
- **Context Recall**: Completeness of retrieved information
- **Context Precision**: Quality of retrieved context

**Quality Gates:**
- Automated performance monitoring
- CI/CD integration ready
- Threshold-based model validation
