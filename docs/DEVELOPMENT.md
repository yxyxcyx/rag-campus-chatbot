# Development Guide

This guide provides comprehensive setup instructions for developing the RAG Campus Chatbot.

---

## Quick Start with Make

All development commands are consolidated in a **Makefile** for consistency across platforms.

```bash
# See all available commands
make help

# Quick setup
make dev          # Create venv and install dependencies
make up           # Start all Docker services
make ingest       # Ingest documents from data/
make test         # Run all tests
```

---

## Quick Setup

Choose your preferred development method:

### Option 1: Docker (Recommended)

**Prerequisites:**
- Docker Desktop installed ([Download here](https://www.docker.com/products/docker-desktop))
- GNU Make (`make --version` to check)
- `.env` file with `GROQ_API_KEY` configured

**One-command setup:**
```bash
# 1. Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 2. Start all services
make up

# 3. Access services
# Frontend: http://localhost:8501
# Backend API: http://localhost:8000/docs
```

**Development workflow:**
```bash
make up-dev       # Start with hot reload
make logs         # View logs
make down         # Stop services
make restart      # Restart all services
make clean        # Stop and remove volumes
```

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
# 1. Setup development environment
make dev

# 2. Configure environment  
cp .env.example .env
# Edit .env with your GROQ_API_KEY

# 3. Start Redis server (separate terminal)
redis-server
```

**Run services:**
```bash
make api          # Start FastAPI backend
make worker       # Start Celery worker
make ui           # Start Streamlit frontend
```

---

## Make Commands Reference

### Setup
| Command | Description |
|---------|-------------|
| `make dev` | Create venv and install all dependencies |
| `make install` | Install Python dependencies only |
| `make check-env` | Verify environment configuration |

### Docker
| Command | Description |
|---------|-------------|
| `make up` | Start all services with Docker Compose |
| `make up-dev` | Start development services (hot reload) |
| `make down` | Stop all Docker services |
| `make logs` | Show logs from all services |
| `make logs-api` | Show API logs only |
| `make restart` | Restart all services |
| `make clean` | Stop services and remove volumes |

### Local Development
| Command | Description |
|---------|-------------|
| `make api` | Start API server locally |
| `make worker` | Start Celery worker locally |
| `make ui` | Start Streamlit UI locally |
| `make shell` | Open Python shell with project context |

### Ingestion
| Command | Description |
|---------|-------------|
| `make ingest` | Ingest documents from data/ folder |
| `make ingest-file FILE=path/to/file.pdf` | Ingest a specific file |
| `make ingest-stats` | Show database statistics |
| `make ingest-clear` | Clear all documents from database |

### Testing
| Command | Description |
|---------|-------------|
| `make test` | Run all tests with pytest |
| `make test-arch` | Run architecture validation tests |
| `make test-eval` | Run system evaluation (requires API) |
| `make test-quick` | Quick smoke test |

### Code Quality
| Command | Description |
|---------|-------------|
| `make lint` | Run linters (ruff, flake8) |
| `make format` | Format code with black and isort |
| `make typecheck` | Run type checking with mypy |

---

## Project Structure

```
rag-campus-chatbot/
├── Makefile                      # Development commands (start here!)
├── src/                          # Core application code
│   ├── main.py                   # FastAPI backend + API endpoints
│   ├── app.py                    # Streamlit frontend
│   ├── config.py                 # Centralized configuration (Pydantic)
│   ├── logging_config.py         # Structured logging
│   ├── rag_pipeline.py           # RAG pipeline + hybrid search (RRF)
│   ├── enhanced_rag_engine.py    # Query analysis, conversation memory, citations
│   ├── ingestion_worker.py       # Celery worker for ingestion
│   ├── celery_config.py          # Celery configuration
│   ├── enhanced_document_loader.py  # OCR document loading
│   ├── table_aware_loader.py     # Table extraction for PDFs
│   └── sentence_window_retrieval.py # Sentence-window chunking
├── scripts/                      # Utility scripts
│   ├── smart_ingest.py           # Smart ingestion with auto table detection
│   ├── direct_ingest.py          # Basic ingestion (no table detection)
│   ├── trigger_ingestion.py      # Celery-based ingestion
│   ├── evaluate.py               # RAG evaluation (RAGAs)
│   ├── check_metrics.py          # Performance gating
│   ├── check_task_status.py      # Celery task monitoring
│   └── shell/                    # Shell scripts
├── tests/                        # All test files
│   ├── test_architecture.py      # Architecture validation
│   ├── test_system_evaluation.py # System evaluation tests
│   └── ...
├── docs/                         # Documentation
│   ├── DEVELOPMENT.md            # This file
│   └── UI.png                    # Screenshot
├── requirements/                 # Split dependencies
│   ├── base.txt                  # Common dependencies
│   ├── api.txt                   # FastAPI service
│   ├── worker.txt                # Celery worker
│   ├── ui.txt                    # Streamlit UI
│   └── dev.txt                   # Development tools
├── data/                         # Document storage (gitignored)
├── docker-compose.yml            # Production Docker config
├── docker-compose.dev.yml        # Development Docker config
└── Dockerfile.*                  # Container definitions
```

---

## Development Workflow

### 1. Document Ingestion
```bash
# Direct ingestion (no Celery required)
make ingest

# Ingest specific file
make ingest-file FILE=data/handbook.pdf

# Check database status
make ingest-stats
```

### 2. Testing
```bash
# Run all tests
make test

# Architecture tests (config, logging, error handling)
make test-arch

# System evaluation (requires running API)
make up
make test-eval
```

### 3. Code Quality
```bash
# Lint code
make lint

# Format code
make format
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
- BM25 keyword search + Vector similarity with Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- Diversity filtering for result quality

**Features:**
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

---

## CI/CD Pipeline

### GitHub Actions Workflow

The project includes a comprehensive MLOps pipeline at `.github/workflows/mlops-pipeline.yml`.

**Pipeline triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

**Pipeline jobs:**

1. **code-quality**: Runs flake8, black, and isort checks
2. **unit-tests**: Executes pytest test suite
3. **rag-evaluation**: Runs RAGAs evaluation and enforces quality thresholds
4. **build-images**: Builds Docker images for api, worker, and ui services
5. **integration-tests**: Tests full system with Docker Compose (main branch only)
6. **security-scan**: Runs Trivy vulnerability scanner
7. **deployment-ready**: Final validation and deployment notification

### Setting Up CI/CD

**1. Add GitHub Secret:**
```bash
Repository Settings → Secrets and variables → Actions → New secret
Name: GROQ_API_KEY
Value: <your-groq-api-key>
```

**2. Enable GitHub Container Registry (optional for image publishing):**

By default, images are built but NOT pushed to registry.

To enable image publishing to GHCR:
- Settings → Actions → General → Workflow permissions
- Select "Read and write permissions"
- Update workflow file to set `push: true` in Docker build step
- See main README for detailed instructions

**3. Configure Branch Protection (recommended):**
```bash
Settings → Branches → Add rule
- Require status checks to pass
- Require pull request reviews
- Enable "rag-evaluation" and "code-quality" checks
```

### Local Testing Before Push

Run the same checks locally:

```bash
# Code quality
flake8 src/ scripts/
black --check src/ scripts/
isort --check-only src/ scripts/

# Tests
pytest tests/ -v

# RAG evaluation
python scripts/evaluate.py
python scripts/check_metrics.py

# Docker builds
docker compose -f docker-compose.yml build
```

### Monitoring Pipeline Results

- **Action runs**: GitHub Actions tab
- **Artifacts**: Download evaluation results (30-day retention)
- **Images**: `ghcr.io/<username>/rag-campus-chatbot-{api,worker,ui}`
- **Security**: Security tab → Code scanning alerts

### Quality Thresholds

Defined in `scripts/check_metrics.py`:
- Context Precision: ≥ 0.70
- Context Recall: ≥ 0.70  
- Faithfulness: ≥ 0.70
- Answer Relevancy: ≥ 0.70

Pipeline will **fail** if any metric is below threshold.
