@echo off
REM dev-start.bat - Windows version of dev-start.sh

echo 🚀 Starting RAG Development Environment
echo ======================================

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

REM Check if .env exists
if not exist .env (
    echo ⚠️  .env file not found. Creating from template...
    copy .env.example .env
    echo 📝 Please edit .env and add your GROQ_API_KEY
    echo    Then run this script again.
    pause
    exit /b 1
)

REM Build and start services
echo 🔨 Building containers...
docker compose -f docker-compose.dev.yml build

echo 🚀 Starting all services...
docker compose -f docker-compose.dev.yml up -d redis chroma worker

echo ⏳ Waiting for services to be ready...
timeout /t 10 /nobreak >nul

echo 📊 Checking database status...
docker compose -f docker-compose.dev.yml exec -T backend python -c "import chromadb; client = chromadb.PersistentClient(path='/app/chroma_db'); print('Chunks:', client.get_collection('collection').count() if 'collection' in [c.name for c in client.list_collections()] else 0)" 2>nul

echo.
echo ✅ Development environment ready!
echo =================================
echo.
echo 🖥️  TERMINAL 1 - Backend:
echo    docker compose -f docker-compose.dev.yml up backend
echo.
echo 🌐 TERMINAL 2 - Frontend:
echo    docker compose -f docker-compose.dev.yml up frontend
echo.
echo 📡 Access URLs:
echo    - Frontend UI: http://localhost:8501
echo    - Backend API: http://localhost:8000
echo    - API Docs: http://localhost:8000/docs
echo.
echo 🛠️  Services running in background:
echo    - Redis: localhost:6379
echo    - ChromaDB: localhost:8001
echo    - Celery Worker: Enhanced OCR enabled
echo.
echo 🔧 Development commands:
echo    - View logs: docker compose -f docker-compose.dev.yml logs -f [service]
echo    - Stop all: docker compose -f docker-compose.dev.yml down
echo    - Rebuild: docker compose -f docker-compose.dev.yml build
echo.
pause
