#!/bin/bash
# dev-start.sh - One command to rule them all!

echo "🚀 Starting RAG Development Environment"
echo "======================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.example .env
    echo "📝 Please edit .env and add your GROQ_API_KEY"
    echo "   Then run this script again."
    exit 1
fi

# Build and start services
echo "🔨 Building containers..."
docker compose -f docker-compose.dev.yml build

echo "🚀 Starting all services..."
docker compose -f docker-compose.dev.yml up -d redis chroma worker

echo "⏳ Waiting for services to be ready..."
sleep 10

# Check if data needs ingestion
echo "🔍 Checking database status..."
CHUNK_COUNT=$(docker compose -f docker-compose.dev.yml exec -T backend python -c "
import chromadb
try:
    client = chromadb.PersistentClient(path='/app/chroma_db')
    collection = client.get_collection('collection')
    print(collection.count())
except:
    print('0')
" 2>/dev/null || echo "0")

echo "📊 Current chunks in database: $CHUNK_COUNT"

if [ "$CHUNK_COUNT" -lt "100" ]; then
    echo "📥 Database appears empty. Triggering enhanced ingestion..."
    docker compose -f docker-compose.dev.yml exec worker python -c "
from enhanced_ingestion_worker import process_document
task = process_document.delay('/app/data')
result = task.get(timeout=300)
print('✅ Ingestion result:', result)
    "
fi

echo ""
echo "✅ Development environment ready!"
echo "================================="
echo ""
echo "🖥️  TERMINAL 1 - Backend:"
echo "   docker compose -f docker-compose.dev.yml up backend"
echo ""
echo "🌐 TERMINAL 2 - Frontend:"
echo "   docker compose -f docker-compose.dev.yml up frontend"
echo ""
echo "📡 Access URLs:"
echo "   - Frontend UI: http://localhost:8501"
echo "   - Backend API: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo ""
echo "🛠️  Services running in background:"
echo "   - Redis: localhost:6379"
echo "   - ChromaDB: localhost:8001"
echo "   - Celery Worker: Enhanced OCR enabled"
echo ""
echo "🔧 Development commands:"
echo "   - View logs: docker compose -f docker-compose.dev.yml logs -f [service]"
echo "   - Stop all: docker compose -f docker-compose.dev.yml down"
echo "   - Rebuild: docker compose -f docker-compose.dev.yml build"
echo ""
