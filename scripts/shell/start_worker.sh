#!/bin/bash
# start_worker.sh - Start Celery worker with proper configuration

echo "🛑 Stopping any existing workers..."
pkill -9 -f "celery.*ingestion_worker" 2>/dev/null
sleep 1

echo " Starting Celery worker with solo pool (macOS compatible)..."
echo "Press Ctrl+C to stop the worker"
echo ""

# Activate venv if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "  Virtual environment not activated!"
    echo "Run: source venv/bin/activate"
    exit 1
fi

# Start worker
celery -A ingestion_worker worker --loglevel=info --pool=solo
