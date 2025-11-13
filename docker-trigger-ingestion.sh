#!/bin/bash
# docker-trigger-ingestion.sh
# 
# Trigger document ingestion in the Dockerized environment
# Usage: ./docker-trigger-ingestion.sh <path_to_document_or_folder>

if [ -z "$1" ]; then
    echo "Usage: ./docker-trigger-ingestion.sh <path_to_document_or_folder>"
    echo "Example: ./docker-trigger-ingestion.sh data/"
    exit 1
fi

echo "🚀 Triggering ingestion for: $1"
echo "📋 Running in Docker worker container..."

docker compose exec worker python -c "
from ingestion_worker import process_document
import sys

task = process_document.delay('$1')
print(f'Task ID: {task.id}')
print('Waiting for task to complete...')

try:
    result = task.get(timeout=3600)
    print('\n' + '='*60)
    print('✅ INGESTION COMPLETE')
    print('='*60)
    print(f'Status: {result.get(\"status\")}')
    print(f'Message: {result.get(\"message\")}')
    print(f'Documents processed: {result.get(\"documents_processed\", \"N/A\")}')
    print(f'Chunks added: {result.get(\"chunks_added\", 0)}')
    print(f'Total chunks in DB: {result.get(\"total_chunks_in_db\", \"N/A\")}')
    print('='*60)
except Exception as e:
    print(f'\n❌ Error: {e}')
    sys.exit(1)
"
