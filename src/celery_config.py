# celery_config.py

"""
Celery Configuration

This module configures the Celery task queue system for asynchronous
document ingestion. It uses Redis as the message broker.

Configuration is loaded from the centralized config module.
"""

from celery import Celery

from config import get_settings

# Load validated configuration
settings = get_settings()

# Redis URL from centralized config
REDIS_URL = settings.redis_url

# Initialize Celery application
celery_app = Celery(
    'rag_chatbot',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=['ingestion_worker']
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour timeout
    task_soft_time_limit=3300,  # 55 minutes soft timeout
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
    # macOS compatibility: use solo pool to avoid forking issues with ML libraries
    worker_pool='solo',
)

if __name__ == '__main__':
    celery_app.start()
