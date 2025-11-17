#!/usr/bin/env python3
# trigger_ingestion.py

"""
Document Ingestion Trigger

This script dispatches document processing jobs to the Celery task queue.

Usage:
    python trigger_ingestion.py <path_to_document_or_folder>
    python trigger_ingestion.py data/
    python trigger_ingestion.py --clear  # Clear the collection
    python trigger_ingestion.py --stats  # Get collection statistics
"""

import sys
import os
import argparse
from celery.result import AsyncResult

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from ingestion_worker import process_document, clear_collection, get_collection_stats
from celery_config import celery_app


def main():
    parser = argparse.ArgumentParser(
        description='Trigger document ingestion tasks'
    )
    parser.add_argument(
        'path',
        nargs='?',
        help='Path to document or folder to ingest'
    )
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear all documents from the collection'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Get collection statistics'
    )
    parser.add_argument(
        '--async',
        dest='async_mode',
        action='store_true',
        help='Run in async mode (don\'t wait for result)'
    )

    args = parser.parse_args()

    # Handle clear operation
    if args.clear:
        print("🗑️  Clearing collection...")
        task = clear_collection.delay()
        
        if not args.async_mode:
            result = task.get(timeout=300)
            print(f" Result: {result}")
        else:
            print(f"📋 Task ID: {task.id}")
            print("Task dispatched. Check status with task ID.")
        return

    # Handle stats operation
    if args.stats:
        print(" Fetching collection statistics...")
        task = get_collection_stats.delay()
        
        if not args.async_mode:
            result = task.get(timeout=30)
            print(f" Result: {result}")
        else:
            print(f"📋 Task ID: {task.id}")
        return

    # Handle ingestion operation
    if not args.path:
        parser.print_help()
        sys.exit(1)

    print(f"📄 Dispatching ingestion task for: {args.path}")
    task = process_document.delay(args.path)
    print(f"📋 Task ID: {task.id}")

    if not args.async_mode:
        print("⏳ Waiting for task to complete...")
        print("(This may take a while for large documents)")
        
        try:
            result = task.get(timeout=3600)  # 1 hour timeout
            
            print("\n" + "="*60)
            print(" INGESTION COMPLETE")
            print("="*60)
            print(f"Status: {result.get('status')}")
            print(f"Message: {result.get('message')}")
            print(f"Documents processed: {result.get('documents_processed', 'N/A')}")
            print(f"Windows added: {result.get('windows_added', 0)}")
            print(f"Total characters: {result.get('total_characters', 'N/A'):,}")
            print(f"Total windows in DB: {result.get('total_windows_in_db', 'N/A')}")
            print(f"Technique: {result.get('technique', 'N/A')}")
            print("="*60)
            
        except Exception as e:
            print(f"\n Error waiting for task: {e}")
            print(f"Task ID {task.id} may still be running.")
            print("Check worker logs for details.")
    else:
        print("Task dispatched in async mode.")
        print("Check task status with:")
        print(f"  python check_task_status.py {task.id}")


if __name__ == '__main__':
    main()
