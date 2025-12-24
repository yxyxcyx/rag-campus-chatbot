#!/usr/bin/env python3
# trigger_table_ingestion.py

"""
Trigger Table-Aware Ingestion via Celery

This script triggers the table-aware ingestion task through Celery,
which is the recommended way to ingest documents in production.

Usage:
    python scripts/trigger_table_ingestion.py [data_folder]
    
    Options:
        --clear    Clear existing data before ingestion
        --no-version    Disable document versioning
"""

import os
import sys
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def trigger_ingestion(data_folder: str, clear_first: bool = False, use_versioning: bool = True):
    """Trigger table-aware ingestion via Celery."""
    from celery_config import celery_app
    
    print(f"Triggering table-aware ingestion for: {data_folder}")
    print(f"  Clear first: {clear_first}")
    print(f"  Use versioning: {use_versioning}")
    
    # Clear if requested
    if clear_first:
        print("\nClearing existing collection...")
        clear_task = celery_app.send_task(
            'ingestion_worker.clear_collection'
        )
        result = clear_task.get(timeout=60)
        print(f"Clear result: {result}")
    
    # Trigger table-aware ingestion
    print("\nTriggering table-aware ingestion...")
    task = celery_app.send_task(
        'ingestion_worker.process_with_tables',
        args=[data_folder],
        kwargs={
            'window_size': 3,
            'use_versioning': use_versioning
        }
    )
    
    print(f"Task ID: {task.id}")
    print("Waiting for completion...")
    
    result = task.get(timeout=600)  # 10 minute timeout
    
    print("\n" + "="*50)
    print("INGESTION RESULT")
    print("="*50)
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    return result


def check_status(task_id: str):
    """Check the status of an ingestion task."""
    from celery_config import celery_app
    
    result = celery_app.AsyncResult(task_id)
    
    print(f"Task ID: {task_id}")
    print(f"Status: {result.status}")
    
    if result.ready():
        print(f"Result: {result.result}")
    else:
        print("Task is still running...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trigger table-aware document ingestion")
    parser.add_argument("data_folder", nargs="?", default="data/", 
                        help="Path to folder containing documents")
    parser.add_argument("--clear", action="store_true",
                        help="Clear existing data before ingestion")
    parser.add_argument("--no-version", action="store_true",
                        help="Disable document versioning")
    parser.add_argument("--status", metavar="TASK_ID",
                        help="Check status of a task instead of triggering")
    
    args = parser.parse_args()
    
    if args.status:
        check_status(args.status)
    else:
        # Resolve data folder path
        data_folder = args.data_folder
        if not os.path.isabs(data_folder):
            # Try relative to script location first
            script_dir = os.path.dirname(__file__)
            relative_path = os.path.join(script_dir, '..', data_folder)
            if os.path.exists(relative_path):
                data_folder = os.path.abspath(relative_path)
        
        if not os.path.exists(data_folder):
            print(f"Error: Data folder not found: {data_folder}")
            sys.exit(1)
        
        trigger_ingestion(
            data_folder,
            clear_first=args.clear,
            use_versioning=not args.no_version
        )
