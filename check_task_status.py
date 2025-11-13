#!/usr/bin/env python3
# check_task_status.py

"""
Task Status Checker

Check the status of a Celery task by its ID.

Usage:
    python check_task_status.py <task_id>
"""

import sys
from celery.result import AsyncResult
from celery_config import celery_app


def main():
    if len(sys.argv) != 2:
        print("Usage: python check_task_status.py <task_id>")
        sys.exit(1)

    task_id = sys.argv[1]
    result = AsyncResult(task_id, app=celery_app)

    print(f"\n{'='*60}")
    print(f"Task ID: {task_id}")
    print(f"{'='*60}")
    print(f"Status: {result.state}")

    if result.state == 'PENDING':
        print("Task is waiting in queue or doesn't exist")
    elif result.state == 'STARTED':
        print("Task is currently running")
    elif result.state == 'SUCCESS':
        print(f"Task completed successfully")
        print(f"\nResult:")
        print(result.result)
    elif result.state == 'FAILURE':
        print(f"Task failed")
        print(f"\nError:")
        print(result.result)
    else:
        print(f"Task state: {result.state}")
        if result.info:
            print(f"Info: {result.info}")

    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
