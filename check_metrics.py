#!/usr/bin/env python3
# check_metrics.py

"""
Model Performance Gate

This script reads RAGAs evaluation results and enforces quality thresholds.
It's designed to fail CI/CD pipelines if model performance degrades below
acceptable levels, preventing bad models from reaching production.

Thresholds:
- Context Precision: >= 0.70
- Context Recall: >= 0.70
- Faithfulness: >= 0.70
- Answer Relevancy: >= 0.70
"""

import sys
import os
import pandas as pd
from pathlib import Path


# Performance thresholds
THRESHOLDS = {
    'context_precision': 0.70,
    'context_recall': 0.70,
    'faithfulness': 0.70,
    'answer_relevancy': 0.70
}


def check_metrics(csv_path: str) -> bool:
    """
    Check if metrics meet minimum thresholds.
    
    Args:
        csv_path: Path to the evaluation results CSV
        
    Returns:
        True if all metrics pass, False otherwise
    """
    if not os.path.exists(csv_path):
        print(f"❌ Error: Evaluation results not found at {csv_path}")
        return False
    
    print(f"📊 Reading evaluation results from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*70)
    print("MODEL PERFORMANCE EVALUATION")
    print("="*70)
    
    all_passed = True
    
    for metric, threshold in THRESHOLDS.items():
        if metric in df.columns:
            # Get mean value for the metric
            mean_value = df[metric].mean()
            passed = mean_value >= threshold
            status = "✅ PASS" if passed else "❌ FAIL"
            
            print(f"{metric:20s}: {mean_value:.4f} (threshold: {threshold:.2f}) {status}")
            
            if not passed:
                all_passed = False
        else:
            print(f"{metric:20s}: ⚠️  NOT FOUND IN RESULTS")
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n✅ All metrics passed! Model performance is acceptable.")
        return True
    else:
        print("\n❌ Some metrics failed! Model performance is below threshold.")
        print("Please improve the model before deploying to production.")
        return False


def main():
    # Look for the most recent evaluation results
    results_dir = Path("evaluation_results")
    
    if not results_dir.exists():
        print("❌ Error: evaluation_results directory not found")
        print("Please run evaluate.py first to generate evaluation results.")
        sys.exit(1)
    
    # Find the most recent CSV file
    csv_files = list(results_dir.glob("ragas_results_*.csv"))
    
    if not csv_files:
        print("❌ Error: No evaluation results found in evaluation_results/")
        print("Please run evaluate.py first to generate evaluation results.")
        sys.exit(1)
    
    # Get the most recent file
    latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
    
    print(f"📄 Using evaluation results: {latest_csv}")
    
    # Check metrics
    if check_metrics(str(latest_csv)):
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == '__main__':
    main()
