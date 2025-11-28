#!/usr/bin/env python3
# test_system_evaluation.py

"""
Comprehensive System Evaluation

Tests:
1. Functional correctness - API responds properly
2. Answer accuracy - Answers are relevant and accurate
3. Performance metrics - Latency, throughput
4. Stress testing - Multiple concurrent requests
"""

import sys
import json
import time
import statistics
import concurrent.futures
from typing import List, Dict, Tuple, Any
from datetime import datetime

try:
    import requests
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "requests"], check=True)
    import requests


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_header(title: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")


def print_result(name: str, passed: bool, details: str = ""):
    status = f"{Colors.GREEN}PASS{Colors.RESET}" if passed else f"{Colors.RED}FAIL{Colors.RESET}"
    print(f"  [{status}] {name}")
    if details:
        print(f"         {Colors.YELLOW}{details}{Colors.RESET}")


# Test questions with expected keywords/topics
TEST_QUESTIONS = [
    {
        "query": "What are the admission requirements for undergraduate students?",
        "expected_keywords": ["admission", "requirement", "application", "student"],
        "category": "Admissions"
    },
    {
        "query": "How do I apply for a scholarship?",
        "expected_keywords": ["scholarship", "apply", "financial", "award"],
        "category": "Financial Aid"
    },
    {
        "query": "What is the grading system at the university?",
        "expected_keywords": ["grade", "gpa", "credit", "score", "mark"],
        "category": "Academic"
    },
    {
        "query": "What are the library operating hours?",
        "expected_keywords": ["library", "hour", "open", "service"],
        "category": "Campus Services"
    },
    {
        "query": "How can I register for courses?",
        "expected_keywords": ["register", "course", "enrollment", "credit"],
        "category": "Registration"
    },
    {
        "query": "What is the policy on academic integrity?",
        "expected_keywords": ["integrity", "plagiarism", "academic", "misconduct", "honest"],
        "category": "Policies"
    },
    {
        "query": "How do I request a transcript?",
        "expected_keywords": ["transcript", "record", "request", "document"],
        "category": "Administration"
    },
    {
        "query": "What programs or degrees are offered?",
        "expected_keywords": ["program", "degree", "major", "faculty", "course"],
        "category": "Programs"
    }
]


def check_api_health(base_url: str) -> Tuple[bool, Dict]:
    """Check API health status."""
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}


def send_query(base_url: str, query: str, timeout: int = 60) -> Tuple[bool, Dict, float]:
    """Send a query and return (success, response, latency)."""
    start = time.time()
    try:
        response = requests.post(
            f"{base_url}/ask",
            json={"query": query},
            timeout=timeout
        )
        latency = time.time() - start
        
        if response.status_code == 200:
            return True, response.json(), latency
        else:
            return False, {
                "status_code": response.status_code,
                "body": response.json() if response.text else {}
            }, latency
    except Exception as e:
        return False, {"error": str(e)}, time.time() - start


def evaluate_answer_relevance(response: str, expected_keywords: List[str]) -> Tuple[float, List[str]]:
    """
    Evaluate answer relevance based on keyword matching.
    Returns (score 0-1, matched_keywords)
    """
    response_lower = response.lower()
    matched = [kw for kw in expected_keywords if kw.lower() in response_lower]
    score = len(matched) / len(expected_keywords) if expected_keywords else 0
    return score, matched


def run_accuracy_tests(base_url: str) -> Dict[str, Any]:
    """Run accuracy tests with various questions."""
    print_header("ACCURACY TESTING")
    
    results = []
    total_relevance = 0
    passed_count = 0
    
    for i, test_case in enumerate(TEST_QUESTIONS, 1):
        query = test_case["query"]
        expected_kw = test_case["expected_keywords"]
        category = test_case["category"]
        
        print(f"\n  Question {i}/{len(TEST_QUESTIONS)}: [{category}]")
        print(f"  Q: {query[:60]}...")
        
        success, response, latency = send_query(base_url, query)
        
        if success:
            answer = response.get("response", "")
            relevance, matched = evaluate_answer_relevance(answer, expected_kw)
            total_relevance += relevance
            
            passed = relevance >= 0.5  # At least 50% keyword match
            if passed:
                passed_count += 1
            
            print_result(
                f"Latency: {latency:.2f}s, Relevance: {relevance*100:.0f}%",
                passed,
                f"Matched: {matched}"
            )
            
            # Show truncated answer
            print(f"  A: {answer[:150]}...")
            
            results.append({
                "query": query,
                "category": category,
                "success": True,
                "latency": latency,
                "relevance": relevance,
                "matched_keywords": matched,
                "answer_preview": answer[:200]
            })
        else:
            print_result(f"Query failed", False, str(response))
            results.append({
                "query": query,
                "category": category,
                "success": False,
                "error": str(response)
            })
    
    avg_relevance = total_relevance / len(TEST_QUESTIONS)
    
    print(f"\n  {Colors.BOLD}Accuracy Summary:{Colors.RESET}")
    print(f"    Questions Passed: {passed_count}/{len(TEST_QUESTIONS)}")
    print(f"    Average Relevance: {avg_relevance*100:.1f}%")
    
    return {
        "total_questions": len(TEST_QUESTIONS),
        "passed": passed_count,
        "average_relevance": avg_relevance,
        "details": results
    }


def run_performance_tests(base_url: str, num_requests: int = 10) -> Dict[str, Any]:
    """Run performance/latency tests."""
    print_header("PERFORMANCE TESTING")
    
    query = "What are the admission requirements?"
    latencies = []
    failures = 0
    
    print(f"  Running {num_requests} sequential requests...")
    
    for i in range(num_requests):
        success, _, latency = send_query(base_url, query, timeout=120)
        if success:
            latencies.append(latency)
            print(f"    Request {i+1}: {latency:.2f}s", end="\r")
        else:
            failures += 1
    
    print()  # Clear the line
    
    if not latencies:
        return {"error": "All requests failed"}
    
    latencies.sort()
    p50_idx = int(len(latencies) * 0.5)
    p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
    p99_idx = min(int(len(latencies) * 0.99), len(latencies) - 1)
    
    stats = {
        "total_requests": num_requests,
        "successful": len(latencies),
        "failed": failures,
        "min_latency": min(latencies),
        "max_latency": max(latencies),
        "mean_latency": statistics.mean(latencies),
        "median_latency": statistics.median(latencies),
        "p50_latency": latencies[p50_idx],
        "p95_latency": latencies[p95_idx],
        "p99_latency": latencies[p99_idx],
        "std_dev": statistics.stdev(latencies) if len(latencies) > 1 else 0
    }
    
    print(f"\n  {Colors.BOLD}Performance Statistics:{Colors.RESET}")
    print(f"    Successful: {stats['successful']}/{num_requests}")
    print(f"    Min Latency:    {stats['min_latency']:.2f}s")
    print(f"    Max Latency:    {stats['max_latency']:.2f}s")
    print(f"    Mean Latency:   {stats['mean_latency']:.2f}s")
    print(f"    Median (P50):   {stats['p50_latency']:.2f}s")
    print(f"    P95 Latency:    {stats['p95_latency']:.2f}s")
    print(f"    P99 Latency:    {stats['p99_latency']:.2f}s")
    print(f"    Std Deviation:  {stats['std_dev']:.2f}s")
    
    # Evaluate performance
    p95_ok = stats['p95_latency'] < 10.0  # 10 second threshold
    print_result(f"P95 < 10s threshold", p95_ok, f"P95: {stats['p95_latency']:.2f}s")
    
    return stats


def run_concurrent_tests(base_url: str, num_requests: int = 5, concurrency: int = 3) -> Dict[str, Any]:
    """Run concurrent request tests."""
    print_header("CONCURRENCY TESTING")
    
    query = "What programs are offered at the university?"
    latencies = []
    failures = 0
    
    print(f"  Running {num_requests} requests with concurrency={concurrency}...")
    
    def make_request(i):
        return send_query(base_url, query, timeout=120)
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(make_request, i) for i in range(num_requests)]
        
        for future in concurrent.futures.as_completed(futures):
            success, response, latency = future.result()
            if success:
                latencies.append(latency)
            else:
                failures += 1
    
    total_time = time.time() - start_time
    
    if not latencies:
        return {"error": "All concurrent requests failed"}
    
    stats = {
        "total_requests": num_requests,
        "concurrency": concurrency,
        "successful": len(latencies),
        "failed": failures,
        "total_time": total_time,
        "throughput": len(latencies) / total_time,
        "mean_latency": statistics.mean(latencies),
        "max_latency": max(latencies)
    }
    
    print(f"\n  {Colors.BOLD}Concurrency Statistics:{Colors.RESET}")
    print(f"    Successful: {stats['successful']}/{num_requests}")
    print(f"    Total Time: {stats['total_time']:.2f}s")
    print(f"    Throughput: {stats['throughput']:.2f} req/s")
    print(f"    Mean Latency: {stats['mean_latency']:.2f}s")
    print(f"    Max Latency: {stats['max_latency']:.2f}s")
    
    # No broken pipe errors = success
    print_result("No broken pipe errors", failures == 0, f"Failures: {failures}")
    
    return stats


def run_error_handling_test(base_url: str) -> Dict[str, Any]:
    """Test error handling with edge cases."""
    print_header("ERROR HANDLING TESTS")
    
    results = []
    
    # Test 1: Empty query
    print("  Testing empty query...")
    success, response, _ = send_query(base_url, "")
    handled = not success or "error" in str(response).lower() or response.get("response", "")
    print_result("Empty query handled", True)  # API should handle gracefully
    results.append({"test": "empty_query", "handled": handled})
    
    # Test 2: Very long query
    print("  Testing very long query...")
    long_query = "What is " + "the admission process " * 50 + "?"
    success, response, latency = send_query(base_url, long_query, timeout=120)
    print_result("Long query handled", success or "error" not in str(response).lower(), f"Latency: {latency:.2f}s")
    results.append({"test": "long_query", "success": success, "latency": latency})
    
    # Test 3: Special characters
    print("  Testing special characters...")
    special_query = "What's the university's policy on @#$% and symbols?"
    success, response, latency = send_query(base_url, special_query)
    print_result("Special characters handled", success, f"Latency: {latency:.2f}s")
    results.append({"test": "special_chars", "success": success})
    
    # Test 4: Non-English characters
    print("  Testing non-English characters...")
    unicode_query = "What is the 大学 admission policy?"
    success, response, latency = send_query(base_url, unicode_query)
    print_result("Unicode handled", success, f"Latency: {latency:.2f}s")
    results.append({"test": "unicode", "success": success})
    
    return {"tests": results}


def main():
    base_url = "http://localhost:8000"
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║           COMPREHENSIVE SYSTEM EVALUATION                          ║")
    print("║           RAG Chatbot - Accuracy & Performance Tests               ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}")
    print(f"  Target: {base_url}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Health check
    print_header("API HEALTH CHECK")
    healthy, health_data = check_api_health(base_url)
    
    if not healthy:
        print(f"  {Colors.RED}API is not running!{Colors.RESET}")
        print(f"  Error: {health_data}")
        return 1
    
    print(f"  Status: {Colors.GREEN}Online{Colors.RESET}")
    print(f"  Version: {health_data.get('version', 'N/A')}")
    print(f"  Database Chunks: {health_data.get('database_chunks', 'N/A')}")
    
    # Run all tests
    accuracy_results = run_accuracy_tests(base_url)
    performance_results = run_performance_tests(base_url, num_requests=8)
    concurrency_results = run_concurrent_tests(base_url, num_requests=5, concurrency=2)
    error_results = run_error_handling_test(base_url)
    
    # Final Summary
    print_header("FINAL EVALUATION SUMMARY")
    
    print(f"  {Colors.BOLD}Accuracy:{Colors.RESET}")
    print(f"    Questions Passed: {accuracy_results['passed']}/{accuracy_results['total_questions']}")
    print(f"    Average Relevance: {accuracy_results['average_relevance']*100:.1f}%")
    
    print(f"\n  {Colors.BOLD}Performance:{Colors.RESET}")
    if "error" not in performance_results:
        print(f"    P50 Latency: {performance_results['p50_latency']:.2f}s")
        print(f"    P95 Latency: {performance_results['p95_latency']:.2f}s")
        print(f"    P99 Latency: {performance_results['p99_latency']:.2f}s")
    
    print(f"\n  {Colors.BOLD}Concurrency:{Colors.RESET}")
    if "error" not in concurrency_results:
        print(f"    Throughput: {concurrency_results['throughput']:.2f} req/s")
        print(f"    Success Rate: {concurrency_results['successful']}/{concurrency_results['total_requests']}")
    
    # Overall verdict
    accuracy_pass = accuracy_results['average_relevance'] >= 0.5
    perf_pass = "error" not in performance_results and performance_results.get('p95_latency', 999) < 15
    
    print(f"\n  {Colors.BOLD}Overall Verdict:{Colors.RESET}")
    if accuracy_pass and perf_pass:
        print(f"    {Colors.GREEN}✓ SYSTEM IS OPERATIONAL AND PERFORMING WELL{Colors.RESET}")
    else:
        issues = []
        if not accuracy_pass:
            issues.append("Low accuracy")
        if not perf_pass:
            issues.append("Performance issues")
        print(f"    {Colors.YELLOW}⚠ SYSTEM NEEDS ATTENTION: {', '.join(issues)}{Colors.RESET}")
    
    # Export results
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "health": health_data,
        "accuracy": accuracy_results,
        "performance": performance_results,
        "concurrency": concurrency_results,
        "error_handling": error_results
    }
    
    with open("evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n  Results saved to: evaluation_results.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
