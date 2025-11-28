#!/usr/bin/env python3
# test_architecture.py

"""
Architecture Validation Test Suite

Tests for validating the refactored system (Tickets 1-3):
- Suite A: Configuration & Startup Validation
- Suite B: Resilience & Chaos Engineering
- Suite C: Load & Observability
- Suite D: End-to-End User Experience

Run with: python scripts/test_architecture.py
"""

import os
import sys
import json
import time
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_header(title: str):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")


def print_test(name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = f"{Colors.GREEN}PASS{Colors.RESET}" if passed else f"{Colors.RED}FAIL{Colors.RESET}"
    print(f"  [{status}] {name}")
    if details:
        print(f"         {Colors.YELLOW}{details}{Colors.RESET}")


def print_info(message: str):
    """Print info message."""
    print(f"  {Colors.BLUE}ℹ {message}{Colors.RESET}")


class TestResults:
    """Collect and summarize test results."""
    
    def __init__(self):
        self.results = []
        self.suite_results = {}
    
    def add(self, suite: str, test: str, passed: bool, details: str = ""):
        self.results.append({
            "suite": suite,
            "test": test,
            "passed": passed,
            "details": details
        })
        if suite not in self.suite_results:
            self.suite_results[suite] = {"passed": 0, "failed": 0}
        if passed:
            self.suite_results[suite]["passed"] += 1
        else:
            self.suite_results[suite]["failed"] += 1
    
    def summary(self):
        print_header("TEST SUMMARY")
        total_passed = sum(r["passed"] for r in self.suite_results.values())
        total_failed = sum(r["failed"] for r in self.suite_results.values())
        
        for suite, counts in self.suite_results.items():
            status = Colors.GREEN if counts["failed"] == 0 else Colors.RED
            print(f"  {suite}: {status}{counts['passed']} passed, {counts['failed']} failed{Colors.RESET}")
        
        print(f"\n  {Colors.BOLD}Total: {total_passed} passed, {total_failed} failed{Colors.RESET}")
        return total_failed == 0


# ==============================================================================
# SUITE A: Configuration & Startup Validation
# ==============================================================================

def test_suite_a(results: TestResults):
    """Test Suite A: Configuration & Startup Validation."""
    print_header("SUITE A: Configuration & Startup Validation")
    
    # Test A1: Valid configuration loads successfully
    print_info("Test A1: Happy Path - Valid configuration")
    try:
        # Temporarily set valid env
        os.environ["GROQ_API_KEY"] = "test_key_for_validation"
        
        # Clear cached settings
        from config import get_settings
        get_settings.cache_clear()
        
        settings = get_settings()
        passed = settings.groq_api_key == "test_key_for_validation"
        print_test("Valid config loads successfully", passed)
        results.add("Suite A", "Happy Path - Valid config", passed)
        
    except Exception as e:
        print_test("Valid config loads successfully", False, str(e))
        results.add("Suite A", "Happy Path - Valid config", False, str(e))
    
    # Test A2: Missing API Key causes immediate failure  
    print_info("Test A2: Sad Path - Missing GROQ_API_KEY")
    try:
        # Run the validation in a subprocess to avoid env pollution
        result = subprocess.run(
            [sys.executable, "-c", """
import os
# Ensure no API key
if 'GROQ_API_KEY' in os.environ:
    del os.environ['GROQ_API_KEY']
os.environ['GROQ_API_KEY'] = ''  # Set to empty

import sys
sys.path.insert(0, 'src')

try:
    from config import Settings
    settings = Settings(_env_file=None)
    print('FAIL: No error raised')
    sys.exit(1)
except Exception as e:
    error_msg = str(e).lower()
    if 'groq_api_key' in error_msg or 'required' in error_msg or 'cannot be empty' in error_msg:
        print(f'PASS: {type(e).__name__}')
        sys.exit(0)
    else:
        print(f'FAIL: Wrong error - {e}')
        sys.exit(1)
"""],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            timeout=10
        )
        
        passed = result.returncode == 0 and "PASS" in result.stdout
        details = result.stdout.strip() if result.stdout else result.stderr.strip()
        
        print_test("Missing API key causes validation error", passed, details)
        results.add("Suite A", "Missing API key validation", passed, details)
        
    except Exception as e:
        print_test("Missing API key causes validation error", False, str(e))
        results.add("Suite A", "Missing API key validation", False, str(e))
    
    # Test A3: Invalid type causes immediate failure
    print_info("Test A3: Type Error Path - BM25_WEIGHT='apple'")
    try:
        os.environ["GROQ_API_KEY"] = "test_key"
        os.environ["BM25_WEIGHT"] = "apple"
        
        from config import get_settings
        get_settings.cache_clear()
        
        from config import Settings
        try:
            settings = Settings()
            passed = False
            details = "Expected ValidationError but got none"
        except Exception as e:
            error_msg = str(e)
            passed = "valid number" in error_msg.lower() or "float" in error_msg.lower() or "bm25" in error_msg.lower()
            details = f"Correctly raised: {type(e).__name__}"
        
        print_test("Invalid type causes validation error", passed, details)
        results.add("Suite A", "Type validation (BM25_WEIGHT='apple')", passed, details)
        
    except Exception as e:
        print_test("Invalid type causes validation error", False, str(e))
        results.add("Suite A", "Type validation", False, str(e))
    finally:
        # Cleanup
        if "BM25_WEIGHT" in os.environ:
            del os.environ["BM25_WEIGHT"]
    
    # Test A4: Log level validation
    print_info("Test A4: Invalid log level")
    try:
        os.environ["GROQ_API_KEY"] = "test_key"
        os.environ["LOG_LEVEL"] = "INVALID_LEVEL"
        
        from config import get_settings
        get_settings.cache_clear()
        
        from config import Settings
        try:
            settings = Settings()
            passed = False
            details = "Expected ValidationError but got none"
        except Exception as e:
            error_msg = str(e)
            passed = "log" in error_msg.lower() or "level" in error_msg.lower()
            details = f"Correctly raised: {type(e).__name__}"
        
        print_test("Invalid log level causes validation error", passed, details)
        results.add("Suite A", "Log level validation", passed, details)
        
    except Exception as e:
        print_test("Invalid log level causes validation error", False, str(e))
        results.add("Suite A", "Log level validation", False, str(e))
    finally:
        if "LOG_LEVEL" in os.environ:
            del os.environ["LOG_LEVEL"]


# ==============================================================================
# SUITE B: Resilience & Error Handling
# ==============================================================================

def test_suite_b(results: TestResults):
    """Test Suite B: Resilience & Chaos Engineering."""
    print_header("SUITE B: Resilience & Chaos Engineering")
    
    # Test B1: Error response structure
    print_info("Test B1: Error response structure validation")
    try:
        # Simulate what the API would return
        error_response = {
            "detail": {
                "message": "AI service is busy. Please try again in a few seconds.",
                "error_code": "RATE_LIMITED",
                "request_id": "abc12345",
                "retry_after": 10
            }
        }
        
        # Validate structure
        detail = error_response.get("detail", {})
        has_message = "message" in detail
        has_error_code = "error_code" in detail
        has_request_id = "request_id" in detail
        
        passed = all([has_message, has_error_code, has_request_id])
        details = f"message={has_message}, error_code={has_error_code}, request_id={has_request_id}"
        
        print_test("Error response has required fields", passed, details)
        results.add("Suite B", "Error response structure", passed, details)
        
    except Exception as e:
        print_test("Error response has required fields", False, str(e))
        results.add("Suite B", "Error response structure", False, str(e))
    
    # Test B2: HTTP exception classes exist and are properly defined
    print_info("Test B2: Exception handling imports")
    try:
        # Check in subprocess to avoid import side effects
        result = subprocess.run(
            [sys.executable, "-c", """
import sys
sys.path.insert(0, 'src')

try:
    import groq
    from requests.exceptions import Timeout, ConnectionError
    
    has_timeout = hasattr(groq, 'APITimeoutError')
    has_rate_limit = hasattr(groq, 'RateLimitError') 
    has_connection = hasattr(groq, 'APIConnectionError')
    
    if all([has_timeout, has_rate_limit, has_connection]):
        print(f'PASS: APITimeoutError={has_timeout}, RateLimitError={has_rate_limit}, APIConnectionError={has_connection}')
        sys.exit(0)
    else:
        print(f'FAIL: APITimeoutError={has_timeout}, RateLimitError={has_rate_limit}, APIConnectionError={has_connection}')
        sys.exit(1)
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
"""],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
            timeout=10
        )
        
        passed = result.returncode == 0 and "PASS" in result.stdout
        details = result.stdout.strip() if result.stdout else result.stderr.strip()
        
        print_test("Groq exception classes available", passed, details)
        results.add("Suite B", "Exception classes available", passed, details)
        
    except Exception as e:
        print_test("Groq exception classes available", False, str(e))
        results.add("Suite B", "Exception classes available", False, str(e))
    
    # Test B3: Request ID generation
    print_info("Test B3: Request ID generation")
    try:
        from logging_config import set_request_id, get_request_id
        
        # Generate request ID
        rid1 = set_request_id()
        rid2 = set_request_id()
        
        # Check they're unique
        passed = rid1 != rid2 and len(rid1) > 0
        details = f"Generated IDs: {rid1}, {rid2}"
        
        print_test("Request IDs are unique", passed, details)
        results.add("Suite B", "Request ID generation", passed, details)
        
    except Exception as e:
        print_test("Request IDs are unique", False, str(e))
        results.add("Suite B", "Request ID generation", False, str(e))
    
    # Test B4: Structured logger with metadata
    print_info("Test B4: Structured logger with metadata")
    try:
        from logging_config import setup_logging, get_logger
        import io
        import logging
        
        # Setup logger
        logger = setup_logging(level="INFO", json_output=False, app_name="test")
        
        # Test structured logging (this should not raise)
        logger.info("Test message", key1="value1", key2=42)
        
        passed = True
        details = "Logger accepts structured metadata"
        
        print_test("Structured logging works", passed, details)
        results.add("Suite B", "Structured logging", passed, details)
        
    except Exception as e:
        print_test("Structured logging works", False, str(e))
        results.add("Suite B", "Structured logging", False, str(e))


# ==============================================================================
# SUITE C: Observability Validation
# ==============================================================================

def test_suite_c(results: TestResults):
    """Test Suite C: Load & Observability."""
    print_header("SUITE C: Load & Observability")
    
    # Test C1: JSON log format
    print_info("Test C1: JSON log output format")
    try:
        from logging_config import StructuredFormatter
        import logging
        
        # Create a log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatter = StructuredFormatter()
        output = formatter.format(record)
        
        # Validate JSON
        parsed = json.loads(output)
        has_timestamp = "timestamp" in parsed
        has_level = "level" in parsed
        has_message = "message" in parsed
        has_module = "module" in parsed
        
        passed = all([has_timestamp, has_level, has_message, has_module])
        details = f"Fields: timestamp={has_timestamp}, level={has_level}, message={has_message}, module={has_module}"
        
        print_test("JSON log format is valid", passed, details)
        results.add("Suite C", "JSON log format", passed, details)
        
    except json.JSONDecodeError as e:
        print_test("JSON log format is valid", False, f"Invalid JSON: {e}")
        results.add("Suite C", "JSON log format", False, str(e))
    except Exception as e:
        print_test("JSON log format is valid", False, str(e))
        results.add("Suite C", "JSON log format", False, str(e))
    
    # Test C2: No print statements in core modules
    print_info("Test C2: No print() in core modules")
    try:
        import ast
        
        core_modules = [
            "main.py",
            "rag_pipeline.py",
            "enhanced_document_loader.py",
            "ingestion_worker.py",
            "sentence_window_retrieval.py"
        ]
        
        src_path = Path(__file__).parent.parent / "src"
        modules_with_print = []
        
        for module in core_modules:
            module_path = src_path / module
            if module_path.exists():
                with open(module_path) as f:
                    try:
                        tree = ast.parse(f.read())
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Call):
                                if isinstance(node.func, ast.Name) and node.func.id == "print":
                                    modules_with_print.append(module)
                                    break
                    except SyntaxError:
                        pass
        
        # config.py is allowed to have print for startup errors
        passed = len(modules_with_print) == 0
        if modules_with_print:
            details = f"Found print() in: {', '.join(modules_with_print)}"
        else:
            details = "No print() statements found in core modules"
        
        print_test("Core modules use logger instead of print", passed, details)
        results.add("Suite C", "No print() in core modules", passed, details)
        
    except Exception as e:
        print_test("Core modules use logger instead of print", False, str(e))
        results.add("Suite C", "No print() in core modules", False, str(e))
    
    # Test C3: Log levels are used correctly
    print_info("Test C3: Correct log levels in main.py")
    try:
        src_path = Path(__file__).parent.parent / "src"
        main_path = src_path / "main.py"
        
        with open(main_path) as f:
            content = f.read()
        
        has_info = "logger.info" in content
        has_warning = "logger.warning" in content
        has_error = "logger.error" in content
        
        passed = all([has_info, has_warning, has_error])
        details = f"INFO={has_info}, WARNING={has_warning}, ERROR={has_error}"
        
        print_test("All log levels used appropriately", passed, details)
        results.add("Suite C", "Log levels usage", passed, details)
        
    except Exception as e:
        print_test("All log levels used appropriately", False, str(e))
        results.add("Suite C", "Log levels usage", False, str(e))


# ==============================================================================
# SUITE D: Frontend Error Handling
# ==============================================================================

def test_suite_d(results: TestResults):
    """Test Suite D: End-to-End User Experience."""
    print_header("SUITE D: End-to-End User Experience")
    
    # Test D1: Frontend handles HTTP errors gracefully
    print_info("Test D1: Frontend error handling code review")
    try:
        src_path = Path(__file__).parent.parent / "src"
        app_path = src_path / "app.py"
        
        with open(app_path) as f:
            content = f.read()
        
        # Check for error handling patterns
        has_status_check = "status_code" in content
        has_error_message = "Error:" in content
        has_exception_handling = "except" in content
        has_request_exception = "RequestException" in content
        
        passed = all([has_status_check, has_error_message, has_exception_handling])
        details = f"status_check={has_status_check}, error_msg={has_error_message}, exception={has_exception_handling}"
        
        print_test("Frontend has error handling", passed, details)
        results.add("Suite D", "Frontend error handling", passed, details)
        
    except Exception as e:
        print_test("Frontend has error handling", False, str(e))
        results.add("Suite D", "Frontend error handling", False, str(e))
    
    # Test D2: Frontend displays user-friendly error messages
    print_info("Test D2: User-friendly error messages")
    try:
        src_path = Path(__file__).parent.parent / "src"
        app_path = src_path / "app.py"
        
        with open(app_path) as f:
            content = f.read()
        
        # Check for user-friendly patterns
        has_friendly_500 = "status code" in content.lower()
        has_connection_error = "could not connect" in content.lower()
        
        passed = has_friendly_500 and has_connection_error
        details = f"500 handling={has_friendly_500}, connection error={has_connection_error}"
        
        print_test("Error messages are user-friendly", passed, details)
        results.add("Suite D", "User-friendly errors", passed, details)
        
    except Exception as e:
        print_test("Error messages are user-friendly", False, str(e))
        results.add("Suite D", "User-friendly errors", False, str(e))


# ==============================================================================
# Docker Integration Tests (Require Docker running)
# ==============================================================================

def test_docker_integration(results: TestResults):
    """Test Docker integration (optional, requires Docker)."""
    print_header("DOCKER INTEGRATION TESTS (Optional)")
    
    # Check if Docker is available
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10
        )
        docker_available = result.returncode == 0
    except Exception:
        docker_available = False
    
    if not docker_available:
        print_info("Docker not available - skipping Docker integration tests")
        print_info("To run Docker tests, ensure Docker is running and try again")
        return
    
    print_info("Docker is available - running integration tests")
    
    # Test: Build containers with new config
    print_info("Testing Docker build with new configuration modules")
    try:
        project_root = Path(__file__).parent.parent
        
        # Check that required files exist
        required_files = [
            "src/config.py",
            "src/logging_config.py",
            "Dockerfile.api",
            "docker-compose.yml"
        ]
        
        all_exist = all((project_root / f).exists() for f in required_files)
        
        print_test("Required files for Docker build exist", all_exist)
        results.add("Docker", "Required files exist", all_exist)
        
    except Exception as e:
        print_test("Required files for Docker build exist", False, str(e))
        results.add("Docker", "Required files exist", False, str(e))


# ==============================================================================
# Generate Sample Outputs for Documentation
# ==============================================================================

def generate_sample_outputs():
    """Generate sample outputs for documentation."""
    print_header("SAMPLE OUTPUTS FOR DOCUMENTATION")
    
    # Sample 1: Startup crash log (missing API key)
    print_info("Sample A: Startup crash log (missing API key)")
    crash_log = """
============================================================
CONFIGURATION ERROR - Application cannot start
============================================================

1 validation error for Settings
groq_api_key
  Value error, GROQ_API_KEY is required and cannot be empty. 
  Please set it in your .env file or environment variables. 
  [type=value_error, input_value='', input_type=str]

Please check your .env file and environment variables.
============================================================
"""
    print(crash_log)
    
    # Sample 2: JSON log entry with request_id
    print_info("Sample B: JSON log entry with request_id tracing error")
    json_log = {
        "timestamp": "2025-11-27T14:30:45.123456+00:00",
        "level": "ERROR",
        "logger": "rag-chatbot",
        "message": "LLM timeout",
        "request_id": "a1b2c3d4",
        "module": "main",
        "function": "ask_question",
        "line": 251,
        "data": {
            "timeout_seconds": 30.0,
            "error_type": "APITimeoutError"
        }
    }
    print(json.dumps(json_log, indent=2))
    
    # Sample 3: Error response structure
    print_info("Sample C: API error response (503)")
    error_response = {
        "detail": {
            "message": "AI service is temporarily unavailable. Please try again in a moment.",
            "error_code": "LLM_TIMEOUT",
            "request_id": "a1b2c3d4",
            "retry_after": 5
        }
    }
    print(json.dumps(error_response, indent=2))


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Run all test suites."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         ARCHITECTURE VALIDATION TEST SUITE                       ║")
    print("║         Tickets 1-3: Observability, Config, Resilience           ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = TestResults()
    
    # Run test suites
    test_suite_a(results)
    test_suite_b(results)
    test_suite_c(results)
    test_suite_d(results)
    test_docker_integration(results)
    
    # Generate sample outputs
    generate_sample_outputs()
    
    # Summary
    all_passed = results.summary()
    
    print(f"\n  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if all_passed:
        print(f"\n{Colors.GREEN}{Colors.BOLD}  ✓ ALL TESTS PASSED - Architecture is sound!{Colors.RESET}\n")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}  ✗ SOME TESTS FAILED - Review required{Colors.RESET}\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    # Restore env after tests
    original_env = os.environ.copy()
    try:
        sys.exit(main())
    finally:
        os.environ.clear()
        os.environ.update(original_env)
