# Makefile for RAG Campus Chatbot
# ================================
# This provides a single, self-documenting interface for all developers.
# Run `make help` to see available commands.

.PHONY: help install dev up down logs restart clean \
        ingest ingest-clear ingest-stats \
        test test-arch test-eval lint format \
        api worker ui shell

# Colors for help output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RESET := \033[0m

# Default Python interpreter
PYTHON := ./venv/bin/python
PIP := ./venv/bin/pip

# =============================================================================
# HELP
# =============================================================================

help: ## Show this help message
	@echo ""
	@echo "$(GREEN)RAG Campus Chatbot - Development Commands$(RESET)"
	@echo "==========================================="
	@echo ""
	@echo "$(YELLOW)Setup:$(RESET)"
	@grep -E '^(install|dev):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Docker:$(RESET)"
	@grep -E '^(up|down|logs|restart|clean):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Development (Local):$(RESET)"
	@grep -E '^(api|worker|ui):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Ingestion:$(RESET)"
	@grep -E '^(ingest|ingest-clear|ingest-stats):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Testing:$(RESET)"
	@grep -E '^(test|test-arch|test-eval):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Code Quality:$(RESET)"
	@grep -E '^(lint|format):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# =============================================================================
# SETUP
# =============================================================================

install: ## Install Python dependencies
	@echo "$(GREEN)Installing dependencies...$(RESET)"
	$(PIP) install -r requirements/api.txt
	$(PIP) install -r requirements/worker.txt
	$(PIP) install -r requirements/dev.txt 2>/dev/null || true
	@echo "$(GREEN)Done!$(RESET)"

dev: ## Setup development environment (venv + deps)
	@echo "$(GREEN)Setting up development environment...$(RESET)"
	python3 -m venv venv
	$(PIP) install --upgrade pip
	$(MAKE) install
	@echo "$(GREEN)Development environment ready!$(RESET)"
	@echo "Activate with: source venv/bin/activate"

# =============================================================================
# DOCKER COMMANDS
# =============================================================================

up: ## Start all services with Docker Compose
	@echo "$(GREEN)Starting services...$(RESET)"
	docker compose up -d
	@echo "$(GREEN)Services started!$(RESET)"
	@echo "  API: http://localhost:8000"
	@echo "  UI:  http://localhost:8501"

up-dev: ## Start development services (with hot reload)
	@echo "$(GREEN)Starting development services...$(RESET)"
	docker compose -f docker-compose.dev.yml up -d
	@echo "$(GREEN)Development services started!$(RESET)"

down: ## Stop all Docker services
	@echo "$(YELLOW)Stopping services...$(RESET)"
	docker compose down
	docker compose -f docker-compose.dev.yml down 2>/dev/null || true
	@echo "$(GREEN)Services stopped.$(RESET)"

logs: ## Show logs from all services
	docker compose logs -f

logs-api: ## Show API logs only
	docker compose logs -f api

logs-worker: ## Show worker logs only
	docker compose logs -f worker

restart: ## Restart all services
	$(MAKE) down
	$(MAKE) up

clean: ## Stop services and remove volumes
	@echo "$(YELLOW)Cleaning up...$(RESET)"
	docker compose down -v
	docker compose -f docker-compose.dev.yml down -v 2>/dev/null || true
	rm -rf src/chroma_db/*
	@echo "$(GREEN)Cleanup complete.$(RESET)"

# =============================================================================
# LOCAL DEVELOPMENT (without Docker)
# =============================================================================

api: ## Start API server locally (requires Redis)
	@echo "$(GREEN)Starting API server...$(RESET)"
	$(PYTHON) -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload --app-dir src

worker: ## Start Celery worker locally (requires Redis)
	@echo "$(GREEN)Starting Celery worker...$(RESET)"
	$(PYTHON) -m celery -A celery_config worker --loglevel=info --workdir src

ui: ## Start Streamlit UI locally
	@echo "$(GREEN)Starting Streamlit UI...$(RESET)"
	$(PYTHON) -m streamlit run src/app.py

shell: ## Open Python shell with project context
	cd src && $(PYTHON) -i -c "from config import get_settings; from logging_config import get_logger; print('Context loaded: get_settings(), get_logger()')"

# =============================================================================
# INGESTION
# =============================================================================

ingest: ## Smart ingest - auto-detects tables and regular text
	@echo "$(GREEN)Running smart ingestion (auto-detects tables)...$(RESET)"
	$(PYTHON) scripts/smart_ingest.py data/
	@echo "$(GREEN)Ingestion complete!$(RESET)"

ingest-basic: ## Basic ingestion without table detection (faster)
	@echo "$(GREEN)Ingesting documents (basic mode)...$(RESET)"
	$(PYTHON) scripts/direct_ingest.py data/
	@echo "$(GREEN)Ingestion complete!$(RESET)"

ingest-file: ## Ingest a specific file (usage: make ingest-file FILE=path/to/file.pdf)
	@echo "$(GREEN)Ingesting $(FILE)...$(RESET)"
	$(PYTHON) scripts/direct_ingest.py $(FILE)

ingest-celery: ## Trigger ingestion via Celery (requires worker running)
	@echo "$(GREEN)Triggering Celery ingestion...$(RESET)"
	$(PYTHON) scripts/trigger_ingestion.py data/

ingest-clear: ## Clear database and re-ingest with smart detection
	@echo "$(YELLOW)Clearing database and re-ingesting...$(RESET)"
	$(PYTHON) scripts/smart_ingest.py data/ --clear
	@echo "$(GREEN)Re-ingestion complete!$(RESET)"

ingest-stats: ## Show database statistics
	$(PYTHON) scripts/smart_ingest.py --stats

# =============================================================================
# TESTING
# =============================================================================

test: ## Run all tests with pytest
	@echo "$(GREEN)Running tests...$(RESET)"
	$(PYTHON) -m pytest tests/ -v

test-arch: ## Run architecture validation tests
	@echo "$(GREEN)Running architecture tests...$(RESET)"
	$(PYTHON) tests/test_architecture.py

test-eval: ## Run system evaluation tests (requires running API)
	@echo "$(GREEN)Running evaluation tests...$(RESET)"
	$(PYTHON) tests/test_system_evaluation.py

test-quick: ## Quick smoke test
	@echo "$(GREEN)Running quick tests...$(RESET)"
	$(PYTHON) -c "import sys; sys.path.insert(0, 'src'); from config import get_settings; from logging_config import get_logger; print('✓ Imports OK')"
	curl -s http://localhost:8000/ | grep -q "online" && echo "✓ API OK" || echo "✗ API not running"

test-enhanced: ## Test enhanced RAG features (requires running API)
	@echo "$(GREEN)Testing enhanced features...$(RESET)"
	$(PYTHON) scripts/test_enhanced_features.py

test-api: ## Quick API health check
	@echo "$(GREEN)Testing API...$(RESET)"
	@curl -s http://localhost:8000/ | grep -q "online" && echo "✓ API is running" || echo "✗ API not running"

# =============================================================================
# CODE QUALITY
# =============================================================================

lint: ## Run linters (ruff)
	@echo "$(GREEN)Running linters...$(RESET)"
	$(PYTHON) -m ruff check src/ tests/ scripts/ || true
	$(PYTHON) -m flake8 src/ --max-line-length=120 --ignore=E501,W503 || true

format: ## Format code with black and isort
	@echo "$(GREEN)Formatting code...$(RESET)"
	$(PYTHON) -m black src/ tests/ scripts/ --line-length=120 || echo "black not installed"
	$(PYTHON) -m isort src/ tests/ scripts/ || echo "isort not installed"

typecheck: ## Run type checking with mypy
	$(PYTHON) -m mypy src/ --ignore-missing-imports || echo "mypy not installed"

# =============================================================================
# UTILITIES
# =============================================================================

check-env: ## Verify environment configuration
	@echo "$(GREEN)Checking environment...$(RESET)"
	@test -f .env && echo "✓ .env exists" || echo "✗ .env missing (copy from .env.example)"
	@test -d venv && echo "✓ venv exists" || echo "✗ venv missing (run: make dev)"
	@docker info > /dev/null 2>&1 && echo "✓ Docker running" || echo "✗ Docker not running"

docs: ## Open documentation
	@echo "Opening README.md..."
	@open README.md 2>/dev/null || cat README.md
