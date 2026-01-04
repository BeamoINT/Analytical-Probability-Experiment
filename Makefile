.PHONY: help install test lint format type-check clean run api report

help:  ## Show this help message
	@echo "PolyB0T - Makefile commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies with Poetry
	poetry install

install-dev:  ## Install with development dependencies
	poetry install --with dev

test:  ## Run tests
	poetry run pytest -v

test-cov:  ## Run tests with coverage
	poetry run pytest --cov=polyb0t --cov-report=html --cov-report=term

lint:  ## Run linting checks
	poetry run ruff check polyb0t tests

format:  ## Format code with black
	poetry run black polyb0t tests

format-check:  ## Check code formatting
	poetry run black --check polyb0t tests

type-check:  ## Run mypy type checking
	poetry run mypy polyb0t

check-all: lint format-check type-check test  ## Run all checks

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.db" -delete

db-init:  ## Initialize database
	poetry run polyb0t db init

db-reset:  ## Reset database (WARNING: deletes all data)
	poetry run polyb0t db reset

run:  ## Run paper trading bot
	poetry run polyb0t run --paper

api:  ## Start API server
	poetry run polyb0t api

report:  ## Generate trading report
	poetry run polyb0t report --today

universe:  ## Show tradable universe
	poetry run polyb0t universe

docker-build:  ## Build Docker image (if Dockerfile exists)
	docker build -t polyb0t:latest .

docker-run:  ## Run in Docker container
	docker run -it --rm --env-file .env polyb0t:latest

.DEFAULT_GOAL := help

