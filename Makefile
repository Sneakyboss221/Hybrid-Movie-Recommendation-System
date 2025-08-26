.PHONY: help install install-dev test lint format clean build docs serve-docs

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black and isort"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build package"
	@echo "  docs         - Build documentation"
	@echo "  serve-docs   - Serve documentation locally"
	@echo "  run-web      - Run Streamlit web interface"
	@echo "  run-cli      - Run terminal interface"
	@echo "  demo         - Run demo script"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

# Testing
test:
	python test_system.py

test-coverage:
	pytest --cov=. --cov-report=html --cov-report=term

# Code quality
lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
	mypy .

format:
	black .
	isort .

format-check:
	black --check --diff .
	isort --check-only --diff .

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Building
build: clean
	python -m build

# Documentation
docs:
	cd docs && sphinx-build -b html . _build/html

serve-docs:
	cd docs/_build/html && python -m http.server 8000

# Running
run-web:
	streamlit run streamlit_app.py

run-cli:
	python main.py

demo:
	python demo.py

# Development setup
setup-dev: install-dev
	pre-commit install

# Security checks
security:
	bandit -r .
	safety check

# All checks
check-all: format-check lint test security

# Quick start
quick-start: install-dev
	@echo "Setting up development environment..."
	pre-commit install
	@echo "Running tests..."
	python test_system.py
	@echo "Starting web interface..."
	streamlit run streamlit_app.py
