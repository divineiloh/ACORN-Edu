# ACORN-Edu Research Pipeline Makefile
# Gold-standard automation for research simulation

.PHONY: setup clean format lint test run all help

# Default target
all: format lint test run

# Setup Python virtual environment and install dependencies
setup:
	@echo "Setting up ACORN-Edu research environment..."
	python -m venv venv
	venv\Scripts\activate && pip install --upgrade pip
	venv\Scripts\activate && pip install -r requirements.txt
	@echo "Setup complete! Activate with: venv\Scripts\activate"

# Clean output directory and Python cache files
clean:
	@echo "Cleaning output directory and cache files..."
	if exist output rmdir /s /q output
	if exist __pycache__ rmdir /s /q __pycache__
	if exist src\__pycache__ rmdir /s /q src\__pycache__
	if exist tests\__pycache__ rmdir /s /q tests\__pycache__
	@echo "Clean complete!"

# Format Python code using black
format:
	@echo "Formatting Python code with black..."
	venv\Scripts\activate && black src/ tests/
	@echo "Code formatting complete!"

# Lint Python code using flake8
lint:
	@echo "Linting Python code with flake8..."
	venv\Scripts\activate && flake8 src/ tests/
	@echo "Linting complete!"

# Run unit tests using pytest
test:
	@echo "Running unit tests with pytest..."
	venv\Scripts\activate && python -m pytest tests/ -v --cov=src
	@echo "Testing complete!"

# Execute the full research pipeline
run:
	@echo "Running ACORN-Edu research pipeline..."
	venv\Scripts\activate && python -m src.simulation_harness
	@echo "Research pipeline complete!"

# Show help information
help:
	@echo "ACORN-Edu Research Pipeline Commands:"
	@echo "  make setup    - Create virtual environment and install dependencies"
	@echo "  make clean    - Delete output directory and Python cache files"
	@echo "  make format   - Format all Python code using black"
	@echo "  make lint     - Lint all Python code using flake8"
	@echo "  make test     - Run all unit tests using pytest"
	@echo "  make run      - Execute the full research pipeline"
	@echo "  make all      - Run format, lint, test, and run in sequence"
	@echo "  make help     - Show this help information"
