.PHONY: help install lint format typecheck test test-fast test-gpu docker-build docker-train docker-inference clean

help:
	@echo "Available targets:"
	@echo "  install        - Install Python dependencies"
	@echo "  lint           - Run ruff linter"
	@echo "  format         - Format code with ruff"
	@echo "  typecheck      - Run mypy type checker"
	@echo "  test           - Run full pytest suite"
	@echo "  test-fast      - Run non-slow tests"
	@echo "  test-gpu       - Run GPU tests"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-train   - Run training in Docker"
	@echo "  docker-inference - Run inference in Docker"
	@echo "  clean          - Remove generated artifacts"

install:
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts
	ruff check --fix src tests scripts

typecheck:
	mypy src

test:
	pytest -v

test-fast:
	pytest -v -m "not slow and not gpu and not requires_model"

test-gpu:
	pytest -v -m gpu

docker-build:
	docker compose build

docker-train:
	docker compose up train

docker-inference:
	docker compose up inference

clean:
	rm -rf outputs/ logs/ runs/ checkpoints/ .pytest_cache/ htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
