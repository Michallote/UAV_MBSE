.PHONY: lint
lint:
	@echo "Running linters for this project"
	black .
	isort .
	ruff check . --fix

.PHONY: tests
tests:
	@echo "Running test suite for the project"
	pytest --rootdir='.' --cov=src tests/ --cov-report=html:logs/cov/
	