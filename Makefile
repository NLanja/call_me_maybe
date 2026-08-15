all: install run

install:
	uv sync

run:
	uv run -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache
	find . -type f -name "*.pyc" -delete

lint:
	uv run flake8 --exclude=".venv,llm_sdk" .
	uv run mypy --exclude '.venv|llm_sdk' --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs .


lint-strict:
	uv run flake8 --exclude=".venv" .
	uv run mypy --strict .

.PHONY: install run debug lint lint-strict clean