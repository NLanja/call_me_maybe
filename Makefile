export UV_CACHE_DIR=/goinfre/$(USER)/.cache/uv
export HF_HOME=/goinfre/$(USER)/.cache/huggingface


all: install run

install:
	uv sync

run:
	uv run python -m src $(ARGS)

debug:
	uv run python -m pdb -m src -- $(ARGS)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	uv run flake8 --exclude=".venv,llm_sdk" .
	uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs .

lint-strict:
	uv run flake8 --exclude=".venv,llm_sdk" .
	uv run mypy --strict .

.PHONY: all install run debug lint lint-strict clean