.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	flake8 --exclude .venv,llm_sdk .
	mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude ".venv|llm_sdk" --follow-imports=skip .

lint-strict:
	flake8 --exclude .venv,llm_sdk .
	mypy --strict --exclude ".venv|llm_sdk" --follow-imports=skip .