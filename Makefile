.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src index --max_chunk_size 2000

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	find . -type f -name "*.pyc" -delete
	rm -rf src/.mypy_cache

lint:
	flake8 src
	mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
