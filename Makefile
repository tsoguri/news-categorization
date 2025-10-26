# Format code with Ruff
format:
	@echo "Formatting code with Ruff..."
	uv run ruff format .
	uv run ruff check --select I --fix .


# Lint code with Ruff
lint:
	@echo "Linting code with Ruff..."
	uv run ruff check . --fix