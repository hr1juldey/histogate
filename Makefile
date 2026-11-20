# HistoGate — see docs/PLAN.md for the PR sequence each target serves
.PHONY: setup lint test train ssl eval export gate serve mcp publish

setup:
	uv sync --all-extras

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests || true

test:
	uv run pytest -m "not gpu"

test-gpu:
	uv run pytest

train:
	uv run histogate-train

ssl:
	uv run histogate-train --ssl

eval:
	uv run histogate-eval

export:
	uv run histogate-export

gate:
	uv run histogate-gate

serve:
	uv run histogate-serve --api

mcp:
	uv run histogate-serve --mcp

publish:
	@echo "PR-15: README/model-card lint + hf push (implemented at PR-15)"
