.PHONY: setup run docker clean format lint typecheck test ci deployment


setup:
	( command -v python3.11 >/dev/null 2>&1 && python3.11 -m venv .venv ) || python3 -m venv .venv
	source .venv/bin/activate && \
	python -V && \
	pip install -U pip && \
	pip install -e '.[dev]'

run: setup
	set -a; source .env; set +a; \
	.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker:
	docker build -t terror-reco:latest .
	docker run --env-file .env -p 8000:8000 terror-reco:latest

clean:
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

format:
	# Auto-fix what we can, but don't fail the target if some issues remain
	ruff check . --fix || true
	black . || true

lint:
	ruff check .
	black --check .

typecheck:
	mypy app

test:
	pytest -q

deployment:
	python tests/manual_deployment.py
	python tests/manual_deployment_simple.py || true
	python tests/deployment_checklist.py || true

ci: lint typecheck test