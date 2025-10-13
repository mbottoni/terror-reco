.PHONY: setup run docker clean

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
	ruff check . --fix
	black .
	mypy app
	pytest