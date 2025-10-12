.PHONY: setup run docker clean

setup:
	python3 -m venv .venv
	source .venv/bin/activate && \
	pip install -U pip && \
	pip install -e '.[dev]'

run: setup
	set -a; source .env; set +a
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker:
	docker build -t terror-reco:latest .
	docker run --env-file .env -p 8000:8000 terror-reco:latest

clean:
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete