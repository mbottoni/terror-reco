.PHONY: setup run docker clean format lint typecheck test ci deployment corpus corpus-validate eval tune migrate migration migrate-check


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

# Build the horror corpus offline (resumable; safe to re-run after a failure)
corpus:
	set -a; source .env; set +a; \
	.venv/bin/python scripts/build_corpus.py --target 500 --resume

corpus-validate:
	.venv/bin/python scripts/build_corpus.py --validate-only

# Database migrations
migrate:
	.venv/bin/python -m alembic upgrade head

migration:
	@test -n "$(m)" || (echo 'usage: make migration m="describe the change"'; exit 1)
	.venv/bin/python -m alembic revision --autogenerate -m "$(m)"

migrate-check:
	.venv/bin/python -m alembic check

# Grid-search blend weights (see docs/evaluation-baseline.md before shipping any)
tune:
	.venv/bin/python scripts/tune_weights.py

# Score the recommender against the gold test set (see docs/evaluation-baseline.md)
eval:
	.venv/bin/python scripts/run_eval.py --runs 10

deployment:
	python tests/manual_deployment.py
	python tests/manual_deployment_simple.py || true
	python tests/deployment_checklist.py || true

ci: lint typecheck test
