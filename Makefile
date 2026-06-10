.PHONY: install test run-api run-dashboard docker-up airflow-init airflow-up dvc-repro lint

install:
	pip install -r requirements.txt

test:
	pytest -q

run-api:
	uvicorn firewall.api.main:app --app-dir src --reload --host 0.0.0.0 --port 8000

run-dashboard:
	streamlit run src/dashboard/app.py --server.port 8501

docker-up:
	docker compose up -d api dashboard postgres mlflow

airflow-init:
	docker compose -f docker-compose.airflow.yml run --rm airflow-init

airflow-up:
	docker compose -f docker-compose.airflow.yml up -d webserver scheduler

dvc-repro:
	python -m dvc repro ingest validate

lint:
	ruff check src tests pipelines scripts dags
