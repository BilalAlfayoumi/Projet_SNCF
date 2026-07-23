# Makefile du projet agent RAG SNCF
# Usage : make <cible>          Ex : make backend, make eval LABEL=test1
#         make help             liste les cibles disponibles

# Variables surchargables : make ingest DATASET=tarifs-intercites WHERE="1=1"
DATASET ?= regularite-mensuelle-tgv-aqst
WHERE   ?= date>='2024-01'
LABEL   ?= run
LIMIT   ?=
K       ?=

.PHONY: help install lint test ingest ingest-checkpoint ingest-no-embed backend front eval eval-smoke

help:  ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Installe les dependances + extras eval (uv sync)
	uv sync --extra eval

lint:  ## Lint du code (ruff)
	uv run ruff check .

test:  ## Lance les tests (pytest)
	uv run pytest

ingest:  ## Ingestion complete d'un dataset (DATASET, WHERE)
	uv run python -m sncf_agent.ingestion.pipeline $(DATASET) --where "$(WHERE)"

ingest-checkpoint:  ## Reprise depuis le checkpoint JSON (re-embedding seulement)
	uv run python -m sncf_agent.ingestion.pipeline $(DATASET) --from-checkpoint

ingest-no-embed:  ## Extraction+parsing+chunking, s'arrete au checkpoint
	uv run python -m sncf_agent.ingestion.pipeline $(DATASET) --where "$(WHERE)" --no-embed

ingest-docs:  ## Ingestion d'un corpus documentaire web/PDF (CORPUS=abonnement-max-jeune)
	uv run python -m sncf_agent.ingestion.docs $(CORPUS)

backend:  ## Lance l'API FastAPI sur http://127.0.0.1:8000 (charge le modele, ~1 min)
	uv run uvicorn sncf_agent.backend.app:app --host 127.0.0.1 --port 8000

front:  ## Lance le frontend Streamlit sur http://127.0.0.1:8501
	uv run streamlit run src/sncf_agent/frontend/app.py

eval:  ## Eval RAGAS complete du golden set (LABEL=nom, K=passages, LIMIT=nb questions)
	uv run python -m eval.harness --label $(LABEL) $(if $(LIMIT),--limit $(LIMIT)) $(if $(K),--k $(K))

eval-smoke:  ## Eval rapide (3 questions) pour verifier que tout marche
	uv run python -m eval.harness --limit 3 --label smoke
