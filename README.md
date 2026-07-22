# Agent conversationnel RAG SNCF

Agent conversationnel qui repond a des questions sur les donnees open data SNCF
(horaires, perturbations, gares, conditions d'abonnement) via un pipeline RAG et
un agent LangGraph.

Projet portfolio. La feuille de route technique complete est dans
[`plan-projet-sncf.md`](plan-projet-sncf.md) et le calendrier de realisation dans
les instructions du depot ([`CLAUDE.md`](CLAUDE.md)).

## Stack (prototype)

- **Backend** : FastAPI
- **Frontend** : Streamlit
- **Agent / RAG** : LangGraph + LangChain (LCEL)
- **Vector store** : FAISS (local)
- **Embeddings** : multilingual-e5 ou BGE-M3 via HuggingFace
- **LLM** : Anthropic (appel direct en prototype ; gateway Portkey plus tard)
- **Observabilite** : LangSmith

## Installation

Le projet utilise [uv](https://docs.astral.sh/uv/).

```bash
# Installe les dependances et cree le venv
uv sync

# Copie et renseigne les variables d'environnement
cp .env.example .env
```

Extras optionnels (installes a la demande selon la phase) :

```bash
uv sync --extra eval    # Semaine 4 : RAGAS
uv sync --extra scale   # Semaine 6 : Qdrant, Redis, Portkey
```

## Structure

```
src/sncf_agent/
  config.py          Configuration centralisee (pydantic-settings, lit .env)
  ingestion/         Pipeline API SNCF -> parsing -> chunking -> checkpoint -> embedding -> FAISS
  rag/               Retriever + agent LangGraph
  backend/           API FastAPI
  frontend/          Interface Streamlit
eval/                Golden set + evaluation RAGAS (Semaine 4)
data/                Donnees generees (raw, checkpoints, index) ; non commite
tests/               Tests
```

## Commandes

```bash
uv run uvicorn sncf_agent.backend.app:app --reload   # backend (a venir)
uv run streamlit run src/sncf_agent/frontend/app.py  # frontend (a venir)
uv run pytest                                         # tests
uv run ruff check .                                   # lint
```
