"""Harnais d'evaluation RAGAS de l'agent RAG SNCF.

Principe : on ne peut pas optimiser ce qu'on ne mesure pas. Ce harnais fait tourner
l'agent sur le golden set, recupere reponse + contextes, puis calcule les metriques
RAGAS (juge = Groq, embeddings = e5 local). On relance ce script apres chaque
changement (chunking, retrieval, prompt) pour valider qu'on ameliore, pas qu'on degrade.

Usage :
    uv run python -m eval.harness                 # golden set complet
    uv run python -m eval.harness --limit 3       # smoke test rapide (3 questions)
    uv run python -m eval.harness --label baseline # nomme le run pour comparaison
"""

from __future__ import annotations

# --- Shim de compatibilite RAGAS 0.4.3 / langchain-community 0.4.x ---
# RAGAS importe langchain_community.chat_models.vertexai.ChatVertexAI, chemin supprime
# dans langchain-community recent. On injecte un module factice AVANT d'importer ragas.
# ChatVertexAI n'est pas utilise par notre pipeline (juge = Groq).
import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _shim = types.ModuleType("langchain_community.chat_models.vertexai")
    _shim.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _shim

import argparse  # noqa: E402
import json  # noqa: E402
import warnings  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import structlog  # noqa: E402
from langchain.chat_models import init_chat_model  # noqa: E402
from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402

from sncf_agent.config import PROJECT_ROOT, settings  # noqa: E402
from sncf_agent.ingestion.embedding import get_embeddings  # noqa: E402
from sncf_agent.rag.agent import ask_with_contexts, build_agent, build_multi_agent  # noqa: E402

log = structlog.get_logger(__name__)

GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden_set.json"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


def load_golden_set() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _build_metrics():
    """Metriques RAGAS. On evite les imports deprecies via ragas.metrics.collections."""
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    return [
        Faithfulness(),  # la reponse est-elle fidele aux contextes recuperes (grounding)
        ResponseRelevancy(),  # la reponse repond-elle a la question
        LLMContextPrecisionWithReference(),  # les contextes recuperes sont-ils pertinents
        LLMContextRecall(),  # les contextes couvrent-ils la reference
    ]


def build_dataset(
    index_name: str, items: list[dict], k: int, limit: int | None
) -> EvaluationDataset:
    """Fait tourner l'agent sur chaque question et construit le dataset RAGAS.

    L'agent teste utilise le LLM d'eval (OpenAI) et non le provider de l'app,
    pour deux raisons : quota/latence des providers locaux ou gratuits, et coherence
    des runs (baseline vs optimisations mesurees avec le meme LLM).

    index_name == "multi" : evalue l'agent multi-corpus (routage inclus), c'est le
    mode par defaut du golden set v0.3+.
    """
    if index_name == "multi":
        agent = build_multi_agent(
            k=k, provider=settings.eval_llm_provider, model=settings.eval_llm_model
        )
    else:
        agent = build_agent(
            index_name, k=k, provider=settings.eval_llm_provider, model=settings.eval_llm_model
        )
    samples: list[SingleTurnSample] = []
    subset = items[:limit] if limit else items
    for i, item in enumerate(subset, 1):
        question = item["question"]
        log.info("eval_echantillon", i=i, total=len(subset), question=question[:60])
        # Contextes REELLEMENT utilises par l'agent (extraits de ses appels d'outil).
        response, contexts = ask_with_contexts(agent, question)
        # RAGAS exige au moins un contexte : pour une question hors domaine que l'agent
        # decline sans rien chercher, on met un marqueur explicite.
        if not contexts:
            contexts = ["(aucun passage recupere : question hors du perimetre du corpus)"]
        samples.append(
            SingleTurnSample(
                user_input=question,
                response=response,
                retrieved_contexts=contexts,
                reference=item["reference"],
            )
        )
    return EvaluationDataset(samples=samples)


def run_eval(limit: int | None = None, label: str = "run", k: int | None = None) -> Path:
    golden = load_golden_set()
    index_name = golden["index"]
    k = k or settings.retrieval_k

    dataset = build_dataset(index_name, golden["items"], k=k, limit=limit)

    evaluator_llm = LangchainLLMWrapper(
        init_chat_model(settings.eval_llm_model, model_provider=settings.eval_llm_provider)
    )
    evaluator_emb = LangchainEmbeddingsWrapper(get_embeddings())

    log.info("ragas_evaluate_debut", n=len(dataset), k=k, index=index_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = evaluate(
            dataset=dataset,
            metrics=_build_metrics(),
            llm=evaluator_llm,
            embeddings=evaluator_emb,
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"{stamp}_{label}.json"

    df = result.to_pandas()
    scores = {c: float(df[c].mean()) for c in df.columns if df[c].dtype.kind == "f"}
    payload = {
        "label": label,
        "timestamp": stamp,
        "index": index_name,
        "k": k,
        "n_questions": len(dataset),
        "scores_moyens": scores,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(RESULTS_DIR / f"{stamp}_{label}_details.csv", index=False)

    log.info("ragas_evaluate_fini", scores=scores, out=str(out))
    print("\n=== SCORES MOYENS (" + label + ") ===")
    for metric, val in scores.items():
        print(f"  {metric}: {val:.3f}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation RAGAS de l'agent SNCF")
    parser.add_argument("--limit", type=int, default=None, help="nb de questions (smoke test)")
    parser.add_argument("--label", default="run", help="nom du run (ex. baseline)")
    parser.add_argument("--k", type=int, default=None, help="nb de passages recuperes")
    args = parser.parse_args()
    run_eval(limit=args.limit, label=args.label, k=args.k)


if __name__ == "__main__":
    main()
