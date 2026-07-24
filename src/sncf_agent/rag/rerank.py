"""Reranking cross-encodeur du retrieval.

Le retrieval vectoriel (bi-encodeur e5) encode requete et passage SEPAREMENT : rapide,
mais approximatif. Un cross-encodeur lit la PAIRE (requete, passage) ensemble et juge
leur pertinence bien plus finement. D'ou la strategie retrieve-then-rerank : le
bi-encodeur ratisse large (rerank_fetch_k candidats), le cross-encodeur reordonne, on
ne garde que les top-k passages transmis au LLM.

Le cross-encodeur tourne sur CPU (comme e5) : on borne rerank_fetch_k pour maitriser le
cout. Le modele est un singleton, charge une seule fois (comme get_embeddings).
"""

from __future__ import annotations

import threading

import structlog
from langchain_core.documents import Document

from sncf_agent.config import settings

log = structlog.get_logger(__name__)

_RERANKER = None
_RERANKER_LOCK = threading.Lock()


def get_reranker():
    """Renvoie une instance partagee du cross-encodeur (chargee une seule fois).

    Thread-safe (double-check locking) : l'echauffement au boot et une requete
    concurrente ne doivent pas charger deux instances du modele.
    """
    global _RERANKER
    if _RERANKER is None:
        with _RERANKER_LOCK:
            if _RERANKER is None:  # double-check apres acquisition du verrou
                # Import local : ne charge torch/sentence-transformers que si on rerank.
                from sentence_transformers import CrossEncoder

                log.info("chargement_reranker", model=settings.rerank_model)
                _RERANKER = CrossEncoder(settings.rerank_model, device="cpu")
    return _RERANKER


def rerank(query: str, docs: list[Document], top_k: int) -> list[Document]:
    """Reordonne les passages par pertinence (cross-encodeur) et garde les top_k.

    docs : candidats issus du retrieval vectoriel (deja filtres par seuil de pertinence).
    Renvoie au plus top_k documents, du plus pertinent au moins pertinent.
    """
    if not docs:
        return []
    if len(docs) <= 1:
        return docs[:top_k]
    reranker = get_reranker()
    scores = reranker.predict([(query, d.page_content) for d in docs])
    ranked = sorted(zip(docs, scores, strict=True), key=lambda pair: float(pair[1]), reverse=True)
    return [doc for doc, _ in ranked[:top_k]]
