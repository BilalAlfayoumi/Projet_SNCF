"""Acces au vector store, isole derriere une seule fonction : LE point de bascule
FAISS <-> Qdrant.

L'agent depend de l'interface retriever de LangChain, jamais d'une implementation. Le
backend est choisi par settings.vector_backend (VECTOR_BACKEND) : FAISS local pour le
dev offline, Qdrant Cloud pour la prod/scale. Changer de backend ne touche ni le
retriever, ni l'agent, ni les guardrails.

Semantique des scores (gere par search_relevant / est_pertinent) :
- FAISS  : distance L2, plus PETIT = plus proche (garde si score <= seuil L2).
- Qdrant : similarite cosinus, plus GRAND = plus proche (garde si score >= seuil cos).
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

from sncf_agent.config import settings


def get_vectorstore(name: str, embeddings: Embeddings | None = None) -> VectorStore:
    """Renvoie le vector store pour un index/collection, selon le backend configure."""
    if settings.is_qdrant:
        from sncf_agent.ingestion.qdrant_store import load_qdrant

        return load_qdrant(name, embeddings=embeddings)
    from sncf_agent.ingestion.embedding import load_faiss

    return load_faiss(name, embeddings=embeddings)


def get_retriever(
    name: str,
    k: int = 4,
    embeddings: Embeddings | None = None,
) -> VectorStoreRetriever:
    """Renvoie un retriever LangChain (les k passages les plus proches)."""
    store = get_vectorstore(name, embeddings=embeddings)
    return store.as_retriever(search_kwargs={"k": k})


def est_pertinent(score: float) -> bool:
    """Vrai si un passage est assez proche pour etre transmis a l'agent, selon le sens
    du score du backend actif (L2 pour FAISS, cosinus pour Qdrant)."""
    if settings.is_qdrant:
        return score >= settings.qdrant_score_threshold
    return score <= settings.retrieval_score_threshold


def search_relevant(
    name: str,
    query: str,
    k: int = 4,
    embeddings: Embeddings | None = None,
):
    """Recherche filtree par le seuil de pertinence du backend actif. Liste vide si rien
    n'est assez proche, ce qui permet a l'agent de refuser honnetement plutot que de
    raisonner sur du hors-sujet."""
    store = get_vectorstore(name, embeddings=embeddings)
    results = store.similarity_search_with_score(query, k=k)
    return [doc for doc, score in results if est_pertinent(float(score))]


def search_with_scores(
    name: str,
    query: str,
    k: int = 5,
    embeddings: Embeddings | None = None,
) -> list[tuple[str, str, float]]:
    """Recherche brute avec scores, pour inspecter la qualite du retrieval.

    Renvoie (texte, source, score). Sens du score selon le backend (voir en-tete).
    """
    store = get_vectorstore(name, embeddings=embeddings)
    results = store.similarity_search_with_score(query, k=k)
    return [
        (doc.page_content, doc.metadata.get("source", "inconnue"), float(score))
        for doc, score in results
    ]
