"""Acces au vector store, isole derriere une seule fonction.

C'est LE point de bascule FAISS -> Qdrant. L'agent depend de l'interface retriever de
LangChain (VectorStoreRetriever), jamais de FAISS directement. En Semaine 6 (version
scalable), on remplace le contenu de get_vectorstore() par un QdrantVectorStore : le
reste du code (retriever, agent) ne change pas, car les deux implementent la meme
interface VectorStore de LangChain.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

from sncf_agent.ingestion.embedding import load_faiss


def get_vectorstore(name: str, embeddings: Embeddings | None = None) -> VectorStore:
    """Renvoie le vector store pour un index donne.

    Prototype : FAISS local. Version scalable (Semaine 6) : remplacer par Qdrant ici,
    sans toucher au retriever ni a l'agent.
    """
    return load_faiss(name, embeddings=embeddings)


def get_retriever(
    name: str,
    k: int = 4,
    embeddings: Embeddings | None = None,
) -> VectorStoreRetriever:
    """Renvoie un retriever LangChain (les k passages les plus proches)."""
    store = get_vectorstore(name, embeddings=embeddings)
    return store.as_retriever(search_kwargs={"k": k})


def search_relevant(
    name: str,
    query: str,
    k: int = 4,
    threshold: float | None = None,
    embeddings: Embeddings | None = None,
):
    """Recherche filtree par seuil de pertinence : ne renvoie que les documents dont la
    distance L2 est sous le seuil. Liste vide si rien n'est assez proche, ce qui permet
    a l'agent de refuser honnetement au lieu de raisonner sur du hors-sujet.
    """
    from sncf_agent.config import settings

    seuil = threshold if threshold is not None else settings.retrieval_score_threshold
    store = get_vectorstore(name, embeddings=embeddings)
    results = store.similarity_search_with_score(query, k=k)
    return [doc for doc, score in results if float(score) <= seuil]


def search_with_scores(
    name: str,
    query: str,
    k: int = 5,
    embeddings: Embeddings | None = None,
) -> list[tuple[str, str, float]]:
    """Recherche brute avec scores, pour inspecter la qualite du retrieval.

    Renvoie une liste de (texte, source, score). Le score FAISS est une distance L2 :
    plus il est PETIT, plus le passage est proche de la requete.
    """
    store = get_vectorstore(name, embeddings=embeddings)
    results = store.similarity_search_with_score(query, k=k)
    return [
        (doc.page_content, doc.metadata.get("source", "inconnue"), float(score))
        for doc, score in results
    ]
