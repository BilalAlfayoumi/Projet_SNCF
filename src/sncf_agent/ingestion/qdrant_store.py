"""Backend vectoriel Qdrant Cloud (alternative a FAISS, pour la prod/scale).

Meme interface de lecture que FAISS (via langchain), donc l'agent ne voit aucune
difference. Avantages sur FAISS : mises a jour incrementales (ajout/suppression d'un
passage sans tout re-embedder), recherche concurrente scalable, serveur manage.

Difference importante a garder en tete : Qdrant renvoie une SIMILARITE COSINUS
(plus grand = plus proche), la ou FAISS renvoie une DISTANCE L2 (plus petit = plus
proche). La logique de seuil (search_relevant) gere les deux sens.
"""

from __future__ import annotations

import uuid

import structlog
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from sncf_agent.config import settings
from sncf_agent.ingestion.chunking import Chunk
from sncf_agent.ingestion.embedding import get_embeddings

log = structlog.get_logger(__name__)

# Namespace stable pour deriver un id de point Qdrant (UUID) depuis notre id de chunk
# (hex 16 car., non valide comme UUID). Idempotence : meme chunk -> meme point.
_NS = uuid.UUID("5f2b1e00-0000-4000-8000-000000000001")


def _client() -> QdrantClient:
    if not settings.qdrant_url or not settings.qdrant_api_key:
        raise RuntimeError("QDRANT_CLUSTER_ENDPOINT / QDRANT_API_KEY manquants dans .env")
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def list_collections() -> list[str]:
    """Liste les collections presentes sur le cluster Qdrant."""
    return sorted(c.name for c in _client().get_collections().collections)


def build_qdrant(chunks: list[Chunk], name: str, embeddings: Embeddings | None = None) -> str:
    """Embedde les chunks et (re)cree la collection Qdrant `name`. Renvoie le nom."""
    emb = embeddings or get_embeddings()
    client = _client()

    # Dimension deduite d'un embedding reel (e5-large = 1024), collection en cosinus.
    dim = len(emb.embed_query("dimension"))
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    store = QdrantVectorStore(client=client, collection_name=name, embedding=emb)
    ids = [str(uuid.uuid5(_NS, c.id)) for c in chunks]
    metadatas = [{"id": c.id, "source": c.source, **c.metadata} for c in chunks]
    log.info("embedding_debut_qdrant", name=name, n_chunks=len(chunks))
    store.add_texts(texts=[c.text for c in chunks], metadatas=metadatas, ids=ids)
    log.info("collection_qdrant_ecrite", name=name, n_vectors=len(chunks))
    return name


def load_qdrant(name: str, embeddings: Embeddings | None = None) -> QdrantVectorStore:
    """Charge une collection Qdrant existante comme vector store langchain."""
    emb = embeddings or get_embeddings()
    client = _client()
    if not client.collection_exists(name):
        raise FileNotFoundError(f"collection Qdrant absente : {name}")
    return QdrantVectorStore(client=client, collection_name=name, embedding=emb)
