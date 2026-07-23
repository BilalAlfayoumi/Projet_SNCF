"""Embedding et construction de l'index FAISS.

Les embeddings tournent en local (sentence-transformers), sans cout d'API. Le modele
multilingual-e5 exige des prefixes distincts pour les requetes ("query: ") et les
documents ("passage: ") : sans eux, la qualite du retrieval chute. On encapsule cette
regle dans E5Embeddings, qui respecte l'interface Embeddings de LangChain pour que le
FAISS reste natif LangChain (reutilisable tel quel par le retriever de l'agent).
"""

from __future__ import annotations

from pathlib import Path

import structlog
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from sncf_agent.config import settings
from sncf_agent.ingestion.chunking import Chunk

log = structlog.get_logger(__name__)


class E5Embeddings(Embeddings):
    """Embeddings sentence-transformers avec prefixes e5 et normalisation L2.

    Compatible avec les modeles de la famille e5 (multilingual-e5-*). Pour un modele
    sans prefixes (ex. BGE-M3), instancier avec query_prefix="" et passage_prefix="".
    """

    def __init__(
        self,
        model_name: str | None = None,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        device: str | None = None,
    ) -> None:
        # Import local : evite de charger torch/sentence-transformers tant qu'on n'embedde pas.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or settings.embedding_model
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        # CPU par defaut, deliberement : (1) mesure faite, MPS n'est pas plus rapide pour
        # e5-large ici ; (2) sur Mac a memoire unifiee, laisser e5 sur MPS entre en
        # concurrence avec le LLM Ollama (qwen ~6 Go) et peut faire tuer le backend par
        # pression memoire. Passer device="mps" explicitement pour forcer si besoin.
        self._model = SentenceTransformer(self.model_name, device=device or "cpu")
        log.info("modele_embedding_charge", model=self.model_name, device=self._model.device.type)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,  # cosine via produit scalaire dans FAISS
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 500,
        )
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([f"{self.passage_prefix}{t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([f"{self.query_prefix}{text}"])[0]


_EMBEDDINGS: E5Embeddings | None = None


def get_embeddings() -> E5Embeddings:
    """Renvoie une instance partagee d'embeddings (chargee une seule fois).

    Le modele e5 pese ~2 Go : le backend doit le charger une fois et le reutiliser
    pour toutes les requetes (agent + inspection retrieval), pas a chaque appel.
    """
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = E5Embeddings()
    return _EMBEDDINGS


def index_dir(name: str) -> Path:
    """Repertoire de l'index FAISS pour un lot donne."""
    settings.ensure_data_dirs()
    return settings.index_dir / name


def list_indexes() -> list[str]:
    """Liste les index FAISS disponibles sur disque."""
    settings.ensure_data_dirs()
    return sorted(
        p.name for p in settings.index_dir.iterdir() if p.is_dir() and (p / "index.faiss").exists()
    )


def build_faiss(chunks: list[Chunk], name: str, embeddings: Embeddings | None = None) -> Path:
    """Embedde les chunks et ecrit un index FAISS sur disque. Renvoie le chemin."""
    emb = embeddings or get_embeddings()
    texts = [c.text for c in chunks]
    metadatas = [{"id": c.id, "source": c.source, **c.metadata} for c in chunks]

    log.info("embedding_debut", name=name, n_chunks=len(chunks))
    store = FAISS.from_texts(
        texts=texts, embedding=emb, metadatas=metadatas, ids=[c.id for c in chunks]
    )
    path = index_dir(name)
    store.save_local(str(path))
    log.info("index_faiss_ecrit", name=name, n_vectors=len(chunks), path=str(path))
    return path


def load_faiss(name: str, embeddings: Embeddings | None = None) -> FAISS:
    """Recharge un index FAISS existant."""
    emb = embeddings or get_embeddings()
    path = index_dir(name)
    if not path.exists():
        raise FileNotFoundError(f"index FAISS absent : {path}")
    # allow_dangerous_deserialization : l'index est produit localement par nous, donc sur.
    return FAISS.load_local(str(path), emb, allow_dangerous_deserialization=True)
