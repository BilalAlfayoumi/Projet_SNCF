"""Chunking : decoupe les Passages en morceaux de taille homogene avec overlap.

Les passages issus de donnees structurees sont souvent deja courts (un enregistrement
= un passage). Le chunking sert surtout aux documents textuels longs (regles
d'abonnement, CGV). On s'appuie sur le splitter recursif de LangChain, avec un id
stable par chunk pour tracer la provenance.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from sncf_agent.config import settings
from sncf_agent.ingestion.parsing import Passage


@dataclass(slots=True)
class Chunk:
    """Un morceau de texte pret a etre embedde, avec un id stable et sa provenance."""

    id: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        return cls(id=d["id"], text=d["text"], source=d["source"], metadata=d.get("metadata", {}))


def _chunk_id(source: str, text: str, position: int) -> str:
    """Id deterministe : meme contenu au meme endroit = meme id (idempotence)."""
    digest = hashlib.sha1(f"{source}|{position}|{text}".encode()).hexdigest()
    return digest[:16]


def chunk_passages(
    passages: list[Passage],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Decoupe une liste de Passages en Chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for passage in passages:
        pieces = splitter.split_text(passage.text)
        for position, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            meta = dict(passage.metadata)
            meta["n_chunks"] = len(pieces)
            chunks.append(
                Chunk(
                    id=_chunk_id(passage.source, piece, position),
                    text=piece,
                    source=passage.source,
                    metadata=meta,
                )
            )
    return chunks
