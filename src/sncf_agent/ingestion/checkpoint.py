"""Checkpoint JSON intermediaire du pipeline d'ingestion.

Choix delibere (voir plan-projet-sncf.md) : on sauvegarde les chunks en JSON APRES le
parsing/chunking et AVANT l'embedding. Si l'embedding plante (modele, memoire, reseau),
on repart du checkpoint sans re-telecharger ni re-parser toute la source.
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from sncf_agent.config import settings
from sncf_agent.ingestion.chunking import Chunk

log = structlog.get_logger(__name__)


def checkpoint_path(name: str) -> Path:
    """Chemin du checkpoint pour un lot d'ingestion donne (ex. dataset_id)."""
    settings.ensure_data_dirs()
    return settings.checkpoints_dir / f"{name}.json"


def save_chunks(chunks: list[Chunk], name: str) -> Path:
    """Ecrit les chunks dans un checkpoint JSON. Renvoie le chemin."""
    path = checkpoint_path(name)
    payload = [c.to_dict() for c in chunks]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("checkpoint_ecrit", name=name, n_chunks=len(chunks), path=str(path))
    return path


def load_chunks(name: str) -> list[Chunk]:
    """Relit les chunks depuis un checkpoint JSON."""
    path = checkpoint_path(name)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint absent : {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = [Chunk.from_dict(d) for d in payload]
    log.info("checkpoint_lu", name=name, n_chunks=len(chunks), path=str(path))
    return chunks
