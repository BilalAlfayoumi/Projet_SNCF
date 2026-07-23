"""Orchestrateur du pipeline d'ingestion, utilisable en CLI.

Enchaine : extraction (OpenDataConnector) -> parsing -> chunking -> checkpoint JSON
-> embedding -> index FAISS. Le checkpoint permet de reprendre a l'embedding sans
re-telecharger la source (option --from-checkpoint).

Exemples :
    uv run python -m sncf_agent.ingestion.pipeline regularite-mensuelle-tgv-aqst \\
        --where "gare_depart='PARIS MONTPARNASSE'"
    uv run python -m sncf_agent.ingestion.pipeline tarifs-intercites --from-checkpoint
"""

from __future__ import annotations

import argparse

import structlog

from sncf_agent.ingestion.checkpoint import load_chunks, save_chunks
from sncf_agent.ingestion.chunking import chunk_passages
from sncf_agent.ingestion.embedding import build_faiss
from sncf_agent.ingestion.parsing import to_passage
from sncf_agent.ingestion.sources import OpenDataConnector

log = structlog.get_logger(__name__)


def ingest_dataset(
    dataset_id: str,
    where: str | None = None,
    select: str | None = None,
    order_by: str | None = None,
    from_checkpoint: bool = False,
    embed: bool = True,
) -> None:
    """Ingere un dataset open data jusqu'a l'index FAISS."""
    if from_checkpoint:
        chunks = load_chunks(dataset_id)
        log.info("reprise_depuis_checkpoint", dataset=dataset_id, n_chunks=len(chunks))
    else:
        with OpenDataConnector() as src:
            records = list(src.fetch(dataset_id, select=select, where=where, order_by=order_by))
        log.info("extraction_terminee", dataset=dataset_id, n_records=len(records))

        passages = [p for r in records if (p := to_passage(r)) is not None]
        log.info("parsing_termine", dataset=dataset_id, n_passages=len(passages))

        chunks = chunk_passages(passages)
        log.info("chunking_termine", dataset=dataset_id, n_chunks=len(chunks))

        save_chunks(chunks, dataset_id)

    if embed:
        build_faiss(chunks, dataset_id)
        log.info("ingestion_complete", dataset=dataset_id, n_chunks=len(chunks))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline d'ingestion open data SNCF")
    parser.add_argument("dataset_id", help="dataset_id data.sncf.com (ex. tarifs-intercites)")
    parser.add_argument("--where", default=None, help="clause ODSQL de filtrage a la source")
    parser.add_argument("--select", default=None, help="clause ODSQL de selection de champs")
    parser.add_argument("--order-by", default=None, help="clause ODSQL de tri")
    parser.add_argument(
        "--from-checkpoint",
        action="store_true",
        help="repartir du checkpoint JSON (saute extraction/parsing/chunking)",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="s'arreter au checkpoint, sans construire l'index FAISS",
    )
    args = parser.parse_args()

    ingest_dataset(
        dataset_id=args.dataset_id,
        where=args.where,
        select=args.select,
        order_by=args.order_by,
        from_checkpoint=args.from_checkpoint,
        embed=not args.no_embed,
    )


if __name__ == "__main__":
    main()
