"""Ingestion de corpus DOCUMENTAIRES (pages web + PDF locaux) vers FAISS.

Complementaire du pipeline datasets (pipeline.py) : ici la source est du texte long
(regles d'abonnement, CGV, FAQ), la ou les datasets sont des enregistrements
structures. C'est sur ce corpus que le chunking joue vraiment son role.

Deux provenances par corpus :
- URLs publiques accessibles aux scripts (ex. FAQ sncf-connect.com).
- PDF deposes MANUELLEMENT dans data/raw/docs/<corpus>/ : certains sites SNCF
  (sncf-voyageurs.com, dont les CGV) repondent 403 aux scripts ; on les telecharge
  au navigateur et le pipeline les ingere depuis le disque.

Usage :
    uv run python -m sncf_agent.ingestion.docs abonnement-max-jeune
"""

from __future__ import annotations

import argparse
import re

import httpx
import structlog
from bs4 import BeautifulSoup

from sncf_agent.config import settings
from sncf_agent.ingestion.checkpoint import save_chunks
from sncf_agent.ingestion.chunking import chunk_passages
from sncf_agent.ingestion.embedding import build_index
from sncf_agent.ingestion.parsing import Passage

log = structlog.get_logger(__name__)

# Corpus documentaires connus : nom -> URLs a recuperer. Les PDF manuels vont dans
# data/raw/docs/<nom>/. Le nom sert aussi de nom d'index FAISS et de checkpoint.
CORPORA: dict[str, list[str]] = {
    # NB : les CGV PDF (sncf-voyageurs.com) repondent 403 aux scripts. Les telecharger
    # au navigateur et les deposer dans data/raw/docs/abonnement-max-jeune/ : le
    # pipeline les ingerera automatiquement au prochain run.
    "abonnement-max-jeune": [
        # FAQ officielle (accessible aux scripts, contrairement a sncf-voyageurs.com)
        "https://www.sncf-connect.com/aide/max-jeune",
    ],
}

_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def _html_to_text(html: str) -> str:
    """Extrait le texte utile d'une page HTML (sans scripts, styles, menus)."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Compacte les lignes vides multiples et les espaces
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def _pdf_to_text(path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def fetch_corpus(corpus: str) -> list[Passage]:
    """Recupere les documents d'un corpus (web + PDF locaux) en Passages."""
    if corpus not in CORPORA:
        raise SystemExit(f"corpus inconnu : {corpus}. Connus : {sorted(CORPORA)}")

    passages: list[Passage] = []

    # 1. Pages web
    with httpx.Client(headers=_UA, follow_redirects=True, timeout=30) as client:
        for url in CORPORA[corpus]:
            resp = client.get(url)
            if resp.status_code != 200:
                log.warning("page_inaccessible", url=url, status=resp.status_code)
                continue
            text = _html_to_text(resp.text)
            titre = re.search(r"<title>(.*?)</title>", resp.text, re.S)
            titre_txt = titre.group(1).strip() if titre else url
            passages.append(
                Passage(
                    text=text,
                    source=f"webdoc:{url}",
                    metadata={"dataset": corpus, "titre": titre_txt, "url": url},
                )
            )
            log.info("page_recuperee", url=url, caracteres=len(text))

    # 2. PDF deposes manuellement
    pdf_dir = settings.raw_dir / "docs" / corpus
    if pdf_dir.exists():
        for pdf in sorted(pdf_dir.glob("*.pdf")):
            text = _pdf_to_text(pdf)
            passages.append(
                Passage(
                    text=text,
                    source=f"pdf:{pdf.name}",
                    metadata={"dataset": corpus, "titre": pdf.stem, "fichier": pdf.name},
                )
            )
            log.info("pdf_lu", fichier=pdf.name, caracteres=len(text))
    else:
        log.info("pas_de_pdf_locaux", dossier=str(pdf_dir))

    return passages


def ingest_docs(corpus: str, embed: bool = True) -> None:
    """Pipeline complet : documents -> chunking -> checkpoint -> FAISS."""
    passages = fetch_corpus(corpus)
    if not passages:
        raise SystemExit("aucun document recupere, rien a ingerer")
    log.info("documents_recuperes", corpus=corpus, n_docs=len(passages))

    chunks = chunk_passages(passages)
    log.info("chunking_termine", corpus=corpus, n_chunks=len(chunks))
    save_chunks(chunks, corpus)

    if embed:
        build_index(chunks, corpus)
        log.info("ingestion_docs_complete", corpus=corpus, n_chunks=len(chunks))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestion de corpus documentaires SNCF")
    parser.add_argument("corpus", choices=sorted(CORPORA), help="corpus a ingerer")
    parser.add_argument("--no-embed", action="store_true", help="s'arreter au checkpoint")
    args = parser.parse_args()
    ingest_docs(args.corpus, embed=not args.no_embed)


if __name__ == "__main__":
    main()
