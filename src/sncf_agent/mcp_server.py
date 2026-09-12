"""Serveur MCP exposant l'agent RAG SNCF comme outils, en plus de l'API REST FastAPI.

Additif : ne remplace ni ne modifie `backend/app.py`. Les deux serveurs partagent la
meme configuration (`config.settings`) et les memes fonctions RAG (`rag/agent.py`,
`rag/vectorstore.py`, `rag/guardrails.py`) ; ils peuvent tourner simultanement, sur
des ports differents.

Deux outils :
- search_sncf_corpus : retrieval brut (meme fonction que l'endpoint /search).
- ask_sncf_agent     : reponse de l'agent multi-corpus (meme fonction que /chat),
  avec les MEMES guardrails d'entree (injection, PII) et de sortie (sources citees).

Note de version du SDK : dans `mcp` 2.x, la classe `FastMCP` (v1) a ete renommee
`MCPServer` (meme interface : constructeur, decorateur `@tool()`, `.run(transport=...)`)
- voir https://py.sdk.modelcontextprotocol.io/v2/migration/#fastmcp-renamed-to-mcpserver.
C'est cette version qui est installee ici (uv add mcp -> mcp>=2).
"""

from __future__ import annotations

from typing import Annotated

import structlog
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from sncf_agent.config import settings
from sncf_agent.ingestion.embedding import list_indexes
from sncf_agent.rag.agent import ask_with_contexts, build_multi_agent
from sncf_agent.rag.guardrails import scan_input, verify_sources
from sncf_agent.rag.vectorstore import search_with_scores

log = structlog.get_logger(__name__)

mcp = MCPServer(
    name="sncf-agent",
    version="0.1.0",
    instructions=(
        "Outils de recherche et de question-reponse sur les donnees ouvertes SNCF "
        "(regularite des TGV, tarifs Intercites, regles de l'abonnement MAX JEUNE)."
    ),
)

# Modeles par defaut par provider quand ask_sncf_agent ne fournit qu'un nom de provider
# (pas de cle BYOK ici : ce serveur MCP tourne cote serveur, avec les cles configurees
# dans .env). Alignes sur les modeles deja utilises ailleurs dans le projet pour ces
# providers (backend/app.py::_BYOK_MODELS pour groq/openai, config.py pour ollama).
_DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "ollama": settings.llm_model,
}

# Cache des agents multi-corpus construits, par provider. Meme logique que
# backend/app.py::_AGENTS : evite de reconstruire l'agent (et son LLM) a chaque appel.
_AGENTS: dict[str, object] = {}


def _get_multi_agent(provider: str):
    if provider not in _AGENTS:
        log.info("construction_agent_mcp", provider=provider)
        model = _DEFAULT_MODELS.get(provider)
        _AGENTS[provider] = build_multi_agent(model=model, provider=provider)
    return _AGENTS[provider]


class SearchHit(BaseModel):
    """Un passage recupere, avec sa source et son score de similarite.

    Meme forme que `SearchHit` dans `backend/app.py` (endpoint /search), pour que les
    deux interfaces (REST et MCP) exposent le meme contrat de donnees.
    """

    text: str
    source: str
    score: float


@mcp.tool()
def search_sncf_corpus(
    query: Annotated[
        str,
        Field(min_length=1, max_length=1000, description="Question ou mots-cles a rechercher."),
    ],
    k: Annotated[
        int, Field(default=5, ge=1, le=20, description="Nombre de passages a renvoyer.")
    ] = 5,
) -> list[SearchHit]:
    """Recherche des passages pertinents dans les corpus documentaires SNCF (regularite
    des TGV, tarifs Intercites, regles de l'abonnement MAX JEUNE).

    Retrieval BRUT (pas de generation LLM) : renvoie les k passages les plus proches de
    la requete, avec leur source et leur score de similarite, tous corpus confondus.
    Utile pour inspecter ce que "voit" l'agent avant de lui poser une question, ou pour
    recuperer directement des extraits sourcés sans passer par une reponse generee.

    Le sens du score depend du backend vectoriel configure sur ce serveur : distance L2
    (plus PETIT = plus proche) pour FAISS, similarite cosinus (plus GRAND = plus proche)
    pour Qdrant. Aucun filtre de pertinence n'est applique ici (contrairement a l'agent) :
    les k meilleurs candidats sont renvoyes meme s'ils sont eloignes de la requete.
    """
    indexes = list_indexes()
    if not indexes:
        raise ToolError(
            "Aucun corpus indexe disponible sur ce serveur : lancer une ingestion "
            "(sncf_agent.ingestion.pipeline / .docs) avant de rechercher."
        )

    hits: list[tuple[str, str, float]] = []
    for index_name in indexes:
        hits.extend(search_with_scores(index_name, query, k=k))

    # Un seul backend vectoriel actif pour tout le serveur (settings.vector_backend) :
    # le sens de tri est donc le meme pour tous les corpus fusionnes ici.
    hits.sort(key=lambda h: h[2], reverse=settings.is_qdrant)
    return [SearchHit(text=text, source=source, score=score) for text, source, score in hits[:k]]


@mcp.tool()
def ask_sncf_agent(
    question: Annotated[
        str,
        Field(min_length=1, max_length=1000, description="Question en francais sur la SNCF."),
    ],
    provider: Annotated[
        str,
        Field(
            pattern="^(groq|openai|ollama)$",
            description="Provider LLM (cle deja configuree cote serveur, pas de BYOK ici).",
        ),
    ] = "groq",
) -> str:
    """Pose une question a l'agent conversationnel RAG SNCF (routage multi-corpus).

    L'agent choisit lui-meme le corpus pertinent (regularite TGV, tarifs Intercites,
    regles MAX JEUNE), cite ses sources, et refuse honnetement si les donnees ne
    couvrent pas la question plutot que d'inventer une reponse.

    Applique les MEMES guardrails que l'endpoint REST /chat :
    - en entree : rejette les tentatives d'injection de prompt et les questions
      contenant des donnees personnelles (email, telephone, IBAN, carte bancaire...) ;
    - en sortie : signale si une source citee dans la reponse n'a pas ete retrouvee
      dans les passages effectivement recuperes (citation potentiellement inventee).

    provider : "groq" (defaut, rapide et gratuit en usage modere), "openai" ou "ollama"
    (local). La cle API correspondante doit deja etre configuree sur ce serveur (.env) ;
    ce n'est pas un mode BYOK (pas de cle transmise par l'appelant).
    """
    verdict = scan_input(question)
    if verdict.blocked:
        if verdict.reason == "pii":
            log.warning("guardrail_pii_detecte_mcp", types=verdict.pii_types)
        else:
            log.warning("guardrail_injection_detecte_mcp", question=question[:80])
        raise ToolError(verdict.message)

    try:
        agent = _get_multi_agent(provider)
    except RuntimeError as exc:
        # build_multi_agent leve RuntimeError si aucun index n'est disponible.
        raise ToolError(str(exc)) from exc

    try:
        answer, contexts = ask_with_contexts(agent, question)
    except Exception as exc:  # noqa: BLE001 - traduit en erreur MCP explicite ci-dessous
        msg = str(exc)
        if "401" in msg or "invalid_api_key" in msg.lower() or "incorrect api key" in msg.lower():
            raise ToolError(
                f"Cle API invalide ou manquante pour le provider {provider} (a configurer cote "
                "serveur)."
            ) from exc
        if "rate_limit" in msg or "429" in msg:
            log.warning("llm_quota_atteint_mcp", provider=provider, error=msg[:200])
            raise ToolError(
                f"Quota du provider LLM ({provider}) atteint. Reessayer plus tard ou "
                "choisir un autre provider (groq | openai | ollama)."
            ) from exc
        log.error("erreur_agent_mcp", provider=provider, error=msg[:300])
        raise ToolError(f"Erreur de l'agent : {msg[:200]}") from exc

    controle = verify_sources(answer, contexts)
    if not controle.ok:
        log.warning("guardrail_source_non_verifiee_mcp", urls=controle.unsupported)
        answer += (
            "\n\n_(Note : une ou plusieurs sources citees n'ont pas pu etre verifiees "
            "dans la base documentaire ; verifiez-les avant de vous y fier.)_"
        )
    return answer


def main() -> None:
    log.info("mcp_server_demarre", host=settings.api_host, port=settings.mcp_port)
    mcp.run(transport="streamable-http", host=settings.api_host, port=settings.mcp_port)


if __name__ == "__main__":
    main()
