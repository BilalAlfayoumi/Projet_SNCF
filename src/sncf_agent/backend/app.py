"""API FastAPI exposant l'agent RAG SNCF.

Endpoints penses pour tester ET optimiser :
- POST /chat   : reponse de l'agent (teste prompt + generation).
- POST /search : retrieval brut avec scores (teste chunking + embedding + retrieval).
- GET  /indexes: index FAISS disponibles.
- GET  /health : sonde de vie.

Le modele d'embedding (~2 Go) et les agents sont charges paresseusement puis mis en
cache par index, pour ne payer le cout qu'une fois.
"""

from __future__ import annotations

import re
from contextlib import asynccontextmanager
from typing import Any

import logfire
import structlog
from fastapi import FastAPI, HTTPException
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from sncf_agent.config import settings
from sncf_agent.ingestion.embedding import get_embeddings, list_indexes
from sncf_agent.rag.agent import ask, build_agent, build_multi_agent
from sncf_agent.rag.vectorstore import search_with_scores

log = structlog.get_logger(__name__)

# Valeur speciale d'index : agent multi-corpus (route seul vers le bon corpus).
MULTI_INDEX = "multi"

# Guardrail d'entree (prototype, regles simples comme prevu au plan ; le gateway LLM
# prendra le relais en version scalable) : motifs d'injection de prompt courants.
_INJECTION_RE = re.compile(
    r"(ignore\s+(les|tes|toutes)\s+.{0,20}instructions"
    r"|oublie\s+(les|tes)\s+.{0,20}instructions"
    r"|system\s*prompt|prompt\s*syst[eè]me"
    r"|tu\s+es\s+maintenant\s+"
    r"|reveal\s+your\s+instructions)",
    re.IGNORECASE,
)

_REPONSE_GUARDRAIL = (
    "Je ne peux pas traiter cette demande. Posez-moi une question sur les donnees "
    "SNCF : regularite des TGV, tarifs Intercites ou abonnement MAX JEUNE."
)

# Cache des agents construits, par (index, k). Evite de reconstruire a chaque requete.
_AGENTS: dict[tuple[str, int], CompiledStateGraph] = {}


def _get_agent(index_name: str, k: int) -> CompiledStateGraph:
    key = (index_name, k)
    if key not in _AGENTS:
        log.info("construction_agent", index=index_name, k=k)
        if index_name == MULTI_INDEX:
            _AGENTS[key] = build_multi_agent(k=k)
        else:
            _AGENTS[key] = build_agent(index_name, k=k)
    return _AGENTS[key]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Precharge le modele d'embedding au demarrage pour que la 1re requete soit rapide.
    log.info("prechargement_embeddings")
    emb = get_embeddings()
    # ECHAUFFEMENT : la premiere inference paie l'initialisation torch (mesure en prod :
    # 143 s sur 2 vCPU partages !). On l'absorbe ici, pendant le boot, pour que la
    # premiere vraie question de l'utilisateur soit deja rapide.
    import time

    t0 = time.time()
    emb.embed_query("echauffement du modele")
    log.info("echauffement_embeddings_fini", duree_s=round(time.time() - t0, 1))
    # Prechauffe aussi le LLM local : charge le modele en GPU des maintenant (sinon la
    # premiere question paie le chargement, ~20-30 s avec Ollama).
    if settings.llm_provider == "ollama":
        try:
            from langchain.chat_models import init_chat_model

            log.info("prechauffage_llm", model=settings.llm_model)
            init_chat_model(
                settings.llm_model,
                model_provider="ollama",
                keep_alive="2h",
            ).invoke("ok")
        except Exception as exc:  # noqa: BLE001
            log.warning("prechauffage_llm_echoue", error=str(exc)[:150])
    log.info("backend_pret", indexes=list_indexes())
    yield
    _AGENTS.clear()


app = FastAPI(title="Agent RAG SNCF", version="0.1.0", lifespan=lifespan)

# --- Observabilite applicative : Pydantic Logfire ---
# Complementaire de LangSmith (qui trace le cote LLM/agent) : Logfire trace le cote
# application (requetes HTTP, erreurs, latences, metriques systeme).
# "if-token-present" : no-op si LOGFIRE_TOKEN absent (CI, autre poste), donc sans risque.
logfire.configure(service_name="sncf-agent-backend", send_to_logfire="if-token-present")
logfire.instrument_fastapi(app)
logfire.instrument_system_metrics()


# Modeles par defaut pour le mode BYOK (cle API fournie par l'utilisateur).
_BYOK_MODELS = {"groq": "llama-3.3-70b-versatile", "openai": "gpt-4o-mini"}


class ChatRequest(BaseModel):
    # Guardrail : borne la taille de la question (anti-saturation du contexte).
    question: str = Field(min_length=1, max_length=1000)
    index: str
    k: int = Field(default=settings.retrieval_k, ge=1, le=20)
    # BYOK : le visiteur fournit SA cle API (demo publique sans facturer le serveur).
    # La cle transite par requete, n'est ni stockee, ni loguee, ni mise en cache.
    provider: str | None = Field(default=None, pattern="^(groq|openai)$")
    api_key: str | None = Field(default=None, min_length=8, max_length=256)


class ChatResponse(BaseModel):
    answer: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    index: str
    k: int = Field(default=5, ge=1, le=20)


class SearchHit(BaseModel):
    text: str
    source: str
    score: float


class SearchResponse(BaseModel):
    hits: list[SearchHit]


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "indexes": list_indexes()}


@app.get("/indexes")
def indexes() -> dict[str, list[str]]:
    return {"indexes": list_indexes()}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    """Retrieval brut avec scores, pour inspecter la qualite du retrieval."""
    if req.index not in list_indexes():
        raise HTTPException(status_code=404, detail=f"index inconnu : {req.index}")
    hits = search_with_scores(req.index, req.query, k=req.k)
    return SearchResponse(
        hits=[SearchHit(text=t, source=s, score=score) for t, s, score in hits]
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Reponse de l'agent RAG (index precis, ou "multi" pour l'agent multi-corpus)."""
    if req.index != MULTI_INDEX and req.index not in list_indexes():
        raise HTTPException(status_code=404, detail=f"index inconnu : {req.index}")
    # Guardrail d'entree : motifs d'injection de prompt -> message de repli clair.
    if _INJECTION_RE.search(req.question):
        log.warning("guardrail_injection_detecte", question=req.question[:80])
        return ChatResponse(answer=_REPONSE_GUARDRAIL)

    if req.api_key:
        # Mode BYOK : agent construit a la volee avec la cle du visiteur, PAS de cache
        # (le cache retiendrait la cle en memoire). Le provider est requis.
        if not req.provider:
            raise HTTPException(status_code=400, detail="provider requis avec api_key")
        log.info("chat_byok", provider=req.provider, index=req.index)  # jamais la cle
        if req.index == MULTI_INDEX:
            agent = build_multi_agent(
                k=req.k,
                provider=req.provider,
                model=_BYOK_MODELS[req.provider],
                api_key=req.api_key,
            )
        else:
            agent = build_agent(
                req.index,
                k=req.k,
                provider=req.provider,
                model=_BYOK_MODELS[req.provider],
                api_key=req.api_key,
            )
    else:
        # Mode "defaut serveur" : si le provider par defaut exige une cle absente
        # (deploiement BYOK-only), guider le visiteur au lieu d'une erreur brute.
        if settings.llm_provider == "groq" and not settings.groq_api_key:
            return ChatResponse(
                answer=(
                    "Ce serveur de demonstration fonctionne avec VOTRE cle API : creez "
                    "une cle Groq gratuite sur console.groq.com, puis saisissez-la dans "
                    "les reglages (barre laterale, section Cle API personnelle)."
                )
            )
        agent = _get_agent(req.index, req.k)
    try:
        answer = ask(agent, req.question)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        # Cle API invalide (mode BYOK notamment) : erreur explicite pour l'utilisateur.
        if "401" in msg or "invalid_api_key" in msg.lower() or "incorrect api key" in msg.lower():
            raise HTTPException(
                status_code=401,
                detail="Cle API invalide ou expiree. Verifiez la cle saisie dans les reglages.",
            ) from exc
        # Cas frequent en dev : quota journalier du provider LLM atteint.
        if "rate_limit" in msg or "429" in msg:
            log.warning("llm_quota_atteint", provider=settings.llm_provider, error=msg[:200])
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Quota du provider LLM ({settings.llm_provider}) atteint. "
                    "Basculer l'app sur un autre provider via LLM_PROVIDER dans .env "
                    "(ex. openai), ou reessayer plus tard."
                ),
            ) from exc
        log.error("erreur_agent", error=msg[:300])
        raise HTTPException(status_code=500, detail=f"Erreur de l'agent : {msg[:200]}") from exc
    return ChatResponse(answer=answer)
