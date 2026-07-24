"""Fabrique centralisee du modele de chat, avec routage Portkey.

Trois chemins, un seul point de decision :
1. BYOK (cle fournie par le visiteur)  -> appel DIRECT au provider avec sa cle.
   Sa cle ne passe JAMAIS par Portkey (vie privee, ses tokens restent a lui).
2. Provider par defaut + Portkey actif  -> via le gateway Portkey (fallback entre
   providers, cache semantique, observabilite des couts), avec les cles du serveur
   stockees cote Portkey (Virtual Keys).
3. Provider par defaut sans Portkey     -> appel direct classique (Ollama, Groq...).

Ce module isole toute la logique provider : agent et eval passent par get_chat_model().
"""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from sncf_agent.config import settings


def _ollama_kwargs() -> dict:
    # Reglages locaux Ollama : deterministe, modele garde chaud, contexte borne, sortie
    # plafonnee (voir historique des correctifs de latence).
    return {"temperature": 0, "keep_alive": "2h", "num_ctx": 6144, "num_predict": 400}


def get_chat_model(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0,
) -> BaseChatModel:
    """Construit le modele de chat selon le contexte (BYOK / Portkey / direct)."""
    effective_provider = provider or settings.llm_provider
    effective_model = model or settings.llm_model

    # 1. BYOK : appel direct avec la cle du visiteur, jamais via Portkey.
    if api_key:
        return init_chat_model(
            effective_model,
            model_provider=effective_provider,
            temperature=temperature,
            api_key=api_key,
        )

    # 2. Provider par defaut + Portkey actif : passe par le gateway (OpenAI-compatible).
    #    La Config Portkey porte le fallback (Groq->OpenAI) et le cache ; le code ne
    #    fait que pointer base_url + headers. On garde init_chat_model('openai') car le
    #    gateway expose une API compatible OpenAI.
    if settings.portkey_active:
        from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

        return init_chat_model(
            effective_model,
            model_provider="openai",
            temperature=temperature,
            base_url=PORTKEY_GATEWAY_URL,
            api_key="via-portkey",  # ignore : Portkey utilise ses Virtual Keys
            default_headers=createHeaders(
                api_key=settings.portkey_api_key,
                config=settings.portkey_config,
            ),
        )

    # 3. Direct classique (Ollama local, ou Groq/OpenAI sans gateway).
    kwargs = {"temperature": temperature}
    if effective_provider == "ollama":
        kwargs.update(_ollama_kwargs())
    return init_chat_model(effective_model, model_provider=effective_provider, **kwargs)
