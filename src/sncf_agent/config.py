"""Configuration centralisee du projet.

Charge les variables depuis .env via pydantic-settings. Toute cle secrete passe
par ici, jamais en dur dans le code.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Racine du projet (deux niveaux au-dessus de ce fichier : src/sncf_agent/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Charge le .env dans os.environ pour que les clients tiers (Groq, OpenAI, LangSmith...)
# qui lisent os.environ y trouvent les cles, en plus de l'objet Settings type ci-dessous.
load_dotenv(PROJECT_ROOT / ".env")

# Pont Logfire : la cle est stockee sous PYDANTIC_LOGFIRE_API_KEY dans le .env, mais le
# SDK Logfire lit LOGFIRE_TOKEN. On mappe sans ecraser un LOGFIRE_TOKEN deja defini.
if os.environ.get("PYDANTIC_LOGFIRE_API_KEY") and not os.environ.get("LOGFIRE_TOKEN"):
    os.environ["LOGFIRE_TOKEN"] = os.environ["PYDANTIC_LOGFIRE_API_KEY"]


class Settings(BaseSettings):
    """Parametres applicatifs, surcharges par le .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    # App interactive sur Ollama local (qwen2.5:7b) : gratuit, sans quota, tool-calling
    # fiable. Groq (llama-3.3-70b-versatile) reste dispo mais son free tier est limite a
    # 100k tokens/JOUR, vite epuise. Surchargables via .env (LLM_PROVIDER / LLM_MODEL)
    # sans toucher au code. Penses pour langchain.init_chat_model(model, model_provider=...).
    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")  # ollama | groq | openai
    llm_model: str = Field(default="qwen2.5:7b", alias="LLM_MODEL")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    # Token HuggingFace : optionnel (modeles d'embedding gated). Les embeddings tournent
    # en local via sentence-transformers, donc sans cout d'API.
    huggingfacehub_api_token: str = Field(default="", alias="HUGGINGFACEHUB_API_TOKEN")

    # --- Donnees SNCF : DEUX sources distinctes, ne pas confondre ---
    # 1. data.sncf.com (plateforme Opendatasoft) : 167 datasets ouverts, fonctionne
    #    SANS cle (quota partage 50k/jour). Une cle gratuite et instantanee releve le
    #    quota. Optionnelle. C'est la source du prototype.
    opendata_base_url: str = "https://data.sncf.com/api/explore/v2.1"
    opendata_api_key: str = Field(default="", alias="OPENDATA_API_KEY")
    # 2. api.sncf.com (Navitia) : itineraires + perturbations TEMPS REEL. Necessite le
    #    token demande via formulaire (en attente). Branche derriere NavitiaConnector,
    #    integre plus tard sans toucher au reste du pipeline.
    navitia_base_url: str = "https://api.sncf.com/v1"
    sncf_api_key: str = Field(default="", alias="SNCF_API_KEY")

    # --- Embeddings (choix a figer en Semaine 0 : e5 ou BGE-M3) ---
    embedding_model: str = "intfloat/multilingual-e5-large"

    # --- Chunking ---
    # 1400 : un enregistrement structure enrichi (~1000 car.) tient dans UN chunk (pas
    # de second chunk orphelin sans en-tete trajet/periode), tout en restant sous la
    # fenetre de 512 tokens d'e5 (~1800 car. en francais).
    chunk_size: int = 1400
    chunk_overlap: int = 150

    # --- Chemins des donnees ---
    raw_dir: Path = DATA_DIR / "raw"
    checkpoints_dir: Path = DATA_DIR / "checkpoints"
    index_dir: Path = DATA_DIR / "index"

    # --- Retrieval (agent) ---
    retrieval_k: int = 4  # nombre de passages recuperes par requete
    # Seuil de pertinence (distance L2, FAISS + e5 normalise : plus petit = plus proche).
    # Mesure sur les 3 corpus : pertinent <= 0.253, hors-sujet >= 0.391. Un passage
    # au-dela du seuil n'est pas transmis a l'agent, qui peut alors refuser honnetement.
    # NB : a recalibrer si on change d'embedding, et a convertir si on passe a Qdrant
    # (qui renvoie une similarite cosinus, plus grand = plus proche).
    retrieval_score_threshold: float = 0.35

    # --- LLM d'evaluation (RAGAS) ---
    # L'eval est gourmande en tokens (juge appele plusieurs fois par question). On la
    # sort du provider de l'app : OpenAI (fiable, supporte n>1) plutot que Groq (quota
    # journalier 100k tokens en free tier, vite atteint par RAGAS).
    eval_llm_provider: str = "openai"
    eval_llm_model: str = "gpt-4o-mini"

    # --- Backend / Frontend ---
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    backend_url: str = Field(default="http://127.0.0.1:8000", alias="BACKEND_URL")

    # --- Observabilite (LangSmith) ---
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="sncf-agent", alias="LANGSMITH_PROJECT")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com", alias="LANGSMITH_ENDPOINT"
    )

    def ensure_data_dirs(self) -> None:
        """Cree les repertoires de donnees s'ils n'existent pas."""
        for directory in (self.raw_dir, self.checkpoints_dir, self.index_dir):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
