"""Configuration centralisee du projet.

Charge les variables depuis .env via pydantic-settings. Toute cle secrete passe
par ici, jamais en dur dans le code.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Racine du projet (deux niveaux au-dessus de ce fichier : src/sncf_agent/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    """Parametres applicatifs, surcharges par le .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    # On demarre avec Groq (rapide, peu couteux) ; OpenAI en secours ; Anthropic plus tard.
    # provider + model sont penses pour langchain.init_chat_model(model, model_provider=...).
    llm_provider: str = "groq"  # "groq" | "openai" | "anthropic"
    llm_model: str = "llama-3.3-70b-versatile"  # a ajuster selon le catalogue Groq du moment
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    # Token HuggingFace : optionnel (modeles d'embedding gated). Les embeddings tournent
    # en local via sentence-transformers, donc sans cout d'API.
    huggingfacehub_api_token: str = Field(default="", alias="HUGGINGFACEHUB_API_TOKEN")

    # --- API open data SNCF ---
    sncf_api_key: str = Field(default="", alias="SNCF_API_KEY")

    # --- Embeddings (choix a figer en Semaine 0 : e5 ou BGE-M3) ---
    embedding_model: str = "intfloat/multilingual-e5-large"

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- Chemins des donnees ---
    raw_dir: Path = DATA_DIR / "raw"
    checkpoints_dir: Path = DATA_DIR / "checkpoints"
    index_dir: Path = DATA_DIR / "index"

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
