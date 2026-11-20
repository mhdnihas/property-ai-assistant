"""Application settings, loaded from environment variables (and `.env` locally)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent

# The value shipped in .env.example; keep the two in sync.
PLACEHOLDER_KEY = "sk-or-v1-your-key-here"

DEFAULT_MODELS = (
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3.5-super-vl:free",
    "google/gemma-4-31b-it:free",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM (OpenRouter) ---
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Comma-separated in the environment; tried in order until one responds.
    openrouter_models: str = ",".join(DEFAULT_MODELS)
    llm_temperature: float = Field(0.2, ge=0, le=2)
    llm_max_tokens: int = Field(1500, ge=64, le=4096)
    llm_disable_reasoning: bool = True  # thinking tokens eat the max_tokens budget
    llm_timeout_s: float = Field(60.0, gt=0)
    # Sent to OpenRouter as HTTP-Referer (app attribution).
    public_url: str = "https://property-ai-assistant.onrender.com"
    app_title: str = "Property AI Assistant"

    # --- Retrieval ---
    index_dir: Path = ROOT_DIR / "data" / "index"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: Path = ROOT_DIR / ".cache" / "fastembed"
    enable_embeddings: bool = True
    top_k: int = Field(8, ge=1, le=20)

    # --- API guard rails ---
    max_message_chars: int = Field(1000, ge=10)
    max_history_turns: int = Field(6, ge=0, le=20)
    rate_limit_per_minute: int = Field(20, ge=1)
    log_level: str = "INFO"

    @field_validator("openrouter_api_key", mode="before")
    @classmethod
    def _blank_key_is_none(cls, v: object) -> object:
        """Treat an empty value or the untouched .env.example placeholder as "no key",
        so the app reports "AI not configured" instead of failing on every request."""
        if isinstance(v, str) and (not v.strip() or v.strip() == PLACEHOLDER_KEY):
            return None
        return v

    @property
    def models(self) -> list[str]:
        return [m.strip() for m in self.openrouter_models.split(",") if m.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
