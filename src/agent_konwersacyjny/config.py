"""Application settings, read from environment variables (prefix `AGENT_`) and `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Models -----------------------------------------------------------------
    # Any `provider:model` string understood by `langchain.chat_models.init_chat_model`.
    model: str = "anthropic:claude-haiku-5-5"
    orchestrator_model: str | None = None
    rag_model: str | None = None
    finance_model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096

    # --- Data -------------------------------------------------------------------
    docs_dir: Path = PROJECT_ROOT / "data" / "docs"
    finance_dir: Path = PROJECT_ROOT / "data" / "finance"
    skills_dir: Path = PROJECT_ROOT / "skills"
    config_dir: Path = PROJECT_ROOT / "config"

    # --- Storage ----------------------------------------------------------------
    storage_dir: Path = PROJECT_ROOT / ".agent_data"
    default_user_id: str = "default"

    # --- RAG --------------------------------------------------------------------
    chunk_size: int = 1200
    chunk_overlap: int = 150
    # Optional `provider:model` for `langchain.embeddings.init_embeddings`
    # (e.g. `openai:text-embedding-3-small`). Empty = BM25 only, no embedding cost.
    embeddings_model: str | None = None

    # --- Conversation -----------------------------------------------------------
    # Upper bound of agent-to-agent handoffs inside one user turn (loop guard).
    max_handoffs_per_turn: int = Field(default=3, ge=1)

    # --- MCP --------------------------------------------------------------------
    mcp_config: Path = PROJECT_ROOT / "mcp_servers.json"

    @property
    def checkpoint_db(self) -> Path:
        return self.storage_dir / "checkpoints.sqlite"

    @property
    def store_db(self) -> Path:
        return self.storage_dir / "store.sqlite"

    def model_for(self, agent_name: str) -> str:
        override = {
            "orchestrator": self.orchestrator_model,
            "rag": self.rag_model,
            "finance": self.finance_model,
        }.get(agent_name)
        return override or self.model


@lru_cache
def get_settings() -> Settings:
    return Settings()
