"""Chat model factory. Every agent can use a different `provider:model` string."""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from agent_konwersacyjny.config import Settings


def build_model(settings: Settings, agent_name: str) -> BaseChatModel:
    return init_chat_model(
        settings.model_for(agent_name),
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )
