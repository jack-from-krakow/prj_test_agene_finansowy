"""Virtual filesystem shared by all agents (deepagents `CompositeBackend`).

| path           | backend                         | lifetime / access                    |
|----------------|---------------------------------|--------------------------------------|
| `/`            | StateBackend                    | scratch, one agent run               |
| `/workspace/`  | StoreBackend (user, thread)     | session memory, read-write           |
| `/memories/`   | StoreBackend (user)             | procedural memory, read-write        |
| `/finance/`    | FilesystemBackend(finance_dir)  | read-only                            |
| `/docs/`       | FilesystemBackend(docs_dir)     | read-only                            |
| `/skills/`     | FilesystemBackend(skills_dir)   | read-only                            |
| `/config/`     | FilesystemBackend(config_dir)   | read-only                            |
"""

from __future__ import annotations

from deepagents import FilesystemPermission
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.state import StateBackend
from deepagents.backends.store import StoreBackend
from langgraph.store.base import BaseStore

from agent_konwersacyjny.config import Settings
from agent_konwersacyjny.memory.namespaces import procedural_fs_ns, workspace_fs_ns

READ_ONLY_PREFIXES = ("/finance/", "/docs/", "/skills/", "/config/")
PROCEDURAL_MEMORY_PATH = "/memories/AGENTS.md"
STATIC_INSTRUCTIONS_PATH = "/config/AGENTS.md"


def build_backend(settings: Settings, store: BaseStore | None = None) -> CompositeBackend:
    """`store=None` resolves the graph's store at call time (e.g. under `langgraph dev`)."""
    for directory in (settings.finance_dir, settings.docs_dir, settings.skills_dir, settings.config_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return CompositeBackend(
        default=StateBackend(),
        routes={
            "/workspace/": StoreBackend(namespace=workspace_fs_ns, store=store),
            "/memories/": StoreBackend(namespace=procedural_fs_ns, store=store),
            "/finance/": FilesystemBackend(root_dir=settings.finance_dir, virtual_mode=True),
            "/docs/": FilesystemBackend(root_dir=settings.docs_dir, virtual_mode=True),
            "/skills/": FilesystemBackend(root_dir=settings.skills_dir, virtual_mode=True),
            "/config/": FilesystemBackend(root_dir=settings.config_dir, virtual_mode=True),
        },
    )


def default_permissions() -> list[FilesystemPermission]:
    """Source data and repo-managed instructions are read-only for every agent."""
    return [
        FilesystemPermission(operations=["write"], paths=[f"{p}**" for p in READ_ONLY_PREFIXES], mode="deny")
    ]
