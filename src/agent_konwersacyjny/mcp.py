"""Load tools from MCP servers declared in `mcp_servers.json`.

Format (connection fields are passed to `langchain_mcp_adapters.MultiServerMCPClient`):

```json
{
  "servers": {
    "time": {
      "transport": "stdio",
      "command": "uvx",
      "args": ["mcp-server-time"],
      "agents": ["orchestrator"]
    }
  }
}
```

`agents` lists which agents get the server's tools (`orchestrator`, `rag`, `finance`);
omit it or use `["*"]` for all of them.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


def read_mcp_config(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    servers = json.loads(path.read_text(encoding="utf-8")).get("servers", {})
    if not isinstance(servers, dict):
        raise ValueError(f"{path}: 'servers' must be an object")
    return servers


async def load_mcp_tools(path: Path) -> dict[str, list[BaseTool]]:
    """Return tools grouped by agent name (`*` = every agent). Failing servers are skipped."""
    servers = read_mcp_config(path)
    if not servers:
        return {}
    from langchain_mcp_adapters.client import MultiServerMCPClient

    grouped: dict[str, list[BaseTool]] = defaultdict(list)
    for name, raw in servers.items():
        connection = {k: v for k, v in raw.items() if k != "agents"}
        agents = raw.get("agents") or ["*"]
        try:
            tools = await MultiServerMCPClient({name: connection}).get_tools(server_name=name)
        except Exception:  # noqa: BLE001 - one broken server must not take the app down
            logger.exception("MCP server %r failed to load; skipping it", name)
            continue
        logger.info("Loaded %d tools from MCP server %r for %s", len(tools), name, agents)
        for agent in agents:
            grouped[agent].extend(tools)
    return dict(grouped)
