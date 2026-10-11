import json
import sys
from pathlib import Path

from agent_konwersacyjny.mcp import load_mcp_tools

SERVER = str(Path(__file__).parent / "mcp_echo_server.py")


async def test_missing_config_means_no_tools(tmp_path):
    assert await load_mcp_tools(tmp_path / "nope.json") == {}


async def test_tools_are_grouped_per_agent_and_broken_servers_skipped(tmp_path):
    config = tmp_path / "mcp_servers.json"
    config.write_text(
        json.dumps(
            {
                "servers": {
                    "echo": {
                        "transport": "stdio",
                        "command": sys.executable,
                        "args": [SERVER],
                        "agents": ["finance"],
                    },
                    "broken": {"transport": "stdio", "command": "/nonexistent/binary", "args": []},
                }
            }
        ),
        encoding="utf-8",
    )
    tools = await load_mcp_tools(config)
    assert list(tools) == ["finance"]
    echo = tools["finance"][0]
    assert echo.name == "echo"
    result = await echo.ainvoke({"text": "cześć"})
    assert "cześć" in str(result)
