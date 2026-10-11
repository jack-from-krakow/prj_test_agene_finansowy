"""Interactive terminal chat: `agent-chat` (or `python -m agent_konwersacyjny`)."""

from __future__ import annotations

import argparse
import asyncio
import logging

from dotenv import load_dotenv

from agent_konwersacyjny.app import AgentApp, create_app
from agent_konwersacyjny.config import get_settings

HELP = """Komendy:
  /new          nowa rozmowa (nowy wątek)
  /history      historia bieżącej rozmowy
  /thread       identyfikator bieżącego wątku
  /exit         wyjście"""


async def _repl(app: AgentApp, thread_id: str, user_id: str) -> None:
    print(f"Wątek: {thread_id} | użytkownik: {user_id}\n{HELP}\n")
    loop = asyncio.get_running_loop()
    while True:
        try:
            text = (await loop.run_in_executor(None, input, "Ty> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if text == "/exit":
            return
        if text == "/new":
            thread_id = app.new_thread_id()
            print(f"Nowy wątek: {thread_id}")
            continue
        if text == "/thread":
            print(thread_id)
            continue
        if text == "/history":
            for speaker, body in await app.history(thread_id):
                print(f"[{speaker}] {body}\n")
            continue
        try:
            result = await app.chat(text, thread_id=thread_id, user_id=user_id)
        except Exception as exc:  # noqa: BLE001 - keep the session alive
            logging.getLogger(__name__).exception("Turn failed")
            print(f"Błąd: {exc}")
            continue
        for reply in result.replies:
            print(f"\n[{reply.agent}] {reply.text}\n")
        if result.active_agent != "orchestrator":
            print(f"(rozmowę prowadzi teraz: {result.active_agent})")


async def _main(args: argparse.Namespace) -> None:
    async with create_app(get_settings(), load_mcp=not args.no_mcp) as app:
        await _repl(app, args.thread or app.new_thread_id(), args.user or app.settings.default_user_id)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Czat z agentem konwersacyjnym")
    parser.add_argument("--thread", help="kontynuuj istniejący wątek")
    parser.add_argument("--user", help="identyfikator użytkownika (zakres pamięci długoterminowej)")
    parser.add_argument("--no-mcp", action="store_true", help="nie ładuj serwerów MCP")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
