"""Chatbot Baseline (Lab 3, Phase 2).

A deliberately minimal LLM chatbot: a single `generate()` call per user input,
NO tools and NO reasoning loop. Its purpose is to be the baseline that the ReAct
agent is compared against. On multi-step movie questions it is expected to
hallucinate facts (episode counts, runtimes, ratings) -- that failure is the
motivation for building the agent.

Usage:
    python chatbot.py                 # interactive REPL
    python chatbot.py --demo          # run preset multi-step movie questions
    python chatbot.py "your question" # one-shot
    python chatbot.py --provider google --demo
"""

import argparse
import sys

from dotenv import load_dotenv

from src.core.factory import get_provider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker

SYSTEM_PROMPT = (
    "You are a helpful movie and TV assistant. Answer the user's question directly "
    "and concisely. If a question requires multiple facts or calculations, do your best "
    "to answer in a single response."
)

# Multi-step questions that a tool-less chatbot tends to get wrong by hallucinating.
DEMO_QUESTIONS = [
    "How many total hours would it take to binge-watch all episodes of Breaking Bad?",
    "Which has a higher TVmaze rating, Breaking Bad or Game of Thrones, and by how much?",
    "What is the combined number of episodes of Breaking Bad and Stranger Things?",
]


class Chatbot:
    """Single-shot chatbot with no tools and no ReAct loop."""

    def __init__(self, provider):
        self.provider = provider

    def chat(self, user_input: str) -> str:
        logger.log_event(
            "CHATBOT_REQUEST",
            {"input": user_input, "model": self.provider.model_name},
        )
        result = self.provider.generate(user_input, system_prompt=SYSTEM_PROMPT)

        tracker.track_request(
            provider=result.get("provider", "unknown"),
            model=self.provider.model_name,
            usage=result.get("usage", {}),
            latency_ms=result.get("latency_ms", 0),
        )
        return result.get("content", "")


def run_demo(bot: Chatbot) -> None:
    print("=== Chatbot Baseline: Multi-step Demo ===\n")
    for i, question in enumerate(DEMO_QUESTIONS, 1):
        print(f"[{i}] User: {question}")
        answer = bot.chat(question)
        print(f"    Bot: {answer}\n")


def run_repl(bot: Chatbot) -> None:
    print("=== Chatbot Baseline (type 'exit' to quit) ===")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        print(f"Bot: {bot.chat(user_input)}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Chatbot baseline for Lab 3.")
    parser.add_argument("question", nargs="?", help="One-shot question to ask.")
    parser.add_argument("--provider", help="openai | google | local (defaults to env).")
    parser.add_argument("--demo", action="store_true", help="Run preset multi-step questions.")
    args = parser.parse_args()

    try:
        provider = get_provider(args.provider)
    except Exception as e:
        print(f"Failed to initialize provider: {e}")
        sys.exit(1)

    bot = Chatbot(provider)

    if args.demo:
        run_demo(bot)
    elif args.question:
        print(bot.chat(args.question))
    else:
        run_repl(bot)


if __name__ == "__main__":
    main()
