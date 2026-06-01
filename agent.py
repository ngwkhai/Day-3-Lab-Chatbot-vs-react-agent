"""ReAct Agent runner (Lab 3, Phase 3).

Wires the TVmaze tools into the ReAct loop and runs it against the same
questions used by the chatbot baseline, so the two can be compared directly.

Usage:
    python agent.py                  # interactive REPL
    python agent.py --demo           # run preset multi-step movie questions
    python agent.py "your question"  # one-shot
    python agent.py --provider google --max-steps 8 --demo
"""

import argparse
import sys

from dotenv import load_dotenv

from chatbot import DEMO_QUESTIONS
from src.agent.agent import ReActAgent
from src.core.factory import get_provider
from src.tools import TOOLS


def run_demo(agent: ReActAgent) -> None:
    print("=== ReAct Agent: Multi-step Demo ===\n")
    for i, question in enumerate(DEMO_QUESTIONS, 1):
        print(f"[{i}] Question: {question}")
        answer = agent.run(question)
        print(f"    Final Answer: {answer}\n")


def run_repl(agent: ReActAgent) -> None:
    print("=== ReAct Agent (type 'exit' to quit) ===")
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
        print(f"Answer: {agent.run(user_input)}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="ReAct agent runner for Lab 3.")
    parser.add_argument("question", nargs="?", help="One-shot question to ask.")
    parser.add_argument("--provider", help="openai | google | local (defaults to env).")
    parser.add_argument("--max-steps", type=int, default=6, help="Max ReAct loop steps.")
    parser.add_argument(
        "--prompt-version", choices=["v1", "v2"], default="v2",
        help="Which system prompt to use (v1 = baseline, v2 = hardened).",
    )
    parser.add_argument("--demo", action="store_true", help="Run preset multi-step questions.")
    args = parser.parse_args()

    try:
        provider = get_provider(args.provider)
    except Exception as e:
        print(f"Failed to initialize provider: {e}")
        sys.exit(1)

    agent = ReActAgent(
        llm=provider, tools=TOOLS, max_steps=args.max_steps,
        prompt_version=args.prompt_version,
    )
    print(f"(prompt_version={args.prompt_version}, max_steps={args.max_steps})")

    if args.demo:
        run_demo(agent)
    elif args.question:
        print(agent.run(args.question))
    else:
        run_repl(agent)


if __name__ == "__main__":
    main()
