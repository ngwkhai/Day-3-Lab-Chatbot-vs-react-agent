"""Evaluation harness (Lab 3, Phase 5/6).

Runs the SAME test suite through three systems -- the chatbot baseline, ReAct
agent v1 and ReAct agent v2 -- and produces a data-driven comparison:
  - Accuracy (auto-graded against real TVmaze ground truth)
  - Avg tokens / request
  - Avg LLM calls (loop length)
  - Avg wall-clock latency
  - Total estimated cost
Plus a per-question Chatbot-vs-Agent winner table.

Results are printed and written to report/evaluation_results.md.

Ground-truth note: expected values were captured from the live TVmaze API
(Breaking Bad: rating 9.2 / 62 eps / 62h / 2008; Game of Thrones: 8.9;
Stranger Things: 42 eps). Update them if TVmaze data changes.

Usage:
    python evaluate.py
    python evaluate.py --provider google
"""

import argparse
import re
import time
from typing import Callable, Dict, List

from dotenv import load_dotenv

from chatbot import Chatbot
from src.agent.agent import ReActAgent
from src.core.factory import get_provider
from src.telemetry.metrics import tracker
from src.tools import TOOLS

RESULTS_PATH = "report/evaluation_results.md"

# Each case: question, list of substrings that must ALL appear in a correct
# answer, and difficulty (simple = single fact, multi = needs multiple tools).
TEST_SUITE = [
    {"q": "What year did the show Breaking Bad premiere?",
     "expected": ["2008"], "type": "simple"},
    {"q": "What is the TVmaze rating of Breaking Bad?",
     "expected": ["9.2"], "type": "simple"},
    {"q": "How many total hours would it take to binge-watch all episodes of Breaking Bad?",
     "expected": ["62"], "type": "multi"},
    {"q": "Which has a higher TVmaze rating, Breaking Bad or Game of Thrones, and by how much?",
     "expected": ["9.2", "8.9", "0.3"], "type": "multi"},
    {"q": "What is the combined number of episodes of Breaking Bad and Stranger Things?",
     "expected": ["104"], "type": "multi"},
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower())


def is_correct(answer: str, expected: List[str]) -> bool:
    a = _norm(answer)
    return all(_norm(e) in a for e in expected)


def run_system(runner: Callable[[str], str]) -> List[Dict]:
    rows = []
    for case in TEST_SUITE:
        start = len(tracker.session_metrics)
        t0 = time.time()
        try:
            answer = runner(case["q"])
        except Exception as e:
            answer = f"ERROR: {e}"
        wall_ms = int((time.time() - t0) * 1000)

        calls = tracker.session_metrics[start:]
        rows.append({
            "q": case["q"],
            "type": case["type"],
            "answer": answer,
            "correct": is_correct(answer, case["expected"]),
            "tokens": sum(m["total_tokens"] for m in calls),
            "llm_calls": len(calls),
            "wall_ms": wall_ms,
            "cost": sum(m["cost_estimate"] for m in calls),
        })
    return rows


def aggregate(rows: List[Dict]) -> Dict:
    n = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    return {
        "accuracy": round(100 * correct / n) if n else 0,
        "correct": correct,
        "total": n,
        "avg_tokens": round(sum(r["tokens"] for r in rows) / n) if n else 0,
        "avg_llm_calls": round(sum(r["llm_calls"] for r in rows) / n, 1) if n else 0,
        "avg_wall_ms": round(sum(r["wall_ms"] for r in rows) / n) if n else 0,
        "total_cost": round(sum(r["cost"] for r in rows), 5),
    }


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluation harness for Lab 3.")
    parser.add_argument("--provider", help="openai | google | local (defaults to env).")
    parser.add_argument("--max-steps", type=int, default=6)
    args = parser.parse_args()

    provider = get_provider(args.provider)

    systems = {
        "Chatbot": Chatbot(provider).chat,
        "Agent v1": ReActAgent(provider, TOOLS, args.max_steps, prompt_version="v1").run,
        "Agent v2": ReActAgent(provider, TOOLS, args.max_steps, prompt_version="v2").run,
    }

    print(f"Running {len(TEST_SUITE)} questions x {len(systems)} systems "
          f"on model '{provider.model_name}'...\n")

    results = {name: run_system(runner) for name, runner in systems.items()}
    aggs = {name: aggregate(rows) for name, rows in results.items()}

    # ---- Console summary
    header = f"{'System':<10} | {'Acc':>6} | {'AvgTok':>7} | {'AvgCalls':>8} | {'AvgLat':>8} | {'Cost':>8}"
    print(header)
    print("-" * len(header))
    for name, a in aggs.items():
        print(f"{name:<10} | {a['accuracy']:>5}% | {a['avg_tokens']:>7} | "
              f"{a['avg_llm_calls']:>8} | {a['avg_wall_ms']:>6}ms | ${a['total_cost']:>7}")

    # ---- Per-question Chatbot vs Agent v2 winner
    print("\nPer-question: Chatbot vs Agent v2")
    for i, case in enumerate(TEST_SUITE):
        cb = results["Chatbot"][i]["correct"]
        ag = results["Agent v2"][i]["correct"]
        winner = "Draw" if cb == ag else ("Agent" if ag else "Chatbot")
        mark = lambda b: "OK" if b else "X"
        print(f"  [{case['type']:6}] cb={mark(cb)} agent={mark(ag)} -> {winner:7} | {case['q']}")

    write_markdown(results, aggs)
    print(f"\nWrote {RESULTS_PATH}")


def write_markdown(results: Dict[str, List[Dict]], aggs: Dict[str, Dict]) -> None:
    lines = ["# Evaluation Results: Chatbot vs ReAct Agent", ""]
    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| System | Accuracy | Avg tokens | Avg LLM calls | Avg latency | Total cost |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for name, a in aggs.items():
        lines.append(f"| {name} | {a['accuracy']}% ({a['correct']}/{a['total']}) | "
                     f"{a['avg_tokens']} | {a['avg_llm_calls']} | {a['avg_wall_ms']} ms | "
                     f"${a['total_cost']} |")

    lines += ["", "## Per-question results", ""]
    lines.append("| # | Type | Question | Chatbot | Agent v1 | Agent v2 | Winner (cb vs v2) |")
    lines.append("| :-- | :-- | :-- | :-- | :-- | :-- | :-- |")
    for i, case in enumerate(TEST_SUITE):
        cb = results["Chatbot"][i]["correct"]
        v1 = results["Agent v1"][i]["correct"]
        v2 = results["Agent v2"][i]["correct"]
        winner = "Draw" if cb == v2 else ("Agent" if v2 else "Chatbot")
        ok = lambda b: "correct" if b else "WRONG"
        q = case["q"].replace("|", "/")
        lines.append(f"| {i + 1} | {case['type']} | {q} | {ok(cb)} | {ok(v1)} | {ok(v2)} | {winner} |")

    lines += ["", "## Sample answers (Agent v2)", ""]
    for i, case in enumerate(TEST_SUITE):
        ans = results["Agent v2"][i]["answer"].replace("\n", " ")
        lines.append(f"- **{case['q']}**\n  - {ans}")

    lines.append("")
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
