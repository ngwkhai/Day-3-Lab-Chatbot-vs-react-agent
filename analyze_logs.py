"""Log analyzer for Lab 3 (Phase 4 - Failure Analysis).

Parses the structured JSON logs in logs/ and produces:
  1. An aggregate telemetry dashboard (tokens, latency P50/P99, cost) for the
     chatbot baseline vs the ReAct agent -- the numbers needed by EVALUATION.md
     and the GROUP_REPORT telemetry section.
  2. A per-run breakdown of every agent run with detected error codes.
  3. Auto-extracted Markdown traces (one success + one failure) written to
     report/traces/, ready to paste into the report (Trace Quality, 9 pts).

Usage:
    python analyze_logs.py                 # latest log file
    python analyze_logs.py logs/2026-06-01.log
"""

import glob
import json
import os
import re
import statistics
import sys
from typing import Dict, List, Optional

TRACE_DIR = os.path.join("report", "traces")
# Words that signal a question references more than one entity / needs >=2 tool calls.
MULTI_ENTITY_HINT = re.compile(r"\b(and|combined|vs|versus|or|both|compare|each)\b", re.IGNORECASE)


def load_events(path: str) -> List[dict]:
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarize_metrics(metrics: List[dict]) -> Dict[str, float]:
    if not metrics:
        return {}
    latencies = [m["latency_ms"] for m in metrics]
    total_tokens = [m["total_tokens"] for m in metrics]
    return {
        "requests": len(metrics),
        "total_tokens": sum(total_tokens),
        "avg_tokens": round(statistics.mean(total_tokens), 1),
        "prompt_tokens": sum(m["prompt_tokens"] for m in metrics),
        "completion_tokens": sum(m["completion_tokens"] for m in metrics),
        "latency_p50_ms": round(percentile(latencies, 50)),
        "latency_p99_ms": round(percentile(latencies, 99)),
        "latency_avg_ms": round(statistics.mean(latencies)),
        "latency_max_ms": max(latencies),
        "total_cost": round(sum(m["cost_estimate"] for m in metrics), 5),
    }


def parse_runs(events: List[dict]):
    """Group events into agent runs and a chatbot metric bucket."""
    runs = []
    chatbot_metrics = []
    current = None

    for ev in events:
        etype, data = ev["event"], ev["data"]
        if etype == "AGENT_START":
            current = {"input": data["input"], "model": data["model"],
                       "version": data.get("version", "v1"),
                       "steps": [], "metrics": [], "status": None}
            runs.append(current)
        elif etype == "AGENT_END":
            if current:
                current["status"] = data.get("status")
                current["reported_steps"] = data.get("steps")
            current = None
        elif etype == "AGENT_STEP":
            if current:
                current["steps"].append(data)
        elif etype == "LLM_METRIC":
            (current["metrics"] if current else chatbot_metrics).append(data)
        # CHATBOT_REQUEST just precedes a chatbot LLM_METRIC; nothing to store.

    return runs, chatbot_metrics


def classify_run(run: dict) -> dict:
    """Attach error flags and a verdict to an agent run."""
    actions = [s["action"] for s in run["steps"] if s.get("action")]
    distinct_tools = {a.split("(")[0] for a in actions}
    distinct_calls = set(actions)  # full tool_name(args), so same tool on 2 shows counts twice
    errors = [s["error"] for s in run["steps"] if s.get("error")]
    final = next((s["final_answer"] for s in run["steps"] if s.get("final_answer")), None)

    # Quality flags decide the verdict: they signal the FINAL answer is likely wrong.
    quality_flags = []
    if MULTI_ENTITY_HINT.search(run["input"]) and len(distinct_calls) < 2:
        quality_flags.append("SUSPECTED_UNDER_TOOLING")
    if run.get("status") == "timeout":
        quality_flags.append("TIMEOUT")

    # Process flags are mid-run hiccups the agent may have recovered from (info only).
    process_flags = errors

    run["actions"] = actions
    run["distinct_tools"] = sorted(distinct_tools)
    run["distinct_calls"] = sorted(distinct_calls)
    run["flags"] = quality_flags + process_flags
    run["quality_flags"] = quality_flags
    run["process_flags"] = process_flags
    run["final_answer"] = final
    run["verdict"] = "CLEAN" if not quality_flags else "SUSPECT"
    return run


def render_trace_md(run: dict, kind: str) -> str:
    lines = [
        f"# {kind} Trace",
        "",
        f"- **Question**: {run['input']}",
        f"- **Model**: {run['model']}",
        f"- **Reported status**: {run.get('status')}  (steps={run.get('reported_steps')})",
        f"- **Distinct tools used**: {', '.join(run['distinct_tools']) or 'none'}",
        f"- **Detected flags**: {', '.join(run['flags']) or 'none'}",
        "",
        "## Step-by-step trace",
        "",
    ]
    for s in run["steps"]:
        lines.append(f"### Step {s.get('step')}")
        if s.get("thought"):
            lines.append(f"- **Thought**: {s['thought']}")
        if s.get("action"):
            lines.append(f"- **Action**: `{s['action']}`")
        if s.get("observation"):
            lines.append(f"- **Observation**: `{s['observation']}`")
        if s.get("error"):
            lines.append(f"- **ERROR**: `{s['error']}`")
        if s.get("raw"):
            lines.append(f"- **Raw model output**: {s['raw']}")
        if s.get("final_answer"):
            lines.append(f"- **Final Answer**: {s['final_answer']}")
        lines.append("")
    lines.append("## Analysis (fill in)")
    lines.append("- **Root cause**: _why did the model behave this way?_")
    lines.append("- **Fix (v1 -> v2)**: _prompt / tool change to try._")
    lines.append("")
    return "\n".join(lines)


def print_dashboard(title: str, summary: Dict[str, float]) -> None:
    print(f"\n## {title}")
    if not summary:
        print("  (no requests)")
        return
    print(f"  Requests           : {summary['requests']}")
    print(f"  Total tokens       : {summary['total_tokens']} "
          f"(prompt={summary['prompt_tokens']}, completion={summary['completion_tokens']})")
    print(f"  Avg tokens/request : {summary['avg_tokens']}")
    print(f"  Latency P50 / P99  : {summary['latency_p50_ms']} ms / {summary['latency_p99_ms']} ms")
    print(f"  Latency avg / max  : {summary['latency_avg_ms']} ms / {summary['latency_max_ms']} ms")
    print(f"  Total cost (est.)  : ${summary['total_cost']}")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        candidates = sorted(glob.glob("logs/*.log"))
        if not candidates:
            print("No log files found in logs/.")
            sys.exit(1)
        path = candidates[-1]

    print(f"Analyzing: {path}")
    events = load_events(path)
    runs, chatbot_metrics = parse_runs(events)
    runs = [classify_run(r) for r in runs]

    # ---- Aggregate dashboards
    print("\n" + "=" * 60)
    print("TELEMETRY DASHBOARD")
    print("=" * 60)
    print_dashboard("Chatbot baseline", summarize_metrics(chatbot_metrics))
    agent_metrics = [m for r in runs for m in r["metrics"]]
    print_dashboard("ReAct agent", summarize_metrics(agent_metrics))

    # ---- Reliability
    print("\n" + "=" * 60)
    print("AGENT RUN BREAKDOWN")
    print("=" * 60)
    clean = sum(1 for r in runs if r["verdict"] == "CLEAN")
    print(f"Total runs: {len(runs)} | clean: {clean} | suspect: {len(runs) - clean}\n")
    for i, r in enumerate(runs, 1):
        print(f"[{i}] {r['version']:3} | {r['verdict']:7} | tool_calls={len(r['distinct_calls'])} "
              f"| flags={r['flags'] or '-'} | steps={r.get('reported_steps')}")
        print(f"    Q: {r['input']}")
        print(f"    A: {r['final_answer']}\n")

    # ---- Reliability comparison per prompt version (v1 vs v2)
    print("-" * 60)
    print("RELIABILITY BY PROMPT VERSION")
    print("-" * 60)
    versions = sorted({r["version"] for r in runs})
    for v in versions:
        group = [r for r in runs if r["version"] == v]
        n = len(group)
        clean_n = sum(1 for r in group if r["verdict"] == "CLEAN")
        under = sum(1 for r in group if "SUSPECTED_UNDER_TOOLING" in r["quality_flags"])
        parse_err = sum(1 for r in group if "PARSE_ERROR" in r["process_flags"])
        guard = sum(1 for r in group if "NO_TOOL_USED" in r["process_flags"])
        rate = round(100 * clean_n / n) if n else 0
        print(f"  {v}: runs={n} | clean={clean_n} ({rate}%) | under_tooling={under} "
              f"| parse_errors={parse_err} | guardrail_trips={guard}")

    # ---- Export one success + one failure trace
    os.makedirs(TRACE_DIR, exist_ok=True)
    success = next((r for r in runs if r["verdict"] == "CLEAN" and r["distinct_tools"]), None)
    failure = next((r for r in runs if r["verdict"] == "SUSPECT"), None)

    if success:
        with open(os.path.join(TRACE_DIR, "trace_success.md"), "w", encoding="utf-8") as f:
            f.write(render_trace_md(success, "Successful"))
        print(f"Wrote {TRACE_DIR}/trace_success.md")
    if failure:
        with open(os.path.join(TRACE_DIR, "trace_failure.md"), "w", encoding="utf-8") as f:
            f.write(render_trace_md(failure, "Failed"))
        print(f"Wrote {TRACE_DIR}/trace_failure.md")

    if not failure:
        print("No failing/suspect run detected in this log.")


if __name__ == "__main__":
    main()
