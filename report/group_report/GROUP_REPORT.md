# Group Report: Lab 3 - Production-Grade Agentic System

- **Team Name**: [TODO: your team name]
- **Team Members**: [TODO: Member 1, Member 2, ...]
- **Deployment Date**: 2026-06-01

> Domain: **TV shows / series** (real data via the public TVmaze API, no API key).
> Rename this file to `GROUP_REPORT_[TEAM_NAME].md` before submitting.

---

## 1. Executive Summary

We built a ReAct agent that answers questions about TV shows by calling real
tools against the TVmaze API, and compared it against a tool-less chatbot
baseline on the same 5-question test suite (auto-graded against live TVmaze
ground truth).

- **Success Rate**: Chatbot **40%** -> Agent v1 **60%** -> Agent v2 **80%** (4/5).
- **Key Outcome**: The agent matches the chatbot on simple single-fact questions
  but decisively wins on multi-step questions (combined episode counts, runtime
  math) because it grounds every number in a tool Observation instead of
  hallucinating. Prompt + guardrail hardening (v1 -> v2) raised accuracy by
  20 points and doubled the rate of "clean" (fully tool-grounded) runs.

---

## 2. System Architecture & Tooling

### 2.1 ReAct Loop Implementation

Implemented in `src/agent/agent.py`. Each step: build prompt (question +
scratchpad) -> `LLM.generate` (Thought + Action) -> clean response (strip code
fences, cut hallucinated Observations) -> detect `Final Answer` or parse
`Action: tool_name(args)` -> execute tool -> append `Observation` to the
scratchpad -> repeat until a final answer or `max_steps`.

See the full diagram in [`report/flowchart.md`](../flowchart.md).

```
Question -> [Thought + Action] -> Parse -> Execute Tool -> Observation
   ^------------------------------------------------------------|
   (loop until "Final Answer:" or max_steps; v2 guardrail blocks
    any Final Answer that is not grounded in a tool result)
```

### 2.2 Tool Definitions (Inventory)

All tools hit the live TVmaze API and return short, sentinel-guarded strings
(`NOT_FOUND`, `-1`, `INVALID_INPUT`, `API_ERROR`). Each lives in its own module
under `src/tools/`.

| Tool Name | Input Format | Returns | Use Case |
| :--- | :--- | :--- | :--- |
| `get_show_info` | `title='Breaking Bad'` | JSON (year, rating, runtime, genres, network, summary) | Rich metadata lookup |
| `get_show_rating` | `title='Game of Thrones'` | float 0-10 | Numeric rating for comparisons |
| `get_episode_stats` | `title='Breaking Bad'` | JSON (season_count, total_episodes, total_runtime_minutes/hours) | Binge-time / episode math |
| `get_cast` | `title='Breaking Bad'` | "Actor as Character; ..." | Cast lookup |

### 2.3 LLM Providers Used

- **Primary**: `deepseek-v4-flash` via an OpenAI-compatible endpoint
  (`OPENAI_BASE_URL`), wired through `OpenAIProvider`.
- **Secondary (Backup)**: Gemini (`gemini-1.5-flash`) and local GGUF (Phi-3)
  available through the same `LLMProvider` interface; switchable via
  `src/core/factory.py` and the `DEFAULT_PROVIDER` env var.

---

## 3. Telemetry & Performance Dashboard

Metrics captured automatically in `logs/` (events `LLM_METRIC`, `AGENT_STEP`,
`AGENT_START/END`) and aggregated by `analyze_logs.py` / `evaluate.py`.

Per-task aggregates over the 5-question suite (`report/evaluation_results.md`):

| System | Accuracy | Avg tokens/task | Avg LLM calls (loop len) | Avg latency/task | Total cost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Chatbot | 40% | 1647 | 1.0 | 12898 ms | $0.082 |
| Agent v1 | 60% | 1549 | 1.8 | 5695 ms | $0.077 |
| Agent v2 | 80% | 3222 | 2.6 | 9891 ms | $0.161 |

Per-LLM-call latency distribution (agent, from `analyze_logs.py`):

- **Latency P50**: ~2476 ms
- **Latency P99**: ~4524 ms

Observations:
- The chatbot has the **highest per-task latency** (~12.9 s) despite a single
  call, because it generates long verbose completions (~4.7k completion tokens
  across the baseline demo).
- The agent makes more (but shorter) calls; v2 trades extra tokens/cost for the
  highest accuracy.

---

## 4. Root Cause Analysis (RCA) - Failure Traces

Full extracted traces in `report/traces/` (`trace_success.md`, `trace_failure.md`).

### Case Study: Hallucinated rating for the second show
- **Input**: "Which has a higher TVmaze rating, Breaking Bad or Game of Thrones, and by how much?"
- **Observed (v1)**: 0 tool calls; answer "higher by 0.5" — both ratings invented.
- **Observed (v2)**: calls `get_show_rating('Breaking Bad') = 9.2` (correct) but
  then either emits a final answer without the `Final Answer:` prefix
  (`PARSE_ERROR`) or skips the tool for Game of Thrones and states "9.1" instead
  of the real **8.9**.
- **Root Cause**: For multi-entity questions the model tends to call a tool for
  the first entity, then fall back to parametric memory for the rest. The v1
  prompt did not force a per-entity tool call, and the model occasionally
  answers in free text without the required prefix.
- **Detection**: caught in logs by the `SUSPECTED_UNDER_TOOLING` heuristic
  (multi-entity question solved with < 2 distinct tool calls) and `PARSE_ERROR`.

---

## 5. Ablation Studies & Experiments

### Experiment 1: Prompt v1 vs Prompt v2 (+ guardrail)
- **Diff**:
  1. v2 prompt rule: "call a tool SEPARATELY for EACH show before comparing"
     + "never state a number from memory" + mandatory `Final Answer:` prefix +
     a multi-show few-shot example.
  2. Code guardrail: reject any `Final Answer` produced before at least one
     successful tool Observation (logged as `NO_TOOL_USED`).
- **Result** (3-question demo, `analyze_logs.py`):
  | Version | Clean (tool-grounded) runs | Under-tooling | Guardrail trips |
  | :--- | :--- | :--- | :--- |
  | v1 | 1/3 (33%) | 2 | 0 |
  | v2 | 2/3 (67%) | 1 | 1 |
  Full-suite accuracy rose from 60% (v1) to **80%** (v2).

### Experiment 2: Chatbot vs Agent (per-question)
| # | Type | Question | Chatbot | Agent v2 | Winner |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | simple | Year Breaking Bad premiered | correct | correct | Draw |
| 2 | simple | TVmaze rating of Breaking Bad | WRONG | correct | **Agent** |
| 3 | multi | Total binge hours of Breaking Bad | correct | correct | Draw |
| 4 | multi | Higher rating: BB vs GoT, by how much | WRONG | WRONG | Draw |
| 5 | multi | Combined episodes BB + Stranger Things | WRONG | correct | **Agent** |

Insight: the chatbot is competitive on **simple single-fact** questions but
fails whenever an answer requires combining or computing multiple real values;
the agent wins precisely there by grounding each number in a tool call.

---

## 6. Production Readiness Review

- **Security**: Tool arguments are parsed defensively (`parse_title`) and tools
  never execute arbitrary code; only whitelisted tool names are callable.
- **Guardrails**: `max_steps` caps the loop (no infinite billing); the v2
  guardrail blocks ungrounded answers; tools return sentinels instead of raising.
- **Reliability gap (known)**: multi-entity questions can still under-call tools
  (question 4). Next step: force tool calls >= number of detected entities, or
  use a stronger model / structured tool-calling API.
- **Scaling**: move to structured/function-calling tool APIs and a framework
  such as LangGraph for branching; add a vector store for tool retrieval once
  the tool count grows; cache TVmaze responses to cut latency and cost.

---

> [!NOTE]
> Submit this report by renaming it to `GROUP_REPORT_[TEAM_NAME].md` and placing it in this folder.
