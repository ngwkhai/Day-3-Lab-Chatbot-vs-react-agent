# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Nguyen Dinh Khai
- **Student ID**: 2A202600671
- **Date**: 2026-06-01

---

## I. Technical Contribution (15 Points)

- **Modules Implemented**:
  - `src/tools/` — TVmaze-backed tools (real data, no API key), one file per tool:
    `show_info_tool.py`, `show_rating_tool.py`, `episode_stats_tool.py`,
    `cast_tool.py`, plus shared `_client.py` (HTTP, title parsing, HTML cleaning)
    and the `TOOLS` registry in `__init__.py`.
  - `src/agent/agent.py` — the ReAct loop, the v1 and v2 system prompts, the
    Thought/Action/Observation parsing, and the v2 "no ungrounded answer" guardrail.
  - `chatbot.py` (baseline), `agent.py` (runner), `evaluate.py` (auto-graded
    test suite) and `analyze_logs.py` (telemetry/failure analysis).
  - `src/core/factory.py` (provider switching) and `OpenAIProvider` `base_url`
    support, so the lab runs on an OpenAI-compatible endpoint.

- **Code Highlights**:
  - Tools return sentinel values (`NOT_FOUND`, `-1`, `INVALID_INPUT`,
    `API_ERROR`) instead of raising, so a tool failure never breaks the ReAct
    loop — the agent can reason about it.
  - Tool descriptions are written precisely (argument format, units, error
    values) since the LLM only "sees" a tool through its description.
  - The v2 guardrail in the loop refuses any final answer that is not grounded
    in a tool result, which is the key fix for hallucinated numbers:

```141:147:src/agent/agent.py
                # Guardrail (v2): never accept an answer that is not grounded in a tool result.
                if guard_active and successful_tools == 0 and step < self.max_steps:
                    observation = (
                        "ERROR[NO_TOOL_USED]: You answered without calling any tool. "
                        "You MUST call a tool and use its Observation before giving a Final Answer. "
                        "Call the appropriate tool now."
                    )
```

- **Documentation**: tools register as `{name, description, func}` in
  `src/tools/__init__.py` (`TOOLS`), which the agent consumes directly; the loop
  feeds each tool `Observation` back into the prompt for the next step.

---

## II. Debugging Case Study (10 Points)

*A real failure found via the logging system (`logs/` + `analyze_logs.py`).*

- **Problem Description**: On the multi-show question "Which has a higher TVmaze
  rating, Breaking Bad or Game of Thrones, and by how much?", the agent called
  `get_show_rating('Breaking Bad') = 9.2` (correct) but then **did not call the
  tool for Game of Thrones** and stated "8.5"/"9.1" instead of the real **8.9** —
  a hallucinated argument/value.
- **Log Source**: `report/traces/trace_failure.md` (extracted from
  `logs/2026-06-01.log`); flagged by `analyze_logs.py` as
  `SUSPECTED_UNDER_TOOLING` and sometimes `PARSE_ERROR`.
- **Diagnosis**: Not a tool bug. The v1 system prompt did not force a per-entity
  tool call, so the model answered the second entity from parametric memory.
  Occasionally it also wrote the answer without the `Final Answer:` prefix,
  tripping the parser.
- **Solution (v1 -> v2)**:
  1. Prompt rule: "call a tool SEPARATELY for EACH show before comparing" +
     "never state a number from memory" + mandatory `Final Answer:` prefix +
     a multi-show few-shot example.
  2. Code guardrail: reject any `Final Answer` emitted before a successful tool
     Observation (logged as `NO_TOOL_USED`).
- **Measured effect**: clean (tool-grounded) runs 33% -> 67%; full-suite
  accuracy 60% -> 80%. (Residual: question 4 still under-calls occasionally —
  honest limitation, see Section IV.)

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

1.  **Reasoning**: The `Thought` block was the biggest difference. The plain
    chatbot jumps straight to an answer, so when a question needs a real number
    it simply guesses from memory (it gave 49.6 hours for Breaking Bad instead
    of the real 62). The agent's `Thought` forces it to first decide *which tool
    to call* and *with what argument* before committing to any number. This
    "think before answering" step is what turns a guess into a grounded answer,
    and it is also what makes the agent's behaviour auditable in the logs.

2.  **Reliability**: The agent was not always better. On simple single-fact
    questions (for example, the year a show premiered) both systems were correct,
    so it was a Draw — the extra reasoning brings no benefit there. The agent is
    also more expensive: it makes more LLM calls per task and uses more tokens
    (about 3222 vs 1647 for v2), so for trivial questions the chatbot is cheaper
    and faster. The agent also still failed the BB-vs-GoT rating question when
    the model skipped the second tool call, which shows the loop is only as
    reliable as the model's willingness to follow the tool-use instructions.

3.  **Observation**: The tool Observations directly steered the next step. Once
    `get_episode_stats` returned `total_runtime_hours = 62.0` or
    `get_show_rating` returned `9.2`, the agent stopped guessing and reused those
    exact values in its arithmetic and final answer. When an Observation was an
    error such as `NOT_FOUND`, the agent could see the failure and retry instead
    of inventing data. In short, the environment feedback replaced the model's
    internal (often wrong) memory with verified facts.

---

## IV. Future Improvements (5 Points)

- **Scalability**: Replace the regex-based `Action: tool(args)` parsing with a
  structured / function-calling tool API so arguments are always valid JSON,
  which removes most `PARSE_ERROR` cases. Run independent tool calls
  asynchronously and cache TVmaze responses to cut latency and cost on repeated
  queries.
- **Safety**: Add a "supervisor" check that requires the agent to make at least
  as many tool calls as there are entities detected in the question (this would
  fix the remaining BB-vs-GoT gap), and have a second LLM audit the final answer
  against the collected Observations before returning it.
- **Performance**: Use a stronger base model than `deepseek-v4-flash`, which
  often skipped tool calls and hallucinated. As the number of tools grows, store
  tool descriptions in a vector database and retrieve only the most relevant
  tools per question instead of listing all of them in every prompt.

---