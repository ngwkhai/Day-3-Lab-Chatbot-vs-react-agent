# ReAct Agent — Logic Flowchart

This diagram shows the Thought → Action → Observation loop implemented in
`src/agent/agent.py`, including the v2 guardrail and error paths captured by
telemetry.

```mermaid
flowchart TD
    A([User question]) --> B[Build prompt:\nQuestion + scratchpad]
    B --> C[LLM.generate\nThought + Action]
    C --> D[Clean response:\nstrip code fences,\ncut hallucinated Observation]
    D --> E{Contains\nFinal Answer?}

    E -- Yes --> G{v2 guardrail:\nany tool used yet?}
    G -- No tool used --> H[Append ERROR NO_TOOL_USED\nforce a tool call]
    H --> L
    G -- Tool was used --> Z([Return Final Answer\nlog AGENT_END success])

    E -- No --> F{Parse Action\ntool_name args ?}
    F -- Parse fails --> P[Observation =\nERROR PARSE_ERROR]
    F -- Unknown tool --> U[Observation =\nERROR UNKNOWN_TOOL]
    F -- Valid tool --> T[Execute tool\nrun TVmaze API]
    T --> O[Observation = tool result\nNOT_FOUND / value]

    P --> L[Append step to scratchpad\nlog AGENT_STEP + metrics]
    U --> L
    O --> L
    L --> M{step < max_steps?}
    M -- Yes --> B
    M -- No --> X([Return timeout\nlog AGENT_END TIMEOUT])
```

## Reading the trace

Every node above emits a structured JSON log event:

- `AGENT_START` / `AGENT_END` — run boundaries (with `version`, `status`).
- `AGENT_STEP` — one Thought/Action/Observation, or an error code
  (`PARSE_ERROR`, `UNKNOWN_TOOL`, `NO_TOOL_USED`).
- `LLM_METRIC` — tokens, latency and cost for each `LLM.generate` call.

`analyze_logs.py` reconstructs runs from these events to compute reliability
per prompt version.
