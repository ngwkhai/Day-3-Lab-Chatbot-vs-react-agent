import re
from typing import Any, Dict, List, Optional, Tuple

from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker

MAX_OBSERVATION_LEN = 600


class ReActAgent:
    """A ReAct-style agent following the Thought -> Action -> Observation loop.

    The agent feeds tool observations back into the prompt until the LLM emits a
    "Final Answer:". Every step is logged via telemetry so failures (parser errors,
    unknown tools, timeouts) can be analyzed from the structured logs.
    """

    def __init__(
        self,
        llm: LLMProvider,
        tools: List[Dict[str, Any]],
        max_steps: int = 5,
        prompt_version: str = "v2",
    ):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.prompt_version = prompt_version
        self.tools_by_name = {t["name"]: t for t in tools}

    # ------------------------------------------------------------------ prompt
    def _tool_block(self) -> Tuple[str, str]:
        tool_descriptions = "\n".join(
            f"- {t['name']}: {t['description']}" for t in self.tools
        )
        tool_names = ", ".join(self.tools_by_name.keys())
        return tool_descriptions, tool_names

    def get_system_prompt(self) -> str:
        if self.prompt_version == "v1":
            return self._system_prompt_v1()
        return self._system_prompt_v2()

    def _system_prompt_v1(self) -> str:
        tool_descriptions, tool_names = self._tool_block()
        return (
            "You are a reasoning agent that answers questions about TV shows by using tools.\n\n"
            "You have access to the following tools:\n"
            f"{tool_descriptions}\n\n"
            "Solve the task step by step using this EXACT format:\n\n"
            "Thought: your reasoning about what to do next.\n"
            "Action: tool_name(argument)\n\n"
            "After each Action, STOP. The system runs the tool and replies with:\n"
            "Observation: <tool result>\n\n"
            "Use the Observation to decide the next Thought/Action. Repeat as needed.\n"
            "When you have enough information, reply with:\n\n"
            "Thought: I now know the final answer.\n"
            "Final Answer: <concise answer>\n\n"
            "Rules:\n"
            f"- Only use these tools: {tool_names}. Never invent a tool name.\n"
            "- Emit exactly ONE Action per step, formatted as tool_name(argument).\n"
            "- Do NOT write the Observation yourself; the system provides it.\n"
            "- Do NOT wrap your output in markdown or code fences.\n"
            "- Base every factual number on tool Observations, not your own memory.\n"
            "- If a tool returns NOT_FOUND or an error, reconsider your arguments or try another tool."
        )

    def _system_prompt_v2(self) -> str:
        """v2: hardened against the two failures observed in v1 logs --
        (1) PARSE_ERROR from answers missing the 'Final Answer:' prefix, and
        (2) hallucinated numbers when the model skipped per-entity tool calls.
        """
        tool_descriptions, tool_names = self._tool_block()
        return (
            "You are a reasoning agent that answers questions about TV shows by using tools.\n\n"
            "You have access to the following tools:\n"
            f"{tool_descriptions}\n\n"
            "Solve the task step by step using this EXACT format:\n\n"
            "Thought: your reasoning about what to do next.\n"
            "Action: tool_name(argument)\n\n"
            "After each Action, STOP. The system runs the tool and replies with:\n"
            "Observation: <tool result>\n\n"
            "Use the Observation to decide the next Thought/Action. Repeat as needed.\n"
            "When you have enough information, reply with EXACTLY:\n\n"
            "Thought: I now know the final answer.\n"
            "Final Answer: <concise answer>\n\n"
            "CRITICAL RULES:\n"
            f"- Only use these tools: {tool_names}. Never invent a tool name.\n"
            "- Emit exactly ONE Action per step, formatted as tool_name(argument).\n"
            "- Do NOT write the Observation yourself; the system provides it.\n"
            "- Do NOT wrap your output in markdown or code fences.\n"
            "- NEVER state a rating, episode count, runtime, or year from your own memory. "
            "If you do not yet have an Observation for a fact, you MUST call a tool for it.\n"
            "- For ANY question that mentions more than one show, call a tool SEPARATELY for "
            "EACH show before you compare or combine. Do not produce a Final Answer until you "
            "have an Observation for every show mentioned.\n"
            "- Your final response MUST begin with the literal prefix 'Final Answer:'. "
            "Do not state the answer in any other way.\n"
            "- If a tool returns NOT_FOUND or an error, reconsider your arguments or try another tool.\n\n"
            "Worked example (two shows):\n"
            "Question: Which is rated higher, Show A or Show B, and by how much?\n"
            "Thought: I need each show's rating from the tool. Start with Show A.\n"
            "Action: get_show_rating(title='Show A')\n"
            "Observation: 8.0\n"
            "Thought: Now I need Show B's rating.\n"
            "Action: get_show_rating(title='Show B')\n"
            "Observation: 7.5\n"
            "Thought: I now know the final answer.\n"
            "Final Answer: Show A is rated higher (8.0 vs 7.5), by 0.5 points."
        )

    # -------------------------------------------------------------------- loop
    def run(self, user_input: str) -> str:
        logger.log_event(
            "AGENT_START",
            {"input": user_input, "model": self.llm.model_name, "version": self.prompt_version},
        )

        scratchpad = ""
        successful_tools = 0
        guard_active = self.prompt_version == "v2"
        for step in range(1, self.max_steps + 1):
            prompt = f"Question: {user_input}\n\n{scratchpad}".strip()

            result = self.llm.generate(prompt, system_prompt=self.get_system_prompt())
            tracker.track_request(
                provider=result.get("provider", "unknown"),
                model=self.llm.model_name,
                usage=result.get("usage", {}),
                latency_ms=result.get("latency_ms", 0),
            )

            text = self._clean_response(result.get("content", ""))
            thought = self._parse_thought(text)

            # 1) Final answer?
            final = re.search(r"Final\s*Answer\s*:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
            if final:
                answer = self._clean_final_answer(final.group(1))
                # Guardrail (v2): never accept an answer that is not grounded in a tool result.
                if guard_active and successful_tools == 0 and step < self.max_steps:
                    observation = (
                        "ERROR[NO_TOOL_USED]: You answered without calling any tool. "
                        "You MUST call a tool and use its Observation before giving a Final Answer. "
                        "Call the appropriate tool now."
                    )
                    logger.log_event(
                        "AGENT_STEP",
                        {"step": step, "thought": thought, "error": "NO_TOOL_USED",
                         "rejected_answer": answer},
                    )
                    scratchpad += f"{text.strip()}\nObservation: {observation}\n"
                    continue

                logger.log_event(
                    "AGENT_STEP",
                    {"step": step, "thought": thought, "final_answer": answer},
                )
                logger.log_event(
                    "AGENT_END",
                    {"steps": step, "status": "success", "version": self.prompt_version},
                )
                return answer

            # 2) Parse and execute an Action
            parsed = self._parse_action(text)
            if not parsed:
                observation = (
                    "ERROR[PARSE_ERROR]: No valid Action found. Respond with "
                    "'Action: tool_name(argument)' or 'Final Answer: ...'."
                )
                logger.log_event(
                    "AGENT_STEP",
                    {"step": step, "thought": thought, "error": "PARSE_ERROR", "raw": text[:400]},
                )
            else:
                tool_name, args = parsed
                observation = self._execute_tool(tool_name, args)
                if not observation.startswith("ERROR[") and observation not in ("NOT_FOUND", ""):
                    successful_tools += 1
                error_code = "UNKNOWN_TOOL" if observation.startswith("ERROR[UNKNOWN_TOOL]") else None
                logger.log_event(
                    "AGENT_STEP",
                    {
                        "step": step,
                        "thought": thought,
                        "action": f"{tool_name}({args})",
                        "observation": observation,
                        **({"error": error_code} if error_code else {}),
                    },
                )

            observation = observation[:MAX_OBSERVATION_LEN]
            scratchpad += f"{text.strip()}\nObservation: {observation}\n"

        logger.log_event(
            "AGENT_END",
            {"steps": self.max_steps, "status": "timeout", "error": "TIMEOUT",
             "version": self.prompt_version},
        )
        return "I could not complete the task within the allowed number of steps."

    # ---------------------------------------------------------------- parsing
    @staticmethod
    def _clean_response(raw: str) -> str:
        """Strip code fences and cut off any hallucinated Observation block."""
        if not raw:
            return ""
        text = raw.strip()
        # Remove markdown code fences (```json ... ``` etc.)
        text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "")
        # The model must not produce its own Observation; ignore anything from there on.
        cut = re.search(r"\n\s*Observation\s*:", text, re.IGNORECASE)
        if cut:
            text = text[: cut.start()]
        return text.strip()

    @staticmethod
    def _clean_final_answer(answer: str) -> str:
        """Trim a final answer: drop any repeated 'Final Answer:' the model echoes."""
        answer = answer.strip()
        # Some models duplicate the sentence and repeat the prefix; keep the first part.
        dup = re.search(r"\bFinal\s*Answer\s*:", answer, re.IGNORECASE)
        if dup:
            answer = answer[: dup.start()].strip()
        return answer

    @staticmethod
    def _parse_thought(text: str) -> Optional[str]:
        match = re.search(
            r"Thought\s*:\s*(.*?)(?:\n\s*Action\s*:|\n\s*Final\s*Answer\s*:|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        return match.group(1).strip() if match else None

    @staticmethod
    def _parse_action(text: str) -> Optional[Tuple[str, str]]:
        """Extract (tool_name, args) from an 'Action: tool_name(args)' line."""
        # Preferred format: tool_name(args)
        match = re.search(
            r"Action\s*:\s*([A-Za-z_]\w*)\s*\((.*?)\)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip(), match.group(2).strip()

        # Fallback: "Action: tool_name" possibly followed by "Action Input: ..."
        name_match = re.search(r"Action\s*:\s*([A-Za-z_]\w*)", text, re.IGNORECASE)
        if name_match:
            input_match = re.search(r"Action\s*Input\s*:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
            args = input_match.group(1).strip() if input_match else ""
            return name_match.group(1).strip(), args

        return None

    # --------------------------------------------------------------- execution
    def _execute_tool(self, tool_name: str, args: str) -> str:
        tool = self.tools_by_name.get(tool_name)
        if tool is None:
            available = ", ".join(self.tools_by_name.keys())
            return f"ERROR[UNKNOWN_TOOL]: '{tool_name}' is not a tool. Available: {available}."
        try:
            return str(tool["func"](args))
        except Exception as e:  # tools should not raise, but guard the loop anyway
            return f"ERROR[TOOL_EXCEPTION]: {e}"
