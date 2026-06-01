"""Web UI + JSON API for Lab 3 (Chatbot vs ReAct Agent).

This is the Vercel serverless entrypoint. Vercel loads the top-level `app`
(a Flask/WSGI application) and turns it into a serverless function. It can also
be run locally with `python api/index.py`.

Routes:
    GET  /            -> single-page comparison UI (HTML)
    GET  /api/health  -> which providers are configured
    POST /api/ask     -> run the chatbot and/or the ReAct agent on a question

The heavy lab logic (providers, tools, ReAct loop, telemetry) is reused as-is
from `src/`. Only OpenAI- and Gemini-style providers work in a serverless
environment; the local GGUF provider needs a model file on disk and is rejected.
"""

import json
import logging
import os
import sys
import time

# Make the project root importable so `src.*` resolves both locally and on Vercel.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flask import Flask, jsonify, request

from src.agent.agent import ReActAgent
from src.core.factory import get_provider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker
from src.tools import TOOLS

app = Flask(__name__)

# Baseline chatbot system prompt (no tools) -- the cinema booking persona.
CHATBOT_SYSTEM_PROMPT = (
    "Bạn là trợ lý đặt vé xem phim thân thiện của một rạp chiếu. Trả lời ngắn gọn, "
    "lịch sự bằng tiếng Việt. Bạn KHÔNG có quyền truy cập hệ thống đặt vé thật, nên hãy "
    "tư vấn dựa trên hiểu biết chung và nhắc khách rằng để xem suất chiếu/ghế chính xác "
    "và đặt vé thì cần dùng chế độ Agent (có công cụ). Khi khách muốn đặt vé, hãy hỏi "
    "đủ thông tin: tên phim, suất chiếu, số ghế và tên khách hàng."
)

DEMO_QUESTIONS = [
    "Hôm nay có những phim nào đang chiếu?",
    "Phim Mai có những suất chiếu nào và còn ghế không?",
    "Đặt 2 vé phim Mai suất 17:30 cho Khai.",
]

ALLOWED_PROVIDERS = {"openai", "google", "gemini"}

# Cache providers per-name so a warm serverless container reuses clients.
_PROVIDER_CACHE = {}


def _get_provider(name):
    name = (name or os.getenv("DEFAULT_PROVIDER", "openai")).lower()
    if name not in ALLOWED_PROVIDERS:
        raise ValueError(
            f"Provider '{name}' is not available in the web app. "
            "Use 'openai' or 'google'."
        )
    if name not in _PROVIDER_CACHE:
        _PROVIDER_CACHE[name] = get_provider(name)
    return _PROVIDER_CACHE[name]


class _EventCapture(logging.Handler):
    """Collect structured telemetry events emitted during a single request.

    The agent and the metrics tracker log JSON payloads to the shared logger.
    We temporarily attach this handler, run the agent, then read back the
    events to build the reasoning trace shown in the UI.
    """

    def __init__(self):
        super().__init__()
        self.events = []

    def emit(self, record):
        try:
            self.events.append(json.loads(record.getMessage()))
        except (ValueError, TypeError):
            pass


# Keep only the most recent turns so prompts stay small (token budget / timeout).
MAX_HISTORY_TURNS = 6


def _format_history(history):
    """Turn a [{role, content}] list into a plain-text transcript for prompting.

    Both providers expose a single-prompt `generate(prompt, system_prompt)` API,
    so multi-turn context is embedded as text rather than a structured message list.
    """
    if not history:
        return ""
    turns = history[-MAX_HISTORY_TURNS * 2:]
    lines = []
    for turn in turns:
        role = (turn.get("role") or "user").lower()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        speaker = "User" if role == "user" else "Assistant"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def run_chatbot(provider, question, history=None):
    """One LLM call, no tools -- the baseline the agent is compared against."""
    start = time.time()
    convo = _format_history(history)
    prompt = f"{convo}\nUser: {question}" if convo else question
    result = provider.generate(prompt, system_prompt=CHATBOT_SYSTEM_PROMPT)
    usage = result.get("usage", {})
    tracker.track_request(
        provider=result.get("provider", "unknown"),
        model=provider.model_name,
        usage=usage,
        latency_ms=result.get("latency_ms", 0),
    )
    return {
        "answer": result.get("content", ""),
        "model": provider.model_name,
        "latency_ms": result.get("latency_ms", int((time.time() - start) * 1000)),
        "total_tokens": usage.get("total_tokens", 0),
    }


def run_agent(provider, question, prompt_version="v2", max_steps=6, history=None):
    capture = _EventCapture()
    logger.logger.addHandler(capture)
    start = time.time()
    convo = _format_history(history)
    # Give the agent prior context, but keep the actual task clearly marked so the
    # ReAct loop still answers the latest question.
    user_input = (
        f"Conversation so far:\n{convo}\n\nCurrent question: {question}"
        if convo
        else question
    )
    try:
        agent = ReActAgent(
            llm=provider,
            tools=TOOLS,
            max_steps=max_steps,
            prompt_version=prompt_version,
        )
        answer = agent.run(user_input)
    finally:
        logger.logger.removeHandler(capture)

    # Build a readable trace + token totals from the captured events.
    trace = []
    total_tokens = 0
    status = "success"
    for ev in capture.events:
        kind = ev.get("event")
        data = ev.get("data", {})
        if kind == "AGENT_STEP":
            trace.append(
                {
                    "step": data.get("step"),
                    "thought": data.get("thought"),
                    "action": data.get("action"),
                    "observation": data.get("observation"),
                    "final_answer": data.get("final_answer"),
                    "error": data.get("error"),
                }
            )
        elif kind == "AGENT_END":
            status = data.get("status", status)
        elif kind == "LLM_METRIC":
            total_tokens += data.get("total_tokens", 0)

    return {
        "answer": answer,
        "model": provider.model_name,
        "prompt_version": prompt_version,
        "status": status,
        "steps": len(trace),
        "latency_ms": int((time.time() - start) * 1000),
        "total_tokens": total_tokens,
        "trace": trace,
    }


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "default_provider": os.getenv("DEFAULT_PROVIDER", "openai"),
            "providers": {
                "openai": bool(os.getenv("OPENAI_API_KEY")),
                "google": bool(os.getenv("GEMINI_API_KEY")),
            },
            "demo_questions": DEMO_QUESTIONS,
        }
    )


@app.post("/api/ask")
def ask():
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'question'."}), 400

    mode = (body.get("mode") or "both").lower()  # both | chatbot | agent
    provider_name = body.get("provider")
    prompt_version = body.get("prompt_version", "v2")
    max_steps = int(body.get("max_steps", 6))
    history = body.get("history") or []
    if not isinstance(history, list):
        history = []

    try:
        provider = _get_provider(provider_name)
    except Exception as e:  # noqa: BLE001 - surface config errors to the UI
        return jsonify({"error": str(e)}), 400

    response = {"question": question, "provider": (provider_name or os.getenv("DEFAULT_PROVIDER", "openai")).lower()}
    try:
        if mode in ("both", "chatbot"):
            response["chatbot"] = run_chatbot(provider, question, history)
        if mode in ("both", "agent"):
            response["agent"] = run_agent(provider, question, prompt_version, max_steps, history)
    except Exception as e:  # noqa: BLE001 - LLM/network errors -> readable message
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 502

    return jsonify(response)


@app.get("/")
def index():
    return INDEX_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}


INDEX_HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>CinemaBot · Trợ lý đặt vé phim</title>
<style>
  :root {
    --bg: #faf8f2;
    --panel: #fffdf8;
    --panel-2: #f4f0e6;
    --border: #e6e0d2;
    --text: #2c2a26;
    --muted: #8c8675;
    --accent: #5b7cf0;
    --accent-2: #1fa97f;
    --warn: #cf8a2c;
    --err: #d9536a;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
    background: radial-gradient(1200px 600px at 80% -10%, #fffefb 0%, var(--bg) 55%);
    color: var(--text);
    min-height: 100vh;
  }
  header {
    padding: 28px 24px 8px;
    max-width: 1100px;
    margin: 0 auto;
  }
  h1 { margin: 0; font-size: 26px; letter-spacing: -0.3px; }
  .sub { color: var(--muted); margin-top: 6px; font-size: 14px; }
  main { max-width: 1100px; margin: 0 auto; padding: 16px 24px 60px; }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px;
  }
  .controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end; margin-bottom: 14px; }
  .field { display: flex; flex-direction: column; gap: 6px; }
  label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
  select, input, textarea {
    background: var(--panel-2);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 10px;
    padding: 10px 12px;
    font-size: 14px;
    outline: none;
  }
  select:focus, input:focus, textarea:focus { border-color: var(--accent); }
  option:disabled { color: #b8b2a2; }
  textarea { width: 100%; resize: vertical; min-height: 64px; }
  .row { display: flex; gap: 10px; margin-top: 12px; }
  .row textarea { flex: 1; }
  button {
    background: linear-gradient(180deg, var(--accent), #4f6fe6);
    color: white; border: none; border-radius: 10px;
    padding: 12px 20px; font-size: 14px; font-weight: 600; cursor: pointer;
    white-space: nowrap;
  }
  button:disabled { opacity: 0.6; cursor: not-allowed; }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
  .chip {
    background: var(--panel-2); border: 1px solid var(--border); color: var(--muted);
    border-radius: 999px; padding: 7px 12px; font-size: 12.5px; cursor: pointer;
  }
  .chip:hover { border-color: var(--accent); color: var(--text); }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 18px; }
  @media (max-width: 820px) { .grid { grid-template-columns: 1fr; } }
  .result h2 { font-size: 15px; margin: 0 0 4px; display: flex; align-items: center; gap: 8px; }
  .badge { font-size: 11px; padding: 3px 8px; border-radius: 999px; font-weight: 600; }
  .badge.base { background: rgba(255,180,84,0.15); color: var(--warn); }
  .badge.agent { background: rgba(56,211,159,0.15); color: var(--accent-2); }
  .meta { color: var(--muted); font-size: 12px; margin: 2px 0 12px; }
  .answer {
    background: var(--panel-2); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px; font-size: 14.5px; line-height: 1.55; white-space: pre-wrap; min-height: 48px;
  }
  .answer.final { border-color: rgba(56,211,159,0.4); }
  .trace { margin-top: 12px; }
  .trace summary { cursor: pointer; color: var(--muted); font-size: 13px; }
  .step {
    border-left: 2px solid var(--border); padding: 8px 0 8px 12px; margin-top: 10px; font-size: 13px;
  }
  .step .k { color: var(--accent); font-weight: 600; }
  .step .act { color: var(--accent-2); }
  .step .obs { color: var(--muted); white-space: pre-wrap; }
  .step .err { color: var(--err); }
  .placeholder { color: var(--muted); font-style: italic; }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid var(--muted);
    border-top-color: var(--accent); border-radius: 50%; animation: spin 0.7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .err-box { color: var(--err); }
  footer { text-align: center; color: var(--muted); font-size: 12px; padding: 24px; }

  /* ---- Conversation ---- */
  .chat {
    display: flex; flex-direction: column; gap: 14px;
    min-height: 240px; max-height: 56vh; overflow-y: auto;
    padding: 6px 4px 4px; margin-bottom: 8px;
  }
  .msg { display: flex; flex-direction: column; max-width: 86%; }
  .msg.user { align-self: flex-end; align-items: flex-end; }
  .msg.assistant { align-self: flex-start; align-items: flex-start; }
  .who { font-size: 11px; color: var(--muted); margin: 0 4px 4px; text-transform: uppercase; letter-spacing: 0.5px; }
  .bubble {
    border-radius: 14px; padding: 11px 14px; font-size: 14.5px; line-height: 1.55;
    white-space: pre-wrap; border: 1px solid var(--border); background: var(--panel-2);
  }
  .msg.user .bubble {
    background: linear-gradient(180deg, var(--accent), #4f6fe6); color: #fff; border-color: transparent;
    border-bottom-right-radius: 4px;
  }
  .msg.assistant .bubble { border-bottom-left-radius: 4px; }
  .bubble.agent { border-color: rgba(31,169,127,0.45); }
  .bubble.error { border-color: var(--err); color: var(--err); }
  .b-head { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
  .b-meta { color: var(--muted); font-size: 11.5px; margin-top: 6px; }
  .empty-state { color: var(--muted); font-style: italic; text-align: center; margin: auto; }
  .ghost {
    background: transparent; color: var(--muted); border: 1px solid var(--border);
    border-radius: 10px; padding: 8px 14px; font-weight: 500;
  }
  .ghost:hover { color: var(--text); border-color: var(--accent); }
</style>
</head>
<body>
<header>
  <h1>CinemaBot · Trợ lý đặt vé phim</h1>
  <div class="sub">Trò chuyện để xem phim đang chiếu, kiểm tra suất &amp; ghế, và đặt vé. Chọn chế độ <b>Agent</b> để dùng công cụ đặt vé thật (giữ ngữ cảnh qua các câu hỏi).</div>
</header>
<main>
  <div class="card">
    <div class="controls">
      <div class="field">
        <label for="provider">Provider</label>
        <select id="provider">
          <option value="openai">OpenAI</option>
          <option value="google" disabled>Google Gemini (không khả dụng)</option>
        </select>
      </div>
      <div class="field">
        <label for="version">Agent prompt</label>
        <select id="version">
          <option value="v2">v2 (hardened)</option>
          <option value="v1">v1 (baseline)</option>
        </select>
      </div>
      <div class="field">
        <label for="mode">Run</label>
        <select id="mode">
          <option value="both">Both</option>
          <option value="chatbot">Chatbot only</option>
          <option value="agent">Agent only</option>
        </select>
      </div>
      <div class="field" style="margin-left:auto;">
        <label>&nbsp;</label>
        <button id="clear" class="ghost" type="button">Cuộc trò chuyện mới</button>
      </div>
    </div>

    <div class="chat" id="chat"></div>

    <div class="row">
      <textarea id="question" placeholder="Nhập tin nhắn… (Ctrl/⌘ + Enter để gửi)"></textarea>
      <button id="ask">Gửi</button>
    </div>
    <div class="chips" id="chips"></div>
  </div>
</main>
<footer>Lab 3 · Agentic AI · TVmaze API (no key)</footer>

<script>
const DEMOS = [
  "Hôm nay có những phim nào đang chiếu?",
  "Phim Mai có những suất chiếu nào và còn ghế không?",
  "Đặt 2 vé phim Mai suất 17:30 cho Khai.",
  "Kiểm tra mã vé BK-ABC123",
];
const WELCOME = "Xin chào! Mình là CinemaBot — trợ lý đặt vé phim của rạp. "
  + "Mình có thể giúp bạn:\n"
  + "• Xem các phim đang chiếu hôm nay\n"
  + "• Kiểm tra suất chiếu, giá vé và ghế trống\n"
  + "• Đặt vé và tra cứu mã vé\n\n"
  + "Bạn muốn xem phim gì hôm nay? (Mẹo: chọn chế độ \"Agent\" để đặt vé thật, hoặc bấm một gợi ý bên dưới.)";
const $ = (id) => document.getElementById(id);

// Running conversation context sent back to the server on every turn.
let history = [];

const chips = $("chips");
DEMOS.forEach((q) => {
  const c = document.createElement("span");
  c.className = "chip";
  c.textContent = q;
  c.onclick = () => { $("question").value = q; $("question").focus(); };
  chips.appendChild(c);
});

function esc(s) {
  return (s ?? "").toString().replace(/[&<>]/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[m]));
}

function scrollToBottom() {
  const chat = $("chat");
  chat.scrollTop = chat.scrollHeight;
}

function hideEmpty() {
  const e = $("empty");
  if (e) e.remove();
}

function renderTrace(trace) {
  if (!trace || !trace.length) return "";
  const steps = trace.map((s) => {
    let html = `<div class="step"><span class="k">Step ${esc(s.step)}</span>`;
    if (s.thought) html += `<div><b>Thought:</b> ${esc(s.thought)}</div>`;
    if (s.action) html += `<div class="act"><b>Action:</b> ${esc(s.action)}</div>`;
    if (s.observation) html += `<div class="obs"><b>Obs:</b> ${esc(s.observation)}</div>`;
    if (s.error) html += `<div class="err"><b>Error:</b> ${esc(s.error)}</div>`;
    if (s.final_answer) html += `<div class="act"><b>Final:</b> ${esc(s.final_answer)}</div>`;
    return html + `</div>`;
  }).join("");
  return `<div class="trace"><details><summary>Reasoning trace (${trace.length} steps)</summary>${steps}</details></div>`;
}

function renderWelcome() {
  const wrap = document.createElement("div");
  wrap.className = "msg assistant";
  wrap.innerHTML = `<div class="who">CinemaBot</div><div class="bubble">${esc(WELCOME)}</div>`;
  $("chat").appendChild(wrap);
  scrollToBottom();
}

function addUserMessage(text) {
  hideEmpty();
  const wrap = document.createElement("div");
  wrap.className = "msg user";
  wrap.innerHTML = `<div class="who">Bạn</div><div class="bubble">${esc(text)}</div>`;
  $("chat").appendChild(wrap);
  scrollToBottom();
}

// Returns the container so we can replace its content when the response arrives.
function addAssistantPlaceholder() {
  hideEmpty();
  const wrap = document.createElement("div");
  wrap.className = "msg assistant";
  wrap.innerHTML = `<div class="who">Trợ lý</div><div class="bubble"><span class="spinner"></span> Đang xử lý…</div>`;
  $("chat").appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function chatbotBubble(c) {
  return `<div class="bubble">`
    + `<div class="b-head"><span class="badge base">Chatbot · no tools</span></div>`
    + `${esc(c.answer || "(empty)")}`
    + `<div class="b-meta">${esc(c.model)} · ${esc(c.latency_ms)} ms · ${esc(c.total_tokens)} tokens</div>`
    + `</div>`;
}

function agentBubble(a) {
  return `<div class="bubble agent">`
    + `<div class="b-head"><span class="badge agent">ReAct Agent · TVmaze tools</span></div>`
    + `${esc(a.answer || "(empty)")}`
    + `<div class="b-meta">${esc(a.model)} (${esc(a.prompt_version)}) · ${esc(a.steps)} steps · `
    + `${esc(a.latency_ms)} ms · ${esc(a.total_tokens)} tokens · ${esc(a.status)}</div>`
    + renderTrace(a.trace)
    + `</div>`;
}

async function ask() {
  const question = $("question").value.trim();
  if (!question) { $("question").focus(); return; }
  const mode = $("mode").value;
  const btn = $("ask");
  btn.disabled = true; btn.textContent = "Đang gửi…";

  addUserMessage(question);
  const sendHistory = history.slice();   // context BEFORE this turn
  history.push({ role: "user", content: question });
  $("question").value = "";
  const slot = addAssistantPlaceholder();

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question, mode,
        history: sendHistory,
        provider: $("provider").value,
        prompt_version: $("version").value,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || ("HTTP " + res.status));

    let html = "";
    if (data.chatbot) html += chatbotBubble(data.chatbot);
    if (data.agent) html += agentBubble(data.agent);
    slot.innerHTML = `<div class="who">Trợ lý</div>` + (html || `<div class="bubble">(empty)</div>`);

    // Use the agent's grounded answer for context when available, else the chatbot's.
    const assistantText = (data.agent && data.agent.answer) || (data.chatbot && data.chatbot.answer) || "";
    history.push({ role: "assistant", content: assistantText });
    scrollToBottom();
  } catch (e) {
    slot.innerHTML = `<div class="who">Trợ lý</div><div class="bubble error">Lỗi: ${esc(e.message)}</div>`;
    // Roll back the user turn so a failed request doesn't poison future context.
    history.pop();
    scrollToBottom();
  } finally {
    btn.disabled = false; btn.textContent = "Gửi";
    $("question").focus();
  }
}

function clearConversation() {
  history = [];
  $("chat").innerHTML = "";
  renderWelcome();
  $("question").focus();
}

$("ask").onclick = ask;
$("clear").onclick = clearConversation;
$("question").addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") ask();
});

// Greet the user with an AI message as soon as the page loads.
renderWelcome();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
