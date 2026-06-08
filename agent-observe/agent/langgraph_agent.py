"""
LangGraph ReAct Agent — Trace AI integration test
==================================================

A medium-complexity research agent that demonstrates:
  - Multi-step ReAct reasoning loop (think → act → observe → repeat)
  - 4 diverse tools: web search, Wikipedia, calculator, text analyzer
  - Multiple LLM calls per run (typically 3-6)
  - Real API failures and retries
  - Full Trace AI observability via manual span management +
    LangGraph-aware callback handler

Architecture:
    __start__
        │
      agent  ← ─ ─ ─ ┐
        │              │
   tools_condition     │
    ├── "tools" ──────┘
    └── "__end__"
"""

import os
import sys
import math
import requests
from datetime import datetime

from dotenv import load_dotenv

# LangGraph / LangChain
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.callbacks.base import BaseCallbackHandler

# Trace AI
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from tracer import trace_manager

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

LANGSEARCH_KEY = os.getenv("LANGSEARCH_API_KEY", "")
GOOGLE_KEY     = os.getenv("GOOGLE_API_KEY", "")
XAI_KEY        = os.getenv("XAI_API_KEY", "")
GROQ_KEY       = os.getenv("GROQ_API_KEY", "")


# ══════════════════════════════════════════════════════════════════════
#  TOOLS
# ══════════════════════════════════════════════════════════════════════

@tool
def web_search(query: str) -> str:
    """Search the web for current information, news, and facts about any topic.
    Use this to find up-to-date information that may not be in training data."""
    if not LANGSEARCH_KEY:
        return "Web search unavailable: LANGSEARCH_API_KEY not configured."
    try:
        resp = requests.post(
            "https://api.langsearch.com/v1/web-search",
            headers={
                "Authorization": f"Bearer {LANGSEARCH_KEY}",
                "Content-Type":  "application/json",
            },
            json={"query": query, "summary": True, "count": 5},
            timeout=10,
        )
        if resp.status_code == 429:
            return f"Web search rate-limited. Retry in a moment or use wikipedia_search instead."
        if resp.status_code != 200:
            return f"Web search failed: HTTP {resp.status_code} — {resp.text[:120]}"

        try:
            data = resp.json()
        except Exception:
            return f"Web search returned malformed response for: {query}"

        # LangSearch wraps results under "data" key
        web_data = data.get("data", data)   # fallback to top-level if no "data" key
        pages = web_data.get("webPages", {}).get("value", [])
        if not pages:
            total = web_data.get("webPages", {}).get("totalEstimatedMatches", 0) or 0
            if not total:
                return (
                    f"No web results returned for '{query}'. "
                    f"Try a shorter or different query, or use wikipedia_search."
                )
            return f"API found ~{total} matches but returned no page data. Try a simpler query."

        parts = []
        for p in pages[:4]:
            title   = p.get("name", "")
            snippet = p.get("summary") or p.get("snippet", "")
            url     = p.get("url", "")
            if snippet:
                # Truncate each snippet to avoid Wikipedia full-article dumps
                snippet = snippet[:800] + ("…" if len(snippet) > 800 else "")
                parts.append(f"**{title}**\n{snippet}\nSource: {url}")
            elif title and url:
                parts.append(f"**{title}** — {url}")

        if parts:
            result = "\n\n---\n\n".join(parts)
            # Hard cap at 4000 chars — avoid blowing the LLM context window
            if len(result) > 4000:
                result = result[:4000] + "\n\n[...truncated for context length]"
            return result
        # Pages present but no snippets — return bare links
        return "\n".join(f"- {p.get('name','')}: {p.get('url','')}" for p in pages[:4])

    except requests.exceptions.Timeout:
        return "Web search timed out (>10s). Try wikipedia_search instead."
    except Exception as e:
        return f"Web search error: {e}"


@tool
def wikipedia_search(topic: str) -> str:
    """Look up detailed encyclopedic background information about a topic.
    Best for understanding concepts, history, and well-established facts."""
    # Wikipedia requires a descriptive User-Agent — plain requests gets 403
    HEADERS = {
        "User-Agent": "TraceAI-ResearchAgent/1.0 (https://github.com/trace-ai; contact@trace.ai)"
    }
    try:
        api = "https://en.wikipedia.org/w/api.php"

        # Step 1 — find article title
        search = requests.get(api, params={
            "action": "opensearch", "search": topic,
            "limit": 1, "namespace": 0, "format": "json",
        }, headers=HEADERS, timeout=8)
        if search.status_code == 403:
            return f"Wikipedia blocked the request (403). Topic: '{topic}'."
        try:
            results = search.json()
        except Exception:
            return f"Wikipedia search failed for '{topic}' (empty or invalid response)."
        titles  = results[1] if len(results) > 1 else []
        if not titles:
            return f"No Wikipedia article found for '{topic}'."

        title = titles[0]

        # Step 2 — fetch intro extract
        extract_resp = requests.get(api, params={
            "action": "query", "prop": "extracts",
            "exintro": True, "explaintext": True,
            "titles": title, "format": "json", "redirects": 1,
        }, headers=HEADERS, timeout=8)
        try:
            pages = extract_resp.json().get("query", {}).get("pages", {})
        except Exception:
            return f"Wikipedia returned an invalid response for '{title}'."
        for pid, page in pages.items():
            if pid != "-1":
                text = page.get("extract", "").strip()
                if text:
                    # Cap at 1200 chars so we don't overload the LLM context
                    preview = text[:1200] + ("…" if len(text) > 1200 else "")
                    return f"**{title}** (Wikipedia)\n\n{preview}"

        return f"Article '{title}' has no extractable content."
    except requests.exceptions.Timeout:
        return f"Wikipedia lookup timed out for '{topic}'."
    except Exception as e:
        return f"Wikipedia error: {e}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Supports: +, -, *, /, **, sqrt, log, log10, sin, cos, tan, pi, e, abs, round.

    Examples:
      calculator("2 ** 10")            → 1024
      calculator("sqrt(144)")          → 12.0
      calculator("(1 + 0.07) ** 10")   → compound growth
      calculator("log10(1000000)")      → 6.0
    """
    safe_globals = {
        "__builtins__": {},
        "abs": abs, "round": round, "pow": pow,
        "max": max, "min": min, "sum": sum,
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "log2": math.log2, "exp": math.exp,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "floor": math.floor, "ceil": math.ceil,
        "pi": math.pi, "e": math.e,
    }
    try:
        result = eval(str(expression).strip(), safe_globals)
        # Format nicely
        if isinstance(result, float):
            if result == int(result) and abs(result) < 1e15:
                return f"{expression} = {int(result)}"
            return f"{expression} = {result:.6g}"
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "Error: Division by zero."
    except NameError as e:
        return f"Unknown function or variable: {e}"
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def text_analyzer(text: str) -> str:
    """Analyze a piece of text and return useful statistics.

    Returns: word count, character count, sentence count, paragraph count,
    estimated reading time, average words per sentence, and a vocabulary
    richness score.
    """
    if not text or not text.strip():
        return "No text provided."

    words      = text.split()
    sentences  = max(1, text.count('.') + text.count('!') + text.count('?'))
    chars      = len(text)
    paragraphs = max(1, text.count('\n\n') + 1)
    read_secs  = round(len(words) / 200 * 60)    # ~200 wpm average
    read_min   = read_secs // 60
    read_sec   = read_secs % 60
    unique     = len(set(w.lower().strip('.,!?;:"\'') for w in words))
    vocab_rich = round(unique / max(len(words), 1) * 100, 1)

    return (
        f"Word count:          {len(words)}\n"
        f"Character count:     {chars}\n"
        f"Sentence count:      {sentences}\n"
        f"Paragraph count:     {paragraphs}\n"
        f"Unique words:        {unique}\n"
        f"Vocabulary richness: {vocab_rich}%\n"
        f"Avg words/sentence:  {round(len(words)/sentences, 1)}\n"
        f"Reading time:        {read_min}m {read_sec}s (at 200 wpm)"
    )


TOOLS = [web_search, wikipedia_search, calculator, text_analyzer]

COST_PER_1M = {
    "grok-3":                   {"input": 3.00,  "output": 15.00},
    "grok-3-fast":              {"input": 0.60,  "output": 4.00},
    "grok-3-mini":              {"input": 0.30,  "output": 0.50},
    "grok-2-1212":              {"input": 2.00,  "output": 10.00},
    "openai/gpt-oss-120b":      {"input": 0.15,  "output": 0.75},
    "qwen/qwen3-32b":           {"input": 0.29,  "output": 0.59},
    "llama-3.3-70b-versatile":  {"input": 0.59,  "output": 0.79},
    "llama-3.1-8b-instant":     {"input": 0.05,  "output": 0.08},
    "gemini-2.5-flash":         {"input": 0.15,  "output": 0.60},
    "gemini-2.5-pro":           {"input": 1.25,  "output": 10.00},
    "gemini-2.0-flash":         {"input": 0.10,  "output": 0.40},
}

SYSTEM_PROMPT = """You are a thorough research assistant with access to web search, Wikipedia, a calculator, and a text analyzer.

When answering a question:
1. Use web_search to find current, up-to-date information
2. Use wikipedia_search to understand background context and established facts
3. Use calculator whenever numerical computation or comparison is needed
4. Use text_analyzer if you need statistics about content you retrieved
5. Always synthesize information from multiple sources before giving your final answer
6. Be specific — include numbers, dates, and sources in your answer

Today's date: {date}
"""


# ══════════════════════════════════════════════════════════════════════
#  TRACE AI CALLBACK — LangGraph-aware
#  Only handles LLM and tool events.
#  The outer trace lifecycle is managed manually by run_langgraph_agent().
# ══════════════════════════════════════════════════════════════════════

class LangGraphTraceCallback(BaseCallbackHandler):
    """Lightweight callback that traces individual LLM calls and tool calls
    within the ReAct loop. Does NOT manage the top-level trace lifecycle —
    that is handled manually so we get clean session-aware traces."""

    def __init__(self):
        super().__init__()
        self._spans:           dict = {}   # run_id → span
        self._model_per_run:   dict = {}   # run_id → model name (for cost calc)
        self.llm_call_count:   int  = 0
        self.tool_call_count:  int  = 0

    # ── LLM ──────────────────────────────────────────────────────────
    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self.llm_call_count += 1
        model = (
            (serialized.get("kwargs") or {}).get("model")
            or (serialized.get("kwargs") or {}).get("model_name")
            or (serialized.get("id") or ["unknown"])[-1]
            or "gemini-2.0-flash"
        )
        self._model_per_run[str(run_id)] = model
        span = trace_manager.start_span("llm_call", f"reasoning_step_{self.llm_call_count}", {
            "model":        model,
            "step":         self.llm_call_count,
            "prompt_chars": sum(len(p) for p in prompts),
        })
        self._spans[str(run_id)] = span

    def on_llm_end(self, response, *, run_id, **kwargs):
        in_tok, out_tok = 0, 0

        # Path 1 — llm_output dict (OpenAI-style)
        llm_out = getattr(response, "llm_output", None) or {}
        usage   = llm_out.get("token_usage") or llm_out.get("usage") or {}
        in_tok  = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
        out_tok = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)

        # Path 2 — Gemini generation_info
        if not in_tok:
            for gen_list in (getattr(response, "generations", None) or []):
                for gen in (gen_list if isinstance(gen_list, list) else [gen_list]):
                    info = getattr(gen, "generation_info", None) or {}
                    meta = info.get("usage_metadata") or {}
                    if meta:
                        in_tok  = meta.get("prompt_token_count", 0)
                        out_tok = meta.get("candidates_token_count", 0)
                        break
                if in_tok:
                    break

        model   = self._model_per_run.pop(str(run_id), "gemini-2.0-flash")
        pricing = COST_PER_1M.get(model, {"input": 0.15, "output": 0.60})
        cost    = round(
            (in_tok  / 1_000_000 * pricing["input"]) +
            (out_tok / 1_000_000 * pricing["output"]),
            6
        )

        trace_manager.end_span("success", {
            "input_tokens":  in_tok,
            "output_tokens": out_tok,
            "total_tokens":  in_tok + out_tok,
            "cost_usd":      cost,
            "model":         model,
            "pricing_note":  f"${pricing['input']}/M in · ${pricing['output']}/M out",
        })
        self._spans.pop(str(run_id), None)

    def on_llm_error(self, error, *, run_id, **kwargs):
        trace_manager.end_span("failed", {"error": str(error)})
        self._spans.pop(str(run_id), None)

    # ── Tools ─────────────────────────────────────────────────────────
    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self.tool_call_count += 1
        name = serialized.get("name", "tool")
        span = trace_manager.start_span("tool_call", name, {
            "input":    str(input_str)[:400],
            "tool_num": self.tool_call_count,
        })
        self._spans[str(run_id)] = span

    def on_tool_end(self, output, *, run_id, **kwargs):
        # In newer LangGraph versions, output may be a ToolMessage object
        if hasattr(output, "content"):
            out_str = str(output.content)
        else:
            out_str = str(output)
        trace_manager.end_span("success", {
            "output_preview": out_str[:400],
            "output_chars":   len(out_str),
        })
        self._spans.pop(str(run_id), None)

    def on_tool_error(self, error, *, run_id, **kwargs):
        trace_manager.end_span("failed", {"error": str(error)})
        self._spans.pop(str(run_id), None)


# ══════════════════════════════════════════════════════════════════════
#  GRAPH DEFINITION
# ══════════════════════════════════════════════════════════════════════

def _build_graph():
    """Build the LangGraph ReAct graph with automatic model fallback."""

    def call_agent(state: MessagesState):
        """
        Agent node — tries Groq first, falls back to Gemini models.
        Rate-limited (429 / RESOURCE_EXHAUSTED) providers are skipped.
        Any other error is re-raised immediately.
        """
        last_err = None

        def _is_text_tool_call(msg) -> bool:
            """Return True if the model output a tool call as plain text
            instead of a structured tool_calls object.
            llama-3.3-70b-versatile sometimes does this."""
            content = getattr(msg, "content", "") or ""
            if isinstance(content, list):
                content = " ".join(
                    (item.get("text", "") if isinstance(item, dict) else str(item))
                    for item in content
                )
            has_text_call = (
                "<function(" in content
                or "<tool_call>" in content
                or content.strip().startswith("<function")
            )
            has_real_call = bool(getattr(msg, "tool_calls", None))
            return has_text_call and not has_real_call

        # ── 1. Groq (primary — free tier, 14K req/day) ─────────────────
        # Model selection notes (verified against Groq's live model list):
        #   openai/gpt-oss-120b      → Best structured tool-calling on Groq.
        #   qwen/qwen3-32b           → Reliable structured tool calling.
        #   llama-3.3-70b-versatile  → Capable but OCCASIONALLY emits text-format
        #                              tool calls; we detect + skip those below.
        #   llama-3.1-8b-instant     → Fast last resort before Gemini.
        if GROQ_KEY:
            for groq_model in [
                "openai/gpt-oss-120b",       # best structured tool calling
                "qwen/qwen3-32b",            # reliable tool calling
                "llama-3.3-70b-versatile",   # capable, text-call detection guards it
                "llama-3.1-8b-instant",      # fast last resort
            ]:
                try:
                    from langchain_groq import ChatGroq
                    llm = ChatGroq(
                        model=groq_model,
                        groq_api_key=GROQ_KEY,
                        temperature=0,
                    ).bind_tools(TOOLS)
                    response_msg = llm.invoke(state["messages"])

                    # If the model output a TEXT-format tool call instead of
                    # a structured one, LangGraph can't execute it — skip this model.
                    if _is_text_tool_call(response_msg):
                        print(f"[LangGraph] {groq_model} returned text-format tool call "
                              f"(not structured JSON) — skipping to next model.")
                        last_err = ValueError(f"{groq_model}: text-format tool call")
                        continue

                    return {"messages": [response_msg]}
                except Exception as e:
                    err = str(e)
                    _groq_skip = ("429", "rate_limit", "quota", "RateLimitError",
                                  "403", "credits", "spending",
                                  "tool_use_failed", "invalid_request_error", "400",
                                  "text-format tool call")
                    if any(x in err for x in _groq_skip):
                        last_err = e
                        print(f"[LangGraph] {groq_model} unavailable ({err[:80]}), trying next...")
                        continue
                    raise

        # ── 2. Gemini fallback ─────────────────────────────────────────
        for gemini_model in ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]:
            try:
                llm = ChatGoogleGenerativeAI(
                    model=gemini_model,
                    temperature=0,
                    google_api_key=GOOGLE_KEY,
                ).bind_tools(TOOLS)
                return {"messages": [llm.invoke(state["messages"])]}
            except Exception as e:
                err = str(e)
                if "RESOURCE_EXHAUSTED" in err or "429" in err or "quota" in err.lower():
                    last_err = e
                    print(f"[LangGraph] {gemini_model} quota-exhausted, trying next…")
                    continue
                raise

        raise last_err or RuntimeError("All models (xAI + Groq + Gemini) are quota-exhausted.")

    builder = StateGraph(MessagesState)
    builder.add_node("agent",  call_agent)
    builder.add_node("tools",  ToolNode(TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile()


# Pre-compile once — reused across all runs
_graph = _build_graph()


# ══════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_langgraph_agent(
    user_query:  str,
    session_id:  str = None,
    user_id:     str = None,
    turn_number: int = None,
) -> str:
    """
    Run the LangGraph ReAct agent and return the final answer.

    Span hierarchy:
        agent_start  (root)
          react_loop (phase)
            reasoning_step_1 (llm_call)   — decide first action
            web_search       (tool_call)   — execute
            reasoning_step_2 (llm_call)   — review, decide next
            wikipedia_search (tool_call)   — execute
            reasoning_step_3 (llm_call)   — synthesize answer
          agent_response (final)
    """
    # ── Start trace (session-aware) ───────────────────────────────────
    trace_manager.start_trace(
        "langgraph_react_agent",
        session_id  = session_id,
        user_id     = user_id,
        turn_number = turn_number,
    )
    primary = ("openai/gpt-oss-120b" if GROQ_KEY else "gemini-2.5-flash")
    trace_manager.start_span("agent_start", "react_agent_root", {
        "query":         user_query,
        "framework":     "langgraph",
        "primary_model": primary,
        "fallback_chain":"Groq → Gemini",
        "tools":         [t.name for t in TOOLS],
    })

    callback = LangGraphTraceCallback()

    try:
        # ── ReAct reasoning loop ──────────────────────────────────────
        trace_manager.start_span("phase", "react_loop", {
            "description": "LangGraph ReAct — iterative think→act→observe loop",
            "max_iterations": 10,
        })

        system = SystemMessage(content=SYSTEM_PROMPT.format(
            date=datetime.now().strftime("%B %d, %Y")
        ))
        human  = HumanMessage(content=user_query)

        result = _graph.invoke(
            {"messages": [system, human]},
            config={
                "callbacks":       [callback],
                "recursion_limit": 12,
            },
        )

        trace_manager.end_span("success", {
            "llm_calls":  callback.llm_call_count,
            "tool_calls": callback.tool_call_count,
        })

        # ── Extract final answer ──────────────────────────────────────
        # Scan messages in reverse to find the last AIMessage with real text.
        # Skip: empty messages, tool-call-only messages, and messages whose
        # content is a raw text-format tool call (e.g. <function(web_search)...>)
        # — those mean the model failed to produce structured tool calls.
        from langchain_core.messages import AIMessage as _AIMsg
        answer = ""
        for msg in reversed(result["messages"]):
            content = getattr(msg, "content", "") or ""
            if isinstance(content, list):
                text = " ".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                ).strip()
            else:
                text = str(content).strip()

            # Skip tool-call-only messages (empty text, non-empty tool_calls)
            if not text or getattr(msg, "tool_calls", None):
                continue

            # Skip raw text-format tool calls — these are broken model outputs,
            # not real answers (e.g. "<function(web_search){...}>")
            if "<function(" in text or "<tool_call>" in text:
                continue

            answer = text
            break

        # Server-side log so we can confirm the extraction
        print(f"[LangGraph] messages={len(result['messages'])} "
              f"answer_len={len(answer)} "
              f"last_msg_type={type(result['messages'][-1]).__name__} "
              f"answer_preview={answer[:80]!r}")

        trace_manager.start_span("agent_response", "final_response", {
            "response_preview": answer[:500],
            "total_llm_calls":  callback.llm_call_count,
            "total_tool_calls": callback.tool_call_count,
        })
        trace_manager.end_span("success")

        # ── Close root ────────────────────────────────────────────────
        trace_manager.end_span("success")   # agent_start
        trace_manager.end_trace("success")
        return answer

    except Exception as exc:
        err = str(exc)
        # Force-close any open spans before recording failure
        trace_manager.start_span("failure", "agent_failure", {
            "error":       err,
            "llm_calls":  callback.llm_call_count,
            "tool_calls": callback.tool_call_count,
        })
        trace_manager.end_span("failed")
        trace_manager.end_trace("failed")
        return f"Agent error: {err}"


# ── Quick test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    query = (
        "What is the current market cap of NVIDIA? "
        "Also find out who founded the company and in what year. "
        "Calculate how many years ago that was from today."
    )
    print(f"\nQuery: {query}\n{'-'*60}")
    answer = run_langgraph_agent(query)
    print(f"\nAnswer:\n{answer}")
