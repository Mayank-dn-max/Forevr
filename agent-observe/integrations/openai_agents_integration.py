"""
integrations/openai_agents_integration.py
==========================================
Trace AI adapter for the OpenAI Agents SDK (the `agents` package).

How it works:
    The OpenAI Agents SDK has a built-in tracing interface: TracingProcessor.
    It fires on_trace_start/end and on_span_start/end for every operation.
    We implement that interface and translate the events to Trace AI spans.

Usage:
    from integrations.openai_agents_integration import TraceAIProcessor
    from agents import Runner, add_trace_processor

    # Register once at startup — instruments ALL agent runs globally
    add_trace_processor(TraceAIProcessor())

    # Then run agents normally
    result = await Runner.run(agent, "What is the weather in NYC?")

    # Full trace appears in Trace AI dashboard automatically.

OpenAI Agents SDK span types we handle:
    - agent       → agent_start span
    - generation  → llm_call span (with token counts)
    - tool_call   → tool_call span
    - handoff     → phase span
    - guardrail   → tool_call span
    - custom      → phase span
"""


try:
    from agents.tracing import TracingProcessor, Trace, Span
except ImportError:
    try:
        # Older versions of the package
        from openai.agents.tracing import TracingProcessor, Trace, Span
    except ImportError:
        raise ImportError(
            "The OpenAI Agents SDK is required for this integration.\n"
            "Install it with: pip install openai-agents"
        )

from traceai._tracer import trace_manager


# Maps OpenAI Agents SDK span type → our span type
_SPAN_TYPE_MAP = {
    "agent":      "agent_start",
    "generation": "llm_call",
    "tool_call":  "tool_call",
    "handoff":    "phase",
    "guardrail":  "tool_call",
    "custom":     "phase",
}


class TraceAIProcessor(TracingProcessor):
    """
    OpenAI Agents SDK tracing processor.

    Register once with add_trace_processor() and every agent run
    is automatically traced in Trace AI.
    """

    # ── Trace lifecycle ────────────────────────────────────────────────
    def on_trace_start(self, trace: "Trace") -> None:
        name = getattr(trace, "name", None) or "openai_agent"
        trace_manager.start_trace(name)
        trace_manager.start_span("agent_start", name, {
            "framework":  "openai_agents",
            "trace_id":   getattr(trace, "trace_id", ""),
            "group_id":   getattr(trace, "group_id", "") or "",
        })

    def on_trace_end(self, trace: "Trace") -> None:
        error = getattr(trace, "error", None)
        if error:
            trace_manager.end_span("failed", {"error": str(error)})
            trace_manager.end_trace("failed")
        else:
            trace_manager.end_span("success")
            trace_manager.end_trace("success")

    # ── Span lifecycle ─────────────────────────────────────────────────
    def on_span_start(self, span: "Span") -> None:
        try:
            data      = getattr(span, "span_data", None) or span
            raw_type  = getattr(data, "type", None) or getattr(span, "type", "custom")
            our_type  = _SPAN_TYPE_MAP.get(raw_type, "phase")

            # Extract name from span data
            name = (
                getattr(data, "name", None)
                or getattr(data, "model", None)
                or getattr(data, "tool_name", None)
                or raw_type
            )

            # Build metadata from span data fields
            metadata = {"openai_span_type": raw_type}

            # LLM generation spans
            if raw_type == "generation":
                metadata["model"] = getattr(data, "model", "")
                inp = getattr(data, "input", None)
                if inp:
                    metadata["input_preview"] = str(inp)[:200]

            # Tool call spans
            elif raw_type == "tool_call":
                metadata["tool_name"]  = getattr(data, "name", "")
                tool_in = getattr(data, "input", None)
                if tool_in:
                    metadata["input"] = str(tool_in)[:200]

            trace_manager.start_span(our_type, name, metadata)

        except Exception:
            pass  # observability must never crash the agent

    def on_span_end(self, span: "Span") -> None:
        try:
            data  = getattr(span, "span_data", None) or span
            error = getattr(span, "error", None)

            meta_update = {}

            if error:
                meta_update["error"] = str(error)
                trace_manager.end_span("failed", meta_update)
                return

            raw_type = getattr(data, "type", "custom")

            # Capture token usage from generation spans
            if raw_type == "generation":
                usage = getattr(data, "usage", None)
                if usage:
                    in_tok  = getattr(usage, "input_tokens",  0) or 0
                    out_tok = getattr(usage, "output_tokens", 0) or 0
                    meta_update.update({
                        "input_tokens":  in_tok,
                        "output_tokens": out_tok,
                        "total_tokens":  in_tok + out_tok,
                    })
                output = getattr(data, "output", None)
                if output:
                    meta_update["output_preview"] = str(output)[:300]

            # Capture tool output
            elif raw_type == "tool_call":
                output = getattr(data, "output", None)
                if output:
                    meta_update["output_preview"] = str(output)[:300]

            trace_manager.end_span("success", meta_update or None)

        except Exception:
            pass
