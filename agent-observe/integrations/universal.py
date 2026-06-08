"""
integrations/universal.py
==========================
Framework-agnostic Trace AI instrumentation.

Use this when:
  - You have a custom agent (not LangChain / CrewAI / OpenAI Agents)
  - You want fine-grained control over what gets traced
  - You're building your own framework

Three patterns available:

PATTERN 1 — Context Manager (most common)
    with trace_span("tool_call", "web_search", {"query": q}):
        result = search(q)
    # span auto-closes on exit, even if an exception is raised

PATTERN 2 — Decorator (cleanest for tool functions)
    @observe("tool_call", name="web_search")
    def web_search(query: str):
        return search(query)

PATTERN 3 — Full manual control (most flexibility)
    agent_trace = new_trace("my_agent")
    with agent_trace:
        with trace_span("phase", "retrieval"):
            with trace_span("tool_call", "search"):
                results = search(query)

These all produce the same span format that Trace AI dashboard understands.
They work with async functions too.
"""

import functools
import contextlib

from traceai._tracer import trace_manager


# ── Pattern 1: Context Manager ─────────────────────────────────────────────

@contextlib.contextmanager
def trace_span(
    span_type: str,
    name: str,
    metadata: dict = None,
    caused_by: str = None,
    trigger: str = None,
    relation: str = None,
):
    """
    Context manager. Span opens on enter, closes on exit.
    Automatically marks as 'failed' if an exception is raised.

    Args:
        span_type: "tool_call" | "llm_call" | "phase" | "agent_start" | "agent_response"
        name:      Display name in dashboard (e.g. "web_search", "gpt-4o")
        metadata:  Any key-value data to attach (latency, tokens, URLs, etc.)
        caused_by: span_id of the span that caused this one (for causal graph)
        trigger:   Human label: "fallback_after_timeout", "retry_on_503", etc.
        relation:  "retry" | "fallback" | "compensation"

    Example:
        import time
        with trace_span("tool_call", "web_search", {"query": q}) as span:
            t0 = time.time()
            results = real_search(q)
            # Update span metadata before it closes:
            trace_manager.update_span({
                "results_found": len(results),
                "latency_ms": round((time.time()-t0)*1000)
            })
    """
    s = trace_manager.start_span(
        span_type, name, metadata or {},
        caused_by=caused_by, trigger=trigger, relation=relation,
    )
    try:
        yield s
    except Exception as e:
        trace_manager.end_span("failed", {
            "error":          str(e),
            "exception_type": type(e).__name__,
        })
        raise
    else:
        trace_manager.end_span("success")


# ── Pattern 2: Decorator ───────────────────────────────────────────────────

def observe(
    span_type: str = "tool_call",
    name: str = None,
    metadata_fn=None,    # callable(*args, **kwargs) → dict
):
    """
    Decorator. Wraps any function (sync or async) with a span.

    Args:
        span_type:   Type of span (default: "tool_call")
        name:        Span name (default: function name)
        metadata_fn: Optional callable that receives the function's arguments
                     and returns a dict of metadata to attach.

    Example:
        @observe("tool_call", name="web_search",
                 metadata_fn=lambda q: {"query": q})
        def web_search(query: str):
            return real_search(query)

        @observe("llm_call", name="gpt-4o")
        async def call_llm(prompt: str):
            return await openai_client.chat.completions.create(...)
    """
    def decorator(func):
        _name = name or func.__name__

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            meta = metadata_fn(*args, **kwargs) if metadata_fn else {}
            trace_manager.start_span(span_type, _name, meta)
            try:
                result = func(*args, **kwargs)
                trace_manager.end_span("success")
                return result
            except Exception as e:
                trace_manager.end_span("failed", {
                    "error":          str(e),
                    "exception_type": type(e).__name__,
                })
                raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            meta = metadata_fn(*args, **kwargs) if metadata_fn else {}
            trace_manager.start_span(span_type, _name, meta)
            try:
                result = await func(*args, **kwargs)
                trace_manager.end_span("success")
                return result
            except Exception as e:
                trace_manager.end_span("failed", {
                    "error":          str(e),
                    "exception_type": type(e).__name__,
                })
                raise

        import asyncio
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# ── Pattern 3: Full trace lifecycle ────────────────────────────────────────

class agent_trace:
    """
    Context manager that wraps a complete agent run with a trace.

    Usage:
        with agent_trace("my_custom_agent", {"user_id": "u123"}):
            # run your agent here
            result = my_agent.run(query)

    On clean exit  → trace status = "success"
    On exception   → trace status = "failed", exception re-raised
    """

    def __init__(self, name: str = "agent", metadata: dict = None):
        self.name     = name
        self.metadata = metadata or {}

    def __enter__(self):
        trace_manager.start_trace(self.name)
        trace_manager.start_span("agent_start", self.name, {
            "framework": "custom",
            **self.metadata,
        })
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            trace_manager.end_span("failed", {"error": str(exc_val)})
            trace_manager.end_trace("failed")
        else:
            trace_manager.end_span("success")
            trace_manager.end_trace("success")
        return False   # don't suppress the exception


# ── Convenience re-exports ──────────────────────────────────────────────────
# So you can do: from integrations.universal import *
__all__ = ["trace_span", "observe", "agent_trace", "trace_manager"]
