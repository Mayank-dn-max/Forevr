"""
forevr.universal — Works with any agent / framework

Three patterns for any custom agent:

PATTERN 1 — Context Manager
    from forevr.universal import span

    with span("tool_call", "database_lookup", {"table": "users"}):
        result = db.query(q)

PATTERN 2 — Decorator
    from forevr.universal import watch

    @watch("tool_call", name="web_search", metadata_fn=lambda q: {"query": q})
    def web_search(query: str):
        return real_search(query)

PATTERN 3 — Full trace lifecycle
    from forevr.universal import Trace, span

    with Trace("my_agent", {"user_id": "u123"}):
        with span("phase", "retrieval"):
            with span("tool_call", "search"):
                results = search(query)
"""

import functools
import contextlib

from forevr._tracer import trace_manager


@contextlib.contextmanager
def span(
    span_type: str,
    name: str,
    metadata: dict = None,
    caused_by: str = None,
    trigger: str = None,
    relation: str = None,
):
    """
    Context manager — span opens on enter, closes on exit.
    Auto-marks as 'failed' if an exception is raised.
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


def watch(span_type: str = "tool_call", name: str = None, metadata_fn=None):
    """
    Decorator — wraps any sync or async function with a span.
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


class Trace:
    """
    Context manager wrapping a complete agent run with a trace.

    Usage:
        with Trace("my_agent"):
            result = my_agent.run(query)
    """

    def __init__(
        self,
        name:        str  = "agent",
        session_id:  str  = None,
        user_id:     str  = None,
        turn_number: int  = None,
        metadata:    dict = None,
    ):
        self.name        = name
        self.session_id  = session_id
        self.user_id     = user_id
        self.turn_number = turn_number
        self.metadata    = metadata or {}

    def __enter__(self):
        trace_manager.start_trace(
            self.name,
            session_id  = self.session_id,
            user_id     = self.user_id,
            turn_number = self.turn_number,
        )
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
        return False


__all__ = ["span", "watch", "Trace", "trace_manager"]
