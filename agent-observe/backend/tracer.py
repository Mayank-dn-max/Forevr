"""
tracer.py — Trace AI Observability SDK
=======================================

Architecture:
  - ContextVar-based span propagation  (async-safe, Problem #2 fixed)
  - @observe decorator + context manager  (no manual spans, Problem #1 fixed)
  - Causal fields: caused_by / trigger / relation  (Problem #3 fixed)
  - Hierarchical cost aggregation per phase  (Problem #4 fixed)
  - execution_pattern classification  (Problem #5 fixed)
"""

import time
import uuid
import functools
import contextlib
from contextvars import ContextVar
from typing import Optional, Dict

def _db_save(trace):
    try:
        from database import save_trace
        save_trace(trace)
    except Exception as e:
        print(f"[tracer] Save failed: {e}")

# ── All state in ContextVars — safe for concurrent requests ──────────────────
_current_trace: ContextVar[Optional[Dict]] = ContextVar("_tracer_current_trace", default=None)
_span_tokens:   ContextVar[Optional[Dict]] = ContextVar("_tracer_span_tokens",   default=None)
_active_span:   ContextVar[Optional[Dict]] = ContextVar("_tracer_active_span",   default=None)

traces = []


class TraceManager:
    """Stateless — all execution state lives in ContextVars."""

    # ══════════════════════════════════════════════════════════════════════════
    # TRACE LIFECYCLE
    # ══════════════════════════════════════════════════════════════════════════

    def start_trace(
        self,
        name:         str,
        session_id:   Optional[str] = None,
        user_id:      Optional[str] = None,
        turn_number:  Optional[int] = None,
    ) -> Dict:
        trace = {
            "trace_id":    str(uuid.uuid4()),
            "name":        name,
            "start_time":  time.time(),
            "spans":       [],
            "status":      "running",
            "session_id":  session_id,
            "user_id":     user_id,
            "turn_number": turn_number,
        }
        _current_trace.set(trace)
        _span_tokens.set({})
        _active_span.set(None)
        return trace

    def end_trace(self, status: str = "success") -> Dict:
        while _active_span.get() is not None:
            self.end_span("failed")

        t = _current_trace.get()
        if t is None:
            return {}
        t["status"]   = status
        t["end_time"] = time.time()
        t["latency"]  = round(t["end_time"] - t["start_time"], 2)

        spans = t["spans"]

        # ── Span-level summary ────────────────────────────────────────────
        t["spans_total"]   = len(spans)
        t["spans_success"] = sum(1 for s in spans if s["status"] == "success")
        t["spans_failed"]  = sum(1 for s in spans if s["status"] == "failed")
        t["spans_skipped"] = sum(1 for s in spans if s["status"] == "skipped")

        # ── Token aggregation ─────────────────────────────────────────────
        t["total_input_tokens"]  = sum(s["metadata"].get("input_tokens",  0) for s in spans)
        t["total_output_tokens"] = sum(s["metadata"].get("output_tokens", 0) for s in spans)
        t["total_cost_usd"]      = round(sum(s["metadata"].get("cost_usd", 0) for s in spans), 6)

        # ── Problem #4: Hierarchical cost per phase ───────────────────────
        phase_costs = {}
        for s in spans:
            if s["type"] == "phase":
                descendants = self._get_descendants(spans, s["span_id"])
                phase_costs[s["name"]] = round(
                    sum(d["metadata"].get("cost_usd", 0) for d in descendants), 6
                )
        t["phase_costs"] = phase_costs

        # ── Problem #5: Trace-level execution pattern ─────────────────────
        patterns = [s["execution_pattern"] for s in spans if s.get("execution_pattern")]
        t["execution_patterns"] = list(dict.fromkeys(patterns))  # unique, ordered

        _current_trace.set(None)
        _span_tokens.set(None)
        traces.append(t)
        _db_save(t)
        return t

    # ══════════════════════════════════════════════════════════════════════════
    # SPAN LIFECYCLE  (low-level; prefer context manager or decorator above)
    # ══════════════════════════════════════════════════════════════════════════

    def start_span(
        self,
        span_type:  str,
        name:       str,
        metadata:   Optional[Dict] = None,
        # ── Problem #3: Causal fields ────────────────────────────────────
        caused_by:  Optional[str]  = None,   # span_id that triggered this span
        trigger:    Optional[str]  = None,   # human label: "fallback_after_timeout"
        relation:   Optional[str]  = None,   # "retry" | "fallback" | "compensation"
    ) -> Dict:
        """
        Open a new span.
        Automatically becomes a child of the currently active span.
        Safe to call from async code — uses ContextVar, not a plain list.
        """
        t = _current_trace.get()
        if t is None:
            t = self.start_trace("unnamed_agent")

        tokens = _span_tokens.get()
        if tokens is None:
            tokens = {}
            _span_tokens.set(tokens)

        parent    = _active_span.get()
        parent_id = parent["span_id"] if parent else None
        depth     = (parent.get("depth", -1) + 1) if parent else 0

        span: Dict = {
            "span_id":           str(uuid.uuid4())[:12],
            "parent_span_id":    parent_id,
            "depth":             depth,
            "type":              span_type,
            "name":              name,
            "start_time":        time.time(),
            "end_time":          None,
            "duration_ms":       None,
            "metadata":          dict(metadata or {}),
            "status":            "running",
            "retry_count":       0,
            "caused_by":         caused_by,
            "trigger":           trigger,
            "relation":          relation,
            "execution_pattern": None,
        }

        t["spans"].append(span)
        token = _active_span.set(span)
        tokens[span["span_id"]] = token
        return span

    def end_span(
        self,
        status:          str              = "success",
        metadata_update: Optional[Dict]   = None,
    ) -> Optional[Dict]:
        """
        Close the currently active span.
        Restores the parent span as the active context automatically.
        """
        span = _active_span.get()
        if span is None:
            return None

        tokens = _span_tokens.get() or {}
        token  = tokens.pop(span["span_id"], None)
        if token is not None:
            _active_span.reset(token)

        span["status"]     = status
        span["end_time"]   = time.time()
        span["duration_ms"] = round((span["end_time"] - span["start_time"]) * 1000, 1)

        if metadata_update:
            span["metadata"].update(metadata_update)

        # Problem #5: classify what kind of execution this was
        span["execution_pattern"] = self._classify_pattern(span)

        return span

    def update_span(self, metadata_update: Dict) -> None:
        """Update metadata on the currently active span without closing it."""
        span = _active_span.get()
        if span and metadata_update:
            span["metadata"].update(metadata_update)

    def record_retry(self, attempt: int, reason: str, wait_ms: int = 0) -> None:
        """Annotate the active span with retry metadata."""
        span = _active_span.get()
        if span:
            span["retry_count"] = attempt
            span["metadata"]["retry_attempt"] = attempt
            span["metadata"]["retry_reason"]  = reason
            span["metadata"]["retry_wait_ms"] = wait_ms

    # ══════════════════════════════════════════════════════════════════════════
    # PROBLEM #1: DECORATOR + CONTEXT MANAGER  (no more manual start/end)
    # ══════════════════════════════════════════════════════════════════════════

    @contextlib.contextmanager
    def span(
        self,
        span_type:  str,
        name:       str,
        metadata:   Optional[Dict] = None,
        caused_by:  Optional[str]  = None,
        trigger:    Optional[str]  = None,
        relation:   Optional[str]  = None,
    ):
        """
        Context manager — automatically closes the span (success or failure).

        Usage:
            with trace_manager.span("tool_call", "web_search", {"query": q}) as s:
                result = do_search(q)
                trace_manager.update_span({"results": len(result)})
        """
        s = self.start_span(span_type, name, metadata or {},
                            caused_by=caused_by, trigger=trigger, relation=relation)
        try:
            yield s
        except Exception as e:
            self.end_span("failed", {
                "error":          str(e),
                "exception_type": type(e).__name__,
            })
            raise
        else:
            self.end_span("success")

    def observe(
        self,
        span_type:   str  = "tool_call",
        name:        Optional[str]  = None,
        metadata_fn  = None,   # callable(*args, **kwargs) → dict
        caused_by:   Optional[str]  = None,
        trigger:     Optional[str]  = None,
        relation:    Optional[str]  = None,
    ):
        """
        Decorator — zero-boilerplate instrumentation.
        Works on both sync and async functions.

        Usage:
            @trace_manager.observe("tool_call", name="validate_sources")
            def validate(result):
                ...

            @trace_manager.observe("tool_call", name="async_search",
                                   metadata_fn=lambda q: {"query": q})
            async def async_search(query):
                ...
        """
        def decorator(func):
            _name = name or func.__name__

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                _meta = metadata_fn(*args, **kwargs) if metadata_fn else {}
                self.start_span(span_type, _name, _meta,
                                caused_by=caused_by, trigger=trigger, relation=relation)
                try:
                    result = func(*args, **kwargs)
                    self.end_span("success")
                    return result
                except Exception as e:
                    self.end_span("failed", {
                        "error":          str(e),
                        "exception_type": type(e).__name__,
                    })
                    raise

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                _meta = metadata_fn(*args, **kwargs) if metadata_fn else {}
                self.start_span(span_type, _name, _meta,
                                caused_by=caused_by, trigger=trigger, relation=relation)
                try:
                    result = await func(*args, **kwargs)
                    self.end_span("success")
                    return result
                except Exception as e:
                    self.end_span("failed", {
                        "error":          str(e),
                        "exception_type": type(e).__name__,
                    })
                    raise

            import asyncio
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    # ══════════════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _classify_pattern(self, span: Dict) -> str:
        """
        Problem #5: Replace raw event labels with execution semantics.

        Instead of storing 'event: retry', we store what it MEANS operationally.
        This enables failure pattern analytics across many traces over time.
        """
        meta         = span.get("metadata", {})
        status       = span.get("status", "")
        retry_count  = span.get("retry_count", 0)
        relation     = span.get("relation", "")
        trigger      = span.get("trigger", "") or ""
        err          = str(meta.get("error", "")).lower()
        retry_reason = str(meta.get("retry_reason", "")).lower()

        if "429" in retry_reason or "rate_limit" in retry_reason:
            return "rate_limit_recovery"

        if relation == "retry" and status == "success":
            return "recovered_via_retry"

        if relation == "retry" and status == "failed":
            return "exhausted_retry"

        if relation == "fallback":
            return "fallback_chain"

        if "timeout" in err or "timeout" in trigger:
            return "timeout_failure"

        if status == "failed" and span.get("type") == "phase":
            return "degraded_phase"

        if retry_count > 0 and status == "success":
            return "recovered_via_retry"

        if retry_count > 0 and status == "failed":
            return "exhausted_retry"

        if status == "failed":
            return "tool_failure"

        if status == "success":
            return "normal_execution"

        return "unknown"

    def _get_descendants(self, spans, span_id: str):
        """Return all spans that are children/grandchildren of span_id."""
        result = []
        queue  = [span_id]
        while queue:
            current_id = queue.pop()
            children   = [s for s in spans if s.get("parent_span_id") == current_id]
            result.extend(children)
            queue.extend(c["span_id"] for c in children)
        return result

    def current_span_id(self) -> Optional[str]:
        """Return the ID of the currently active span (useful for caused_by links)."""
        s = _active_span.get()
        return s["span_id"] if s else None


# ── Module-level singleton ────────────────────────────────────────────────────
trace_manager = TraceManager()
