"""
forevr.langchain — LangChain / LangGraph integration

Usage:
    from forevr.langchain import Tracer

    # Works with AgentExecutor:
    agent = AgentExecutor(agent=..., tools=..., callbacks=[Tracer()])

    # Works with LCEL chains:
    chain = prompt | llm | parser
    chain.invoke({"input": "..."}, config={"callbacks": [Tracer()]})

    # Works directly on the LLM:
    llm = ChatOpenAI(model="gpt-4o", callbacks=[Tracer()])
"""

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
except ImportError:
    raise ImportError(
        "langchain-core is required for the LangChain integration.\n"
        "Install it with:  pip install langchain-core"
    )

from forevr._tracer import trace_manager


class Tracer(BaseCallbackHandler):
    """
    Drop-in LangChain callback handler.
    Add to any chain, agent, or LLM — tracing is automatic.
    """

    def __init__(self, agent_name: str = "langchain_agent"):
        super().__init__()
        self.agent_name   = agent_name
        self._run_to_span = {}
        self._trace_open  = False

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        if not self._trace_open:
            trace_manager.start_trace(self.agent_name)
            self._trace_open = True
            span = trace_manager.start_span("agent_start", self.agent_name, {
                "framework":     "langchain",
                "input_preview": str(inputs)[:300],
            })
        else:
            name = (
                serialized.get("name")
                or (serialized.get("id") or ["chain"])[-1]
                or "chain"
            )
            span = trace_manager.start_span("phase", name, {
                "input_preview": str(inputs)[:200],
            })
        self._run_to_span[str(run_id)] = span["span_id"]

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        trace_manager.end_span("success", {"output_preview": str(outputs)[:200]})
        self._run_to_span.pop(str(run_id), None)
        if not self._run_to_span:
            self._trace_open = False
            trace_manager.end_trace("success")

    def on_chain_error(self, error, *, run_id, **kwargs):
        trace_manager.end_span("failed", {"error": str(error)})
        self._run_to_span.pop(str(run_id), None)
        if not self._run_to_span:
            self._trace_open = False
            trace_manager.end_trace("failed")

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        ids        = serialized.get("id") or ["unknown"]
        model_name = ids[-1] if ids else "unknown"
        span = trace_manager.start_span("llm_call", model_name, {
            "model":        model_name,
            "prompt_count": len(prompts),
            "prompt_chars": sum(len(p) for p in prompts),
            "framework":    "langchain",
        })
        self._run_to_span[str(run_id)] = span["span_id"]

    def on_llm_end(self, response, *, run_id, **kwargs):
        llm_out = getattr(response, "llm_output", None) or {}
        usage   = llm_out.get("token_usage", {})
        in_tok  = usage.get("prompt_tokens",    0)
        out_tok = usage.get("completion_tokens", 0)
        trace_manager.end_span("success", {
            "input_tokens":  in_tok,
            "output_tokens": out_tok,
            "total_tokens":  in_tok + out_tok,
        })
        self._run_to_span.pop(str(run_id), None)

    def on_llm_error(self, error, *, run_id, **kwargs):
        trace_manager.end_span("failed", {"error": str(error)})
        self._run_to_span.pop(str(run_id), None)

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        name = serialized.get("name", "tool")
        span = trace_manager.start_span("tool_call", name, {
            "input": str(input_str)[:300],
        })
        self._run_to_span[str(run_id)] = span["span_id"]

    def on_tool_end(self, output, *, run_id, **kwargs):
        trace_manager.end_span("success", {"output_preview": str(output)[:300]})
        self._run_to_span.pop(str(run_id), None)

    def on_tool_error(self, error, *, run_id, **kwargs):
        trace_manager.end_span("failed", {"error": str(error)})
        self._run_to_span.pop(str(run_id), None)

    def on_agent_finish(self, finish, *, run_id, **kwargs):
        trace_manager.start_span("agent_response", "final_response", {
            "response_preview": str(finish.return_values)[:400],
        })
        trace_manager.end_span("success")
