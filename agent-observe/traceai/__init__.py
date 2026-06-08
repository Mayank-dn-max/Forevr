"""
traceai — Trace AI SDK
======================
Universal observability for AI agents.

Quick start:

    import traceai
    traceai.init(
        api_key = "tai-xxxxxxxxxxxxxxxxxxxx",   # from dashboard.traceai.dev
        project = "my-research-agent",
    )

    # LangChain / LangGraph
    from traceai.langchain import Tracer
    agent = AgentExecutor(..., callbacks=[Tracer()])

    # CrewAI
    from traceai.crewai import trace_crew
    result = trace_crew(crew, inputs={...})

    # OpenAI Agents SDK
    from traceai.openai_agents import Processor
    from agents import add_trace_processor
    add_trace_processor(Processor())

    # Any custom agent
    from traceai.universal import Trace, span, watch
    with Trace("my_agent"):
        with span("tool_call", "search"):
            result = search(query)

    @watch("tool_call")
    def my_tool(query): ...

Environment variable alternative (no code change needed):
    export TRACEAI_API_KEY=tai-xxxxxxxxxxxxxxxxxxxx
    export TRACEAI_PROJECT=my-agent
"""

__version__ = "0.1.0"
__author__  = "Trace AI"

# Expose init() at the top level so users can call traceai.init(...)
from traceai.config import init, get as get_config

__all__ = ["init", "get_config", "__version__"]
