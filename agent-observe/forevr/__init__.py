"""
forevr — Agent Observability SDK
==================================
Universal observability for AI agents.

Quick start:

    import forevr
    forevr.init(
        api_key = "tai-xxxxxxxxxxxxxxxxxxxx",
        project = "my-research-agent",
    )

    # LangChain / LangGraph
    from forevr.langchain import Tracer
    agent = AgentExecutor(..., callbacks=[Tracer()])

    # CrewAI
    from forevr.crewai import trace_crew
    result = trace_crew(crew, inputs={...})

    # OpenAI Agents SDK
    from forevr.openai_agents import Processor
    from agents import add_trace_processor
    add_trace_processor(Processor())

    # Any custom agent
    from forevr.universal import Trace, span, watch
    with Trace("my_agent"):
        with span("tool_call", "search"):
            result = search(query)

    @watch("tool_call")
    def my_tool(query): ...

Environment variable alternative (no code change needed):
    export FOREVR_API_KEY=tai-xxxxxxxxxxxxxxxxxxxx
    export FOREVR_PROJECT=my-agent
"""

__version__ = "0.1.3"
__author__  = "Forevr"

from forevr.config import init, get as get_config

__all__ = ["init", "get_config", "__version__"]
