# Forevr
forevr
Observability SDK for AI agents. Auto-traces LangChain, CrewAI, OpenAI Agents, and any custom agent — with a full dashboard for latency, cost, errors, and LLM-as-judge scoring.

Install
pip install forevr
Quick start
import forevr

forevr.init(
    api_key  = "tai-xxxxxxxxxxxxxxxxxxxx",
    project  = "my-agent",
)
Integrations
LangChain / LangGraph

from forevr.langchain import Tracer

agent = AgentExecutor(agent=..., tools=..., callbacks=[Tracer()])
CrewAI

from forevr.crewai import trace_crew

result = trace_crew(crew, inputs={"topic": "AI trends"})
OpenAI Agents SDK

from forevr.openai_agents import Processor
from agents import add_trace_processor

add_trace_processor(Processor())
Any custom agent

from forevr.universal import Trace, span

with Trace("my_agent"):
    with span("tool_call", "web_search"):
        result = search(query)
Local dev (no API key needed)
Run the backend locally and traces appear in the dashboard automatically:

# No init() needed — defaults to http://127.0.0.1:8000
from forevr.langchain import Tracer
Environment variables
export FOREVR_API_KEY=tai-xxxxxxxxxxxxxxxxxxxx
export FOREVR_PROJECT=my-agent
