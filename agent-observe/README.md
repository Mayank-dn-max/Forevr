# forevr

Observability SDK for AI agents. Auto-traces LangChain, CrewAI, OpenAI Agents, and any custom agent — with a full dashboard for latency, cost, errors, and LLM-as-judge scoring.

## Install

```bash
pip install forevr
```

## Quick start

```python
import forevr

forevr.init(
    api_key  = "tai-xxxxxxxxxxxxxxxxxxxx",
    project  = "my-agent",
)
```

## Integrations

**LangChain / LangGraph**
```python
from forevr.langchain import Tracer

agent = AgentExecutor(agent=..., tools=..., callbacks=[Tracer()])
```

**CrewAI**
```python
from forevr.crewai import trace_crew

result = trace_crew(crew, inputs={"topic": "AI trends"})
```

**OpenAI Agents SDK**
```python
from forevr.openai_agents import Processor
from agents import add_trace_processor

add_trace_processor(Processor())
```

**Any custom agent**
```python
from forevr.universal import Trace, span

with Trace("my_agent"):
    with span("tool_call", "web_search"):
        result = search(query)
```

## Local dev (no API key needed)

Run the backend locally and traces appear in the dashboard automatically:

```python
# No init() needed — defaults to http://127.0.0.1:8000
from forevr.langchain import Tracer
```

## Environment variables

```bash
export FOREVR_API_KEY=tai-xxxxxxxxxxxxxxxxxxxx
export FOREVR_PROJECT=my-agent
```
