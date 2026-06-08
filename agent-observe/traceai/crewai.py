"""
traceai.crewai — CrewAI integration
=====================================

Usage:
    from traceai.crewai import trace_crew

    crew   = Crew(agents=[...], tasks=[...])
    result = trace_crew(crew, inputs={"topic": "AI trends"})
    # Replaces crew.kickoff() — identical return value, full tracing added.

    # Optional: also instrument the LLM for token counts
    from traceai.langchain import Tracer
    from langchain_openai import ChatOpenAI

    llm   = ChatOpenAI(model="gpt-4o", callbacks=[Tracer()])
    agent = Agent(role="researcher", llm=llm, ...)
"""

try:
    from crewai import Crew
except ImportError:
    raise ImportError(
        "crewai is required for the CrewAI integration.\n"
        "Install it with:  pip install crewai"
    )

from traceai._tracer import trace_manager


def _make_step_callback():
    def step_callback(agent_output):
        try:
            tool    = getattr(agent_output, "tool",       None)
            tool_in = getattr(agent_output, "tool_input", None)
            log     = getattr(agent_output, "log",        "") or ""
            if tool:
                trace_manager.start_span("tool_call", tool, {
                    "tool":        tool,
                    "tool_input":  str(tool_in)[:300] if tool_in else "",
                    "log_preview": log[:200],
                })
                trace_manager.end_span("success")
            else:
                trace_manager.start_span("agent_response", "final_response", {
                    "response_preview": log[:400],
                })
                trace_manager.end_span("success")
        except Exception:
            pass
    return step_callback


def _make_task_callback():
    def task_callback(task_output):
        try:
            raw  = getattr(task_output, "raw",         "") or ""
            desc = getattr(task_output, "description", "") or ""
            name = desc[:60] or "task"
            trace_manager.start_span("phase", f"task: {name}", {
                "output_preview": raw[:300],
            })
            trace_manager.end_span("success")
        except Exception:
            pass
    return task_callback


def trace_crew(crew: "Crew", inputs: dict = None, crew_name: str = "crewai_agent"):
    """
    Run a CrewAI crew with full Trace AI observability.

    Drop-in replacement for crew.kickoff():
        # Before
        result = crew.kickoff(inputs=inputs)

        # After  ← identical result, full tracing added
        from traceai.crewai import trace_crew
        result = trace_crew(crew, inputs=inputs)
    """
    orig_step = getattr(crew, "step_callback", None)
    orig_task = getattr(crew, "task_callback", None)

    crew.step_callback = _make_step_callback()
    crew.task_callback = _make_task_callback()

    trace_manager.start_trace(crew_name)
    trace_manager.start_span("agent_start", crew_name, {
        "framework":  "crewai",
        "num_agents": len(crew.agents),
        "num_tasks":  len(crew.tasks),
        "inputs":     str(inputs)[:200] if inputs else "",
    })

    try:
        result = crew.kickoff(inputs=inputs or {})
        trace_manager.end_span("success", {"result_preview": str(result)[:300]})
        trace_manager.end_trace("success")
        return result
    except Exception as e:
        trace_manager.end_span("failed", {"error": str(e)})
        trace_manager.end_trace("failed")
        raise
    finally:
        crew.step_callback = orig_step
        crew.task_callback = orig_task


# backwards-compat alias
instrument_crew = trace_crew
