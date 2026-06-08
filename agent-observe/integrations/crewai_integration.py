"""
integrations/crewai_integration.py
====================================
Trace AI adapter for CrewAI.

How it works:
    CrewAI exposes three hook points:
      1. step_callback   — fires after each agent step (tool use / LLM call)
      2. task_callback   — fires when a task completes
      3. Wrapping kickoff() — captures the full crew run lifecycle

    This adapter uses all three plus wraps the LLM via the LangChain
    callback handler (since CrewAI agents use LangChain LLMs internally).

Usage:
    from integrations.crewai_integration import instrument_crew

    crew = Crew(agents=[...], tasks=[...])
    result = instrument_crew(crew, inputs={"topic": "AI trends"})

    # That's it. Full trace appears in Trace AI dashboard.

    # Optional — also instrument the LLM directly for token counts:
    from integrations.langchain_integration import TraceAICallbackHandler
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o", callbacks=[TraceAICallbackHandler()])
    agent = Agent(role="researcher", llm=llm, ...)
"""

import functools

try:
    from crewai import Crew
except ImportError:
    raise ImportError(
        "crewai is required for the CrewAI integration.\n"
        "Install it with: pip install crewai"
    )

from traceai._tracer import trace_manager


def _make_step_callback():
    """Returns a step callback that records each agent action as a span."""
    def step_callback(agent_output):
        """Called after every agent step (thought + action + observation)."""
        try:
            # AgentOutput has .tool, .tool_input, .text (final answer)
            tool      = getattr(agent_output, "tool",       None)
            tool_in   = getattr(agent_output, "tool_input", None)
            log       = getattr(agent_output, "log",        "") or ""

            if tool:
                # This step used a tool
                span_name = tool or "agent_step"
                trace_manager.start_span("tool_call", span_name, {
                    "tool":        tool,
                    "tool_input":  str(tool_in)[:300] if tool_in else "",
                    "log_preview": log[:200],
                })
                trace_manager.end_span("success")
            else:
                # Final answer step
                trace_manager.start_span("agent_response", "final_response", {
                    "response_preview": log[:400],
                })
                trace_manager.end_span("success")
        except Exception:
            pass   # never crash the agent because of observability

    return step_callback


def _make_task_callback():
    """Returns a task callback that records task completion."""
    def task_callback(task_output):
        """Called when a CrewAI task finishes."""
        try:
            raw    = getattr(task_output, "raw",         "") or ""
            desc   = getattr(task_output, "description", "") or ""
            name   = desc[:60] or "task"

            trace_manager.start_span("phase", f"task: {name}", {
                "output_preview": raw[:300],
            })
            trace_manager.end_span("success")
        except Exception:
            pass

    return task_callback


def instrument_crew(crew: "Crew", inputs: dict = None, crew_name: str = "crewai_agent"):
    """
    Run a CrewAI crew with full Trace AI observability.

    Replaces:
        result = crew.kickoff(inputs=inputs)

    With:
        result = instrument_crew(crew, inputs=inputs)

    Args:
        crew:       Your CrewAI Crew instance
        inputs:     The inputs dict passed to crew.kickoff()
        crew_name:  Label shown in the Trace AI dashboard

    Returns:
        The same result as crew.kickoff()
    """
    # Inject callbacks
    original_step = getattr(crew, "step_callback", None)
    original_task = getattr(crew, "task_callback", None)

    crew.step_callback = _make_step_callback()
    crew.task_callback = _make_task_callback()

    # Start trace
    trace_manager.start_trace(crew_name)
    trace_manager.start_span("agent_start", crew_name, {
        "framework":   "crewai",
        "num_agents":  len(crew.agents),
        "num_tasks":   len(crew.tasks),
        "inputs":      str(inputs)[:200] if inputs else "",
    })

    try:
        result = crew.kickoff(inputs=inputs or {})
        trace_manager.end_span("success", {
            "result_preview": str(result)[:300],
        })
        trace_manager.end_trace("success")
        return result

    except Exception as e:
        trace_manager.end_span("failed", {"error": str(e)})
        trace_manager.end_trace("failed")
        raise

    finally:
        # Restore original callbacks
        crew.step_callback = original_step
        crew.task_callback = original_task
