import sys
import os
import csv
import io
import uuid
import re
import json as _json

# Add parent dir so `agent/` package is importable when running locally.
# In production the agent folder is not present — all agent imports are
# guarded with try/except so this never crashes a deployed backend.
_parent = os.path.join(os.path.dirname(__file__), '..')
if _parent not in sys.path:
    sys.path.insert(0, _parent)

# Load .env if available (needed for GROQ_API_KEY / GOOGLE_API_KEY in judge endpoint)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from database import (
    init_db, load_all_traces, clear_all_traces, get_trace_count, save_trace,
    save_score, get_all_scores,
    save_dataset, get_all_datasets, delete_dataset as db_delete_dataset,
    add_to_dataset, remove_from_dataset, get_dataset_members, get_all_memberships,
    save_judgment, get_all_judgments,
    get_all_sessions, get_session_traces, delete_session as db_delete_session,
    create_api_key, validate_api_key, has_any_api_keys, list_api_keys, revoke_api_key,
)

# Admin secret: set ADMIN_TOKEN env var in Railway to protect key mgmt and db/clear.
# If not set, those endpoints are open (local dev mode).
_ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")


def _require_admin(request: Request) -> None:
    """Raise 401/403 if ADMIN_TOKEN is set and the request doesn't supply it."""
    if not _ADMIN_TOKEN:
        return  # no token configured → open (local dev)
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Admin token required")
    if auth[7:].strip() != _ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid admin token")
from storage import get_traces
from analyzer import analyzer
import tracer   # need the module so we can extend tracer.traces


# ── Startup / Shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the server starts.

    1. init_db()          → create traces.db + table if they don't exist
    2. load_all_traces()  → read every saved trace from SQLite
    3. tracer.traces.extend(...)  → populate the in-memory list so the
                                    dashboard instantly shows past runs
    """
    init_db()
    existing = load_all_traces()
    tracer.traces.extend(existing)
    print(f"[startup] Restored {len(existing)} trace(s) from SQLite")
    yield
    # Nothing needed on shutdown — writes happen immediately per trace


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Import agent lazily — only available in local dev, not production ──────────
def _get_run_agent():
    try:
        from agent.agent import run_agent
        return run_agent
    except ImportError:
        raise HTTPException(status_code=501, detail="Test agent not available in this deployment")


class RunRequest(BaseModel):
    query:       str = "What is LangChain?"
    session_id:  str = None
    user_id:     str = None
    turn_number: int = None
    agent_type:  str = "default"   # "default" | "langgraph"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/traces")
def traces():
    """Return all traces (in-memory list, pre-loaded from SQLite on startup)."""
    return get_traces()


@app.post("/run")
def run_agent_endpoint(body: RunRequest):
    """Trigger an agent run. Trace is saved to SQLite automatically."""
    kwargs = dict(session_id=body.session_id, user_id=body.user_id, turn_number=body.turn_number)
    if body.agent_type == "langgraph":
        try:
            from agent.langgraph_agent import run_langgraph_agent
        except ImportError:
            raise HTTPException(status_code=501, detail="LangGraph agent not available in this deployment")
        result = run_langgraph_agent(body.query, **kwargs)
    else:
        result = _get_run_agent()(body.query, **kwargs)
    return {"result": result}


@app.get("/insights")
def insights():
    """
    Run all 6 pattern detectors against the current trace history.
    Returns ranked insights: latency spikes, error clusters, cost anomalies,
    output drift, retry storms, failure rate spikes.
    """
    return analyzer.analyze(get_traces())


@app.get("/health")
def health():
    """Health check for Railway / load balancers."""
    return {"status": "ok"}


@app.get("/db/stats")
def db_stats(request: Request):
    """Return SQLite database stats — useful for debugging."""
    _require_admin(request)
    return {
        "traces_on_disk":   get_trace_count(),
        "traces_in_memory": len(get_traces()),
    }


@app.delete("/db/clear")
def db_clear(request: Request):
    """Delete ALL traces from SQLite and memory. Use before a fresh demo."""
    _require_admin(request)
    deleted = clear_all_traces()
    tracer.traces.clear()
    return {"deleted": deleted, "message": f"Cleared {deleted} trace(s)"}


# ── API Key management ────────────────────────────────────────────────────────

class KeyCreateRequest(BaseModel):
    name: str = "default"

@app.post("/keys")
def create_key(body: KeyCreateRequest, request: Request):
    """
    Generate a new API key. Returns the full key ONCE — store it immediately.
    Subsequent calls to GET /keys only show the prefix (tai-xxxxxxxx...).
    Requires ADMIN_TOKEN if set.
    """
    _require_admin(request)
    result = create_api_key(body.name)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to create key")
    return result

@app.get("/keys")
def get_keys(request: Request):
    """List all API keys (masked — prefix only, never the raw key). Requires ADMIN_TOKEN if set."""
    _require_admin(request)
    return list_api_keys()

@app.delete("/keys/{key_id}")
def delete_key(key_id: str, request: Request):
    """Revoke an API key by its key_id. Requires ADMIN_TOKEN if set."""
    _require_admin(request)
    ok = revoke_api_key(key_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Key not found")
    return {"status": "revoked", "key_id": key_id}


# ── Remote SDK ingest ─────────────────────────────────────────────────────────
class IngestRequest(BaseModel):
    trace:   dict
    project: str = "default"

@app.post("/ingest")
def ingest_trace(body: IngestRequest, request: Request = None):
    """
    Receive a trace from a remote traceai SDK user.

    Auth behaviour:
      - No keys in DB  → accept all (local dev / self-hosted mode)
      - Keys exist     → require Authorization: Bearer tai-xxxx
    """
    # Enforce auth only when at least one key has been created
    if has_any_api_keys():
        auth = (request.headers.get("Authorization") or "") if request else ""
        if not auth.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing API key — add Authorization: Bearer <key>")
        raw_key = auth[7:].strip()
        if not validate_api_key(raw_key):
            raise HTTPException(status_code=401, detail="Invalid or revoked API key")

    trace = body.trace
    if not trace or not trace.get("trace_id"):
        raise HTTPException(status_code=400, detail="trace_id is required")

    trace["project"] = body.project
    save_trace(trace)
    tracer.traces.append(trace)

    return {"status": "ok", "trace_id": trace["trace_id"]}


# ── CSV Export ────────────────────────────────────────────────────────────────

@app.get("/export/csv")
def export_csv():
    """Download all traces as a flat CSV file."""
    data = get_traces()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([
        "trace_id", "name", "status", "start_time", "latency_s",
        "spans_total", "spans_success", "spans_failed",
        "total_input_tokens", "total_output_tokens", "total_cost_usd",
        "execution_patterns",
    ])
    for t in data:
        w.writerow([
            t.get("trace_id", ""),
            t.get("name", ""),
            t.get("status", ""),
            t.get("start_time", ""),
            t.get("latency", ""),
            t.get("spans_total", ""),
            t.get("spans_success", ""),
            t.get("spans_failed", ""),
            t.get("total_input_tokens", ""),
            t.get("total_output_tokens", ""),
            t.get("total_cost_usd", ""),
            "|".join(t.get("execution_patterns", [])),
        ])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="traces.csv"'},
    )


# ── Manual Scoring ────────────────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    score: int   # 1 = thumbs up, -1 = thumbs down, 0 = remove
    label: str = ""
    comment: str = ""

@app.post("/score/{trace_id}")
def post_score(trace_id: str, body: ScoreRequest):
    save_score(trace_id, body.score, body.label, body.comment)
    return {"ok": True}

@app.get("/scores")
def get_scores():
    return get_all_scores()


# ── Dataset Builder ───────────────────────────────────────────────────────────

class DatasetCreate(BaseModel):
    name: str
    description: str = ""

class DatasetAddTrace(BaseModel):
    trace_id: str
    split: str = "train"

@app.get("/datasets")
def list_datasets():
    return get_all_datasets()

@app.get("/datasets/memberships")
def dataset_memberships():
    """Returns {trace_id: [dataset_id, ...]} for all memberships."""
    return get_all_memberships()

@app.post("/datasets")
def create_dataset(body: DatasetCreate):
    did = str(uuid.uuid4())[:8]
    save_dataset(did, body.name, body.description)
    return {"dataset_id": did, "name": body.name, "description": body.description, "trace_count": 0}

@app.delete("/datasets/{dataset_id}")
def delete_dataset_route(dataset_id: str):
    db_delete_dataset(dataset_id)
    return {"ok": True}

@app.post("/datasets/{dataset_id}/traces")
def add_trace_to_dataset(dataset_id: str, body: DatasetAddTrace):
    add_to_dataset(dataset_id, body.trace_id, body.split)
    return {"ok": True}

@app.delete("/datasets/{dataset_id}/traces/{trace_id}")
def remove_trace_from_dataset(dataset_id: str, trace_id: str):
    remove_from_dataset(dataset_id, trace_id)
    return {"ok": True}

@app.get("/datasets/{dataset_id}/export")
def export_dataset(dataset_id: str):
    """Export a dataset as JSON enriched with input/output/context per trace."""
    datasets = {d["dataset_id"]: d for d in get_all_datasets()}
    ds = datasets.get(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    all_scores = get_all_scores()
    all_judg   = get_all_judgments()
    members    = get_dataset_members(dataset_id)

    rows = []
    for t in members:
        spans     = t.get("spans", [])
        resp_span = next((s for s in spans if s.get("type") == "agent_response"), None)
        output    = resp_span.get("metadata", {}).get("response_preview", "") if resp_span else ""
        context   = []
        for s in spans:
            if s.get("type") == "tool_call" and s.get("status") == "success":
                meta = s.get("metadata", {})
                for k in ("result", "output", "content", "summary", "output_preview"):
                    if meta.get(k):
                        context.append(str(meta[k])[:500])
                        break
        tid = t.get("trace_id", "")
        rows.append({
            "trace_id":    tid,
            "split":       t.get("_split", "train"),
            "name":        t.get("name", ""),
            "status":      t.get("status", ""),
            "latency":     t.get("latency"),
            "input":       t.get("name", ""),
            "output":      output,
            "context":     context[:5],
            "human_score": all_scores.get(tid, {}).get("score"),
            "judgment":    all_judg.get(tid),
        })

    return {"dataset": ds, "traces": rows}


# ── LLM-as-Judge (groundedness) ───────────────────────────────────────────────

@app.post("/judge/{trace_id}")
def judge_trace(trace_id: str):
    """
    Run a Groq groundedness judge on a trace's agent response.

    Tries models in order. Falls back to Gemini if Groq is rate-limited.
    Results are cached in SQLite — clicking Judge again returns instantly.
    """
    traces_list = get_traces()
    trace = next((t for t in traces_list if t["trace_id"] == trace_id), None)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")

    # ── Return cached judgment (saves quota) ─────────────────────────────────
    existing = get_all_judgments()
    if trace_id in existing:
        j = existing[trace_id]
        return {
            "trace_id":  trace_id,
            "score":     j.get("score"),
            "verdict":   j.get("verdict"),
            "reasoning": j.get("reasoning"),
            "model":     j.get("model", "cached"),
            "cached":    True,
        }

    # ── Extract response text ─────────────────────────────────────────────────
    spans     = trace.get("spans", [])
    resp_span = next((s for s in spans if s.get("type") == "agent_response"), None)
    response_text = (
        resp_span.get("metadata", {}).get("response_preview", "") if resp_span else ""
    )

    if not response_text:
        return {"error": "No agent_response span with response_preview found in this trace"}

    # ── Extract retrieved context from tool calls ─────────────────────────────
    context_parts = []
    for s in spans:
        if s.get("type") == "tool_call" and s.get("status") == "success":
            meta = s.get("metadata", {})
            for k in ("result", "output", "content", "summary", "output_preview"):
                if meta.get(k):
                    context_parts.append(str(meta[k])[:600])
                    break
    context_text = "\n\n".join(context_parts[:5]) if context_parts else ""

    groq_key   = os.getenv("GROQ_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    if not groq_key and not google_key:
        return {"error": "No API key found — set GROQ_API_KEY or GOOGLE_API_KEY"}

    prompt = f"""You are a groundedness evaluator for AI agent responses.

RETRIEVED CONTEXT (sources the agent used):
{context_text if context_text else "No context was retrieved by the agent."}

AGENT RESPONSE:
{response_text[:1500]}

Score how well the agent response is grounded in the retrieved context.
- 1.0 = fully grounded: every claim is supported by the context
- 0.5-0.9 = partially grounded: most claims supported, some unsupported
- 0.0-0.4 = ungrounded: claims not supported by context, or no context used

Respond ONLY with valid JSON (no markdown, no explanation):
{{"score": 0.85, "verdict": "grounded", "reasoning": "one sentence explanation"}}

verdict must be exactly one of: grounded, partial, ungrounded"""

    def _parse_judge_text(text):
        m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if m:
            parsed  = _json.loads(m.group())
            score   = float(max(0.0, min(1.0, parsed.get("score", 0.5))))
            verdict = parsed.get("verdict", "partial")
            if verdict not in ("grounded", "partial", "ungrounded"):
                verdict = "partial"
            reasoning = str(parsed.get("reasoning", ""))[:300]
            return score, verdict, reasoning
        return 0.5, "partial", text[:200]

    def _is_rate_limit(err_str):
        return any(x in err_str for x in ("429", "rate_limit", "quota", "RateLimitError", "rate limit"))

    last_error = None

    # ── 1. Groq (primary — fast, free) ───────────────────────────────────────
    if groq_key:
        from groq import Groq
        groq_client = Groq(api_key=groq_key)
        for model_name in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
            try:
                completion = groq_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=200,
                )
                text  = completion.choices[0].message.content.strip()
                score, verdict, reasoning = _parse_judge_text(text)
                save_judgment(trace_id, score, verdict, reasoning, model_name)
                return {
                    "trace_id":  trace_id,
                    "score":     score,
                    "verdict":   verdict,
                    "reasoning": reasoning,
                    "model":     model_name,
                    "cached":    False,
                }
            except Exception as e:
                err_str    = str(e)
                last_error = err_str
                if _is_rate_limit(err_str):
                    continue
                return {"error": err_str}

    # ── 2. Gemini fallback ────────────────────────────────────────────────────
    if google_key:
        import google.generativeai as genai
        genai.configure(api_key=google_key)
        for model_name in ["gemini-2.0-flash-lite", "gemini-2.0-flash"]:
            try:
                model  = genai.GenerativeModel(model_name)
                result = model.generate_content(prompt)
                text   = result.text.strip()
                score, verdict, reasoning = _parse_judge_text(text)
                save_judgment(trace_id, score, verdict, reasoning, model_name)
                return {
                    "trace_id":  trace_id,
                    "score":     score,
                    "verdict":   verdict,
                    "reasoning": reasoning,
                    "model":     model_name,
                    "cached":    False,
                }
            except Exception as e:
                err_str    = str(e)
                last_error = err_str
                if _is_rate_limit(err_str):
                    continue
                return {"error": err_str}

    return {
        "error":       "rate_limit",
        "message":     f"All judge models are rate-limited. Try again in a minute.",
        "last_error":  last_error,
    }


@app.get("/judgments")
def get_judgments():
    return get_all_judgments()


# ── Sessions ──────────────────────────────────────────────────────────────────

@app.get("/sessions")
def list_sessions():
    """Return all sessions with their aggregated metadata."""
    return get_all_sessions()


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    """Return a single session with all its traces (turns) in order."""
    sessions = {s["session_id"]: s for s in get_all_sessions()}
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    traces = get_session_traces(session_id)
    return {"session": session, "traces": traces}


@app.delete("/sessions/{session_id}")
def delete_session_route(session_id: str):
    """Delete a session record. Traces are kept but unlinked from the session."""
    db_delete_session(session_id)
    return {"ok": True}
