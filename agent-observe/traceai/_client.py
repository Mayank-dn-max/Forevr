"""
traceai/_client.py — HTTP trace sender

Sends completed traces to the Trace AI backend (local or cloud).
Always uses HTTP — never imports backend modules directly.

Local dev (no API key):  POST http://127.0.0.1:8000/ingest
Cloud (API key set):     POST https://api.traceai.dev/ingest

Runs in a background daemon thread so it never blocks the agent.
Fails silently — observability must never crash the user's code.
"""

import threading


def send_trace(trace: dict) -> None:
    """POST a completed trace to the backend. Non-blocking, silent on failure."""
    from traceai.config import get
    cfg      = get()
    base_url = cfg.get("base_url", "http://127.0.0.1:8000")
    api_key  = cfg.get("api_key")
    project  = cfg.get("project", "default")

    def _post():
        try:
            import requests
            headers = {
                "Content-Type":  "application/json",
                "X-TraceAI-SDK": "python/0.1.0",
            }
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            resp = requests.post(
                f"{base_url}/ingest",
                json={"trace": trace, "project": project},
                headers=headers,
                timeout=8,
            )
            if resp.status_code not in (200, 201):
                print(f"[traceai] Ingest warning: HTTP {resp.status_code} from {base_url}")
        except Exception as e:
            print(f"[traceai] Could not deliver trace to {base_url}: {e}")
            print(f"[traceai] Is the backend running? Start it with: uvicorn backend.main:app --reload")

    threading.Thread(target=_post, daemon=True).start()
