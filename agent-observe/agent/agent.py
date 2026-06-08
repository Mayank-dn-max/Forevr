import os
import sys
import time
import random
import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from langchain_google_genai import ChatGoogleGenerativeAI
from tracer import trace_manager

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

LANGSEARCH_KEY = os.getenv("LANGSEARCH_API_KEY", "")

# ── Gemini pricing (USD per 1M tokens) ───────────────────────────────
COST_PER_1M = {
    "gemini-2.5-flash": {"input": 0.15,  "output": 0.60},
    "gemini-2.5-pro":   {"input": 1.25,  "output": 10.00},
    "gemini-2.0-flash": {"input": 0.10,  "output": 0.40},
}

MODELS = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]

def get_llm(model):
    return ChatGoogleGenerativeAI(model=model, temperature=0)

# ─────────────────────────────────────────────────────────────────────
#  TOOL FAILURE PROBABILITIES  (tweak these to change how often things fail)
#
#  Each tool rolls a random number. If it falls below the threshold → FAIL.
#  This makes every run genuinely different.
#
#  Tool                Fail %    Why it might fail in the real world
#  ─────────────────── ───────   ─────────────────────────────────────
#  web_search          15 %      Network timeout / DNS failure
#  validate_sources    40 %      Source domain not in allowlist
#  fact_check          45 %      Third-party fact-check API is flaky
#  summarise           10 %      Input text exceeds token limit
# ─────────────────────────────────────────────────────────────────────
FAIL_RATES = {
    # web_search is now REAL — no fake fail rate needed
    "validate_sources": 0.40,   # simulated: 40% chance source fails allowlist
    "fact_check":       0.45,   # simulated: 45% chance fact-check service is down
    "summarise":        0.10,   # simulated: 10% chance input too long
}

def coin(tool_name):
    """Returns True if this tool should FAIL on this run."""
    return random.random() < FAIL_RATES[tool_name]


# ── Real LLM call: token tracking + retry tracking ────────────────────
def invoke_llm_with_retry(user_query, summary):
    """
    Calls Gemini with automatic model fallback.
    Records: real input/output tokens, cost estimate, and retry metadata.
    This is a REAL API call — tokens and cost are genuine.
    """
    trace_manager.start_span("llm_call", "gemini_generation", {
        "query_length":   len(user_query),
        "context_length": len(summary),
    })

    for model in MODELS:
        llm = get_llm(model)
        print(f"\n[>>] Trying model: {model}")

        for attempt in range(1, 3):   # max 2 attempts per model (was 3)
            try:
                response = llm.invoke(
                    f"Answer the user query.\n\nQuery:\n{user_query}\n\nContext:\n{summary}"
                )

                # ── Real token counts from Gemini response ──────────
                usage   = getattr(response, "usage_metadata", None) or {}
                in_tok  = getattr(usage, "input_tokens",            0) or usage.get("input_tokens", 0)
                out_tok = getattr(usage, "output_tokens",           0) or usage.get("output_tokens", 0)
                cached  = getattr(usage, "cache_read_input_tokens", 0) or usage.get("cache_read_input_tokens", 0)

                pricing  = COST_PER_1M.get(model, {"input": 0.15, "output": 0.60})
                cost_usd = round(
                    (in_tok  / 1_000_000 * pricing["input"]) +
                    (out_tok / 1_000_000 * pricing["output"]),
                    6
                )

                print(f"[OK] {model} — in:{in_tok} out:{out_tok} cost:${cost_usd}")

                trace_manager.end_span("success", {
                    "model":        model,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "cached_tokens": cached,
                    "total_tokens": in_tok + out_tok,
                    "cost_usd":     cost_usd,
                    "pricing_note": f"${pricing['input']}/M in · ${pricing['output']}/M out",
                })
                return response, model

            except Exception as e:
                err = str(e)
                if "RESOURCE_EXHAUSTED" in err or "429" in err:
                    # Short wait — don't burn minutes on free-tier quota
                    wait = 2
                    print(f"[!!] Rate limited on {model} (attempt {attempt}/2). Waiting {wait}s...")
                    trace_manager.record_retry(attempt, f"Rate limited (429) — quota exhausted on {model}", wait_ms=wait * 1000)
                    time.sleep(wait)
                    if attempt >= 1:
                        print(f"[X] Skipping {model} — quota exhausted")
                        break

                elif ("Server disconnected" in err
                      or "RemoteProtocol" in err
                      or "Connection reset" in err
                      or "ConnectionError" in err
                      or "EOF" in err):
                    # Transient network error from Gemini — worth one retry on next model
                    # These are NOT quota errors; Gemini's server dropped the connection.
                    # Cause: internal timeout on Google's side, malformed payload, or instability.
                    wait = 1
                    print(f"[!!] Gemini dropped connection on {model}: {err[:60]}. Trying next model...")
                    trace_manager.record_retry(
                        attempt,
                        f"Server disconnected (connection reset) — moving to next model",
                        wait_ms=wait * 1000
                    )
                    time.sleep(wait)
                    break  # try next model immediately

                elif "NOT_FOUND" in err or "INVALID_ARGUMENT" in err:
                    print(f"[X] Model {model} unavailable, trying next...")
                    break
                else:
                    trace_manager.end_span("failed", {"error": err, "model": model})
                    raise

    trace_manager.end_span("failed", {
        "error": "All models quota-exhausted. Wait a minute and try again, or get a paid API key.",
        "hint":  "Free tier: ~15 requests/minute per model. Try again shortly."
    })
    raise RuntimeError("All Gemini models are rate-limited. Please wait 1-2 minutes and try again.")


# ══════════════════════════════════════════════════════════════════════
#  SIMULATED TOOLS
#  These do NOT call real APIs — they simulate what a real tool would do.
#  Latency (time.sleep) mimics real-world response times.
#  Success/failure is now random based on FAIL_RATES above.
# ══════════════════════════════════════════════════════════════════════

def tool_web_search(query):
    """
    REAL — calls the LangSearch Web Search API (api.langsearch.com).
    Measures actual network latency. Returns real results from the web.
    Falls back gracefully if the API key is missing or the call fails.
    """
    trace_manager.start_span("tool_call", "web_search", {"query": query})
    t0 = time.time()

    try:
        response = requests.post(
            "https://api.langsearch.com/v1/web-search",
            headers={
                "Authorization": f"Bearer {LANGSEARCH_KEY}",
                "Content-Type":  "application/json",
            },
            json={"query": query, "summary": True, "count": 5},
            timeout=10,
        )
        latency_ms = round((time.time() - t0) * 1000, 1)

        if response.status_code != 200:
            trace_manager.end_span("failed", {
                "error":      f"LangSearch API error: HTTP {response.status_code}",
                "http_code":  response.status_code,
                "latency_ms": latency_ms,
            })
            return None

        data      = response.json()
        web_data  = data.get("data", data)   # LangSearch wraps under "data" key
        pages     = web_data.get("webPages", {}).get("value", [])
        total     = web_data.get("webPages", {}).get("totalEstimatedMatches", 0) or 0

        # Build a rich context string from real snippets
        snippets = []
        urls     = []
        for p in pages[:5]:
            title   = p.get("name", "")
            snippet = p.get("summary") or p.get("snippet", "")
            url     = p.get("url", "")
            if snippet:
                snippets.append(f"[{title}]\n{snippet}")
            if url:
                urls.append(url)

        if snippets:
            context = "\n\n".join(snippets)
        elif pages:
            # Pages returned but no snippets — use titles + URLs as minimal context
            context = "\n".join(
                f"{p.get('name','')}: {p.get('url','')}" for p in pages
            )
        else:
            # Truly no results — use a descriptive fallback, NOT the query string
            context = f"No web search results found for: {query}. Answer from general knowledge."

        context_chars = len(context)
        trace_manager.end_span("success", {
            "results_found":    len(pages),
            "snippets_found":   len(snippets),
            "total_estimated":  total,
            "latency_ms":       latency_ms,
            "context_chars":    context_chars,
            "top_urls":         urls[:3],
            "source":           "LangSearch API (langsearch.com)",
        })
        print(f"[web_search] {len(pages)} results, {len(snippets)} snippets, {context_chars} chars in {latency_ms}ms")
        return context

    except requests.exceptions.Timeout:
        latency_ms = round((time.time() - t0) * 1000, 1)
        trace_manager.end_span("failed", {
            "error":      "LangSearch API timeout (>10s)",
            "latency_ms": latency_ms,
            "http_code":  504,
        })
        return None

    except Exception as e:
        latency_ms = round((time.time() - t0) * 1000, 1)
        trace_manager.end_span("failed", {
            "error":      str(e),
            "latency_ms": latency_ms,
        })
        return None


def tool_validate_sources(result):
    """
    SIMULATED — mimics a source allowlist / trust-score validator.
    In production, this would check domain reputation, robots.txt, etc.
    Fail rate: 40% (source not trusted)
    """
    latency = random.randint(100, 300)
    checked = random.randint(3, 8)
    trace_manager.start_span("tool_call", "validate_sources", {"sources_to_check": checked})
    time.sleep(latency / 1000)

    if coin("validate_sources"):
        trusted = random.randint(0, checked - 1)   # some trusted, but not all
        trace_manager.end_span("failed", {
            "error":    f"Only {trusted}/{checked} sources pass trust threshold",
            "trusted":  trusted,
            "checked":  checked,
            "allowlist": ["arxiv.org", "wikipedia.org", "github.com"],
        })
        return False
    else:
        trace_manager.end_span("success", {
            "trusted":  checked,
            "checked":  checked,
            "latency_ms": latency,
        })
        return True


def tool_fact_check(text):
    """
    SIMULATED — mimics a fact-checking microservice (e.g. ClaimBuster API).
    In production, this would POST the text to a fact-checking endpoint.
    Fail rate: 45% (service flaky / timeout)
    Retries: 0–2 times before giving up (also random).
    """
    latency = random.randint(200, 500)
    trace_manager.start_span("tool_call", "fact_check", {"text_length": len(text)})

    # Random number of retries (0, 1, or 2) to simulate flaky behaviour
    max_retries = random.randint(0, 2)

    if coin("fact_check"):
        # This run will fail — do some retries first to make it realistic
        for attempt in range(1, max_retries + 1):
            time.sleep(0.15)
            trace_manager.record_retry(
                attempt,
                random.choice([
                    "HTTP 503 Service Unavailable",
                    "Fact-check API timeout (300ms)",
                    "Connection reset by peer",
                ]),
                wait_ms=150
            )
        time.sleep(latency / 1000)
        trace_manager.end_span("failed", {
            "error":        "Fact-check service unavailable",
            "retry_count":  max_retries,
            "last_http_code": random.choice([503, 504, 429]),
            "latency_ms":   latency,
        })
        return False
    else:
        time.sleep(latency / 1000)
        claims = random.randint(2, 6)
        verified = random.randint(claims - 1, claims)
        trace_manager.end_span("success", {
            "claims_found":    claims,
            "claims_verified": verified,
            "accuracy_score":  round(verified / claims, 2),
            "latency_ms":      latency,
        })
        return True


def tool_summarise(text):
    """
    SIMULATED — mimics a summarisation step (could be a smaller LLM or extractive summariser).
    In production, this might call a cheap fast model (e.g. gemini-flash) just for summarisation.
    Fail rate: 10% (input too long / malformed)
    """
    latency = random.randint(150, 350)
    trace_manager.start_span("tool_call", "summarise", {"input_chars": len(text)})
    time.sleep(latency / 1000)

    # Keep up to 4000 chars — enough for the LLM to work with without
    # exceeding typical context limits. Real summarisation would use
    # a cheap/fast model here (e.g. gemini-flash) to compress semantically.
    MAX_CHARS = 4000
    summary = text[:MAX_CHARS] + "..." if len(text) > MAX_CHARS else text

    if coin("summarise"):
        trace_manager.end_span("failed", {
            "error":      "Summarisation model timeout",
            "input_chars": len(text),
            "latency_ms": latency,
        })
        return summary   # still return best-effort even on "failure"
    else:
        trace_manager.end_span("success", {
            "input_chars":       len(text),
            "output_chars":      len(summary),
            "compression_ratio": round(len(summary) / max(len(text), 1), 2),
            "latency_ms":        latency,
            "truncated":         len(text) > MAX_CHARS,
        })
        return summary


# ── Phase status helper ───────────────────────────────────────────────
def phase_status(child_results):
    """
    Derive the phase outcome from its children's results.
    If ALL children pass   → success
    If SOME children fail  → failed  (could also be "degraded" in a real SDK)
    """
    return "success" if all(child_results) else "failed"


# ── Main agent ────────────────────────────────────────────────────────
def run_agent(user_query, session_id=None, user_id=None, turn_number=None):
    trace_manager.start_trace(
        "research_agent",
        session_id  = session_id,
        user_id     = user_id,
        turn_number = turn_number,
    )
    trace_manager.start_span("agent_start", "agent_root", {"query": user_query})

    try:
        # ── Phase 1: Retrieval ─────────────────────────────────────
        trace_manager.start_span("phase", "retrieval_phase", {
            "tools": ["web_search", "validate_sources"]
        })
        search_result   = tool_web_search(user_query)
        sources_valid   = tool_validate_sources(search_result or "")
        trace_manager.end_span(phase_status([search_result is not None, sources_valid]))

        # Fall back to a default result if web_search failed
        if search_result is None:
            search_result = "LangChain is a framework for building AI agents."

        # ── Phase 2: Processing ────────────────────────────────────
        trace_manager.start_span("phase", "processing_phase", {
            "tools": ["fact_check", "summarise"]
        })
        facts_ok = tool_fact_check(search_result)
        summary  = tool_summarise(search_result)
        trace_manager.end_span(phase_status([facts_ok, summary is not None]))

        # ── Phase 3: LLM Generation ────────────────────────────────
        trace_manager.start_span("phase", "generation_phase", {"models": MODELS})
        response, model_used = invoke_llm_with_retry(user_query, summary)

        trace_manager.start_span("agent_response", "final_response", {"model": model_used})
        trace_manager.end_span("success", {"response_preview": response.content[:300]})

        trace_manager.end_span("success")   # generation_phase
        trace_manager.end_span("success")   # agent_root
        trace_manager.end_trace("success")
        return response.content

    except Exception as e:
        trace_manager.start_span("failure", "agent_failure", {"error": str(e)})
        trace_manager.end_span("failed")
        trace_manager.end_trace("failed")
        return str(e)


if __name__ == "__main__":
    result = run_agent("What is LangChain?")
    print(f"\nAgent Response:\n{result}")
