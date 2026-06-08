"""
database.py — SQLite persistence layer for Trace AI
====================================================

Stores every completed trace as a JSON blob in a local SQLite database.
SQLite is built into Python — no install needed, no external service.

Schema (single table, simple):
    traces
        trace_id   TEXT  PRIMARY KEY   — unique ID per run
        name       TEXT               — agent name
        status     TEXT               — success / failed
        start_time REAL               — unix timestamp (for sorting)
        data       TEXT               — full trace dict serialized as JSON

Why JSON blob (not normalized tables)?
    - Traces have deeply nested spans — flattening adds complexity with zero benefit at this scale
    - Schema changes (new span fields) don't require SQL migrations
    - Fast enough for thousands of traces
    - Single SELECT to load everything
"""

import sqlite3
import json
import os
import time
import hashlib
import secrets

# ── Database location ──────────────────────────────────────────────────────
# Stored next to the backend files so it's easy to find and back up.
DB_PATH = os.environ.get(
    "DB_PATH",
    os.path.join(os.path.dirname(__file__), "traces.db")
)


def _connect() -> sqlite3.Connection:
    """Open a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    return conn


def init_db() -> None:
    """
    Create all tables if they don't exist yet.
    Safe to call on every startup — CREATE TABLE IF NOT EXISTS is idempotent.
    """
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                trace_id   TEXT  PRIMARY KEY,
                name       TEXT,
                status     TEXT,
                start_time REAL,
                data       TEXT  NOT NULL,
                saved_at   REAL  DEFAULT (unixepoch('now'))
            )
        """)
        # Add session columns to traces if they don't exist yet (safe migration)
        for col, typ in [("session_id", "TEXT"), ("user_id", "TEXT"), ("turn_number", "INTEGER")]:
            try:
                conn.execute(f"ALTER TABLE traces ADD COLUMN {col} {typ}")
            except Exception:
                pass  # column already exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,
                user_id     TEXT DEFAULT '',
                started_at  REAL,
                last_active REAL,
                turn_count  INTEGER DEFAULT 0,
                total_cost  REAL DEFAULT 0.0,
                has_failure INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                trace_id  TEXT PRIMARY KEY,
                score     INTEGER NOT NULL,
                label     TEXT DEFAULT '',
                comment   TEXT DEFAULT '',
                scored_at REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id  TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                description TEXT DEFAULT '',
                created_at  REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dataset_members (
                dataset_id TEXT NOT NULL,
                trace_id   TEXT NOT NULL,
                split      TEXT DEFAULT 'train',
                added_at   REAL,
                PRIMARY KEY (dataset_id, trace_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS judgments (
                trace_id    TEXT PRIMARY KEY,
                score       REAL NOT NULL,
                verdict     TEXT,
                reasoning   TEXT,
                judge_model TEXT,
                judged_at   REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                key_id     TEXT PRIMARY KEY,
                name       TEXT NOT NULL,
                key_hash   TEXT NOT NULL UNIQUE,
                key_prefix TEXT NOT NULL,
                created_at REAL,
                last_used  REAL,
                active     INTEGER DEFAULT 1
            )
        """)
        conn.commit()
    print(f"[DB] SQLite ready at: {DB_PATH}")


def _upsert_session(trace: dict) -> None:
    """Auto-create or update a session record whenever a session-tagged trace is saved."""
    sid = trace.get("session_id")
    if not sid:
        return
    try:
        with _connect() as conn:
            existing = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?", (sid,)
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO sessions (session_id, user_id, started_at, last_active, turn_count, total_cost, has_failure) VALUES (?, ?, ?, ?, 1, ?, ?)",
                    (
                        sid,
                        trace.get("user_id", ""),
                        trace.get("start_time", time.time()),
                        trace.get("end_time", time.time()),
                        trace.get("total_cost_usd", 0.0),
                        1 if trace.get("status") == "failed" else 0,
                    ),
                )
            else:
                conn.execute(
                    """UPDATE sessions SET
                        last_active = MAX(last_active, ?),
                        turn_count  = turn_count + 1,
                        total_cost  = total_cost + ?,
                        has_failure = CASE WHEN ? = 1 THEN 1 ELSE has_failure END
                    WHERE session_id = ?""",
                    (
                        trace.get("end_time", time.time()),
                        trace.get("total_cost_usd", 0.0),
                        1 if trace.get("status") == "failed" else 0,
                        sid,
                    ),
                )
            conn.commit()
    except Exception as e:
        print(f"[DB] Warning — failed to upsert session: {e}")


def save_trace(trace: dict) -> None:
    """
    Persist a completed trace to SQLite.
    Called by tracer.py at the end of every agent run.

    Uses INSERT OR REPLACE so if a trace_id already exists
    (e.g. after a hot-reload mid-run), it's safely overwritten.
    """
    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO traces
                    (trace_id, name, status, start_time, data, saved_at, session_id, user_id, turn_number)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.get("trace_id", "unknown"),
                    trace.get("name", ""),
                    trace.get("status", "unknown"),
                    trace.get("start_time", time.time()),
                    json.dumps(trace),
                    time.time(),
                    trace.get("session_id"),
                    trace.get("user_id"),
                    trace.get("turn_number"),
                )
            )
            conn.commit()
        _upsert_session(trace)
    except Exception as e:
        # Never crash the agent run because of a DB write failure
        print(f"[DB] Warning — failed to save trace {trace.get('trace_id')}: {e}")


def load_all_traces() -> list:
    """
    Load every trace from SQLite, ordered oldest → newest.
    Called once on server startup to restore the in-memory traces list.

    Returns a list of trace dicts (same format tracer.py produces).
    """
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT data FROM traces ORDER BY start_time ASC"
            ).fetchall()
        traces = [json.loads(row["data"]) for row in rows]
        print(f"[DB] Loaded {len(traces)} trace(s) from SQLite")
        return traces
    except Exception as e:
        print(f"[DB] Warning — failed to load traces: {e}")
        return []


def delete_trace(trace_id: str) -> bool:
    """
    Delete a single trace by ID.
    Useful for a future 'clear' button in the dashboard.
    Returns True if a row was deleted.
    """
    try:
        with _connect() as conn:
            cursor = conn.execute(
                "DELETE FROM traces WHERE trace_id = ?", (trace_id,)
            )
            conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"[DB] Warning — failed to delete trace {trace_id}: {e}")
        return False


def clear_all_traces() -> int:
    """
    Delete ALL traces from the database.
    Returns the number of rows deleted.
    """
    try:
        with _connect() as conn:
            cursor = conn.execute("DELETE FROM traces")
            conn.commit()
        print(f"[DB] Cleared {cursor.rowcount} trace(s) from SQLite")
        return cursor.rowcount
    except Exception as e:
        print(f"[DB] Warning — failed to clear traces: {e}")
        return 0


def get_trace_count() -> int:
    """Return total number of stored traces."""
    try:
        with _connect() as conn:
            row = conn.execute("SELECT COUNT(*) as n FROM traces").fetchone()
        return row["n"] if row else 0
    except Exception:
        return 0


# ── Scores ─────────────────────────────────────────────────────────────────

def save_score(trace_id: str, score: int, label: str = "", comment: str = "") -> None:
    try:
        with _connect() as conn:
            if score == 0:
                conn.execute("DELETE FROM scores WHERE trace_id = ?", (trace_id,))
            else:
                conn.execute(
                    "INSERT OR REPLACE INTO scores (trace_id, score, label, comment, scored_at) VALUES (?, ?, ?, ?, ?)",
                    (trace_id, score, label, comment, time.time()),
                )
            conn.commit()
    except Exception as e:
        print(f"[DB] Warning — failed to save score: {e}")


def get_all_scores() -> dict:
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT trace_id, score, label, comment, scored_at FROM scores"
            ).fetchall()
        return {
            r["trace_id"]: {
                "score": r["score"], "label": r["label"],
                "comment": r["comment"], "scored_at": r["scored_at"],
            }
            for r in rows
        }
    except Exception as e:
        print(f"[DB] Warning — failed to load scores: {e}")
        return {}


# ── Datasets ───────────────────────────────────────────────────────────────

def save_dataset(dataset_id: str, name: str, description: str = "") -> None:
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO datasets (dataset_id, name, description, created_at) VALUES (?, ?, ?, ?)",
                (dataset_id, name, description, time.time()),
            )
            conn.commit()
    except Exception as e:
        print(f"[DB] Warning — failed to save dataset: {e}")


def get_all_datasets() -> list:
    try:
        with _connect() as conn:
            rows = conn.execute("""
                SELECT d.dataset_id, d.name, d.description, d.created_at,
                       COUNT(m.trace_id) as trace_count
                FROM datasets d
                LEFT JOIN dataset_members m ON d.dataset_id = m.dataset_id
                GROUP BY d.dataset_id
                ORDER BY d.created_at DESC
            """).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"[DB] Warning — failed to load datasets: {e}")
        return []


def delete_dataset(dataset_id: str) -> None:
    try:
        with _connect() as conn:
            conn.execute("DELETE FROM dataset_members WHERE dataset_id = ?", (dataset_id,))
            conn.execute("DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,))
            conn.commit()
    except Exception as e:
        print(f"[DB] Warning — failed to delete dataset: {e}")


def add_to_dataset(dataset_id: str, trace_id: str, split: str = "train") -> None:
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO dataset_members (dataset_id, trace_id, split, added_at) VALUES (?, ?, ?, ?)",
                (dataset_id, trace_id, split, time.time()),
            )
            conn.commit()
    except Exception as e:
        print(f"[DB] Warning — failed to add to dataset: {e}")


def remove_from_dataset(dataset_id: str, trace_id: str) -> None:
    try:
        with _connect() as conn:
            conn.execute(
                "DELETE FROM dataset_members WHERE dataset_id = ? AND trace_id = ?",
                (dataset_id, trace_id),
            )
            conn.commit()
    except Exception as e:
        print(f"[DB] Warning — failed to remove from dataset: {e}")


def get_dataset_members(dataset_id: str) -> list:
    try:
        with _connect() as conn:
            rows = conn.execute("""
                SELECT m.trace_id, m.split, m.added_at, t.data
                FROM dataset_members m
                JOIN traces t ON m.trace_id = t.trace_id
                WHERE m.dataset_id = ?
                ORDER BY m.added_at ASC
            """, (dataset_id,)).fetchall()
        result = []
        for r in rows:
            trace = json.loads(r["data"])
            trace["_split"] = r["split"]
            trace["_added_at"] = r["added_at"]
            result.append(trace)
        return result
    except Exception as e:
        print(f"[DB] Warning — failed to get dataset members: {e}")
        return []


def get_all_memberships() -> dict:
    """Returns {trace_id: [dataset_id, ...]} for all memberships."""
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT dataset_id, trace_id FROM dataset_members"
            ).fetchall()
        result: dict = {}
        for r in rows:
            result.setdefault(r["trace_id"], []).append(r["dataset_id"])
        return result
    except Exception as e:
        print(f"[DB] Warning — failed to load memberships: {e}")
        return {}


# ── Judgments ──────────────────────────────────────────────────────────────

def save_judgment(
    trace_id: str, score: float, verdict: str, reasoning: str, judge_model: str
) -> None:
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO judgments (trace_id, score, verdict, reasoning, judge_model, judged_at) VALUES (?, ?, ?, ?, ?, ?)",
                (trace_id, score, verdict, reasoning, judge_model, time.time()),
            )
            conn.commit()
    except Exception as e:
        print(f"[DB] Warning — failed to save judgment: {e}")


def get_all_judgments() -> dict:
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT trace_id, score, verdict, reasoning, judge_model, judged_at FROM judgments"
            ).fetchall()
        return {r["trace_id"]: dict(r) for r in rows}
    except Exception as e:
        print(f"[DB] Warning — failed to load judgments: {e}")
        return {}


# ── Sessions ───────────────────────────────────────────────────────────────

def get_all_sessions() -> list:
    """Return all sessions ordered by most recently active."""
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY last_active DESC"
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"[DB] Warning — failed to load sessions: {e}")
        return []


def get_session_traces(session_id: str) -> list:
    """Return all traces belonging to a session, ordered by start_time ASC (turn order)."""
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT data FROM traces WHERE session_id = ? ORDER BY start_time ASC",
                (session_id,),
            ).fetchall()
        return [json.loads(r["data"]) for r in rows]
    except Exception as e:
        print(f"[DB] Warning — failed to load session traces: {e}")
        return []


def delete_session(session_id: str) -> None:
    """Delete a session and disassociate its traces (traces are kept, just unlinked)."""
    try:
        with _connect() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
    except Exception as e:
        print(f"[DB] Warning — failed to delete session: {e}")


# ── API Keys ────────────────────────────────────────────────────────────────

def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def create_api_key(name: str) -> dict:
    """
    Generate a new API key, store its hash, return the full key once.
    The raw key is NEVER stored — only the SHA-256 hash.

    Returns:
        { "key_id": "...", "name": "...", "key": "tai-xxxx...", "prefix": "tai-xxxx" }
    """
    raw_key    = "tai-" + secrets.token_hex(32)   # tai- + 64 hex chars
    key_prefix = raw_key[:12]                      # "tai-xxxxxxxx" — shown in list
    key_hash   = _hash_key(raw_key)
    key_id     = str(secrets.token_hex(8))
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO api_keys (key_id, name, key_hash, key_prefix, created_at, active) VALUES (?, ?, ?, ?, ?, 1)",
                (key_id, name, key_hash, key_prefix, time.time()),
            )
            conn.commit()
        return {"key_id": key_id, "name": name, "key": raw_key, "prefix": key_prefix}
    except Exception as e:
        print(f"[DB] Warning — failed to create API key: {e}")
        return {}


def validate_api_key(raw_key: str) -> bool:
    """Return True if the key exists, is active, and updates last_used."""
    try:
        key_hash = _hash_key(raw_key)
        with _connect() as conn:
            row = conn.execute(
                "SELECT key_id FROM api_keys WHERE key_hash = ? AND active = 1",
                (key_hash,),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE api_keys SET last_used = ? WHERE key_id = ?",
                    (time.time(), row["key_id"]),
                )
                conn.commit()
                return True
        return False
    except Exception as e:
        print(f"[DB] Warning — failed to validate API key: {e}")
        return False


def has_any_api_keys() -> bool:
    """True if at least one active key exists. Used to decide whether to enforce auth."""
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM api_keys WHERE active = 1 LIMIT 1"
            ).fetchone()
        return row is not None
    except Exception:
        return False


def list_api_keys() -> list:
    """Return all keys with masked display (prefix only, never the hash)."""
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT key_id, name, key_prefix, created_at, last_used, active FROM api_keys ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"[DB] Warning — failed to list API keys: {e}")
        return []


def revoke_api_key(key_id: str) -> bool:
    """Soft-delete: mark as inactive. Returns True if found."""
    try:
        with _connect() as conn:
            cursor = conn.execute(
                "UPDATE api_keys SET active = 0 WHERE key_id = ?", (key_id,)
            )
            conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"[DB] Warning — failed to revoke API key: {e}")
        return False
