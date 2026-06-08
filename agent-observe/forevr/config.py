"""
forevr/config.py — SDK Configuration

Handles API key, project, and endpoint configuration.

Priority order (highest → lowest):
  1. forevr.init(api_key=...) called explicitly in code
  2. Environment variables (FOREVR_API_KEY, FOREVR_PROJECT, FOREVR_BASE_URL)
  3. Defaults (localhost for self-hosted)
"""

import os

_config = {
    "api_key":  None,
    "project":  "default",
    "base_url": "http://127.0.0.1:8000",
    "enabled":  True,
}


def init(
    api_key:  str = None,
    project:  str = "default",
    base_url: str = None,
):
    """
    Configure the forevr SDK.

    Call this ONCE at the top of your script before running any agents.

    Args:
        api_key:  Your forevr API key (get it from your dashboard)
                  Format: tai-xxxxxxxxxxxxxxxxxxxxxxxx
        project:  Project name shown in the dashboard (default: "default")
        base_url: Override the API endpoint.
                  For self-hosted: pass your own server URL.

    Example:
        import forevr
        forevr.init(
            api_key = "tai-xxxxxxxxxxxxxxxxxxxx",
            project = "my-research-agent",
        )
    """
    _config["api_key"] = api_key.strip() if api_key else None
    _config["project"] = project or "default"

    if base_url:
        _config["base_url"] = base_url.rstrip("/")
    else:
        if api_key:
            _config["base_url"] = "https://api.forevr.dev"
        else:
            _config["base_url"] = "http://127.0.0.1:8000"

    print(
        f"[forevr] Initialized — project: '{_config['project']}' "
        f"→ {_config['base_url']}"
    )


def _load_from_env():
    """Auto-load config from environment variables if init() hasn't been called."""
    if os.getenv("FOREVR_API_KEY") and not _config["api_key"]:
        _config["api_key"]  = os.getenv("FOREVR_API_KEY")
        _config["project"]  = os.getenv("FOREVR_PROJECT", "default")
        _config["base_url"] = os.getenv(
            "FOREVR_BASE_URL",
            "https://api.forevr.dev" if _config["api_key"] else "http://127.0.0.1:8000"
        )


def get() -> dict:
    """Return the current config. Loads from env vars if not yet initialized."""
    _load_from_env()
    return _config.copy()


def is_cloud_mode() -> bool:
    """True if sending to a remote backend (API key is set)."""
    _load_from_env()
    return bool(_config.get("api_key"))
