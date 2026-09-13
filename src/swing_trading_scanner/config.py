"""Environment configuration shared by command-line tools."""

import os
from pathlib import Path

from dotenv import load_dotenv


def load_environment():
    """Load .env from the working directory; preserve exported variables."""
    load_dotenv(Path.cwd() / ".env", override=False)


def require_credentials():
    """Return configured market-data credentials or raise a useful error."""
    load_environment()
    key = os.getenv("ALPACA_API_KEY", "").strip()
    secret = os.getenv("ALPACA_SECRET_KEY", "").strip()
    if not key or not secret:
        raise ValueError(
            "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env or your environment. "
            "See .env.example."
        )
    return key, secret
