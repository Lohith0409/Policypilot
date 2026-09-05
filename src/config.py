"""
Centralized configuration for PolicyPilot.

Why this file exists: instead of every module calling os.getenv() directly
(which scatters "where do settings come from" logic across the whole codebase
and makes typos in env var names fail silently at random points), every other
file imports its settings from HERE. One source of truth. If a key is missing,
we fail loudly at startup — not three files deep into a request at 2am in prod.
"""

import os
from dotenv import load_dotenv

# Reads the .env file in the project root and loads its contents into
# the process's environment variables, so os.getenv() below can see them.
load_dotenv()


def _require_env(key: str) -> str:
    """Fetch a required env var, or crash immediately with a clear message.

    Why fail fast: if GROQ_API_KEY is missing, we want the app to refuse
    to start with a clear error — not silently run and fail confusingly
    later when the first LLM call happens.
    """
    value = os.getenv(key)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {key}. "
            f"Did you create a .env file?"
        )
    return value


GROQ_API_KEY = _require_env("GROQ_API_KEY")
QDRANT_URL = _require_env("QDRANT_URL")
QDRANT_API_KEY = _require_env("QDRANT_API_KEY")
HUGGINGFACEHUB_API_TOKEN = _require_env("HUGGINGFACEHUB_API_TOKEN")

# Model choices centralized here too, so swapping models later is a
# one-line change, not a find-and-replace across the whole codebase.
LLM_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QDRANT_COLLECTION_NAME = "policy_docs"