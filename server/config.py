"""Server configuration: the OpenAI-facing model names and what they map to."""

import os

# Requests per minute allowed per client IP (override with RATE_LIMIT_PER_MINUTE).
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))

# When the server has no session, should it pop a visible browser window for
# interactive sign-in (the first request then blocks until you finish logging
# in)? On by default for local single-user use. Set to "0"/"false" for headless
# deployments, where it instead returns a 503 telling the caller to run
# `python -m deepseek.auth`.
SERVER_INTERACTIVE_LOGIN = os.getenv("SERVER_INTERACTIVE_LOGIN", "1").lower() not in (
    "0", "false", "no", "off",
)

# Public model ids the server advertises (via /v1/models) and accepts, mapped to
# DeepSeek's `model_type` wire value. This is the MODEL axis ONLY — it picks
# which model answers. DeepThink and web Search are orthogonal toggles requested
# per call via `thinking` / `search`, never encoded in the model name.
#
# ONE entry, deliberately. DeepSeek merged Fast / Expert / Vision on 2026-09-10:
# the backend still accepts `model_type` "expert" and "vision", but the model
# config the web app downloads marks both `enabled: false, switchable: false`, so
# they are retired from the product surface and may begin failing without notice.
# `default` (快速模式) is the single live mode, and the capabilities Expert used
# to gate — DeepThink reasoning and web search — are requested per call instead.
#
# To re-expose a model, add an id -> wire value pair here.
MODEL_MAP = {
    "deepseek-chat": "default",   # 快速模式 — the single live model
}

DEFAULT_MODEL = "deepseek-chat"


def is_known_model(name: str) -> bool:
    """Whether `name` is a model id we accept (used to 404 unknown models)."""
    return name in MODEL_MAP


def resolve_model_type(name: str) -> str:
    """Translate a public model id to DeepSeek's `model_type` wire value.

    Caller must check `is_known_model` first; this raises KeyError otherwise.
    """
    return MODEL_MAP[name]
