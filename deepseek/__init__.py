"""Unofficial OpenAI-compatible client for chat.deepseek.com."""

# Must run before .client builds its httpx.Client — see deepseek/_envfix.py.
from ._envfix import sanitize_no_proxy as _sanitize_no_proxy

_sanitize_no_proxy()

from .auth import Session, get_session, login  # noqa: E402
from .client import DeepSeekClient, DeepSeekError, Reply  # noqa: E402
from .pow import DeepSeekPow  # noqa: E402

__all__ = ["Session", "get_session", "login", "DeepSeekClient", "DeepSeekError",
           "Reply", "DeepSeekPow"]
