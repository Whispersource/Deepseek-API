"""Workaround for an httpx-incompatible NO_PROXY entry on Windows.

httpx builds a URL pattern for *every* entry in NO_PROXY. A bracketed IPv6 entry
such as ``[::1]`` becomes the pattern ``all://*[::1]``, which httpx cannot parse
and rejects with::

    httpx.InvalidURL: Invalid port: ':1]'

That error is raised while constructing ``httpx.Client`` — i.e. before a single
byte is sent — so every DeepSeek request fails with a 500.

The bare ``::1`` spelling is handled correctly and already covers IPv6 loopback,
so dropping the bracketed duplicates is enough. This module is imported from
``deepseek/__init__.py`` so the fix applies to the library, the examples and the
server alike.

It is applied in Python rather than in the shell because a Windows process
environment can hold both ``NO_PROXY`` and ``no_proxy`` as separate entries, and
``os.environ`` normalises keys to upper case — so a stale lower-case entry wins
over anything a shell script assigns. Rewriting ``os.environ`` is what httpx
actually reads, so it is the reliable place to do it.
"""

from __future__ import annotations

import os

FALLBACK = "localhost,127.0.0.1,::1"


def sanitize_no_proxy() -> list[tuple[str, str, str]]:
    """Drop bracketed IPv6 entries from every NO_PROXY spelling.

    Returns the (key, before, after) of each variable that was actually changed,
    so callers can log it. Idempotent, so it is safe to call from several
    entry points.
    """
    changes: list[tuple[str, str, str]] = []
    for key in [k for k in os.environ if k.lower() == "no_proxy"]:
        raw = os.environ.get(key, "")
        kept = [
            part for part in raw.split(",")
            if part.strip() and not part.strip().startswith("[")
        ]
        cleaned = ",".join(kept) or FALLBACK
        if cleaned != raw:
            os.environ[key] = cleaned
            changes.append((key, raw, cleaned))
    return changes
