"""Deployment launcher for the DeepSeek OpenAI-compatible API server.

Extra entry point on top of `python app.py` for one reason: `app.py` relies on
the environment it inherits, and Windows environments can carry an
httpx-incompatible NO_PROXY entry (`[::1]`) that makes every request fail with
`InvalidURL: Invalid port: ':1]'` before reaching DeepSeek. Importing `deepseek`
repairs that first — see deepseek/_envfix.py for the full explanation.

Run:
    .\\venv\\Scripts\\python.exe run_server.py
or just:
    .\\start.ps1
"""

from __future__ import annotations

import os

# Import for the side effect of repairing NO_PROXY, and report what changed.
from deepseek._envfix import sanitize_no_proxy

for _key, _before, _after in sanitize_no_proxy():
    print(f"[deploy] repaired {_key}: {_before!r} -> {_after!r}", flush=True)

import uvicorn  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    print(f"[deploy] DeepSeek OpenAI-compatible API on http://{host}:{port}", flush=True)
    print(f"[deploy]   health: http://{host}:{port}/healthz", flush=True)
    print(f"[deploy]   models: http://{host}:{port}/v1/models", flush=True)
    uvicorn.run("server.api:app", host=host, port=port, reload=False)
