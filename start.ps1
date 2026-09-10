# start.ps1 — launch the DeepSeek OpenAI-compatible API server.
#
# Usage:
#   .\start.ps1
#
# Optional overrides (set before running, or edit .env):
#   $env:PORT = "8080"; $env:HOST = "0.0.0.0"; .\start.ps1
#
# The server is started through run_server.py, which repairs the NO_PROXY
# environment variable before httpx reads it (see run_server.py for why).

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$python = Join-Path $root "venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtualenv not found at $python. Create it with: python -m venv venv; .\venv\Scripts\python.exe -m pip install -r requirements.txt"
}

& $python (Join-Path $root "run_server.py")
