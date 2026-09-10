"""Example 6 — use the official OpenAI SDK against the server.

The whole point of the server is OpenAI compatibility: point any OpenAI client at
it and existing code works unchanged.

Install the SDK first:

    pip install openai

Start the server in another terminal:

    python app.py            # or: .\\start.ps1

Then run this from the project root:

    python examples/06_server_openai_sdk.py
"""

# Make the project importable when this file is run directly, and repair NO_PROXY
# before the SDK builds its httpx client — see deepseek/_envfix.py for why. Any
# client that talks to the server over httpx needs this, not just this example.
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from deepseek._envfix import sanitize_no_proxy

sanitize_no_proxy()

# The reply below is in Hindi, which a CP936/GBK console cannot encode — that
# raises UnicodeEncodeError as soon as output is piped. Be explicit about UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from openai import OpenAI

# Point base_url at the server. api_key is required by the SDK but ignored here.
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

completion = client.chat.completions.create(
    # deepseek-chat is the only model the server exposes. thinking (DeepThink)
    # and search (web) are independent toggles; they ride in extra_body, since
    # they're outside OpenAI's schema.
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful agent who always replies in Hindi"},
        {"role": "user", "content": "what is better macbook or framework."},
    ],
    extra_body={"thinking": True, "search": True},
)
print(completion.choices[0].message.content)

# conversation_id is outside OpenAI's schema, so the SDK keeps it in model_extra.
extra = getattr(completion, "model_extra", None) or {}
cid = extra.get("conversation_id")
print("conversation_id:", cid)

# To continue that conversation, send the id back via extra_body too:
#
#   client.chat.completions.create(
#       model="deepseek-chat",
#       messages=[{"role": "user", "content": "What's my name?"}],
#       extra_body={"conversation_id": cid, "thinking": True},
#   )
