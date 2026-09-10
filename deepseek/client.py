"""
Pure-HTTP DeepSeek chat client.

Speaks chat.deepseek.com's internal API directly using a captured signed-in
session (see `deepseek.auth`). For each message it:

    1. creates a chat session   (POST /api/v0/chat_session/create)
    2. fetches a PoW challenge   (POST /api/v0/chat/create_pow_challenge)
    3. solves it via the WASM    (deepseek.pow.DeepSeekPow)
    4. POSTs the completion       with the x-ds-pow-response header
    5. parses the SSE stream      into text

    from deepseek.auth import get_session
    from deepseek.client import DeepSeekClient

    client = DeepSeekClient(get_session())
    print(client.chat("Hello!"))                 # full reply
    for chunk in client.stream("Tell a joke"):   # streamed
        print(chunk, end="", flush=True)
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Iterator, Optional

import httpx

from .auth import Session, get_session
from .pow import DeepSeekPow

BASE = "https://chat.deepseek.com"
COMPLETION_PATH = "/api/v0/chat/completion"

# DeepSeek's model selector, sent as `model_type` in the completion body.
# "default" (快速模式) is the only mode the web app still offers: DeepSeek merged
# Fast / Expert / Vision on 2026-09-10, leaving "expert" and "vision" accepted by
# the backend but disabled in the model config. Omitting the field lets the
# backend pick, so we always send one explicitly.
DEFAULT_MODEL_TYPE = "default"

# A conversation_id is an opaque "<chat_session_id>:<last_message_id>" token. It
# carries everything needed to resume a thread, so the client stays stateless.
_CID_SEP = ":"


def _encode_cid(session_id: str, message_id: Optional[int]) -> str:
    if message_id is None:
        return session_id
    return f"{session_id}{_CID_SEP}{message_id}"


def _decode_cid(conversation_id: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    """Split a conversation_id back into (chat_session_id, parent_message_id)."""
    if not conversation_id:
        return None, None
    session_id, _, msg = conversation_id.partition(_CID_SEP)
    parent = int(msg) if msg.isdigit() else None
    return (session_id or None), parent


class DeepSeekError(RuntimeError):
    """An error DeepSeek reported inside the completion stream.

    The chat backend signals upstream failures (rate limiting, content policy)
    as a `hint` event carrying no fragments at all. Left unhandled the stream
    just ends, so the caller sees a successful-but-empty answer instead of the
    real reason. `finish_reason` carries DeepSeek's code, e.g.
    "rate_limit_reached".
    """

    def __init__(self, message: str, finish_reason: Optional[str] = None):
        super().__init__(message)
        self.finish_reason = finish_reason


@dataclass
class Reply:
    """A completed chat reply plus the id to resume the conversation.

    `reasoning` holds the DeepThink chain-of-thought when thinking is enabled
    (empty otherwise); it is reported separately from the answer, mirroring
    DeepSeek's own `reasoning_content` field.
    """

    text: str
    conversation_id: str
    reasoning: str = ""

    def __str__(self) -> str:  # so print(reply) shows the text
        return self.text


def _biz(data: dict) -> dict:
    """Unwrap DeepSeek's `data.biz_data` envelope, raising on API-level errors."""
    if data.get("code") != 0:
        raise RuntimeError(f"DeepSeek API error: {data.get('msg') or data}")
    biz = data.get("data", {}).get("biz_data")
    if biz is None:
        raise RuntimeError(f"Unexpected response shape: {data}")
    return biz


class DeepSeekClient:
    def __init__(
        self,
        session: Optional[Session] = None,
        allow_interactive: bool = True,
    ):
        # `allow_interactive=False` makes session resolution non-blocking: it
        # uses a cached/headless session and raises LoginRequired instead of
        # opening a browser window. The server passes False (see server/api.py).
        self.session = session or get_session(allow_interactive=allow_interactive)
        self._pow = DeepSeekPow()
        # The wasmtime Store behind the PoW solver is not reentrant; serialise
        # access so concurrent server requests don't corrupt it.
        self._pow_lock = threading.Lock()
        self._http = httpx.Client(
            base_url=BASE,
            headers=self._base_headers(),
            cookies=self.session.cookies,
            timeout=httpx.Timeout(120.0, read=300.0),
        )

    def _base_headers(self) -> dict:
        return {
            "authorization": f"Bearer {self.session.token}",
            "accept": "*/*",
            "content-type": "application/json",
            "user-agent": self.session.user_agent,
            "origin": BASE,
            "referer": f"{BASE}/",
            "x-app-version": "2.0.0",
            "x-client-version": "2.0.0",
            "x-client-platform": "web",
            "x-client-locale": "en_US",
            "x-client-bundle-id": "com.deepseek.chat",
            "x-client-timezone-offset": "19800",
        }

    # --- protocol steps -----------------------------------------------------

    def create_chat_session(self) -> str:
        r = self._http.post("/api/v0/chat_session/create", json={})
        r.raise_for_status()
        return _biz(r.json())["chat_session"]["id"]

    def _pow_header(self, target_path: str = COMPLETION_PATH) -> str:
        r = self._http.post(
            "/api/v0/chat/create_pow_challenge", json={"target_path": target_path}
        )
        r.raise_for_status()
        challenge = _biz(r.json())["challenge"]
        with self._pow_lock:
            return self._pow.make_header(challenge)

    # --- public API ---------------------------------------------------------

    def stream(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        thinking: bool = False,
        search: bool = False,
    ) -> "_Stream":
        """Stream a reply. Iterate it for text chunks; read `.conversation_id`
        afterwards to resume the thread. Pass an existing `conversation_id` to
        continue a previous conversation.

        `model` is DeepSeek's model_type wire value; "default" is the only live
        one. It defaults to "default" on a NEW thread. It cannot be combined
        with `conversation_id` — a thread's model is fixed when it's created, so
        resuming keeps the original model. `thinking` enables DeepThink reasoning
        and `search` enables web search; both are independent of the model.
        """
        if conversation_id and model is not None:
            raise ValueError(
                "`model` cannot be set together with `conversation_id`; a thread's "
                "model is fixed when it is created. Pass `model` only on the first turn."
            )
        session_id, parent_id = _decode_cid(conversation_id)
        if session_id is None:
            # New thread: select the model (default when unspecified).
            session_id = self.create_chat_session()
            model_type: Optional[str] = model or DEFAULT_MODEL_TYPE
        else:
            # Resuming: let the existing thread's model stand (send no model_type).
            model_type = None
        return _Stream(self, prompt, session_id, parent_id, model_type, thinking, search)

    def chat(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        thinking: bool = False,
        search: bool = False,
    ) -> Reply:
        """Return the complete reply (`.text`), its `.conversation_id`, and — when
        thinking is enabled — the chain-of-thought in `.reasoning`."""
        s = self.stream(prompt, conversation_id=conversation_id,
                        model=model, thinking=thinking, search=search)
        reasoning: list = []
        text: list = []
        for kind, delta in s.events():
            (reasoning if kind == "think" else text).append(delta)
        return Reply(text="".join(text), conversation_id=s.conversation_id,
                     reasoning="".join(reasoning))

    def close(self) -> None:
        self._http.close()


class _Stream:
    """A single streamed reply.

    Consume it with `events()` to get every delta tagged as either "think"
    (DeepThink chain-of-thought) or "response" (the answer), or iterate it
    directly to get answer text only. It can only be consumed ONCE — `.events()`
    performs the HTTP request — and afterwards `.conversation_id` holds the token
    for resuming the conversation.
    """

    def __init__(self, client: "DeepSeekClient", prompt: str, session_id: str,
                 parent_id: Optional[int], model: str,
                 thinking: bool, search: bool):
        self._client = client
        self._prompt = prompt
        self._session_id = session_id
        self._parent_id = parent_id
        self._model = model
        self._thinking = thinking
        self._search = search
        self._message_id: Optional[int] = None

    def events(self) -> Iterator[tuple]:
        """Yield ``(kind, text)`` deltas, kind being "think" or "response".

        Runs the request, so this is a one-shot generator: calling it twice
        starts a second completion.
        """
        body = {
            "chat_session_id": self._session_id,
            "parent_message_id": self._parent_id,
            "prompt": self._prompt,
            "ref_file_ids": [],
            "thinking_enabled": self._thinking,
            "search_enabled": self._search,
            "action": None,
            "preempt": False,
        }
        # Only select a model on a new thread; on resume the thread keeps its own.
        if self._model is not None:
            body["model_type"] = self._model
        # PoW challenges are short-lived, so solve right before the request.
        headers = {"x-ds-pow-response": self._client._pow_header()}
        meta: dict = {}
        with self._client._http.stream(
            "POST", COMPLETION_PATH, json=body, headers=headers
        ) as resp:
            resp.raise_for_status()
            yield from _parse_sse(resp.iter_lines(), meta)
        if meta.get("message_id") is not None:
            self._message_id = meta["message_id"]

    def __iter__(self) -> Iterator[str]:
        """Answer text only — the chain-of-thought is skipped. Use `events()` if
        you need the reasoning as well."""
        for kind, text in self.events():
            if kind == "response":
                yield text

    @property
    def conversation_id(self) -> str:
        return _encode_cid(self._session_id, self._message_id)


_FRAGMENT_KINDS = {"RESPONSE": "response", "THINK": "think", "THINKING": "think"}


def _kind_of(fragment_type) -> Optional[str]:
    """Map a fragment type to our two channels: "think" or "response"."""
    return _FRAGMENT_KINDS.get(fragment_type)


def _parse_sse(lines, meta: Optional[dict] = None) -> Iterator[tuple]:
    """Turn DeepSeek's SSE completion stream into ``(kind, text)`` deltas.

    The stream sends an initial snapshot frame whose `v` is the full response
    object (with `fragments[].content`), then a series of append frames:
      * {"p":"response/fragments/-1/content","o":"APPEND","v":" what"}  (sets path)
      * {"v":"'s"}                                                       (appends to it)

    A response is a *list* of fragments, and they are not all the same thing:
    with DeepThink on, a THINK fragment (the chain-of-thought) streams first and a
    RESPONSE fragment (the actual answer) is appended after it. Both are addressed
    through the same "-1" (newest) index, so the path alone cannot tell them
    apart — every delta is tagged "think" or "response" so callers can route the
    reasoning separately instead of either mixing it into the answer or losing it.

    If `meta` is given, the assistant's `message_id` is recorded into it (used to
    build the resumable conversation_id). The exact field location can vary, so
    we look in a few plausible spots defensively.
    """
    active_path: Optional[str] = None
    # Type of each fragment seen so far, in order, so an append frame's "-1"
    # (newest) index can be resolved back to the fragment it points at.
    fragment_types: list = []
    # Indexes of fragments whose opening content we already emitted, so a
    # re-sent snapshot mid-stream cannot duplicate text. Keyed by position in the
    # growing fragment list, which stays stable across snapshots and appends.
    opened_fragments: set = set()

    def _newest_kind() -> Optional[str]:
        return _kind_of(fragment_types[-1]) if fragment_types else None

    for line in lines:
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue

        v = obj.get("v")

        # Error hint frame: DeepSeek reports upstream failures as a `hint` event
        # with `{"type": "error", "finish_reason": ...}` and no fragments at all.
        # Surface it — otherwise the stream ends silently and the caller gets an
        # empty answer with no idea that anything went wrong.
        if obj.get("type") == "error":
            raise DeepSeekError(
                obj.get("content") or "DeepSeek returned an error",
                obj.get("finish_reason"),
            )

        # Snapshot frame: full response object.
        if isinstance(v, dict) and "response" in v:
            if meta is not None:
                _capture_message_id(meta, v)
            fragments = v["response"].get("fragments", [])
            fragment_types = [frag.get("type") for frag in fragments]
            base = 0
            if fragments:
                active_path = "response/fragments/-1/content"
            for j, frag in enumerate(fragments):
                idx = base + j
                kind, content = _kind_of(frag.get("type")), frag.get("content")
                if kind and content and idx not in opened_fragments:
                    opened_fragments.add(idx)
                    yield kind, content
            continue

        # Whole new fragment(s) appended to the response.
        if obj.get("p") == "response/fragments" and isinstance(v, list):
            if obj.get("o") == "SET":
                fragment_types = [frag.get("type") for frag in v]
                base = 0
            else:
                base = len(fragment_types)
                fragment_types.extend(frag.get("type") for frag in v)
            if v:
                active_path = "response/fragments/-1/content"
            # A fragment arrives with its opening text already in it, so emit that
            # here; later deltas come through the path frames below.
            for j, frag in enumerate(v):
                idx = base + j
                kind, content = _kind_of(frag.get("type")), frag.get("content")
                if kind and content and idx not in opened_fragments:
                    opened_fragments.add(idx)
                    yield kind, content
            continue

        # Path-setting append frame.
        if "p" in obj:
            path = obj["p"]
            if path.endswith("content"):
                active_path = path
            if meta is not None and path.endswith("message_id") \
                    and isinstance(v, int):
                meta["message_id"] = v
            # `o` is often omitted on these frames — the real stream emits a bare
            # {"p":"response/fragments/-1/content","v":"!"} after appending a new
            # fragment. Requiring o == "APPEND" silently swallowed that trailing
            # text (the answer came back as "Hello" instead of "Hello!"). Only an
            # explicit SET means replace.
            if isinstance(v, str) and v and path.endswith("content") \
                    and obj.get("o") != "SET":
                kind = _newest_kind()
                if kind:
                    yield kind, v
            continue

        # Bare append to the current path.
        if isinstance(v, str) and active_path and active_path.endswith("content"):
            kind = _newest_kind()
            if kind:
                yield kind, v


def _capture_message_id(meta: dict, snapshot: dict) -> None:
    """Best-effort: pull the assistant message_id out of a snapshot frame.

    DeepSeek nests the assistant message under `response`; we check there first,
    then the snapshot root, accepting `message_id` or `id`.
    """
    for container in (snapshot.get("response"), snapshot):
        if isinstance(container, dict):
            mid = container.get("message_id", container.get("id"))
            if isinstance(mid, int):
                meta["message_id"] = mid
                return
