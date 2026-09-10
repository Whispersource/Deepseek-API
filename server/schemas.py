"""Pydantic models for the OpenAI-compatible request/response shapes we support."""

from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel

from .config import DEFAULT_MODEL


class ChatMessage(BaseModel):
    model_config = {"extra": "allow"}

    role: str
    # content is a plain string, or a list of parts (OpenAI vision-style). We only
    # read text parts; non-text parts are ignored. Can be None when tool_calls are present.
    content: Union[str, List[dict], None] = None
    tool_calls: Optional[List[dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "allow"}

    model: str = DEFAULT_MODEL
    messages: List[ChatMessage]
    stream: bool = False
    # Standard OpenAI Function Calling / Tools parameters
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None
    # Pass a conversation_id from a previous response to resume that thread.
    conversation_id: Optional[str] = None
    # Tools to enable for this request, independent of the model. OpenAI clients
    # pass these via extra_body: `thinking` (DeepThink), `search` (web).
    #
    # `thinking` defaults ON: a generic OpenAI-compatible client cannot send
    # extra_body at all, so with the old default of False it could never get
    # reasoning. The chain-of-thought comes back separately in
    # `reasoning_content`; send "thinking": false to skip it and answer faster.
    thinking: bool = True
    search: bool = False
    # Accepted for compatibility but not all are forwarded to DeepSeek.
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    user: Optional[str] = None
