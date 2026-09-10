"""Translate between OpenAI's chat-completions shapes and our DeepSeek client.

DeepSeek's web protocol has no system/role channel or native tool-calling — just
a single `prompt` string. So we:
1. Format tools and messages into one prompt DeepSeek can answer (injecting tool
   definitions and calling instructions when tools are enabled);
2. Parse the output (streaming and non-streaming) to extract tool calls into
   standard OpenAI `tool_calls` structures with `finish_reason='tool_calls'`.
   Supports:
   - DeepSeek native DSML XML: <｜｜DSML｜｜ calls>, <｜｜DSML｜｜tool_calls>, <tool calls>, <tool_calls>
   - JSON format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
   - Code block format: ```tool_call ... ```
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Iterable, List, Optional, Tuple

from .schemas import ChatMessage

_ROLE_LABELS = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool",
}

# Matches opening of XML-style tool containers:
# e.g. <｜｜DSML｜｜ calls>, <｜｜DSML｜｜tool_calls>, <tool calls>, <tool_calls>, <tools>
_XML_CONTAINER_OPEN = re.compile(
    r'<(?:\s*[｜|]{1,2}\s*DSML\s*[｜|]{1,2}\s*)?(?:tool_calls|tool calls|calls|tools)(?:\s*>|\s)',
    re.IGNORECASE,
)
_XML_CONTAINER_CLOSE = re.compile(
    r'</(?:\s*[｜|]{1,2}\s*DSML\s*[｜|]{1,2}\s*)?(?:tool_calls|tool calls|calls|tools)\s*>',
    re.IGNORECASE,
)

_XML_INVOKE_PATTERN = re.compile(
    r'<(?:\s*[｜|]{1,2}\s*DSML\s*[｜|]{1,2}\s*)?invoke\s+name=["\']([^"\']+)["\']>([\s\S]*?)(?:</(?:\s*[｜|]{1,2}\s*DSML\s*[｜|]{1,2}\s*)?invoke>|$)',
    re.IGNORECASE,
)
_XML_PARAM_PATTERN = re.compile(
    r'<(?:\s*[｜|]{1,2}\s*DSML\s*[｜|]{1,2}\s*)?parameter\s+name=["\']([^"\']+)["\'](?:\s+string=["\'](true|false)["\'])?>([\s\S]*?)(?:</(?:\s*[｜|]{1,2}\s*DSML\s*[｜|]{1,2}\s*)?parameter>|$)',
    re.IGNORECASE,
)


def _text_of(content) -> str:
    """Extract plain text from a message's content (string or list-of-parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text", ""))
    return "\n".join(parts)


def format_tools_system_prompt(tools: List[dict]) -> str:
    """Generate system instructions and tool definitions for Function Calling."""
    tool_defs = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if t.get("type") == "function" and "function" in t else t
        if isinstance(fn, dict):
            tool_defs.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            })
    tools_json = json.dumps(tool_defs, ensure_ascii=False, indent=2)
    return (
        "## Tools Available\n"
        "You have access to the following tools:\n"
        f"```json\n{tools_json}\n```\n\n"
        "## Tool Calling Convention\n"
        "When you need to call a tool, you MUST use the following format:\n"
        "<tool_call>\n"
        '{"name": "tool_name", "arguments": {"param_name": "param_value"}}\n'
        "</tool_call>\n\n"
        "Rules:\n"
        "1. You may write reasoning or explanation before outputting tool calls.\n"
        "2. Only invoke tools listed in the tools definitions above.\n"
        "3. When you decide to invoke a tool, output the tool call block directly without asking for confirmation."
    )


def messages_to_prompt(messages: List[ChatMessage], tools: Optional[List[dict]] = None) -> str:
    """Flatten a chat history into a single prompt DeepSeek can answer."""
    tools_prompt = format_tools_system_prompt(tools) if tools else ""

    if len(messages) == 1 and messages[0].role == "user" and not tools:
        return _text_of(messages[0].content)

    lines = []
    system_injected = False

    for m in messages:
        role = (m.role or "user").lower()
        if role == "system":
            sys_text = _text_of(m.content)
            if tools_prompt and not system_injected:
                sys_text = f"{sys_text}\n\n{tools_prompt}" if sys_text else tools_prompt
                system_injected = True
            lines.append(f"System: {sys_text}")
        elif role == "assistant":
            parts = []
            text = _text_of(m.content)
            if text:
                parts.append(text)
            if m.tool_calls:
                parts.append("<tool_calls>")
                for tc in m.tool_calls:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    tc_name = fn.get("name") or tc.get("name", "")
                    tc_args = fn.get("arguments") or tc.get("arguments", "{}")
                    if isinstance(tc_args, str):
                        try:
                            tc_args = json.loads(tc_args)
                        except Exception:
                            tc_args = {"command": tc_args}
                    parts.append(f'<invoke name="{tc_name}">')
                    if isinstance(tc_args, dict):
                        for k, v in tc_args.items():
                            is_s = "true" if isinstance(v, str) else "false"
                            val_s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                            parts.append(f'<parameter name="{k}" string="{is_s}">{val_s}</parameter>')
                    parts.append("</invoke>")
                parts.append("</tool_calls>")
            content_str = "\n".join(parts)
            lines.append(f"Assistant: {content_str}")
        elif role == "tool":
            call_id = getattr(m, "tool_call_id", None) or getattr(m, "name", None) or "result"
            lines.append(f"Tool ({call_id}): {_text_of(m.content)}")
        else:
            label = _ROLE_LABELS.get(role, role.capitalize())
            lines.append(f"{label}: {_text_of(m.content)}")

    if tools_prompt and not system_injected:
        lines.insert(0, f"System: {tools_prompt}")

    lines.append("Assistant:")
    return "\n\n".join(lines)


def parse_xml_tool_calls(block: str) -> List[Tuple[str, str]]:
    """Parse an XML-style tool calls block into [(name, args_json_str)]."""
    calls = []
    for m in _XML_INVOKE_PATTERN.finditer(block):
        tool_name = m.group(1).strip()
        body = m.group(2)
        args = {}
        for p in _XML_PARAM_PATTERN.finditer(body):
            p_name = p.group(1).strip()
            is_str = p.group(2) == "true"
            p_val = p.group(3).strip()
            if is_str:
                args[p_name] = p_val
            else:
                try:
                    args[p_name] = json.loads(p_val)
                except Exception:
                    args[p_name] = p_val
        calls.append((tool_name, json.dumps(args, ensure_ascii=False)))
    return calls


def parse_json_tool_payload(raw: str) -> Optional[Tuple[str, str]]:
    """Parse a tool call payload string into (tool_name, arguments_json_str)."""
    raw = raw.strip()
    if not raw:
        return None

    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            if lines[-1].strip() == "```":
                raw = "\n".join(lines[1:-1]).strip()
            else:
                raw = "\n".join(lines[1:]).strip()

    data = None
    try:
        data = json.loads(raw)
    except Exception:
        s = raw.find("{")
        e = raw.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                data = json.loads(raw[s:e + 1])
            except Exception:
                pass

    if not isinstance(data, dict):
        return None

    name = data.get("name")
    if not name or not isinstance(name, str):
        return None

    args = data.get("arguments")
    if args is None:
        args = data.get("parameters")
    if args is None:
        args = data.get("args")
    if args is None:
        args = {k: v for k, v in data.items() if k != "name"}

    if isinstance(args, str):
        args_str = args
    elif isinstance(args, (dict, list)):
        args_str = json.dumps(args, ensure_ascii=False)
    else:
        args_str = str(args)

    return name, args_str


def extract_all_tool_calls(text: str) -> Tuple[str, List[dict]]:
    """Extract all tool calls (XML and JSON) from completed text."""
    tool_calls = []

    # 1. XML variants (<tool_calls>, <tool calls>, <｜｜DSML｜｜ calls>, etc.)
    def xml_repl(m):
        parsed = parse_xml_tool_calls(m.group(0))
        for name, args in parsed:
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": name, "arguments": args},
            })
        return ""

    text = re.sub(
        r'<(?:\s*[｜|]{1,2}\s*DSML\s*[｜|]{1,2}\s*)?(?:tool_calls|tool calls|calls|tools)(?:\s*>|\s)[\s\S]*?(?:</(?:\s*[｜|]{1,2}\s*DSML\s*[｜|]{1,2}\s*)?(?:tool_calls|tool calls|calls|tools)\s*>|$)',
        xml_repl,
        text,
        flags=re.IGNORECASE,
    )

    # 2. <tool_call> and ```tool_call
    def tc_repl(m):
        payload = m.group(1) or m.group(2) or ""
        parsed = parse_json_tool_payload(payload)
        if parsed:
            name, args = parsed
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": name, "arguments": args},
            })
            return ""
        return m.group(0)

    text = re.sub(
        r'<tool_call>(.*?)(?:</tool_call>|$)|```tool_call\s*(.*?)\s*(?:```|$)',
        tc_repl,
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    return text.strip(), tool_calls


def _now() -> int:
    return int(time.time())


def _id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def _est_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) — DeepSeek's web API gives us no count."""
    return max(1, len(text) // 4)


def completion_response(model: str, content: str, prompt: str,
                        conversation_id: str = None, reasoning: str = "",
                        has_tools: bool = False) -> dict:
    """A full (non-streaming) OpenAI chat.completion object."""
    pt = _est_tokens(prompt)
    ct = max(1, (len(content) + len(reasoning)) // 4)

    tool_calls = []
    finish_reason = "stop"

    if has_tools:
        cleaned_content, tool_calls = extract_all_tool_calls(content)
        if tool_calls:
            content = cleaned_content
            finish_reason = "tool_calls"

    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": _id(),
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "conversation_id": conversation_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        },
    }


def error_chunk(message: str, err_type: str = "upstream_error") -> str:
    """An SSE error frame."""
    payload = {"error": {"message": message, "type": err_type}}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def stream_chunks(model: str, stream, has_tools: bool = False) -> Iterable[str]:
    """Yield OpenAI SSE lines (`data: {...}\\n\\n`) for a streamed completion.

    Handles reasoning, normal content, and intercepted tool calls in both
    XML format (<tool_calls>, <tool calls>, <｜｜DSML｜｜ calls>) and <tool_call> JSON format.
    """
    cid, created = _id(), _now()

    def frame(delta: dict, finish=None, extra: dict = None) -> str:
        obj = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if extra:
            obj.update(extra)
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    # First frame announces the assistant role.
    yield frame({"role": "assistant", "content": ""})

    if not has_tools:
        for kind, d in stream.events():
            if not d:
                continue
            yield frame({"reasoning_content": d} if kind == "think" else {"content": d})
        conversation_id = getattr(stream, "conversation_id", None)
        yield frame({}, finish="stop", extra={"conversation_id": conversation_id})
        yield "data: [DONE]\n\n"
        return

    # Tools enabled: state machine to parse XML and <tool_call> blocks
    state = "TEXT"  # "TEXT" | "XML" | "JSON_TAG" | "JSON_FENCE"
    text_buf = ""
    tool_buf = ""
    tool_calls_emitted = 0

    for kind, d in stream.events():
        if not d:
            continue
        if kind == "think":
            yield frame({"reasoning_content": d})
            continue

        if state == "TEXT":
            text_buf += d

            while True:
                m_xml = _XML_CONTAINER_OPEN.search(text_buf)
                idx_tag = text_buf.lower().find("<tool_call>")
                idx_fence = text_buf.lower().find("```tool_call")

                matches = []
                if m_xml:
                    matches.append((m_xml.start(), m_xml.end(), "XML"))
                if idx_tag != -1:
                    matches.append((idx_tag, idx_tag + len("<tool_call>"), "JSON_TAG"))
                if idx_fence != -1:
                    matches.append((idx_fence, idx_fence + len("```tool_call"), "JSON_FENCE"))

                if matches:
                    matches.sort(key=lambda x: x[0])
                    start, end, style = matches[0]

                    content_before = text_buf[:start]
                    if content_before:
                        yield frame({"content": content_before})

                    tool_buf = text_buf[end:]
                    text_buf = ""
                    state = style
                    break
                else:
                    # Check for partial tool-opening prefix at end of text_buf
                    held = 0
                    lower = text_buf.lower()
                    for l in range(min(25, len(text_buf)), 0, -1):
                        sub = lower[-l:]
                        if any(sub.startswith(p) or p.startswith(sub) for p in ["<|", "<｜", "<tool", "<call", "```tool"]):
                            held = l
                            break
                        if sub == "<":
                            held = 1
                            break

                    if held > 0:
                        safe = text_buf[:-held]
                        text_buf = text_buf[-held:]
                    else:
                        safe = text_buf
                        text_buf = ""
                    if safe:
                        yield frame({"content": safe})
                    break

        elif state == "XML":
            tool_buf += d
            m_close = _XML_CONTAINER_CLOSE.search(tool_buf)
            if m_close:
                payload = tool_buf[:m_close.start()]
                remainder = tool_buf[m_close.end():]
                calls = parse_xml_tool_calls(payload)
                for name, args in calls:
                    yield frame({
                        "tool_calls": [{
                            "index": tool_calls_emitted,
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        }]
                    })
                    tool_calls_emitted += 1
                state = "TEXT"
                text_buf = remainder
                tool_buf = ""

        elif state == "JSON_TAG":
            tool_buf += d
            close_idx = tool_buf.lower().find("</tool_call>")
            if close_idx != -1:
                payload = tool_buf[:close_idx]
                remainder = tool_buf[close_idx + len("</tool_call>"):]
                parsed = parse_json_tool_payload(payload)
                if parsed:
                    name, args = parsed
                    yield frame({
                        "tool_calls": [{
                            "index": tool_calls_emitted,
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        }]
                    })
                    tool_calls_emitted += 1
                state = "TEXT"
                text_buf = remainder
                tool_buf = ""

        elif state == "JSON_FENCE":
            tool_buf += d
            close_idx = tool_buf.find("```")
            if close_idx != -1:
                payload = tool_buf[:close_idx]
                remainder = tool_buf[close_idx + 3:]
                parsed = parse_json_tool_payload(payload)
                if parsed:
                    name, args = parsed
                    yield frame({
                        "tool_calls": [{
                            "index": tool_calls_emitted,
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        }]
                    })
                    tool_calls_emitted += 1
                state = "TEXT"
                text_buf = remainder
                tool_buf = ""

    # End of stream flush
    if state == "XML" and tool_buf:
        calls = parse_xml_tool_calls(tool_buf)
        for name, args in calls:
            yield frame({
                "tool_calls": [{
                    "index": tool_calls_emitted,
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                }]
            })
            tool_calls_emitted += 1
    elif state in ("JSON_TAG", "JSON_FENCE") and tool_buf:
        parsed = parse_json_tool_payload(tool_buf)
        if parsed:
            name, args = parsed
            yield frame({
                "tool_calls": [{
                    "index": tool_calls_emitted,
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                }]
            })
            tool_calls_emitted += 1
    elif state == "TEXT" and text_buf:
        yield frame({"content": text_buf})

    finish = "tool_calls" if tool_calls_emitted > 0 else "stop"
    conversation_id = getattr(stream, "conversation_id", None)
    yield frame({}, finish=finish, extra={"conversation_id": conversation_id})
    yield "data: [DONE]\n\n"
