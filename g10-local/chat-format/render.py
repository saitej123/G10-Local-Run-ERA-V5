"""Render and parse the G10 chat format v0.

The string is a sequence of atomic `<|era_*|>` tokens plus plain text.
Effort is one of 1k, 4k, 16k, 64k. The matching effort token is the first
token of every assistant turn. The budget covers the think span and the tool
trace. It does not cover the final response. This module records the budget.
The trainer enforces the count.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_TOKENS_PATH = Path(__file__).resolve().parents[1] / "special-tokens" / "tokens.py"
_spec = importlib.util.spec_from_file_location("era_special_tokens", _TOKENS_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load special tokens from {_TOKENS_PATH}")
_tokens = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tokens)

EFFORT_BUDGETS: dict[str, int] = dict(_tokens.EFFORT_BUDGETS)
T: dict[str, str] = _tokens.by_name()
ROLES = ("system", "user", "assistant", "tool")
_SURFACES = sorted((token for token in T.values()), key=len, reverse=True)


class RenderError(ValueError):
    """The transcript is not valid chat-format v0."""


def effort_limit(effort: str) -> int:
    try:
        return EFFORT_BUDGETS[effort]
    except KeyError as exc:
        raise RenderError(
            f"effort must be one of {', '.join(EFFORT_BUDGETS)}"
        ) from exc


def _reject_specials(text: str, where: str) -> None:
    for token in _SURFACES:
        if token in text:
            raise RenderError(f"{where} contains special token {token}")


def _call_json(call: object) -> str:
    if not isinstance(call, dict):
        raise RenderError("tool call must be an object")
    name = call.get("name")
    arguments = call.get("arguments", {})
    if not isinstance(name, str) or name == "":
        raise RenderError("tool call name must be a non-empty string")
    if not isinstance(arguments, dict):
        raise RenderError("tool call arguments must be an object")
    try:
        payload = json.dumps(
            {"name": name, "arguments": arguments},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except TypeError as exc:
        raise RenderError("tool call arguments must be JSON values") from exc
    _reject_specials(payload, "tool call")
    return payload


def render(messages: list[dict], effort: str = "4k") -> str:
    """Render a transcript. `effort` is 1k, 4k, 16k, or 64k."""
    effort_limit(effort)
    if not isinstance(messages, list) or not messages:
        raise RenderError("messages must be a non-empty list")

    parts: list[str] = [T["bos"]]
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise RenderError(f"message {index} must be an object")
        role = message.get("role")
        if role not in ROLES:
            raise RenderError(f"message {index} has role {role!r}")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise RenderError(f"message {index} content must be a string")
        _reject_specials(content, f"message {index} content")

        if role == "assistant":
            parts.append(T["assistant"])
            parts.append(T[f"effort_{effort}"])
            if message.get("think"):
                think = message["think"]
                if not isinstance(think, str):
                    raise RenderError(f"message {index} think must be a string")
                _reject_specials(think, f"message {index} think")
                parts.extend((T["think"], think, T["think_end"]))
            for call in message.get("tool_calls") or []:
                payload = _call_json(call)
                parts.extend((T["tool_call"], payload, T["tool_call_end"]))
            if content:
                parts.extend((T["response"], content))
            parts.append(T["end"])
        elif role == "tool":
            if content == "":
                raise RenderError(f"message {index} tool content is empty")
            parts.extend(
                (T["tool"], T["tool_result"], content, T["tool_result_end"], T["end"])
            )
        else:
            parts.extend((T[role], content, T["end"]))
    parts.append(T["eos"])
    return "".join(parts)


def _token_at(text: str, pos: int) -> tuple[str, int] | None:
    for token in _SURFACES:
        if text.startswith(token, pos):
            name = next(key for key, value in T.items() if value == token)
            return name, pos + len(token)
    return None


def _until(text: str, pos: int, closing: str) -> tuple[str, int]:
    found = _token_at(text, pos)
    if found is None or found[0] != closing:
        # Content sits between here and the closing token.
        end = text.find(T[closing], pos)
        if end < 0:
            raise RenderError(f"missing {T[closing]}")
        if _token_at(text, end) is None or _token_at(text, end)[0] != closing:
            raise RenderError(f"missing {T[closing]}")
        content = text[pos:end]
        _reject_specials(content, "parsed text")
        return content, end + len(T[closing])
    return "", found[1]


def parse(text: str) -> dict:
    """Inverse of `render` for transcripts this module produced."""
    if not isinstance(text, str) or not text.startswith(T["bos"]) or not text.endswith(T["eos"]):
        raise RenderError("transcript must start with bos and end with eos")
    pos = len(T["bos"])
    end = len(text) - len(T["eos"])
    messages: list[dict] = []
    effort: str | None = None

    while pos < end:
        opened = _token_at(text, pos)
        if opened is None or opened[0] not in ROLES:
            raise RenderError(f"expected a role token at offset {pos}")
        role, pos = opened

        if role == "assistant":
            effort_token = _token_at(text, pos)
            if effort_token is None or not effort_token[0].startswith("effort_"):
                raise RenderError("assistant turn is missing an effort token")
            this_effort = effort_token[0].removeprefix("effort_")
            if this_effort not in EFFORT_BUDGETS:
                raise RenderError(f"unknown effort token {effort_token[0]}")
            if effort is None:
                effort = this_effort
            elif effort != this_effort:
                raise RenderError("assistant turns disagree on the effort budget")
            pos = effort_token[1]
            message: dict = {"role": "assistant"}
            while True:
                nxt = _token_at(text, pos)
                if nxt is None:
                    raise RenderError("unterminated assistant turn")
                name, nxt_pos = nxt
                if name == "end":
                    pos = nxt_pos
                    break
                if name == "think":
                    content, pos = _until(text, nxt_pos, "think_end")
                    message["think"] = content
                elif name == "tool_call":
                    payload, pos = _until(text, nxt_pos, "tool_call_end")
                    try:
                        call = json.loads(payload)
                    except json.JSONDecodeError as exc:
                        raise RenderError("tool call is not JSON") from exc
                    message.setdefault("tool_calls", []).append(call)
                elif name == "response":
                    content, pos = _until(text, nxt_pos, "end")
                    message["content"] = content
                    break
                else:
                    raise RenderError(f"unexpected token {T[name]} in assistant turn")
            messages.append(message)
        elif role == "tool":
            marker = _token_at(text, pos)
            if marker is None or marker[0] != "tool_result":
                raise RenderError("tool turn is missing tool_result")
            content, pos = _until(text, marker[1], "tool_result_end")
            closer = _token_at(text, pos)
            if closer is None or closer[0] != "end":
                raise RenderError("tool turn is missing end")
            messages.append({"role": "tool", "content": content})
            pos = closer[1]
        else:
            content, pos = _until(text, pos, "end")
            messages.append({"role": role, "content": content})

    if pos != end:
        raise RenderError("trailing bytes before eos")
    return {"effort": effort, "messages": messages}
