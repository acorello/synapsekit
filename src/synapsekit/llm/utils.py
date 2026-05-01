from __future__ import annotations

from typing import Any


def extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value

    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for item in value:
            text = extract_text(item)
            if text:
                parts.append(text)
        return "".join(parts)

    if isinstance(value, dict):
        dict_text: str | None = value.get("text")
        if isinstance(dict_text, str):
            return dict_text

        content = value.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, (list, tuple)):
            return extract_text(content)
        return ""

    text_attr = getattr(value, "text", None)
    if isinstance(text_attr, str):
        return text_attr

    content_attr = getattr(value, "content", None)
    if isinstance(content_attr, str):
        return content_attr
    if isinstance(content_attr, (list, tuple)):
        return extract_text(content_attr)

    return ""


def extract_reasoning(obj: Any) -> str | None:
    if obj is None:
        return None

    for attr in ("reasoning", "reasoning_content", "thinking"):
        text = extract_text(getattr(obj, attr, None))
        if text:
            return text
    return None


def extract_reasoning_tokens(usage: Any) -> int:
    if usage is None:
        return 0
    value = getattr(usage, "reasoning_tokens", None)
    if value is not None:
        return int(value or 0)
    details = getattr(usage, "completion_tokens_details", None)
    if isinstance(details, dict):
        return int(details.get("reasoning_tokens", 0) or 0)
    if details is not None:
        return int(getattr(details, "reasoning_tokens", 0) or 0)
    return 0
